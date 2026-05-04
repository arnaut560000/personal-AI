from __future__ import annotations

from pathlib import Path
import tempfile
import wave
from typing import Any

from config import AppConfig


def check_microphone_available() -> tuple[bool, str]:
    try:
        import sounddevice as sd
    except Exception as exc:
        return (False, f"sounddevice import failed: {exc}")
    try:
        devices = sd.query_devices()
    except Exception as exc:
        return (False, f"Microphone query failed: {exc}")
    if not devices:
        return (False, "No audio devices were found.")
    for device in devices:
        if float(device.get("max_input_channels", 0)) > 0:
            return (True, "Microphone detected.")
    return (False, "No input-capable microphone device detected.")


def save_wav_file(audio_path: Path, audio_data: Any, config: AppConfig) -> None:
    import numpy as np

    pcm_audio = np.clip(audio_data * 32767, -32768, 32767).astype(np.int16)
    with wave.open(str(audio_path), "wb") as wav_file:
        wav_file.setnchannels(config.channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(config.sample_rate)
        wav_file.writeframes(pcm_audio.tobytes())


def record_until_silence(config: AppConfig) -> Path:
    import numpy as np
    import sounddevice as sd

    silence_limit_samples = int(config.silence_limit_seconds * config.sample_rate)
    max_record_samples = config.max_record_seconds * config.sample_rate
    min_audio_samples = int(config.min_audio_seconds * config.sample_rate)
    chunk_size = 1024
    silence_threshold = 0.01

    frames: list[Any] = []
    heard_voice = False
    silence_samples = 0
    total_samples = 0

    print("Listening... speak now.")
    with sd.InputStream(
        samplerate=config.sample_rate,
        channels=config.channels,
        dtype="float32",
    ) as stream:
        while total_samples < max_record_samples:
            chunk, _ = stream.read(chunk_size)
            frames.append(chunk.copy())
            total_samples += len(chunk)

            volume = float(np.max(np.abs(chunk)))
            if volume >= silence_threshold:
                heard_voice = True
                silence_samples = 0
            elif heard_voice:
                silence_samples += len(chunk)
                if silence_samples >= silence_limit_samples and total_samples >= min_audio_samples:
                    break

    if not heard_voice:
        raise RuntimeError("I could not hear clear speech. Please try again a little closer to the microphone.")

    audio_data = np.concatenate(frames, axis=0)
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
        temp_path = Path(temp_file.name)
    save_wav_file(temp_path, audio_data, config)
    return temp_path
