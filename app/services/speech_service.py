from __future__ import annotations

from pathlib import Path
from queue import Queue
import re
import subprocess
import tempfile
import threading
import wave
from typing import Any

from app.config import AppConfig


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


def create_whisper_model(config: AppConfig) -> Any:
    try:
        from faster_whisper import WhisperModel

        return WhisperModel(
            config.whisper_model,
            device="cpu",
            compute_type="int8",
            cpu_threads=config.whisper_cpu_threads,
        )
    except Exception as exc:
        raise RuntimeError(
            "Whisper could not start. Install voice dependencies and verify the "
            f'"{config.whisper_model}" model can be loaded.'
        ) from exc


def transcribe_audio_file(whisper_model: Any, audio_path: Path, config: AppConfig) -> str:
    try:
        segments, _ = whisper_model.transcribe(
            str(audio_path),
            beam_size=config.whisper_beam_size,
            vad_filter=True,
            language="en",
            initial_prompt=(
                "This is a voice assistant conversation about locations, nearby restaurants, "
                "lists, cafes, hotels, and places to eat."
            ),
        )
    except Exception as exc:
        raise RuntimeError(
            "RoomAI could not transcribe that audio file. "
            "Please try again with a clearer recording."
        ) from exc

    transcript = " ".join(segment.text.strip() for segment in segments).strip()
    if not transcript:
        raise RuntimeError("I heard audio, but I could not understand the words. Please try again clearly.")
    return transcript


def transcribe_microphone_audio(whisper_model: Any, config: AppConfig) -> str:
    audio_path = record_until_silence(config)
    try:
        return transcribe_audio_file(whisper_model, audio_path, config)
    finally:
        audio_path.unlink(missing_ok=True)


def speak_with_powershell(text: str) -> None:
    escaped_text = text.replace("'", "''")
    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        f"$speaker.Speak('{escaped_text}')"
    )
    result = subprocess.run(
        ["powershell", "-Command", command],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        error_output = result.stderr.strip() or result.stdout.strip() or "Unknown speech error."
        raise RuntimeError(f"Windows speech fallback failed: {error_output}")


def create_tts_engine() -> tuple[str, Any]:
    pyttsx3_import_error: str | None = None
    pyttsx3_init_error: str | None = None
    try:
        import pyttsx3
    except Exception as exc:
        pyttsx3_import_error = str(exc)
    else:
        try:
            return ("pyttsx3", pyttsx3.init())
        except Exception as exc:
            pyttsx3_init_error = str(exc)
    try:
        speak_with_powershell("RoomAI voice is ready.")
        return ("powershell", None)
    except RuntimeError as exc:
        details = []
        if pyttsx3_import_error:
            details.append(f"pyttsx3 import failed: {pyttsx3_import_error}")
        if pyttsx3_init_error:
            details.append(f"pyttsx3 init failed: {pyttsx3_init_error}")
        details.append(str(exc))
        raise RuntimeError("Text-to-speech is unavailable. " + " | ".join(details)) from exc


def speak_text(tts_backend: str, engine: Any, text: str) -> None:
    try:
        if tts_backend == "pyttsx3":
            engine.say(text)
            engine.runAndWait()
        else:
            speak_with_powershell(text)
    except Exception as exc:
        raise RuntimeError("RoomAI could not speak the reply.") from exc


def check_tts_available() -> tuple[bool, str]:
    try:
        backend, _ = create_tts_engine()
    except RuntimeError as exc:
        return (False, str(exc))
    return (True, f"TTS backend available: {backend}")


class SpeechService:
    def __init__(self, config: AppConfig):
        self.config = config
        self._whisper_model: Any | None = None

    def get_whisper_model(self) -> Any:
        if self._whisper_model is None:
            self._whisper_model = create_whisper_model(self.config)
        return self._whisper_model

    def transcribe_microphone(self) -> str:
        return transcribe_microphone_audio(self.get_whisper_model(), self.config)

    def transcribe_file(self, audio_path: Path) -> str:
        return transcribe_audio_file(self.get_whisper_model(), audio_path, self.config)

    def diagnostics(self) -> list[str]:
        mic_ok, mic_message = check_microphone_available()
        whisper_ready = False
        try:
            self.get_whisper_model()
            whisper_ready = True
            whisper_message = f"Whisper model '{self.config.whisper_model}' is ready."
        except RuntimeError as exc:
            whisper_message = str(exc)
        tts_ok, tts_message = check_tts_available()
        return [
            f"Microphone: {'OK' if mic_ok else 'WARN'} - {mic_message}",
            f"Whisper: {'OK' if whisper_ready else 'WARN'} - {whisper_message}",
            f"TTS: {'OK' if tts_ok else 'WARN'} - {tts_message}",
        ]


class StreamingSpeaker:
    def __init__(self, tts_backend: str, engine: Any):
        self.tts_backend = tts_backend
        self.engine = engine
        self._worker_engine: Any = None
        self._queue: Queue[str | None] = Queue()
        self._buffer = ""
        self._stop_requested = threading.Event()
        self._thread = threading.Thread(target=self._worker, daemon=True)
        self._thread.start()

    def feed(self, chunk: str) -> None:
        if not chunk:
            return
        self._buffer += chunk
        parts = re.split(r"(?<=[.!?])\s+|\n+", self._buffer)
        if len(parts) <= 1:
            return
        self._buffer = parts[-1]
        for part in parts[:-1]:
            text = part.strip()
            if text:
                self._queue.put(text)

    def finish(self) -> None:
        if self._stop_requested.is_set():
            self._buffer = ""
            self._queue.put(None)
            self._thread.join(timeout=10)
            return
        remaining = self._buffer.strip()
        if remaining:
            self._queue.put(remaining)
        self._buffer = ""
        self._queue.put(None)
        self._thread.join(timeout=10)

    def stop(self) -> None:
        self._stop_requested.set()
        self._buffer = ""
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except Exception:
                break
        if self.tts_backend == "pyttsx3" and self.engine is not None:
            try:
                self.engine.stop()
            except Exception:
                pass
        if self.tts_backend == "pyttsx3" and self._worker_engine is not None:
            try:
                self._worker_engine.stop()
            except Exception:
                pass
        self._queue.put(None)
        self._thread.join(timeout=10)

    def _worker(self) -> None:
        local_engine = self.engine
        if self.tts_backend == "pyttsx3":
            try:
                import pyttsx3

                local_engine = pyttsx3.init()
            except Exception:
                local_engine = self.engine
        self._worker_engine = local_engine

        while True:
            item = self._queue.get()
            if item is None:
                break
            if self._stop_requested.is_set():
                continue
            try:
                speak_text(self.tts_backend, local_engine, item)
            except Exception:
                try:
                    speak_with_powershell(item)
                except Exception:
                    pass
