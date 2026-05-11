from __future__ import annotations

from pathlib import Path
from queue import Queue
import html
import json
import shutil
import re
import subprocess
import tempfile
import threading
from urllib import request as urlrequest
from urllib.error import URLError
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


def _powershell_quote(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _pitch_shift_wav(source_path: Path, target_path: Path, pitch_factor: float = 1.12) -> None:
    """Create a brighter, younger-sounding WAV by raising playback pitch locally."""
    try:
        import numpy as np
    except Exception:
        shutil.copyfile(source_path, target_path)
        return

    with wave.open(str(source_path), "rb") as source:
        channels = source.getnchannels()
        sample_width = source.getsampwidth()
        frame_rate = source.getframerate()
        frame_count = source.getnframes()
        frames = source.readframes(frame_count)

    if sample_width != 2:
        shutil.copyfile(source_path, target_path)
        return

    audio = np.frombuffer(frames, dtype=np.int16)
    if channels > 1:
        audio = audio.reshape(-1, channels)

    sample_count = audio.shape[0]
    shifted_count = max(1, int(sample_count / pitch_factor))
    old_positions = np.arange(sample_count)
    new_positions = np.linspace(0, sample_count - 1, shifted_count)

    if channels > 1:
        shifted_channels = [
            np.interp(new_positions, old_positions, audio[:, channel])
            for channel in range(channels)
        ]
        shifted = np.stack(shifted_channels, axis=1)
    else:
        shifted = np.interp(new_positions, old_positions, audio)

    shifted = np.clip(shifted, -32768, 32767).astype(np.int16)
    with wave.open(str(target_path), "wb") as target:
        target.setnchannels(channels)
        target.setsampwidth(sample_width)
        target.setframerate(frame_rate)
        target.writeframes(shifted.tobytes())


def _wav_has_sound(audio: bytes, minimum_peak: int = 200) -> bool:
    """Return False when a generated WAV is only silence or too quiet to hear."""
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_file:
            temp_path = Path(temp_file.name)
            temp_file.write(audio)
        try:
            with wave.open(str(temp_path), "rb") as wav_file:
                sample_width = wav_file.getsampwidth()
                frame_count = wav_file.getnframes()
                frames = wav_file.readframes(frame_count)
        finally:
            temp_path.unlink(missing_ok=True)
    except Exception:
        return True

    if sample_width != 2 or not frames:
        return True

    import struct

    sample_count = len(frames) // 2
    if sample_count == 0:
        return False
    samples = struct.unpack("<" + "h" * sample_count, frames)
    return max(abs(sample) for sample in samples) >= minimum_peak


def synthesize_with_powershell_wav(text: str) -> bytes:
    """Generate a local Windows WAV file without opening any extra terminal windows."""
    with tempfile.TemporaryDirectory(prefix="roomai_voice_") as temp_dir:
        temp_path = Path(temp_dir)
        raw_wav = temp_path / "roomai_raw.wav"
        cute_wav = temp_path / "roomai_cute.wav"
        text_path = temp_path / "roomai_text.txt"
        text_path.write_text(text, encoding="utf-8")
        command = (
            "Add-Type -AssemblyName System.Speech; "
            "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            "$speaker.Rate = -1; "
            "$speaker.Volume = 95; "
            "try { $speaker.SelectVoice('Microsoft Zira Desktop') } catch { "
            "try { $speaker.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) } catch { } }; "
            f"$text = Get-Content -Raw -LiteralPath {_powershell_quote(str(text_path))}; "
            f"$speaker.SetOutputToWaveFile({_powershell_quote(str(raw_wav))}); "
            "$speaker.Speak($text); "
            "$speaker.SetOutputToNull();"
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
        return raw_wav.read_bytes()


def speak_with_powershell(text: str, childlike: bool = True) -> None:
    escaped_text = text.replace("'", "''")
    escaped_ssml_text = html.escape(text, quote=False).replace("'", "''")
    ssml = (
        "<speak version='1.0' xml:lang='en-US' "
        "xmlns='http://www.w3.org/2001/10/synthesis'>"
        "<prosody pitch='+18%' rate='-8%' volume='soft'>"
        f"{escaped_ssml_text}"
        "</prosody>"
        "</speak>"
    ).replace("'", "''")

    if childlike:
        with tempfile.TemporaryDirectory(prefix="roomai_voice_") as temp_dir:
            temp_path = Path(temp_dir)
            raw_wav = temp_path / "roomai_raw.wav"
            cute_wav = temp_path / "roomai_cute.wav"
            text_path = temp_path / "roomai_text.txt"
            text_path.write_text(text, encoding="utf-8")
            command = (
                "Add-Type -AssemblyName System.Speech; "
                "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
                "$speaker.Rate = -1; "
                "$speaker.Volume = 88; "
                "try { $speaker.SelectVoice('Microsoft Zira Desktop') } catch { "
                "try { $speaker.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) } catch { } }; "
                f"$text = Get-Content -Raw -LiteralPath {_powershell_quote(str(text_path))}; "
                f"$speaker.SetOutputToWaveFile({_powershell_quote(str(raw_wav))}); "
                "$speaker.Speak($text); "
                "$speaker.SetOutputToNull();"
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
            play_result = subprocess.run(
                [
                    "powershell",
                    "-Command",
                    (
                        "Add-Type -AssemblyName System.Windows.Forms; "
                        f"$player = New-Object System.Media.SoundPlayer({_powershell_quote(str(raw_wav))}); "
                        "$player.PlaySync();"
                    ),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            if play_result.returncode != 0:
                error_output = play_result.stderr.strip() or play_result.stdout.strip() or "Unknown playback error."
                raise RuntimeError(f"Windows speech playback failed: {error_output}")
            return

    command = (
        "Add-Type -AssemblyName System.Speech; "
        "$speaker = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
        "$speaker.Rate = -1; "
        "$speaker.Volume = 88; "
        "try { $speaker.SelectVoiceByHints([System.Speech.Synthesis.VoiceGender]::Female) } catch { }; "
        f"$plain = '{escaped_text}'; "
        f"$ssml = '{ssml}'; "
        "try { $speaker.SpeakSsml($ssml) } catch { $speaker.Speak($plain) }"
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


def configure_cute_voice(engine: Any) -> None:
    """Tune local TTS toward a brighter, softer companion voice when the OS allows it."""
    try:
        voices = engine.getProperty("voices") or []
        preferred_voice = None
        preferred_markers = ("zira", "female", "hazel", "susan", "eva", "aria", "jenny", "girl", "child")
        for voice in voices:
            voice_text = " ".join(
                str(getattr(voice, attr, "")).lower()
                for attr in ("id", "name", "gender", "languages")
            )
            if any(marker in voice_text for marker in preferred_markers):
                preferred_voice = voice
                break
        if preferred_voice is not None:
            engine.setProperty("voice", preferred_voice.id)
    except Exception:
        pass

    # pyttsx3 has no reliable pitch control, so keep this fallback calm and readable.
    for property_name, value in (("rate", 165), ("volume", 0.88)):
        try:
            engine.setProperty(property_name, value)
        except Exception:
            pass


def synthesize_with_gpt_sovits(config: AppConfig, text: str) -> bytes:
    """Call a local GPT-SoVITS API server and return generated WAV audio bytes."""
    if config.tts_backend != "gpt_sovits":
        raise RuntimeError("GPT-SoVITS TTS is not enabled.")
    if not config.gpt_sovits_ref_audio_path or not config.gpt_sovits_prompt_text:
        raise RuntimeError(
            "GPT-SoVITS needs ROOMAI_GPT_SOVITS_REF_AUDIO_PATH and ROOMAI_GPT_SOVITS_PROMPT_TEXT."
        )

    payload = {
        "text": text,
        "text_lang": config.gpt_sovits_text_lang,
        "ref_audio_path": config.gpt_sovits_ref_audio_path,
        "prompt_text": config.gpt_sovits_prompt_text,
        "prompt_lang": config.gpt_sovits_prompt_lang,
        "text_split_method": "cut5",
        "batch_size": 1,
        "media_type": "wav",
        "streaming_mode": False,
        "speed_factor": 0.88,
    }
    data = json.dumps(payload).encode("utf-8")
    api_request = urlrequest.Request(
        config.gpt_sovits_url,
        data=data,
        headers={"Content-Type": "application/json", "Accept": "audio/wav"},
        method="POST",
    )
    try:
        with urlrequest.urlopen(api_request, timeout=90) as response:
            audio = response.read()
    except URLError as exc:
        raise RuntimeError("GPT-SoVITS server is not reachable. Start api_v2.py on port 9880.") from exc

    if not audio:
        raise RuntimeError("GPT-SoVITS returned empty audio.")
    if not _wav_has_sound(audio):
        raise RuntimeError("GPT-SoVITS returned silent audio.")
    return audio


def create_tts_engine() -> tuple[str, Any]:
    powershell_error: str | None = None
    try:
        speak_with_powershell("RoomAI voice is ready.")
        return ("powershell", None)
    except RuntimeError as exc:
        powershell_error = str(exc)

    pyttsx3_import_error: str | None = None
    pyttsx3_init_error: str | None = None
    try:
        import pyttsx3
    except Exception as exc:
        pyttsx3_import_error = str(exc)
    else:
        try:
            engine = pyttsx3.init()
            configure_cute_voice(engine)
            return ("pyttsx3", engine)
        except Exception as exc:
            pyttsx3_init_error = str(exc)

    details = []
    if powershell_error:
        details.append(f"Windows SSML speech failed: {powershell_error}")
    if pyttsx3_import_error:
        details.append(f"pyttsx3 import failed: {pyttsx3_import_error}")
    if pyttsx3_init_error:
        details.append(f"pyttsx3 init failed: {pyttsx3_init_error}")
    raise RuntimeError("Text-to-speech is unavailable. " + " | ".join(details))


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

    def synthesize_reply_audio(self, text: str) -> bytes:
        # Prefer GPT-SoVITS for the custom companion voice, but never send a silent
        # file to the browser. Some incomplete GPT-SoVITS setups return a valid
        # one-second WAV with no samples, so RoomAI falls back to Windows TTS.
        if self.config.tts_backend == "gpt_sovits":
            try:
                return synthesize_with_gpt_sovits(self.config, text)
            except RuntimeError:
                return synthesize_with_powershell_wav(text)
        return synthesize_with_powershell_wav(text)

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
        if self.config.tts_backend == "gpt_sovits":
            if self.config.gpt_sovits_ref_audio_path and self.config.gpt_sovits_prompt_text:
                tts_message = f"GPT-SoVITS configured at {self.config.gpt_sovits_url}."
            else:
                tts_message = "GPT-SoVITS selected, but reference audio or prompt text is missing."
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
                configure_cute_voice(local_engine)
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
