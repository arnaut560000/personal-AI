from __future__ import annotations

from queue import Queue
import re
import subprocess
import threading
from typing import Any

from audio import record_until_silence
from config import AppConfig


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


def transcribe_microphone_audio(whisper_model: Any, config: AppConfig) -> str:
    audio_path = record_until_silence(config)
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
        transcript = " ".join(segment.text.strip() for segment in segments).strip()
    finally:
        audio_path.unlink(missing_ok=True)
    if not transcript:
        raise RuntimeError("I heard audio, but I could not understand the words. Please try again clearly.")
    return transcript


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
        backend, engine = create_tts_engine()
    except RuntimeError as exc:
        return (False, str(exc))
    return (True, f"TTS backend available: {backend}")


class StreamingSpeaker:
    """Speaks incoming streamed text in sentence-sized chunks."""

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
        # Drain any queued but unsaid chunks.
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
            # pyttsx3 is more reliable when initialized/used on the same thread.
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
                # Last-resort fallback so speech still works even if the preferred backend fails mid-stream.
                try:
                    speak_with_powershell(item)
                except Exception:
                    pass
