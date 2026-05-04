from __future__ import annotations

from pathlib import Path
import tempfile

from fastapi import APIRouter, File, HTTPException, Request, UploadFile

from app.services.ai_service import AIService


router = APIRouter(tags=["voice"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


def _guess_suffix(upload: UploadFile) -> str:
    filename = upload.filename or ""
    suffix = Path(filename).suffix.strip()
    if suffix:
        return suffix

    content_type = (upload.content_type or "").lower()
    if "webm" in content_type:
        return ".webm"
    if "ogg" in content_type:
        return ".ogg"
    if "wav" in content_type:
        return ".wav"
    if "mpeg" in content_type or "mp3" in content_type:
        return ".mp3"
    return ".bin"


@router.post("/voice/transcribe")
async def transcribe_voice(request: Request, audio: UploadFile = File(...)) -> dict[str, object]:
    suffix = _guess_suffix(audio)
    temp_path: Path | None = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
            temp_path = Path(temp_file.name)
            while True:
                chunk = await audio.read(1024 * 1024)
                if not chunk:
                    break
                temp_file.write(chunk)

        if temp_path.stat().st_size == 0:
            raise HTTPException(status_code=400, detail="Uploaded audio file is empty.")

        transcript = _service(request).speech_service.transcribe_file(temp_path)
        return {
            "success": True,
            "transcript": transcript,
            "filename": audio.filename or temp_path.name,
            "content_type": audio.content_type or "application/octet-stream",
        }
    except HTTPException:
        raise
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail="Voice transcription failed unexpectedly.") from exc
    finally:
        await audio.close()
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
