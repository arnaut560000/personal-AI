from __future__ import annotations

from fastapi import APIRouter, HTTPException, Request

from app.models.schemas import FactCreate, MemorySummaryResponse, MemoryWriteResponse, NoteCreate
from app.services.ai_service import AIService


router = APIRouter(tags=["memory"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.get("/memory", response_model=MemorySummaryResponse)
def get_memory(request: Request) -> MemorySummaryResponse:
    return MemorySummaryResponse(**_service(request).memory_service.get_summary())


@router.post("/memory/note", response_model=MemoryWriteResponse)
def save_note(payload: NoteCreate, request: Request) -> MemoryWriteResponse:
    _service(request).memory_service.save_note(payload.content)
    return MemoryWriteResponse(status="saved", message="Note saved.")


@router.post("/memory/fact", response_model=MemoryWriteResponse)
def save_fact(payload: FactCreate, request: Request) -> MemoryWriteResponse:
    _service(request).memory_service.set_fact(payload.key, payload.value)
    return MemoryWriteResponse(status="saved", message="Fact saved.")


@router.delete("/memory/fact/{key}", response_model=MemoryWriteResponse)
def delete_fact(key: str, request: Request) -> MemoryWriteResponse:
    deleted = _service(request).memory_service.delete_fact(key)
    if not deleted:
        raise HTTPException(status_code=404, detail="Saved fact not found.")
    return MemoryWriteResponse(status="deleted", message=f'Fact "{key}" deleted.')
