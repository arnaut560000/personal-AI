from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["chat"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return _service(request).process_text(
        text=payload.text,
        session_id=payload.session_id,
        use_web_search=payload.use_web_search,
        force_location_refresh=payload.force_location_refresh,
        require_gps=payload.require_gps,
    )
