from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import StatusResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["status"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.get("/status", response_model=StatusResponse)
def status(request: Request) -> StatusResponse:
    return _service(request).get_status()
