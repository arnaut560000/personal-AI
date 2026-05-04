from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import CommandRequest, CommandResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["command"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.post("/command", response_model=CommandResponse)
def execute_command(payload: CommandRequest, request: Request) -> CommandResponse:
    result = _service(request).command_service.process(payload.text)
    return CommandResponse(
        handled=result.handled,
        response=result.response,
        should_exit=result.should_exit,
        metadata=result.metadata or {},
    )
