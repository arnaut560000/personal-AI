from __future__ import annotations

from fastapi import APIRouter, Request

from app.config import get_config, update_env_settings
from app.models.schemas import SettingsResponse, SettingsUpdateRequest, SettingsUpdateResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["settings"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.get("/settings", response_model=SettingsResponse)
def get_settings(request: Request) -> SettingsResponse:
    return _service(request).get_settings()


@router.post("/settings", response_model=SettingsUpdateResponse)
def update_settings(payload: SettingsUpdateRequest, request: Request) -> SettingsUpdateResponse:
    updates = payload.model_dump(exclude_none=True)
    update_env_settings(updates)

    refreshed_config = get_config()
    request.app.state.config = refreshed_config
    request.app.state.ai_service = AIService(refreshed_config)

    return SettingsUpdateResponse(
        status="updated",
        message="Settings updated successfully.",
        settings=request.app.state.ai_service.get_settings(),
    )
