from __future__ import annotations

from fastapi import APIRouter, Request

from app.models.schemas import ClientLocationRequest, ClientLocationResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["location"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.post("/location/client", response_model=ClientLocationResponse)
def save_client_location(payload: ClientLocationRequest, request: Request) -> ClientLocationResponse:
    location = _service(request).location_service.save_client_location(
        latitude=payload.latitude,
        longitude=payload.longitude,
        accuracy_meters=payload.accuracy,
    )
    return ClientLocationResponse(
        status="saved",
        message="Client GPS location saved.",
        location=location,
    )
