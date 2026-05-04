from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    text: str = Field(..., min_length=1)
    session_id: str = Field(default="default", min_length=1)
    use_web_search: bool = False
    force_location_refresh: bool = False
    require_gps: bool = False


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    handled_by: str
    model: str
    success: bool = True
    metadata: dict[str, Any] = Field(default_factory=dict)


class FeatureAvailability(BaseModel):
    ollama: bool
    memory: bool
    commands: bool
    voice_input: bool
    text_to_speech: bool
    web_search: bool
    location: bool


class StatusResponse(BaseModel):
    status: str
    backend: str
    ollama_online: bool
    ollama_status: str
    model_name: str
    memory_path: str
    feature_availability: FeatureAvailability
    diagnostics: list[str]


class NoteCreate(BaseModel):
    content: str = Field(..., min_length=1)


class FactCreate(BaseModel):
    key: str = Field(..., min_length=1)
    value: str = Field(..., min_length=1)


class MemorySummaryResponse(BaseModel):
    notes: list[dict[str, Any]]
    facts: list[dict[str, str]]
    counts: dict[str, int]
    summary: str


class MemoryWriteResponse(BaseModel):
    status: str
    message: str


class CommandRequest(BaseModel):
    text: str = Field(..., min_length=1)


class CommandResponse(BaseModel):
    handled: bool
    response: str
    should_exit: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class SettingsResponse(BaseModel):
    ollama_url: str
    ollama_model: str
    history_turns: int
    allow_local_actions: bool
    api_host: str
    api_port: int
    fixed_location: str
    city: str
    region: str
    country: str


class SettingsUpdateRequest(BaseModel):
    ollama_url: str | None = None
    ollama_model: str | None = None
    history_turns: int | None = Field(default=None, ge=1, le=50)
    allow_local_actions: bool | None = None
    fixed_location: str | None = None
    city: str | None = None
    region: str | None = None
    country: str | None = None


class SettingsUpdateResponse(BaseModel):
    status: str
    message: str
    settings: SettingsResponse


class ClientLocationRequest(BaseModel):
    latitude: float = Field(..., ge=-90, le=90)
    longitude: float = Field(..., ge=-180, le=180)
    accuracy: float | None = Field(default=None, ge=0)


class ClientLocationResponse(BaseModel):
    status: str
    message: str
    location: dict[str, Any]
