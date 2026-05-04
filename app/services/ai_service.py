from __future__ import annotations

from typing import Any

from app.config import (
    LOCATION_TRIGGER_KEYWORDS,
    RESTAURANT_TRIGGER_KEYWORDS,
    SYSTEM_PROMPT,
    AppConfig,
    read_env_settings,
)
from app.models.schemas import ChatResponse, FeatureAvailability, SettingsResponse, StatusResponse
from app.services.command_service import CommandService
from app.services.location_service import LocationService
from app.services.memory_service import MemoryService
from app.services.ollama_client import OllamaClient
from app.services.search_service import SearchService
from app.services.speech_service import (
    SpeechService,
    check_microphone_available,
    check_tts_available,
)
from app.utils import normalize_transcript


class AIService:
    def __init__(self, config: AppConfig):
        self.config = config
        self.memory_service = MemoryService(str(config.local_memory_path), max_turns=config.history_turns)
        self.command_service = CommandService(config, self.memory_service)
        self.location_service = LocationService(config)
        self.ollama_client = OllamaClient(config)
        self.search_service = SearchService(config, self.ollama_client, self.location_service)
        self.speech_service = SpeechService(config)
        self.model_name = config.ollama_model

    def startup_checks(self) -> tuple[bool, list[str], str]:
        diagnostics: list[str] = []
        ollama_ok, ollama_message = self.ollama_client.check_health()
        diagnostics.append(f"Ollama: {'OK' if ollama_ok else 'FAIL'} - {ollama_message}")
        if ollama_ok:
            self.model_name = self.ollama_client.resolve_model()
        diagnostics.extend(self.speech_service.diagnostics())
        return ollama_ok, diagnostics, ollama_message

    def get_status(self) -> StatusResponse:
        ollama_ok, diagnostics, ollama_message = self.startup_checks()
        microphone_ok, _ = check_microphone_available()
        tts_ok, _ = check_tts_available()
        return StatusResponse(
            status="ok" if ollama_ok else "degraded",
            backend="running",
            ollama_online=ollama_ok,
            ollama_status=ollama_message,
            model_name=self.model_name,
            memory_path=str(self.config.local_memory_path),
            feature_availability=FeatureAvailability(
                ollama=ollama_ok,
                memory=True,
                commands=True,
                voice_input=microphone_ok,
                text_to_speech=tts_ok,
                web_search=True,
                location=True,
            ),
            diagnostics=diagnostics,
        )

    def get_settings(self) -> SettingsResponse:
        env_settings = read_env_settings()
        return SettingsResponse(
            ollama_url=self.config.ollama_url,
            ollama_model=self.config.ollama_model,
            history_turns=self.config.history_turns,
            allow_local_actions=self.config.allow_local_actions,
            api_host=self.config.api_host,
            api_port=self.config.api_port,
            fixed_location=env_settings.get("fixed_location", ""),
            city=env_settings.get("city", ""),
            region=env_settings.get("region", ""),
            country=env_settings.get("country", ""),
        )

    def _conversation_context(self, session_id: str) -> list[dict[str, str]]:
        memory_context = self.memory_service.build_memory_context()
        history = self.memory_service.get_history(session_id)
        return [
            {
                "role": "system",
                "content": (
                    "Local user memory context from past sessions:\n"
                    f"{memory_context}\n"
                    "Use this context only when relevant. If uncertain, ask for confirmation."
                ),
            },
            *history.to_messages(),
        ]

    def _graceful_chat_response(
        self,
        session_id: str,
        user_text: str,
        message: str,
        handled_by: str,
        metadata: dict[str, Any] | None = None,
    ) -> ChatResponse:
        self.memory_service.add_turn(session_id, user_text, message)
        return ChatResponse(
            session_id=session_id,
            answer=message,
            handled_by=handled_by,
            model=self.model_name,
            success=False,
            metadata=metadata or {},
        )

    def process_text(
        self,
        text: str,
        session_id: str = "default",
        use_web_search: bool = False,
        force_location_refresh: bool = False,
        require_gps: bool = False,
    ) -> ChatResponse:
        clean_text = normalize_transcript(text)

        command_result = self.command_service.process(clean_text)
        if command_result.handled:
            if clean_text and command_result.response:
                self.memory_service.add_turn(session_id, clean_text, command_result.response)
            return ChatResponse(
                session_id=session_id,
                answer=command_result.response,
                handled_by="command_service",
                model=self.model_name,
                success=True,
                metadata=command_result.metadata or {},
            )

        try:
            if clean_text.lower().startswith("web:") or use_web_search:
                query_text = clean_text[4:].strip() if clean_text.lower().startswith("web:") else clean_text
                if not query_text:
                    raise RuntimeError('Please add a question after "web:".')
                answer, web_context = self.search_service.ask_with_web_notes(
                    query_text,
                    history_messages=self._conversation_context(session_id),
                )
                self.memory_service.add_turn(session_id, clean_text, answer)
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    handled_by="search_service",
                    model=self.model_name,
                    success=True,
                    metadata={"web_context": web_context},
                )

            if self.location_service.is_location_request(clean_text, LOCATION_TRIGGER_KEYWORDS):
                location = self.location_service.get_live_location(
                    force_refresh=force_location_refresh,
                    require_gps=require_gps,
                )
                answer = self.location_service.build_location_answer(location)
                self.memory_service.add_turn(session_id, clean_text, answer)
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    handled_by="location_service",
                    model=self.model_name,
                    success=True,
                    metadata={"location": location or {}},
                )

            if self.search_service.is_restaurant_request(clean_text, RESTAURANT_TRIGGER_KEYWORDS):
                location = self.location_service.get_live_location(force_refresh=force_location_refresh)
                if not location:
                    answer = "I could not determine your location well enough to suggest nearby restaurants."
                else:
                    try:
                        restaurants = self.search_service.fetch_nearby_restaurants(location)
                        answer = self.search_service.build_restaurant_answer(restaurants, location)
                    except RuntimeError:
                        answer = self.search_service.fetch_restaurants_from_web_search(
                            clean_text,
                            location,
                            history_messages=self._conversation_context(session_id),
                        )
                self.memory_service.add_turn(session_id, clean_text, answer)
                return ChatResponse(
                    session_id=session_id,
                    answer=answer,
                    handled_by="search_service",
                    model=self.model_name,
                    success=True,
                    metadata={"location": location or {}},
                )

            answer = self.ollama_client.ask(
                system_prompt=SYSTEM_PROMPT,
                user_text=clean_text,
                history_messages=self._conversation_context(session_id),
            )
            self.memory_service.add_turn(session_id, clean_text, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                handled_by="ai_service",
                model=self.model_name,
                success=True,
                metadata={},
            )
        except RuntimeError as exc:
            return self._graceful_chat_response(
                session_id=session_id,
                user_text=clean_text,
                message=(
                    "RoomAI is running, but the AI backend is currently unavailable. "
                    "Please check Ollama and try again."
                ),
                handled_by="fallback",
                metadata={"error": str(exc)},
            )
