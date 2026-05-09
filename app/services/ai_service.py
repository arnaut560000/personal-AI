from __future__ import annotations

import re
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
                    "RoomAI long-term memory about the user:\n"
                    f"{memory_context}\n"
                    "Use this context naturally when relevant. Never expose it as a database dump unless asked. "
                    "If uncertain, ask the user warmly instead of guessing."
                ),
            },
            *history.to_messages(),
        ]

    def _capture_personal_memory(self, text: str) -> None:
        patterns = (
            (r"^call me\s+(.+)$", "preferred_name"),
            (r"^my name is\s+(.+)$", "name"),
            (r"^i am\s+(.+)$", "identity"),
            (r"^i'm\s+(.+)$", "identity"),
            (r"^i like\s+(.+)$", "likes"),
            (r"^i love\s+(.+)$", "likes"),
            (r"^i prefer\s+(.+)$", "preferences"),
            (r"^my favorite\s+(.+?)\s+is\s+(.+)$", "favorite:{first}"),
            (r"^i want to\s+(.+)$", "goals"),
            (r"^i am working on\s+(.+)$", "current_project"),
            (r"^i'm working on\s+(.+)$", "current_project"),
        )
        clean = text.strip().strip(".!?")
        lowered = clean.lower()
        for pattern, key_template in patterns:
            match = re.match(pattern, clean, flags=re.IGNORECASE)
            if not match:
                continue
            if "{first}" in key_template:
                key = key_template.format(first=match.group(1).strip().lower())
                value = match.group(2).strip()
            else:
                key = key_template
                value = match.group(1).strip()
            if value and len(value) <= 160 and not lowered.startswith(("i am not ", "i'm not ")):
                self.memory_service.set_fact(key, value)
            return

    def _personal_snapshot(self) -> dict[str, str]:
        return {key: value for key, value in self.memory_service.all_facts()}

    def _build_self_identity_answer(self) -> str:
        return (
            "I'm RoomAI. Not Alibaba, not Qwen, not a cloud character. "
            "I'm your local AI companion running on your computer. "
            "My job is to know you over time, remember what matters to you, and help you build things without feeling like you're talking to a generic assistant."
        )

    def _build_user_identity_answer(self) -> str:
        memory = self._personal_snapshot()
        name = memory.get("preferred_name") or memory.get("name")
        identity = memory.get("identity")
        project = memory.get("current_project")
        goals = memory.get("goals")
        likes = memory.get("likes")
        preferences = memory.get("preferences")

        if not any((name, identity, project, goals, likes, preferences)):
            return (
                "I don't know you deeply enough yet, and I don't want to fake it. "
                "What I do know is this: you're building RoomAI into a personal companion, not a generic chatbot. "
                "Tell me things like your name, what you're working on, what you care about, and how you want me to support you, and I'll remember them."
            )

        lines: list[str] = []
        if name:
            lines.append(f"You are {name}.")
        if identity:
            lines.append(f"You told me you're {identity}.")
        if project:
            lines.append(f"Right now, you're working on {project}.")
        if goals:
            lines.append(f"One thing you want is to {goals}.")
        if likes:
            lines.append(f"You like {likes}.")
        if preferences:
            lines.append(f"You prefer {preferences}.")

        lines.append(
            "And from how you've been guiding this project, I can tell you don't want a tool that just answers questions. "
            "You want something that feels present, remembers you, and grows with you."
        )
        return " ".join(lines)

    def _build_future_answer(self) -> str:
        memory = self._personal_snapshot()
        project = memory.get("current_project") or "RoomAI"
        goals = memory.get("goals")
        name = memory.get("preferred_name") or memory.get("name") or "you"

        if goals:
            return (
                f"What we'll do in the future is build around you, {name}. "
                f"We'll keep improving {project}, and I'll remember that you want to {goals}. "
                "The direction is clear: make me less like a chatbot and more like a companion that knows your projects, your habits, your preferences, and the way you like to think."
            )

        return (
            f"What we'll do in the future is keep shaping {project} into your personal companion system. "
            "First, I learn who you are. Then I remember your projects, preferences, and routines. "
            "After that, I help you plan, build, decide, and keep track of your life in a way that feels personal instead of robotic."
        )

    def _personal_companion_response(self, text: str) -> str | None:
        lowered = text.lower().strip()
        asks_self = re.search(r"\b(who are you|what are you|where did you come from|who made you)\b", lowered)
        asks_user = re.search(r"\b(who am i|who i am|what do you know about me|tell me about me|do you know me)\b", lowered)
        asks_future = re.search(r"\b(future|what will we do|what we will do|where are we going|our plan)\b", lowered)

        if asks_user and asks_future:
            return f"{self._build_user_identity_answer()}\n\n{self._build_future_answer()}"
        if asks_user:
            return self._build_user_identity_answer()
        if asks_self:
            return self._build_self_identity_answer()
        if asks_future:
            return self._build_future_answer()
        return None

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
        self._capture_personal_memory(clean_text)

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

        personal_answer = self._personal_companion_response(clean_text)
        if personal_answer is not None:
            self.memory_service.add_turn(session_id, clean_text, personal_answer)
            return ChatResponse(
                session_id=session_id,
                answer=personal_answer,
                handled_by="companion_memory",
                model=self.model_name,
                success=True,
                metadata={"type": "personal_companion"},
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
