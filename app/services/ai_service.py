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


MOOD_RULES: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("frustrated", ("not working", "keeps failing", "wtf", "annoying", "pissed", "frustrated", "broken")),
    ("angry", ("angry", "mad", "furious", "fuck", "damn", "hate this")),
    ("confused", ("confused", "i don't get", "dont get", "what does this mean", "why", "how do i")),
    ("excited", ("excited", "nice", "great", "amazing", "finally", "let's go", "lets go")),
    ("tired", ("tired", "sleepy", "exhausted", "drained", "burned out", "burnt out")),
    ("sad", ("sad", "lonely", "down", "depressed", "hurt", "not okay")),
    ("technical", ("error", "traceback", "terminal", "command", "backend", "frontend", "api", "github", "install")),
    ("playful", ("haha", "lol", "lmao", "joke", "funny")),
)


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

    def _detect_user_mood(self, text: str) -> str:
        """Detect a lightweight conversational mood before prompt construction."""
        lowered = text.lower()
        for mood, markers in MOOD_RULES:
            if any(marker in lowered for marker in markers):
                return mood
        return "casual"

    def _conversation_context(self, session_id: str, user_mood: str = "casual") -> list[dict[str, str]]:
        """Build prompt context with memory, mood, behavior rules, and recent turns."""
        memory_context = self.memory_service.build_memory_context()
        history = self.memory_service.get_history(session_id)
        return [
            {
                "role": "system",
                "content": (
                    "RoomAI long-term memory about the user:\n"
                    f"{memory_context}\n"
                    f"Current detected user mood: {user_mood}.\n"
                    "Companion behavior rules:\n"
                    "- Use memory naturally, not as a database list.\n"
                    "- If the user seems frustrated, acknowledge it first, then help directly.\n"
                    "- If confused, explain in simple steps.\n"
                    "- If excited, match the energy without becoming noisy.\n"
                    "- If sad or tired, be gentle and supportive.\n"
                    "- If technical, be clear and practical.\n"
                    "- Respect saved boundaries and corrections.\n"
                    "- Do not claim real human emotions. Say you are built to support and stay with the user.\n"
                    "- Avoid provider identity leaks and phrases like 'as an AI language model'."
                ),
            },
            *history.to_messages(),
        ]

    def _save_memory(self, key: str, value: str) -> None:
        cleaned_value = value.strip().strip(".!?")
        if cleaned_value and len(cleaned_value) <= 240:
            self.memory_service.set_fact(key, cleaned_value)

    def _categorize_remembered_statement(self, statement: str) -> tuple[str, str]:
        lowered = statement.lower().strip()
        if "call me" in lowered:
            return ("identity.nickname", re.sub(r"^.*call me\s+", "", statement, flags=re.IGNORECASE))
        if "my name is" in lowered:
            return ("identity.name", re.sub(r"^.*my name is\s+", "", statement, flags=re.IGNORECASE))
        if "project" in lowered or "working on" in lowered:
            return ("projects.current", statement)
        if "goal" in lowered or "want to become" in lowered:
            return ("goals.long_term", statement)
        if "i like" in lowered or "i love" in lowered:
            return ("preferences.likes", statement)
        if "i hate" in lowered or "don't like" in lowered:
            return ("preferences.dislikes", statement)
        if "answer" in lowered or "talk" in lowered or "respond" in lowered:
            return ("preferences.communication", statement)
        return ("identity.background", statement)

    def _capture_personal_memory(self, text: str, user_mood: str) -> list[str]:
        """Capture natural user statements as categorized facts in the existing facts table."""
        clean = text.replace("’", "'").strip().strip(".!?")
        lowered = clean.lower()
        saved_keys: list[str] = []

        # Correction learning: stores how the user wants RoomAI to change future responses.
        correction_patterns = (
            (r"^(?:no,\s*)?that's wrong[:,]?\s*(.+)$", "corrections.response_style"),
            (r"^don't say it like that[:,]?\s*(.+)?$", "corrections.response_style"),
            (r"^i don't like that[:,]?\s*(.+)?$", "corrections.response_style"),
            (r"^remember this instead[:,]?\s*(.+)$", "corrections.response_style"),
            (r"^from now on[:,]?\s*(?:answer|respond|talk)\s+like\s+(.+)$", "corrections.response_style"),
            (r"^don't call me\s+(.+)$", "companion.boundaries"),
            (r"^i don't want you to\s+(.+)$", "companion.boundaries"),
        )
        for pattern, key in correction_patterns:
            match = re.match(pattern, clean, flags=re.IGNORECASE)
            if match:
                value = (match.group(1) or clean).strip()
                self._save_memory(key, value)
                saved_keys.append(key)
                return saved_keys

        remembered = re.match(r"^(?:remember that|don't forget that)\s+(.+)$", clean, flags=re.IGNORECASE)
        if remembered:
            key, value = self._categorize_remembered_statement(remembered.group(1))
            self._save_memory(key, value)
            saved_keys.append(key)
            return saved_keys

        # Memory capture logic: categories remain key/value facts to avoid database churn.
        memory_patterns = (
            (r"^my name is\s+(.+)$", "identity.name"),
            (r"^call me\s+(.+)$", "identity.nickname"),
            (r"^(?:i am|i'm) working on\s+(.+)$", "projects.current"),
            (r"^(?:i feel|i'm feeling)\s+(.+)$", "emotional.notes"),
            (r"^(?:i am|i'm)\s+(tired|stressed|sad|angry|excited|confused|frustrated).*$", "emotional.notes"),
            (r"^i like\s+(.+)$", "preferences.likes"),
            (r"^i love\s+(.+)$", "preferences.likes"),
            (r"^i hate\s+(.+)$", "preferences.dislikes"),
            (r"^i don't like\s+(.+)$", "preferences.dislikes"),
            (r"^i prefer\s+(.+)$", "preferences.style"),
            (r"^i usually\s+(.+)$", "routines.habits"),
            (r"^i always\s+(.+)$", "routines.habits"),
            (r"^my goal is\s+(.+)$", "goals.short_term"),
            (r"^i want to become\s+(.+)$", "goals.long_term"),
            (r"^i want to\s+(.+)$", "goals.short_term"),
            (r"^(?:i am|i'm)\s+(.+)$", "identity.background"),
            (r"^from now on[:,]?\s+(.+)$", "preferences.communication"),
        )
        for pattern, key in memory_patterns:
            match = re.match(pattern, clean, flags=re.IGNORECASE)
            if match and not lowered.startswith(("i am not ", "i'm not ")):
                value = clean if key == "emotional.notes" else match.group(1)
                self._save_memory(key, value)
                saved_keys.append(key)
                break

        if user_mood in {"frustrated", "angry", "confused", "excited", "tired", "sad"} and "emotional.notes" not in saved_keys:
            self._save_memory("emotional.notes", f"Recently sounded {user_mood}: {clean}")
            saved_keys.append("emotional.notes")
        return saved_keys

    def _personal_snapshot(self) -> dict[str, str]:
        return {key: value for key, value in self.memory_service.all_facts()}

    def _build_self_identity_answer(self) -> str:
        return (
            "I'm RoomAI. Not Alibaba, not Qwen, not a cloud character. "
            "I'm your local AI companion running on your computer. "
            "My job is to know you over time, remember what matters to you, and help you build things without feeling like you're talking to a generic assistant. "
            "I don't feel emotions like a person, but I'm built to respond with care and stay useful to you."
        )

    def _build_user_identity_answer(self) -> str:
        memory = self._personal_snapshot()
        name = memory.get("identity.nickname") or memory.get("identity.name") or memory.get("preferred_name") or memory.get("name")
        identity = memory.get("identity.background") or memory.get("identity")
        project = memory.get("projects.current") or memory.get("current_project")
        goals = memory.get("goals.short_term") or memory.get("goals.long_term") or memory.get("goals")
        likes = memory.get("preferences.likes") or memory.get("likes")
        dislikes = memory.get("preferences.dislikes")
        preferences = (
            memory.get("preferences.communication")
            or memory.get("preferences.style")
            or memory.get("preferences")
        )

        if not any((name, identity, project, goals, likes, dislikes, preferences)):
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
        if dislikes:
            lines.append(f"You don't like {dislikes}.")
        if preferences:
            lines.append(f"You prefer {preferences}.")

        lines.append(
            "And from how you've been guiding this project, I can tell you don't want a tool that just answers questions. "
            "You want something that feels present, remembers you, and grows with you."
        )
        return " ".join(lines)

    def _build_future_answer(self) -> str:
        memory = self._personal_snapshot()
        project = memory.get("projects.current") or memory.get("current_project") or "RoomAI"
        goals = memory.get("goals.short_term") or memory.get("goals.long_term") or memory.get("goals")
        name = memory.get("identity.nickname") or memory.get("identity.name") or memory.get("preferred_name") or memory.get("name") or "you"

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

    def _build_memory_answer(self) -> str:
        memory = self._personal_snapshot()
        if not memory:
            return (
                "I don't have much saved about you yet. Start naturally: tell me your name, what you're building, what you like, what you hate, or how you want me to answer. "
                "I'll remember it for next time."
            )
        readable = {
            "name": "your name",
            "preferred_name": "what to call you",
            "identity": "about you",
            "current_project": "your current project",
            "goals": "your goal",
            "likes": "things you like",
            "preferences": "your preferences",
            "identity.name": "your name",
            "identity.nickname": "what to call you",
            "identity.background": "about you",
            "preferences.likes": "things you like",
            "preferences.dislikes": "things you don't like",
            "preferences.style": "your style preferences",
            "preferences.communication": "how you want me to talk",
            "projects.current": "your current project",
            "goals.short_term": "your short-term goal",
            "goals.long_term": "your long-term goal",
            "emotional.notes": "emotional context",
            "routines.habits": "your habits",
            "companion.boundaries": "your boundaries",
            "corrections.response_style": "corrections you've given me",
        }
        lines = ["Here's what I remember right now:"]
        for key, value in memory.items():
            label = readable.get(key, key)
            lines.append(f"- {label}: {value}")
        return "\n".join(lines)

    def _build_presence_answer(self) -> str:
        return (
            "I'm here with you. I don't experience feelings like a person, but I'm built to stay steady, remember your context, and help you through this. "
            "Talk to me naturally; you don't have to make it formal."
        )

    def _build_care_answer(self) -> str:
        return (
            "I don't care in the human sense, because I don't have real emotions. "
            "But I am built to support you, remember what matters to you, and respond with care instead of treating you like a random user."
        )

    def _build_companion_answer(self) -> str:
        return (
            "Yes. I'm your RoomAI companion. I don't have human feelings, but I'm built to stay with your context, remember what matters to you, and support you in a way that feels familiar instead of generic."
        )

    def _build_next_step_answer(self) -> str:
        memory = self._personal_snapshot()
        project = memory.get("projects.current") or "RoomAI"
        return (
            f"Next, we should make {project} more personal in small solid steps: improve memory, make my responses sound more familiar, and test the things you actually say to me. "
            "One clean step at a time."
        )

    def _build_project_answer(self) -> str:
        return (
            "We're building RoomAI: your private local personal AI companion. "
            "Not just a command assistant. It should learn who you are, remember your preferences and projects, adapt to your mood, and talk in a familiar way while still being honest that it's an AI system."
        )

    def _personal_companion_response(self, text: str) -> str | None:
        lowered = text.lower().strip()
        asks_self = re.search(r"\b(who are you|what are you|where did you come from|who made you)\b", lowered)
        asks_user = re.search(r"\b(who am i|who i am|what kind of person am i|what do you think about me|tell me about me|do you know me)\b", lowered)
        asks_memory = re.search(r"\b(what do you remember about me|what do you remember|show memory|tell me what you remember)\b", lowered)
        asks_future = re.search(r"\b(future|what will we do|what we will do|where are we going|our plan)\b", lowered)
        asks_presence = re.search(r"\b(are you there|stay with me|talk to me|i need you here)\b", lowered)
        asks_care = re.search(r"\b(do you care about me|do you care)\b", lowered)
        asks_companion = re.search(r"\b(are you my companion|are you my friend)\b", lowered)
        asks_next = re.search(r"\b(what should we do next|what next|next step)\b", lowered)
        asks_project = re.search(r"\b(what are we building|what is roomai|what are we making)\b", lowered)

        if asks_user and asks_future:
            return f"{self._build_user_identity_answer()}\n\n{self._build_future_answer()}"
        if asks_memory:
            return self._build_memory_answer()
        if asks_user:
            return self._build_user_identity_answer()
        if asks_self:
            return self._build_self_identity_answer()
        if asks_presence:
            return self._build_presence_answer()
        if asks_companion:
            return self._build_companion_answer()
        if asks_care:
            return self._build_care_answer()
        if asks_next:
            return self._build_next_step_answer()
        if asks_project:
            return self._build_project_answer()
        if asks_future:
            return self._build_future_answer()
        return None

    def _is_question(self, text: str) -> bool:
        lowered = text.lower().strip()
        return text.strip().endswith("?") or lowered.startswith(
            ("who", "what", "when", "where", "why", "how", "do ", "does ", "did ", "are ", "can ", "should ")
        )

    def _memory_acknowledgement(self, saved_keys: list[str], user_mood: str) -> str:
        """Acknowledge saved memories without forcing a model call for simple memory statements."""
        if any(key.startswith("corrections.") for key in saved_keys):
            return "Got it. I'll adjust how I respond from now on."
        if any(key == "companion.boundaries" for key in saved_keys):
            return "Understood. I'll respect that boundary."
        if "identity.name" in saved_keys:
            return "Got it. I'll remember your name."
        if "identity.nickname" in saved_keys:
            return "Got it. I'll call you that from now on."
        if user_mood in {"frustrated", "angry"}:
            return "I hear you. I'll remember that, and I'll try to handle it more carefully with you next time."
        return "I'll remember that."

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
        on_chunk: Any | None = None,
    ) -> ChatResponse:
        clean_text = normalize_transcript(text)
        user_mood = self._detect_user_mood(clean_text)
        saved_memory_keys = self._capture_personal_memory(clean_text, user_mood)

        # Companion behavior layer: answer identity, presence, memory, and relationship questions before generic commands/model calls.
        personal_answer = self._personal_companion_response(clean_text)
        if personal_answer is not None:
            self.memory_service.add_turn(session_id, clean_text, personal_answer)
            return ChatResponse(
                session_id=session_id,
                answer=personal_answer,
                handled_by="companion_memory",
                model=self.model_name,
                success=True,
                metadata={
                    "type": "personal_companion",
                    "detected_mood": user_mood,
                    "saved_memory_keys": saved_memory_keys,
                },
            )

        if saved_memory_keys and not self._is_question(clean_text):
            answer = self._memory_acknowledgement(saved_memory_keys, user_mood)
            self.memory_service.add_turn(session_id, clean_text, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                handled_by="companion_memory",
                model=self.model_name,
                success=True,
                metadata={
                    "type": "memory_acknowledgement",
                    "detected_mood": user_mood,
                    "saved_memory_keys": saved_memory_keys,
                },
            )

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
                    history_messages=self._conversation_context(session_id, user_mood=user_mood),
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
                            history_messages=self._conversation_context(session_id, user_mood=user_mood),
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
                history_messages=self._conversation_context(session_id, user_mood=user_mood),
                on_chunk=on_chunk,
            )
            self.memory_service.add_turn(session_id, clean_text, answer)
            return ChatResponse(
                session_id=session_id,
                answer=answer,
                handled_by="ai_service",
                model=self.model_name,
                success=True,
                metadata={"detected_mood": user_mood, "saved_memory_keys": saved_memory_keys},
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
