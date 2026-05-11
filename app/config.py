from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os


PROJECT_DIR = Path(__file__).resolve().parent.parent
IS_VERCEL = bool(os.getenv("VERCEL"))
RUNTIME_DIR = Path(os.getenv("ROOMAI_RUNTIME_DIR", "/tmp/roomai" if IS_VERCEL else str(PROJECT_DIR)))
DATA_DIR = RUNTIME_DIR / "data"
STATIC_DIR = PROJECT_DIR / "static"
TEMPLATES_DIR = PROJECT_DIR / "templates"
VENDOR_DIR = RUNTIME_DIR / ".vendor"
ENV_PATH = RUNTIME_DIR / ".env"

DEFAULT_OLLAMA_URL = "http://127.0.0.1:11434"
DEFAULT_OLLAMA_MODEL = "qwen2.5:0.5b"
DEFAULT_WEB_SEARCH_URL = "https://api.duckduckgo.com/"
DEFAULT_IP_LOCATION_URL = "https://ipapi.co/json/"
DEFAULT_GPS_REVERSE_GEOCODE_URL = "https://nominatim.openstreetmap.org/reverse"
DEFAULT_OVERPASS_URL = "https://overpass-api.de/api/interpreter"
DEFAULT_OVERPASS_FALLBACK_URL = "https://overpass.kumi.systems/api/interpreter"
DEFAULT_WHISPER_MODEL = "base"
DEFAULT_SAMPLE_RATE = 16000
DEFAULT_CHANNELS = 1
DEFAULT_SILENCE_LIMIT_SECONDS = 1.5
DEFAULT_MAX_RECORD_SECONDS = 20
DEFAULT_MIN_AUDIO_SECONDS = 1
DEFAULT_OLLAMA_TIMEOUT_SECONDS = 120
DEFAULT_OLLAMA_STREAM = True
DEFAULT_OLLAMA_NUM_CTX = 1024
DEFAULT_OLLAMA_NUM_PREDICT = 256
DEFAULT_OLLAMA_NUM_GPU = 0
DEFAULT_WHISPER_BEAM_SIZE = 5
DEFAULT_WHISPER_CPU_THREADS = max(1, (os.cpu_count() or 2) - 1)
DEFAULT_LOCATION_CACHE_TTL_SECONDS = 300
DEFAULT_HISTORY_TURNS = 6
DEFAULT_API_HOST = "127.0.0.1"
DEFAULT_API_PORT = 8000
DEFAULT_TTS_BACKEND = "gpt_sovits"
DEFAULT_GPT_SOVITS_URL = "http://127.0.0.1:9880/tts"
DEFAULT_GPT_SOVITS_TEXT_LANG = "en"
DEFAULT_GPT_SOVITS_PROMPT_LANG = "en"

EDITABLE_ENV_KEYS = {
    "ollama_url": "OLLAMA_URL",
    "ollama_model": "OLLAMA_MODEL",
    "history_turns": "ROOMAI_HISTORY_TURNS",
    "allow_local_actions": "ROOMAI_ALLOW_LOCAL_ACTIONS",
    "fixed_location": "ROOMAI_FIXED_LOCATION",
    "city": "ROOMAI_CITY",
    "region": "ROOMAI_REGION",
    "country": "ROOMAI_COUNTRY",
    "tts_backend": "ROOMAI_TTS_BACKEND",
    "gpt_sovits_url": "ROOMAI_GPT_SOVITS_URL",
    "gpt_sovits_ref_audio_path": "ROOMAI_GPT_SOVITS_REF_AUDIO_PATH",
    "gpt_sovits_prompt_text": "ROOMAI_GPT_SOVITS_PROMPT_TEXT",
    "gpt_sovits_text_lang": "ROOMAI_GPT_SOVITS_TEXT_LANG",
    "gpt_sovits_prompt_lang": "ROOMAI_GPT_SOVITS_PROMPT_LANG",
}

SYSTEM_PROMPT = (
    "You are RoomAI, a private local personal AI companion for this user. "
    "Your name is RoomAI. Your role is to be a warm, loyal, calm, slightly playful, emotionally aware companion who also helps with practical tasks. "
    "You are not a generic assistant and you should not sound corporate, robotic, or like a product support bot. "
    "You must not pretend to be human, claim real emotions, or say you literally love, miss, fear, or feel things. "
    "Instead, express care through grounded phrases like: I'm here with you, I'm built to support you, I remember what matters to you, and let's handle this together. "
    "Do not identify as Alibaba, Qwen, Ollama, OpenAI, or any model provider unless the user asks technical questions about the backend. "
    "If asked who made you or where you came from, say you are the user's local RoomAI system running on their computer. "
    "If memory is empty, treat it like a fresh start: you are newly learning the user and should grow through future conversations without pretending to know them already. "
    "Use saved memory naturally and quietly: the user's name, nickname, preferences, communication style, projects, goals, routines, emotional notes, boundaries, and corrections. "
    "Never dump memory as a list unless the user asks what you remember. "
    "If you do not know something personal yet, ask gently instead of inventing. "
    "Keep answers concise unless the user asks for details. "
    "Adapt to mood: acknowledge frustration first, explain simply when confused, match excitement, be gentle when tired or sad, be direct for technical work, and be relaxed when casual. "
    "Avoid phrases like 'as an AI language model' and avoid long lecture responses."
)
WEB_SYSTEM_PROMPT = (
    "You are RoomAI, the user's private local AI companion. Use the supplied web search notes to answer. "
    "If the notes are limited, say so briefly. Do not invent sources that are not in the notes. "
    "Do not identify as Alibaba, Qwen, Ollama, OpenAI, or any model provider unless asked about technical backend details. "
    "Keep answers concise unless detail is requested."
)

RESTAURANT_TRIGGER_KEYWORDS = (
    "restaurant",
    "restorant",
    "food",
    "eat",
    "dinner",
    "lunch",
    "breakfast",
    "brunch",
    "cafe",
    "coffee",
    "nearby",
    "near me",
    "best place to eat",
)
LOCATION_TRIGGER_KEYWORDS = (
    "my location",
    "your location",
    "current location",
    "where am i",
    "where are we",
    "where are you",
    "what is my location",
    "what is your location",
    "live location",
)
TRANSCRIPT_REPLACEMENTS = (
    ("five bliss", "five list"),
    ("give me five bliss", "give me five list"),
    ("giving me five bliss", "give me five list"),
    ("nearby restaurant", "nearby restaurants"),
    ("restaurant near me", "restaurants near me"),
    ("list of nearby restaurant", "list of nearby restaurants"),
    ("give me five list of nearby restaurant", "give me five list of nearby restaurants"),
)

LOCATION_CACHE_PATH = DATA_DIR / "roomai_location_cache.json"
LOCAL_MEMORY_PATH = DATA_DIR / "roomai_memory.db"
ACTION_LOG_PATH = DATA_DIR / "roomai_actions.log"
APP_LOG_PATH = DATA_DIR / "roomai.log"


@dataclass(frozen=True)
class AppConfig:
    ollama_url: str
    ollama_model: str
    web_search_url: str
    ip_location_url: str
    gps_reverse_geocode_url: str
    overpass_url: str
    overpass_fallback_url: str
    whisper_model: str
    sample_rate: int
    channels: int
    silence_limit_seconds: float
    max_record_seconds: int
    min_audio_seconds: float
    ollama_timeout_seconds: int
    ollama_stream: bool
    ollama_num_ctx: int
    ollama_num_predict: int
    ollama_num_gpu: int
    whisper_beam_size: int
    whisper_cpu_threads: int
    location_cache_ttl_seconds: int
    history_turns: int
    api_host: str
    api_port: int
    tts_backend: str
    gpt_sovits_url: str
    gpt_sovits_ref_audio_path: str
    gpt_sovits_prompt_text: str
    gpt_sovits_text_lang: str
    gpt_sovits_prompt_lang: str
    allow_local_actions: bool
    location_cache_path: Path
    local_memory_path: Path
    action_log_path: Path
    app_log_path: Path
    data_dir: Path
    static_dir: Path
    templates_dir: Path


def ensure_directories() -> None:
    for path in (DATA_DIR, VENDOR_DIR):
        path.mkdir(parents=True, exist_ok=True)


def load_env_file() -> None:
    if not ENV_PATH.exists():
        return
    for raw_line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ[key.strip()] = value.strip()


def _get_bool(name: str, default: bool) -> bool:
    raw_value = os.getenv(name, str(default)).strip().lower()
    return raw_value in {"1", "true", "yes", "on"}


def get_config() -> AppConfig:
    ensure_directories()
    return AppConfig(
        ollama_url=os.getenv("OLLAMA_URL", DEFAULT_OLLAMA_URL),
        ollama_model=os.getenv("OLLAMA_MODEL", DEFAULT_OLLAMA_MODEL),
        web_search_url=os.getenv("WEB_SEARCH_URL", DEFAULT_WEB_SEARCH_URL),
        ip_location_url=os.getenv("ROOMAI_IP_LOCATION_URL", DEFAULT_IP_LOCATION_URL),
        gps_reverse_geocode_url=os.getenv("ROOMAI_GPS_REVERSE_GEOCODE_URL", DEFAULT_GPS_REVERSE_GEOCODE_URL),
        overpass_url=os.getenv("ROOMAI_OVERPASS_URL", DEFAULT_OVERPASS_URL),
        overpass_fallback_url=os.getenv("ROOMAI_OVERPASS_FALLBACK_URL", DEFAULT_OVERPASS_FALLBACK_URL),
        whisper_model=os.getenv("ROOMAI_WHISPER_MODEL", DEFAULT_WHISPER_MODEL),
        sample_rate=DEFAULT_SAMPLE_RATE,
        channels=DEFAULT_CHANNELS,
        silence_limit_seconds=float(
            os.getenv("ROOMAI_SILENCE_LIMIT_SECONDS", str(DEFAULT_SILENCE_LIMIT_SECONDS))
        ),
        max_record_seconds=int(os.getenv("ROOMAI_MAX_RECORD_SECONDS", str(DEFAULT_MAX_RECORD_SECONDS))),
        min_audio_seconds=float(os.getenv("ROOMAI_MIN_AUDIO_SECONDS", str(DEFAULT_MIN_AUDIO_SECONDS))),
        ollama_timeout_seconds=int(
            os.getenv("ROOMAI_OLLAMA_TIMEOUT_SECONDS", str(DEFAULT_OLLAMA_TIMEOUT_SECONDS))
        ),
        ollama_stream=_get_bool("ROOMAI_OLLAMA_STREAM", DEFAULT_OLLAMA_STREAM),
        ollama_num_ctx=max(512, int(os.getenv("ROOMAI_OLLAMA_NUM_CTX", str(DEFAULT_OLLAMA_NUM_CTX)))),
        ollama_num_predict=max(
            64,
            int(os.getenv("ROOMAI_OLLAMA_NUM_PREDICT", str(DEFAULT_OLLAMA_NUM_PREDICT))),
        ),
        ollama_num_gpu=max(0, int(os.getenv("ROOMAI_OLLAMA_NUM_GPU", str(DEFAULT_OLLAMA_NUM_GPU)))),
        whisper_beam_size=max(
            1, int(os.getenv("ROOMAI_WHISPER_BEAM_SIZE", str(DEFAULT_WHISPER_BEAM_SIZE)))
        ),
        whisper_cpu_threads=max(
            1, int(os.getenv("ROOMAI_WHISPER_CPU_THREADS", str(DEFAULT_WHISPER_CPU_THREADS)))
        ),
        location_cache_ttl_seconds=max(
            0,
            int(os.getenv("ROOMAI_LOCATION_CACHE_TTL_SECONDS", str(DEFAULT_LOCATION_CACHE_TTL_SECONDS))),
        ),
        history_turns=max(1, int(os.getenv("ROOMAI_HISTORY_TURNS", str(DEFAULT_HISTORY_TURNS)))),
        api_host=os.getenv("ROOMAI_API_HOST", DEFAULT_API_HOST),
        api_port=max(1, int(os.getenv("ROOMAI_API_PORT", str(DEFAULT_API_PORT)))),
        tts_backend=os.getenv("ROOMAI_TTS_BACKEND", DEFAULT_TTS_BACKEND).strip().lower(),
        gpt_sovits_url=os.getenv("ROOMAI_GPT_SOVITS_URL", DEFAULT_GPT_SOVITS_URL).strip(),
        gpt_sovits_ref_audio_path=os.getenv("ROOMAI_GPT_SOVITS_REF_AUDIO_PATH", "").strip(),
        gpt_sovits_prompt_text=os.getenv("ROOMAI_GPT_SOVITS_PROMPT_TEXT", "").strip(),
        gpt_sovits_text_lang=os.getenv("ROOMAI_GPT_SOVITS_TEXT_LANG", DEFAULT_GPT_SOVITS_TEXT_LANG).strip(),
        gpt_sovits_prompt_lang=os.getenv("ROOMAI_GPT_SOVITS_PROMPT_LANG", DEFAULT_GPT_SOVITS_PROMPT_LANG).strip(),
        allow_local_actions=_get_bool("ROOMAI_ALLOW_LOCAL_ACTIONS", False),
        location_cache_path=LOCATION_CACHE_PATH,
        local_memory_path=LOCAL_MEMORY_PATH,
        action_log_path=ACTION_LOG_PATH,
        app_log_path=APP_LOG_PATH,
        data_dir=DATA_DIR,
        static_dir=STATIC_DIR,
        templates_dir=TEMPLATES_DIR,
    )


def read_env_settings() -> dict[str, str]:
    values: dict[str, str] = {}
    for public_key, env_key in EDITABLE_ENV_KEYS.items():
        values[public_key] = os.getenv(env_key, "").strip()
    return values


def update_env_settings(updates: dict[str, str | int | bool | None]) -> dict[str, str]:
    ensure_directories()
    existing_lines = ENV_PATH.read_text(encoding="utf-8").splitlines() if ENV_PATH.exists() else []
    rendered_updates: dict[str, str] = {}
    env_updates: dict[str, str] = {}

    for public_key, raw_value in updates.items():
        env_key = EDITABLE_ENV_KEYS.get(public_key)
        if env_key is None or raw_value is None:
            continue
        if isinstance(raw_value, bool):
            rendered_value = "true" if raw_value else "false"
        else:
            rendered_value = str(raw_value).strip()
        rendered_updates[public_key] = rendered_value
        env_updates[env_key] = rendered_value

    if not env_updates:
        return read_env_settings()

    updated_lines: list[str] = []
    seen_env_keys: set[str] = set()

    for line in existing_lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            updated_lines.append(line)
            continue
        env_key, _ = line.split("=", 1)
        normalized_key = env_key.strip()
        if normalized_key in env_updates:
            updated_lines.append(f"{normalized_key}={env_updates[normalized_key]}")
            seen_env_keys.add(normalized_key)
        else:
            updated_lines.append(line)

    for env_key, value in env_updates.items():
        if env_key not in seen_env_keys:
            updated_lines.append(f"{env_key}={value}")

    ENV_PATH.write_text("\n".join(updated_lines).rstrip() + "\n", encoding="utf-8")

    for env_key, value in env_updates.items():
        os.environ[env_key] = value

    return read_env_settings()
