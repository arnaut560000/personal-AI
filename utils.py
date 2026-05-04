from __future__ import annotations

from pathlib import Path
import json
import logging
from logging.handlers import RotatingFileHandler
import re
from typing import Any
from urllib import error, request

from config import TRANSCRIPT_REPLACEMENTS


def setup_logging(log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    root_logger.setLevel(logging.INFO)
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    file_handler = RotatingFileHandler(log_path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    root_logger.addHandler(stream_handler)


def fetch_json(url: str, timeout: int = 15, user_agent: str = "RoomAI/1.0") -> dict[str, Any]:
    http_request = request.Request(url, headers={"User-Agent": user_agent})
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError("A network request failed while RoomAI was gathering information.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("A service returned unreadable data.") from exc


def fetch_json_post(
    url: str,
    payload: str,
    timeout: int = 20,
    user_agent: str = "RoomAI/1.0",
) -> dict[str, Any]:
    http_request = request.Request(
        url,
        data=payload.encode("utf-8"),
        headers={
            "User-Agent": user_agent,
            "Content-Type": "application/x-www-form-urlencoded; charset=utf-8",
        },
        method="POST",
    )
    try:
        with request.urlopen(http_request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except error.URLError as exc:
        raise RuntimeError("A network request failed while RoomAI was gathering information.") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError("A service returned unreadable data.") from exc


def normalize_transcript(user_text: str) -> str:
    normalized = " ".join(user_text.strip().split())
    lowered = normalized.lower()
    for source_text, replacement_text in TRANSCRIPT_REPLACEMENTS:
        if source_text in lowered:
            start = lowered.index(source_text)
            end = start + len(source_text)
            normalized = normalized[:start] + replacement_text + normalized[end:]
            lowered = normalized.lower()
    return normalized


def contains_trigger_phrase(user_text: str, phrase: str) -> bool:
    """
    Match trigger phrases using word boundaries to avoid substring false positives.
    Example: 'eat' should not match 'great'.
    """
    escaped = re.escape(phrase.strip().lower())
    escaped = escaped.replace(r"\ ", r"\s+")
    pattern = rf"\b{escaped}\b"
    return re.search(pattern, user_text.lower()) is not None
