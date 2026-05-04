from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import sqlite3
import time
from typing import Any


@dataclass
class SessionTurn:
    user: str
    assistant: str
    timestamp: float


class SessionHistory:
    def __init__(self, max_turns: int = 6):
        self.max_turns = max(1, max_turns)
        self._turns: deque[SessionTurn] = deque(maxlen=self.max_turns)

    def add_turn(self, user: str, assistant: str) -> None:
        self._turns.append(SessionTurn(user=user, assistant=assistant, timestamp=time.time()))

    def to_messages(self) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        for turn in self._turns:
            messages.append({"role": "user", "content": turn.user})
            messages.append({"role": "assistant", "content": turn.assistant})
        return messages

    def clear(self) -> None:
        self._turns.clear()


class MemoryStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)

    def _initialize(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS facts (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS notes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            conn.commit()

    def set_fact(self, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO facts (key, value, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(key) DO UPDATE SET
                    value=excluded.value,
                    updated_at=excluded.updated_at
                """,
                (key.strip().lower(), value.strip(), time.time()),
            )
            conn.commit()

    def get_fact(self, key: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute("SELECT value FROM facts WHERE key = ?", (key.strip().lower(),)).fetchone()
        return str(row[0]) if row else None

    def all_facts(self) -> list[tuple[str, str]]:
        with self._connect() as conn:
            rows = conn.execute("SELECT key, value FROM facts ORDER BY updated_at DESC LIMIT 20").fetchall()
        return [(str(k), str(v)) for k, v in rows]

    def save_note(self, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO notes (content, created_at) VALUES (?, ?)",
                (content.strip(), time.time()),
            )
            conn.commit()

    def read_notes(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, content, created_at FROM notes ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [{"id": int(row[0]), "content": str(row[1]), "created_at": float(row[2])} for row in rows]

    def build_memory_context(self) -> str:
        facts = self.all_facts()
        if not facts:
            return "No saved user facts."
        lines = ["Saved user facts:"]
        for key, value in facts:
            lines.append(f"- {key}: {value}")
        return "\n".join(lines)
