from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import re
import subprocess
from typing import Callable
import webbrowser

from app.config import AppConfig
from app.services.memory_service import MemoryService


@dataclass
class CommandResult:
    handled: bool
    response: str
    should_exit: bool = False
    metadata: dict[str, str] | None = None


class CommandService:
    def __init__(self, config: AppConfig, memory_service: MemoryService):
        self.config = config
        self.memory_service = memory_service
        self.pending_confirmation: dict[str, str] | None = None
        self.app_allow_list: dict[str, list[str]] = {
            "notepad": ["notepad.exe"],
            "calculator": ["calc.exe"],
            "paint": ["mspaint.exe"],
            "explorer": ["explorer.exe"],
        }
        self.dangerous_apps = {"cmd", "powershell", "regedit"}

    def process(self, user_text: str, status_callback: Callable[[str], None] | None = None) -> CommandResult:
        text = user_text.strip()
        lowered = text.lower()
        if not text:
            return CommandResult(False, "")

        if lowered in {"exit", "quit"}:
            self._log_action(text, "allowed", "exit")
            return CommandResult(True, "Goodbye!", should_exit=True, metadata={"type": "exit"})

        if self._looks_like_unsafe_shell(lowered):
            self._log_action(text, "blocked", "unsafe_shell")
            return CommandResult(
                True,
                "I blocked that because arbitrary shell execution is disabled for safety.",
                metadata={"type": "blocked"},
            )

        if lowered == "confirm":
            if not self.pending_confirmation:
                return CommandResult(True, "There is no pending action to confirm.")
            action = dict(self.pending_confirmation)
            self.pending_confirmation = None
            return self._execute_confirmed_action(action, status_callback=status_callback)

        if lowered in {"cancel", "deny"}:
            if self.pending_confirmation:
                self._log_action("pending_action", "cancelled", self.pending_confirmation.get("type", "unknown"))
                self.pending_confirmation = None
                return CommandResult(True, "Cancelled the pending action.")
            return CommandResult(True, "There is no pending action to cancel.")

        if self._is_time_or_date_request(lowered):
            now = datetime.now()
            response = f"It is {now.strftime('%I:%M %p')} on {now.strftime('%A, %B %d, %Y')}."
            self._log_action(text, "allowed", "datetime")
            return CommandResult(True, response, metadata={"type": "datetime"})

        note_match = re.match(r"^(save note|note)\s*[:\-]\s*(.+)$", text, flags=re.IGNORECASE)
        if note_match:
            note_content = note_match.group(2).strip()
            if not note_content:
                return CommandResult(True, "Please provide text after save note.")
            self.memory_service.save_note(note_content)
            self._log_action(text, "allowed", "save_note")
            return CommandResult(True, "Saved your note.", metadata={"type": "save_note"})

        if lowered in {"read notes", "show notes", "list notes"}:
            notes = self.memory_service.read_notes(limit=10)
            if not notes:
                return CommandResult(True, "You do not have any saved notes yet.", metadata={"type": "read_notes"})
            lines = ["Here are your recent notes:"]
            for note in notes:
                lines.append(f"- [{note['id']}] {note['content']}")
            self._log_action(text, "allowed", "read_notes")
            return CommandResult(True, "\n".join(lines), metadata={"type": "read_notes"})

        remember_match = re.match(r"^remember\s+that\s+(.+?)\s+is\s+(.+)$", text, flags=re.IGNORECASE)
        if remember_match:
            key = remember_match.group(1).strip()
            value = remember_match.group(2).strip()
            self.memory_service.set_fact(key, value)
            self._log_action(text, "allowed", "save_fact")
            return CommandResult(
                True,
                f"Okay, I will remember that {key} is {value}.",
                metadata={"type": "save_fact"},
            )

        name_set_match = re.match(r"^(my name is)\s+(.+)$", text, flags=re.IGNORECASE)
        if name_set_match:
            name = name_set_match.group(2).strip().strip(".!?")
            if not name:
                return CommandResult(True, "Please tell me your name after 'my name is'.")
            self.memory_service.set_fact("name", name)
            self._log_action(text, "allowed", "save_name")
            return CommandResult(
                True,
                f"Nice to meet you, {name}. I will remember your name.",
                metadata={"type": "save_name"},
            )

        if re.search(r"\b(what is my name|what's my name|do you know my name)\b", lowered):
            value = self.memory_service.get_fact("name")
            self._log_action(text, "allowed", "recall_name")
            if value is None:
                return CommandResult(True, "You have not told me your name yet.", metadata={"type": "recall_name"})
            return CommandResult(True, f"Your name is {value}.", metadata={"type": "recall_name"})

        recall_match = re.match(r"^(what do you remember about|recall)\s+(.+)$", text, flags=re.IGNORECASE)
        if recall_match:
            key = recall_match.group(2).strip()
            value = self.memory_service.get_fact(key)
            self._log_action(text, "allowed", "recall_fact")
            if value is None:
                return CommandResult(
                    True,
                    f"I do not have anything saved for {key} yet.",
                    metadata={"type": "recall_fact"},
                )
            return CommandResult(True, f"You told me that {key} is {value}.", metadata={"type": "recall_fact"})

        website_match = re.match(r"^open\s+(website|site|url)?\s*(.+)$", text, flags=re.IGNORECASE)
        if website_match:
            raw_url = website_match.group(2).strip()
            if "." not in raw_url and not raw_url.startswith("http"):
                return CommandResult(True, "Please provide a valid website address, like open website example.com.")
            url = raw_url if raw_url.startswith(("http://", "https://")) else f"https://{raw_url}"
            if not self.config.allow_local_actions:
                self._log_action(text, "blocked", "open_website_disabled")
                return CommandResult(
                    True,
                    f"Website launch is disabled in backend mode. Target URL: {url}",
                    metadata={"type": "open_website_disabled"},
                )
            webbrowser.open(url)
            self._log_action(text, "allowed", "open_website")
            return CommandResult(True, f"Opening {url}", metadata={"type": "open_website"})

        app_match = re.match(r"^open app\s+(.+)$", text, flags=re.IGNORECASE) or re.match(
            r"^open\s+(.+)$", text, flags=re.IGNORECASE
        )
        if app_match:
            app_name = app_match.group(1).strip().lower()
            if app_name in {"website", "site", "url"}:
                return CommandResult(False, "")
            if not self.config.allow_local_actions:
                self._log_action(text, "blocked", "open_app_disabled")
                return CommandResult(
                    True,
                    "Opening local desktop apps is disabled in backend mode.",
                    metadata={"type": "open_app_disabled"},
                )
            if app_name in self.dangerous_apps:
                self.pending_confirmation = {"type": "open_app", "name": app_name}
                self._log_action(text, "pending_confirmation", "open_app_danger")
                return CommandResult(
                    True,
                    f'Opening "{app_name}" may be unsafe. Type "confirm" to continue or "cancel" to stop.',
                    metadata={"type": "pending_confirmation"},
                )
            return self._open_safe_app(text, app_name)

        return CommandResult(False, "")

    def _open_safe_app(self, raw_command: str, app_name: str) -> CommandResult:
        program = self.app_allow_list.get(app_name)
        if not program:
            self._log_action(raw_command, "blocked", "app_not_allowlisted")
            return CommandResult(
                True,
                (
                    f'I can only open allow-listed apps for safety. Supported: {", ".join(sorted(self.app_allow_list))}. '
                    'Use "open app <name>".'
                ),
                metadata={"type": "app_not_allowlisted"},
            )
        subprocess.Popen(program, shell=False)
        self._log_action(raw_command, "allowed", "open_app")
        return CommandResult(True, f'Opening "{app_name}".', metadata={"type": "open_app"})

    def _execute_confirmed_action(
        self,
        action: dict[str, str],
        status_callback: Callable[[str], None] | None = None,
    ) -> CommandResult:
        action_type = action.get("type", "")
        if action_type == "open_app":
            app_name = action.get("name", "")
            command = ["powershell.exe"] if app_name == "powershell" else [f"{app_name}.exe"]
            if status_callback:
                status_callback(f"Running confirmed action: {app_name}")
            subprocess.Popen(command, shell=False)
            self._log_action(app_name, "allowed_after_confirmation", "open_app_danger")
            return CommandResult(True, f'Confirmed. Opening "{app_name}".', metadata={"type": "open_app"})
        self._log_action("unknown", "blocked", "unknown_confirmed_action")
        return CommandResult(True, "I could not run that confirmed action.", metadata={"type": "blocked"})

    def _is_time_or_date_request(self, lowered: str) -> bool:
        triggers = (
            "what time",
            "current time",
            "time now",
            "what date",
            "today's date",
            "todays date",
            "current date",
        )
        return any(trigger in lowered for trigger in triggers)

    def _looks_like_unsafe_shell(self, lowered: str) -> bool:
        blocked_phrases = (
            "run cmd",
            "run powershell",
            "execute command",
            "execute shell",
            "run shell",
            "rm ",
            "del ",
            "rmdir ",
            "format c:",
        )
        return any(phrase in lowered for phrase in blocked_phrases)

    def _log_action(self, command: str, outcome: str, action: str) -> None:
        event = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "command": command,
            "action": action,
            "outcome": outcome,
        }
        with self.config.action_log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(event) + "\n")
