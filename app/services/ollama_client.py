from __future__ import annotations

import json
import logging
from typing import Any, Callable
from urllib import error, request

from app.config import AppConfig


logger = logging.getLogger(__name__)


class OllamaClient:
    def __init__(self, config: AppConfig):
        self.config = config

    def fetch_installed_models(self) -> list[str]:
        try:
            with request.urlopen(f"{self.config.ollama_url}/api/tags", timeout=5) as response:
                data = json.loads(response.read().decode("utf-8"))
        except error.URLError as exc:
            raise RuntimeError(
                "Could not connect to Ollama. Make sure Ollama is running on "
                f"{self.config.ollama_url}."
            ) from exc
        return [model["name"] for model in data.get("models", []) if "name" in model]

    def resolve_model(self) -> str:
        installed_models = self.fetch_installed_models()
        if not installed_models:
            raise RuntimeError(
                "Ollama is running, but no model is installed yet. "
                "Install one first, for example: ollama pull llama3"
            )
        if self.config.ollama_model in installed_models:
            return self.config.ollama_model
        raise RuntimeError(
            f'Ollama model "{self.config.ollama_model}" is not installed. '
            f"Installed models: {', '.join(installed_models)}"
        )

    def ask(
        self,
        system_prompt: str,
        user_text: str,
        history_messages: list[dict[str, str]] | None = None,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        messages = [{"role": "system", "content": system_prompt}]
        if history_messages:
            messages.extend(history_messages)
        messages.append({"role": "user", "content": user_text})
        return self.ask_with_messages(messages, on_chunk=on_chunk)

    def ask_with_messages(
        self,
        messages: list[dict[str, str]],
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        payload = json.dumps(
            {
                "model": self.config.ollama_model,
                "stream": self.config.ollama_stream,
                "messages": messages,
            }
        ).encode("utf-8")
        http_request = request.Request(
            f"{self.config.ollama_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with request.urlopen(http_request, timeout=self.config.ollama_timeout_seconds) as response:
                if self.config.ollama_stream:
                    return self._read_stream(response, on_chunk=on_chunk)
                data = json.loads(response.read().decode("utf-8"))
        except error.HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode("utf-8", errors="ignore").strip()
            except Exception:
                body = ""
            detail = f"HTTP {exc.code}" + (f" - {body}" if body else "")
            raise RuntimeError(
                "Ollama returned an HTTP error while answering your question. "
                f"{detail}"
            ) from exc
        except error.URLError as exc:
            raise RuntimeError(
                "Failed to reach Ollama while sending your question. "
                "Please check that Ollama is still running."
            ) from exc

        message = data.get("message", {})
        content = message.get("content", "").strip()
        if not content:
            raise RuntimeError("Ollama returned an empty response.")
        return content

    def check_health(self) -> tuple[bool, str]:
        try:
            models = self.fetch_installed_models()
        except RuntimeError as exc:
            return (False, str(exc))
        if self.config.ollama_model not in models:
            return (
                False,
                f'Model "{self.config.ollama_model}" is not installed. Installed: {", ".join(models)}',
            )
        return (True, f'Ollama OK with model "{self.config.ollama_model}"')

    def _read_stream(
        self,
        http_response: Any,
        on_chunk: Callable[[str], None] | None = None,
    ) -> str:
        chunks: list[str] = []
        for raw_line in http_response:
            if not raw_line:
                continue
            line = raw_line.decode("utf-8").strip()
            if not line:
                continue
            try:
                packet = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "error" in packet:
                err_text = str(packet.get("error") or "").strip()
                if err_text:
                    raise RuntimeError(err_text)
                continue
            message = packet.get("message", {})
            content = message.get("content", "")
            if content:
                chunks.append(content)
                if on_chunk is not None:
                    on_chunk(content)
        full_text = "".join(chunks).strip()
        if not full_text:
            logger.warning("Ollama stream returned no text.")
            raise RuntimeError("Ollama returned an empty response.")
        return full_text
