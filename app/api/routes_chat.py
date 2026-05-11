from __future__ import annotations

import json
from queue import Queue
import threading

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from app.models.schemas import ChatRequest, ChatResponse
from app.services.ai_service import AIService


router = APIRouter(tags=["chat"])


def _service(request: Request) -> AIService:
    return request.app.state.ai_service


@router.post("/chat", response_model=ChatResponse)
def chat(payload: ChatRequest, request: Request) -> ChatResponse:
    return _service(request).process_text(
        text=payload.text,
        session_id=payload.session_id,
        use_web_search=payload.use_web_search,
        force_location_refresh=payload.force_location_refresh,
        require_gps=payload.require_gps,
    )


@router.post("/chat/stream")
def chat_stream(payload: ChatRequest, request: Request) -> StreamingResponse:
    service = _service(request)
    events: Queue[dict[str, object] | None] = Queue()

    def publish(event: dict[str, object]) -> None:
        events.put(event)

    def on_chunk(chunk: str) -> None:
        if chunk:
            publish({"type": "chunk", "content": chunk})

    def worker() -> None:
        try:
            response = service.process_text(
                text=payload.text,
                session_id=payload.session_id,
                use_web_search=payload.use_web_search,
                force_location_refresh=payload.force_location_refresh,
                require_gps=payload.require_gps,
                on_chunk=on_chunk,
            )
            response_data = response.model_dump() if hasattr(response, "model_dump") else response.dict()
            publish({"type": "done", "response": response_data})
        except Exception as exc:
            publish({"type": "error", "detail": str(exc)})
        finally:
            events.put(None)

    def event_stream():
        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        while True:
            event = events.get()
            if event is None:
                break
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(event_stream(), media_type="application/x-ndjson")
