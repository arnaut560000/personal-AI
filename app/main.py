from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.api.routes_chat import router as chat_router
from app.api.routes_commands import router as commands_router
from app.api.routes_location import router as location_router
from app.api.routes_memory import router as memory_router
from app.api.routes_settings import router as settings_router
from app.api.routes_status import router as status_router
from app.api.routes_voice import router as voice_router
from app.config import get_config, load_env_file
from app.services.ai_service import AIService
from app.utils import setup_logging


@asynccontextmanager
async def lifespan(application: FastAPI):
    load_env_file()
    config = get_config()
    setup_logging(config.app_log_path)
    application.state.config = config
    application.state.ai_service = AIService(config)
    yield


def create_app() -> FastAPI:
    app = FastAPI(
        title="RoomAI Backend",
        version="2.0.0",
        description="Backend-first assistant architecture for future mobile clients.",
        lifespan=lifespan,
    )
    app.include_router(status_router)
    app.include_router(chat_router)
    app.include_router(location_router)
    app.include_router(memory_router)
    app.include_router(commands_router)
    app.include_router(settings_router)
    app.include_router(voice_router)

    @app.get("/")
    def root() -> FileResponse:
        return FileResponse(app.state.config.static_dir / "index.html")

    return app


app = create_app()
