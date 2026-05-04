from __future__ import annotations

import uvicorn

from app.config import get_config, load_env_file


def main() -> None:
    load_env_file()
    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.api_host,
        port=config.api_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
