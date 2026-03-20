"""Entry point: uv run python -m hue_visualizer"""

import logging

from dotenv import load_dotenv
import uvicorn

from hue_visualizer.core.config import Settings


def main():
    load_dotenv()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    settings = Settings()

    from hue_visualizer.server.app import app

    uvicorn.run(app, host=settings.server_host, port=settings.server_port, log_level="info")


if __name__ == "__main__":
    main()
