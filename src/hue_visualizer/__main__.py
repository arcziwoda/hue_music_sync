"""Entry point: uv run python -m hue_visualizer"""

import logging

from dotenv import load_dotenv
import uvicorn

from hue_visualizer.core.config import Settings


def main():
    load_dotenv()

    # Configure logging: route our app logs through a named logger,
    # set root logger to WARNING to suppress hue_entertainment_pykit's
    # noisy root-level logging (logs every set_input call at INFO, ~300/s).
    root = logging.getLogger()
    root.setLevel(logging.WARNING)

    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter(
        "%(asctime)s [%(name)s] %(levelname)s: %(message)s"
    ))

    # Our app loggers at INFO
    for name in ("hue_visualizer", "uvicorn"):
        lg = logging.getLogger(name)
        lg.setLevel(logging.INFO)
        lg.addHandler(handler)

    settings = Settings()

    from hue_visualizer.server.app import app

    uvicorn.run(app, host=settings.server_host, port=settings.server_port, log_level="info")


if __name__ == "__main__":
    main()
