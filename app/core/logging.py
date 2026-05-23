import logging
import sys

from app.core.config import get_settings


APP_LOG_HANDLER_NAME = "meeting_app_console"


def configure_app_logging() -> None:
    settings = get_settings()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    logger = logging.getLogger("app")
    logger.setLevel(level)
    logger.propagate = False

    existing_handler = next(
        (
            handler
            for handler in logger.handlers
            if handler.get_name() == APP_LOG_HANDLER_NAME
        ),
        None,
    )
    if existing_handler is None:
        existing_handler = logging.StreamHandler(sys.stdout)
        existing_handler.set_name(APP_LOG_HANDLER_NAME)
        logger.addHandler(existing_handler)

    existing_handler.setLevel(level)
    existing_handler.setFormatter(
        logging.Formatter(
            "%(asctime)s %(levelname)s [%(name)s] %(message)s"
        )
    )
