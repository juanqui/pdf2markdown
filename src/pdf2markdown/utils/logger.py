"""Logging configuration for PDF to Markdown converter."""

import logging
import logging.handlers
from pathlib import Path


def setup_logging(
    level: str = "INFO", log_file: Path | None = None, format_string: str | None = None
) -> None:
    """Configure application logging.

    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
        format_string: Optional format string for log messages
    """
    if format_string is None:
        format_string = "[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s"

    handlers = [logging.StreamHandler()]

    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(
            logging.handlers.RotatingFileHandler(
                log_file, maxBytes=10_485_760, backupCount=5  # 10MB
            )
        )

    logging.basicConfig(
        level=getattr(logging, level.upper()), format=format_string, handlers=handlers
    )

    # Set specific loggers
    logging.getLogger("pdf2markdown").setLevel(getattr(logging, level.upper()))

    # Reduce noise from other libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
