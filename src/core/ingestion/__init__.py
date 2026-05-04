import logging

from src.config.config import LOG_DATE_FORMAT, LOG_FORMAT, LOG_LEVEL


def setup_logging() -> None:
    """Configure root logging once for ingestion scripts and workers."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    level = getattr(logging, LOG_LEVEL, logging.INFO)
    logging.basicConfig(
        level=level,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
    )

    # Keep third-party HTTP transport logs quiet unless explicit DEBUG mode.
    external_level = logging.DEBUG if level <= logging.DEBUG else logging.WARNING
    logging.getLogger("httpx").setLevel(external_level)
    logging.getLogger("httpcore").setLevel(external_level)
