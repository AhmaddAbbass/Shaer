import logging
import os


def configure_logging() -> None:
    """Configure root logging once."""
    if logging.getLogger().handlers:
        return

    log_level = os.getenv("SHAER_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, log_level, logging.INFO)

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
