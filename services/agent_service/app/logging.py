from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


_CONFIGURED = False


def configure_logging(level: str = "INFO") -> None:
    """
    Configure logging to both stdout and a rolling log file under logs/agent_service.log.
    Safe to call multiple times.
    """
    global _CONFIGURED
    if _CONFIGURED:
        return

    log_dir = Path(__file__).resolve().parents[3] / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / "agent_service.log"

    logger = logging.getLogger()
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")

    # Console handler
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # File handler (append mode)
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    _CONFIGURED = True


def get_logger(name: Optional[str] = None) -> logging.Logger:
    return logging.getLogger(name or "agent_service")
