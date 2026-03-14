"""Logging setup for SPD-RAG: console (INFO), file (DEBUG), and error file (ERROR) via loguru."""

import sys
import logging
from loguru import logger
from backend.shared.constants import LOGS_DIR, BACKEND_LOG_PATH_STR, BACKEND_ERROR_LOG_PATH_STR

LOGS_DIR.mkdir(parents=True, exist_ok=True)
logger.remove()

logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

logger.add(
    BACKEND_LOG_PATH_STR,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="5 days",
    encoding="utf-8"
)

logger.add(
    BACKEND_ERROR_LOG_PATH_STR,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="10 MB",
    retention="5 days",
    encoding="utf-8"
)

def get_logger(context_tag=None):
    """Return a loguru logger, optionally bound to a context tag (e.g. module name)."""
    if context_tag:
        return logger.bind(name=context_tag)
    return logger


logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
