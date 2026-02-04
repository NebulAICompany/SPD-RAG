import sys
import logging
from loguru import logger
from backend.shared.constants import LOGS_DIR, BACKEND_LOG_PATH_STR, BACKEND_ERROR_LOG_PATH_STR

# Create logs directory if it doesn't exist
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Remove default logger
logger.remove()

# Add console handler with colors
logger.add(
    sys.stdout,
    format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
    level="INFO",
    colorize=True
)

# Add file handler for all logs
logger.add(
    BACKEND_LOG_PATH_STR,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="DEBUG",
    rotation="10 MB",
    retention="5 days",
    encoding="utf-8"
)

# Add separate error log file
logger.add(
    BACKEND_ERROR_LOG_PATH_STR,
    format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
    level="ERROR",
    rotation="10 MB",
    retention="5 days",
    encoding="utf-8"
)

def get_logger(context_tag=None):
    """Get logger instance with optional context tag"""
    if context_tag:
        return logger.bind(name=context_tag)
    return logger


# Quiet Azure SDK HTTP logging noise
logging.getLogger("azure").setLevel(logging.WARNING)
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
