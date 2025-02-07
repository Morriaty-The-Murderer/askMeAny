import os
import sys
from pathlib import Path

from loguru import logger

# Remove default logger
logger.remove()

# Get project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Create logs directory if it doesn't exist
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)

# Configure logging format
LOG_FORMAT = ("<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
              "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>")

# Configure default logger
logger.add(
    sys.stderr,
    format=LOG_FORMAT,
    level="INFO",
    colorize=True,
)

# Add file logging with rotation
logger.add(
    LOG_DIR / "app_{time}.log",
    format=LOG_FORMAT,
    level="DEBUG",
    rotation="1 day",
    retention="7 days",
    compression="zip",
    enqueue=True,
)

# Error logging to separate file
logger.add(
    LOG_DIR / "error_{time}.log",
    format=LOG_FORMAT,
    level="ERROR",
    rotation="1 day",
    retention="30 days",
    compression="zip",
    enqueue=True,
)

# Environment-specific logging
if os.getenv("ENVIRONMENT") == "development":
    # More verbose logging for development
    logger.add(
        LOG_DIR / "debug_{time}.log",
        format=LOG_FORMAT,
        level="DEBUG",
        filter=lambda record: record["level"].name == "DEBUG",
        rotation="1 day",
        retention="3 days",
        compression="zip",
        enqueue=True,
    )


def get_logger(name: str = None):
    """Get a logger instance with the specified name.
    
    Args:
        name: Optional name for the logger. Defaults to calling module name.
    
    Returns:
        Logger instance configured with project settings
    """
    return logger.bind(name=name or "main")


# Export configured logger
__all__ = ["logger", "get_logger"]
