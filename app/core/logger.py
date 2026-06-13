import sys
import os
from loguru import logger


def setup_logging():
    logger.remove()
    logger.add(
        sys.stdout,
        level="INFO",
        colorize=True,
        format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
    )
    os.makedirs("logs", exist_ok=True)
    logger.add(
        "logs/weld-vision.log",
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        enqueue=True,
    )
    return logger
