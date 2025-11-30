import logging
from typing import Optional


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    # Allow legacy positional usage where the first argument was a level.
    if isinstance(name, int):
        level, name = name, None

    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger
