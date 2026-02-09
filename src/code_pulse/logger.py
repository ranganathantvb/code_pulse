import json
import logging
import os
import time
import urllib.request
from typing import Optional


class NewRelicLogHandler(logging.Handler):
    def __init__(self, api_key: str, endpoint: str) -> None:
        super().__init__()
        self.api_key = api_key
        self.endpoint = endpoint
        self.service_name = os.getenv("CODEPULSE_SERVICE_NAME", "codepulse-agent-fastapi")

    def emit(self, record: logging.LogRecord) -> None:
        try:
            payload = [
                {
                    "message": record.getMessage(),
                    "level": record.levelname,
                    "logger": record.name,
                    "timestamp": int(record.created * 1000),
                    "service": self.service_name,
                }
            ]
            data = json.dumps(payload).encode("utf-8")
            request = urllib.request.Request(
                self.endpoint,
                data=data,
                headers={
                    "Content-Type": "application/json",
                    "Api-Key": self.api_key,
                },
            )
            with urllib.request.urlopen(request, timeout=1.5):
                return
        except Exception:
            return


def setup_logging(name: Optional[str] = None, level: int = logging.INFO) -> logging.Logger:
    # Allow legacy positional usage where the first argument was a level.
    if isinstance(name, int):
        level, name = name, None

    root_logger = logging.getLogger()
    if not root_logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
        )
        handler.setFormatter(formatter)
        root_logger.addHandler(handler)
        if os.getenv("CODEPULSE_ENABLE_LOG_FORWARDING", "true").lower() in {"1", "true", "yes", "on"}:
            api_key = os.getenv("NEW_RELIC_LOG_API_KEY") or os.getenv("NEW_RELIC_USER_KEY")
            endpoint = os.getenv("NEW_RELIC_LOG_API_ENDPOINT", "https://log-api.newrelic.com/log/v1")
            if api_key:
                root_logger.addHandler(NewRelicLogHandler(api_key, endpoint))
    root_logger.setLevel(level)

    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = True
    return logger


# Shared default logger for modules that only need a simple logger instance.
logger = setup_logging("code_pulse")
