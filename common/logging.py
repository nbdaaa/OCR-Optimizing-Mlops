import logging
import json
import sys
from datetime import datetime, timezone

class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return json.dumps({
            "ts":      datetime.now(timezone.utc).isoformat(),
            "level":   record.levelname,
            "module":  record.name,
            "message": record.getMessage(),
            **({"exc": self.formatException(record.exc_info)}
               if record.exc_info else {}),
        })

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(_JsonFormatter())
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
        logger.propagate = False
    return logger
