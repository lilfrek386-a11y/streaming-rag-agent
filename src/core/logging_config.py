import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

LOG_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"
MAX_BYTES = 1_000_000
BACKUP_COUNT = 3


def setup_logging(log_dir: str, level: str = "INFO") -> None:
    directory = Path(log_dir)
    directory.mkdir(parents=True, exist_ok=True)

    handler = RotatingFileHandler(
        directory / "app.log",
        maxBytes=MAX_BYTES,
        backupCount=BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter(LOG_FORMAT))

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))

    for existing in list(root.handlers):
        if isinstance(existing, RotatingFileHandler):
            root.removeHandler(existing)
            existing.close()

    root.addHandler(handler)

    for noisy in ("httpx", "httpcore", "urllib3", "sentence_transformers"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
