"""Dev-only daily log file (resets at server-local midnight) and tail reader."""

from __future__ import annotations

import logging
from datetime import date
from pathlib import Path

from light_house.config import Settings

_LOG_FORMAT = "%(asctime)s %(levelname)s %(name)s %(message)s"
_HANDLER_MARKER = "_light_house_dev_log_handler"
# Access paths omitted from the dev log file (avoids poll feedback in dev-log.html).
_DEV_LOG_FILE_SKIP_ACCESS = (
    "GET /v1/dev/log",
    "GET /dev-log.html",
)


class DevLogFileNoiseFilter(logging.Filter):
    """Drop noisy access lines from the dev log file only (stderr unchanged)."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "uvicorn.access":
            return True
        msg = record.getMessage()
        return not any(skip in msg for skip in _DEV_LOG_FILE_SKIP_ACCESS)


class MidnightResetFileHandler(logging.Handler):
    """Append to a log file; truncate at server-local midnight on the next write."""

    def __init__(self, path: Path) -> None:
        super().__init__()
        self._path = path.resolve()
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._current_date = date.today()
        self._stream = self._open_stream(truncate=False)

    def _open_stream(self, *, truncate: bool):
        mode = "w" if truncate else "a"
        return self._path.open(mode, encoding="utf-8")

    def _rotate_if_new_day(self) -> None:
        today = date.today()
        if today == self._current_date:
            return
        self._current_date = today
        if self._stream is not None:
            self._stream.close()
        self._stream = self._open_stream(truncate=True)

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._rotate_if_new_day()
            msg = self.format(record)
            assert self._stream is not None
            self._stream.write(msg + "\n")
            self._stream.flush()
        except Exception:
            self.handleError(record)

    def close(self) -> None:
        if self._stream is not None:
            self._stream.close()
            self._stream = None
        super().close()


def _logger_has_dev_handler(log: logging.Logger) -> bool:
    return any(getattr(h, _HANDLER_MARKER, False) for h in log.handlers)


def setup_dev_file_logging(settings: Settings) -> Path | None:
    """Attach midnight-reset file handler to app + uvicorn loggers."""
    if not settings.dev_log_enabled:
        return None

    path = settings.dev_log_path.resolve()
    level = getattr(logging, settings.log_level.upper(), logging.INFO)
    formatter = logging.Formatter(_LOG_FORMAT)

    handler = MidnightResetFileHandler(path)
    setattr(handler, _HANDLER_MARKER, True)
    handler.setFormatter(formatter)
    handler.setLevel(level)
    handler.addFilter(DevLogFileNoiseFilter())

    root = logging.getLogger()
    if not _logger_has_dev_handler(root):
        root.addHandler(handler)

    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        logging.getLogger(name).setLevel(level)

    logging.getLogger(__name__).info(
        "Dev log file: %s (resets at local midnight)",
        path,
    )
    return path


def read_dev_log_tail(path: Path, *, max_lines: int) -> tuple[str, int, bool]:
    """Return (content, line_count, truncated)."""
    if not path.is_file():
        return "", 0, False
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    total = len(lines)
    truncated = total > max_lines
    if truncated:
        lines = lines[-max_lines:]
    return "\n".join(lines), len(lines), truncated
