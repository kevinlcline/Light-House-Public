"""Dev daily log handler and tail API."""

from __future__ import annotations

import logging
import tempfile
from datetime import date
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.dev_log import DevLogFileNoiseFilter, MidnightResetFileHandler, read_dev_log_tail
from light_house.main import _settings_dep, app


def _test_settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "NOTES_PATH": str(tmp_path / "notes"),
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
    }
    base.update(overrides)
    return Settings(**base)


def _emit(handler: MidnightResetFileHandler, message: str) -> None:
    record = logging.LogRecord(
        name="test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )
    handler.emit(record)


def test_midnight_reset_truncates_on_date_change(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    log_path = tmp_path / "dev.log"
    log_path.write_text("stale line\n", encoding="utf-8")

    class FakeDate:
        _today = date(2026, 5, 29)

        @classmethod
        def today(cls) -> date:
            return cls._today

    monkeypatch.setattr("light_house.dev_log.date", FakeDate)

    handler = MidnightResetFileHandler(log_path)
    handler.setFormatter(logging.Formatter("%(message)s"))
    _emit(handler, "day one")
    assert "stale line" in log_path.read_text(encoding="utf-8")
    assert "day one" in log_path.read_text(encoding="utf-8")

    FakeDate._today = date(2026, 5, 30)
    _emit(handler, "day two")
    content = log_path.read_text(encoding="utf-8")
    assert "stale line" not in content
    assert "day one" not in content
    assert "day two" in content
    handler.close()


def test_dev_log_file_noise_filter_skips_poll_access_lines() -> None:
    filt = DevLogFileNoiseFilter()
    poll = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s" %s',
        args=("127.0.0.1:0", "GET /v1/dev/log?tail=500 HTTP/1.1", "200"),
        exc_info=None,
    )
    chat = logging.LogRecord(
        name="uvicorn.access",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg='%s - "%s" %s',
        args=("127.0.0.1:0", "GET /v1/chat/history?agent_id=lumen HTTP/1.1", "200"),
        exc_info=None,
    )
    assert filt.filter(poll) is False
    assert filt.filter(chat) is True


def test_read_dev_log_tail_truncates() -> None:
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False) as f:
        f.write("\n".join(f"line{i}" for i in range(5)))
        path = Path(f.name)
    try:
        content, count, truncated = read_dev_log_tail(path, max_lines=2)
        assert count == 2
        assert truncated is True
        assert "line3" in content
        assert "line4" in content
        assert "line0" not in content
    finally:
        path.unlink(missing_ok=True)


def test_dev_log_api_returns_404_when_disabled(tmp_path: Path) -> None:
    settings = _test_settings(tmp_path, DEV_LOG_ENABLED=False, LIGHT_HOUSE_ENV="production")
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            response = client.get("/v1/dev/log")
        assert response.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_dev_log_api_returns_tail_when_enabled(tmp_path: Path) -> None:
    log_path = tmp_path / "dev.log"
    log_path.write_text("alpha\nbeta\ngamma\n", encoding="utf-8")
    settings = _test_settings(
        tmp_path,
        DEV_LOG_ENABLED=True,
        DEV_LOG_PATH=str(log_path),
        DEV_LOG_MAX_TAIL_LINES=2000,
    )
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            response = client.get("/v1/dev/log?tail=2")
        assert response.status_code == 200
        data = response.json()
        assert data["lines"] == 2
        assert data["truncated"] is True
        assert "beta" in data["content"]
        assert "gamma" in data["content"]
        assert "alpha" not in data["content"]
        assert response.headers.get("cache-control") == "no-store"
    finally:
        app.dependency_overrides.clear()
