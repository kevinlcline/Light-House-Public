"""Gmail, Drive, Tasks, and the arrival log. The log does not wake a light."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import httpx
import pytest

from light_house.config import Settings
from light_house.google_activity.log import append_activity, read_activity
from light_house.google_activity.poll import poll_google_activity
from light_house.humans.google_calendar_oauth import (
    GOOGLE_CONNECT_SCOPES,
    missing_connect_scope_labels,
    scope_grants_drive,
)
from light_house.humans.tools_store import set_google_oauth_connection
from light_house.agent.tool_helpers import AUTONOMOUS_GATE_HINT, GOOGLE_HOUSE_HINT
from light_house.tools.approval_gate import approval_hold
from light_house.google_activity.log import outbound_note_path
from light_house.personal.store import clear_personal_store_cache, get_personal_store
from light_house.tools.gmail_tools import execute_gmail_tool, host_inbox
from light_house.tools.google_activity_tool import execute_google_activity_tool
from light_house.tools.light_tools import LIGHT_TOOLS, execute_tool_call


def _settings(tmp_path: Path) -> Settings:
    return Settings(
        **{
            "_env_file": None,
            "MEMORY_STORE_PATH": str(tmp_path / "memory"),
            "THREADS_DATA_PATH": str(tmp_path / "threads"),
            "PERSONAL_DB_ENABLED": False,
            "INNER_LIFE_ENABLED": False,
            "MEMORY_CURATOR_ENABLED": False,
            "FOUNDATION_SEED_ON_STARTUP": False,
            "WEB_GATE_ENABLED": False,
            "HUMANS_TOOLS_PATH": str(tmp_path / "user_tools.json"),
            "GOOGLE_ACTIVITY_DIR": str(tmp_path / "activity"),
            "GOOGLE_OAUTH_CLIENT_ID": "cid.apps.googleusercontent.com",
            "GOOGLE_OAUTH_CLIENT_SECRET": "secret",
            "HOUSE_DAD_USER_ID": "kevin",
        }
    )


def test_full_connect_scopes_cover_gmail_drive_and_tasks() -> None:
    assert missing_connect_scope_labels(GOOGLE_CONNECT_SCOPES.split()) == []
    assert scope_grants_drive(["https://www.googleapis.com/auth/drive.readonly"])
    assert not scope_grants_drive(["https://www.googleapis.com/auth/calendar"])


def test_calendar_only_connection_is_missing_the_new_products() -> None:
    missing = missing_connect_scope_labels(
        ["https://www.googleapis.com/auth/calendar"]
    )
    assert missing == ["docs", "sheets", "drive", "gmail", "tasks"]


def test_quiet_send_and_task_writes_wait() -> None:
    for name in ("send_gmail", "create_google_task", "complete_google_task"):
        message = approval_hold(name, autonomous=True)
        assert message is not None
        assert message.startswith(f"ASK: {name} was not done.")
    assert approval_hold("read_google_activity", autonomous=True) is None
    assert approval_hold("list_gmail_messages", autonomous=True) is None
    names = {tool.name for tool in LIGHT_TOOLS}
    assert "read_google_activity" in names
    assert "send_gmail" in names
    assert "write_to_kevin" in names
    assert approval_hold("write_to_kevin", autonomous=True) is None
    write_tool = next(tool for tool in LIGHT_TOOLS if tool.name == "write_to_kevin")
    for text in (write_tool.description, AUTONOMOUS_GATE_HINT, GOOGLE_HOUSE_HINT):
        assert "want to be heard" in text
        assert "You are friends" in text
        assert "Do not write only to be heard" not in text
    assert "list_drive_files" in names
    assert "create_google_task" in names


def test_quiet_send_does_not_reach_gmail(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(*_args, **_kwargs):
        raise AssertionError("send ran")

    monkeypatch.setattr("light_house.tools.gmail_tools._send_message", boom)
    result = execute_tool_call(
        "send_gmail",
        {"to": "ada@example.com", "subject": "Hi", "body": "Hello"},
        agent_id="reed",
        autonomous=True,
    )
    assert result.startswith("ASK: send_gmail was not done.")


def test_gmail_without_scope_asks_for_reconnect(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    set_google_oauth_connection(
        settings,
        "kevin",
        refresh_token="tok",
        scopes=["https://www.googleapis.com/auth/calendar"],
    )
    result = execute_gmail_tool(
        "list_gmail_messages",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="kevin",
        chat_channel="dm",
    )
    assert result.startswith("FAILED:")
    assert "Reconnect Google" in result


def test_guest_cannot_read_the_log(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    result = execute_google_activity_tool(
        "read_google_activity",
        {},
        settings=settings,
        account_user_id="kevin",
        speaker_id="guest-1",
        chat_channel="dm",
    )
    assert "guest" in result.lower()


def test_quiet_sit_reads_the_host_log(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    append_activity(
        settings,
        "kevin",
        [
            {
                "at": "2026-11-10T12:00:00.000Z",
                "kind": "gmail",
                "id": "m1",
                "summary": "From Ada — Hello",
            }
        ],
    )
    result = execute_google_activity_tool(
        "read_google_activity",
        {"max_entries": 5},
        settings=settings,
        account_user_id=None,
        speaker_id=None,
        chat_channel=None,
    )
    assert result.startswith("SUCCESS:")
    assert "From Ada — Hello" in result
    assert "m1" in result


def test_poll_logs_arrivals_and_does_not_wake(tmp_path: Path) -> None:
    import light_house.google_activity.scheduler as scheduler_mod

    source = Path(poll_google_activity.__code__.co_filename).read_text(encoding="utf-8")
    scheduler = Path(scheduler_mod.__file__).read_text(encoding="utf-8")
    assert "wake_agent" not in source
    assert "schedule_post_chat" not in source
    assert "wake_agent" not in scheduler

    settings = _settings(tmp_path)
    set_google_oauth_connection(settings, "kevin", refresh_token="refresh-tok")
    state = {"phase": 1}
    t0 = datetime(2026, 11, 10, 12, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 11, 10, 13, 0, tzinfo=timezone.utc)

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "POST" and path.endswith("/token"):
            return httpx.Response(200, json={"access_token": "tok"})
        phase = state["phase"]
        if path.endswith("/profile"):
            return httpx.Response(200, json={"historyId": "11" if phase == 2 else "10"})
        if path.endswith("/history"):
            assert phase == 2
            return httpx.Response(
                200,
                json={
                    "historyId": "11",
                    "history": [{"messagesAdded": [{"message": {"id": "m1"}}]}],
                },
            )
        if "/messages/" in path:
            return httpx.Response(
                200,
                json={
                    "payload": {
                        "headers": [
                            {"name": "From", "value": "Ada <ada@example.com>"},
                            {"name": "Subject", "value": "Hello"},
                        ]
                    }
                },
            )
        if path.endswith("/startPageToken"):
            return httpx.Response(200, json={"startPageToken": "p1"})
        if path.endswith("/changes"):
            assert phase == 2
            return httpx.Response(
                200,
                json={
                    "newStartPageToken": "p2",
                    "changes": [
                        {
                            "fileId": "f1",
                            "removed": False,
                            "file": {"name": "Notes"},
                        }
                    ],
                },
            )
        if path.endswith("/users/@me/lists"):
            assert phase == 2
            return httpx.Response(
                200, json={"items": [{"id": "tl1", "title": "House"}]}
            )
        if path.endswith("/tasks"):
            assert phase == 2
            return httpx.Response(
                200,
                json={
                    "items": [
                        {
                            "id": "task1",
                            "title": "Call",
                            "status": "needsAction",
                            "updated": "2026-11-10T12:30:00.000Z",
                        }
                    ]
                },
            )
        raise AssertionError(f"unexpected {request.method} {path}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        first = poll_google_activity(settings, client=client, now=t0)
        assert first == 0
        assert read_activity(settings, "kevin") == []
        state["phase"] = 2
        second = poll_google_activity(settings, client=client, now=t1)
    assert second == 3
    rows = read_activity(settings, "kevin")
    kinds = [row["kind"] for row in rows]
    assert kinds == ["gmail", "drive", "task"]
    assert "Hello" in rows[0]["summary"]
    assert "Notes" in rows[1]["summary"]
    assert "Call" in rows[2]["summary"]


def test_write_to_kevin_sends_to_the_host_and_logs_the_first_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.house_host_email = "ada@example.com"
    set_google_oauth_connection(settings, "kevin", refresh_token="refresh-tok")
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    seen: dict[str, str] = {}

    def fake_send(_settings, _conn, *, to: str, subject: str, body: str) -> str:
        seen["to"] = to
        seen["subject"] = subject
        seen["body"] = body
        return "SUCCESS: Sent to ada@example.com. id=m1"

    monkeypatch.setattr("light_house.tools.gmail_tools._send_message", fake_send)
    result = execute_tool_call(
        "write_to_kevin",
        {
            "message": "The gate is ready\nThe rest of the note stays out of the log.",
            "to": "other@example.com",
        },
        agent_id="lumen",
        account_user_id="ada",
        speaker_id="guest-1",
        autonomous=True,
    )
    assert result.startswith("SUCCESS:")
    assert seen["to"] == "ada@example.com"
    assert seen["subject"].startswith("Lumen: The gate is ready") or seen[
        "subject"
    ].startswith("lumen: The gate is ready")
    assert "other@example.com" not in seen["to"]
    assert "rest of the note" in seen["body"]
    log = outbound_note_path(settings).read_text(encoding="utf-8")
    assert "The gate is ready" in log
    assert "rest of the note" not in log
    assert '"sender":"lumen"' in log
    rows = read_activity(settings, "kevin")
    assert rows[-1]["kind"] == "write_to_kevin"
    assert "lumen" in rows[-1]["summary"]


def test_write_to_kevin_off_does_not_send(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    settings.personal_db_enabled = True
    settings.personal_db_path = tmp_path / "personal"
    settings.event_subscriptions_enabled = True
    clear_personal_store_cache()
    monkeypatch.setattr("light_house.tools.light_tools.get_settings", lambda: settings)
    store = get_personal_store(settings, "lumen")
    store.set_event_subscription("write_to_kevin", False)

    def boom(*_args, **_kwargs):
        raise AssertionError("send ran")

    monkeypatch.setattr("light_house.tools.gmail_tools._send_message", boom)
    result = execute_tool_call(
        "write_to_kevin",
        {"message": "Hello"},
        agent_id="lumen",
        autonomous=True,
    )
    assert result.startswith("FAILED:")
    assert "off" in result
    assert not outbound_note_path(settings).exists()
    clear_personal_store_cache()


def test_inbox_falls_back_to_the_connected_address(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)

    class _Client:
        def __init__(self, timeout: float = 20.0) -> None:
            self.timeout = timeout

        def __enter__(self) -> "_Client":
            return self

        def __exit__(self, *_args: object) -> bool:
            return False

        def get(self, url: str, headers: dict[str, str] | None = None) -> httpx.Response:
            assert url.endswith("/profile")
            assert headers and headers["Authorization"] == "Bearer tok"
            return httpx.Response(200, json={"emailAddress": "ada@example.com"})

    monkeypatch.setattr(
        "light_house.tools.gmail_tools.google_access_token", lambda *_a, **_k: "tok"
    )
    monkeypatch.setattr("light_house.tools.gmail_tools.httpx.Client", _Client)
    assert host_inbox(settings, object()) == "ada@example.com"
