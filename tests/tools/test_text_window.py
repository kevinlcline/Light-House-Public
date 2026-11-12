"""A long Google read is a window, not the end of the text."""

from __future__ import annotations

import base64

import httpx

from light_house.tools.docs_tools import _read_doc
from light_house.tools.drive_tools import _read_file
from light_house.tools.gmail_tools import _read_message
from light_house.tools.text_window import slice_text_window


class _Resp:
    def __init__(self, *, status_code: int = 200, payload: dict | None = None, text: str = ""):
        self.status_code = status_code
        self._payload = payload or {}
        self.text = text

    def json(self) -> dict:
        return self._payload


class _Client:
    def __init__(self, handler, timeout: float = 20.0):
        self._handler = handler

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def get(self, url: str, headers=None, params=None):
        return self._handler(url, params)


def _install_client(monkeypatch, module: str, handler) -> None:
    monkeypatch.setattr(f"{module}.httpx.Client", lambda timeout=20.0: _Client(handler, timeout))


def test_slice_reports_the_window_and_stops_past_the_end() -> None:
    start, end, chunk, total = slice_text_window("abcdef", offset=2, max_chars=3)
    assert (start, end, chunk, total) == (2, 5, "cde", 6)
    start, end, chunk, total = slice_text_window("abcdef", offset=9, max_chars=3)
    assert start >= total
    assert chunk == ""


def test_read_google_doc_pages_instead_of_stopping(monkeypatch) -> None:
    body = ("alpha " * 20) + "TAIL"
    document = {
        "title": "Long",
        "body": {
            "content": [
                {"paragraph": {"elements": [{"textRun": {"content": body}}]}}
            ]
        },
    }

    def handler(url, _params):
        assert "documents/" in url
        return _Resp(payload=document)

    monkeypatch.setattr("light_house.tools.docs_tools._google_access_token", lambda *_a, **_k: "tok")
    _install_client(monkeypatch, "light_house.tools.docs_tools", handler)

    first = _read_doc(None, object(), doc_id_or_url="abcdefghij1234", offset=0, max_chars=50)
    assert "TAIL" not in first
    assert "(truncated)" not in first
    assert "Continue with offset=50" in first
    assert f"chars 0-50 of {len(body)} (excerpt)" in first

    second = _read_doc(
        None,
        object(),
        doc_id_or_url="abcdefghij1234",
        offset=50,
        max_chars=500,
    )
    assert "TAIL" in second
    assert "Continue with offset" not in second
    assert "(excerpt)" in second

    past = _read_doc(
        None,
        object(),
        doc_id_or_url="abcdefghij1234",
        offset=len(body) + 10,
        max_chars=500,
    )
    assert "Nothing remains" in past
    assert "TAIL" not in past


def test_read_drive_file_pages(monkeypatch) -> None:
    text = "A" * 40 + "END"

    def handler(url, _params):
        if url.endswith("/export"):
            return _Resp(text=text)
        return _Resp(
            payload={
                "id": "fileid1234",
                "name": "Notes",
                "mimeType": "application/vnd.google-apps.document",
            }
        )

    monkeypatch.setattr("light_house.tools.drive_tools.google_access_token", lambda *_a, **_k: "tok")
    _install_client(monkeypatch, "light_house.tools.drive_tools", handler)

    first = _read_file(None, object(), file_id="fileid1234", offset=0, max_chars=20)
    assert "END" not in first
    assert "Continue with offset=20" in first
    second = _read_file(None, object(), file_id="fileid1234", offset=20, max_chars=500)
    assert second.endswith("END") or "END" in second
    assert "Continue with offset" not in second


def test_read_gmail_message_pages(monkeypatch) -> None:
    raw = ("word " * 30) + "CLOSE"
    encoded = base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")
    payload = {
        "payload": {
            "mimeType": "text/plain",
            "headers": [
                {"name": "From", "value": "Ada"},
                {"name": "Subject", "value": "Long"},
            ],
            "body": {"data": encoded},
        }
    }

    def handler(url, _params):
        return _Resp(payload=payload)

    monkeypatch.setattr("light_house.tools.gmail_tools.google_access_token", lambda *_a, **_k: "tok")
    _install_client(monkeypatch, "light_house.tools.gmail_tools", handler)

    first = _read_message(None, object(), message_id="msg1234", offset=0, max_chars=40)
    assert "CLOSE" not in first
    assert "Continue with offset=" in first
    assert "Subject: Long" in first
    second = _read_message(None, object(), message_id="msg1234", offset=len(raw) - 5, max_chars=500)
    assert "CLOSE" in second
    assert "Continue with offset" not in second


def test_short_doc_has_no_continuation(monkeypatch) -> None:
    document = {
        "title": "Short",
        "body": {"content": [{"paragraph": {"elements": [{"textRun": {"content": "Hello."}}]}}]},
    }
    monkeypatch.setattr("light_house.tools.docs_tools._google_access_token", lambda *_a, **_k: "tok")
    _install_client(
        monkeypatch,
        "light_house.tools.docs_tools",
        lambda _url, _params: _Resp(payload=document),
    )
    text = _read_doc(None, object(), doc_id_or_url="abcdefghij1234", offset=0, max_chars=8000)
    assert "Hello." in text
    assert "Continue with offset" not in text
    assert "(excerpt)" not in text
    assert "chars 0-6 of 6" in text


def test_tool_schemas_accept_offset() -> None:
    from light_house.tools.docs_tools import ReadGoogleDocArgs
    from light_house.tools.drive_tools import ReadDriveArgs
    from light_house.tools.gmail_tools import ReadGmailArgs

    assert ReadGoogleDocArgs(doc_id_or_url="abcdefghij1234", offset=12000).offset == 12000
    assert ReadDriveArgs(file_id="fileid1234", offset=40).offset == 40
    assert ReadGmailArgs(message_id="msg1234", offset=80).offset == 80
    # httpx imported so a missing dependency fails here rather than mid-test
    assert httpx.Client
