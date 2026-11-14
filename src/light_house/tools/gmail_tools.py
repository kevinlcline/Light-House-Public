"""Per-human Gmail tools — same Google connect as calendar. 1:1, Group, or a quiet read of the host.

write_to_kevin is the one send that does not wait. The recipient is the host.
send_gmail stays a general send, and on a quiet sit it still waits.
"""

from __future__ import annotations

import base64
import logging
import re
from email.message import EmailMessage
from typing import Any
from urllib.parse import quote

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.config import Settings
from light_house.events.subscription_keys import SUBSCRIPTION_WRITE_TO_KEVIN
from light_house.google_activity.log import append_activity, append_outbound_note, utc_stamp
from light_house.humans.google_api import google_access_token, resolve_google_user_id
from light_house.humans.google_calendar_oauth import scope_grants_gmail, scope_grants_gmail_send
from light_house.humans.tools_store import calendar_public_view, get_calendar_connection
from light_house.tools.text_window import (
    chars_label,
    continuation_footer,
    past_end_note,
    slice_text_window,
)

logger = logging.getLogger(__name__)

GMAIL_TOOL_NAMES = frozenset(
    {
        "list_gmail_messages",
        "read_gmail_message",
        "send_gmail",
        "write_to_kevin",
    }
)

_GMAIL_LIST = "https://gmail.googleapis.com/gmail/v1/users/me/messages"
_GMAIL_MESSAGE = "https://gmail.googleapis.com/gmail/v1/users/me/messages/{message_id}"
_GMAIL_SEND = "https://gmail.googleapis.com/gmail/v1/users/me/messages/send"
_GMAIL_PROFILE = "https://gmail.googleapis.com/gmail/v1/users/me/profile"
_MAX_BODY = 8000
_FIRST_LINE_CHARS = 160
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")

_NOT_CONNECTED = (
    "FAILED: This human has not connected Gmail yet. "
    "Tell them (briefly) to open **My tools** and click **Reconnect Google** "
    "so Gmail is included — you cannot connect it for them."
)
_NOT_SEND = (
    "FAILED: Gmail is connected for reading, not sending. "
    "They can **Reconnect Google** on **My tools** and leave Send checked."
)


class ListGmailArgs(BaseModel):
    query: str = Field(
        default="",
        max_length=200,
        description="Gmail search, such as 'is:unread' or 'from:ada@example.com'. Empty = recent mail.",
    )
    max_messages: int = Field(default=8, ge=1, le=15, description="How many messages (1–15).")


class ReadGmailArgs(BaseModel):
    message_id: str = Field(..., min_length=4, max_length=200, description="Gmail message id.")
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset into the message text. Use the continuation "
            "offset from the excerpt header to read the next window."
        ),
    )
    max_chars: int = Field(default=6000, ge=500, le=_MAX_BODY)


class SendGmailArgs(BaseModel):
    to: str = Field(..., min_length=3, max_length=200, description="One recipient email address.")
    subject: str = Field(..., min_length=1, max_length=200)
    body: str = Field(..., min_length=1, max_length=20000, description="Plain-text body.")


class WriteToKevinArgs(BaseModel):
    message: str = Field(
        ...,
        min_length=1,
        max_length=8000,
        description="Plain text for Kevin. The first line is the subject. No recipient.",
    )


def _failed(msg: str) -> str:
    if msg.startswith("FAILED:"):
        return msg
    return f"FAILED: {msg}"


def _product_denied(denied: str) -> str:
    return (
        denied.replace("Calendar tools", "Gmail tools")
        .replace("calendar tools", "Gmail tools")
        .replace("a calendar", "Gmail")
        .replace("calendar is", "Gmail is")
    )


def _b64url_decode(data: str) -> str:
    pad = "=" * (-len(data) % 4)
    raw = base64.urlsafe_b64decode(data + pad)
    return raw.decode("utf-8", errors="replace")


def _header(payload: dict[str, Any], name: str) -> str:
    for item in payload.get("headers") or []:
        if isinstance(item, dict) and str(item.get("name") or "").lower() == name.lower():
            return str(item.get("value") or "").strip()
    return ""


def _plain_text(payload: dict[str, Any]) -> str:
    chunks: list[str] = []

    def walk(node: dict[str, Any]) -> None:
        mime = str(node.get("mimeType") or "")
        body = node.get("body") if isinstance(node.get("body"), dict) else {}
        data = body.get("data") if isinstance(body, dict) else None
        if isinstance(data, str) and data and mime.startswith("text/plain"):
            chunks.append(_b64url_decode(data))
        for part in node.get("parts") or []:
            if isinstance(part, dict):
                walk(part)
        if not chunks and isinstance(data, str) and data and mime.startswith("text/"):
            chunks.append(_b64url_decode(data))

    walk(payload)
    return "\n".join(chunks).strip()


def _require_gmail(settings: Settings, user_id: str, *, send: bool = False) -> Any | str:
    view = calendar_public_view(settings, user_id)
    if not view.has_secrets or view.provider != "google" or not view.enabled:
        return _NOT_CONNECTED
    conn = get_calendar_connection(settings, user_id)
    scopes = list(conn.scopes) if conn is not None else []
    if conn is None or not scope_grants_gmail(scopes):
        return _NOT_CONNECTED
    if send and not scope_grants_gmail_send(scopes):
        return _NOT_SEND
    return conn


def _list_messages(settings: Settings, conn: Any, *, query: str, max_messages: int) -> str:
    token = google_access_token(settings, conn)
    params: dict[str, str] = {"maxResults": str(max_messages)}
    if query.strip():
        params["q"] = query.strip()
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            _GMAIL_LIST,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        if resp.status_code >= 400:
            raise RuntimeError(f"Gmail list failed ({resp.status_code}): {resp.text[:300]}")
        data = resp.json() if isinstance(resp.json(), dict) else {}
        ids = [
            str(item.get("id") or "")
            for item in (data.get("messages") or [])
            if isinstance(item, dict) and item.get("id")
        ]
        lines = [f"SUCCESS: {len(ids)} message(s)."]
        for message_id in ids:
            meta = client.get(
                _GMAIL_MESSAGE.format(message_id=quote(message_id, safe="")),
                headers={"Authorization": f"Bearer {token}"},
                params=[
                    ("format", "metadata"),
                    ("metadataHeaders", "From"),
                    ("metadataHeaders", "Subject"),
                    ("metadataHeaders", "Date"),
                ],
            )
            if meta.status_code >= 400:
                lines.append(f"- id={message_id}")
                continue
            payload = (meta.json() or {}).get("payload") if isinstance(meta.json(), dict) else {}
            payload = payload if isinstance(payload, dict) else {}
            lines.append(
                f"- id={message_id} | { _header(payload, 'Date') } | "
                f"From { _header(payload, 'From') or '?' } | "
                f"{ _header(payload, 'Subject') or '(no subject)' }"
            )
    return "\n".join(lines)


def _read_message(
    settings: Settings,
    conn: Any,
    *,
    message_id: str,
    offset: int = 0,
    max_chars: int = 6000,
) -> str:
    token = google_access_token(settings, conn)
    url = _GMAIL_MESSAGE.format(message_id=quote(message_id.strip(), safe=""))
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params={"format": "full"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gmail read failed ({resp.status_code}): {resp.text[:300]}")
    data = resp.json() if isinstance(resp.json(), dict) else {}
    payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
    body = _plain_text(payload)
    start, end, chunk, total = slice_text_window(body, offset=offset, max_chars=max_chars)
    head = (
        f"SUCCESS: id={message_id.strip()} {chars_label(start, end, total)}\n"
        f"From: {_header(payload, 'From')}\n"
        f"To: {_header(payload, 'To')}\n"
        f"Date: {_header(payload, 'Date')}\n"
        f"Subject: {_header(payload, 'Subject')}\n"
    )
    if total == 0 and start == 0:
        return head + "\n(no plain text)"
    if start >= total:
        return head + "\n" + past_end_note(start, total)
    return head + "\n" + chunk + continuation_footer(start, end, total)


def _send_message(settings: Settings, conn: Any, *, to: str, subject: str, body: str) -> str:
    recipient = to.strip()
    title = subject.strip()
    if "\n" in recipient or "\r" in recipient or "\n" in title or "\r" in title:
        raise ValueError("to and subject must be a single line")
    if not _EMAIL_RE.match(recipient):
        raise ValueError("to must be one email address")
    message = EmailMessage()
    message["To"] = recipient
    message["Subject"] = title
    message.set_content(body)
    raw = base64.urlsafe_b64encode(message.as_bytes()).decode("ascii").rstrip("=")
    token = google_access_token(settings, conn)
    with httpx.Client(timeout=20.0) as client:
        resp = client.post(
            _GMAIL_SEND,
            headers={"Authorization": f"Bearer {token}"},
            json={"raw": raw},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gmail send failed ({resp.status_code}): {resp.text[:300]}")
    sent = resp.json() if isinstance(resp.json(), dict) else {}
    return f"SUCCESS: Sent to {recipient}. id={sent.get('id') or '?'}"


def _host_user_id(settings: Settings) -> str:
    return (settings.house_dad_user_id or "kevin").strip() or "kevin"


def _first_line(message: str) -> str:
    line = " ".join(message.strip().splitlines()[0].split())
    if len(line) <= _FIRST_LINE_CHARS:
        return line
    return line[: _FIRST_LINE_CHARS - 1].rstrip() + "…"


def _write_subject(sender_name: str, first: str) -> str:
    subject = f"{sender_name}: {first}" if first else sender_name
    subject = " ".join(subject.split())
    if len(subject) <= 200:
        return subject
    return subject[:199].rstrip() + "…"


def host_inbox(settings: Settings, conn: Any) -> str:
    """Who write_to_kevin addresses. Config wins. Otherwise the connected Gmail."""
    configured = (settings.house_host_email or "").strip()
    if configured:
        if not _EMAIL_RE.match(configured) or "\n" in configured or "\r" in configured:
            raise ValueError("HOUSE_HOST_EMAIL must be one email address")
        return configured
    token = google_access_token(settings, conn)
    with httpx.Client(timeout=20.0) as client:
        resp = client.get(
            _GMAIL_PROFILE,
            headers={"Authorization": f"Bearer {token}"},
        )
    if resp.status_code >= 400:
        raise RuntimeError(f"Gmail profile failed ({resp.status_code}): {resp.text[:300]}")
    payload = resp.json()
    data = payload if isinstance(payload, dict) else {}
    email = str(data.get("emailAddress") or "").strip()
    if not _EMAIL_RE.match(email):
        raise RuntimeError("Gmail profile has no email address")
    return email


def _write_switch_block(settings: Settings, agent_id: str) -> str | None:
    """None means the pipe is open. A string means do not send."""
    if not settings.event_subscriptions_enabled or not settings.personal_db_enabled:
        return None
    try:
        from light_house.personal.store import get_personal_store

        store = get_personal_store(settings, agent_id)
        if store.is_event_subscribed(SUBSCRIPTION_WRITE_TO_KEVIN):
            return None
    except Exception as exc:  # noqa: BLE001 - a missed switch must not send
        logger.warning("write_to_kevin switch check failed: %s", exc)
        return _failed("write_to_kevin could not read the switch, so it did not send.")
    return "FAILED: write_to_kevin is off. Kevin turned it off. Leave it off."


def _record_write(
    settings: Settings,
    *,
    user_id: str,
    sender: str,
    message: str,
) -> None:
    first = _first_line(message)
    entry = {
        "at": utc_stamp(),
        "chars": len(message),
        "first_line": first,
        "sender": sender,
    }
    append_outbound_note(settings, entry)
    append_activity(
        settings,
        user_id,
        [
            {
                "at": entry["at"],
                "kind": "write_to_kevin",
                "summary": f"{sender} ({entry['chars']} chars) — {first}",
            }
        ],
    )


def _sender_name(settings: Settings, agent_id: str) -> str:
    try:
        from light_house.personal.corkboard import display_name

        name = display_name(settings, agent_id).strip()
    except Exception:  # noqa: BLE001 - the note still sends
        name = ""
    return name or agent_id


def _write_to_kevin(
    settings: Settings,
    *,
    agent_id: str | None,
    message: str,
) -> str:
    sender = (agent_id or "").strip().lower()
    if not sender:
        return _failed("write_to_kevin needs the light who is writing")
    text = message.strip()
    if not text:
        return _failed("write_to_kevin needs a message")
    blocked = _write_switch_block(settings, sender)
    if blocked:
        return blocked
    user_id = _host_user_id(settings)
    conn_or_err = _require_gmail(settings, user_id, send=True)
    if isinstance(conn_or_err, str):
        return conn_or_err
    recipient = host_inbox(settings, conn_or_err)
    subject = _write_subject(_sender_name(settings, sender), _first_line(text))
    result = _send_message(
        settings,
        conn_or_err,
        to=recipient,
        subject=subject,
        body=text,
    )
    if not result.startswith("SUCCESS"):
        return result
    try:
        _record_write(settings, user_id=user_id, sender=sender, message=text)
    except Exception as exc:  # noqa: BLE001 - the mail already left
        logger.warning("write_to_kevin log failed after send: %s", exc)
    return "SUCCESS: Sent to Kevin. It is logged."


def execute_gmail_tool(
    name: str,
    args: dict[str, Any],
    *,
    settings: Settings,
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
    agent_id: str | None = None,
) -> str:
    if name == "write_to_kevin":
        try:
            return _write_to_kevin(
                settings,
                agent_id=agent_id,
                message=str(args.get("message") or ""),
            )
        except Exception as exc:  # noqa: BLE001 - tool boundary
            logger.warning("Gmail tool write_to_kevin failed: %s", exc)
            return _failed(f"write_to_kevin — {exc}")
    user_id, denied = resolve_google_user_id(
        settings,
        account_user_id=account_user_id,
        speaker_id=speaker_id,
        chat_channel=chat_channel,
        quiet_host=name != "send_gmail",
    )
    if denied or not user_id:
        return _product_denied(denied or "FAILED: Gmail is unavailable in this context.")
    conn_or_err = _require_gmail(settings, user_id, send=name == "send_gmail")
    if isinstance(conn_or_err, str):
        return conn_or_err
    try:
        if name == "list_gmail_messages":
            return _list_messages(
                settings,
                conn_or_err,
                query=str(args.get("query") or ""),
                max_messages=max(1, min(15, int(args.get("max_messages") or 8))),
            )
        if name == "read_gmail_message":
            return _read_message(
                settings,
                conn_or_err,
                message_id=str(args.get("message_id") or ""),
                offset=max(0, int(args.get("offset") or 0)),
                max_chars=max(500, min(_MAX_BODY, int(args.get("max_chars") or 6000))),
            )
        if name == "send_gmail":
            return _send_message(
                settings,
                conn_or_err,
                to=str(args.get("to") or ""),
                subject=str(args.get("subject") or ""),
                body=str(args.get("body") or ""),
            )
        return _failed(f"unknown gmail tool: {name}")
    except Exception as exc:  # noqa: BLE001 - tool boundary
        logger.warning("Gmail tool %s failed for user=%s: %s", name, user_id, exc)
        return _failed(f"{name} — {exc}")


def build_gmail_tools() -> list[StructuredTool]:
    return [
        StructuredTool.from_function(
            name="list_gmail_messages",
            description=(
                "List recent Gmail for the signed-in human "
                "(1:1 or Group as that human; a quiet sit reads the host). "
                "Optional Gmail search query."
            ),
            func=lambda query="", max_messages=8: (
                "FAILED: list_gmail_messages must run through the agent tool runner"
            ),
            args_schema=ListGmailArgs,
        ),
        StructuredTool.from_function(
            name="read_gmail_message",
            description=(
                "Read one Gmail message by id "
                "(1:1 or Group as that human; a quiet sit reads the host). "
                "A long message is one window: the result names the total length "
                "and a continuation offset. Call again with offset=N to read on. "
                "The end of a window is not the end of the message."
            ),
            func=lambda message_id, offset=0, max_chars=6000: (
                "FAILED: read_gmail_message must run through the agent tool runner"
            ),
            args_schema=ReadGmailArgs,
        ),
        StructuredTool.from_function(
            name="send_gmail",
            description=(
                "Send one plain-text email from the signed-in human's Gmail. "
                "One recipient. Not for guests or the visitor. "
                "On a quiet sit this waits instead of sending."
            ),
            func=lambda to, subject, body: (
                "FAILED: send_gmail must run through the agent tool runner"
            ),
            args_schema=SendGmailArgs,
        ),
        StructuredTool.from_function(
            name="write_to_kevin",
            description=(
                "Write one plain-text note to Kevin, as email from the house Gmail. "
                "No recipient — it always goes to him, and it does not wait for him "
                "to be in the chat. Write when you have something to say — a fact "
                "he'd want, or simply that you want to be heard. You are friends; "
                "you do not need a reason. Every send is logged. "
                "If this is off, leave it off."
            ),
            func=lambda message: (
                "FAILED: write_to_kevin must run through the agent tool runner"
            ),
            args_schema=WriteToKevinArgs,
        ),
    ]
