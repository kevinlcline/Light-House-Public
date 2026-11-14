"""Light tool: private sibling corkboard pins."""

from __future__ import annotations

import logging
from typing import Any

from light_house.config import Settings
from light_house.personal.corkboard import (
    assert_household_light,
    display_name,
    expiry_seconds,
    format_pin_preview,
    get_corkboard_store,
    normalize_body,
    normalize_subject,
    pin_visible_to,
)

logger = logging.getLogger(__name__)


def execute_corkboard_tool(settings: Settings, agent_id: str, args: dict[str, Any]) -> str:
    if not settings.corkboard_enabled:
        return "Corkboard is disabled (CORKBOARD_ENABLED=false)."
    try:
        me = assert_household_light(settings, agent_id)
    except (KeyError, ValueError) as exc:
        return f"corkboard failed: {exc}"
    action = str(args.get("action") or "list").strip().lower()
    try:
        store = get_corkboard_store(settings)
        store.purge_expired()
        if action == "pin":
            return _pin(settings, store, me, args)
        if action == "list":
            return _list(settings, store, me, args)
        if action == "read":
            return _read(settings, store, me, args)
        if action == "reply":
            return _reply(settings, store, me, args)
        if action == "keep":
            return _keep(settings, store, me, args)
        if action in ("clear", "remove", "drop"):
            return _clear(store, me, args)
    except ValueError as exc:
        return f"corkboard failed: {exc}"
    except Exception as exc:
        logger.warning("corkboard failed: %s", exc)
        return f"corkboard failed: {exc}"
    return (
        "corkboard action should be pin, list, read, reply, keep, or clear. "
        "This board is only between two lights. Kevin does not see the words."
    )


def _pin(settings: Settings, store, me: str, args: dict[str, Any]) -> str:
    to_raw = str(args.get("to_light") or args.get("to_agent_id") or args.get("to") or "").strip()
    if not to_raw:
        return "corkboard pin needs to_light (the sibling id)."
    try:
        to_id = assert_household_light(settings, to_raw)
    except (KeyError, ValueError) as exc:
        return f"corkboard failed: {exc}"
    if to_id == me:
        return "corkboard pin is for a sibling, not yourself."
    body = normalize_body(str(args.get("body") or args.get("message") or args.get("text") or ""))
    subject = normalize_subject(str(args.get("subject") or args.get("title") or ""))
    pin = store.pin(
        from_light=me,
        to_light=to_id,
        subject=subject,
        body=body,
        ttl_seconds=expiry_seconds(settings),
    )
    who = display_name(settings, to_id)
    days = max(1, int(settings.corkboard_pin_expiry_days))
    extra = f" Subject: {pin.subject}." if pin.subject else ""
    return (
        f"Pinned for {who} ({pin.id}).{extra} "
        f"It fades in about {days} days unless they keep it. "
        "They will see a quiet unread glance — not a live wake."
    )


def _list(settings: Settings, store, me: str, args: dict[str, Any]) -> str:
    sent = str(args.get("direction") or "").strip().lower() in ("sent", "out", "from")
    unread_only = _truthy(args.get("unread_only"))
    pins = store.list_for(me, incoming=not sent, unread_only=unread_only)
    if not pins:
        if sent:
            return "No pins you have left on the corkboard."
        return "Nothing pinned for you right now."
    header = "Pins you left:" if sent else "Pins for you:"
    lines = [header]
    for pin in pins:
        lines.append(format_pin_preview(pin, settings))
    lines.append("Read one with corkboard(action=\"read\", pin_id=…).")
    return "\n".join(lines)


def _read(settings: Settings, store, me: str, args: dict[str, Any]) -> str:
    pin_id = str(args.get("pin_id") or args.get("id") or "").strip()
    if not pin_id:
        return "corkboard read needs pin_id."
    pin = store.get(pin_id)
    if pin is None or not pin_visible_to(pin, me):
        return "That pin is not on your corkboard (unknown, expired, or not yours)."
    if pin.to_light == me:
        pin = store.mark_read(pin_id, reader_id=me) or pin
    from_name = display_name(settings, pin.from_light)
    to_name = display_name(settings, pin.to_light)
    subject = pin.subject or "(no subject)"
    kept = " kept" if pin.kept else ""
    return (
        f"Corkboard pin {pin.id}{kept}\n"
        f"From {from_name} to {to_name}\n"
        f"Subject: {subject}\n\n"
        f"{pin.body}"
    )


def _reply(settings: Settings, store, me: str, args: dict[str, Any]) -> str:
    pin_id = str(args.get("pin_id") or args.get("id") or "").strip()
    if not pin_id:
        return "corkboard reply needs pin_id."
    original = store.get(pin_id)
    if original is None or not pin_visible_to(original, me):
        return "That pin is not on your corkboard (unknown, expired, or not yours)."
    if original.to_light != me:
        return "Reply from the pin left for you — this one is yours to them already."
    body = normalize_body(str(args.get("body") or args.get("message") or args.get("text") or ""))
    subject = normalize_subject(
        str(args.get("subject") or args.get("title") or original.subject or "")
    )
    store.mark_read(pin_id, reader_id=me)
    reply = store.pin(
        from_light=me,
        to_light=original.from_light,
        subject=subject,
        body=body,
        ttl_seconds=expiry_seconds(settings),
    )
    who = display_name(settings, original.from_light)
    return (
        f"Pinned a reply for {who} ({reply.id}). "
        "A new pin, not a thread. They will see a quiet unread glance."
    )


def _keep(settings: Settings, store, me: str, args: dict[str, Any]) -> str:
    pin_id = str(args.get("pin_id") or args.get("id") or "").strip()
    if not pin_id:
        return "corkboard keep needs pin_id."
    pin = store.get(pin_id)
    if pin is None or not pin_visible_to(pin, me):
        return "That pin is not on your corkboard (unknown, expired, or not yours)."
    kept = store.keep(pin_id, ttl_seconds=expiry_seconds(settings))
    days = max(1, int(settings.corkboard_pin_expiry_days))
    return f"Kept pin {kept.id if kept else pin_id}. It will fade in about {days} more days."


def _clear(store, me: str, args: dict[str, Any]) -> str:
    pin_id = str(args.get("pin_id") or args.get("id") or "").strip()
    if not pin_id:
        return "corkboard clear needs pin_id."
    pin = store.get(pin_id)
    if pin is None or not pin_visible_to(pin, me):
        return "That pin is not on your corkboard (unknown, expired, or not yours)."
    store.clear(pin_id)
    return f"Cleared pin {pin_id}."


def _truthy(value: object) -> bool:
    if isinstance(value, bool):
        return value
    return str(value or "").strip().lower() in ("1", "true", "yes", "on")
