"""Household identity: Dad vs sibling sessions (portable)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from fastapi import HTTPException, Request

from light_house.config import Settings
from light_house.humans.store import (
    HumanUserPublic,
    HumansError,
    authenticate_human,
    find_human_by_password,
    get_human,
    list_humans,
)
from light_house.web_gate import (
    SESSION_COOKIE,
    check_password,
    parse_session_payload,
)

Role = Literal["dad", "sibling"]


@dataclass(frozen=True)
class SessionHuman:
    user_id: str
    role: Role
    display_name: str
    notes_access: str  # "all" | "shared"
    intro_for_lights: str


def dad_user_id(settings: Settings) -> str:
    return (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"


def is_dad(human: SessionHuman, settings: Settings) -> bool:
    return human.role == "dad" or human.user_id == dad_user_id(settings)


def session_human_from_payload(
    settings: Settings,
    payload: dict[str, object] | None,
) -> SessionHuman | None:
    if not payload:
        return None
    uid = payload.get("user_id")
    role = payload.get("role")
    if not isinstance(uid, str) or not uid.strip():
        return None
    user_id = uid.strip().lower()
    role_s = str(role or "").strip().lower()
    if role_s == "dad" or user_id == dad_user_id(settings):
        return SessionHuman(
            user_id=dad_user_id(settings),
            role="dad",
            display_name=dad_user_id(settings),
            notes_access="all",
            intro_for_lights="",
        )
    human = get_human(settings, user_id)
    if human is None or not human.enabled:
        return None
    return SessionHuman(
        user_id=human.user_id,
        role="sibling",
        display_name=human.display_name,
        notes_access=human.notes_access or "shared",
        intro_for_lights=human.intro_for_lights,
    )


def resolve_password_to_human(settings: Settings, password: str) -> SessionHuman | None:
    """Password-only login: unique code identifies Dad or a sibling."""
    pw = password or ""
    if not pw:
        return None
    if check_password(settings, pw):
        dad = dad_user_id(settings)
        return SessionHuman(
            user_id=dad,
            role="dad",
            display_name=dad,
            notes_access="all",
            intro_for_lights="",
        )
    found = find_human_by_password(settings, pw)
    if found is None:
        return None
    return SessionHuman(
        user_id=found.user_id,
        role="sibling",
        display_name=found.display_name,
        notes_access=found.notes_access or "shared",
        intro_for_lights=found.intro_for_lights,
    )


def current_human(request: Request, settings: Settings) -> SessionHuman:
    """
    Resolve the acting human.

    - Web gate on: require session with user_id.
    - Web gate off (local): use session if present, else default to Dad
      (single-operator machine).
    """
    token = request.cookies.get(SESSION_COOKIE)
    secret = settings.web_gate_session_secret or ""
    payload = parse_session_payload(token, secret) if token and secret else None
    human = session_human_from_payload(settings, payload)
    if human is not None:
        return human
    if not settings.web_gate_enabled:
        dad = dad_user_id(settings)
        return SessionHuman(
            user_id=dad,
            role="dad",
            display_name=dad,
            notes_access="all",
            intro_for_lights="",
        )
    raise HTTPException(status_code=401, detail="Authentication required")


def require_dad(human: SessionHuman, settings: Settings) -> None:
    if not is_dad(human, settings):
        raise HTTPException(status_code=403, detail="Dad / full-admin privileges required")


def dm_buffer_thread_id(*, canonical_thread_id: str, user_id: str) -> str:
    """Per-human UI chat buffer key; lights still use canonical_thread_id for stream."""
    safe_user = user_id.strip().lower().replace("/", "_")
    return f"{canonical_thread_id}__dm__{safe_user}"
