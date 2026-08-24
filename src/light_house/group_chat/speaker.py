"""Resolve who a group utterance is attributed to (account vs present guest)."""

from __future__ import annotations

import re
from typing import Any

_GUEST_SPEAKER_RE = re.compile(r"^guest-[1-9]\d{0,2}$")
_DISPLAY_NAME_RE = re.compile(r"^[\w][\w .'\-]{0,39}$", re.UNICODE)
_MAX_DISPLAY_NAME = 40
_MAX_PRESENT = 3  # account + guest-1 + guest-2


def _validate_display_name(dname: str) -> str:
    name = (dname or "").strip()
    if not name:
        raise ValueError("display_name is required when speaking as a guest")
    if len(name) > _MAX_DISPLAY_NAME:
        raise ValueError(f"display_name must be at most {_MAX_DISPLAY_NAME} characters")
    if not _DISPLAY_NAME_RE.match(name):
        raise ValueError(
            "display_name may use letters, numbers, spaces, periods, hyphens, apostrophes"
        )
    return name


def resolve_group_utterance_speaker(
    *,
    account_user_id: str,
    account_display_name: str,
    speaker_id: str | None = None,
    display_name: str | None = None,
) -> tuple[str, str]:
    """
    Return (speaker_id, display_name) for transcript / lights.

    Account identity still drives auth elsewhere. Guests are session-present
    names only: speaker_id must be guest-1, guest-2, … with a display_name.
    """
    account_id = (account_user_id or "").strip()
    if not account_id:
        raise ValueError("account user id is required")
    account_label = (account_display_name or "").strip() or account_id

    sid = (speaker_id or "").strip()
    dname = (display_name or "").strip()
    if not sid or sid == account_id:
        return account_id, account_label

    if not _GUEST_SPEAKER_RE.match(sid):
        raise ValueError(
            "speaker_id must be your account id or a guest slot (guest-1, guest-2, …)"
        )
    return sid, _validate_display_name(dname)


def normalize_present_humans(
    *,
    account_user_id: str,
    account_display_name: str,
    present: list[dict[str, Any]] | None = None,
) -> list[dict[str, str]]:
    """
    Build the room roster: account holder first, then validated guest slots.

    Ignores invalid rows rather than failing the whole send (guests are soft).
    """
    account_id = (account_user_id or "").strip()
    if not account_id:
        raise ValueError("account user id is required")
    account_label = (account_display_name or "").strip() or account_id
    out: list[dict[str, str]] = [
        {"speaker_id": account_id, "display_name": account_label, "description": ""}
    ]
    seen = {account_id.lower()}

    for row in present or []:
        if len(out) >= _MAX_PRESENT:
            break
        if not isinstance(row, dict):
            continue
        sid = str(row.get("speaker_id") or "").strip()
        dname = str(row.get("display_name") or "").strip()
        if not sid or sid.lower() in seen:
            continue
        if sid == account_id:
            continue
        if not _GUEST_SPEAKER_RE.match(sid):
            continue
        try:
            label = _validate_display_name(dname)
        except ValueError:
            continue
        blurb = re.sub(r"\s+", " ", str(row.get("description") or "").strip())
        if len(blurb) > 160:
            blurb = blurb[:160].rstrip()
        out.append(
            {
                "speaker_id": sid,
                "display_name": label,
                "description": blurb,
            }
        )
        seen.add(sid.lower())
    return out


def format_presence_for_prompt(present: list[dict[str, str]] | None) -> str:
    """Markdown block telling lights who is physically in the room."""
    people = list(present or [])
    if not people:
        return ""
    if len(people) == 1:
        name = people[0].get("display_name") or people[0].get("speaker_id") or "the host"
        return (
            f"## Who is present\n"
            f"Only **{name}** is in the room (no guests).\n"
        )
    names = ", ".join(
        f"**{p.get('display_name') or p.get('speaker_id')}**" for p in people
    )
    guest_blurbs = [
        p
        for p in people
        if _GUEST_SPEAKER_RE.match(str(p.get("speaker_id") or ""))
        and str(p.get("description") or "").strip()
    ]
    about = ""
    if guest_blurbs:
        about_lines = [
            f"- **{p.get('display_name') or p.get('speaker_id')}** — "
            f"{str(p.get('description') or '').strip()}"
            for p in guest_blurbs
        ]
        about = (
            "About the guests (tone notes only — see **Who is speaking** for who "
            "just talked):\n" + "\n".join(about_lines) + "\n"
        )
    return (
        f"## Who is present\n"
        f"People in the room right now: {names}.\n"
        f"This is the room roster, not necessarily who just spoke.\n"
        f"{about}"
        f"Adjust tone for the company — keep private, romantic, or highly personal "
        f"affection muted while guests are present unless the host clearly invites it.\n"
    )


def format_current_speaker_for_prompt(
    *,
    human_id: str,
    human_display_name: str,
    present_humans: list[dict[str, str]] | None = None,
) -> str:
    """Markdown block: who owns the current human utterance (host vs guest speak-as)."""
    sid = (human_id or "").strip()
    name = (human_display_name or "").strip() or sid
    if not name:
        return ""

    if _GUEST_SPEAKER_RE.match(sid):
        blurb = ""
        for person in present_humans or []:
            if str(person.get("speaker_id") or "").strip() == sid:
                blurb = str(person.get("description") or "").strip()
                break
        note = ""
        if blurb:
            note = (
                f"Guest note (tone only — still address them as **{name}**): {blurb}\n"
            )
        return (
            f"## Who is speaking (human)\n"
            f"**{name}** is a guest signed into the house (slot `{sid}`). "
            f"They are speaking through the host's device — address **{name}** by that "
            f"name, not the host account, even if the host is also listed under "
            f"**Who is present**.\n"
            f"{note}"
        )

    people = list(present_humans or [])
    if len(people) > 1:
        return (
            f"## Who is speaking (human)\n"
            f"**{name}** is speaking now (`{sid}`). Address them as **{name}**. "
            f"Others under **Who is present** are in the room but did not say this line.\n"
        )
    return (
        f"## Who is speaking (human)\n"
        f"**{name}** is speaking (`{sid}`).\n"
    )
