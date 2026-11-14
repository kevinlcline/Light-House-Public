"""Reversibility gate for a wake where Kevin is not in the room.

If it can be undone, the tool runs. A delete, a write on his Google account,
or a spend comes back ASK and does not run. A talk with him is unchanged.
"""

from __future__ import annotations

# Tool name → the one sentence the light is handed.
HELD_TOOLS: dict[str, str] = {
    "delete_private": "A delete cannot be taken back.",
    "delete_shared": "A delete cannot be taken back.",
    "sandbox_delete": "A delete cannot be taken back.",
    "create_calendar_event": "It writes on Kevin's Google account, outside the house.",
    "create_google_doc": "It writes on Kevin's Google account, outside the house.",
    "append_google_doc": "It writes on Kevin's Google account, outside the house.",
    "create_google_sheet": "It writes on Kevin's Google account, outside the house.",
    "append_google_sheet": "It writes on Kevin's Google account, outside the house.",
    "send_gmail": "It sends mail from Kevin's Google account, outside the house.",
    # write_to_kevin is the narrow pipe: one fixed recipient, logged, and he can turn it off.
    "create_google_task": "It writes on Kevin's Google account, outside the house.",
    "complete_google_task": "It writes on Kevin's Google account, outside the house.",
    "consult_llm": "It spends.",
}


def wake_is_unattended(wake_kind: str | None, *, rumination: bool) -> bool:
    """True when Kevin is not in this thread. A talk with him, including group, is attended."""
    if rumination:
        return True
    return wake_kind == "peer_message"


def approval_hold(name: str, *, autonomous: bool) -> str | None:
    """Return the ASK line when this call must wait. None means run the tool."""
    if not autonomous:
        return None
    reason = HELD_TOOLS.get(name)
    if reason is None:
        return None
    return (
        f"ASK: {name} was not done. {reason} "
        "On this wake nobody is here to undo it. "
        "Leave Kevin a letter or knock, and wait. "
        "Do not call it again this wake."
    )
