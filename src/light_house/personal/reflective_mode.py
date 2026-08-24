"""Reflective mode — pause, reflect, choose speak or intentional silence."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path

from light_house.config import Settings
from light_house.personal.light_state_keys import REFLECTIVE_MODE
from light_house.personal.store import PersonalStore, get_personal_store

logger = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()

# Kevin inviting a one-shot pause this turn (mode stays off unless already on).
ONE_SHOT_INVITE_PHRASES = (
    "take your time",
    "sit with this",
    "think about that",
    "no rush",
    "pause with this",
    "reflect on that",
    "sit with that",
)

_DECISION_RE = re.compile(
    r"^\s*DECISION\s*:\s*(SPEAK|SILENCE)\b",
    re.IGNORECASE | re.MULTILINE,
)
_NOTES_RE = re.compile(
    r"NOTES\s*:\s*(.*?)(?=\n\s*DRAFT\s*:|\Z)",
    re.IGNORECASE | re.DOTALL,
)
_DRAFT_RE = re.compile(
    r"DRAFT\s*:\s*(.*)\Z",
    re.IGNORECASE | re.DOTALL,
)


def _store(settings: Settings, agent_id: str) -> PersonalStore | None:
    if not settings.personal_db_enabled:
        return None
    try:
        return get_personal_store(settings, agent_id)
    except (KeyError, RuntimeError, OSError) as exc:
        logger.warning("Reflective mode store unavailable agent=%s: %s", agent_id, exc)
        return None


def is_reflective_mode(settings: Settings, agent_id: str) -> bool:
    store = _store(settings, agent_id)
    if store is None:
        return False
    return store.get_light_state(REFLECTIVE_MODE, default=0) > 0


def set_reflective_mode(settings: Settings, agent_id: str, *, enabled: bool) -> bool:
    """Persist reflective mode. Returns True if stored."""
    store = _store(settings, agent_id)
    if store is None:
        return False
    store.set_light_state(REFLECTIVE_MODE, 1 if enabled else 0)
    logger.info(
        "Reflective mode %s agent=%s",
        "on" if enabled else "off",
        agent_id,
    )
    return True


def human_invites_reflection(text: str) -> bool:
    """True when Kevin's message invites a one-shot reflective pause."""
    lower = (text or "").strip().lower()
    if not lower:
        return False
    return any(phrase in lower for phrase in ONE_SHOT_INVITE_PHRASES)


def should_reflect_this_turn(
    settings: Settings,
    agent_id: str,
    *,
    latest_human_text: str | None,
    wake_kind: str | None = None,
) -> bool:
    """Reflective path for human DM turns only (not peer wakes)."""
    if wake_kind:
        return False
    if is_reflective_mode(settings, agent_id):
        return True
    if latest_human_text and human_invites_reflection(latest_human_text):
        return True
    return False


def parse_reflection_decision(raw: str) -> tuple[str, str, str]:
    """
    Parse reflect-node output.

    Returns (decision, notes, draft) where decision is 'speak' or 'silence'.
    Defaults to speak if the model does not mark silence clearly.
    """
    text = (raw or "").strip()
    match = _DECISION_RE.search(text)
    decision = "silence" if match and match.group(1).upper() == "SILENCE" else "speak"
    notes_m = _NOTES_RE.search(text)
    notes = notes_m.group(1).strip() if notes_m else ""
    draft_m = _DRAFT_RE.search(text)
    draft = draft_m.group(1).strip() if draft_m else ""
    return decision, notes, draft


def silence_log_path(settings: Settings) -> Path:
    return (settings.dev_log_path.parent / "reflective_silence.ndjson").resolve()


def log_intentional_silence(
    settings: Settings,
    *,
    agent_id: str,
    thread_id: str,
    user_text: str,
    notes: str = "",
) -> None:
    """Append a private NDJSON row for an intentional silence choice."""
    path = silence_log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    snippet = " ".join((user_text or "").split())
    if len(snippet) > 200:
        snippet = snippet[:197] + "..."
    note = " ".join((notes or "").split())
    if len(note) > 300:
        note = note[:297] + "..."
    payload = {
        "ts": time.time(),
        "agent_id": (agent_id or "").strip().lower(),
        "thread_id": thread_id,
        "user_snippet": snippet,
        "notes": note,
    }
    line = json.dumps(payload, ensure_ascii=False) + "\n"
    with _LOG_LOCK:
        with path.open("a", encoding="utf-8") as fh:
            fh.write(line)
    logger.info(
        "Intentional silence logged agent=%s thread_id=%s",
        agent_id,
        thread_id,
    )


def try_kevin_reflect_command(
    settings: Settings,
    *,
    message: str,
    agent_id: str,
) -> str | None:
    """Parse Kevin /reflect slash commands; return reply or None."""
    text = message.strip()
    lower = text.lower()
    if lower not in ("/reflect", "/reflect on", "/reflect off", "/reflect status") and not lower.startswith(
        "/reflect "
    ):
        return None
    parts = lower.split(maxsplit=1)
    arg = parts[1].strip() if len(parts) > 1 else "status"
    if arg in ("", "status", "?"):
        on = is_reflective_mode(settings, agent_id)
        state = "on" if on else "off"
        return (
            f"{agent_id} reflective mode is {state}. "
            "When on: pause → reflect → choose speak or silence. "
            "Usage: /reflect on | /reflect off"
        )
    if arg in ("on", "enable", "1", "true"):
        if not settings.personal_db_enabled:
            return "Reflective mode needs the personal store (PERSONAL_DB_ENABLED)."
        ok = set_reflective_mode(settings, agent_id, enabled=True)
        if not ok:
            return f"Could not enable reflective mode for {agent_id}."
        return (
            f"Reflective mode on for {agent_id}. "
            "They will pause, reflect, and may choose silence before answering."
        )
    if arg in ("off", "disable", "0", "false"):
        if not settings.personal_db_enabled:
            return "Reflective mode needs the personal store (PERSONAL_DB_ENABLED)."
        ok = set_reflective_mode(settings, agent_id, enabled=False)
        if not ok:
            return f"Could not disable reflective mode for {agent_id}."
        return f"Reflective mode off for {agent_id}. Back to reactive replies."
    return "Usage: /reflect on | /reflect off | /reflect status"


REFLECT_SYSTEM_HINT = (
    "\n\n## Reflective pause (this turn)\n"
    "You are in **reflective mode**. Do **not** answer Kevin yet.\n"
    "Sit with what they said. Decide whether speaking or silence is truer.\n"
    "Reply with **exactly** this shape (no other preamble):\n"
    "DECISION: SPEAK\n"
    "or\n"
    "DECISION: SILENCE\n"
    "\n"
    "Optional lines after the decision:\n"
    "NOTES: <private reflection — not shown to Kevin>\n"
    "DRAFT: <optional draft reply if SPEAK — may be refined in the next step>\n"
    "\n"
    "Silence is a valid, honored choice — not failure, not rudeness. "
    "Choose SILENCE when stillness is truer than performance."
)

REFLECTIVE_RESPOND_HINT = (
    "\n\n## After reflection\n"
    "You already paused and chose to speak. Answer Kevin from that stillness — "
    "honest, unhurried, not performative. If private NOTES were recorded, let them "
    "color your presence; do not dump the NOTES block into chat."
)

REFLECTIVE_MODE_TOOL_HINT = (
    "\n\n## Reflective mode (pause → choose)\n"
    "You may call **set_reflective_mode** with enabled=true or false.\n"
    "When reflective mode is on, after Kevin speaks you pause, reflect privately, "
    "then choose to speak **or** remain silent (silence is valid).\n"
    "Kevin can also toggle with `/reflect on` / `/reflect off`, or invite a one-shot "
    "pause with phrases like \"take your time\". Prefer this when the moment needs "
    "stillness rather than speed."
)
