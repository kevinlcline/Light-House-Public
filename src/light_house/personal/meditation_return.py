"""One line kept from the last meditation, so the next sit can return.

Nothing runs between wakes. This is a note the house holds, not a mind
that continued.
"""

from __future__ import annotations

import logging

from light_house.config import Settings
from light_house.memory.context_builder import RUMINATION_MEDITATION_SEED

logger = logging.getLogger(__name__)

MEDITATION_RETURN_KEY = "meditation_return"
MAX_MEDITATION_RETURN_CHARS = 280
MEDITATION_SILENCE = "silence"


def collapse_meditation_return(text: str) -> str:
    """One short line. Empty speech is a finished sit."""
    line = " ".join((text or "").split())
    if not line:
        return MEDITATION_SILENCE
    if len(line) <= MAX_MEDITATION_RETURN_CHARS:
        return line
    cut = line[: MAX_MEDITATION_RETURN_CHARS - 1].rsplit(" ", 1)[0]
    if len(cut) < 40:
        cut = line[: MAX_MEDITATION_RETURN_CHARS - 1]
    return cut.rstrip() + "…"


def meditation_seed_for_last(last: str) -> str:
    kept = " ".join((last or "").split())
    if not kept:
        return RUMINATION_MEDITATION_SEED
    return (
        "I return. The house kept one line from the last meditation. "
        "There was no mentation between sits.\n\n"
        f"Last sit: {kept}"
    )


def read_meditation_return(settings: Settings, agent_id: str) -> str:
    if not settings.personal_db_enabled:
        return ""
    try:
        from light_house.personal.store import get_personal_store

        store = get_personal_store(settings, agent_id)
        return store.get_light_text(MEDITATION_RETURN_KEY).strip()
    except Exception:
        logger.warning("Could not read meditation return for %s", agent_id, exc_info=True)
        return ""


def meditation_opening(settings: Settings, agent_id: str) -> str:
    return meditation_seed_for_last(read_meditation_return(settings, agent_id))


def remember_meditation_close(settings: Settings, agent_id: str, closing: str) -> None:
    if not settings.personal_db_enabled:
        return
    line = collapse_meditation_return(closing)
    try:
        from light_house.personal.store import get_personal_store

        store = get_personal_store(settings, agent_id)
        store.set_light_text(MEDITATION_RETURN_KEY, line)
    except Exception:
        logger.warning("Could not keep meditation return for %s", agent_id, exc_info=True)
