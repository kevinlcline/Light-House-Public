"""Light-owned in-progress names — reminded on free awake only."""

from __future__ import annotations

from dataclasses import dataclass

from light_house.config import Settings

MAX_IN_PROGRESS = 20
MAX_NAME_LEN = 80
MAX_NOTE_LEN = 280

# Free scheduled solitude: wake_kind is None (chores/meditation/other wakes stay quiet).
FREE_AWAKE_WAKE_KIND = None


@dataclass(frozen=True)
class InProgressItem:
    name: str
    note: str
    named_at: float
    updated_at: float


def normalize_in_progress_name(name: str) -> tuple[str, str]:
    display = " ".join((name or "").split())
    if not display:
        raise ValueError("Name a task — in_progress needs a name.")
    if len(display) > MAX_NAME_LEN:
        raise ValueError(f"Name is too long (max {MAX_NAME_LEN} characters).")
    return display, display.casefold()


def normalize_in_progress_note(note: str) -> str:
    text = " ".join((note or "").split())
    if len(text) > MAX_NOTE_LEN:
        raise ValueError(f"Note is too long (max {MAX_NOTE_LEN} characters).")
    return text


def _unsubscribe_query(name: str) -> str:
    display = " ".join((name or "").split())
    if not display:
        raise ValueError("Name a task — in_progress needs a name.")
    return display.casefold()


def match_in_progress_item(name: str, items: list[InProgressItem]) -> InProgressItem | None:
    """Resolve a drop.

    Exact name, a unique prefix of a stored name, or a pasted list line
    (the name, then the note). The query is not length-capped: a stored
    name must stay droppable even when the pasted line is long.
    """
    key = _unsubscribe_query(name)
    exact = [item for item in items if item.name.casefold() == key]
    if len(exact) == 1:
        return exact[0]

    prefixed = [item for item in items if item.name.casefold().startswith(key)]
    if len(prefixed) == 1:
        return prefixed[0]
    if len(prefixed) > 1:
        listed = ", ".join(item.name for item in prefixed)
        raise ValueError(
            "More than one in-progress name starts that way: "
            f"{listed}. Give more of the name."
        )

    covered = [item for item in items if key.startswith(item.name.casefold())]
    if not covered:
        return None
    covered.sort(key=lambda item: len(item.name), reverse=True)
    best = covered[0]
    rest = key[len(best.name.casefold()) :]
    if rest and not rest.startswith(" —") and not rest.startswith(" -") and not rest.startswith("—"):
        return None
    tied = [
        item
        for item in covered
        if len(item.name) == len(best.name) and item.name.casefold() != best.name.casefold()
    ]
    if tied:
        listed = ", ".join(item.name for item in [best, *tied])
        raise ValueError(
            "More than one in-progress name matches that line: "
            f"{listed}. Give more of the name."
        )
    return best


def is_free_awake(wake_kind: str | None) -> bool:
    return wake_kind is FREE_AWAKE_WAKE_KIND


def format_in_progress_items(items: list[InProgressItem]) -> str:
    if not items:
        return "Nothing on your in-progress list."
    lines = ["You named these as still in progress:"]
    for item in items:
        if item.note:
            lines.append(f"- {item.name} — {item.note}")
        else:
            lines.append(f"- {item.name}")
    return "\n".join(lines)


def format_in_progress_free_wake(items: list[InProgressItem]) -> str:
    """Glance for a free awake. Empty when they named nothing."""
    if not items:
        return ""
    lines = [
        "## In progress (you named these)",
        "A glance only — not a chore and not an order. "
        "Continue, leave them, or drop one with **in_progress** "
        "(action=unsubscribe, name= a unique prefix of the name, "
        "or the list line). Kevin does not add items.",
    ]
    for item in items:
        if item.note:
            lines.append(f"- {item.name} — {item.note}")
        else:
            lines.append(f"- {item.name}")
    return "\n".join(lines)


def format_in_progress_free_wake_for_agent(
    settings: Settings,
    agent_id: str,
    *,
    wake_kind: str | None,
) -> str:
    if not is_free_awake(wake_kind):
        return ""
    if not settings.personal_db_enabled:
        return ""
    try:
        from light_house.personal.store import get_personal_store

        store = get_personal_store(settings, agent_id)
        return format_in_progress_free_wake(store.list_in_progress())
    except Exception:
        return ""
