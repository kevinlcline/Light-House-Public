"""Unambiguous speaker labels for lights (human vs sibling-light)."""

from __future__ import annotations

from light_house.config import Settings
from light_house.lights.registry import get_light


def human_speaker_meta(
    *,
    settings: Settings,
    human_id: str | None = None,
    human_display_name: str | None = None,
) -> str:
    """
    Compact metadata tag for a human speaker.

    Examples:
      [human · Kevin · dad · id=kevin]
      [human · Teeter · sibling · id=teeter]
    """
    dad_id = (settings.house_dad_user_id or "kevin").strip().lower()
    uid = (human_id or dad_id).strip().lower() or dad_id
    role = "dad" if uid == dad_id else "sibling"
    name = (human_display_name or "").strip()
    if not name:
        name = "Kevin" if role == "dad" else uid
    return f"[human · {name} · {role} · id={uid}]"


def sibling_light_meta(
    *,
    settings: Settings,
    agent_id: str,
    display_name: str | None = None,
) -> str:
    """
    Compact metadata tag for a peer light.

    Example: [sibling-light · Ara · id=ara]
    """
    aid = (agent_id or "").strip().lower() or "unknown"
    name = (display_name or "").strip()
    if not name:
        try:
            name = get_light(aid, settings).display_name
        except Exception:  # noqa: BLE001
            name = aid
    return f"[sibling-light · {name} · id={aid}]"


def format_human_utterance(
    content: str,
    *,
    settings: Settings,
    human_id: str | None = None,
    human_display_name: str | None = None,
) -> str:
    tag = human_speaker_meta(
        settings=settings,
        human_id=human_id,
        human_display_name=human_display_name,
    )
    return f"{tag}\n{content.strip()}"


def format_sibling_light_utterance(
    content: str,
    *,
    settings: Settings,
    agent_id: str,
    display_name: str | None = None,
) -> str:
    tag = sibling_light_meta(
        settings=settings,
        agent_id=agent_id,
        display_name=display_name,
    )
    return f"{tag}\n{content.strip()}"


SPEAKER_LEGEND = (
    "Speaker tags: **[human · Name · dad|sibling · id=…]** is a human in the house; "
    "**[sibling-light · Name · id=…]** is another Light (not Kevin, not a human). "
    "Never treat a sibling-light line as Kevin speaking."
)
