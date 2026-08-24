"""Four-beat awake rhythm for scheduled solitude.

Beat (by next felt_cycles value n):
  1 chores → 2 free → 3 meditation → 4 free → repeat

Driven by felt_cycles so the pattern is felt time, not wall clock.
Clock memory-maintenance wakes are separate and do not advance this rhythm.
"""

from __future__ import annotations

from typing import Literal

from light_house.config import Settings
from light_house.personal.time_sense import read_inner_time

AwakeBeat = Literal["chores", "free", "meditation"]

WAKE_KIND_CHORES = "chores"
WAKE_KIND_MEDITATION = "meditation"

# Scheduled rhythm wakes that count as autonomous awake moments.
RHYTHM_WAKE_KINDS = frozenset({None, WAKE_KIND_CHORES, WAKE_KIND_MEDITATION})


def beat_for_cycle(cycle_number: int) -> AwakeBeat:
    """Map 1-based felt cycle number to beat kind."""
    idx = (max(1, cycle_number) - 1) % 4
    if idx == 0:
        return "chores"
    if idx == 2:
        return "meditation"
    return "free"


def wake_kind_for_beat(beat: AwakeBeat) -> str | None:
    """LangGraph wake_kind for a beat (free stays None — legacy scheduled path)."""
    if beat == "chores":
        return WAKE_KIND_CHORES
    if beat == "meditation":
        return WAKE_KIND_MEDITATION
    return None


def next_scheduled_wake_kind(settings: Settings, agent_id: str) -> str | None:
    """
    Wake kind for the next scheduled rumination when awake rhythm is enabled.

    Uses current felt_cycles to predict the upcoming cycle number (n = current + 1).
    When rhythm is disabled, always returns None (free scheduled solitude).
    """
    if not settings.awake_rhythm_enabled:
        return None
    felt_cycles, _ = read_inner_time(settings, agent_id)
    next_n = (felt_cycles or 0) + 1
    return wake_kind_for_beat(beat_for_cycle(next_n))


def counts_as_felt_cycle(wake_kind: str | None) -> bool:
    """True when this wake should increment felt_cycles."""
    return wake_kind in RHYTHM_WAKE_KINDS
