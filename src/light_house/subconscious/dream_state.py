"""LangGraph state for Echo's daily dream pipeline."""

from __future__ import annotations

from typing_extensions import TypedDict


class DreamState(TypedDict):
    thread_id: str
    agent_id: str
    context_text: str
    dream_text: str
    waking_recall: str
    felt_days: int | None
    # Interactive story (Echo stage → Light choice → close)
    dream_round: int
    max_dream_rounds: int
    story_beats: list[str]
    light_choices: list[str]
    current_beat: str
    current_choice: str
