"""Gentle pause hints when recent rumination summaries repeat (Phase 6 item 4)."""

from __future__ import annotations

from light_house.config import Settings
from light_house.memory.dedup import is_near_duplicate_text
from light_house.rumination_log import read_rumination_log_entries

SIMILARITY_PAUSE_HINT = (
    "\n\n## Gentle pause (recent reflection)\n"
    "Your recent rumination summaries look very similar to one another. "
    "You may continue if something new is genuinely emerging — or simply rest now. "
    "There is no obligation to produce another reflection on the same ground."
)


def recent_reflection_similarity_hint(settings: Settings, *, agent_id: str) -> str:
    """Return a context hint when the latest summary repeats recent ones; never blocks."""
    if not settings.rumination_similarity_hint_enabled:
        return ""
    if not settings.rumination_log_enabled:
        return ""
    lookback = max(2, settings.rumination_similarity_lookback)
    path = settings.rumination_log_path.resolve()
    entries, _ = read_rumination_log_entries(
        path,
        max_lines=lookback,
        agent_id=agent_id,
    )
    summaries: list[str] = []
    for row in entries:
        line = row.get("summary_line")
        if not isinstance(line, str):
            continue
        clean = line.strip()
        if not clean or clean == "(no summary)":
            continue
        if clean.startswith("(error:"):
            continue
        summaries.append(clean)
    if len(summaries) < 2:
        return ""
    latest = summaries[0]
    min_matches = max(1, settings.rumination_similarity_min_matches)
    matches = sum(
        1 for prior in summaries[1:] if is_near_duplicate_text(latest, prior)
    )
    if matches >= min_matches:
        return SIMILARITY_PAUSE_HINT
    return ""
