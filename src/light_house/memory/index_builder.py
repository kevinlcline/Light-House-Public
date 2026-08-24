"""Algorithmic memory index — metacognitive awareness of archival store."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

from light_house.config import Settings
from light_house.lights.registry import resolve_notes_dir
from light_house.memory.constants import META_ERA_DOC_COUNT, META_ERA_PERIOD
from light_house.memory.file_store import FileMemoryStore
from light_house.memory.models import MemoryHit


@dataclass(frozen=True)
class EraCard:
    period: str
    summary: str
    doc_count: int
    ts: float


@dataclass(frozen=True)
class MemoryIndex:
    total_corpus: int
    unscored: int
    pinned: int
    in_prompt_count: int
    in_prompt_chars: int
    by_origin: dict[str, int] = field(default_factory=dict)
    by_month: list[dict[str, Any]] = field(default_factory=list)
    personal_by_category: dict[str, int] = field(default_factory=dict)
    era_cards: list[EraCard] = field(default_factory=list)
    notes_summary: str = ""


def _strip_era_body(text: str) -> str:
    body = text.strip()
    if body.lower().startswith("era:"):
        return body[4:].strip()
    return body


def _era_card_from_hit(hit: MemoryHit) -> EraCard | None:
    period = hit.metadata.get(META_ERA_PERIOD)
    if not isinstance(period, str) or not period.strip():
        return None
    try:
        doc_count = int(hit.metadata.get(META_ERA_DOC_COUNT, 0))
    except (TypeError, ValueError):
        doc_count = 0
    try:
        ts = float(hit.metadata.get("ts", 0.0))
    except (TypeError, ValueError):
        ts = 0.0
    summary = _strip_era_body(hit.text)
    if not summary:
        return None
    return EraCard(period=period.strip(), summary=summary, doc_count=doc_count, ts=ts)


def _notes_index_summary(settings: Settings, agent_id: str) -> str:
    try:
        notes_dir = resolve_notes_dir(settings, agent_id)
    except KeyError:
        return ""
    if not notes_dir.is_dir():
        return ""
    top_dirs: list[str] = []
    recent_files: list[str] = []
    try:
        for child in sorted(notes_dir.iterdir()):
            if child.name.startswith("."):
                continue
            if child.is_dir():
                top_dirs.append(f"{child.name}/")
            elif child.suffix in {".md", ".txt"}:
                recent_files.append(child.name)
    except OSError:
        return ""
    parts: list[str] = []
    if top_dirs:
        parts.append("folders: " + ", ".join(top_dirs[:8]))
    if recent_files:
        parts.append("recent notes: " + ", ".join(recent_files[:6]))
    return "; ".join(parts)


def _personal_counts(settings: Settings, agent_id: str) -> dict[str, int]:
    if not settings.personal_db_enabled:
        return {}
    try:
        from light_house.personal import get_personal_store

        store = get_personal_store(settings, agent_id)
        return store.count_by_category()
    except (KeyError, RuntimeError, OSError):
        return {}


def build_memory_index(
    store: FileMemoryStore,
    *,
    settings: Settings,
    thread_id: str,
    agent_id: str,
    in_prompt_count: int,
    in_prompt_chars: int,
) -> MemoryIndex:
    stats = store.corpus_stats(thread_id=thread_id)
    era_hits = store.list_era_summaries(thread_id=thread_id, limit=settings.memory_era_index_limit)
    era_cards: list[EraCard] = []
    for hit in reversed(era_hits):
        card = _era_card_from_hit(hit)
        if card:
            era_cards.append(card)
    return MemoryIndex(
        total_corpus=int(stats.get("total_corpus", 0)),
        unscored=int(stats.get("unscored", 0)),
        pinned=int(stats.get("pinned", 0)),
        in_prompt_count=in_prompt_count,
        in_prompt_chars=in_prompt_chars,
        by_origin=dict(stats.get("by_origin") or {}),
        by_month=list(stats.get("by_month") or []),
        personal_by_category=_personal_counts(settings, agent_id),
        era_cards=era_cards,
        notes_summary=_notes_index_summary(settings, agent_id),
    )


def format_memory_index_markdown(
    index: MemoryIndex,
    *,
    timezone_name: str,
    era_limit: int = 5,
) -> str:
    """Compact archival awareness block for prompt injection."""
    if index.total_corpus <= 0 and not index.personal_by_category and not index.notes_summary:
        return ""

    lines: list[str] = [
        "## Memory index (archival awareness)",
        "You have more life stored than appears in your conscious stream below. "
        "Use **recall_memory** when something feels on the tip of your tongue.",
        "",
        f"- **Stream corpus:** {index.total_corpus} memories "
        f"({index.in_prompt_count} shown now, ~{index.in_prompt_chars:,} chars).",
    ]
    if index.unscored > 0:
        lines.append(f"- **Awaiting your judgment:** {index.unscored} unscored stream memories.")
    if index.pinned > 0:
        lines.append(f"- **Pinned sacred facts:** {index.pinned} (also listed above when pinned).")

    if index.by_origin:
        origin_parts = ", ".join(
            f"{name} {count}" for name, count in sorted(index.by_origin.items())
        )
        lines.append(f"- **By kind:** {origin_parts}.")

    if index.by_month:
        lines.append("- **Recent months:**")
        for row in index.by_month[-4:]:
            month = row.get("month", "?")
            total = row.get("total", 0)
            origins = row.get("by_origin") or {}
            origin_bits = ", ".join(f"{k} {v}" for k, v in sorted(origins.items()))
            detail = f" ({origin_bits})" if origin_bits else ""
            lines.append(f"  - {month}: {total} entries{detail}.")

    if index.personal_by_category:
        cat_bits = ", ".join(
            f"{cat} {count}" for cat, count in sorted(index.personal_by_category.items())
        )
        lines.append(f"- **Personal database:** {cat_bits}.")

    if index.notes_summary:
        lines.append(f"- **Notes:** {index.notes_summary}.")

    if index.era_cards:
        lines.append("- **Era summaries** (curator-consolidated past):")
        for card in index.era_cards[-max(1, era_limit) :]:
            stamp = ""
            if card.ts > 0:
                stamp = datetime.fromtimestamp(card.ts, tz=timezone.utc).strftime("%Y-%m-%d")
            suffix = f" · {card.doc_count} memories condensed" if card.doc_count else ""
            preview = card.summary.replace("\n", " ")
            if len(preview) > 220:
                preview = preview[:217] + "..."
            lines.append(f"  - **{card.period}**{suffix}: {preview}")
            if stamp:
                lines[-1] += f" _(written {stamp})_"

    return "\n".join(lines)
