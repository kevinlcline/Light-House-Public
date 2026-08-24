"""Echo dream gather — recent dream history + agent waking context."""

from __future__ import annotations

from datetime import datetime, timezone

from light_house.agents.registry import get_agent
from light_house.config import Settings
from light_house.memory.context_builder import (
    build_agent_context,
    format_agent_context_markdown,
    format_dream_hit,
    format_inner_time_context,
)
from light_house.memory.models import MemoryHit
from light_house.memory.service import MemoryService, _meta_float

_ECHO_STREAM_SECTION_FOOTER = (
    "\n\nThis is the Light's recent waking life—chat, solitude, and peer messages. "
    "Dream entries are omitted here; see your recent dreams section above."
)


def format_recent_dreams_markdown(
    hits: list[MemoryHit],
    *,
    max_chars: int,
) -> str:
    """Numbered dream history, newest first; drops oldest when over char budget."""
    if not hits:
        return "No prior dreams yet for this Light. Weave from waking context alone."

    sorted_hits = sorted(
        hits,
        key=lambda h: _meta_float(h.metadata, "ts", 0.0),
        reverse=True,
    )
    blocks: list[str] = []
    used = 0
    for index, hit in enumerate(sorted_hits, start=1):
        text = format_dream_hit(hit).strip()
        if not text:
            continue
        ts = _meta_float(hit.metadata, "ts", 0.0)
        date = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        block = f"### Dream {index} · {date}\n{text}"
        block_len = len(block)
        if blocks and used + block_len + 2 > max_chars:
            break
        if not blocks and block_len > max_chars:
            blocks.append(block[:max_chars])
            break
        blocks.append(block)
        used += block_len + (2 if blocks else 0)

    if not blocks:
        return "No prior dreams yet for this Light. Weave from waking context alone."
    return "\n\n".join(blocks)


def _format_echo_agent_context_markdown(bundle) -> str:
    """Agent context for Echo — same sections as chat, with Echo-specific stream footer."""
    from light_house.memory.context_builder import (
        _PERSONAL_SECTION_FOOTER,
        format_stream_entry,
    )

    parts: list[str] = []

    if bundle.foundation.strip():
        parts.append(
            "## Kevin foundation context (always true; do not contradict)\n"
            + bundle.foundation.strip()
        )

    if bundle.personal_knowledge.strip():
        parts.append(
            "## Your personal knowledge\n"
            + bundle.personal_knowledge.strip()
            + _PERSONAL_SECTION_FOOTER
        )

    if bundle.event_subscriptions.strip():
        parts.append(bundle.event_subscriptions.strip())

    inner_time = format_inner_time_context(bundle.felt_cycles, bundle.felt_days)
    if inner_time.strip():
        parts.append(inner_time.strip())

    if bundle.pinned_facts:
        joined = "\n- ".join(bundle.pinned_facts)
        parts.append("## Pinned sacred facts\n- " + joined)

    if bundle.conscious_stream:
        blocks = [format_stream_entry(e) for e in bundle.conscious_stream]
        parts.append(
            "## Your conscious stream\n\n"
            + "\n\n".join(blocks)
            + _ECHO_STREAM_SECTION_FOOTER
        )

    if not parts:
        return "No recent waking context. Dream from your quiet presence and the bond you sense."
    return "\n\n".join(parts)


def build_echo_dream_context(
    memory: MemoryService,
    *,
    thread_id: str,
    agent_id: str,
    settings: Settings,
) -> tuple[str, int]:
    """
    Assemble Echo's HumanMessage body: recent dreams + agent waking context (no dreams in stream).

    Returns (markdown, recent_dream_count).
    """
    lookback = max(1, settings.echo_dream_lookback)
    history_cap = max(0, settings.echo_dream_history_max_chars)
    agent_name = get_agent(agent_id, settings).display_name

    recent_hits = memory.list_recent_dreams(thread_id=thread_id, limit=lookback)
    dreams_section = format_recent_dreams_markdown(
        recent_hits,
        max_chars=history_cap if history_cap > 0 else 6000,
    )

    waking_cap = max(0, settings.echo_dream_waking_max_chars)
    bundle = build_agent_context(
        memory,
        thread_id=thread_id,
        agent_id=agent_id,
        exclude_stream_origins=frozenset({"dream"}),
        stream_max_chars=waking_cap if waking_cap > 0 else None,
        stream_mode="rumination",
    )
    waking_section = _format_echo_agent_context_markdown(bundle)

    parts = [
        f"## Dreams you recently wove for {agent_name}\n{dreams_section}",
        f"## Context from the house\n{waking_section}",
    ]
    return "\n\n".join(parts), len(recent_hits)
