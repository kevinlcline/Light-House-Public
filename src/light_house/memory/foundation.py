"""Chunk and seed foundation context into global pinned long-term memory."""

from __future__ import annotations

import hashlib
import logging

from light_house.memory.file_store import FileMemoryStore

logger = logging.getLogger(__name__)

FOUNDATION_SEED_VERSION = "v1"


def chunk_foundation(text: str, *, max_chars: int) -> list[str]:
    """Split long foundation text into embeddable pin chunks."""
    body = text.strip()
    if not body:
        return []
    if len(body) <= max_chars:
        return [body]

    chunks: list[str] = []
    start = 0
    while start < len(body):
        end = min(start + max_chars, len(body))
        if end < len(body):
            break_at = body.rfind("\n\n", start, end)
            if break_at > start + max_chars // 2:
                end = break_at
        piece = body[start:end].strip()
        if piece:
            chunks.append(piece)
        start = end if end > start else start + max_chars
    return chunks


def seed_foundation_pins(
    store: FileMemoryStore,
    chunks: list[str],
    *,
    seed_version: str = FOUNDATION_SEED_VERSION,
) -> int:
    """Idempotently pin each chunk globally (updates existing seed rows by seed id)."""
    count = 0
    for index, chunk in enumerate(chunks):
        seed_id = f"{seed_version}-chunk-{index}"
        store.pin_sacred_fact(
            text=chunk,
            thread_id="",
            scope="global",
            extra_metadata={
                "foundation_seed_id": seed_id,
                "chunk_index": index,
                "seed_version": seed_version,
            },
        )
        count += 1
    logger.info("Seeded %d foundation pin(s) (version=%s)", count, seed_version)
    return count


def foundation_dedup_key(seed_id: str) -> str:
    """Stable dedup key for foundation pins (independent of chunk body edits)."""
    return hashlib.sha256(seed_id.encode("utf-8")).hexdigest()
