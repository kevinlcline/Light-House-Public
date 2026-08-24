"""Shared deduplication helpers for memory retrieval and short-term buffering."""

from __future__ import annotations


def normalize_snippet(text: str, *, max_len: int = 800) -> str:
    """Normalize text for overlap checks (case/whitespace insensitive)."""
    return " ".join(text.lower().split())[:max_len]


def dedupe_lines(lines: list[str]) -> list[str]:
    """Drop later lines whose normalized form was already seen."""
    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        key = normalize_snippet(line)
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(line)
    return out


def is_near_duplicate_text(a: str, b: str, *, min_len: int = 8) -> bool:
    """Cheap near-duplicate check without embeddings (normalized equality or prefix)."""
    na = normalize_snippet(a)
    nb = normalize_snippet(b)
    if len(na) < min_len or len(nb) < min_len:
        return na == nb and bool(na)
    if na == nb:
        return True
    shorter, longer = (na, nb) if len(na) <= len(nb) else (nb, na)
    return longer.startswith(shorter) and len(shorter) / len(longer) > 0.92


def dedupe_assistant_messages(messages: list) -> list:
    """Drop earlier assistant echoes; keep the newest occurrence of each reply."""
    if not messages:
        return messages
    seen: set[str] = set()
    kept_rev: list = []
    for msg in reversed(messages):
        role = getattr(msg, "role", None)
        content = getattr(msg, "content", "")
        if role == "assistant" and isinstance(content, str):
            key = normalize_snippet(content)
            if key in seen:
                continue
            if any(
                is_near_duplicate_text(content, prior.content)
                for prior in kept_rev
                if getattr(prior, "role", None) == "assistant"
            ):
                continue
            seen.add(key)
        kept_rev.append(msg)
    kept_rev.reverse()
    return kept_rev
