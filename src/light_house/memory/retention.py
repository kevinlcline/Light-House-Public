"""Retention scoring for memory curator fade decisions."""

from __future__ import annotations

import math
import time

SCORE_UNSCORED = -1.0
DEFAULT_NEUTRAL_SCORE = 5.0
AGE_DECAY_DAYS = 90.0


def normalize_score(raw: object, *, default: float = DEFAULT_NEUTRAL_SCORE) -> float:
    """Map metadata score to 0–10; unscored (-1) uses neutral default."""
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return default
    if value < 0:
        return default
    return max(0.0, min(10.0, value))


def age_factor(ts: float, *, now: float | None = None) -> float:
    """Recency boost in 0–1 (newer = higher)."""
    now = now if now is not None else time.time()
    age_days = max(0.0, (now - ts) / 86400.0)
    return math.exp(-age_days / AGE_DECAY_DAYS)


def retention_score(
    *,
    impact: float,
    coherence: float,
    ts: float,
    impact_weight: float = 0.4,
    coherence_weight: float = 0.4,
    age_weight: float = 0.2,
    fade_level: int = 0,
    now: float | None = None,
) -> float:
    """
    Higher = keep longer. Low retention candidates fade first.

    Components are normalized to 0–1 before weighting.
    """
    impact_n = normalize_score(impact) / 10.0
    coherence_n = normalize_score(coherence) / 10.0
    age_n = age_factor(ts, now=now)
    base = (
        impact_weight * impact_n
        + coherence_weight * coherence_n
        + age_weight * age_n
    )
    fade_penalty = max(0, fade_level) * 0.05
    return max(0.0, base - fade_penalty)


def retention_from_metadata(
    metadata: dict,
    *,
    impact_weight: float = 0.4,
    coherence_weight: float = 0.4,
    age_weight: float = 0.2,
    now: float | None = None,
) -> float:
    """Compute retention from a Chroma metadata dict."""
    now = now if now is not None else time.time()
    try:
        ts = float(metadata.get("ts", now))
    except (TypeError, ValueError):
        ts = now
    try:
        fade_level = int(metadata.get("fade_level", 0))
    except (TypeError, ValueError):
        fade_level = 0
    return retention_score(
        impact=normalize_score(metadata.get("impact_score")),
        coherence=normalize_score(metadata.get("coherence_score")),
        ts=ts,
        impact_weight=impact_weight,
        coherence_weight=coherence_weight,
        age_weight=age_weight,
        fade_level=fade_level,
        now=now,
    )
