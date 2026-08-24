"""Retention ranking tests."""

import time

from light_house.memory.retention import retention_from_metadata, retention_score


def test_low_impact_fades_before_high():
    now = time.time()
    low = retention_score(impact=2.0, coherence=2.0, ts=now - 86400 * 30, now=now)
    high = retention_score(impact=9.0, coherence=8.0, ts=now - 86400 * 30, now=now)
    assert high > low


def test_unscored_metadata_uses_neutral_default():
    now = time.time()
    score = retention_from_metadata({"impact_score": -1, "coherence_score": -1, "ts": now}, now=now)
    assert score > 0


def test_fade_level_reduces_retention():
    now = time.time()
    crisp = retention_from_metadata(
        {"impact_score": 5, "coherence_score": 5, "ts": now, "fade_level": 0},
        now=now,
    )
    faded = retention_from_metadata(
        {"impact_score": 5, "coherence_score": 5, "ts": now, "fade_level": 3},
        now=now,
    )
    assert crisp > faded
