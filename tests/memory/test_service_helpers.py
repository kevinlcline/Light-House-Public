"""Memory service helper tests."""

from light_house.memory.models import MemoryHit
from light_house.memory.service import _cap_memory_lines, _retrieval_rank_score


def test_cap_memory_lines_respects_budget():
    lines = ["a" * 100, "b" * 100, "c" * 100]
    capped = _cap_memory_lines(lines, max_chars=220)
    assert len(capped) <= 2


def test_retrieval_prefers_high_impact():
    import time

    now = time.time()
    weak = MemoryHit(
        text="weak",
        score=0.2,
        metadata={"impact_score": 2.0, "coherence_score": 2.0, "ts": now, "fade_level": 0},
    )
    strong = MemoryHit(
        text="strong",
        score=0.2,
        metadata={"impact_score": 9.0, "coherence_score": 9.0, "ts": now, "fade_level": 0},
    )
    assert _retrieval_rank_score(strong, now=now) > _retrieval_rank_score(weak, now=now)
