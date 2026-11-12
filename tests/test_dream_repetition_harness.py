"""Dream-repetition harness.

Each test resolves its own file before it asserts anything.
The speaking-arc seed stays in the arc tests, not here.
"""

from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from light_house.config import Settings
from light_house.memory.models import MemoryHit

REPO = Path(__file__).resolve().parents[1]
SUB = REPO / "src" / "light_house" / "subconscious"
ARCHIVE = REPO / "data" / "memory" / "memory.db"

CHOICE = "I take the quieter path along the north stair."
BEAT = (
    "The attic holds one lamp and a shut window, and the dust in the light does not fall."
)
UNFROZEN = (
    "You are Lumen. This choice prompt is unfrozen. "
    "Say the step the night actually gives you, in your own words."
)
ANCHOR_KEYS = frozenset({"place_id", "place-id", "seed", "residue"})
_PLACE_TOKEN = re.compile(r"place[_-]id", re.IGNORECASE)
_WAKING_SPLIT = re.compile(r"\n\s*\[waking recall\]\s*", re.IGNORECASE)


def _require(name: str) -> Path:
    path = SUB / name
    assert path.is_file(), str(path)
    return path


def _night_body(text: str) -> str:
    body = (text or "").strip()
    if body.lower().startswith("dream:"):
        body = body[len("dream:") :].strip()
    return _WAKING_SPLIT.split(body, maxsplit=1)[0].strip()


def _open_archive() -> sqlite3.Connection:
    if not ARCHIVE.is_file():
        pytest.skip("house dream archive is not on this machine")
    con = sqlite3.connect(f"file:{ARCHIVE}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def _nodes():
    from light_house.subconscious import dream_nodes as dn

    settings = Settings(
        _env_file=None,
        ECHO_DREAM_INTERACTIVE_ENABLED=True,
        ECHO_DREAM_STORY_ROUNDS=3,
        PERSONAL_DB_ENABLED=False,
    )
    seen: list[list] = []

    class FakeClient:
        def invoke(self, messages):
            seen.append(list(messages))
            n = len(seen)
            if n == 1:
                return AIMessage(content=BEAT)
            if n == 2:
                return AIMessage(content=CHOICE)
            return AIMessage(content="The lamp leaned, and the window stayed shut.")

    stack = (
        patch.object(dn, "build_echo_dream_llm_chain", return_value=[("mock", FakeClient())]),
        patch.object(dn, "get_agent", return_value=MagicMock(display_name="Lumen")),
        patch.object(dn, "load_echo_persona", return_value="Echo"),
        patch.object(dn, "_light_choice_system", return_value=UNFROZEN),
    )
    return dn, settings, MagicMock(), seen, stack


def _blank_state(**extra) -> dict:
    state = {
        "thread_id": "lumen-home",
        "agent_id": "lumen",
        "context_text": "",
        "dream_text": "",
        "waking_recall": "",
        "felt_days": None,
        "dream_round": 0,
        "max_dream_rounds": 3,
        "story_beats": [],
        "light_choices": [],
        "current_beat": "",
        "current_choice": "",
    }
    state.update(extra)
    return state


def _human(messages: list) -> str:
    for message in messages:
        if isinstance(message, HumanMessage):
            return str(message.content)
    raise AssertionError("Echo was handed no human message")


def test_within_night_prompt_keeps_the_choice_out_of_prior_beats_after_unfreeze() -> None:
    """The prompt body, not the stored night.

    Unfreezing the choice prompt changes the words of the choice call.
    The choice still sits in its own section. Prior beats stay raw Echo text.
    """
    _require("dream_nodes.py")
    _require("dream_story.py")
    dn, settings, memory, seen, stack = _nodes()
    from light_house.subconscious.dream_story import assemble_dream_story

    with stack[0], stack[1], stack[2], stack[3]:
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        state = _blank_state()
        state = {**state, **nodes["echo_dream_beat"](state)}
        state = {**state, **nodes["light_choose_path"](state)}
        choice_system = str(seen[1][0].content)
        state = {**state, **nodes["echo_dream_beat"](state)}
        prompt = _human(seen[2])

    assert choice_system == UNFROZEN
    prior, section = prompt.split("## The Light's choice", 1)
    assert "## Prior beats" in prior
    assert CHOICE not in prior
    assert f"— {CHOICE}" not in prior
    assert CHOICE in section
    stored = assemble_dream_story(state["story_beats"], state["light_choices"])
    assert f"— {CHOICE}" in stored


def test_earlier_night_hinge_is_absent_from_the_prompt_after_unfreeze() -> None:
    """Night N is handed no welded choice from nights 1..N-1.

    The stored body still welds the choice. Lookback strips that line.
    Unfreezing the choice prompt does not put the line back. A sentence
    that only contains an em dash stays with the beat.
    """
    _require("dream_nodes.py")
    _require("dream_story.py")
    _require("echo_context.py")
    dn, settings, memory, seen, stack = _nodes()
    from light_house.subconscious.dream_story import assemble_dream_story
    from light_house.subconscious.echo_context import format_recent_dreams_markdown

    beat = "The attic holds one lamp — the dust in its light does not fall."
    stored = assemble_dream_story([beat], [CHOICE])
    hinge = f"— {CHOICE}"
    assert hinge in stored
    lookback = format_recent_dreams_markdown(
        [
            MemoryHit(
                text=f"dream: {stored}",
                score=0.0,
                metadata={
                    "stream_source": "dream",
                    "ts": 1_700_000_000.0,
                    "reflection_summary": "I woke remembering the attic lamp.",
                },
            )
        ],
        max_chars=6000,
    )
    assert hinge not in lookback
    assert beat in lookback
    assert "I woke remembering the attic lamp." in lookback

    with stack[0], stack[1], stack[2], stack[3]:
        assert dn._light_choice_system("Lumen") == UNFROZEN
        nodes = dn.build_echo_dream_nodes(settings=settings, memory=memory)
        nodes["echo_dream_beat"](
            _blank_state(
                context_text=f"## Dreams you recently wove for Lumen\n{lookback}",
            )
        )
        prompt = _human(seen[0])

    assert UNFROZEN not in prompt
    assert beat in prompt
    assert "Dreams you recently wove" in prompt
    assert hinge not in prompt
    assert CHOICE not in prompt


def test_canned_exact_body_is_counted_and_a_longer_night_is_not() -> None:
    _require("dream_story.py")
    from light_house.subconscious.dream_story import _CANNED_EMPTY_BEAT_RE

    canon = (
        "The night held still. A door stood ajar, light thin as breath beyond it. "
        "I waited, and waiting was enough."
    )
    assert _CANNED_EMPTY_BEAT_RE.fullmatch(_night_body(canon))
    assert _CANNED_EMPTY_BEAT_RE.fullmatch(_night_body(f"dream: {canon}"))
    recalled = f"dream: {canon}\n\n[waking recall] I woke remembering the door."
    assert _CANNED_EMPTY_BEAT_RE.fullmatch(_night_body(recalled))
    longer = f"{canon}\n\nThe lamp stayed where the night had set it down."
    assert _CANNED_EMPTY_BEAT_RE.search(longer)
    assert _CANNED_EMPTY_BEAT_RE.fullmatch(_night_body(longer)) is None


def test_archive_canned_bodies_are_exact_matches_only() -> None:
    """Historical count: stored nights whose body is only the canned sentence."""
    _require("dream_story.py")
    from light_house.subconscious.dream_story import _CANNED_EMPTY_BEAT_RE

    con = _open_archive()
    try:
        rows = con.execute(
            "SELECT body FROM memory_documents WHERE stream_source = 'dream'"
        ).fetchall()
    finally:
        con.close()
    assert rows
    exact = 0
    for row in rows:
        if _CANNED_EMPTY_BEAT_RE.fullmatch(_night_body(row["body"])):
            exact += 1
    assert exact == 0


def test_dream_state_has_no_place_anchor() -> None:
    path = _require("dream_state.py")
    from light_house.subconscious.dream_state import DreamState

    text = path.read_text(encoding="utf-8")
    assert "class DreamState" in text
    fields = set(DreamState.__annotations__)
    assert fields.isdisjoint(ANCHOR_KEYS)


def test_one_lights_recent_nights_carry_no_place_anchor() -> None:
    """Baseline absence. Passes now. Fails when a stored night grows an anchor."""
    _require("dream_state.py")
    con = _open_archive()
    try:
        thread = con.execute(
            """
            SELECT thread_id FROM memory_documents
            WHERE stream_source = 'dream'
            GROUP BY thread_id
            HAVING COUNT(*) >= 5
            ORDER BY thread_id
            LIMIT 1
            """
        ).fetchone()
        assert thread is not None
        rows = con.execute(
            """
            SELECT body, extra_json FROM memory_documents
            WHERE thread_id = ? AND stream_source = 'dream'
            ORDER BY ts DESC
            LIMIT 5
            """,
            (thread["thread_id"],),
        ).fetchall()
    finally:
        con.close()
    assert len(rows) == 5
    for row in rows:
        extra = json.loads(row["extra_json"] or "{}")
        assert set(extra).isdisjoint(ANCHOR_KEYS)
        assert _PLACE_TOKEN.search(row["body"] or "") is None
