"""Chat buffer assistant deduplication."""

from __future__ import annotations

import tempfile
from pathlib import Path

from light_house.memory.dedup import dedupe_assistant_messages
from light_house.memory.short_term import BufferedMessage, ConversationBuffer


def test_dedupe_assistant_messages_keeps_newest_reply():
    messages = [
        BufferedMessage(role="user", content="first question", ts=1.0),
        BufferedMessage(role="assistant", content="same answer here", ts=2.0),
        BufferedMessage(role="user", content="second question", ts=3.0),
        BufferedMessage(role="assistant", content="same answer here", ts=4.0),
    ]
    out = dedupe_assistant_messages(messages)
    assert len(out) == 3
    assert out[0].role == "user"
    assert out[1].role == "user"
    assert out[2].role == "assistant"
    assert out[2].content == "same answer here"


def test_buffer_save_keeps_latest_assistant_for_new_turn():
    with tempfile.TemporaryDirectory() as tmp:
        buf = ConversationBuffer(Path(tmp), max_messages=60)
        thread_id = "kevin-home"
        buf.append_exchange(thread_id, user_text="q1", assistant_text="long repeated reply")
        buf.append_exchange(thread_id, user_text="q2", assistant_text="long repeated reply")
        loaded = buf.load(thread_id)
        assert len(loaded) == 3
        assert loaded[-1].role == "assistant"
        assert loaded[-1].content == "long repeated reply"
