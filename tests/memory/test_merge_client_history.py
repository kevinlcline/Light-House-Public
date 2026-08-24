"""Server-authoritative chat history merge."""

from __future__ import annotations

import tempfile
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage

from light_house.config import Settings
from light_house.memory.models import HistoryMessage
from light_house.memory.service import MemoryService


def _service(tmp_path: Path) -> MemoryService:
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAL_DB_ENABLED=False,
    )
    return MemoryService(settings)


def test_merge_uses_server_buffer_not_client_history():
    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(Path(tmp))
        thread_id = "kevin-home"
        svc._short_term.append_exchange(
            thread_id,
            user_text="phone user",
            assistant_text="phone reply",
        )
        client_history = [
            HistoryMessage(role="user", content="stale pc user"),
            HistoryMessage(role="assistant", content="stale pc reply"),
        ]
        messages = svc.merge_client_history(
            thread_id=thread_id,
            client_history=client_history,
            latest_user="new question",
        )
        assert len(messages) == 3
        assert isinstance(messages[0], HumanMessage)
        assert messages[0].content == "phone user"
        assert isinstance(messages[1], AIMessage)
        assert messages[1].content == "phone reply"
        assert isinstance(messages[2], HumanMessage)
        assert messages[2].content == "new question"


def test_merge_seeds_empty_buffer_from_client_once():
    with tempfile.TemporaryDirectory() as tmp:
        svc = _service(Path(tmp))
        thread_id = "kevin-home"
        client_history = [
            HistoryMessage(role="user", content="legacy user"),
            HistoryMessage(role="assistant", content="legacy assistant"),
        ]
        messages = svc.merge_client_history(
            thread_id=thread_id,
            client_history=client_history,
            latest_user="follow up",
        )
        assert len(messages) == 3
        assert messages[0].content == "legacy user"
        assert messages[1].content == "legacy assistant"
        assert messages[2].content == "follow up"

        buffered = svc.load_thread_chat_history(thread_id)
        assert len(buffered) == 2
        assert buffered[0].content == "legacy user"

        # Second call: client history ignored; server buffer wins
        stale = [HistoryMessage(role="user", content="wrong")]
        messages2 = svc.merge_client_history(
            thread_id=thread_id,
            client_history=stale,
            latest_user="another",
        )
        assert len(messages2) == 3
        assert messages2[0].content == "legacy user"
        assert messages2[1].content == "legacy assistant"
        assert messages2[2].content == "another"
