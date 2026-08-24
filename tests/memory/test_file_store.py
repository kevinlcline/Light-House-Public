"""FileMemoryStore unit tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.memory.constants import (
    GLOBAL_THREAD_ID,
    MEMORY_KIND_REFLECTION,
    MEMORY_KIND_TURN,
    MEMORY_TAG_PRIVATE_RUMINATION,
    PINNED_TRUE,
    PRIVATE_TRUE,
    STREAM_SOURCE_CHAT,
    STREAM_SOURCE_THOUGHT,
)
from light_house.memory.file_store import FileMemoryStore


@pytest.fixture
def store(tmp_path: Path) -> FileMemoryStore:
    return FileMemoryStore(tmp_path / "memory")


def test_remember_turn_dedup_by_content(store: FileMemoryStore) -> None:
    first = store.remember_turn(
        thread_id="t1",
        user_text="hello",
        assistant_text="hi there",
    )
    second = store.remember_turn(
        thread_id="t1",
        user_text="hello",
        assistant_text="hi there",
    )
    assert first == second
    assert store.count_turns(thread_id="t1") == 1


def test_pin_sacred_fact_global_and_thread(store: FileMemoryStore) -> None:
    store.pin_sacred_fact(text="Always be kind.", thread_id="", scope="global")
    store.pin_sacred_fact(text="Thread fact.", thread_id="t1", scope="thread")
    global_facts = store.list_pinned_facts(thread_id=GLOBAL_THREAD_ID, limit=10)
    thread_facts = store.list_pinned_facts(thread_id="t1", limit=10)
    assert any("Always be kind." in hit.text for hit in global_facts)
    assert any("Thread fact." in hit.text for hit in thread_facts)


def test_list_thread_corpus_orders_by_ts(store: FileMemoryStore) -> None:
    store.remember_stream_event(
        thread_id="t1",
        text="older thought",
        stream_source=STREAM_SOURCE_THOUGHT,
        extra_metadata={"ts": 100.0},
    )
    store.remember_stream_event(
        thread_id="t1",
        text="newer chat",
        stream_source=STREAM_SOURCE_CHAT,
        extra_metadata={"ts": 200.0},
    )
    corpus = store.list_thread_corpus(thread_id="t1")
    assert len(corpus) >= 2
    assert corpus[-1].metadata.get("ts", 0) >= corpus[0].metadata.get("ts", 0)


def test_import_document_and_count(store: FileMemoryStore) -> None:
    store.import_document(
        doc_id="legacy-1",
        body="migrated body",
        metadata={
            "thread_id": GLOBAL_THREAD_ID,
            "memory_kind": MEMORY_KIND_TURN,
            "pinned": PINNED_TRUE,
            "dedup_key": "abc",
            "ts": 1.0,
            "stream_source": STREAM_SOURCE_CHAT,
        },
    )
    assert store.count_all() == 1


def test_list_thread_corpus_excludes_migrated_legacy_reflections(store: FileMemoryStore) -> None:
    """Migrated reflection rows must not duplicate stream copies in context."""
    store.import_document(
        doc_id="legacy-ref",
        body="thought: Quiet evening reflection.",
        metadata={
            "thread_id": "t1",
            "memory_kind": MEMORY_KIND_REFLECTION,
            "memory_tag": MEMORY_TAG_PRIVATE_RUMINATION,
            "private": PRIVATE_TRUE,
            "migrated_to_stream": "true",
            "ts": 100.0,
        },
    )
    store.remember_stream_event(
        thread_id="t1",
        text="thought: Quiet evening reflection.",
        stream_source=STREAM_SOURCE_THOUGHT,
        extra_metadata={"ts": 100.0, "legacy_reflection_id": "legacy-ref"},
    )
    corpus = store.list_thread_corpus(thread_id="t1")
    assert len(corpus) == 1
    assert corpus[0].metadata.get("memory_kind") == "stream"


def test_concurrent_reads_do_not_raise(store: FileMemoryStore) -> None:
    """Group chat loads context for multiple lights in parallel threads."""
    import threading

    store.pin_sacred_fact(text="House rule.", thread_id="t1", scope="thread")
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            for _ in range(20):
                store.list_pinned_facts(thread_id="t1", limit=5)
                store.list_thread_corpus(thread_id="t1")
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(4)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert not errors
