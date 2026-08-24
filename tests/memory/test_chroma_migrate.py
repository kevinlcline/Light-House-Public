"""Chroma → file store migration tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

from light_house.config import Settings
from light_house.memory.chroma_migrate import migrate_chroma_if_needed
from light_house.memory.file_store import FileMemoryStore


def test_migration_skips_when_file_store_nonempty(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    store.import_document(
        doc_id="existing",
        body="already here",
        metadata={"thread_id": "t", "memory_kind": "turn", "dedup_key": "x", "ts": 1.0},
    )
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        CHROMA_PATH=str(tmp_path / "chroma"),
    )
    (tmp_path / "chroma").mkdir()
    assert migrate_chroma_if_needed(settings, store) == 0
    assert store.count_all() == 1


def test_migration_skips_when_no_chroma_dir(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        CHROMA_PATH=str(tmp_path / "chroma"),
    )
    assert migrate_chroma_if_needed(settings, store) == 0


def test_migration_imports_chroma_documents(tmp_path: Path) -> None:
    store = FileMemoryStore(tmp_path / "memory")
    settings = Settings(
        _env_file=None,
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        CHROMA_PATH=str(tmp_path / "chroma"),
    )
    (tmp_path / "chroma").mkdir()

    mock_collection = MagicMock()
    mock_collection.get.return_value = {
        "ids": ["doc-1", "doc-2"],
        "documents": ["first body", "second body"],
        "metadatas": [
            {"thread_id": "t1", "memory_kind": "turn", "dedup_key": "a", "ts": 1.0},
            {"thread_id": "t1", "memory_kind": "stream", "dedup_key": "b", "ts": 2.0},
        ],
    }
    mock_client = MagicMock()
    mock_client.get_or_create_collection.return_value = mock_collection

    with patch("chromadb.PersistentClient", return_value=mock_client):
        count = migrate_chroma_if_needed(settings, store)

    assert count == 2
    assert store.count_all() == 2
