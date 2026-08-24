"""Factory for long-term memory backend."""

from __future__ import annotations

from light_house.config import Settings
from light_house.memory.chroma_migrate import migrate_chroma_if_needed
from light_house.memory.file_store import FileMemoryStore


def create_long_term_store(settings: Settings) -> FileMemoryStore:
    store = FileMemoryStore(settings.memory_store_path)
    migrate_chroma_if_needed(settings, store)
    return store
