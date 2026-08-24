"""One-time migration from Chroma persistent store to file memory."""

from __future__ import annotations

import logging
from pathlib import Path

from light_house.config import Settings
from light_house.memory.file_store import FileMemoryStore

logger = logging.getLogger(__name__)


def migrate_chroma_if_needed(settings: Settings, store: FileMemoryStore) -> int:
    """
    Import documents from legacy Chroma into the file store when the new store is empty.

    Returns the number of documents imported.
    """
    if store.count_all() > 0:
        return 0
    chroma_path = settings.chroma_path.resolve()
    if not chroma_path.is_dir():
        return 0
    try:
        import chromadb
        from chromadb.config import Settings as ChromaSettings
    except ImportError:
        logger.warning("chromadb not installed; skipping Chroma migration")
        return 0

    try:
        client = chromadb.PersistentClient(
            path=str(chroma_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        collection = client.get_or_create_collection(name="light_house_longterm")
        result = collection.get(include=["documents", "metadatas"])
    except Exception as exc:
        logger.warning("Chroma migration skipped (could not read %s): %s", chroma_path, exc)
        return 0

    ids = result.get("ids") or []
    docs = result.get("documents") or []
    metas = result.get("metadatas") or []
    if not ids:
        return 0

    count = 0
    for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
        if not doc_id or doc is None or not isinstance(meta, dict):
            continue
        store.import_document(doc_id=str(doc_id), body=str(doc), metadata=dict(meta))
        count += 1

    if count:
        logger.info(
            "Migrated %d memory document(s) from Chroma (%s) to file store (%s)",
            count,
            chroma_path,
            settings.memory_store_path,
        )
    return count
