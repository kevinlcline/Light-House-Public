"""Optional Chroma sidecar for semantic recall over the conscious stream."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from light_house.memory.constants import CORPUS_MEMORY_KINDS, GLOBAL_THREAD_ID
from light_house.memory.file_store import FileMemoryStore
from light_house.memory.models import MemoryHit

logger = logging.getLogger(__name__)

_INDEXABLE_KINDS = frozenset(CORPUS_MEMORY_KINDS)


class SemanticRecallIndex:
    """Semantic search layer synced from FileMemoryStore corpus documents."""

    def __init__(self, persist_path: Path, *, collection_name: str = "light_house_recall") -> None:
        persist_path.mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=str(persist_path),
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    @staticmethod
    def _scope_where(thread_id: str) -> dict[str, Any]:
        return {"$or": [{"thread_id": thread_id}, {"thread_id": GLOBAL_THREAD_ID}]}

    def upsert(self, *, doc_id: str, text: str, metadata: dict[str, Any]) -> None:
        kind = str(metadata.get("memory_kind", ""))
        if kind not in _INDEXABLE_KINDS:
            return
        body = text.strip()
        if not body:
            return
        meta = {
            "thread_id": str(metadata.get("thread_id", "")),
            "memory_kind": kind,
            "ts": float(metadata.get("ts", 0.0)),
        }
        source = metadata.get("stream_source")
        if isinstance(source, str) and source:
            meta["stream_source"] = source
        self._collection.upsert(ids=[doc_id], documents=[body], metadatas=[meta])

    def delete(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        try:
            self._collection.delete(ids=doc_ids)
        except Exception:
            logger.debug("Recall index delete skipped for %d ids", len(doc_ids))

    def search(
        self,
        query: str,
        *,
        thread_id: str,
        k: int = 10,
        stream_source: str | None = None,
        since_ts: float | None = None,
    ) -> list[MemoryHit]:
        q = query.strip()
        if not q:
            return []
        cap = max(1, min(30, k))
        where: dict[str, Any] = self._scope_where(thread_id)
        if stream_source:
            where = {
                "$and": [
                    self._scope_where(thread_id),
                    {"stream_source": stream_source},
                ]
            }
        result = self._collection.query(
            query_texts=[q],
            n_results=cap,
            where=where,
            include=["documents", "distances", "metadatas"],
        )
        hits: list[MemoryHit] = []
        docs = (result.get("documents") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or []
        metas = (result.get("metadatas") or [[]])[0] or []
        ids = (result.get("ids") or [[]])[0] or []
        for doc_id, doc, dist, meta in zip(ids, docs, dists, metas, strict=False):
            m = dict(meta or {})
            if since_ts is not None:
                try:
                    if float(m.get("ts", 0.0)) < since_ts:
                        continue
                except (TypeError, ValueError):
                    pass
            score = float(dist) if dist is not None else None
            hits.append(
                MemoryHit(text=str(doc), score=score, metadata=m, doc_id=str(doc_id))
            )
        return hits

    def backfill_thread(self, store: FileMemoryStore, *, thread_id: str) -> int:
        """Index all corpus docs for a thread (idempotent upsert)."""
        count = 0
        for hit in store.list_thread_corpus(thread_id=thread_id):
            if not hit.doc_id:
                continue
            self.upsert(doc_id=hit.doc_id, text=hit.text, metadata=hit.metadata)
            count += 1
        return count
