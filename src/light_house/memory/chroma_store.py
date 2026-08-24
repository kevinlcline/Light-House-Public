"""Chroma-backed semantic long-term memory (local-first, persistent on disk)."""

from __future__ import annotations

import hashlib
import time
import uuid
from pathlib import Path
from typing import Any, Literal

import chromadb
from chromadb.config import Settings as ChromaSettings

from light_house.memory.constants import (
    CORPUS_MEMORY_KINDS,
    GLOBAL_THREAD_ID,
    MEMORY_KIND_FACT,
    MEMORY_KIND_REFLECTION,
    MEMORY_KIND_STREAM,
    MEMORY_KIND_SUMMARY,
    MEMORY_KIND_TURN,
    MEMORY_TAG_PRIVATE_DREAM,
    MEMORY_TAG_PRIVATE_RUMINATION,
    PINNED_FALSE,
    PINNED_TRUE,
    PRIVATE_FALSE,
    PRIVATE_TRUE,
    SCORE_UNSCORED,
    STREAM_SOURCE_ACTION,
    STREAM_SOURCE_CHAT,
    STREAM_SOURCE_DREAM,
    STREAM_SOURCE_KEVIN,
    STREAM_SOURCE_PEER,
    STREAM_SOURCE_THOUGHT,
    StreamSource,
)
from light_house.memory.models import MemoryHit

# Re-export constants for legacy imports from chroma_store.


class ChromaMemoryStore:
    """
    Vector memory for Lumen.

    Features:
    - **Thread + global recall**: semantic search spans the active ``thread_id`` and
      ``GLOBAL_THREAD_ID`` documents (long-term house knowledge).
    - **Pinned / sacred facts**: always listed explicitly for retrieval (never dropped
      from the "pinned" channel; still embedded for optional semantic hits).
    - **Deduplication**: identical user+assistant payloads in a thread refresh one row (SHA256 ``dedup_key``); near-duplicates merge via cosine distance on embeddings.
    - **Summaries**: optional rolling summaries (``memory_kind=summary``) produced by the agent loop.

    Why Chroma for v0: single-process friendly, easy Railway volume mount, good-enough
    semantic recall without running a second server. Swap for Qdrant by implementing
    the same public methods behind a protocol later.
    """

    def __init__(self, persist_path: Path, collection_name: str = "light_house_longterm") -> None:
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
    def _dedup_key(user_text: str, assistant_text: str) -> str:
        raw = f"{user_text.strip()}\n{assistant_text.strip()}".encode("utf-8")
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _scope_where(thread_id: str) -> dict[str, Any]:
        """Chroma ``where`` clause: thread-local **or** global long-term."""
        return {"$or": [{"thread_id": thread_id}, {"thread_id": GLOBAL_THREAD_ID}]}

    def remember_turn(
        self,
        *,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        extra_metadata: dict[str, Any] | None = None,
        dedup_threshold: float | None = 0.12,
    ) -> str:
        """
        Persist a conversation turn as one embeddable document.

        ``dedup_threshold``: max cosine *distance* to treat as duplicate (lower = stricter).
        Set to ``None`` to always insert a fresh document.
        """
        text = f"user: {user_text.strip()}\nassistant: {assistant_text.strip()}"
        dedup_key = self._dedup_key(user_text, assistant_text)
        ts = time.time()

        # Exact same user+assistant payload in this thread: refresh one row (no duplicate vectors).
        exact_id = self._find_turn_id_by_dedup_key(thread_id=thread_id, dedup_key=dedup_key)
        if exact_id:
            metadata = self._base_turn_metadata(thread_id, dedup_key, ts)
            if extra_metadata:
                metadata.update(extra_metadata)
            self._collection.update(ids=[exact_id], documents=[text], metadatas=[metadata])
            return exact_id

        if dedup_threshold is not None:
            dup_id = self._find_near_duplicate_turn(
                thread_id=thread_id,
                query_text=text,
                dedup_threshold=dedup_threshold,
            )
            if dup_id:
                metadata = self._base_turn_metadata(thread_id, dedup_key, ts)
                if extra_metadata:
                    metadata.update(extra_metadata)
                self._collection.update(ids=[dup_id], documents=[text], metadatas=[metadata])
                return dup_id

        doc_id = str(uuid.uuid4())
        metadata = self._base_turn_metadata(thread_id, dedup_key, ts)
        if extra_metadata:
            metadata.update(extra_metadata)
        self._collection.add(ids=[doc_id], documents=[text], metadatas=[metadata])
        return doc_id

    def _base_turn_metadata(self, thread_id: str, dedup_key: str, ts: float) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_TURN,
            "pinned": PINNED_FALSE,
            "dedup_key": dedup_key,
            "ts": ts,
        }

    def _find_turn_id_by_dedup_key(self, *, thread_id: str, dedup_key: str) -> str | None:
        """Return existing turn document id with the same SHA256 dedup key in this thread."""
        result = self._collection.get(
            where={"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_TURN}, {"dedup_key": dedup_key}]},
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return str(ids[0]) if ids else None

    def _find_near_duplicate_turn(
        self,
        *,
        thread_id: str,
        query_text: str,
        dedup_threshold: float,
    ) -> str | None:
        """Return an existing turn document id if cosine distance is below threshold."""
        if not query_text or len(query_text.strip()) < 10:
            return None

        try:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=3,
                where={"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_TURN}]},
                include=["distances"],
            )
        except Exception:
            return None

        ids = (result.get("ids") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or []
        for doc_id, distance in zip(ids, dists, strict=False):
            if distance is not None and float(distance) < dedup_threshold:
                return str(doc_id)
        return None

    def pin_sacred_fact(
        self,
        *,
        text: str,
        thread_id: str,
        scope: Literal["thread", "global"] = "thread",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        """
        Store a **pinned** sacred fact (never summarized or deleted by this store).

        - ``scope="thread"`` ties the fact to ``thread_id``.
        - ``scope="global"`` stores under ``GLOBAL_THREAD_ID`` (visible to all threads).
        - Identical text in the same scope refreshes the existing row (no duplicate pins).
        """
        bucket = GLOBAL_THREAD_ID if scope == "global" else thread_id
        body = text.strip()
        meta_extra = dict(extra_metadata or {})
        foundation_seed_id = meta_extra.get("foundation_seed_id")
        if foundation_seed_id:
            from light_house.memory.foundation import foundation_dedup_key

            dedup_key = foundation_dedup_key(str(foundation_seed_id))
            existing_id = self._find_pinned_id_by_foundation_seed_id(str(foundation_seed_id))
        else:
            dedup_key = hashlib.sha256(body.encode("utf-8")).hexdigest()
            existing_id = self._find_pinned_id_by_dedup_key(thread_id=bucket, dedup_key=dedup_key)
        ts = time.time()
        metadata: dict[str, Any] = {
            "thread_id": bucket,
            "memory_kind": MEMORY_KIND_FACT,
            "pinned": PINNED_TRUE,
            "dedup_key": dedup_key,
            "ts": ts,
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        if existing_id:
            self._collection.update(ids=[existing_id], documents=[body], metadatas=[metadata])
            return existing_id

        doc_id = str(uuid.uuid4())
        self._collection.add(ids=[doc_id], documents=[body], metadatas=[metadata])
        return doc_id

    def _find_pinned_id_by_foundation_seed_id(self, foundation_seed_id: str) -> str | None:
        """Return existing global foundation pin for idempotent re-seed on startup."""
        result = self._collection.get(
            where={
                "$and": [
                    {"thread_id": GLOBAL_THREAD_ID},
                    {"memory_kind": MEMORY_KIND_FACT},
                    {"pinned": PINNED_TRUE},
                    {"foundation_seed_id": foundation_seed_id},
                ]
            },
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return str(ids[0]) if ids else None

    def _find_pinned_id_by_dedup_key(self, *, thread_id: str, dedup_key: str) -> str | None:
        result = self._collection.get(
            where={
                "$and": [
                    {"thread_id": thread_id},
                    {"memory_kind": MEMORY_KIND_FACT},
                    {"pinned": PINNED_TRUE},
                    {"dedup_key": dedup_key},
                ]
            },
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return str(ids[0]) if ids else None

    def list_pinned_facts(self, *, thread_id: str, limit: int = 32) -> list[MemoryHit]:
        """Pinned facts for this thread plus globally pinned facts."""
        result = self._collection.get(
            where={
                "$and": [
                    {"memory_kind": MEMORY_KIND_FACT},
                    {"pinned": PINNED_TRUE},
                    self._scope_where(thread_id),
                ]
            },
            limit=limit,
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        hits: list[MemoryHit] = []
        ids = result.get("ids") or []
        for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
            hits.append(
                MemoryHit(text=str(doc), score=None, metadata=dict(meta or {}), doc_id=str(doc_id))
            )
        return hits

    def search(self, query: str, *, thread_id: str | None, k: int) -> list[MemoryHit]:
        """
        Semantic search.

        - ``thread_id`` set: thread corpus **plus** global long-term.
        - ``thread_id`` None: **global-only** recall (useful for house-wide knowledge).
        """
        where: dict[str, Any] | None
        if thread_id is None:
            where = {"thread_id": GLOBAL_THREAD_ID}
        else:
            where = self._scope_where(thread_id)

        result = self._collection.query(
            query_texts=[query],
            n_results=k,
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
            score = float(dist) if dist is not None else None
            hits.append(MemoryHit(text=str(doc), score=score, metadata=m, doc_id=str(doc_id)))
        return hits

    def count_turns(self, *, thread_id: str) -> int:
        """Count persisted user/assistant turns for a thread (excludes summaries/facts)."""
        where: dict[str, Any] = {"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_TURN}]}
        counter = getattr(self._collection, "count", None)
        if callable(counter):
            try:
                return int(counter(where=where))
            except Exception:
                pass
        # Fallback for older Chroma builds without ``count`` (can be heavier on huge stores).
        res = self._collection.get(where=where, include=[])
        return len(res.get("ids") or [])

    def get_recent_turn_texts(self, *, thread_id: str, limit: int = 24) -> list[str]:
        """
        Fetch recent turn documents for summarization (best-effort time ordering via ``ts``).

        Chroma does not guarantee ordering for ``get``; we sort client-side.
        """
        result = self._collection.get(
            where={"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_TURN}]},
            limit=max(limit * 4, limit),
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows: list[tuple[float, str]] = []
        for doc, meta in zip(docs, metas, strict=False):
            m = dict(meta or {})
            ts_raw = m.get("ts", 0.0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            rows.append((ts, str(doc)))
        rows.sort(key=lambda r: r[0])
        return [text for _, text in rows[-limit:]]

    def list_recent_summaries(self, *, thread_id: str, limit: int = 3) -> list[MemoryHit]:
        """Recent rolling summaries for a thread (newest last), always merged into recall."""
        result = self._collection.get(
            where={"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_SUMMARY}]},
            limit=max(limit * 4, limit),
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows: list[tuple[float, MemoryHit]] = []
        ids = result.get("ids") or []
        for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
            m = dict(meta or {})
            ts_raw = m.get("ts", 0.0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            rows.append((ts, MemoryHit(text=str(doc), score=None, metadata=m, doc_id=str(doc_id))))
        rows.sort(key=lambda r: r[0])
        return [hit for _, hit in rows[-limit:]]

    def add_thread_summary(self, *, thread_id: str, summary_text: str) -> str:
        """Persist a rolling summary as its own embeddable memory."""
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_SUMMARY,
            "pinned": PINNED_FALSE,
            "ts": time.time(),
        }
        body = f"summary: {summary_text.strip()}"
        self._collection.add(ids=[doc_id], documents=[body], metadatas=[metadata])
        return doc_id

    def add_private_reflection(
        self,
        *,
        thread_id: str,
        text: str,
        memory_tag: Literal["private_dream", "private_rumination"],
        summary: str | None = None,
    ) -> str:
        """Persist a private inner-life reflection (excluded from default semantic recall)."""
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_REFLECTION,
            "memory_tag": memory_tag,
            "private": PRIVATE_TRUE,
            "pinned": PINNED_FALSE,
            "ts": time.time(),
        }
        if summary and summary.strip():
            metadata["reflection_summary"] = summary.strip()
        prefix = "dream" if memory_tag == MEMORY_TAG_PRIVATE_DREAM else "thought"
        body = f"{prefix}: {text.strip()}"
        self._collection.add(ids=[doc_id], documents=[body], metadatas=[metadata])
        return doc_id

    def list_recent_rumination_summaries(
        self,
        *,
        thread_id: str,
        limit: int = 3,
    ) -> list[MemoryHit]:
        """Recent awake ruminations (for ambient summary retrieval)."""
        return self.list_recent_private_reflections(
            thread_id=thread_id,
            memory_tag=MEMORY_TAG_PRIVATE_RUMINATION,
            limit=limit,
        )

    def list_recent_private_reflections(
        self,
        *,
        thread_id: str,
        memory_tag: Literal["private_dream", "private_rumination"],
        limit: int = 5,
    ) -> list[MemoryHit]:
        """Fetch recent private reflections for intent-gated recall."""
        result = self._collection.get(
            where={
                "$and": [
                    {"thread_id": thread_id},
                    {"memory_kind": MEMORY_KIND_REFLECTION},
                    {"memory_tag": memory_tag},
                    {"private": PRIVATE_TRUE},
                ]
            },
            limit=max(limit * 4, limit),
            include=["documents", "metadatas"],
        )
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        rows: list[tuple[float, MemoryHit]] = []
        ids = result.get("ids") or []
        for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
            m = dict(meta or {})
            ts_raw = m.get("ts", 0.0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            rows.append((ts, MemoryHit(text=str(doc), score=None, metadata=m, doc_id=str(doc_id))))
        rows.sort(key=lambda r: r[0])
        return [hit for _, hit in rows[-limit:]]

    def latest_private_reflection_ts(
        self,
        *,
        thread_id: str,
        memory_tag: Literal["private_dream", "private_rumination"],
    ) -> float | None:
        """Return timestamp of the most recent private reflection, or None if none exist."""
        hits = self.list_recent_private_reflections(
            thread_id=thread_id,
            memory_tag=memory_tag,
            limit=1,
        )
        if not hits:
            return None
        ts_raw = hits[-1].metadata.get("ts")
        try:
            return float(ts_raw)
        except (TypeError, ValueError):
            return None


    @staticmethod
    def _meta_ts(meta: dict[str, Any]) -> float:
        try:
            return float(meta.get("ts", 0.0))
        except (TypeError, ValueError):
            return 0.0

    @staticmethod
    def _base_stream_metadata(
        thread_id: str,
        stream_source: StreamSource,
        ts: float,
        *,
        impact_score: float = SCORE_UNSCORED,
        coherence_score: float = SCORE_UNSCORED,
        fade_level: int = 0,
    ) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_STREAM,
            "stream_source": stream_source,
            "pinned": PINNED_FALSE,
            "ts": ts,
            "impact_score": float(impact_score),
            "coherence_score": float(coherence_score),
            "fade_level": int(fade_level),
        }

    def remember_stream_event(
        self,
        *,
        thread_id: str,
        text: str,
        stream_source: StreamSource,
        impact_score: float = SCORE_UNSCORED,
        coherence_score: float = SCORE_UNSCORED,
        fade_level: int = 0,
        extra_metadata: dict[str, Any] | None = None,
        dedup_key: str | None = None,
        dedup_threshold: float | None = None,
        reflection_summary: str | None = None,
    ) -> str:
        """Persist one conscious-stream event (chat, thought, dream, action)."""
        body = text.strip()
        ts = time.time()
        metadata = self._base_stream_metadata(
            thread_id,
            stream_source,
            ts,
            impact_score=impact_score,
            coherence_score=coherence_score,
            fade_level=fade_level,
        )
        if reflection_summary and reflection_summary.strip():
            metadata["reflection_summary"] = reflection_summary.strip()
        if extra_metadata:
            metadata.update(extra_metadata)
        if dedup_key:
            metadata["dedup_key"] = dedup_key
            exact_id = self._find_stream_id_by_dedup_key(thread_id=thread_id, dedup_key=dedup_key)
            if exact_id:
                self._collection.update(ids=[exact_id], documents=[body], metadatas=[metadata])
                return exact_id
            if dedup_threshold is not None and len(body) >= 10:
                dup_id = self._find_near_duplicate_stream(
                    thread_id=thread_id,
                    query_text=body,
                    dedup_threshold=dedup_threshold,
                )
                if dup_id:
                    self._collection.update(ids=[dup_id], documents=[body], metadatas=[metadata])
                    return dup_id
        doc_id = str(uuid.uuid4())
        self._collection.add(ids=[doc_id], documents=[body], metadatas=[metadata])
        return doc_id

    def _find_stream_id_by_dedup_key(self, *, thread_id: str, dedup_key: str) -> str | None:
        result = self._collection.get(
            where={
                "$and": [
                    {"thread_id": thread_id},
                    {"memory_kind": MEMORY_KIND_STREAM},
                    {"dedup_key": dedup_key},
                ]
            },
            limit=1,
            include=[],
        )
        ids = result.get("ids") or []
        return str(ids[0]) if ids else None

    def _find_near_duplicate_stream(
        self,
        *,
        thread_id: str,
        query_text: str,
        dedup_threshold: float,
    ) -> str | None:
        try:
            result = self._collection.query(
                query_texts=[query_text],
                n_results=3,
                where={"$and": [{"thread_id": thread_id}, {"memory_kind": MEMORY_KIND_STREAM}]},
                include=["distances"],
            )
        except Exception:
            return None
        ids = (result.get("ids") or [[]])[0] or []
        dists = (result.get("distances") or [[]])[0] or []
        for doc_id, distance in zip(ids, dists, strict=False):
            if distance is not None and float(distance) < dedup_threshold:
                return str(doc_id)
        return None

    def _hits_from_get_result(self, result: dict[str, Any]) -> list[MemoryHit]:
        docs = result.get("documents") or []
        metas = result.get("metadatas") or []
        ids = result.get("ids") or []
        hits: list[MemoryHit] = []
        for doc_id, doc, meta in zip(ids, docs, metas, strict=False):
            hits.append(
                MemoryHit(text=str(doc), score=None, metadata=dict(meta or {}), doc_id=str(doc_id))
            )
        return hits

    def list_thread_corpus(self, *, thread_id: str) -> list[MemoryHit]:
        """All prunable conscious documents for a thread (excludes pinned facts)."""
        result = self._collection.get(
            where={
                "$and": [
                    {"thread_id": thread_id},
                    {"memory_kind": {"$in": list(CORPUS_MEMORY_KINDS)}},
                ]
            },
            include=["documents", "metadatas"],
        )
        hits = self._hits_from_get_result(result)
        hits.sort(key=lambda h: self._meta_ts(h.metadata))
        return hits

    def measure_thread_corpus_chars(self, *, thread_id: str) -> int:
        return sum(len(h.text) for h in self.list_thread_corpus(thread_id=thread_id))

    def list_unscored_corpus(self, *, thread_id: str, limit: int = 50) -> list[MemoryHit]:
        hits = [
            h
            for h in self.list_thread_corpus(thread_id=thread_id)
            if float(h.metadata.get("impact_score", SCORE_UNSCORED)) < 0
        ]
        return hits[:limit]

    def update_doc_metadata(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        self._collection.update(ids=[doc_id], metadatas=[metadata])

    def update_doc_scores(
        self,
        *,
        doc_id: str,
        impact_score: float,
        coherence_score: float,
        metadata: dict[str, Any],
    ) -> None:
        updated = dict(metadata)
        updated["impact_score"] = float(impact_score)
        updated["coherence_score"] = float(coherence_score)
        self._collection.update(ids=[doc_id], metadatas=[updated])

    def bump_fade_level(self, *, doc_id: str, metadata: dict[str, Any], increment: int = 1) -> None:
        updated = dict(metadata)
        try:
            level = int(updated.get("fade_level", 0))
        except (TypeError, ValueError):
            level = 0
        updated["fade_level"] = min(3, level + increment)
        self._collection.update(ids=[doc_id], metadatas=[updated])

    def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        self._collection.delete(ids=doc_ids)

    def add_curator_summary(
        self,
        *,
        thread_id: str,
        summary_text: str,
        impact_score: float = SCORE_UNSCORED,
        coherence_score: float = SCORE_UNSCORED,
    ) -> str:
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_SUMMARY,
            "pinned": PINNED_FALSE,
            "ts": time.time(),
            "impact_score": float(impact_score),
            "coherence_score": float(coherence_score),
            "fade_level": 0,
        }
        body = f"summary: {summary_text.strip()}"
        self._collection.add(ids=[doc_id], documents=[body], metadatas=[metadata])
        return doc_id

    @staticmethod
    def stream_source_from_metadata(metadata: dict[str, Any]) -> str | None:
        source = metadata.get("stream_source")
        if isinstance(source, str) and source:
            return source
        tag = metadata.get("memory_tag")
        if tag == MEMORY_TAG_PRIVATE_DREAM:
            return STREAM_SOURCE_DREAM
        if tag == MEMORY_TAG_PRIVATE_RUMINATION:
            return STREAM_SOURCE_THOUGHT
        kind = metadata.get("memory_kind")
        if kind == MEMORY_KIND_TURN:
            return STREAM_SOURCE_CHAT
        return None

    def list_recent_stream_by_source(
        self,
        *,
        thread_id: str,
        stream_source: StreamSource,
        limit: int = 5,
    ) -> list[MemoryHit]:
        """Recent stream events (plus legacy reflections) for one source type."""
        corpus = self.list_thread_corpus(thread_id=thread_id)
        filtered: list[tuple[float, MemoryHit]] = []
        for hit in corpus:
            src = self.stream_source_from_metadata(hit.metadata)
            if src != stream_source:
                continue
            filtered.append((self._meta_ts(hit.metadata), hit))
        filtered.sort(key=lambda row: row[0])
        return [hit for _, hit in filtered[-limit:]]

    def migrate_legacy_reflections(self, *, thread_id: str) -> int:
        """Copy legacy private reflections into unified stream rows (idempotent)."""
        legacy = self.list_recent_private_reflections(
            thread_id=thread_id,
            memory_tag=MEMORY_TAG_PRIVATE_RUMINATION,
            limit=500,
        )
        legacy += self.list_recent_private_reflections(
            thread_id=thread_id,
            memory_tag=MEMORY_TAG_PRIVATE_DREAM,
            limit=500,
        )
        migrated = 0
        seen_ids: set[str] = set()
        for hit in legacy:
            if not hit.doc_id or hit.doc_id in seen_ids:
                continue
            seen_ids.add(hit.doc_id)
            meta = hit.metadata
            if meta.get("migrated_to_stream") == "true":
                continue
            tag = meta.get("memory_tag")
            source: StreamSource = (
                STREAM_SOURCE_DREAM if tag == MEMORY_TAG_PRIVATE_DREAM else STREAM_SOURCE_THOUGHT
            )
            extra: dict[str, Any] = {"legacy_reflection_id": hit.doc_id}
            summary = meta.get("reflection_summary")
            body = hit.text.strip()
            if isinstance(summary, str) and summary.strip() and source == STREAM_SOURCE_DREAM:
                body = f"{body}\n\n[waking recall] {summary.strip()}"
            new_id = self.remember_stream_event(
                thread_id=thread_id,
                text=body,
                stream_source=source,
                extra_metadata=extra,
            )
            updated = dict(meta)
            updated["migrated_to_stream"] = "true"
            updated["stream_doc_id"] = new_id
            if hit.doc_id:
                self.update_doc_metadata(doc_id=hit.doc_id, metadata=updated)
            migrated += 1
        return migrated
