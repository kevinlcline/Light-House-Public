"""Portable SQLite long-term memory store (replaces Chroma)."""

from __future__ import annotations

import json
import logging
import re
import sqlite3
import threading
import time
import uuid
from collections import Counter
from datetime import datetime, timezone
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal

from light_house.memory.constants import (
    CORPUS_MEMORY_KINDS,
    GLOBAL_THREAD_ID,
    MEMORY_KIND_ERA,
    MEMORY_KIND_FACT,
    MEMORY_KIND_REFLECTION,
    MEMORY_KIND_STREAM,
    MEMORY_KIND_SUMMARY,
    MEMORY_KIND_TURN,
    META_ERA_DOC_COUNT,
    META_ERA_PERIOD,
    MEMORY_TAG_PRIVATE_DREAM,
    MEMORY_TAG_PRIVATE_RUMINATION,
    PINNED_FALSE,
    PINNED_TRUE,
    PRIVATE_TRUE,
    SCORE_UNSCORED,
    STREAM_SOURCE_CHAT,
    STREAM_SOURCE_DREAM,
    STREAM_SOURCE_THOUGHT,
    StreamSource,
)
from light_house.memory.dedup import is_near_duplicate_text
from light_house.memory.models import MemoryHit
from light_house.memory.util import body_dedup_key, turn_dedup_key

logger = logging.getLogger(__name__)

_KNOWN_META_KEYS = frozenset(
    {
        "thread_id",
        "memory_kind",
        "pinned",
        "dedup_key",
        "ts",
        "stream_source",
        "memory_tag",
        "private",
        "impact_score",
        "coherence_score",
        "fade_level",
        "reflection_summary",
        "foundation_seed_id",
        "migrated_to_stream",
        "stream_doc_id",
        "legacy_reflection_id",
        META_ERA_PERIOD,
        META_ERA_DOC_COUNT,
    }
)

_SEARCHABLE_MEMORY_KINDS = CORPUS_MEMORY_KINDS


class FileMemoryStore:
    """SQLite-backed conscious stream and pinned facts (no vector DB)."""

    def __init__(self, store_path: Path, *, collection_name: str = "light_house_longterm") -> None:
        self._store_path = store_path.resolve()
        self._store_path.mkdir(parents=True, exist_ok=True)
        self._db_path = self._store_path / "memory.db"
        self._collection_name = collection_name
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self._db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_schema()

    @contextmanager
    def _db(self):
        with self._lock:
            yield self._conn

    def _fetchone(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Row | None:
        with self._db() as conn:
            return conn.execute(sql, params).fetchone()

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[sqlite3.Row]:
        with self._db() as conn:
            return conn.execute(sql, params).fetchall()

    def _execute_commit(self, sql: str, params: tuple[Any, ...] = ()) -> None:
        with self._db() as conn:
            conn.execute(sql, params)
            conn.commit()

    def _init_schema(self) -> None:
        with self._db() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_documents (
                    doc_id TEXT PRIMARY KEY,
                    thread_id TEXT NOT NULL,
                    body TEXT NOT NULL,
                    memory_kind TEXT NOT NULL,
                    pinned INTEGER NOT NULL DEFAULT 0,
                    ts REAL NOT NULL,
                    stream_source TEXT,
                    memory_tag TEXT,
                    private_flag INTEGER NOT NULL DEFAULT 0,
                    dedup_key TEXT,
                    impact_score REAL NOT NULL DEFAULT -1,
                    coherence_score REAL NOT NULL DEFAULT -1,
                    fade_level INTEGER NOT NULL DEFAULT 0,
                    reflection_summary TEXT,
                    foundation_seed_id TEXT,
                    migrated_to_stream TEXT,
                    stream_doc_id TEXT,
                    legacy_reflection_id TEXT,
                    extra_json TEXT NOT NULL DEFAULT '{}'
                );
                CREATE INDEX IF NOT EXISTS idx_mem_thread_kind
                    ON memory_documents(thread_id, memory_kind);
                CREATE INDEX IF NOT EXISTS idx_mem_dedup
                    ON memory_documents(thread_id, dedup_key);
                CREATE INDEX IF NOT EXISTS idx_mem_foundation
                    ON memory_documents(foundation_seed_id);
                CREATE TABLE IF NOT EXISTS memory_meta (
                    key TEXT PRIMARY KEY,
                    value TEXT NOT NULL
                );
                """
            )
            conn.execute(
                "INSERT OR IGNORE INTO memory_meta(key, value) VALUES (?, ?)",
                ("collection_name", self._collection_name),
            )
            conn.commit()
        self._ensure_fts_schema()

    def _ensure_fts_schema(self) -> None:
        with self._db() as conn:
            conn.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS memory_fts USING fts5(
                    doc_id UNINDEXED,
                    thread_id UNINDEXED,
                    body,
                    stream_source UNINDEXED,
                    memory_kind UNINDEXED,
                    ts UNINDEXED,
                    tokenize='porter unicode61'
                )
                """
            )
            conn.commit()
        row = self._fetchone("SELECT COUNT(*) AS c FROM memory_fts")
        indexed = int(row["c"]) if row else 0
        if indexed == 0:
            self._backfill_fts()

    def _backfill_fts(self) -> None:
        placeholders = ",".join("?" for _ in _SEARCHABLE_MEMORY_KINDS)
        rows = self._fetchall(
            f"""
            SELECT doc_id, thread_id, body, stream_source, memory_kind, ts
            FROM memory_documents
            WHERE memory_kind IN ({placeholders})
            """,
            _SEARCHABLE_MEMORY_KINDS,
        )
        if not rows:
            return
        with self._db() as conn:
            for row in rows:
                conn.execute(
                    """
                    INSERT INTO memory_fts(doc_id, thread_id, body, stream_source, memory_kind, ts)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (
                        row["doc_id"],
                        row["thread_id"],
                        row["body"],
                        row["stream_source"],
                        row["memory_kind"],
                        float(row["ts"]),
                    ),
                )
            conn.commit()

    @staticmethod
    def _fts_query_text(query: str) -> str:
        tokens = re.findall(r"[\w']+", query.lower())
        return " ".join(tokens)

    def _sync_fts_row(
        self,
        *,
        doc_id: str,
        thread_id: str,
        body: str,
        stream_source: str | None,
        memory_kind: str,
        ts: float,
    ) -> None:
        if memory_kind not in _SEARCHABLE_MEMORY_KINDS:
            return
        with self._db() as conn:
            conn.execute("DELETE FROM memory_fts WHERE doc_id = ?", (doc_id,))
            conn.execute(
                """
                INSERT INTO memory_fts(doc_id, thread_id, body, stream_source, memory_kind, ts)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (doc_id, thread_id, body, stream_source, memory_kind, ts),
            )
            conn.commit()

    def _delete_fts_rows(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        placeholders = ",".join("?" for _ in doc_ids)
        self._execute_commit(
            f"DELETE FROM memory_fts WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )

    def count_all(self) -> int:
        row = self._fetchone("SELECT COUNT(*) AS c FROM memory_documents")
        return int(row["c"]) if row else 0

    @staticmethod
    def _dedup_key(user_text: str, assistant_text: str) -> str:
        return turn_dedup_key(user_text, assistant_text)

    @staticmethod
    def _meta_ts(meta: dict[str, Any]) -> float:
        try:
            return float(meta.get("ts", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def _split_metadata(self, metadata: dict[str, Any]) -> tuple[dict[str, Any], str]:
        extra = {k: v for k, v in metadata.items() if k not in _KNOWN_META_KEYS}
        return metadata, json.dumps(extra, ensure_ascii=False)

    def _row_to_hit(self, row: sqlite3.Row) -> MemoryHit:
        meta = self._metadata_from_row(row)
        return MemoryHit(text=str(row["body"]), score=None, metadata=meta, doc_id=str(row["doc_id"]))

    def _metadata_from_row(self, row: sqlite3.Row) -> dict[str, Any]:
        meta: dict[str, Any] = {
            "thread_id": row["thread_id"],
            "memory_kind": row["memory_kind"],
            "pinned": PINNED_TRUE if row["pinned"] else PINNED_FALSE,
            "ts": row["ts"],
        }
        if row["dedup_key"]:
            meta["dedup_key"] = row["dedup_key"]
        if row["stream_source"]:
            meta["stream_source"] = row["stream_source"]
        if row["memory_tag"]:
            meta["memory_tag"] = row["memory_tag"]
        if row["private_flag"]:
            meta["private"] = PRIVATE_TRUE
        meta["impact_score"] = float(row["impact_score"])
        meta["coherence_score"] = float(row["coherence_score"])
        meta["fade_level"] = int(row["fade_level"])
        if row["reflection_summary"]:
            meta["reflection_summary"] = row["reflection_summary"]
        if row["foundation_seed_id"]:
            meta["foundation_seed_id"] = row["foundation_seed_id"]
        if row["migrated_to_stream"]:
            meta["migrated_to_stream"] = row["migrated_to_stream"]
        if row["stream_doc_id"]:
            meta["stream_doc_id"] = row["stream_doc_id"]
        if row["legacy_reflection_id"]:
            meta["legacy_reflection_id"] = row["legacy_reflection_id"]
        try:
            extra = json.loads(row["extra_json"] or "{}")
            if isinstance(extra, dict):
                meta.update(extra)
        except json.JSONDecodeError:
            pass
        return meta

    def _insert_row(self, *, doc_id: str, body: str, metadata: dict[str, Any]) -> None:
        _, extra_json = self._split_metadata(metadata)
        with self._db() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO memory_documents (
                    doc_id, thread_id, body, memory_kind, pinned, ts,
                    stream_source, memory_tag, private_flag, dedup_key,
                    impact_score, coherence_score, fade_level, reflection_summary,
                    foundation_seed_id, migrated_to_stream, stream_doc_id,
                    legacy_reflection_id, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    doc_id,
                    str(metadata.get("thread_id", "")),
                    body,
                    str(metadata.get("memory_kind", MEMORY_KIND_STREAM)),
                    1 if metadata.get("pinned") == PINNED_TRUE else 0,
                    float(metadata.get("ts", time.time())),
                    metadata.get("stream_source"),
                    metadata.get("memory_tag"),
                    1 if metadata.get("private") == PRIVATE_TRUE else 0,
                    metadata.get("dedup_key"),
                    float(metadata.get("impact_score", SCORE_UNSCORED)),
                    float(metadata.get("coherence_score", SCORE_UNSCORED)),
                    int(metadata.get("fade_level", 0)),
                    metadata.get("reflection_summary"),
                    metadata.get("foundation_seed_id"),
                    metadata.get("migrated_to_stream"),
                    metadata.get("stream_doc_id"),
                    metadata.get("legacy_reflection_id"),
                    extra_json,
                ),
            )
            conn.commit()
        self._sync_fts_row(
            doc_id=doc_id,
            thread_id=str(metadata.get("thread_id", "")),
            body=body,
            stream_source=metadata.get("stream_source"),
            memory_kind=str(metadata.get("memory_kind", MEMORY_KIND_STREAM)),
            ts=float(metadata.get("ts", time.time())),
        )

    def import_document(self, *, doc_id: str, body: str, metadata: dict[str, Any]) -> None:
        """Import one document (migration from Chroma)."""
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)

    def _find_by_dedup_key(
        self, *, thread_id: str, memory_kind: str, dedup_key: str
    ) -> str | None:
        row = self._fetchone(
            """
            SELECT doc_id FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ? AND dedup_key = ?
            LIMIT 1
            """,
            (thread_id, memory_kind, dedup_key),
        )
        return str(row["doc_id"]) if row else None

    def _find_near_duplicate(
        self,
        *,
        thread_id: str,
        memory_kind: str,
        query_text: str,
        limit: int = 40,
    ) -> str | None:
        if not query_text or len(query_text.strip()) < 10:
            return None
        rows = self._fetchall(
            """
            SELECT doc_id, body FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ?
            ORDER BY ts DESC LIMIT ?
            """,
            (thread_id, memory_kind, limit),
        )
        for row in rows:
            if is_near_duplicate_text(query_text, str(row["body"])):
                return str(row["doc_id"])
        return None

    def remember_turn(
        self,
        *,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        extra_metadata: dict[str, Any] | None = None,
        dedup_threshold: float | None = 0.12,
    ) -> str:
        text = f"user: {user_text.strip()}\nassistant: {assistant_text.strip()}"
        dedup_key = self._dedup_key(user_text, assistant_text)
        ts = time.time()
        exact_id = self._find_by_dedup_key(
            thread_id=thread_id, memory_kind=MEMORY_KIND_TURN, dedup_key=dedup_key
        )
        metadata = self._base_turn_metadata(thread_id, dedup_key, ts)
        if extra_metadata:
            metadata.update(extra_metadata)
        if exact_id:
            self._insert_row(doc_id=exact_id, body=text, metadata=metadata)
            return exact_id
        if dedup_threshold is not None:
            dup_id = self._find_near_duplicate(
                thread_id=thread_id, memory_kind=MEMORY_KIND_TURN, query_text=text
            )
            if dup_id:
                self._insert_row(doc_id=dup_id, body=text, metadata=metadata)
                return dup_id
        doc_id = str(uuid.uuid4())
        self._insert_row(doc_id=doc_id, body=text, metadata=metadata)
        return doc_id

    def _base_turn_metadata(self, thread_id: str, dedup_key: str, ts: float) -> dict[str, Any]:
        return {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_TURN,
            "pinned": PINNED_FALSE,
            "dedup_key": dedup_key,
            "ts": ts,
        }

    def pin_sacred_fact(
        self,
        *,
        text: str,
        thread_id: str,
        scope: Literal["thread", "global"] = "thread",
        extra_metadata: dict[str, Any] | None = None,
    ) -> str:
        bucket = GLOBAL_THREAD_ID if scope == "global" else thread_id
        body = text.strip()
        meta_extra = dict(extra_metadata or {})
        foundation_seed_id = meta_extra.get("foundation_seed_id")
        if foundation_seed_id:
            from light_house.memory.foundation import foundation_dedup_key

            dedup_key = foundation_dedup_key(str(foundation_seed_id))
            existing_id = self._find_by_foundation_seed_id(str(foundation_seed_id))
        else:
            dedup_key = body_dedup_key(body)
            existing_id = self._find_pinned_by_dedup_key(thread_id=bucket, dedup_key=dedup_key)
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
            self._insert_row(doc_id=existing_id, body=body, metadata=metadata)
            return existing_id
        doc_id = str(uuid.uuid4())
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
        return doc_id

    def _find_by_foundation_seed_id(self, foundation_seed_id: str) -> str | None:
        row = self._fetchone(
            """
            SELECT doc_id FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ? AND pinned = 1
              AND foundation_seed_id = ?
            LIMIT 1
            """,
            (GLOBAL_THREAD_ID, MEMORY_KIND_FACT, foundation_seed_id),
        )
        return str(row["doc_id"]) if row else None

    def _find_pinned_by_dedup_key(self, *, thread_id: str, dedup_key: str) -> str | None:
        row = self._fetchone(
            """
            SELECT doc_id FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ? AND pinned = 1 AND dedup_key = ?
            LIMIT 1
            """,
            (thread_id, MEMORY_KIND_FACT, dedup_key),
        )
        return str(row["doc_id"]) if row else None

    def list_pinned_facts(self, *, thread_id: str, limit: int = 32) -> list[MemoryHit]:
        rows = self._fetchall(
            """
            SELECT * FROM memory_documents
            WHERE memory_kind = ? AND pinned = 1
              AND (thread_id = ? OR thread_id = ?)
            ORDER BY ts ASC LIMIT ?
            """,
            (MEMORY_KIND_FACT, thread_id, GLOBAL_THREAD_ID, limit),
        )
        return [self._row_to_hit(row) for row in rows]

    def search(self, query: str, *, thread_id: str | None, k: int) -> list[MemoryHit]:
        """Keyword recall via FTS5 (substring fallback when FTS returns nothing)."""
        if thread_id is None:
            return self.search_stream_corpus(
                query, thread_id=GLOBAL_THREAD_ID, k=k, global_only=True
            )
        return self.search_stream_corpus(query, thread_id=thread_id, k=k)

    def search_stream_corpus(
        self,
        query: str,
        *,
        thread_id: str,
        k: int = 10,
        stream_source: str | None = None,
        since_ts: float | None = None,
        global_only: bool = False,
    ) -> list[MemoryHit]:
        """Full-text search over conscious-stream corpus for one thread."""
        fts_q = self._fts_query_text(query)
        if not fts_q:
            return []
        cap = max(1, min(30, k))
        kind_placeholders = ",".join("?" for _ in _SEARCHABLE_MEMORY_KINDS)
        params: list[Any] = [fts_q]
        if global_only:
            thread_clause = "memory_fts.thread_id = ?"
            params.append(GLOBAL_THREAD_ID)
        else:
            thread_clause = "(memory_fts.thread_id = ? OR memory_fts.thread_id = ?)"
            params.extend([thread_id, GLOBAL_THREAD_ID])
        source_clause = ""
        if stream_source:
            source_clause = " AND memory_fts.stream_source = ?"
            params.append(stream_source)
        since_clause = ""
        if since_ts is not None:
            since_clause = " AND memory_fts.ts >= ?"
            params.append(float(since_ts))
        params.extend(_SEARCHABLE_MEMORY_KINDS)
        params.append(cap)

        rows = self._fetchall(
            f"""
            SELECT d.*
            FROM memory_fts
            JOIN memory_documents d ON d.doc_id = memory_fts.doc_id
            WHERE memory_fts MATCH ? AND {thread_clause}
              AND memory_fts.memory_kind IN ({kind_placeholders})
              {source_clause}
              {since_clause}
            ORDER BY bm25(memory_fts) ASC, memory_fts.ts DESC
            LIMIT ?
            """,
            tuple(params),
        )
        hits = [self._row_to_hit(row) for row in rows]
        if hits:
            return hits

        # Substring fallback for very short queries FTS may miss.
        q = query.strip().lower()
        if not q:
            return []
        fallback_thread = GLOBAL_THREAD_ID if global_only else thread_id
        placeholders = ",".join("?" for _ in _SEARCHABLE_MEMORY_KINDS)
        if global_only:
            fb_rows = self._fetchall(
                f"""
                SELECT * FROM memory_documents
                WHERE thread_id = ? AND memory_kind IN ({placeholders})
                ORDER BY ts DESC LIMIT ?
                """,
                (fallback_thread, *_SEARCHABLE_MEMORY_KINDS, cap * 4),
            )
        else:
            fb_rows = self._fetchall(
                f"""
                SELECT * FROM memory_documents
                WHERE (thread_id = ? OR thread_id = ?) AND memory_kind IN ({placeholders})
                ORDER BY ts DESC LIMIT ?
                """,
                (thread_id, GLOBAL_THREAD_ID, *_SEARCHABLE_MEMORY_KINDS, cap * 4),
            )
        return [
            self._row_to_hit(row) for row in fb_rows if q in str(row["body"]).lower()
        ][:cap]

    def corpus_stats(self, *, thread_id: str) -> dict[str, Any]:
        """Aggregate counts for memory-index injection (algorithmic, no LLM)."""
        hits = self.list_thread_corpus(thread_id=thread_id)
        by_origin: Counter[str] = Counter()
        by_month: Counter[str] = Counter()
        month_origin: dict[str, Counter[str]] = {}
        unscored = 0
        for hit in hits:
            origin = self.stream_source_from_metadata(hit.metadata) or "other"
            by_origin[origin] += 1
            if float(hit.metadata.get("impact_score", SCORE_UNSCORED)) < 0:
                unscored += 1
            ts = self._meta_ts(hit.metadata)
            if ts > 0:
                month_key = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m")
                by_month[month_key] += 1
                month_origin.setdefault(month_key, Counter())[origin] += 1

        pinned = len(self.list_pinned_facts(thread_id=thread_id, limit=10_000))
        recent_months = sorted(by_month.keys(), reverse=True)[:6]
        month_rows = [
            {
                "month": month,
                "total": by_month[month],
                "by_origin": dict(month_origin.get(month, Counter())),
            }
            for month in reversed(recent_months)
        ]
        return {
            "total_corpus": len(hits),
            "unscored": unscored,
            "pinned": pinned,
            "by_origin": dict(by_origin),
            "by_month": month_rows,
        }

    def count_turns(self, *, thread_id: str) -> int:
        row = self._fetchone(
            "SELECT COUNT(*) AS c FROM memory_documents WHERE thread_id = ? AND memory_kind = ?",
            (thread_id, MEMORY_KIND_TURN),
        )
        return int(row["c"]) if row else 0

    def get_recent_turn_texts(self, *, thread_id: str, limit: int = 24) -> list[str]:
        rows = self._fetchall(
            """
            SELECT body, ts FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ?
            ORDER BY ts ASC
            """,
            (thread_id, MEMORY_KIND_TURN),
        )
        texts = [str(row["body"]) for row in rows]
        return texts[-limit:]

    def list_recent_summaries(self, *, thread_id: str, limit: int = 3) -> list[MemoryHit]:
        rows = self._fetchall(
            """
            SELECT * FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ?
            ORDER BY ts ASC
            """,
            (thread_id, MEMORY_KIND_SUMMARY),
        )
        hits = [self._row_to_hit(row) for row in rows]
        return hits[-limit:]

    def add_thread_summary(self, *, thread_id: str, summary_text: str) -> str:
        doc_id = str(uuid.uuid4())
        metadata: dict[str, Any] = {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_SUMMARY,
            "pinned": PINNED_FALSE,
            "ts": time.time(),
        }
        body = f"summary: {summary_text.strip()}"
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
        return doc_id

    def add_private_reflection(
        self,
        *,
        thread_id: str,
        text: str,
        memory_tag: Literal["private_dream", "private_rumination"],
        summary: str | None = None,
    ) -> str:
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
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
        return doc_id

    def list_recent_rumination_summaries(self, *, thread_id: str, limit: int = 3) -> list[MemoryHit]:
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
        rows = self._fetchall(
            """
            SELECT * FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ? AND memory_tag = ? AND private_flag = 1
            ORDER BY ts ASC
            """,
            (thread_id, MEMORY_KIND_REFLECTION, memory_tag),
        )
        hits = [self._row_to_hit(row) for row in rows]
        return hits[-limit:]

    def latest_private_reflection_ts(
        self,
        *,
        thread_id: str,
        memory_tag: Literal["private_dream", "private_rumination"],
    ) -> float | None:
        hits = self.list_recent_private_reflections(
            thread_id=thread_id, memory_tag=memory_tag, limit=1
        )
        if not hits:
            return None
        ts_raw = hits[-1].metadata.get("ts")
        try:
            return float(ts_raw)
        except (TypeError, ValueError):
            return None

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
            exact_id = self._find_by_dedup_key(
                thread_id=thread_id, memory_kind=MEMORY_KIND_STREAM, dedup_key=dedup_key
            )
            if exact_id:
                self._insert_row(doc_id=exact_id, body=body, metadata=metadata)
                return exact_id
            if dedup_threshold is not None and len(body) >= 10:
                dup_id = self._find_near_duplicate(
                    thread_id=thread_id, memory_kind=MEMORY_KIND_STREAM, query_text=body
                )
                if dup_id:
                    self._insert_row(doc_id=dup_id, body=body, metadata=metadata)
                    return dup_id
        doc_id = str(uuid.uuid4())
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
        return doc_id

    def list_thread_corpus(self, *, thread_id: str) -> list[MemoryHit]:
        placeholders = ",".join("?" for _ in CORPUS_MEMORY_KINDS)
        rows = self._fetchall(
            f"""
            SELECT * FROM memory_documents
            WHERE thread_id = ? AND memory_kind IN ({placeholders})
              AND NOT (
                memory_kind = ? AND COALESCE(migrated_to_stream, '') = 'true'
              )
            ORDER BY ts ASC
            """,
            (thread_id, *CORPUS_MEMORY_KINDS, MEMORY_KIND_REFLECTION),
        )
        return [self._row_to_hit(row) for row in rows]

    def measure_thread_corpus_chars(self, *, thread_id: str) -> int:
        return sum(len(h.text) for h in self.list_thread_corpus(thread_id=thread_id))

    def list_unscored_corpus(self, *, thread_id: str, limit: int = 50) -> list[MemoryHit]:
        hits = [
            h
            for h in self.list_thread_corpus(thread_id=thread_id)
            if float(h.metadata.get("impact_score", SCORE_UNSCORED)) < 0
        ]
        return hits[:limit]

    def _get_doc(self, doc_id: str) -> tuple[str, dict[str, Any]] | None:
        row = self._fetchone(
            "SELECT * FROM memory_documents WHERE doc_id = ?", (doc_id,)
        )
        if row is None:
            return None
        return str(row["body"]), self._metadata_from_row(row)

    def update_doc_metadata(self, *, doc_id: str, metadata: dict[str, Any]) -> None:
        existing = self._get_doc(doc_id)
        if existing is None:
            return
        body, _old = existing
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)

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
        self.update_doc_metadata(doc_id=doc_id, metadata=updated)

    def bump_fade_level(self, *, doc_id: str, metadata: dict[str, Any], increment: int = 1) -> None:
        updated = dict(metadata)
        try:
            level = int(updated.get("fade_level", 0))
        except (TypeError, ValueError):
            level = 0
        updated["fade_level"] = min(3, level + increment)
        self.update_doc_metadata(doc_id=doc_id, metadata=updated)

    def delete_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        self._delete_fts_rows(doc_ids)
        placeholders = ",".join("?" for _ in doc_ids)
        self._execute_commit(
            f"DELETE FROM memory_documents WHERE doc_id IN ({placeholders})",
            tuple(doc_ids),
        )

    def add_era_summary(
        self,
        *,
        thread_id: str,
        period_label: str,
        summary_text: str,
        doc_count: int,
        ts_start: float | None = None,
        ts_end: float | None = None,
    ) -> str:
        """Persist a curator-written era card for the memory index (not conscious stream)."""
        doc_id = str(uuid.uuid4())
        now = time.time()
        metadata: dict[str, Any] = {
            "thread_id": thread_id,
            "memory_kind": MEMORY_KIND_ERA,
            "pinned": PINNED_FALSE,
            "ts": now,
            META_ERA_PERIOD: period_label.strip(),
            META_ERA_DOC_COUNT: int(doc_count),
        }
        if ts_start is not None:
            metadata["era_ts_start"] = float(ts_start)
        if ts_end is not None:
            metadata["era_ts_end"] = float(ts_end)
        body = f"era: {summary_text.strip()}"
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
        return doc_id

    def list_era_summaries(self, *, thread_id: str, limit: int = 5) -> list[MemoryHit]:
        rows = self._fetchall(
            """
            SELECT * FROM memory_documents
            WHERE thread_id = ? AND memory_kind = ?
            ORDER BY ts DESC
            LIMIT ?
            """,
            (thread_id, MEMORY_KIND_ERA, max(1, limit)),
        )
        return [self._row_to_hit(row) for row in rows]

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
        self._insert_row(doc_id=doc_id, body=body, metadata=metadata)
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
