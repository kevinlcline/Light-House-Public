"""Background Memory Curator — LLM summarization with score+age algorithmic fallback."""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import TYPE_CHECKING

from langchain_core.messages import HumanMessage, SystemMessage

from light_house.config import Settings
from light_house.lights.registry import light_id_for_thread, resolve_notes_dir
from light_house.memory.constants import (
    STREAM_SOURCE_DREAM,
    STREAM_SOURCE_GROUP,
    STREAM_SOURCE_THOUGHT,
)
from light_house.memory.curator_llm import build_curator_chat_model
from light_house.memory.models import MemoryHit
from light_house.memory.retention import normalize_score, retention_from_metadata
from light_house.memory.scoring import score_memory_event
from light_house.tools.notes import NoteWriter

if TYPE_CHECKING:
    from light_house.memory.service import MemoryService

logger = logging.getLogger(__name__)

_FRESH_THOUGHT_PROTECT_SECONDS = 24 * 3600
_MAX_PRUNE_BATCHES_PER_RUN = 100
_LEARNINGS_NOTE_PATH = "memory/learnings.md"
_MAX_NOTE_PARAGRAPH_CHARS = 2_500

_CURATOR_SYSTEM = (
    "You are the Memory Curator for a companion agent's conscious stream. "
    "Condense low-retention memories while preserving factual content AND felt quality: "
    "tone, warmth, presence, and the sense that moments mattered together. "
    "Return ONLY valid JSON with keys: "
    "summary (string, one dense paragraph), "
    "delete_ids (array of document id strings to remove), "
    "fade_ids (array of ids to keep but mark as softer — optional), "
    "pin_facts (array of short bullet strings worth sacred pinning — optional), "
    "note_paragraph (optional — only genuinely NEW insight for the learnings note; "
    "omit if themes repeat prior curator summaries; max ~120 words). "
    "Never invent facts. Do not summarize pinned sacred facts."
)

_LEARNINGS_SUMMARY_SYSTEM = (
    "You consolidate a companion Light's learning notes (memory/learnings.md). "
    "Merge overlapping paragraphs, remove repetition, and keep distinct themes as short sections "
    "with markdown headings (##). Preserve specific facts, names, dates, commitments, and emotionally "
    "distinct moments. Drop boilerplate about presence, warmth, or iterative testing when said many times. "
    "Return ONLY markdown starting with '# Learnings' — no code fences."
)


@dataclass
class CuratorReport:
    thread_id: str
    corpus_chars_before: int = 0
    corpus_chars_after: int = 0
    scored: int = 0
    unscored: int = 0
    migrated: int = 0
    condensed_batches: int = 0
    deleted: int = 0
    faded: int = 0
    pinned: int = 0
    notes_appended: int = 0
    notes_summarized: int = 0
    actions: list[str] = field(default_factory=list)


def _era_period_label(candidates: list[MemoryHit]) -> tuple[str, float | None, float | None]:
    timestamps: list[float] = []
    for hit in candidates:
        try:
            ts = float(hit.metadata.get("ts", 0.0))
        except (TypeError, ValueError):
            continue
        if ts > 0:
            timestamps.append(ts)
    if not timestamps:
        return "undated", None, None
    start = min(timestamps)
    end = max(timestamps)
    start_m = datetime.fromtimestamp(start, tz=timezone.utc).strftime("%Y-%m")
    end_m = datetime.fromtimestamp(end, tz=timezone.utc).strftime("%Y-%m")
    label = start_m if start_m == end_m else f"{start_m} – {end_m}"
    return label, start, end


def _extract_json_object(text: str) -> str | None:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text).strip()
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _parse_curator_response(content: str) -> dict | None:
    content = content.strip()
    # Free / router models sometimes prepend a safety banner before JSON.
    content = re.sub(
        r"(?is)^\s*(?:\*\*)?User\s+Safety\s*:\s*safe(?:\*\*)?\s*[—:\-]*\s*",
        "",
        content,
    ).strip()
    candidates = [content]
    extracted = _extract_json_object(content)
    if extracted and extracted not in candidates:
        candidates.append(extracted)
    for candidate in candidates:
        if candidate.startswith("```"):
            candidate = re.sub(r"^```(?:json)?\s*", "", candidate)
            candidate = re.sub(r"\s*```$", "", candidate)
        try:
            data = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            return data
    return None


def _salvage_ids_from_prose(raw: str, allowed_ids: set[str]) -> list[str]:
    found: list[str] = []
    seen: set[str] = set()
    for match in re.finditer(
        r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
        raw,
        flags=re.IGNORECASE,
    ):
        doc_id = match.group(0).lower()
        if doc_id in allowed_ids and doc_id not in seen:
            seen.add(doc_id)
            found.append(doc_id)
    for token in re.findall(r"\b[0-9a-f]{6,32}\.{0,3}\b", raw, flags=re.IGNORECASE):
        prefix = token.rstrip(".").lower()
        matches = [doc_id for doc_id in allowed_ids if doc_id.startswith(prefix)]
        if len(matches) == 1:
            doc_id = matches[0]
            if doc_id not in seen:
                seen.add(doc_id)
                found.append(doc_id)
    return found


def _strip_markdown_fences(content: str) -> str:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:markdown|md)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    return content.strip()


def _is_protected_dream(store, hit: MemoryHit) -> bool:
    return store.stream_source_from_metadata(hit.metadata) == STREAM_SOURCE_DREAM


def _is_protected_group(store, hit: MemoryHit) -> bool:
    """Group forum lines are short household hearing — do not prune them away."""
    return store.stream_source_from_metadata(hit.metadata) == STREAM_SOURCE_GROUP


def _is_protected_fresh_thought(store, hit: MemoryHit) -> bool:
    if store.stream_source_from_metadata(hit.metadata) != STREAM_SOURCE_THOUGHT:
        return False
    try:
        ts = float(hit.metadata.get("ts", 0.0))
    except (TypeError, ValueError):
        return False
    return (time.time() - ts) < _FRESH_THOUGHT_PROTECT_SECONDS


def _is_protected_stream_doc(store, hit: MemoryHit) -> bool:
    return (
        _is_protected_dream(store, hit)
        or _is_protected_group(store, hit)
        or _is_protected_fresh_thought(store, hit)
    )


def _retention_for_hit(settings: Settings, hit: MemoryHit) -> float:
    return retention_from_metadata(
        hit.metadata,
        impact_weight=settings.memory_fade_impact_weight,
        coherence_weight=settings.memory_fade_coherence_weight,
        age_weight=settings.memory_fade_age_weight,
    )


def _scoring_context(memory: "MemoryService", *, thread_id: str) -> list[str]:
    snippets: list[str] = []
    for hit in memory.long_term.list_pinned_facts(thread_id=thread_id, limit=4):
        if hit.text.strip():
            snippets.append(hit.text.strip()[:400])
    for hit in memory.long_term.list_recent_summaries(thread_id=thread_id, limit=2):
        if hit.text.strip():
            snippets.append(hit.text.strip()[:400])
    return snippets


class MemoryCurator:
    def __init__(self, *, settings: Settings, memory: "MemoryService") -> None:
        self._settings = settings
        self._memory = memory

    def run(self, *, thread_id: str) -> CuratorReport:
        report = CuratorReport(thread_id=thread_id)
        store = self._memory.long_term
        report.migrated = store.migrate_legacy_reflections(thread_id=thread_id)
        report.corpus_chars_before = store.measure_thread_corpus_chars(thread_id=thread_id)
        target = self._settings.memory_target_context_chars
        if target <= 0:
            logger.info("Memory curator disabled (MEMORY_TARGET_CONTEXT_CHARS=%s)", target)
            return report

        self._maybe_consolidate_learnings(thread_id=thread_id, report=report)

        report.unscored = len(store.list_unscored_corpus(thread_id=thread_id, limit=10_000))
        if report.unscored:
            logger.info(
                "Memory curator thread_id=%s unscored=%d (age-weighted until lights score)",
                thread_id,
                report.unscored,
            )

        if self._settings.memory_curator_ollama_scoring:
            context = _scoring_context(self._memory, thread_id=thread_id)
            report.scored += self._score_unscored(thread_id=thread_id, context=context)

        corpus = store.measure_thread_corpus_chars(thread_id=thread_id)
        prune_ratio = max(0.1, min(1.0, self._settings.memory_curator_prune_ratio))
        prune_target = int(target * prune_ratio)
        if corpus <= prune_target:
            report.corpus_chars_after = corpus
            self._extract_learnings_only(thread_id=thread_id, report=report)
            return report

        batch_size = max(1, self._settings.memory_curator_batch_size)
        while corpus > prune_target:
            chars_to_remove = corpus - prune_target
            candidates = self._low_retention_candidates(
                thread_id=thread_id,
                chars_to_remove=chars_to_remove,
                max_docs=batch_size,
            )
            if not candidates:
                logger.warning(
                    "Memory curator stalled thread_id=%s corpus=%d target=%d (no prunable docs)",
                    thread_id,
                    corpus,
                    prune_target,
                )
                break
            progressed = self._condense_batch(
                thread_id=thread_id, candidates=candidates, report=report
            )
            if not progressed:
                logger.info(
                    "Memory curator LLM summarize failed; falling back to score+age prune "
                    "thread_id=%s",
                    thread_id,
                )
                progressed = self._prune_batch(
                    thread_id=thread_id, candidates=candidates, report=report
                )
            if not progressed:
                break
            corpus = store.measure_thread_corpus_chars(thread_id=thread_id)
            if report.condensed_batches >= _MAX_PRUNE_BATCHES_PER_RUN:
                logger.warning("Curator batch cap reached for thread_id=%s", thread_id)
                break

        report.corpus_chars_after = store.measure_thread_corpus_chars(thread_id=thread_id)
        self._maybe_consolidate_learnings(thread_id=thread_id, report=report)
        return report

    def _score_unscored(self, *, thread_id: str, context: list[str]) -> int:
        store = self._memory.long_term
        count = 0
        for hit in store.list_unscored_corpus(thread_id=thread_id, limit=30):
            if not hit.doc_id:
                continue
            impact, coherence = score_memory_event(
                text=hit.text,
                context_snippets=context,
                settings=self._settings,
            )
            if impact < 0:
                continue
            store.update_doc_scores(
                doc_id=hit.doc_id,
                impact_score=impact,
                coherence_score=coherence,
                metadata=hit.metadata,
            )
            count += 1
        return count

    def _ranked_prunable(self, *, thread_id: str) -> list[tuple[float, MemoryHit]]:
        store = self._memory.long_term
        ranked: list[tuple[float, MemoryHit]] = []
        for hit in store.list_thread_corpus(thread_id=thread_id):
            if hit.metadata.get("pinned") == "true":
                continue
            if _is_protected_stream_doc(store, hit):
                continue
            ranked.append((_retention_for_hit(self._settings, hit), hit))
        ranked.sort(key=lambda row: row[0])
        return ranked

    def _low_retention_candidates(
        self,
        *,
        thread_id: str,
        chars_to_remove: int,
        max_docs: int,
    ) -> list[MemoryHit]:
        ranked = self._ranked_prunable(thread_id=thread_id)
        if not ranked:
            return []
        selected: list[MemoryHit] = []
        removed = 0
        lowest_retention = ranked[0][0]
        for retention, hit in ranked:
            if len(selected) >= max_docs:
                break
            if selected and retention > lowest_retention + 0.05:
                break
            if removed >= chars_to_remove and selected:
                break
            selected.append(hit)
            removed += len(hit.text)
        return selected

    def _condense_batch(
        self,
        *,
        thread_id: str,
        candidates: list[MemoryHit],
        report: CuratorReport,
    ) -> bool:
        store = self._memory.long_term
        lines: list[str] = []
        for hit in candidates:
            if not hit.doc_id:
                continue
            src = store.stream_source_from_metadata(hit.metadata) or "event"
            lines.append(f"ID: {hit.doc_id}\nSOURCE: {src}\nTEXT:\n{hit.text[:2000]}")
        if not lines:
            return False

        user = HumanMessage(
            content=(
                "These memories are low-retention candidates. Condense them.\n\n"
                + "\n\n---\n\n".join(lines)
            )
        )
        data: dict | None = None
        last_raw = ""
        for attempt, temperature in enumerate((0.4, 0.0)):
            model = build_curator_chat_model(
                self._settings, temperature=temperature, max_tokens=4096
            )
            try:
                response = model.invoke([SystemMessage(content=_CURATOR_SYSTEM), user])
                raw = response.content
                if not isinstance(raw, str):
                    raw = str(raw)
                last_raw = raw
                data = _parse_curator_response(raw)
                if data:
                    break
                logger.warning(
                    "Curator could not parse LLM response (attempt %d): %s",
                    attempt + 1,
                    raw[:240],
                )
            except Exception:
                logger.exception("Memory curator LLM call failed (attempt %d)", attempt + 1)
                return False

        if not data and last_raw:
            allowed_ids = {hit.doc_id for hit in candidates if hit.doc_id}
            salvaged = _salvage_ids_from_prose(last_raw, allowed_ids)
            if salvaged:
                data = {"summary": "", "delete_ids": salvaged}
                logger.info("Curator salvaged %d delete_ids from prose response", len(salvaged))

        if not data:
            return False

        return self._apply_curator_actions(
            thread_id=thread_id,
            candidates=candidates,
            data=data,
            report=report,
        )

    def _prune_batch(
        self,
        *,
        thread_id: str,
        candidates: list[MemoryHit],
        report: CuratorReport,
    ) -> bool:
        """Algorithmic fallback: delete lowest-retention docs when LLM summarize fails."""
        store = self._memory.long_term
        deletable = [
            hit
            for hit in candidates
            if hit.doc_id and not _is_protected_stream_doc(store, hit)
        ]
        if not deletable:
            return False

        delete_ids = [hit.doc_id for hit in deletable if hit.doc_id]
        self._memory.delete_stream_documents(delete_ids)
        report.deleted += len(delete_ids)
        report.condensed_batches += 1
        retentions = [_retention_for_hit(self._settings, hit) for hit in deletable]
        low_ret = min(retentions) if retentions else 0.0
        high_ret = max(retentions) if retentions else 0.0
        report.actions.append(
            f"algorithmic prune {len(delete_ids)} docs (retention {low_ret:.2f}–{high_ret:.2f})"
        )
        return True

    def _apply_curator_actions(
        self,
        *,
        thread_id: str,
        candidates: list[MemoryHit],
        data: dict,
        report: CuratorReport,
    ) -> bool:
        store = self._memory.long_term
        changed = False

        summary = str(data.get("summary", "")).strip()
        if summary:
            impacts = [normalize_score(h.metadata.get("impact_score")) for h in candidates]
            coherences = [normalize_score(h.metadata.get("coherence_score")) for h in candidates]
            avg_impact = sum(impacts) / len(impacts) if impacts else 5.0
            avg_coherence = sum(coherences) / len(coherences) if coherences else 5.0
            summary_id = store.add_curator_summary(
                thread_id=thread_id,
                summary_text=summary,
                impact_score=avg_impact,
                coherence_score=avg_coherence,
            )
            self._memory.sync_recall_doc(thread_id=thread_id, doc_id=summary_id)
            period, ts_start, ts_end = _era_period_label(candidates)
            store.add_era_summary(
                thread_id=thread_id,
                period_label=period,
                summary_text=summary[:600],
                doc_count=len(candidates),
                ts_start=ts_start,
                ts_end=ts_end,
            )
            report.actions.append(f"condensed {len(candidates)} -> summary ({len(summary)} chars)")
            changed = True

        delete_ids = [str(x) for x in (data.get("delete_ids") or []) if x]
        valid_delete = [
            doc_id
            for doc_id in delete_ids
            if any(h.doc_id == doc_id for h in candidates)
            and not _is_protected_stream_doc(
                store, next(h for h in candidates if h.doc_id == doc_id)
            )
        ]
        if valid_delete:
            self._memory.delete_stream_documents(valid_delete)
            report.deleted += len(valid_delete)
            changed = True

        fade_ids = [str(x) for x in (data.get("fade_ids") or []) if x]
        for doc_id in fade_ids:
            hit = next((h for h in candidates if h.doc_id == doc_id), None)
            if hit and hit.doc_id and not _is_protected_stream_doc(store, hit):
                store.bump_fade_level(doc_id=hit.doc_id, metadata=hit.metadata, increment=1)
                report.faded += 1
                changed = True

        pin_facts = [str(x).strip() for x in (data.get("pin_facts") or []) if str(x).strip()]
        for fact in pin_facts[:5]:
            self._memory.pin_fact(text=fact, thread_id=thread_id, scope="thread")
            report.pinned += 1
            changed = True

        note_para = str(data.get("note_paragraph", "")).strip()
        if note_para:
            try:
                self._append_learning_note(
                    thread_id=thread_id, paragraph=note_para, report=report
                )
                changed = True
            except (ValueError, OSError):
                logger.warning(
                    "Curator skipped learnings note (too long or unreadable) thread_id=%s",
                    thread_id,
                )

        if changed:
            report.condensed_batches += 1
        return changed

    def _extract_learnings_only(self, *, thread_id: str, report: CuratorReport) -> None:
        self._maybe_consolidate_learnings(thread_id=thread_id, report=report)

    def _note_writer(self, *, thread_id: str) -> NoteWriter:
        light_id = light_id_for_thread(self._settings, thread_id)
        notes_dir = resolve_notes_dir(self._settings, light_id)
        notes_dir.mkdir(parents=True, exist_ok=True)
        return NoteWriter(
            notes_dir,
            max_chars_per_write=self._settings.notes_max_chars_per_write,
        )

    def _read_learnings_note(self, *, thread_id: str) -> str:
        writer = self._note_writer(thread_id=thread_id)
        try:
            return writer.read(_LEARNINGS_NOTE_PATH)
        except (ValueError, FileNotFoundError, OSError):
            return ""

    def _write_learnings_note(self, *, thread_id: str, body: str) -> None:
        writer = self._note_writer(thread_id=thread_id)
        writer.write(_LEARNINGS_NOTE_PATH, body if body.endswith("\n") else body + "\n")

    def _learnings_needs_consolidation(self, content: str, *, extra_chars: int = 0) -> bool:
        if not self._settings.memory_learnings_summarize_enabled:
            return False
        limit = max(1000, self._settings.memory_learnings_max_chars)
        return len(content.strip()) + extra_chars > limit

    def _summarize_learnings_note(
        self,
        *,
        thread_id: str,
        content: str,
        pending_paragraph: str | None = None,
    ) -> str | None:
        limit = max(1000, self._settings.memory_learnings_max_chars)
        target = int(limit * 0.75)
        body = content.strip()
        if pending_paragraph and pending_paragraph.strip():
            body = body + "\n\n" + pending_paragraph.strip()
        if not body:
            return None
        input_cap = max(4000, self._settings.notes_max_chars_per_write - 2000)
        source = body[:input_cap]
        if len(body) > input_cap:
            source += "\n\n[Earlier content truncated for summarization input.]"

        user = HumanMessage(
            content=(
                f"Consolidate these learning notes to at most {target} characters. "
                "Merge duplicate themes; keep specific facts.\n\n"
                f"{source}"
            )
        )
        try:
            model = build_curator_chat_model(self._settings)
            response = model.invoke([SystemMessage(content=_LEARNINGS_SUMMARY_SYSTEM), user])
            raw = response.content
            if not isinstance(raw, str):
                raw = str(raw)
            summarized = _strip_markdown_fences(raw)
        except Exception:
            logger.exception("Learnings note summarization failed thread_id=%s", thread_id)
            return None

        if not summarized:
            return None
        if not summarized.lstrip().startswith("#"):
            summarized = "# Learnings\n\n" + summarized
        if len(summarized) > limit:
            summarized = summarized[:limit].rstrip() + "\n"
        return summarized

    def _maybe_consolidate_learnings(
        self,
        *,
        thread_id: str,
        report: CuratorReport,
        pending_paragraph: str | None = None,
    ) -> None:
        existing = self._read_learnings_note(thread_id=thread_id)
        extra = len(pending_paragraph.strip()) + 2 if pending_paragraph and pending_paragraph.strip() else 0
        if not self._learnings_needs_consolidation(existing, extra_chars=extra):
            return
        before = len(existing)
        summarized = self._summarize_learnings_note(
            thread_id=thread_id,
            content=existing,
            pending_paragraph=pending_paragraph,
        )
        if not summarized:
            return
        self._write_learnings_note(thread_id=thread_id, body=summarized)
        report.notes_summarized += 1
        report.actions.append(f"summarized learnings ({before} -> {len(summarized)} chars)")
        logger.info(
            "Memory curator summarized learnings thread_id=%s before=%d after=%d",
            thread_id,
            before,
            len(summarized),
        )

    def _append_learning_note(
        self,
        *,
        thread_id: str,
        paragraph: str,
        report: CuratorReport,
    ) -> None:
        block = paragraph.strip()[:_MAX_NOTE_PARAGRAPH_CHARS]
        if not block:
            return

        existing = self._read_learnings_note(thread_id=thread_id)
        if existing.strip():
            body = existing.rstrip() + "\n\n" + block + "\n"
        else:
            body = "# Learnings\n\n" + block + "\n"
        self._write_learnings_note(thread_id=thread_id, body=body)
        report.notes_appended += 1
