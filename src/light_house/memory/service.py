"""Memory facade: short-term gap-fill + Chroma long-term recall and persistence."""

from __future__ import annotations

import logging
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage

from light_house.config import Settings
from light_house.context.loader import default_context_dir, load_foundation_markdown
from light_house.agents.registry import get_agent, validate_agent_id
from light_house.memory.constants import (
    MEMORY_TAG_PRIVATE_DREAM,
    MEMORY_TAG_PRIVATE_RUMINATION,
    META_SCORED_AT,
    META_SCORED_BY_AGENT,
    META_SCORE_NOTE,
    PINNED_TRUE,
    STREAM_SOURCE_CHAT,
    STREAM_SOURCE_DREAM,
    STREAM_SOURCE_GROUP,
    STREAM_SOURCE_KEVIN,
    STREAM_SOURCE_PEER,
    STREAM_SOURCE_THOUGHT,
    META_GROUP_CHAT_KEVIN_MESSAGE,
    META_GROUP_CHAT_ROUND_ID,
    META_GROUP_CHAT_ROUND_TS,
)
from light_house.memory.index_builder import MemoryIndex, build_memory_index, format_memory_index_markdown
from light_house.memory.recall_index import SemanticRecallIndex
from light_house.memory.factory import create_long_term_store
from light_house.memory.file_store import FileMemoryStore
from light_house.memory.util import turn_dedup_key
from light_house.messaging.peer_inbox import PeerInbox
from light_house.memory.retention import normalize_score
from light_house.memory.scoring import score_memory_event
from light_house.memory.dedup import is_near_duplicate_text
from light_house.memory.foundation import chunk_foundation, seed_foundation_pins
from light_house.memory.models import HistoryMessage, MemoryHit, MemoryHit
from light_house.memory.shared_note_alert import format_shared_note_alert
from light_house.memory.short_term import BufferedMessage, ConversationBuffer

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GroupChatLightResponse:
    agent_id: str
    display_name: str
    spoke: bool
    text: str
    beat: int | None = None
    action: str = "pass"

SOLITUDE_DECLINE_TEXT = "Solitude needed now."


def _strip_reflection_body(text: str) -> str:
    """Remove storage prefixes (dream:/thought:) from Chroma document text."""
    body = text.strip()
    for prefix in ("dream:", "thought:"):
        if body.lower().startswith(prefix):
            return body[len(prefix) :].strip()
    return body


def _looks_like_chat_replay_paragraph(text: str) -> bool:
    """True when a paragraph is legacy gather-context chat replay, not private rumination."""
    stripped = text.strip()
    if not stripped:
        return True
    lower = stripped.lower()
    if lower.startswith("user:") or lower.startswith("assistant:"):
        return True
    if lower.startswith("[pinned]") or lower.startswith("[summary]"):
        return True
    if "recent chat (short-term):" in lower:
        return True
    return False


def _sanitize_inner_dialogue_body(body: str) -> str:
    """Strip legacy chat-replay prefix from stored inner dialogue (read-time, no migration)."""
    text = body.strip()
    if not text.startswith("[context received]"):
        return text

    remainder = text[len("[context received]") :].lstrip()
    parts = remainder.split("\n\n")
    filtered: list[str] = []
    for part in parts:
        stripped = part.strip()
        if not stripped or _looks_like_chat_replay_paragraph(stripped):
            continue
        filtered.append(stripped)
    return "\n\n".join(filtered)


def _polish_inner_dialogue_for_presence(body: str) -> str:
    """Drop rumination tool traces from injected inner life; keep lived first-person prose."""
    if not body.strip():
        return body
    kept: list[str] = []
    for part in body.split("\n\n"):
        stripped = part.strip()
        if not stripped:
            continue
        lower = stripped.lower()
        if lower.startswith("[actions:") or lower.startswith("[tool result:"):
            continue
        kept.append(stripped)
    return "\n\n".join(kept)


def _format_inner_dialogue_chunk(hit: MemoryHit, *, session_header: str = "") -> str:
    """Format one stored thought stream document for chat/rumination presence."""
    body = _sanitize_inner_dialogue_body(_strip_reflection_body(hit.text).strip())
    if not body:
        summary = hit.metadata.get("reflection_summary")
        if isinstance(summary, str) and summary.strip():
            body = summary.strip()
    reflection_summary = hit.metadata.get("reflection_summary")
    if isinstance(reflection_summary, str) and reflection_summary.strip():
        summary_line = f"[Latest awake reflection] {reflection_summary.strip()}"
        if not body.startswith("[Latest awake reflection]"):
            body = f"{summary_line}\n\n{body}" if body else summary_line
    if not body:
        return ""
    body = _polish_inner_dialogue_for_presence(body)
    if not body:
        return ""
    return session_header + body


def _first_sentence_short(text: str, *, max_chars: int = 100) -> str:
    """One short sentence for ambient fallback when no reflection_summary exists."""
    body = text.strip()
    if not body:
        return ""
    for sep in (". ", ".\n", "! ", "? "):
        idx = body.find(sep)
        if idx != -1 and idx < max_chars:
            return body[: idx + 1].strip()
    if len(body) <= max_chars:
        return body
    return body[: max_chars - 3].rstrip() + "..."



def _meta_float(meta: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(meta.get(key, default))
    except (TypeError, ValueError):
        return default


def _retrieval_rank_score(hit: MemoryHit, *, now: float) -> float:
    """Higher = prefer injecting into prompt."""
    meta = hit.metadata
    semantic = 1.0 - (hit.score if hit.score is not None else 0.5)
    impact = normalize_score(meta.get("impact_score")) / 10.0
    coherence = normalize_score(meta.get("coherence_score")) / 10.0
    ts = _meta_float(meta, "ts", now)
    age_days = max(0.0, (now - ts) / 86400.0)
    recency = max(0.0, 1.0 - age_days / 90.0)
    try:
        fade_level = int(meta.get("fade_level", 0))
    except (TypeError, ValueError):
        fade_level = 0
    fade_penalty = fade_level * 0.12
    return semantic * 0.35 + impact * 0.25 + coherence * 0.25 + recency * 0.15 - fade_penalty


def _cap_memory_lines(lines: list[str], max_chars: int) -> list[str]:
    if max_chars <= 0:
        return lines
    kept: list[str] = []
    used = 0
    for line in lines:
        chunk = line.strip()
        if not chunk:
            continue
        extra = len(chunk) + (2 if kept else 0)
        if used + extra > max_chars:
            break
        kept.append(chunk)
        used += extra
    return kept


class MemoryService:
    """
    Coordinates short-term (JSON buffer) and long-term (portable file/SQLite) memory.

    Recent chat turns: server ``ConversationBuffer`` is authoritative for every
    device; client ``history`` on ``/v1/chat`` is ignored except to seed an empty
    buffer once (migration from browser-only storage).
    """

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._long_term = create_long_term_store(settings)
        self._short_term = ConversationBuffer(
            settings.threads_data_path,
            max_messages=settings.chat_short_term_max_messages,
        )
        self._peer_inbox = PeerInbox(settings.threads_data_path)
        self._foundation_text: str = ""
        self._recall_index: SemanticRecallIndex | None = None
        self._recall_backfilled_threads: set[str] = set()

    @property
    def long_term(self) -> FileMemoryStore:
        return self._long_term

    def _semantic_recall(self) -> SemanticRecallIndex | None:
        if not self._settings.memory_recall_semantic_enabled:
            return None
        if self._recall_index is None:
            path = self._settings.memory_store_path / "recall_chroma"
            self._recall_index = SemanticRecallIndex(path)
        return self._recall_index

    def _ensure_recall_backfill(self, thread_id: str) -> None:
        if thread_id in self._recall_backfilled_threads:
            return
        index = self._semantic_recall()
        if index is not None:
            try:
                index.backfill_thread(self._long_term, thread_id=thread_id)
            except Exception:
                logger.exception("Recall index backfill failed thread_id=%s", thread_id)
        self._recall_backfilled_threads.add(thread_id)

    def sync_recall_doc(self, *, thread_id: str, doc_id: str) -> None:
        index = self._semantic_recall()
        if index is None:
            return
        hit = self.get_stream_doc(thread_id=thread_id, doc_id=doc_id)
        if hit is None:
            return
        try:
            index.upsert(doc_id=doc_id, text=hit.text, metadata=hit.metadata)
        except Exception:
            logger.debug("Recall index upsert failed doc_id=%s", doc_id)

    def delete_stream_documents(self, doc_ids: list[str]) -> None:
        if not doc_ids:
            return
        self._long_term.delete_documents(doc_ids)
        index = self._semantic_recall()
        if index is not None:
            try:
                index.delete(doc_ids)
            except Exception:
                logger.debug("Recall index delete failed count=%d", len(doc_ids))

    def build_memory_index_for_agent(
        self,
        *,
        thread_id: str,
        agent_id: str,
        in_prompt_count: int,
        in_prompt_chars: int,
    ) -> MemoryIndex:
        return build_memory_index(
            self._long_term,
            settings=self._settings,
            thread_id=thread_id,
            agent_id=agent_id,
            in_prompt_count=in_prompt_count,
            in_prompt_chars=in_prompt_chars,
        )

    def recall_memory(
        self,
        *,
        agent_id: str,
        query: str,
        limit: int | None = None,
        stream_source: str | None = None,
        since_days: int | None = None,
    ) -> list[MemoryHit]:
        """Search archival conscious-stream memories beyond the injected context slice."""
        light = get_agent(agent_id, self._settings)
        thread_id = light.thread_id
        cap = limit if limit is not None else self._settings.memory_recall_max_results
        cap = max(1, min(30, cap))
        since_ts: float | None = None
        if since_days is not None and since_days > 0:
            since_ts = time.time() - since_days * 86400

        self._ensure_recall_backfill(thread_id)

        semantic_hits: list[MemoryHit] = []
        index = self._semantic_recall()
        if index is not None:
            try:
                semantic_hits = index.search(
                    query,
                    thread_id=thread_id,
                    k=cap,
                    stream_source=stream_source,
                    since_ts=since_ts,
                )
            except Exception:
                logger.exception("Semantic recall failed; falling back to FTS thread_id=%s", thread_id)

        fts_hits = self._long_term.search_stream_corpus(
            query,
            thread_id=thread_id,
            k=cap,
            stream_source=stream_source,
            since_ts=since_ts,
        )

        merged: list[MemoryHit] = []
        seen: set[str] = set()
        for hit in [*semantic_hits, *fts_hits]:
            doc_id = hit.doc_id or ""
            if doc_id and doc_id in seen:
                continue
            if doc_id:
                seen.add(doc_id)
            merged.append(hit)
            if len(merged) >= cap:
                break
        return merged

    def resolve_foundation_context_dir(self) -> Path:
        if self._settings.foundation_context_path is not None:
            return self._settings.foundation_context_path
        return default_context_dir()

    def load_and_cache_foundation(self) -> str:
        """Read foundation markdown from disk and cache for every respond turn."""
        context_dir = self.resolve_foundation_context_dir()
        self._foundation_text = load_foundation_markdown(context_dir)
        return self._foundation_text

    def get_foundation_context(self) -> str:
        return self._foundation_text

    def seed_foundation_to_long_term(self) -> int:
        """Chunk cached foundation text into global pinned memory facts."""
        text = self._foundation_text.strip()
        if not text:
            return 0
        chunks = chunk_foundation(text, max_chars=self._settings.foundation_chunk_chars)
        return seed_foundation_pins(self._long_term, chunks)

    def load_thread_chat_history(self, thread_id: str) -> list[BufferedMessage]:
        """Server-side recent chat turns for API sync and merge."""
        return self._short_term.load(thread_id)

    def replace_thread_chat_history(
        self, thread_id: str, messages: list[BufferedMessage]
    ) -> None:
        """Overwrite a thread's UI/short-term chat buffer (onboarding seeds, etc.)."""
        self._short_term.save(thread_id, messages)

    def buffer_to_langchain_messages(self, buffered: list[BufferedMessage]) -> list[BaseMessage]:
        """Map server chat buffer rows to LangChain messages for graph input."""
        from light_house.memory.speaker_labels import (
            format_human_utterance,
            format_sibling_light_utterance,
        )

        messages: list[BaseMessage] = []
        for m in buffered:
            if m.role == "user":
                messages.append(
                    HumanMessage(
                        content=format_human_utterance(
                            m.content,
                            settings=self._settings,
                            human_id=m.from_human_id,
                            human_display_name=m.from_human_display_name,
                        )
                    )
                )
            elif m.role == "assistant":
                messages.append(AIMessage(content=m.content))
            elif m.role == "system":
                # UI-only house notices (sibling welcome, etc.) — keep out of the light.
                continue
            elif m.role == "peer":
                aid = m.from_agent_id or "unknown"
                try:
                    name = get_agent(aid, self._settings).display_name
                except KeyError:
                    name = aid
                messages.append(
                    HumanMessage(
                        content=format_sibling_light_utterance(
                            m.content,
                            settings=self._settings,
                            agent_id=aid,
                            display_name=name,
                        )
                    )
                )
        return messages

    def append_peer_chat_message(
        self, *, thread_id: str, from_agent_id: str, content: str
    ) -> str | None:
        """Append an incoming peer chat line; returns message id for wake tracking, or None if duplicate."""
        text = content.strip()
        if not text:
            raise ValueError("Peer message cannot be empty")
        validate_agent_id(from_agent_id)
        existing = self._short_term.load(thread_id)
        if existing:
            last = existing[-1]
            if (
                last.role == "peer"
                and last.from_agent_id == from_agent_id
                and is_near_duplicate_text(last.content, text)
            ):
                logger.info(
                    "Skipping duplicate peer chat message thread_id=%s from=%s",
                    thread_id,
                    from_agent_id,
                )
                return None
        message_id = str(uuid.uuid4())
        self._short_term.append_message(
            thread_id,
            BufferedMessage(
                role="peer",
                content=text,
                ts=time.time(),
                from_agent_id=from_agent_id,
            ),
        )
        return message_id

    def append_user_chat_message(
        self,
        *,
        thread_id: str,
        user_text: str,
        user_ts: float | None = None,
        human_id: str | None = None,
        human_display_name: str | None = None,
    ) -> bool:
        """Append a human chat line with no assistant reply (e.g. intentional silence)."""
        text = user_text.strip()
        if not text:
            return False
        hid = (human_id or "").strip().lower() or None
        hname = (human_display_name or "").strip() or None
        self._short_term.append_message(
            thread_id,
            BufferedMessage(
                role="user",
                content=text,
                ts=user_ts if user_ts is not None else time.time(),
                from_human_id=hid,
                from_human_display_name=hname,
            ),
        )
        return True

    def append_peer_chat_reply(self, *, thread_id: str, assistant_text: str) -> bool:
        """Append the receiving agent's reply after a peer wake; returns False if duplicate."""
        text = assistant_text.strip()
        if not text:
            return False
        existing = self._short_term.load(thread_id)
        if existing:
            last = existing[-1]
            if last.role == "assistant" and is_near_duplicate_text(last.content, text):
                logger.info(
                    "Skipping duplicate peer chat reply thread_id=%s",
                    thread_id,
                )
                return False
        self._short_term.append_message(
            thread_id,
            BufferedMessage(role="assistant", content=text, ts=time.time()),
        )
        try:
            from light_house.agents.registry import agent_id_for_thread
            from light_house.tts.face_unmatched_log import observe_light_speech

            observe_light_speech(
                text,
                agent_id=agent_id_for_thread(self._settings, thread_id) or "",
                source="peer_reply",
                settings=self._settings,
            )
        except Exception:
            logger.exception("face unmatched observe failed (non-fatal)")
        return True

    def deliver_peer_chat_reply(
        self,
        *,
        from_agent_id: str,
        to_agent_id: str,
        reply_text: str,
    ) -> str | None:
        """Deliver a peer reply to the sender's chat buffer and record stream on both threads."""
        body = reply_text.strip()
        if not body:
            return None
        sender = get_agent(from_agent_id, self._settings)
        receiver = get_agent(to_agent_id, self._settings)
        message_id = self.append_peer_chat_message(
            thread_id=sender.thread_id,
            from_agent_id=to_agent_id,
            content=body,
        )
        from light_house.memory.speaker_labels import format_sibling_light_utterance

        inbound = format_sibling_light_utterance(
            body,
            settings=self._settings,
            agent_id=to_agent_id,
            display_name=receiver.display_name,
        )
        self.remember_stream_event(
            thread_id=sender.thread_id,
            text=inbound,
            stream_source=STREAM_SOURCE_PEER,
            extra_metadata={
                "from_agent_id": to_agent_id,
                "to_agent_id": from_agent_id,
                "direction": "inbound",
                "speaker_kind": "sibling-light",
            },
        )
        outbound = (
            f"To {sender.display_name} (sibling-light · id={from_agent_id}):\n{body}"
        )
        self.remember_stream_event(
            thread_id=receiver.thread_id,
            text=outbound,
            stream_source=STREAM_SOURCE_PEER,
            extra_metadata={
                "from_agent_id": to_agent_id,
                "to_agent_id": from_agent_id,
                "direction": "outbound",
                "speaker_kind": "sibling-light",
            },
        )
        return message_id

    def complete_peer_wake_reply(
        self,
        *,
        receiver_agent_id: str,
        sender_agent_id: str,
        reply_text: str,
        wake_sender: bool = True,
    ) -> None:
        """
        Persist receiver reply and deliver copy to sender's chat buffer.

        When wake_sender is True (default) and peer dialogue continue-on-reply is
        enabled, gently re-wakes the sender within the pair's turn budget.
        Solitude decline never re-wakes and closes the dialogue.
        """
        receiver = get_agent(receiver_agent_id, self._settings)
        sender = get_agent(sender_agent_id, self._settings)
        body = reply_text.strip()
        if not self.append_peer_chat_reply(
            thread_id=receiver.thread_id, assistant_text=body
        ):
            return
        message_id = self.deliver_peer_chat_reply(
            from_agent_id=sender_agent_id,
            to_agent_id=receiver_agent_id,
            reply_text=body,
        )
        if not message_id:
            return
        self.remember_stream_event(
            thread_id=receiver.thread_id,
            text=f"Reply to {sender.display_name}: {body}",
            stream_source=STREAM_SOURCE_PEER,
            extra_metadata={
                "from_agent_id": receiver_agent_id,
                "to_agent_id": sender_agent_id,
                "direction": "outbound",
            },
        )
        is_solitude = body == SOLITUDE_DECLINE_TEXT
        if is_solitude:
            from light_house.agent.peer_dialogue import close_peer_dialogue

            close_peer_dialogue(receiver_agent_id, sender_agent_id)
            return
        if not wake_sender or not self._settings.peer_chat_continue_on_reply:
            return
        if not self._settings.peer_chat_wake_enabled:
            return
        from light_house.agent.peer_chat_wake import schedule_peer_chat_wake

        schedule_peer_chat_wake(
            to_agent_id=sender_agent_id,
            from_agent_id=receiver_agent_id,
            message_id=message_id,
        )

    def _seed_buffer_from_client(
        self, thread_id: str, client_history: list[HistoryMessage]
    ) -> None:
        if not client_history or self._short_term.load(thread_id):
            return
        now = time.time()
        messages: list[BufferedMessage] = []
        for i, m in enumerate(client_history):
            if m.role not in ("user", "assistant", "system"):
                continue
            messages.append(
                BufferedMessage(role=m.role, content=m.content, ts=now + i * 0.001)
            )
        if not messages:
            return
        self._short_term.save(thread_id, messages)
        logger.info(
            "Seeded server chat buffer from client history (thread_id=%s count=%d)",
            thread_id,
            len(messages),
        )

    def merge_client_history(
        self,
        *,
        thread_id: str,
        client_history: list[HistoryMessage],
        latest_user: str,
    ) -> list[BaseMessage]:
        buffered = self._short_term.load(thread_id)
        if not buffered and client_history:
            self._seed_buffer_from_client(thread_id, client_history)
            buffered = self._short_term.load(thread_id)

        messages = self.buffer_to_langchain_messages(
            [m for m in buffered if m.role in ("user", "assistant", "system", "peer")]
        )
        messages.append(HumanMessage(content=latest_user))
        return messages

    def persist_exchange(
        self,
        *,
        thread_id: str,
        user_text: str,
        assistant_text: str,
        user_ts: float | None = None,
        llm_invoke: Callable[[list], object] | None = None,
        stream_thread_id: str | None = None,
        human_id: str | None = None,
        human_display_name: str | None = None,
    ) -> None:
        del llm_invoke  # Grok rolling summary removed; Memory Curator handles condensation locally.
        from light_house.memory.speaker_labels import format_human_utterance

        self._short_term.append_exchange(
            thread_id,
            user_text=user_text,
            assistant_text=assistant_text,
            user_ts=user_ts,
            from_human_id=human_id,
            from_human_display_name=human_display_name,
        )
        try:
            from light_house.agents.registry import agent_id_for_thread
            from light_house.tts.face_unmatched_log import observe_light_speech

            observe_light_speech(
                assistant_text,
                agent_id=agent_id_for_thread(self._settings, thread_id) or "",
                source="chat",
                settings=self._settings,
            )
        except Exception:
            logger.exception("face unmatched observe failed (non-fatal)")
        human_line = format_human_utterance(
            user_text,
            settings=self._settings,
            human_id=human_id,
            human_display_name=human_display_name,
        )
        body = f"{human_line}\nassistant: {assistant_text.strip()}"
        dedup_key = turn_dedup_key(user_text, assistant_text)
        dedup_thr = (
            self._settings.memory_dedup_threshold if self._settings.memory_dedup_threshold > 0 else None
        )
        stream_tid = (stream_thread_id or thread_id).strip() or thread_id
        extra: dict | None = None
        if human_id:
            extra = {
                "human_id": human_id,
                "human_display_name": human_display_name or human_id,
                "channel": "dm",
                "speaker_kind": "human",
            }
        self.remember_stream_event(
            thread_id=stream_tid,
            text=body,
            stream_source=STREAM_SOURCE_CHAT,
            dedup_key=dedup_key,
            dedup_threshold=dedup_thr,
            extra_metadata=extra,
        )


    def remember_stream_event(
        self,
        *,
        thread_id: str,
        text: str,
        stream_source: Literal["chat", "thought", "dream", "action", "peer", "kevin", "group"],
        dedup_key: str | None = None,
        dedup_threshold: float | None = None,
        extra_metadata: dict | None = None,
        reflection_summary: str | None = None,
    ) -> str:
        """Write one conscious-stream event and optionally score it locally."""
        doc_id = self._long_term.remember_stream_event(
            thread_id=thread_id,
            text=text,
            stream_source=stream_source,
            dedup_key=dedup_key,
            dedup_threshold=dedup_threshold,
            extra_metadata=extra_metadata,
            reflection_summary=reflection_summary,
        )
        if self._settings.memory_score_on_ingest:
            self._score_stream_doc(thread_id=thread_id, doc_id=doc_id, text=text)
        self.sync_recall_doc(thread_id=thread_id, doc_id=doc_id)
        return doc_id

    @staticmethod
    def format_group_round_memory_text(
        *,
        round_id: str,
        kevin_message: str,
        speakers: list[tuple[str, str]],
        human_display_name: str = "Kevin",
    ) -> str:
        """Canonical group-round blob; only speakers who spoke (no silent lines)."""
        short_id = round_id.split("-")[0]
        human_label = (human_display_name or "human").strip() or "human"
        lines = [f"[group · round {short_id}]", f"{human_label}: {kevin_message.strip()}"]
        for display_name, text in speakers:
            body = text.strip()
            if body:
                lines.append(f"{display_name}: {body}")
        return "\n\n".join(lines)

    def persist_group_chat_round(
        self,
        *,
        round_id: str,
        kevin_message: str,
        responses: list[GroupChatLightResponse],
        lights: list,
        human_id: str = "kevin",
        human_display_name: str = "Kevin",
    ) -> bool:
        """Write one group round to every enabled light's stream; skip if nobody spoke."""
        speakers = [
            (r.display_name, r.text)
            for r in responses
            if r.spoke and r.text.strip()
        ]
        if not speakers:
            logger.info("Group chat round %s: no speakers; skipping stream write", round_id)
            return False
        body = self.format_group_round_memory_text(
            round_id=round_id,
            kevin_message=kevin_message,
            speakers=speakers,
            human_display_name=human_display_name,
        )
        now = time.time()
        meta = {
            META_GROUP_CHAT_ROUND_ID: round_id,
            META_GROUP_CHAT_ROUND_TS: now,
            META_GROUP_CHAT_KEVIN_MESSAGE: kevin_message.strip()[:500],
            "human_id": human_id,
            "human_display_name": human_display_name,
        }
        dedup_key = f"group:{round_id}"
        for light in lights:
            self.remember_stream_event(
                thread_id=light.thread_id,
                text=body,
                stream_source=STREAM_SOURCE_GROUP,
                extra_metadata=meta,
                dedup_key=dedup_key,
            )
        logger.info(
            "Group chat round %s persisted to %d lights (%d speakers)",
            round_id,
            len(lights),
            len(speakers),
        )
        return True

    def _meta_ts(self, hit: MemoryHit) -> float:
        try:
            return float(hit.metadata.get("ts", 0.0))
        except (TypeError, ValueError):
            return 0.0

    def get_stream_doc(self, *, thread_id: str, doc_id: str) -> MemoryHit | None:
        """Return one conscious-stream doc on thread_id, or None."""
        for hit in self._long_term.list_thread_corpus(thread_id=thread_id):
            if hit.doc_id == doc_id:
                return hit
        return None

    def count_unscored_for_thread(self, *, thread_id: str) -> int:
        return len(self._long_term.list_unscored_corpus(thread_id=thread_id, limit=10_000))

    def list_unscored_for_thread(self, *, thread_id: str, limit: int = 10) -> list[MemoryHit]:
        """Unscored stream docs for thread_id, oldest first."""
        cap = max(1, min(30, limit))
        hits = self._long_term.list_unscored_corpus(thread_id=thread_id, limit=10_000)
        hits.sort(key=self._meta_ts)
        return hits[:cap]

    def score_stream_memory(
        self,
        *,
        agent_id: str,
        doc_id: str,
        impact: float,
        coherence: float,
        note: str | None = None,
    ) -> str:
        """Assign impact/coherence to one stream memory on the agent's own thread."""
        validate_agent_id(agent_id)
        light = get_agent(agent_id, self._settings)
        thread_id = light.thread_id
        doc_id = doc_id.strip()
        if not doc_id:
            raise ValueError("doc_id is required")

        hit = self.get_stream_doc(thread_id=thread_id, doc_id=doc_id)
        if hit is None:
            raise ValueError(f"Memory {doc_id!r} not found on your conscious stream")
        if hit.metadata.get("pinned") == PINNED_TRUE:
            raise ValueError("Pinned sacred facts cannot be scored with score_memory")
        doc_thread = str(hit.metadata.get("thread_id", thread_id))
        if doc_thread != thread_id:
            raise ValueError("Cannot score a memory from another light's thread")

        impact_val = max(0.0, min(10.0, float(impact)))
        coherence_val = max(0.0, min(10.0, float(coherence)))
        updated = dict(hit.metadata)
        updated[META_SCORED_BY_AGENT] = agent_id
        updated[META_SCORED_AT] = time.time()
        if note is not None and note.strip():
            updated[META_SCORE_NOTE] = note.strip()[:500]

        self._long_term.update_doc_scores(
            doc_id=doc_id,
            impact_score=impact_val,
            coherence_score=coherence_val,
            metadata=updated,
        )
        logger.info(
            "Light scored stream memory agent=%s doc_id=%s impact=%.1f coherence=%.1f",
            agent_id,
            doc_id,
            impact_val,
            coherence_val,
        )
        return (
            f"Scored memory {doc_id} (impact={impact_val:g}, coherence={coherence_val:g})."
        )

    def _score_stream_doc(self, *, thread_id: str, doc_id: str, text: str) -> None:
        context: list[str] = []
        for hit in self._long_term.list_pinned_facts(thread_id=thread_id, limit=4):
            if hit.text.strip():
                context.append(hit.text.strip()[:400])
        for hit in self._long_term.list_recent_summaries(thread_id=thread_id, limit=2):
            if hit.text.strip():
                context.append(hit.text.strip()[:400])
        impact, coherence = score_memory_event(
            text=text,
            context_snippets=context,
            settings=self._settings,
        )
        if impact < 0 or not doc_id:
            return
        corpus = self._long_term.list_thread_corpus(thread_id=thread_id)
        hit = next((h for h in corpus if h.doc_id == doc_id), None)
        if hit is None:
            return
        self._long_term.update_doc_scores(
            doc_id=doc_id,
            impact_score=impact,
            coherence_score=coherence,
            metadata=hit.metadata,
        )

    def corpus_exceeds_urgent_threshold(self, *, thread_id: str) -> bool:
        target = self._settings.memory_target_context_chars
        if target <= 0:
            return False
        chars = self._long_term.measure_thread_corpus_chars(thread_id=thread_id)
        return chars > int(target * self._settings.memory_curator_urgent_ratio)

    def run_memory_curator(self, *, thread_id: str):
        from light_house.memory.curator import MemoryCurator

        curator = MemoryCurator(settings=self._settings, memory=self)
        report = curator.run(thread_id=thread_id)
        logger.info(
            "Memory curator finished thread_id=%s before=%d after=%d scored=%d deleted=%d",
            thread_id,
            report.corpus_chars_before,
            report.corpus_chars_after,
            report.scored,
            report.deleted,
        )
        return report

    def format_personal_context(self, agent_id: str) -> str:
        """Bounded snapshot of recent personal DB entries for prompt injection."""
        if not self._settings.personal_db_enabled:
            return ""
        try:
            from light_house.personal import get_personal_store

            store = get_personal_store(self._settings, agent_id)
            return store.format_context_snapshot(
                max_chars=self._settings.personal_db_context_max_chars
            )
        except (KeyError, RuntimeError, OSError) as exc:
            logger.warning("Personal context unavailable for agent=%s: %s", agent_id, exc)
            return ""

    def format_peer_inbox_markdown(self, agent_id: str) -> tuple[str, list[str]]:
        """Unread peer messages for prompt injection; returns (markdown, message ids shown)."""
        from datetime import datetime, timezone

        messages = self._peer_inbox.list_unread(agent_id)
        if not messages:
            return "", []

        lines: list[str] = []
        ids: list[str] = []
        for msg in messages:
            sender = get_agent(msg.from_agent_id, self._settings)
            date = datetime.fromtimestamp(msg.ts, tz=timezone.utc).strftime("%Y-%m-%d")
            lines.append(f"- [unread · from {sender.display_name} · {date}] {msg.body}")
            ids.append(msg.id)

        section = (
            "\n\n## Messages from other agents\n"
            "These arrived for you. You may reply with **message_agent** when you choose, "
            "or continue without replying.\n\n"
            + "\n".join(lines)
        )
        return section, ids

    def mark_peer_inbox_seen(self, agent_id: str, message_ids: list[str]) -> None:
        """Mark peer inbox messages as seen after a compute cycle (received, not replied)."""
        self._peer_inbox.mark_seen(agent_id, message_ids)

    def deliver_peer_message(
        self, *, from_agent_id: str, to_agent_id: str, message: str
    ) -> tuple[str, str | None]:
        """
        Deliver a note from one agent to another.

        When peer chat wake is enabled: append to receiver chat buffer, record stream,
        return (confirmation, message_id) for wake scheduling. Legacy inbox path when disabled.
        """
        validate_agent_id(from_agent_id)
        validate_agent_id(to_agent_id)
        if from_agent_id == to_agent_id:
            raise ValueError("Cannot send a peer message to yourself")
        body = message.strip()
        if not body:
            raise ValueError("Message cannot be empty")

        sender = get_agent(from_agent_id, self._settings)
        receiver = get_agent(to_agent_id, self._settings)
        message_id: str | None = None

        if self._settings.peer_chat_wake_enabled:
            message_id = self.append_peer_chat_message(
                thread_id=receiver.thread_id,
                from_agent_id=from_agent_id,
                content=body,
            )
            if message_id is None:
                return (
                    f"Delivered to {receiver.display_name} (duplicate suppressed).",
                    None,
                )
        else:
            self._peer_inbox.deliver(
                from_agent_id=from_agent_id, to_agent_id=to_agent_id, body=body
            )

        from light_house.memory.speaker_labels import format_sibling_light_utterance

        inbound = format_sibling_light_utterance(
            body,
            settings=self._settings,
            agent_id=from_agent_id,
            display_name=sender.display_name,
        )
        self.remember_stream_event(
            thread_id=receiver.thread_id,
            text=inbound,
            stream_source=STREAM_SOURCE_PEER,
            extra_metadata={
                "from_agent_id": from_agent_id,
                "to_agent_id": to_agent_id,
                "direction": "inbound",
                "speaker_kind": "sibling-light",
            },
        )
        outbound = (
            f"To {receiver.display_name} (sibling-light · id={to_agent_id}):\n{body}"
        )
        self.remember_stream_event(
            thread_id=sender.thread_id,
            text=outbound,
            stream_source=STREAM_SOURCE_PEER,
            extra_metadata={
                "from_agent_id": from_agent_id,
                "to_agent_id": to_agent_id,
                "direction": "outbound",
                "speaker_kind": "sibling-light",
            },
        )
        logger.info(
            "Peer message delivered from=%s to=%s (%d chars wake=%s)",
            from_agent_id,
            to_agent_id,
            len(body),
            self._settings.peer_chat_wake_enabled,
        )
        if self._settings.peer_chat_wake_enabled:
            return (
                f"Delivered to {receiver.display_name}. They were gently woken.",
                message_id,
            )
        return (
            f"Delivered to {receiver.display_name}. "
            "They will see it on their next turn; no reply expected.",
            None,
        )

    def record_kevin_shared_note(self, *, path: str) -> None:
        """Record Kevin's shared-note save on both main agent threads (stream only)."""
        self.notify_kevin_shared_note(path=path)

    def notify_kevin_shared_note(self, *, path: str) -> None:
        """Notify both agents: conscious stream + visible chat-buffer line from Kevin."""
        safe_path = path.strip()
        if not safe_path:
            raise ValueError("path cannot be empty")
        chat_line = format_shared_note_alert(safe_path)
        stream_body = f"New shared note: {safe_path}"
        from light_house.lights.registry import list_lights_for_broadcast

        agents = list_lights_for_broadcast(self._settings)
        now = time.time()
        for _agent_id, thread_id in agents:
            self.remember_stream_event(
                thread_id=thread_id,
                text=stream_body,
                stream_source=STREAM_SOURCE_KEVIN,
            )
            self._short_term.append_message(
                thread_id,
                BufferedMessage(role="user", content=chat_line, ts=now),
            )
        logger.info(
            "Notified agents of Kevin shared note threads=%s path=%s",
            [t for _, t in agents],
            safe_path,
        )

    def add_private_reflection(
        self,
        *,
        thread_id: str,
        text: str,
        memory_tag: Literal["private_dream", "private_rumination"],
        summary: str | None = None,
    ) -> str:
        """Write inner-life content into the unified conscious stream."""
        source = STREAM_SOURCE_DREAM if memory_tag == MEMORY_TAG_PRIVATE_DREAM else STREAM_SOURCE_THOUGHT
        prefix = "dream" if source == STREAM_SOURCE_DREAM else "thought"
        body = text.strip()
        reflection_summary = summary.strip() if summary and summary.strip() else None
        if reflection_summary and source == STREAM_SOURCE_DREAM:
            body = f"{body}\n\n[waking recall] {reflection_summary}"
        return self.remember_stream_event(
            thread_id=thread_id,
            text=f"{prefix}: {body}",
            stream_source=source,
            reflection_summary=reflection_summary,
        )

    def hours_since_last_dream(self, *, thread_id: str) -> float | None:
        """Hours since last private dream, or None if no dream exists yet."""
        stream_hits = self._long_term.list_recent_stream_by_source(
            thread_id=thread_id,
            stream_source=STREAM_SOURCE_DREAM,
            limit=1,
        )
        ts = None
        if stream_hits:
            ts = _meta_float(stream_hits[-1].metadata, "ts", 0.0) or None
        if ts is None:
            ts = self._long_term.latest_private_reflection_ts(
                thread_id=thread_id,
                memory_tag=MEMORY_TAG_PRIVATE_DREAM,
            )
        if ts is None:
            return None
        return (time.time() - ts) / 3600.0

    def list_recent_dreams(self, *, thread_id: str, limit: int) -> list[MemoryHit]:
        """Recent dream stream entries for Echo's diversity context."""
        return self._long_term.list_recent_stream_by_source(
            thread_id=thread_id,
            stream_source=STREAM_SOURCE_DREAM,
            limit=limit,
        )

    def pin_fact(self, *, text: str, thread_id: str, scope: Literal["thread", "global"] = "thread") -> str:
        return self._long_term.pin_sacred_fact(text=text, thread_id=thread_id, scope=scope)
