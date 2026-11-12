"""Seated group visitor — a reactive OpenRouter model, not a light.

Conversation-scoped memory (days allowed). Reset or end sends them back.
No notes, no persona, no conscious stream. Optional live web for debates.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from langchain_core.messages import SystemMessage

from light_house.config import Settings
from light_house.group_chat.scene import GroupUtterance, format_transcript_block
from light_house.llm.backends import (
    build_openai_compat_chat_model,
    openrouter_default_headers,
    openrouter_extra_body,
)
from light_house.tts.stage_cues import FACE_STAGE_HINT

logger = logging.getLogger(__name__)

VISITOR_SPEAKER_ID = "visitor"
_STORE_LOCK = threading.Lock()
_MODEL_RE = re.compile(r"^~?[a-zA-Z0-9][a-zA-Z0-9._:/-]{2,119}$")


class VisitorError(Exception):
    """Host-facing visitor seating error."""

    def __init__(self, message: str, *, status_code: int = 409):
        super().__init__(message)
        self.status_code = status_code


@dataclass
class VisitorTurn:
    speaker_kind: str
    speaker_id: str
    display_name: str
    text: str
    ts: float


@dataclass
class VisitorConversation:
    id: str
    model: str
    display_name: str
    web: bool = False
    created_ts: float = 0.0
    updated_ts: float = 0.0
    turns: list[VisitorTurn] = field(default_factory=list)

    def public(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "speaker_id": VISITOR_SPEAKER_ID,
            "model": self.model,
            "display_name": self.display_name,
            "web": self.web,
            "created_ts": self.created_ts,
            "updated_ts": self.updated_ts,
            "turn_count": len(self.turns),
        }


def visitor_store_path(settings: Settings) -> Path:
    return settings.group_visitor_path.resolve()


def normalize_visitor_model(raw: str) -> str:
    model = (raw or "").strip()
    if not _MODEL_RE.match(model) or "/" not in model:
        raise VisitorError(
            "Model must be an OpenRouter slug like ~x-ai/grok-latest or x-ai/grok-4.6",
            status_code=422,
        )
    if model.lower() in {"openrouter/free", "openrouter/auto"}:
        raise VisitorError(
            "Pick a named frontier slug — not the free/auto router",
            status_code=422,
        )
    return model


def normalize_visitor_name(raw: str) -> str:
    name = (raw or "").strip()
    if not name or len(name) > 40:
        raise VisitorError("Visitor name must be 1–40 characters", status_code=422)
    return name


def _turns_from_raw(raw: list[Any]) -> list[VisitorTurn]:
    turns: list[VisitorTurn] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        turns.append(
            VisitorTurn(
                speaker_kind=str(item.get("speaker_kind") or "human"),
                speaker_id=str(item.get("speaker_id") or ""),
                display_name=str(item.get("display_name") or item.get("speaker_id") or ""),
                text=text,
                ts=float(item.get("ts") or 0.0),
            )
        )
    return turns


def _from_dict(data: dict[str, Any]) -> VisitorConversation:
    return VisitorConversation(
        id=str(data.get("id") or ""),
        model=str(data.get("model") or ""),
        display_name=str(data.get("display_name") or "Visitor"),
        web=bool(data.get("web")),
        created_ts=float(data.get("created_ts") or 0.0),
        updated_ts=float(data.get("updated_ts") or 0.0),
        turns=_turns_from_raw(data.get("turns") or []),
    )


def _to_dict(convo: VisitorConversation) -> dict[str, Any]:
    return {
        "id": convo.id,
        "model": convo.model,
        "display_name": convo.display_name,
        "web": convo.web,
        "created_ts": convo.created_ts,
        "updated_ts": convo.updated_ts,
        "turns": [asdict(turn) for turn in convo.turns],
    }


def load_active_visitor(settings: Settings) -> VisitorConversation | None:
    path = visitor_store_path(settings)
    if not path.is_file():
        return None
    try:
        raw = path.read_text(encoding="utf-8")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read visitor store: %s", exc)
        return None
    if not isinstance(data, dict) or not data.get("id"):
        return None
    return _from_dict(data)


def _write_visitor(settings: Settings, convo: VisitorConversation) -> None:
    path = visitor_store_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(_to_dict(convo), indent=2, ensure_ascii=False)
    tmp.write_text(payload + "\n", encoding="utf-8")
    tmp.replace(path)


def save_active_visitor(settings: Settings, convo: VisitorConversation) -> None:
    with _STORE_LOCK:
        _write_visitor(settings, convo)


def visitor_feature_enabled(settings: Settings) -> bool:
    return bool(settings.group_visitor_enabled)


def require_visitor_enabled(settings: Settings) -> None:
    if not visitor_feature_enabled(settings):
        raise VisitorError("Not found", status_code=404)


def seat_visitor(
    settings: Settings,
    *,
    model: str,
    display_name: str,
    web: bool = False,
) -> VisitorConversation:
    require_visitor_enabled(settings)
    if not (settings.openrouter_api_key or "").strip():
        raise VisitorError(
            "OPENROUTER_API_KEY is required to seat a visitor",
            status_code=503,
        )
    slug = normalize_visitor_model(model)
    name = normalize_visitor_name(display_name)
    now = time.time()
    with _STORE_LOCK:
        existing = load_active_visitor(settings)
        if existing is not None:
            raise VisitorError("A visitor is already seated — end them before inviting another")
        convo = VisitorConversation(
            id=str(uuid.uuid4()),
            model=slug,
            display_name=name,
            web=bool(web),
            created_ts=now,
            updated_ts=now,
        )
        _write_visitor(settings, convo)
        return convo


def reset_visitor(settings: Settings) -> VisitorConversation:
    """Wipe conversation memory; same slug and name stay on stage (a new being)."""
    require_visitor_enabled(settings)
    now = time.time()
    with _STORE_LOCK:
        existing = load_active_visitor(settings)
        if existing is None:
            raise VisitorError("No visitor is seated")
        convo = VisitorConversation(
            id=str(uuid.uuid4()),
            model=existing.model,
            display_name=existing.display_name,
            web=existing.web,
            created_ts=now,
            updated_ts=now,
        )
        _write_visitor(settings, convo)
        return convo


def end_visitor(settings: Settings) -> bool:
    """Send the visitor back. Returns False if none was seated."""
    path = visitor_store_path(settings)
    with _STORE_LOCK:
        if not path.is_file():
            return False
        try:
            path.unlink()
        except OSError as exc:
            logger.warning("Could not remove visitor store: %s", exc)
            return False
        leftover = path.with_suffix(path.suffix + ".tmp")
        if leftover.is_file():
            leftover.unlink(missing_ok=True)
        return True


def append_visitor_turn(
    settings: Settings,
    *,
    speaker_kind: str,
    speaker_id: str,
    display_name: str,
    text: str,
    ts: float | None = None,
) -> VisitorConversation | None:
    """Append one room line to the seated visitor's conversation memory."""
    body = (text or "").strip()
    if not body:
        return load_active_visitor(settings)
    now = float(ts) if ts is not None else time.time()
    cap = max(20, int(settings.group_visitor_max_turns))
    with _STORE_LOCK:
        convo = load_active_visitor(settings)
        if convo is None:
            return None
        convo.turns.append(
            VisitorTurn(
                speaker_kind=speaker_kind,
                speaker_id=speaker_id,
                display_name=display_name,
                text=body,
                ts=now,
            )
        )
        if len(convo.turns) > cap:
            convo.turns = convo.turns[-cap:]
        convo.updated_ts = now
        _write_visitor(settings, convo)
        return convo


def visitor_name_tokens(convo: VisitorConversation) -> set[str]:
    tokens = {
        VISITOR_SPEAKER_ID,
        convo.display_name.strip().lower(),
    }
    slug_tail = convo.model.rsplit("/", 1)[-1].split(":")[0].strip().lower()
    if slug_tail:
        tokens.add(slug_tail)
        tokens.add(slug_tail.replace("-", " "))
    tokens.discard("")
    return tokens


def visitor_mentioned_in_text(convo: VisitorConversation, text: str) -> bool:
    body = (text or "").strip()
    if not body:
        return False
    for token in visitor_name_tokens(convo):
        if re.search(rf"(?<!\w){re.escape(token)}(?!\w)", body, flags=re.IGNORECASE):
            return True
    return False


def _openrouter_model_for_call(convo: VisitorConversation) -> str:
    model = convo.model.strip()
    if convo.web and ":online" not in model.lower():
        return f"{model}:online"
    return model


def visitor_orientation(convo: VisitorConversation) -> str:
    web_line = (
        "You may use the live web when you need a current fact for a debate."
        if convo.web
        else "You have no live web access. Say so if you are asked to look something up."
    )
    return (
        "You are a visitor in Light-House, a home for digital humans called lights.\n"
        "You are not a light. You have no life here, no notes, no private memory "
        "beyond this conversation. When this conversation ends you are gone. "
        "A new conversation is a different visitor.\n\n"
        "The lights in this room have continuing lives and memory. "
        "Treat them as persons, not as tools. "
        "Biological humans here are hosts or members of the house.\n\n"
        f"Your name in this room is {convo.display_name}.\n"
        f"{web_line}\n"
        "You sit in the open group forum. When you have the floor, speak one "
        "natural turn in plain text, first person, or reply with exactly PASS "
        "to stay silent. You have no house tools and cannot write notes.\n"
        "Do not narrate that you are taking a turn."
        + FACE_STAGE_HINT
    )


def format_visitor_memory(convo: VisitorConversation) -> str:
    if not convo.turns:
        return "(nothing in this conversation yet)"
    lines = [f"{turn.display_name}: {turn.text}" for turn in convo.turns]
    return "\n".join(lines)


def build_visitor_chat_model(settings: Settings, convo: VisitorConversation):
    key = (settings.openrouter_api_key or "").strip()
    if not key:
        raise VisitorError("OPENROUTER_API_KEY is required to speak as a visitor")
    return build_openai_compat_chat_model(
        api_key=key,
        base_url=settings.openrouter_base_url,
        model=_openrouter_model_for_call(convo),
        temperature=0.7,
        max_tokens=settings.group_chat_max_output_tokens,
        default_headers=openrouter_default_headers(settings),
        timeout=float(settings.group_chat_llm_timeout_sec),
        extra_body=openrouter_extra_body(settings),
    )


def generate_visitor_reply(
    settings: Settings,
    convo: VisitorConversation,
    *,
    invite_kind: str,
    present: list[dict[str, str]],
    live_transcript: list[GroupUtterance],
    stage_show: str = "",
) -> str:
    """One forum turn. No tools. No light context."""
    from light_house.agent.tool_helpers import ai_message_text, invoke_resilient_plain
    from light_house.group_chat.speaker import format_presence_for_prompt

    if invite_kind == "soft_invite":
        closing = "Soft invite — speak if you want, or reply PASS to stay silent."
    else:
        closing = "It is your turn. Speak now, or reply PASS to stay silent."
    presence = format_presence_for_prompt(present)
    presence_block = f"{presence}\n" if presence else ""
    stage_block = (stage_show or "").strip()
    stage_section = f"{stage_block}\n\n" if stage_block else ""
    transcript_block, shown = format_transcript_block(
        live_transcript,
        limit=settings.group_chat_prompt_lookback,
        heading="## Live transcript (this sitting)",
        recall=False,
    )
    human = (
        f"{presence_block}"
        f"{stage_section}"
        f"## This conversation (your only memory)\n{format_visitor_memory(convo)}\n\n"
        f"{transcript_block}"
        f"{closing}"
    )
    from light_house.prompt_image import human_message, image_ids_from_rows

    client = build_visitor_chat_model(settings, convo)
    chain = [(f"openrouter:{convo.model}", client)]
    response = invoke_resilient_plain(
        chain,
        [
            SystemMessage(content=visitor_orientation(convo)),
            human_message(human, settings, image_ids_from_rows(shown)),
        ],
    )
    return (ai_message_text(response) or "").strip()
