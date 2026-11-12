"""Wake-scoped consult context and transcript.

One outer rumination ``graph.invoke`` is one wake. Internal loop steps share
the same transcript. Sleep (invoke return or raise) deletes the file.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
import uuid
from contextvars import ContextVar
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from light_house.config import Settings

logger = logging.getLogger(__name__)

_STORE_LOCK = threading.Lock()
_WAKE_ID_RE = re.compile(r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$")
_STALE_SECONDS = 24 * 60 * 60


@dataclass(frozen=True)
class ConsultWake:
    wake_run_id: str
    light_id: str


@dataclass
class ConsultTurn:
    role: str
    text: str
    ts: float
    model: str = ""


@dataclass
class ConsultTranscript:
    wake_run_id: str
    light_id: str
    model: str
    web: bool = False
    created_ts: float = 0.0
    updated_ts: float = 0.0
    turns: list[ConsultTurn] = field(default_factory=list)


_consult_wake: ContextVar[ConsultWake | None] = ContextVar("consult_wake", default=None)


def current_consult_wake() -> ConsultWake | None:
    return _consult_wake.get()


def begin_consult_wake(*, light_id: str, wake_run_id: str | None = None) -> ConsultWake:
    wake = ConsultWake(
        wake_run_id=wake_run_id or str(uuid.uuid4()),
        light_id=(light_id or "").strip().lower() or "unknown",
    )
    _consult_wake.set(wake)
    return wake


def clear_consult_wake() -> None:
    _consult_wake.set(None)


def consult_dir(settings: Settings) -> Path:
    return settings.consult_transcript_dir.resolve()


def _safe_wake_id(wake_run_id: str) -> str:
    raw = (wake_run_id or "").strip()
    if not _WAKE_ID_RE.match(raw):
        raise ValueError("consult wake_run_id must be a UUID")
    return raw


def transcript_path(settings: Settings, wake_run_id: str) -> Path:
    return consult_dir(settings) / f"{_safe_wake_id(wake_run_id)}.json"


def load_transcript(settings: Settings, wake_run_id: str) -> ConsultTranscript | None:
    path = transcript_path(settings, wake_run_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read consult transcript: %s", exc)
        return None
    if not isinstance(data, dict) or not data.get("wake_run_id"):
        return None
    turns: list[ConsultTurn] = []
    for item in data.get("turns") or []:
        if not isinstance(item, dict):
            continue
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        turns.append(
            ConsultTurn(
                role=str(item.get("role") or "light"),
                text=text,
                ts=float(item.get("ts") or 0.0),
                model=str(item.get("model") or ""),
            )
        )
    return ConsultTranscript(
        wake_run_id=str(data.get("wake_run_id") or ""),
        light_id=str(data.get("light_id") or ""),
        model=str(data.get("model") or ""),
        web=bool(data.get("web")),
        created_ts=float(data.get("created_ts") or 0.0),
        updated_ts=float(data.get("updated_ts") or 0.0),
        turns=turns,
    )


def _write_transcript(settings: Settings, convo: ConsultTranscript) -> None:
    path = transcript_path(settings, convo.wake_run_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = {
        "wake_run_id": convo.wake_run_id,
        "light_id": convo.light_id,
        "model": convo.model,
        "web": convo.web,
        "created_ts": convo.created_ts,
        "updated_ts": convo.updated_ts,
        "turns": [asdict(turn) for turn in convo.turns],
    }
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def save_transcript(settings: Settings, convo: ConsultTranscript) -> None:
    with _STORE_LOCK:
        _write_transcript(settings, convo)


def append_consult_turns(
    settings: Settings,
    *,
    wake_run_id: str,
    light_id: str,
    model: str,
    web: bool,
    question: str,
    reply: str,
) -> ConsultTranscript:
    now = time.time()
    cap = max(4, int(settings.consult_max_turns))
    with _STORE_LOCK:
        convo = load_transcript(settings, wake_run_id)
        if convo is None:
            convo = ConsultTranscript(
                wake_run_id=wake_run_id,
                light_id=light_id,
                model=model,
                web=bool(web),
                created_ts=now,
                updated_ts=now,
            )
        else:
            convo.model = model
            convo.web = bool(web)
        convo.turns.append(ConsultTurn(role="light", text=question, ts=now, model=model))
        convo.turns.append(ConsultTurn(role="consult", text=reply, ts=now, model=model))
        if len(convo.turns) > cap:
            convo.turns = convo.turns[-cap:]
        convo.updated_ts = now
        _write_transcript(settings, convo)
        return convo


def end_wake(settings: Settings, wake_run_id: str) -> bool:
    """Delete the consult transcript for this wake. Missing file is fine."""
    try:
        path = transcript_path(settings, wake_run_id)
    except ValueError:
        return False
    with _STORE_LOCK:
        existed = path.is_file()
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not remove consult transcript %s: %s", path, exc)
            return False
        leftover = path.with_suffix(path.suffix + ".tmp")
        leftover.unlink(missing_ok=True)
        return existed


def sweep_stale_consults(settings: Settings, *, max_age_seconds: float = _STALE_SECONDS) -> int:
    """Remove leftover transcripts if a process died before sleep."""
    root = consult_dir(settings)
    if not root.is_dir():
        return 0
    cutoff = time.time() - max(60.0, float(max_age_seconds))
    removed = 0
    with _STORE_LOCK:
        for path in root.glob("*.json"):
            try:
                if path.stat().st_mtime < cutoff:
                    path.unlink(missing_ok=True)
                    path.with_suffix(path.suffix + ".tmp").unlink(missing_ok=True)
                    removed += 1
            except OSError as exc:
                logger.warning("Could not sweep consult file %s: %s", path, exc)
    if removed:
        logger.info("Swept %d stale consult transcript(s)", removed)
    return removed


def invoke_rumination_graph(graph: Any, state: dict[str, Any], *, settings: Settings) -> Any:
    """Run one rumination wake and delete any consult thread when it sleeps."""
    light_id = str(state.get("agent_id") or "")
    wake = begin_consult_wake(light_id=light_id)
    try:
        try:
            sweep_stale_consults(settings)
        except Exception:
            logger.warning("Consult sweep failed (non-fatal)", exc_info=True)
        return graph.invoke(state)
    finally:
        try:
            end_wake(settings, wake.wake_run_id)
        except Exception:
            logger.warning(
                "Consult cleanup failed wake=%s light=%s",
                wake.wake_run_id,
                wake.light_id,
                exc_info=True,
            )
        clear_consult_wake()
