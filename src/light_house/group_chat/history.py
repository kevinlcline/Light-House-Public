"""Kevin-only group chat round history for UI replay."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from light_house.config import Settings

logger = logging.getLogger(__name__)
_HISTORY_LOCK = threading.Lock()


def _history_path(settings: Settings) -> Path:
    return settings.group_chat_history_path.resolve()


def append_group_round(settings: Settings, record: dict[str, Any]) -> None:
    path = _history_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    with _HISTORY_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _trim_history_file(path, settings.group_chat_ui_history_rounds)


def _trim_history_file(path: Path, max_rounds: int) -> None:
    if max_rounds <= 0 or not path.is_file():
        return
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return
    if len(lines) <= max_rounds:
        return
    tail = lines[-max_rounds:]
    path.write_text("\n".join(tail) + "\n", encoding="utf-8")


def read_group_round_history(settings: Settings, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = _history_path(settings)
    cap = limit if limit is not None else settings.group_chat_ui_history_rounds
    if not path.is_file():
        return []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read group chat history: %s", exc)
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(data)
    if cap > 0 and len(records) > cap:
        records = records[-cap:]
    return records


def _utterance_history_path(settings: Settings) -> Path:
    base = settings.group_chat_history_path.resolve()
    return base.parent / "utterances.ndjson"


def append_group_utterance(settings: Settings, record: dict[str, Any]) -> None:
    """Append one open-forum utterance for UI replay."""
    path = _utterance_history_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False)
    # Keep more lines than rounds — each sitting has many utterances.
    max_lines = max(200, settings.group_chat_ui_history_rounds * 20)
    with _HISTORY_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
        _trim_history_file(path, max_lines)
    if (record.get("speaker_kind") or "") == "light":
        try:
            from light_house.tts.face_unmatched_log import observe_light_speech

            observe_light_speech(
                str(record.get("text") or ""),
                agent_id=str(record.get("speaker_id") or ""),
                source="group",
                settings=settings,
            )
        except Exception:
            logger.exception("face unmatched observe failed (non-fatal)")


def read_group_utterances(settings: Settings, *, limit: int | None = None) -> list[dict[str, Any]]:
    path = _utterance_history_path(settings)
    cap = limit if limit is not None else max(100, settings.group_chat_ui_history_rounds * 10)
    if not path.is_file():
        return []
    try:
        # Prefer a cheap tail read — full-file scans made /forum feel slow as history grew.
        lines = _tail_text_lines(path, max_lines=max(cap * 2, 200))
    except OSError as exc:
        logger.warning("Could not read group utterance history: %s", exc)
        return []
    records: list[dict[str, Any]] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("text"):
            records.append(data)
    if cap > 0 and len(records) > cap:
        records = records[-cap:]
    return records


def _tail_text_lines(path: Path, *, max_lines: int) -> list[str]:
    """Return up to ``max_lines`` trailing non-empty-capable lines from a text file."""
    if max_lines <= 0:
        return []
    size = path.stat().st_size
    if size <= 0:
        return []
    # Small files: just read.
    if size < 256_000:
        return path.read_text(encoding="utf-8").splitlines()
    block = 8192
    data = b""
    with path.open("rb") as f:
        pos = size
        while pos > 0 and data.count(b"\n") <= max_lines:
            read_size = min(block, pos)
            pos -= read_size
            f.seek(pos)
            data = f.read(read_size) + data
            if pos == 0:
                break
    text = data.decode("utf-8", errors="replace")
    lines = text.splitlines()
    if len(lines) > max_lines:
        lines = lines[-max_lines:]
    return lines
