"""Per-agent peer message inbox (immediate delivery, optional async reply)."""

from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from light_house.agents.registry import validate_agent_id


@dataclass(frozen=True)
class PeerMessage:
    id: str
    from_agent_id: str
    to_agent_id: str
    body: str
    ts: float
    seen: bool


class PeerInbox:
    """JSON-backed inbox per agent for unread peer messages."""

    def __init__(self, data_dir: Path) -> None:
        self._root = (data_dir / "peer_inbox").resolve()
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, agent_id: str) -> Path:
        validate_agent_id(agent_id)
        safe = agent_id.replace("/", "_").replace("\\", "_")
        return self._root / f"{safe}.json"

    def _load(self, agent_id: str) -> list[dict]:
        path = self._path(agent_id)
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        return raw if isinstance(raw, list) else []

    def _save(self, agent_id: str, rows: list[dict]) -> None:
        path = self._path(agent_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    @staticmethod
    def _row_to_message(row: dict) -> PeerMessage | None:
        if not isinstance(row, dict):
            return None
        msg_id = row.get("id")
        from_id = row.get("from_agent_id")
        to_id = row.get("to_agent_id")
        body = row.get("body")
        if not isinstance(msg_id, str) or not isinstance(from_id, str) or not isinstance(to_id, str):
            return None
        if not isinstance(body, str) or not body.strip():
            return None
        try:
            ts = float(row.get("ts", 0.0))
        except (TypeError, ValueError):
            ts = 0.0
        seen = bool(row.get("seen", False))
        return PeerMessage(
            id=msg_id,
            from_agent_id=from_id,
            to_agent_id=to_id,
            body=body.strip(),
            ts=ts,
            seen=seen,
        )

    def deliver(self, *, from_agent_id: str, to_agent_id: str, body: str) -> PeerMessage:
        validate_agent_id(from_agent_id)
        validate_agent_id(to_agent_id)
        if from_agent_id == to_agent_id:
            raise ValueError("Cannot send a peer message to yourself")
        text = body.strip()
        if not text:
            raise ValueError("Message cannot be empty")

        msg = PeerMessage(
            id=str(uuid.uuid4()),
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            body=text,
            ts=time.time(),
            seen=False,
        )
        rows = self._load(to_agent_id)
        rows.append(
            {
                "id": msg.id,
                "from_agent_id": msg.from_agent_id,
                "to_agent_id": msg.to_agent_id,
                "body": msg.body,
                "ts": msg.ts,
                "seen": msg.seen,
            }
        )
        self._save(to_agent_id, rows)
        return msg

    def list_unread(self, to_agent_id: str, *, limit: int = 20) -> list[PeerMessage]:
        rows = self._load(to_agent_id)
        out: list[PeerMessage] = []
        for row in rows:
            msg = self._row_to_message(row)
            if msg is None or msg.seen:
                continue
            out.append(msg)
        out.sort(key=lambda m: m.ts)
        return out[-limit:]

    def mark_seen(self, to_agent_id: str, message_ids: list[str]) -> None:
        if not message_ids:
            return
        wanted = set(message_ids)
        rows = self._load(to_agent_id)
        changed = False
        for row in rows:
            if isinstance(row, dict) and row.get("id") in wanted:
                row["seen"] = True
                changed = True
        if changed:
            self._save(to_agent_id, rows)
