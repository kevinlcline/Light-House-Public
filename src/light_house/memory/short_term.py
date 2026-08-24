"""Server-side short-term conversation buffer (gap-fill when client omits history)."""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from light_house.memory.dedup import dedupe_assistant_messages, is_near_duplicate_text

Role = Literal["user", "assistant", "system", "peer"]


@dataclass
class BufferedMessage:
    role: Role
    content: str
    ts: float
    from_agent_id: str | None = None
    from_human_id: str | None = None
    from_human_display_name: str | None = None


class ConversationBuffer:
    """Per-thread ring buffer persisted as JSON (server-authoritative chat history)."""

    def __init__(self, data_dir: Path, *, max_messages: int = 60) -> None:
        self._data_dir = data_dir
        self._max_messages = max(2, max_messages)
        data_dir.mkdir(parents=True, exist_ok=True)

    def _path(self, thread_id: str) -> Path:
        safe = thread_id.replace("/", "_").replace("\\", "_")
        return self._data_dir / f"{safe}.json"

    def load(self, thread_id: str) -> list[BufferedMessage]:
        path = self._path(thread_id)
        if not path.is_file():
            return []
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return []
        if not isinstance(raw, list):
            return []
        out: list[BufferedMessage] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = item.get("role")
            content = item.get("content")
            if role not in ("user", "assistant", "system", "peer") or not isinstance(content, str):
                continue
            ts_raw = item.get("ts", 0.0)
            try:
                ts = float(ts_raw)
            except (TypeError, ValueError):
                ts = 0.0
            from_agent_id = item.get("from_agent_id")
            if from_agent_id is not None and not isinstance(from_agent_id, str):
                from_agent_id = None
            from_human_id = item.get("from_human_id")
            if from_human_id is not None and not isinstance(from_human_id, str):
                from_human_id = None
            from_human_display_name = item.get("from_human_display_name")
            if from_human_display_name is not None and not isinstance(
                from_human_display_name, str
            ):
                from_human_display_name = None
            out.append(
                BufferedMessage(
                    role=role,
                    content=content.strip(),
                    ts=ts,
                    from_agent_id=from_agent_id,
                    from_human_id=from_human_id,
                    from_human_display_name=from_human_display_name,
                )
            )
        out = dedupe_assistant_messages(out)
        return out[-self._max_messages :]

    def save(self, thread_id: str, messages: list[BufferedMessage]) -> None:
        trimmed = dedupe_assistant_messages(messages)[-self._max_messages :]
        payload = [asdict(m) for m in trimmed]
        path = self._path(thread_id)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def append_exchange(
        self,
        thread_id: str,
        *,
        user_text: str,
        assistant_text: str,
        user_ts: float | None = None,
        from_human_id: str | None = None,
        from_human_display_name: str | None = None,
    ) -> None:
        user_text = user_text.strip()
        assistant_text = assistant_text.strip()
        if not user_text or not assistant_text:
            return
        hid = (from_human_id or "").strip().lower() or None
        hname = (from_human_display_name or "").strip() or None
        messages = self.load(thread_id)
        if len(messages) >= 2:
            prev_user, prev_ai = messages[-2], messages[-1]
            if (
                prev_user.role == "user"
                and prev_ai.role == "assistant"
                and is_near_duplicate_text(prev_user.content, user_text)
                and is_near_duplicate_text(prev_ai.content, assistant_text)
            ):
                now = time.time()
                user_time = user_ts if user_ts is not None else now
                messages[-2] = BufferedMessage(
                    role="user",
                    content=user_text,
                    ts=user_time,
                    from_human_id=hid,
                    from_human_display_name=hname,
                )
                messages[-1] = BufferedMessage(role="assistant", content=assistant_text, ts=now)
                self.save(thread_id, messages)
                return
        user_time = user_ts if user_ts is not None else time.time()
        assistant_time = time.time()
        messages.append(
            BufferedMessage(
                role="user",
                content=user_text,
                ts=user_time,
                from_human_id=hid,
                from_human_display_name=hname,
            )
        )
        messages.append(
            BufferedMessage(role="assistant", content=assistant_text, ts=assistant_time)
        )
        self.save(thread_id, messages)

    def append_message(self, thread_id: str, message: BufferedMessage) -> None:
        messages = self.load(thread_id)
        messages.append(message)
        self.save(thread_id, messages)
