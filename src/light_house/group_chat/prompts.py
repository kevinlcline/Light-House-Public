"""Prompts and schema for sequential group chat scenes."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator

from light_house.config import Settings
from light_house.tts.stage_cues import FACE_STAGE_HINT

GROUP_CHAT_TASK_HINT = (
    "\n\n## Group chat scene (shared room)\n"
    "A human spoke in a group chat with you and the other lights.\n"
    "Address the person named under **Who is speaking** — that is who just spoke. "
    "**Who is present** is only the room roster; do not treat the host as the speaker "
    "unless they are also Who is speaking.\n"
    "Honor **Who is present** for tone — guests in the room means tone down private or "
    "romantic affection unless the host clearly invites it.\n"
    "You can see the **scene transcript so far** — what was already said in this scene.\n"
    "You have the **same tools as private 1:1** — **list_notes**, **read_note**, "
    "**web_search**, sandbox, garden, gallery, and the signed-in human's "
    "Calendar, Docs, Sheets, Gmail, Drive, and Tasks. "
    "Use them if you need to look something up before you speak. "
    "Those Google tools belong to the signed-in human speaking as themselves — "
    "not a guest or the visitor. "
    "**read_google_activity** is the house log of new mail and file changes. "
    "That log does not wake anyone.\n"
    "Choose one action after any note lookups:\n"
    "- **speak**: add your voice. You may address the current speaker or reply to another light.\n"
    "- **pass**: stay silent this beat (always allowed).\n"
    "- **close**: optional short closer if the scene feels complete "
    "(still may be cut off by house caps).\n"
    "Prefer a distinct angle, or pass. Do not repeat what was just said.\n\n"
    "When ready, reply with **only** valid JSON (no markdown fences):\n"
    '{"action":"speak","text":"your words"} or '
    '{"action":"pass","text":""} or '
    '{"action":"close","text":"optional brief closer"}'
    + FACE_STAGE_HINT
)

# Backward-compatible hint text for parallel mode (isolated decisions).
GROUP_CHAT_PARALLEL_HINT = (
    "\n\n## Group chat round\n"
    "A human asked one question in a group chat with you and the other lights. "
    "You are deciding **alone** — you cannot see what anyone else will say this round.\n"
    "Address the person named under **Who is speaking**.\n"
    "You have the **same tools as private 1:1**. Use them if you need to look something up "
    "first. Calendar, Docs, Sheets, Gmail, Drive, and Tasks belong to the signed-in human "
    "speaking as themselves — not a guest or the visitor. "
    "The Google activity log does not wake anyone.\n"
    "If you speak, address that person directly. You may mention another light by name, "
    "but this is still your reply to their prompt — not a side conversation.\n"
    "You may stay silent this round and speak on a later one.\n\n"
    "When ready, reply with **only** valid JSON (no markdown fences):\n"
    '{"speak": true, "text": "your words"} or {"speak": false, "text": ""}'
    + FACE_STAGE_HINT
)

def group_worker_boundary(settings: Settings, agent_id: str) -> str:
    """Public template has no cursor worker. The hook stays so group prompts match."""
    del settings, agent_id
    return ""


class GroupChatDecision(BaseModel):
    """Light decision for one beat (speak / pass / close)."""

    action: Literal["speak", "pass", "close"] = Field(
        default="pass",
        description="speak, pass, or close the scene.",
    )
    text: str = Field(default="", description="Words when speaking or closing.")
    # Legacy parallel-mode field; normalized into action in the validator.
    speak: bool | None = Field(default=None, exclude=True)

    @model_validator(mode="before")
    @classmethod
    def _normalize_legacy_speak(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        raw = dict(data)
        if "action" not in raw and "speak" in raw:
            raw["action"] = "speak" if raw.get("speak") else "pass"
        return raw

    @property
    def spoke(self) -> bool:
        return self.action in ("speak", "close") and bool(self.text.strip())
