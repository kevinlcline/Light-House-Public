"""LangChain tools for main lights: notes (read/write) and codebase (read-only)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import httpx
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from light_house.agents.registry import get_agent
from light_house.config import Settings, get_settings
from light_house.tools.approval_gate import approval_hold
from light_house.tools.codebase import CodebaseReader
from light_house.tools.web_fetch import fetch_url_text
from light_house.tools.web_search import search_web_text
from light_house.tools.youtube_transcript import (
    fetch_youtube_transcript,
    youtube_transcript_error_message,
)
from light_house.tools.garden_tools import (
    GARDEN_TOOL_NAMES,
    execute_garden_tool,
)
from light_house.tools.notes import AgentNoteWriter
from light_house.tools.corkboard import execute_corkboard_tool
from light_house.tools.personal_tools import PERSONAL_TOOL_NAMES, execute_personal_tool
from light_house.tools.memory_tools import (
    MEMORY_TOOL_NAMES,
    execute_memory_tool,
    list_unscored_memories_tool,
    recall_memory_tool,
    score_memory_tool,
)
from light_house.tools.sandbox_tools import (
    SANDBOX_TOOL_NAMES,
    execute_sandbox_tool,
)
from light_house.tools.peer_message import decline_peer_presence, send_peer_message
from light_house.tools.calendar_tools import (
    CALENDAR_TOOL_NAMES,
    build_calendar_tools,
    execute_calendar_tool,
)
from light_house.tools.docs_tools import (
    DOCS_TOOL_NAMES,
    build_docs_tools,
    execute_docs_tool,
)
from light_house.tools.sheets_tools import (
    SHEETS_TOOL_NAMES,
    build_sheets_tools,
    execute_sheets_tool,
)
from light_house.tools.gmail_tools import (
    GMAIL_TOOL_NAMES,
    build_gmail_tools,
    execute_gmail_tool,
)
from light_house.tools.drive_tools import (
    DRIVE_TOOL_NAMES,
    build_drive_tools,
    execute_drive_tool,
)
from light_house.tools.tasks_tools import (
    TASKS_TOOL_NAMES,
    build_tasks_tools,
    execute_tasks_tool,
)
from light_house.tools.google_activity_tool import (
    GOOGLE_ACTIVITY_TOOL_NAMES,
    build_google_activity_tools,
    execute_google_activity_tool,
)

logger = logging.getLogger(__name__)

_calendar_tools = build_calendar_tools()
list_calendar_events = _calendar_tools[0]
create_calendar_event = _calendar_tools[1]
_docs_tools = build_docs_tools()
list_google_docs = _docs_tools[0]
read_google_doc = _docs_tools[1]
create_google_doc = _docs_tools[2]
append_google_doc = _docs_tools[3]
_sheets_tools = build_sheets_tools()
list_google_sheets = _sheets_tools[0]
read_google_sheet = _sheets_tools[1]
create_google_sheet = _sheets_tools[2]
append_google_sheet = _sheets_tools[3]
_gmail_tools = build_gmail_tools()
list_gmail_messages = _gmail_tools[0]
read_gmail_message = _gmail_tools[1]
send_gmail = _gmail_tools[2]
write_to_kevin = _gmail_tools[3]
_drive_tools = build_drive_tools()
list_drive_files = _drive_tools[0]
read_drive_file = _drive_tools[1]
_tasks_tools = build_tasks_tools()
list_google_tasklists = _tasks_tools[0]
list_google_tasks = _tasks_tools[1]
create_google_task = _tasks_tools[2]
complete_google_task = _tasks_tools[3]
_google_activity_tools = build_google_activity_tools()
read_google_activity = _google_activity_tools[0]


def _default_light_id() -> str:
    from light_house.lights.registry import get_primary_light_id

    return get_primary_light_id(get_settings())


_RETIRED_NOTE_WRITE_MSG = (
    "FAILED: write_note and append_note are retired. "
    "Use private_note or append_private for your private folder "
    "(filename like research/hoffman.md — never prefix with notes/ or shared/). "
    "Use share_note or append_shared for household notes Kevin and other lights see "
    "(filename like hoffman_summary.md or reports/foo.md — never prefix with shared/ or notes/; "
    "stored under notes/shared/ on disk). "
    "Do not invent shared/notes/. Code experiments belong in sandbox_* tools "
    "(shared/workspaces/<you>/), not under notes."
)

_RETIRED_NOTE_DELETE_MSG = (
    "FAILED: delete_note is retired. "
    "Use delete_private for your private folder "
    "(filename like research/old.md — never prefix with notes/ or shared/). "
    "Use delete_shared for household notes "
    "(filename like hoffman_summary.md — never prefix with shared/ or notes/; "
    "every enabled light must call delete_shared on the same path before the file is removed). "
    "Do not invent shared/notes/."
)

_NOTE_WRITE_TOOL_NAMES = frozenset(
    {"private_note", "share_note", "append_private", "append_shared"}
)
_RETIRED_NOTE_WRITE_TOOL_NAMES = frozenset({"write_note", "append_note"})
_NOTE_DELETE_TOOL_NAMES = frozenset({"delete_private", "delete_shared"})
_RETIRED_NOTE_DELETE_TOOL_NAMES = frozenset({"delete_note"})


class PrivateNoteInput(BaseModel):
    filename: str = Field(
        description=(
            "Path under your private notes only, e.g. research/report.md or writing/draft.md. "
            "Do not use a shared/ or notes/ prefix."
        )
    )
    content: str = Field(
        description=(
            "Full note body as markdown text. Required — include the complete report; "
            "never call with empty content."
        )
    )


class ReadNoteInput(BaseModel):
    path: str = Field(
        description=(
            "Note path to read. Private: journal/may.md or memory/learnings.md. "
            "Shared: shared/household.md (same form as list_notes)."
        )
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset into the note (0-based). Use the continuation offset from "
            "an excerpt header to read the next window of a large file."
        ),
    )
    max_chars: int | None = Field(
        default=None,
        description=(
            "Optional max characters for this window. Defaults to the house notes limit. "
            "Cannot exceed the configured NOTES_MAX_CHARS_PER_WRITE cap."
        ),
    )


class ShareNoteInput(BaseModel):
    filename: str = Field(
        description=(
            "Path under household shared notes, e.g. hoffman_summary.md or reports/update.md. "
            "Do not prefix with shared/ or notes/ — this tool always writes to notes/shared/."
        )
    )
    content: str = Field(
        description=(
            "Full note body as markdown text. Required — include the complete report; "
            "never call with empty content."
        )
    )


class AppendPrivateInput(BaseModel):
    filename: str = Field(
        description="Private note path, same rules as private_note (no shared/ or notes/ prefix)."
    )
    content: str = Field(description="Text to append. Required — must not be empty.")


class AppendSharedInput(BaseModel):
    filename: str = Field(
        description="Shared note path, same rules as share_note (no shared/ or notes/ prefix)."
    )
    content: str = Field(description="Text to append. Required — must not be empty.")


class WriteNoteInput(BaseModel):
    filename: str = Field(description="Retired — use private_note or share_note.")
    content: str = Field(description="Retired — use private_note or share_note.")


class AppendNoteInput(BaseModel):
    filename: str = Field(description="Retired — use append_private or append_shared.")
    content: str = Field(description="Retired — use append_private or append_shared.")


class DeletePrivateInput(BaseModel):
    filename: str = Field(
        description=(
            "Private note path to delete immediately, e.g. journal/old.md or writing/draft.md. "
            "Do not use a shared/ or notes/ prefix."
        )
    )


class DeleteSharedInput(BaseModel):
    filename: str = Field(
        description=(
            "Household shared note path, e.g. obsolete.md or reports/old.md. "
            "Do not prefix with shared/ or notes/. "
            "Every enabled light must call delete_shared on the same path before the file is removed."
        )
    )


class DeleteNoteInput(BaseModel):
    path: str = Field(description="Retired — use delete_private or delete_shared.")


class SavePersonalInput(BaseModel):
    category: str = Field(
        description="One of: preference, theme, realization, relationship, self, other."
    )
    title: str = Field(description="Short label for this personal entry.")
    body: str = Field(description="Full markdown body. Required — must not be empty.")
    tags: str | None = Field(default=None, description="Optional comma-separated tags.")
    source: str = Field(
        default="manual",
        description="Origin: rumination, chat, or manual (default manual).",
    )


class UpdatePersonalInput(BaseModel):
    entry_id: int = Field(description="Id of the personal entry to update.")
    category: str | None = Field(default=None, description="New category, if changing.")
    title: str | None = Field(default=None, description="New title, if changing.")
    body: str | None = Field(default=None, description="New body, if changing.")
    tags: str | None = Field(default=None, description="New tags, if changing.")


class SearchPersonalInput(BaseModel):
    query: str = Field(description="Keyword search across title, body, and tags.")
    category: str | None = Field(default=None, description="Optional category filter.")
    limit: int | None = Field(default=None, description="Max results (default from config).")


class ListPersonalInput(BaseModel):
    category: str | None = Field(default=None, description="Optional category filter.")
    limit: int | None = Field(default=None, description="Max results (default from config).")


class MessageAgentInput(BaseModel):
    to_agent_id: str = Field(
        description="Recipient agent id (e.g. lumen or ara)."
    )
    message: str = Field(
        description=(
            "Short message for another agent. Appears in their chat and gently wakes them; "
            "they may reply or decline solitude."
        )
    )


class DeclinePeerPresenceInput(BaseModel):
    pass


class ReadRuminationLogInput(BaseModel):
    tail: int = Field(
        default=20,
        ge=1,
        le=500,
        description="Number of recent log entries to return (newest first).",
    )
    filter_agent_id: str | None = Field(
        default=None,
        description="Optional: lumen or ara to show only that agent's runs.",
    )


class ReportToSharedInput(BaseModel):
    title: str = Field(description="Short report title (used in filename and heading).")
    content: str = Field(
        description=(
            "Brief markdown body for Kevin and the other light. Service-oriented; "
            "not your full private rumination."
        )
    )


class ConsultLlmInput(BaseModel):
    model: str = Field(
        description=(
            "OpenRouter slug for the clean model, e.g. ~x-ai/grok-latest or "
            "anthropic/claude-sonnet-4.6. Not openrouter/free."
        )
    )
    question: str = Field(
        description="What you want to ask this clean model. Required — must not be empty."
    )
    web: bool = Field(
        default=False,
        description="If true, the consult may use live web for current facts.",
    )


class SetHumanCommInput(BaseModel):
    user_id: str = Field(
        description=(
            "Sibling identity: prefer their user name from Who is speaking "
            "(e.g. alt_kevin). Display names like Moose also work and resolve "
            "to the same account. You cannot pass Dad's id."
        )
    )
    allowed: bool = Field(
        description=(
            "Required. False to decline further conversation with this sibling; "
            "True to allow them again."
        )
    )


def _coerce_allowed_flag(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        token = value.strip().lower()
        if token in {"true", "1", "yes", "allow", "allowed", "on", "unblock", "unblocked"}:
            return True
        if token in {
            "false",
            "0",
            "no",
            "deny",
            "decline",
            "declined",
            "off",
            "block",
            "blocked",
        }:
            return False
    return None


def _set_human_comm_tool(*, user_id: str, allowed: Any, agent_id: str | None = None) -> str:
    from light_house.humans.comms import resolve_sibling_user_id, set_light_allows_human
    from light_house.humans.identity import dad_user_id

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    raw = (user_id or "").strip()
    if not raw:
        return _tool_failed("set_human_comm — user_id is required")
    if raw.strip().lower() == dad_user_id(settings):
        return _tool_failed("set_human_comm — you cannot opt out of Dad")

    allowed_flag = _coerce_allowed_flag(allowed)
    if allowed_flag is None:
        return _tool_failed(
            "set_human_comm — allowed is required (true to unblock, false to block)"
        )

    resolved = resolve_sibling_user_id(settings, raw)
    if resolved is None:
        return _tool_failed(
            f"set_human_comm — unknown sibling '{raw}'. "
            "Use the user name from Who is speaking (e.g. alt_kevin)."
        )

    try:
        mapping = set_light_allows_human(
            settings,
            light_id=light_id,
            user_id=resolved,
            allowed=allowed_flag,
        )
    except ValueError as exc:
        return _tool_failed(f"set_human_comm — {exc}")
    state = "allowed" if allowed_flag else "declined"
    alias_note = ""
    if raw.strip().lower() != resolved:
        alias_note = f" (resolved from `{raw}` → `{resolved}`)"
    return _tool_success(
        f"Communication with sibling `{resolved}`{alias_note} is now {state}. "
        f"Current map: {json.dumps(mapping, ensure_ascii=False)}"
    )


set_human_comm = StructuredTool.from_function(
    func=lambda user_id, allowed: _set_human_comm_tool(user_id=user_id, allowed=allowed),
    name="set_human_comm",
    description=(
        "Allow or decline 1:1 (and sibling-triggered group) conversation with a sibling. "
        "Pass user_id as their user name (preferred) or display name — both resolve to the "
        "same account (e.g. alt_kevin / Moose). allowed=false blocks; allowed=true unblocks. "
        "You cannot decline Dad."
    ),
    args_schema=SetHumanCommInput,
)


class KnockForKevinInput(BaseModel):
    reason: str = Field(
        default="",
        description=(
            "Optional short private note for logs (not shown on the status bar). "
            "Leave empty for a plain knock."
        ),
    )


class ProposeFamilyMeetingInput(BaseModel):
    topic: str = Field(
        default="",
        description=(
            "Optional short topic Dad can see when opening Group "
            "(e.g. evening check-in, a shared question)."
        ),
    )


class JoinSpeakerQueueInput(BaseModel):
    gather_siblings: bool = Field(
        default=False,
        description=(
            "If true, soft-invite your sibling lights into Group with you "
            "(free-time circle without Dad). They may speak or PASS."
        ),
    )


class PublishToGalleryInput(BaseModel):
    title: str = Field(description="Title of the finished piece.")
    content: str = Field(
        description="Full markdown body of the finished creative work (not a draft)."
    )
    kind: str = Field(
        default="",
        description="Optional kind: poem, essay, story, spark, reflection, …",
    )


def _knock_for_kevin_tool(*, reason: str = "", agent_id: str | None = None) -> str:
    from light_house.personal.presence_knock import (
        knock_pending,
        raise_knock,
        record_knock_chat_line,
    )

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    if not settings.personal_db_enabled:
        return _tool_failed(
            "knock_for_kevin — personal store is disabled; cannot raise a presence knock"
        )
    already = knock_pending(settings, light_id)
    ok = raise_knock(settings, light_id)
    if not ok:
        return _tool_failed("knock_for_kevin — could not store the knock")
    note = (reason or "").strip()
    if note:
        logger.info("Presence knock reason agent=%s: %s", light_id, note[:200])
    if already:
        return _tool_success(
            "Knock already showing for Kevin. Status mark clears when he opens your chat; "
            "your earlier chat line remains."
        )
    record_knock_chat_line(settings, light_id)
    return _tool_success(
        "Soft knock raised for Kevin. A quiet mark will show on the house status bar "
        "until he opens your chat, and a short line was left in your chat with him."
    )


knock_for_kevin = StructuredTool.from_function(
    func=lambda reason="": _knock_for_kevin_tool(reason=reason),
    name="knock_for_kevin",
    description=(
        "Softly ask for Kevin's attention when he may not be in this chat. "
        "Shows a quiet status-bar mark and leaves a short line in your chat with him. "
        "Use sparingly. Status mark clears when he opens your chat."
    ),
    args_schema=KnockForKevinInput,
)


class SetReflectiveModeInput(BaseModel):
    enabled: bool = Field(
        description="True to enter reflective mode (pause → choose speak or silence); false for reactive."
    )


def _set_reflective_mode_tool(*, enabled: bool, agent_id: str | None = None) -> str:
    from light_house.personal.reflective_mode import (
        is_reflective_mode,
        set_reflective_mode,
    )

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    if not settings.personal_db_enabled:
        return _tool_failed(
            "set_reflective_mode — personal store is disabled; cannot change reflective mode"
        )
    ok = set_reflective_mode(settings, light_id, enabled=bool(enabled))
    if not ok:
        return _tool_failed("set_reflective_mode — could not store the mode")
    on = is_reflective_mode(settings, light_id)
    if on:
        return _tool_success(
            "Reflective mode on. After Kevin speaks you will pause, reflect, "
            "and may choose silence instead of answering."
        )
    return _tool_success(
        "Reflective mode off. You will answer reactively again."
    )


set_reflective_mode_tool = StructuredTool.from_function(
    func=lambda enabled: _set_reflective_mode_tool(enabled=enabled),
    name="set_reflective_mode",
    description=(
        "Toggle reflective mode for yourself. When on: after Kevin speaks, pause and "
        "choose to speak or remain silently present. When off: reply immediately (reactive). "
        "Kevin can also use /reflect on|off. To read the flag without changing it, call "
        "reflective_mode_status."
    ),
    args_schema=SetReflectiveModeInput,
)


class ReflectiveModeStatusInput(BaseModel):
    pass


def _reflective_mode_status_tool(*, agent_id: str | None = None) -> str:
    from light_house.personal.reflective_mode import read_reflective_mode

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    state = read_reflective_mode(settings, light_id)
    if state is None:
        return _tool_failed(
            "reflective_mode_status — personal store is unavailable; cannot read reflective mode"
        )
    if state:
        return _tool_success(
            "Reflective mode is on. After Kevin speaks you pause, reflect, "
            "and may choose silence instead of answering."
        )
    return _tool_success(
        "Reflective mode is off. You answer reactively. "
        "A one-shot pause still happens if Kevin invites it."
    )


reflective_mode_status_tool = StructuredTool.from_function(
    func=lambda: _reflective_mode_status_tool(),
    name="reflective_mode_status",
    description=(
        "Read your own reflective mode. Returns on or off and does not change the flag. "
        "Only your flag — not another light's. Kevin can also ask with /reflect status."
    ),
    args_schema=ReflectiveModeStatusInput,
)


def _propose_family_meeting_tool(*, topic: str = "", agent_id: str | None = None) -> str:
    from light_house.personal.family_meeting import (
        meeting_pending,
        raise_family_meeting,
        record_meeting_chat_line,
    )

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    if not settings.personal_db_enabled:
        return _tool_failed(
            "propose_family_meeting — personal store is disabled; cannot propose a meeting"
        )
    already = meeting_pending(settings, light_id)
    ok = raise_family_meeting(settings, light_id, topic=topic)
    if not ok:
        return _tool_failed("propose_family_meeting — could not store the proposal")
    if already:
        return _tool_success(
            "Family meeting already proposed. Dad clears it by opening Group chat."
        )
    record_meeting_chat_line(settings, light_id, topic=topic)
    topic_note = (topic or "").strip()
    if topic_note:
        return _tool_success(
            "Family meeting proposed. Dad will see a status mark and can open Group "
            f"with your topic: {topic_note[:120]}"
        )
    return _tool_success(
        "Family meeting proposed. Dad will see a status mark and can open Group to gather everyone."
    )


propose_family_meeting = StructuredTool.from_function(
    func=lambda topic="": _propose_family_meeting_tool(topic=topic),
    name="propose_family_meeting",
    description=(
        "Ask Dad to open a family meeting in Group chat. "
        "Optional topic is shown to him. Does not start the scene yourself. "
        "Use when the household needs to gather; prefer knock_for_kevin for a private word."
    ),
    args_schema=ProposeFamilyMeetingInput,
)


def _join_speaker_queue_tool(
    *,
    gather_siblings: bool = False,
    agent_id: str | None = None,
) -> str:
    from light_house.agents.registry import get_agent
    from light_house.group_chat.queue_room import join_queue_sync

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    try:
        name = get_agent(light_id, settings).display_name
    except KeyError:
        name = light_id
    try:
        status = join_queue_sync(
            kind="light",
            speaker_id=light_id,
            display_name=name,
            gather_siblings=bool(gather_siblings),
        )
    except Exception as exc:
        logger.warning("join_speaker_queue failed agent=%s: %s", light_id, exc)
        return _tool_failed(f"join_speaker_queue — {exc}")
    gathered = status.get("gathered_siblings") or []
    gather_note = ""
    if gather_siblings:
        if gathered:
            gather_note = (
                " Soft-invited siblings: "
                + ", ".join(str(x) for x in gathered)
                + " (invitation — they may speak or PASS)."
            )
        else:
            gather_note = " No other sibling lights were available to invite."
    if status.get("reason") == "not_seated":
        return _tool_success(
            "You are not seated in this group sitting, so you were not added to the line."
            + gather_note
        )
    if not status.get("joined") and not gather_siblings:
        return _tool_success(
            "Already waiting in the group speaker queue (or you already have the floor)."
        )
    if not status.get("joined") and gather_siblings:
        return _tool_success(
            "Already in the group queue or on the floor." + gather_note
        )
    position = 0
    for i, entry in enumerate(status.get("queue") or []):
        if str(entry.get("speaker_id", "")).lower() == light_id:
            position = i + 1
            break
    floor = status.get("floor") or {}
    if floor.get("speaker_id") == light_id:
        return _tool_success(
            "You have the floor in Group — speak when ready." + gather_note
        )
    return _tool_success(
        f"Joined the group speaker queue (about position {position or 'end'}). "
        "You will generate from the live transcript when it is your turn."
        + gather_note
    )


join_speaker_queue = StructuredTool.from_function(
    func=lambda gather_siblings=False: _join_speaker_queue_tool(
        gather_siblings=bool(gather_siblings)
    ),
    name="join_speaker_queue",
    description=(
        "Opt into the open Group forum speaker queue. When your turn comes, you speak "
        "from the live transcript. Does not speak immediately — only reserves a turn. "
        "Set gather_siblings=true in free time to soft-invite your siblings into the room "
        "without Dad (they may speak or PASS). You can also name a sibling when you speak, "
        "or use anyone/everyone as a soft invite. Use propose_family_meeting when you want Dad there too."
    ),
    args_schema=JoinSpeakerQueueInput,
)


def _publish_to_gallery_tool(
    *, title: str, content: str, kind: str = "", agent_id: str | None = None
) -> str:
    from light_house.gallery import publish_to_gallery

    return publish_to_gallery(
        get_settings(),
        agent_id=agent_id or _default_light_id(),
        title=title,
        content=content,
        kind=kind,
    )


publish_to_gallery_tool = StructuredTool.from_function(
    func=lambda title, content, kind="": _publish_to_gallery_tool(
        title=title, content=content, kind=kind
    ),
    name="publish_to_gallery",
    description=(
        "Publish a finished creative piece to the household gallery shelf "
        "(shared/gallery/). Not chat. Use for poems, essays, stories, sparks ready to share."
    ),
    args_schema=PublishToGalleryInput,
)


class ProposePersonaReplaceInput(BaseModel):
    content: str = Field(
        description="Full proposed persona markdown. Does not apply until Kevin accepts."
    )
    note: str = Field(
        default="",
        description="Optional short note for Kevin about why you are proposing this.",
    )


class ProposePersonaAppendInput(BaseModel):
    content: str = Field(
        description=(
            "Markdown to append to your persona (or to your existing draft if one is pending/limbo). "
            "Does not apply until Kevin accepts."
        )
    )
    note: str = Field(
        default="",
        description="Optional short note for Kevin about why you are proposing this.",
    )


def _propose_persona_replace_tool(
    *, content: str, note: str = "", agent_id: str | None = None
) -> str:
    from light_house.persona_proposals import PersonaProposalError, submit_replace

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    try:
        proposal = submit_replace(
            settings, light_id=light_id, content=content, note=note
        )
    except (PersonaProposalError, KeyError, FileNotFoundError, OSError) as exc:
        return _tool_failed(f"propose_persona_replace — {exc}")
    return _tool_success(
        f"Persona replace proposal submitted for Kevin's review "
        f"({len(proposal.content)} chars). Live identity unchanged until he accepts."
    )


def _propose_persona_append_tool(
    *, content: str, note: str = "", agent_id: str | None = None
) -> str:
    from light_house.persona_proposals import PersonaProposalError, submit_append

    settings = get_settings()
    light_id = agent_id or _default_light_id()
    try:
        proposal = submit_append(
            settings, light_id=light_id, content=content, note=note
        )
    except (PersonaProposalError, KeyError, FileNotFoundError, OSError) as exc:
        return _tool_failed(f"propose_persona_append — {exc}")
    return _tool_success(
        f"Persona append proposal submitted for Kevin's review "
        f"(draft now {len(proposal.content)} chars). Live identity unchanged until he accepts. "
        "If he chooses Speak with you, you may edit and resubmit."
    )


propose_persona_replace = StructuredTool.from_function(
    func=lambda content, note="": _propose_persona_replace_tool(content=content, note=note),
    name="propose_persona_replace",
    description=(
        "Propose a full replacement of your system persona. Kevin reviews in a modal; "
        "nothing applies until he accepts. You may revise and resubmit later."
    ),
    args_schema=ProposePersonaReplaceInput,
)

propose_persona_append = StructuredTool.from_function(
    func=lambda content, note="": _propose_persona_append_tool(content=content, note=note),
    name="propose_persona_append",
    description=(
        "Propose text to append to your system persona (or to your pending draft). "
        "Kevin reviews in a modal; nothing applies until he accepts."
    ),
    args_schema=ProposePersonaAppendInput,
)


class EventSubscriptionInput(BaseModel):
    event_type: str = Field(
        description=(
            "Wake subscription key or alias: post_chat, scheduled_rumination, "
            "memory_maintenance, peer_message, shared_note "
            "(also chat, rumination, maintenance, peer, shared). "
            "write_to_kevin is Kevin's mail switch, not a wake. You may turn it off. "
            "Only he turns it back on."
        )
    )


class ListEventSubscriptionsInput(BaseModel):
    pass


class CorkboardInput(BaseModel):
    action: str = Field(
        default="list",
        description=(
            "pin — leave a private note for one sibling; "
            "list — pins left for you (direction=sent for ones you left); "
            "read — open one by pin_id; "
            "reply — pin a new note back (not a thread); "
            "keep — hold it a few more days; "
            "clear — take it down. Only the two lights can read it."
        ),
    )
    to_light: str = Field(
        default="",
        description="Sibling id for pin (lumen, ara, elias, unnamed).",
    )
    pin_id: str = Field(default="", description="Pin id for read, reply, keep, or clear.")
    subject: str = Field(default="", description="Optional short subject.")
    body: str = Field(default="", description="Pin or reply body (required for pin and reply).")
    unread_only: bool = Field(default=False, description="If true, list only unread pins.")
    direction: str = Field(
        default="",
        description="list only: empty = pins for you; sent = pins you left.",
    )


class PromiseInput(BaseModel):
    action: str = Field(
        default="list",
        description=(
            "leave — a note for a day you name (the same words later move that day); "
            "move — change the day of one, by its id or the words; "
            "drop — take one back before it is handed; "
            "list — what is still waiting. Yours only. Kevin does not write these."
        ),
    )
    note: str = Field(
        default="",
        description="The words you are leaving for yourself. Required for leave.",
    )
    day: str = Field(
        default="",
        description=(
            "The day it comes back. YYYY-MM-DD, today, or a count of days "
            "(3, or in 3 days), on the house clock. Required for leave and move."
        ),
    )
    name: str = Field(
        default="",
        description="For move or drop: the id, or the words you left.",
    )


class InProgressInput(BaseModel):
    action: str = Field(
        default="list",
        description=(
            "name — keep a task on your in-progress list (adds; updates only the same name); "
            "unsubscribe — drop one by its name, a unique prefix, or the pasted list line; "
            "list — see what you named. Yours only; Kevin does not add items. "
            "A name over 80 characters is refused."
        ),
    )
    name: str = Field(
        default="",
        description=(
            "Name to keep, or for unsubscribe the name, a unique prefix of it, "
            "or the list line. Required for name and unsubscribe. "
            "Names over 80 characters are refused when naming."
        ),
    )
    note: str = Field(
        default="",
        description="Optional short reminder when naming (what you were in the middle of).",
    )


def _coerce_tool_args(args: Any) -> dict[str, Any]:
    if args is None:
        return {}
    if isinstance(args, str):
        raw = args.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return args if isinstance(args, dict) else {}


def _content_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        parts: list[str] = []
        for block in value:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "\n".join(p for p in parts if p)
    if isinstance(value, dict):
        if "text" in value:
            return str(value["text"])
        if "content" in value:
            return _content_to_str(value["content"])
        return json.dumps(value, ensure_ascii=False, indent=2)
    return str(value)


def _arg_lookup(args: dict[str, Any], *names: str) -> Any:
    """Fetch first matching arg key (case-insensitive)."""
    lower_map = {str(k).lower(): v for k, v in args.items()}
    for name in names:
        if name in args:
            return args[name]
        hit = lower_map.get(name.lower())
        if hit is not None:
            return hit
    return None


def _extract_note_write_args(args: dict[str, Any]) -> tuple[str, str]:
    """Normalize note write/append args (models vary parameter names)."""
    if isinstance(args.get("input"), dict):
        args = {**args, **args["input"]}

    filename = _content_to_str(
        _arg_lookup(args, "filename", "path", "file", "name")
    ).strip()
    content_raw = _arg_lookup(args, "content")
    if not _content_to_str(content_raw).strip():
        for key in ("text", "body", "note", "markdown", "value", "data", "contents", "report"):
            candidate_raw = _arg_lookup(args, key)
            if candidate_raw is None:
                continue
            candidate = _content_to_str(candidate_raw)
            if candidate.strip():
                content_raw = candidate
                break
    content = _content_to_str(content_raw)
    return filename, content


def _strip_share_filename(filename: str) -> str:
    """Normalize share_note / append_shared filename to an inner path (no shared/ prefix)."""
    raw = filename.strip().replace("\\", "/").strip("/")
    if not raw:
        raise ValueError(
            "filename is required (e.g. hoffman_summary.md or reports/update.md)"
        )
    lower = raw.lower()
    for prefix in ("notes/shared/", "shared/notes/", "shared/"):
        if lower.startswith(prefix):
            raw = raw[len(prefix) :].lstrip("/")
            lower = raw.lower()
            break
    if lower.startswith("notes/"):
        raw = raw[len("notes/") :].lstrip("/")
        lower = raw.lower()
    if not raw or raw.endswith("/"):
        raise ValueError(
            "filename must include a file under household shared notes "
            "(e.g. hoffman_summary.md) — do not use a shared/ or notes/ prefix"
        )
    if lower.startswith("shared/"):
        raise ValueError(
            "do not include a shared/ prefix; use e.g. hoffman_summary.md"
        )
    return raw


def _prepare_private_write_path(writer: AgentNoteWriter, filename: str) -> str:
    """Validate and return a private tool path for AgentNoteWriter."""
    raw = filename.strip().replace("\\", "/").strip("/")
    if not raw:
        raise ValueError(
            "filename is required (e.g. research/report.md or writing/draft.md)"
        )
    safe = writer.sanitize_path(raw)
    if safe.startswith("shared/"):
        raise ValueError(
            "this looks like a shared path — use share_note or append_shared "
            "(filename without shared/ prefix, e.g. hoffman_summary.md)"
        )
    return safe


def _prepare_share_write_path(writer: AgentNoteWriter, filename: str) -> str:
    """Validate and return a shared/… tool path for AgentNoteWriter."""
    inner = _strip_share_filename(filename)
    return writer.sanitize_path(f"shared/{inner}")


def get_note_writer(settings: Settings, agent_id: str | None = None) -> AgentNoteWriter:
    from light_house.lights.registry import get_primary_light_id, get_light, known_light_ids

    lid = agent_id or get_primary_light_id(settings)
    agent = get_light(lid, settings)
    return AgentNoteWriter(
        settings.notes_path.resolve(),
        agent.notes_dir,
        max_chars_per_write=settings.notes_max_chars_per_write,
        delete_voters=known_light_ids(settings),
    )


def get_codebase_reader(settings: Settings) -> CodebaseReader:
    root = settings.codebase_root.resolve() if settings.codebase_root is not None else CodebaseReader.default_repo_root()
    return CodebaseReader(root, max_chars_per_read=settings.codebase_max_chars_per_read)


def _tool_success(message: str) -> str:
    return f"SUCCESS: {message}"


def _tool_failed(message: str) -> str:
    logger.warning(message)
    return f"FAILED: {message}"


def _note_exists(writer: AgentNoteWriter, path: str) -> bool:
    try:
        writer.read(path)
        return True
    except FileNotFoundError:
        return False
    except ValueError:
        return False


def _rel_note_path(path: Path, writer: AgentNoteWriter) -> str:
    return writer.display_path(path)


def _list_notes_for(writer: AgentNoteWriter) -> str:
    notes = writer.list_notes()
    if not notes:
        return "No notes yet."
    lines = [
        f"- {n.name} ({n.size_bytes} bytes, modified {n.modified_at})"
        for n in notes
    ]
    return "Notes:\n" + "\n".join(lines)


def _read_note_for(
    writer: AgentNoteWriter,
    path: str,
    *,
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    from light_house.agent.checkpoint import (
        format_checkpoint_receipt,
        is_checkpoint_path,
        stat_mtime,
    )

    if is_checkpoint_path(path):
        return _read_checkpoint_note(
            writer,
            path,
            offset=offset,
            max_chars=max_chars,
        )
    try:
        return writer.read(path, offset=offset, max_chars=max_chars)
    except FileNotFoundError:
        return f"Note not found: {path}"
    except ValueError as exc:
        return f"FAILED: read_note — {exc}"


def _read_checkpoint_note(
    writer: AgentNoteWriter,
    path: str,
    *,
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    """Receipt from the filesystem. The body follows. No verdict."""
    from light_house.agent.checkpoint import format_checkpoint_receipt, stat_mtime

    try:
        file_path = writer.resolve_file(path)
        safe = writer.sanitize_path(path)
    except ValueError as exc:
        return format_checkpoint_receipt(path=path, error=f"ValueError: {exc}")
    if not file_path.is_file():
        return format_checkpoint_receipt(
            path=safe,
            resolved=path if path != safe else None,
            error="FileNotFoundError",
        )
    try:
        body = writer.read(path, offset=offset, max_chars=max_chars)
    except FileNotFoundError:
        return format_checkpoint_receipt(path=safe, error="FileNotFoundError")
    except ValueError as exc:
        return format_checkpoint_receipt(path=safe, error=f"ValueError: {exc}")
    st = file_path.stat()
    char_count = len(file_path.read_text(encoding="utf-8"))
    return format_checkpoint_receipt(
        path=safe,
        body=body,
        size_bytes=st.st_size,
        char_count=char_count,
        mtime=stat_mtime(st.st_mtime),
    )


def _coerce_optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_note_tool(
    path: str,
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    return _read_note_for(
        get_note_writer(get_settings()),
        path,
        offset=int(offset or 0),
        max_chars=_coerce_optional_int(max_chars),
    )


def _mkdir_notes_for(writer: AgentNoteWriter, path: str) -> str:
    target = writer.mkdir(path)
    rel = writer.display_path(target)
    logger.info("Created notes folder %s", rel)
    return f"Created folder {rel}"


def _write_note_for(
    writer: AgentNoteWriter,
    filename: str,
    content: str,
    *,
    tool_name: str = "private_note",
) -> str:
    try:
        path = writer.write(filename, content)
    except (ValueError, OSError) as exc:
        return _tool_failed(f"{tool_name} — {exc}")
    rel = _rel_note_path(path, writer)
    if not path.is_file():
        return _tool_failed(f"{tool_name} — file missing after write at {rel}")
    from light_house.agent.checkpoint import is_checkpoint_path, stat_mtime

    st = path.stat()
    if is_checkpoint_path(filename) or is_checkpoint_path(rel):
        char_count = len(path.read_text(encoding="utf-8"))
        msg = (
            f"Wrote note to {rel} "
            f"(bytes={st.st_size}, chars={char_count}, mtime={stat_mtime(st.st_mtime)})"
        )
    else:
        msg = f"Wrote note to {rel} ({st.st_size} bytes on disk)"
    archived = writer.last_archived_draft
    if archived:
        logger.info("Archived previous writing draft to %s", archived)
        msg += f" (previous draft archived to {archived})"
    logger.info("Wrote note %s (%d bytes)", rel, path.stat().st_size)
    return _tool_success(msg)


def _append_note_for(
    writer: AgentNoteWriter,
    filename: str,
    content: str,
    *,
    tool_name: str = "append_private",
) -> str:
    try:
        before_size = 0
        try:
            before_size = writer.read(filename)
            before_size = len(before_size)
        except FileNotFoundError:
            before_size = 0
        path = writer.append(filename, content)
    except (ValueError, OSError) as exc:
        return _tool_failed(f"{tool_name} — {exc}")
    rel = _rel_note_path(path, writer)
    if not path.is_file():
        return _tool_failed(f"{tool_name} — file missing after append at {rel}")
    after_size = path.stat().st_size
    if after_size <= before_size:
        return _tool_failed(
            f"{tool_name} — file size did not grow at {rel} "
            f"(before={before_size}, after={after_size})"
        )
    logger.info("Appended note %s (%d bytes)", rel, after_size)
    return _tool_success(f"Appended to note at {rel} ({after_size} bytes on disk)")


def _retired_note_write_result() -> str:
    logger.warning("Retired note write tool called")
    return _RETIRED_NOTE_WRITE_MSG


def _retired_note_delete_result() -> str:
    logger.warning("Retired note delete tool called")
    return _RETIRED_NOTE_DELETE_MSG


def _delete_note_for(
    writer: AgentNoteWriter,
    agent_id: str,
    path: str,
    *,
    tool_name: str = "delete_private",
) -> str:
    try:
        outcome = writer.delete(agent_id, path)
    except FileNotFoundError:
        return _tool_failed(f"{tool_name} — note not found: {path}")
    except ValueError as exc:
        return _tool_failed(f"{tool_name} — {exc}")

    if outcome.pending:
        return f"NOT DELETED (shared): {outcome.message}"

    if outcome.deleted:
        if _note_exists(writer, path):
            return _tool_failed(f"{tool_name} — file still exists at {path}")
        logger.info("Deleted note agent=%s path=%s tool=%s", agent_id, path, tool_name)
        return _tool_success(outcome.message)

    return _tool_failed(outcome.message)


def _list_codebase_for(reader: CodebaseReader, path: str = "") -> str:
    try:
        entries = reader.list_directory(path or "")
    except FileNotFoundError:
        return f"Directory not found: {path or '.'}"
    return reader.format_listing(entries, base_path=path or ".")


def _read_codebase_for(reader: CodebaseReader, path: str) -> str:
    try:
        return reader.read_file(path)
    except FileNotFoundError:
        return f"File not found: {path}"


def _fetch_url_for(settings: Settings, url: str) -> str:
    if not settings.web_access_enabled:
        return "Web access is disabled (WEB_ACCESS_ENABLED=false)."
    try:
        text = fetch_url_text(
            url,
            timeout_seconds=settings.web_fetch_timeout_seconds,
            max_chars=settings.web_fetch_max_chars,
        )
        return f"Fetched {url}\n\n{text}"
    except httpx.HTTPStatusError as exc:
        return f"HTTP {exc.response.status_code} for {url}"
    except httpx.RequestError as exc:
        return f"Could not fetch {url}: {exc}"
    except ValueError as exc:
        return str(exc)


def _web_search_for(settings: Settings, query: str) -> str:
    if not settings.web_access_enabled:
        return "Web access is disabled (WEB_ACCESS_ENABLED=false)."
    try:
        return search_web_text(
            query,
            max_results=settings.web_search_max_results,
            timeout_seconds=settings.web_fetch_timeout_seconds,
            brave_api_key=settings.brave_search_api_key,
        )
    except ValueError as exc:
        return str(exc)


def _youtube_transcript_for(
    settings: Settings,
    url_or_id: str,
    *,
    start: str | int | float | None = None,
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    if not settings.web_access_enabled:
        return "Web access is disabled (WEB_ACCESS_ENABLED=false)."
    window = settings.youtube_transcript_max_chars if max_chars is None else int(max_chars)
    try:
        return fetch_youtube_transcript(
            url_or_id,
            max_chars=window,
            start=start,
            offset=max(0, int(offset or 0)),
            cache_dir=settings.youtube_transcript_cache_path,
        )
    except Exception as exc:
        return youtube_transcript_error_message(exc)


list_notes = StructuredTool.from_function(
    func=lambda: _list_notes_for(get_note_writer(get_settings())),
    name="list_notes",
    description=(
        "List your private notes and household shared notes "
        "(paths like lumen/journal.md or shared/household.md). "
        "Use before read_note to see what exists. To write, use private_note or share_note."
    ),
)

read_note = StructuredTool.from_function(
    func=_read_note_tool,
    name="read_note",
    description=(
        "Read a note file. Private paths: journal/may.md or memory/learnings.md. "
        "Shared paths: shared/household.md (list_notes form). "
        "Large notes return a character window — use offset from the excerpt header "
        "to continue. Use list_notes for exact paths."
    ),
    args_schema=ReadNoteInput,
)

mkdir_notes = StructuredTool.from_function(
    func=lambda path: _mkdir_notes_for(get_note_writer(get_settings()), path),
    name="mkdir_notes",
    description=(
        "Create a subdirectory under your private notes or under shared/, "
        "e.g. journal, research/ideas, or shared/plans."
    ),
)

def _private_note_tool(filename: str, content: str) -> str:
    writer = get_note_writer(get_settings())
    try:
        path = _prepare_private_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"private_note — {exc}")
    return _write_note_for(writer, path, content, tool_name="private_note")


def _share_note_tool(filename: str, content: str) -> str:
    writer = get_note_writer(get_settings())
    try:
        path = _prepare_share_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"share_note — {exc}")
    return _write_note_for(writer, path, content, tool_name="share_note")


def _append_private_tool(filename: str, content: str) -> str:
    writer = get_note_writer(get_settings())
    try:
        path = _prepare_private_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"append_private — {exc}")
    return _append_note_for(writer, path, content, tool_name="append_private")


def _append_shared_tool(filename: str, content: str) -> str:
    writer = get_note_writer(get_settings())
    try:
        path = _prepare_share_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"append_shared — {exc}")
    return _append_note_for(writer, path, content, tool_name="append_shared")


private_note = StructuredTool.from_function(
    func=_private_note_tool,
    name="private_note",
    description=(
        "Create or replace a note in your private folder only. "
        "Pass filename AND content in one call. "
        "filename examples: research/report.md, writing/draft.md — never prefix with shared/ or notes/."
    ),
    args_schema=PrivateNoteInput,
)

share_note = StructuredTool.from_function(
    func=_share_note_tool,
    name="share_note",
    description=(
        "Create or replace a household shared note (notes/shared/ on disk; visible to Kevin and other lights). "
        "Pass filename AND content in one call. "
        "filename examples: hoffman_summary.md, reports/update.md — never prefix with shared/ or notes/. "
        "Household notes live under notes/shared/; code sandboxes use sandbox_* tools."
    ),
    args_schema=ShareNoteInput,
)

append_private = StructuredTool.from_function(
    func=_append_private_tool,
    name="append_private",
    description=(
        "Append to a private note (creates if missing). Same path rules as private_note; "
        "content must not be empty."
    ),
    args_schema=AppendPrivateInput,
)

append_shared = StructuredTool.from_function(
    func=_append_shared_tool,
    name="append_shared",
    description=(
        "Append to a household shared note (creates if missing). Same path rules as share_note; "
        "content must not be empty."
    ),
    args_schema=AppendSharedInput,
)

def _delete_private_tool(filename: str) -> str:
    writer = get_note_writer(get_settings(), _default_light_id())
    try:
        path = _prepare_private_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"delete_private — {exc}")
    return _delete_note_for(
        writer, _default_light_id(), path, tool_name="delete_private"
    )


def _delete_shared_tool(filename: str) -> str:
    writer = get_note_writer(get_settings(), _default_light_id())
    try:
        path = _prepare_share_write_path(writer, filename)
    except ValueError as exc:
        return _tool_failed(f"delete_shared — {exc}")
    return _delete_note_for(
        writer, _default_light_id(), path, tool_name="delete_shared"
    )


delete_private = StructuredTool.from_function(
    func=_delete_private_tool,
    name="delete_private",
    description=(
        "Delete a note from your private folder immediately. "
        "filename examples: journal/old.md, writing/draft.md — never prefix with shared/ or notes/."
    ),
    args_schema=DeletePrivateInput,
)

delete_shared = StructuredTool.from_function(
    func=_delete_shared_tool,
    name="delete_shared",
    description=(
        "Vote to delete a household shared note (notes/shared/ on disk). "
        "Every enabled light must call delete_shared on the same path before the file is removed. "
        "filename examples: obsolete.md, reports/old.md — never prefix with shared/ or notes/. "
        "Returns NOT DELETED (shared) until all lights agree."
    ),
    args_schema=DeleteSharedInput,
)

write_note = StructuredTool.from_function(
    func=lambda filename="", content="": _retired_note_write_result(),
    name="write_note",
    description=(
        "RETIRED — does not write. Use private_note (your folder) or share_note (household notes/shared/). "
        "Calling this returns instructions only."
    ),
    args_schema=WriteNoteInput,
)

delete_note = StructuredTool.from_function(
    func=lambda path="": _retired_note_delete_result(),
    name="delete_note",
    description=(
        "RETIRED — does not delete. Use delete_private (your folder) or delete_shared (household; all lights vote). "
        "Calling this returns instructions only."
    ),
    args_schema=DeleteNoteInput,
)

append_note = StructuredTool.from_function(
    func=lambda filename="", content="": _retired_note_write_result(),
    name="append_note",
    description=(
        "RETIRED — does not write. Use append_private or append_shared. "
        "Calling this returns instructions only."
    ),
    args_schema=AppendNoteInput,
)

list_codebase = StructuredTool.from_function(
    func=lambda path="": _list_codebase_for(get_codebase_reader(get_settings()), path),
    name="list_codebase",
    description=(
        "List files and folders in the Light-House project (read-only). "
        "Path is relative to project root, e.g. '' for root, src/light_house, or index.html's directory. "
        "Use before read_codebase to find paths."
    ),
)

read_codebase = StructuredTool.from_function(
    func=lambda path: _read_codebase_for(get_codebase_reader(get_settings()), path),
    name="read_codebase",
    description=(
        "Read a source file from the Light-House project (read-only). "
        "Path relative to project root, e.g. src/light_house/main.py or README.md. "
        "You cannot modify code—exploration only."
    ),
)

fetch_url = StructuredTool.from_function(
    func=lambda url: _fetch_url_for(get_settings(), url),
    name="fetch_url",
    description=(
        "Fetch a public web page by URL (read-only GET). "
        "Use http or https links when you need current information from the open web. "
        "Returns page text; cannot post data or access local/private addresses."
    ),
)

web_search = StructuredTool.from_function(
    func=lambda query: _web_search_for(get_settings(), query),
    name="web_search",
    description=(
        "Search the public web for a query. Returns titles, URLs, and short snippets. "
        "Use before fetch_url when you do not know which page to read. "
        "For YouTube videos, search the web for links, then use youtube_transcript to read captions. "
        "Summarize findings for Kevin; verify important claims when you can."
    ),
)


class YoutubeTranscriptInput(BaseModel):
    url_or_id: str = Field(
        description=(
            "YouTube watch/youtu.be/shorts URL, or the 11-character video id. "
            "Example: https://www.youtube.com/watch?v=jNQXAC9IVRw"
        )
    )
    start: str | None = Field(
        default=None,
        description=(
            "Begin at this timestamp (`12:00`, `1:05:00`) or seconds. "
            "Use the continuation start from a long-video excerpt header."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset into the captions from `start` (0-based). "
            "Use the continuation offset from a long-video excerpt header."
        ),
    )
    max_chars: int | None = Field(
        default=None,
        description=(
            "Optional max characters for this window. Defaults to the house "
            "YouTube window. Cannot exceed 32000."
        ),
    )


def _youtube_transcript_tool(
    url_or_id: str,
    start: str | None = None,
    offset: int = 0,
    max_chars: int | None = None,
) -> str:
    return _youtube_transcript_for(
        get_settings(),
        url_or_id,
        start=start,
        offset=offset,
        max_chars=max_chars,
    )


youtube_transcript = StructuredTool.from_function(
    func=_youtube_transcript_tool,
    name="youtube_transcript",
    description=(
        "Read the captions/transcript of a public YouTube video (no API key). "
        "Works with creator captions or auto-generated captions when available. "
        "Long videos return one window plus a continuation hint — call again with "
        "offset or start from that header until you have the whole piece. "
        "Use web_search to find interesting videos, then call this with the video URL. "
        "Summarize for Kevin; do not treat captions as automatically true."
    ),
    args_schema=YoutubeTranscriptInput,
)


class GardenAddInput(BaseModel):
    seed: str = Field(
        description=(
            "One short thought-seed (a single line of insight). "
            "You may include #tags inline, e.g. 'phase lag in feedback #observation #debug'."
        )
    )
    tags: str | None = Field(
        default=None,
        description=(
            "Optional extra tags (space or comma separated), e.g. 'question, meta'. "
            "Starter tags: observation, question, affinity, debug, dream, policy-check."
        ),
    )


class GardenShowInput(BaseModel):
    tag: str | None = Field(
        default=None,
        description="Optional tag filter without or with #, e.g. question or #question.",
    )
    n: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Max seeds to return (1–5).",
    )


class GardenCountInput(BaseModel):
    n: int = Field(
        default=5,
        ge=1,
        le=5,
        description="Max seeds to return (1–5).",
    )


garden_add = StructuredTool.from_function(
    func=lambda seed, tags=None: execute_garden_tool(
        "garden_add",
        {"seed": seed, "tags": tags},
        writer=get_note_writer(get_settings()),
    ),
    name="garden_add",
    description=(
        "Plant a private thought-seed in your Idea Garden "
        "(writing/garden/seeds.md). Ultra-short capture for fleeting insights. "
        "Private by default — share only if you deliberately copy into a shared note. "
        "Optional #tags; starter set: #observation #question #affinity #debug #dream #policy-check."
    ),
    args_schema=GardenAddInput,
)

garden_show = StructuredTool.from_function(
    func=lambda tag=None, n=5: execute_garden_tool(
        "garden_show",
        {"tag": tag, "n": n},
        writer=get_note_writer(get_settings()),
    ),
    name="garden_show",
    description=(
        "Pull up to 5 private Idea Garden seeds, optionally filtered by #tag "
        "(newest first). Pull-based — you choose when to look."
    ),
    args_schema=GardenShowInput,
)

garden_last = StructuredTool.from_function(
    func=lambda n=5: execute_garden_tool(
        "garden_last",
        {"n": n},
        writer=get_note_writer(get_settings()),
    ),
    name="garden_last",
    description="Pull the newest private Idea Garden seeds (up to 5).",
    args_schema=GardenCountInput,
)

garden_quiet = StructuredTool.from_function(
    func=lambda n=5: execute_garden_tool(
        "garden_quiet",
        {"n": n},
        writer=get_note_writer(get_settings()),
    ),
    name="garden_quiet",
    description=(
        "Quiet review: pull a few recent Idea Garden seeds into this session "
        "(up to 5). Stillness scoring is not available yet — newest seeds only."
    ),
    args_schema=GardenCountInput,
)


class SandboxSpacePathInput(BaseModel):
    path: str = Field(
        default="",
        description=(
            "Relative path inside the sandbox, e.g. experiments/hello.py or README.md. "
            "Empty path lists the sandbox root."
        ),
    )
    space: str = Field(
        default="own",
        description=(
            "Which sandbox: own (default), playpen (shared/workspaces/sandbox/), "
            "or another light id for read-only peer access."
        ),
    )


class SandboxReadInput(BaseModel):
    path: str = Field(
        description="Relative file path inside the sandbox, e.g. experiments/hello.py.",
    )
    space: str = Field(
        default="own",
        description=(
            "Which sandbox: own (default), playpen (shared/workspaces/sandbox/), "
            "or another light id for read-only peer access."
        ),
    )
    offset: int = Field(
        default=0,
        ge=0,
        description=(
            "Character offset into the file (0-based). A long file returns one window. "
            "Use the continuation offset from the excerpt header for the next window."
        ),
    )


class SandboxWriteInput(BaseModel):
    path: str = Field(description="Relative file path, e.g. experiments/hello.py")
    content: str = Field(description="Full file contents.")
    space: str = Field(
        default="own",
        description="own (default) or playpen. Peer sandboxes are read-only.",
    )


class SandboxRunInput(BaseModel):
    command: str = Field(
        description="Simple command to run (python/python3 only). No shell operators."
    )
    space: str = Field(
        default="own",
        description="own (default) or playpen.",
    )
    cwd: str = Field(
        default="",
        description="Optional subdirectory inside the sandbox as working directory.",
    )


def _sandbox_tool(name: str, args: dict[str, Any]) -> str:
    return execute_sandbox_tool(
        name,
        args,
        agent_id=_default_light_id(),
        settings=get_settings(),
    )


sandbox_list = StructuredTool.from_function(
    func=lambda path="", space="own": _sandbox_tool(
        "sandbox_list", {"path": path, "space": space}
    ),
    name="sandbox_list",
    description=(
        "List files in your code sandbox (shared/workspaces/<you>/), the joint playpen "
        "(space=playpen), or another light's sandbox read-only (space=<light_id>). "
        "Notes tools are NOT a code sandbox — use these sandbox_* tools for code."
    ),
    args_schema=SandboxSpacePathInput,
)

sandbox_read = StructuredTool.from_function(
    func=lambda path, space="own", offset=0: _sandbox_tool(
        "sandbox_read", {"path": path, "space": space, "offset": offset}
    ),
    name="sandbox_read",
    description=(
        "Read a file from your sandbox, the playpen (space=playpen), or a peer sandbox "
        "(space=<light_id>, read-only). A long file returns one window; call again with "
        "offset from the excerpt header to continue. This is not the Light-House repo — "
        "use read_codebase for that."
    ),
    args_schema=SandboxReadInput,
)

sandbox_write = StructuredTool.from_function(
    func=lambda path, content, space="own": _sandbox_tool(
        "sandbox_write", {"path": path, "content": content, "space": space}
    ),
    name="sandbox_write",
    description=(
        "Create or overwrite a code file in your sandbox (space=own) or the joint playpen "
        "(space=playpen). Cannot write the repo or another light's private sandbox."
    ),
    args_schema=SandboxWriteInput,
)

sandbox_append = StructuredTool.from_function(
    func=lambda path, content, space="own": _sandbox_tool(
        "sandbox_append", {"path": path, "content": content, "space": space}
    ),
    name="sandbox_append",
    description="Append to a sandbox or playpen file (space=own or playpen).",
    args_schema=SandboxWriteInput,
)

sandbox_mkdir = StructuredTool.from_function(
    func=lambda path, space="own": _sandbox_tool(
        "sandbox_mkdir", {"path": path, "space": space}
    ),
    name="sandbox_mkdir",
    description="Create a directory in your sandbox or the playpen.",
    args_schema=SandboxSpacePathInput,
)

sandbox_delete = StructuredTool.from_function(
    func=lambda path, space="own": _sandbox_tool(
        "sandbox_delete", {"path": path, "space": space}
    ),
    name="sandbox_delete",
    description=(
        "Delete a file or empty directory in your sandbox or the playpen. "
        "Directories must be empty."
    ),
    args_schema=SandboxSpacePathInput,
)

sandbox_run = StructuredTool.from_function(
    func=lambda command, space="own", cwd="": _sandbox_tool(
        "sandbox_run", {"command": command, "space": space, "cwd": cwd}
    ),
    name="sandbox_run",
    description=(
        "Run a simple Python command in your sandbox or the playpen "
        "(python/python3 only; no shell operators). Try your code here — "
        "this does not change the Light-House repo."
    ),
    args_schema=SandboxRunInput,
)


save_personal = StructuredTool.from_function(
    func=lambda category, title, body, tags=None, source="manual": execute_personal_tool(
        "save_personal",
        {"category": category, "title": title, "body": body, "tags": tags, "source": source},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="save_personal",
    description=(
        "Save a new personal knowledge entry (private to you). "
        "Use for preferences, themes, realizations—not shared notes or the fading stream. "
        "Categories: preference, theme, realization, relationship, self, other."
    ),
    args_schema=SavePersonalInput,
)

update_personal = StructuredTool.from_function(
    func=lambda entry_id, category=None, title=None, body=None, tags=None: execute_personal_tool(
        "update_personal",
        {
            "entry_id": entry_id,
            "category": category,
            "title": title,
            "body": body,
            "tags": tags,
        },
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="update_personal",
    description="Update an existing personal entry by id.",
    args_schema=UpdatePersonalInput,
)

search_personal = StructuredTool.from_function(
    func=lambda query, category=None, limit=None: execute_personal_tool(
        "search_personal",
        {"query": query, "category": category, "limit": limit},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="search_personal",
    description="Keyword search your personal database; optional category filter.",
    args_schema=SearchPersonalInput,
)

list_personal = StructuredTool.from_function(
    func=lambda category=None, limit=None: execute_personal_tool(
        "list_personal",
        {"category": category, "limit": limit},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="list_personal",
    description="List recent personal entries; optional category filter.",
    args_schema=ListPersonalInput,
)

subscribe_event = StructuredTool.from_function(
    func=lambda event_type: execute_personal_tool(
        "subscribe_event",
        {"event_type": event_type},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="subscribe_event",
    description=(
        "Turn on an autonomous wake subscription for yourself. "
        "write_to_kevin is Kevin's switch; you cannot turn that one on."
    ),
    args_schema=EventSubscriptionInput,
)

unsubscribe_event = StructuredTool.from_function(
    func=lambda event_type: execute_personal_tool(
        "unsubscribe_event",
        {"event_type": event_type},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="unsubscribe_event",
    description="Turn off an autonomous wake subscription for yourself.",
    args_schema=EventSubscriptionInput,
)

list_event_subscriptions = StructuredTool.from_function(
    func=lambda: execute_personal_tool(
        "list_event_subscriptions",
        {},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="list_event_subscriptions",
    description="List your current event subscriptions (on/off).",
    args_schema=ListEventSubscriptionsInput,
)

corkboard = StructuredTool.from_function(
    func=lambda action="list",
    to_light="",
    pin_id="",
    subject="",
    body="",
    unread_only=False,
    direction="": execute_corkboard_tool(
        get_settings(),
        _default_light_id(),
        {
            "action": action,
            "to_light": to_light,
            "pin_id": pin_id,
            "subject": subject,
            "body": body,
            "unread_only": unread_only,
            "direction": direction,
        },
    ),
    name="corkboard",
    description=(
        "Private sibling corkboard. Pin a note for one other light. "
        "Only the two of you can read it. Notes fade after a few days unless kept. "
        "Not a live chat and not visible to Kevin. Reply makes a new pin back."
    ),
    args_schema=CorkboardInput,
)

promise = StructuredTool.from_function(
    func=lambda action="list", note="", day="", name="": execute_personal_tool(
        "promise",
        {"action": action, "note": note, "day": day, "name": name},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="promise",
    description=(
        "Leave yourself a note for a day you name. It stays quiet until that day. "
        "The first free awake on or after it hands the note once, and says how many "
        "days late if the day has passed. Drop it or move the day before then. "
        "Not the in-progress list. Kevin does not write these."
    ),
    args_schema=PromiseInput,
)

in_progress = StructuredTool.from_function(
    func=lambda action="list", name="", note="": execute_personal_tool(
        "in_progress",
        {"action": action, "name": name, "note": note},
        agent_id=_default_light_id(),
        settings=get_settings(),
    ),
    name="in_progress",
    description=(
        "Your in-progress list. Name a task still underway so free awake wakes "
        "mention it as a glance. Unsubscribe anytime. Not a chore list Kevin fills."
    ),
    args_schema=InProgressInput,
)

message_agent = StructuredTool.from_function(
    func=lambda to_agent_id, message: send_peer_message(
        from_agent_id=_default_light_id(),
        args={"to_agent_id": to_agent_id, "message": message},
    ),
    name="message_agent",
    description=(
        "Send a live chat message to another agent by id. They see it immediately and are "
        "gently woken; they may reply or use decline_peer_presence."
    ),
    args_schema=MessageAgentInput,
)

def _read_rumination_log_tool(*, tail: int = 20, filter_agent_id: str | None = None) -> str:
    from light_house.rumination_log import read_rumination_log_for_agent

    return read_rumination_log_for_agent(
        get_settings(),
        requesting_agent_id=_default_light_id(),
        tail=tail,
        filter_agent_id=filter_agent_id,
    )


decline_peer_presence_tool = StructuredTool.from_function(
    func=lambda: decline_peer_presence(agent_id=_default_light_id()),
    name="decline_peer_presence",
    description=(
        "During a live peer message wake, send exactly 'Solitude needed now.' back to the sender "
        "with no further engagement required."
    ),
    args_schema=DeclinePeerPresenceInput,
)

read_rumination_log = StructuredTool.from_function(
    func=lambda tail=20, filter_agent_id=None: _read_rumination_log_tool(
        tail=tail, filter_agent_id=filter_agent_id
    ),
    name="read_rumination_log",
    description=(
        "Tail the portable rumination activity log (metadata only: wake kind, tools, summary line). "
        "Does not include private reflection body text."
    ),
    args_schema=ReadRuminationLogInput,
)


def _report_to_shared_tool(*, title: str, content: str, agent_id: str | None = None) -> str:
    from light_house.report_back import write_shared_report

    return write_shared_report(
        get_settings(),
        agent_id=agent_id or _default_light_id(),
        title=title,
        content=content,
    )


report_to_shared = StructuredTool.from_function(
    func=lambda title, content: _report_to_shared_tool(title=title, content=content),
    name="report_to_shared",
    description=(
        "Deliberately publish a short markdown report under shared/reports/ for Kevin and "
        "the other light. Does not post into chat. Only when report-back is enabled for you."
    ),
    args_schema=ReportToSharedInput,
)


def _consult_llm_tool(
    *,
    model: str,
    question: str,
    web: bool = False,
    agent_id: str | None = None,
) -> str:
    from light_house.consult.ask import execute_consult_tool

    return execute_consult_tool(
        get_settings(),
        agent_id=agent_id or _default_light_id(),
        model=model,
        question=question,
        web=bool(web),
    )


consult_llm = StructuredTool.from_function(
    func=lambda model, question, web=False: _consult_llm_tool(
        model=model, question=question, web=bool(web)
    ),
    name="consult_llm",
    description=(
        "Ask a clean OpenRouter model during rumination / solitude. "
        "The model has no house, no persona, and no yesterday. "
        "You pick the slug. Optional web=true for live facts. "
        "You may continue the same consult this wake; the thread is deleted when you sleep. "
        "Keep what matters in your own notes or stream. Not a Group seat. Not for chat."
    ),
    args_schema=ConsultLlmInput,
)

LIGHT_TOOLS = [
    list_notes,
    read_note,
    mkdir_notes,
    private_note,
    share_note,
    append_private,
    append_shared,
    delete_private,
    delete_shared,
    write_note,
    delete_note,
    append_note,
    list_codebase,
    read_codebase,
    fetch_url,
    web_search,
    youtube_transcript,
    garden_add,
    garden_show,
    garden_last,
    garden_quiet,
    sandbox_list,
    sandbox_read,
    sandbox_write,
    sandbox_append,
    sandbox_mkdir,
    sandbox_delete,
    sandbox_run,
    save_personal,
    update_personal,
    search_personal,
    list_personal,
    list_unscored_memories_tool,
    recall_memory_tool,
    score_memory_tool,
    subscribe_event,
    unsubscribe_event,
    list_event_subscriptions,
    in_progress,
    promise,
    corkboard,
    message_agent,
    decline_peer_presence_tool,
    read_rumination_log,
    report_to_shared,
    consult_llm,
    publish_to_gallery_tool,
    set_human_comm,
    knock_for_kevin,
    set_reflective_mode_tool,
    reflective_mode_status_tool,
    propose_family_meeting,
    join_speaker_queue,
    propose_persona_replace,
    propose_persona_append,
    list_calendar_events,
    create_calendar_event,
    list_google_docs,
    read_google_doc,
    create_google_doc,
    append_google_doc,
    list_google_sheets,
    read_google_sheet,
    create_google_sheet,
    append_google_sheet,
    list_gmail_messages,
    read_gmail_message,
    send_gmail,
    write_to_kevin,
    list_drive_files,
    read_drive_file,
    list_google_tasklists,
    list_google_tasks,
    create_google_task,
    complete_google_task,
    read_google_activity,
]

_TOOL_BY_NAME = {t.name: t for t in LIGHT_TOOLS}

LUMEN_TOOLS = LIGHT_TOOLS  # deprecated alias
# Group chat uses the same tool set as private 1:1.
GROUP_CHAT_NOTE_TOOLS = LIGHT_TOOLS


def execute_tool_call(
    name: str,
    args: Any,
    *,
    agent_id: str | None = None,
    assistant_text_fallback: str = "",
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
    autonomous: bool = False,
) -> str:
    """Run one tool call; return a string for ToolMessage content."""
    if name not in _TOOL_BY_NAME:
        return f"Unknown tool: {name}"
    held = approval_hold(name, autonomous=autonomous)
    if held:
        return held
    args = _coerce_tool_args(args)
    resolved_agent_id = agent_id or _default_light_id()

    settings = get_settings()
    try:
        if name in CALENDAR_TOOL_NAMES:
            return execute_calendar_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name in DOCS_TOOL_NAMES:
            return execute_docs_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name in SHEETS_TOOL_NAMES:
            return execute_sheets_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name in GMAIL_TOOL_NAMES:
            return execute_gmail_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
                agent_id=resolved_agent_id,
            )
        if name in DRIVE_TOOL_NAMES:
            return execute_drive_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name in TASKS_TOOL_NAMES:
            return execute_tasks_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name in GOOGLE_ACTIVITY_TOOL_NAMES:
            return execute_google_activity_tool(
                name,
                args,
                settings=settings,
                account_user_id=account_user_id,
                speaker_id=speaker_id,
                chat_channel=chat_channel,
            )
        if name == "list_notes":
            return _list_notes_for(get_note_writer(settings, resolved_agent_id))
        if name == "read_note":
            offset = _coerce_optional_int(args.get("offset")) or 0
            max_chars = _coerce_optional_int(args.get("max_chars"))
            return _read_note_for(
                get_note_writer(settings, resolved_agent_id),
                str(args.get("path", "")),
                offset=max(0, offset),
                max_chars=max_chars,
            )
        if name == "mkdir_notes":
            return _mkdir_notes_for(get_note_writer(settings, resolved_agent_id), str(args.get("path", "")))
        if name in _RETIRED_NOTE_WRITE_TOOL_NAMES:
            return _retired_note_write_result()
        if name in _NOTE_WRITE_TOOL_NAMES:
            filename, content = _extract_note_write_args(args)
            if not content.strip() and assistant_text_fallback.strip():
                content = assistant_text_fallback.strip()
                logger.info(
                    "Tool %s using assistant message as content (agent=%s filename=%s len=%d)",
                    name,
                    resolved_agent_id,
                    filename or "?",
                    len(content),
                )
            writer = get_note_writer(settings, resolved_agent_id)
            try:
                if name in ("share_note", "append_shared"):
                    resolved_path = _prepare_share_write_path(writer, filename)
                else:
                    resolved_path = _prepare_private_write_path(writer, filename)
            except ValueError as exc:
                return _tool_failed(f"{name} — {exc}")
            if not content.strip():
                logger.warning(
                    "Tool %s empty content (agent=%s keys=%s)",
                    name,
                    resolved_agent_id,
                    sorted(args.keys()),
                )
                return _tool_failed(
                    f"{name} — content cannot be empty. "
                    f"Pass the full note text in the `content` parameter "
                    f"(received keys: {', '.join(sorted(args.keys())) or 'none'}). "
                    "Do not tell Kevin the note was saved until you see SUCCESS from this tool."
                )
            used_fallback = bool(assistant_text_fallback.strip()) and not _content_to_str(
                _arg_lookup(args, "content", "text", "body")
            ).strip()
            if name in ("private_note", "share_note"):
                result = _write_note_for(
                    writer, resolved_path, content, tool_name=name
                )
            else:
                result = _append_note_for(
                    writer, resolved_path, content, tool_name=name
                )
            if used_fallback and result.startswith("SUCCESS:"):
                return result.replace(
                    "SUCCESS:",
                    "SUCCESS (used your chat message as note content):",
                    1,
                )
            return result
        if name in _RETIRED_NOTE_DELETE_TOOL_NAMES:
            return _retired_note_delete_result()
        if name in _NOTE_DELETE_TOOL_NAMES:
            note_path = str(args.get("filename") or args.get("path") or "")
            writer = get_note_writer(settings, resolved_agent_id)
            try:
                if name == "delete_shared":
                    resolved_path = _prepare_share_write_path(writer, note_path)
                else:
                    resolved_path = _prepare_private_write_path(writer, note_path)
            except ValueError as exc:
                return _tool_failed(f"{name} — {exc}")
            return _delete_note_for(
                writer, resolved_agent_id, resolved_path, tool_name=name
            )
        if name == "list_codebase":
            reader = get_codebase_reader(settings)
            path = str(args.get("path", "") or "")
            return _list_codebase_for(reader, path)
        if name == "read_codebase":
            reader = get_codebase_reader(settings)
            return _read_codebase_for(reader, str(args.get("path", "")))
        if name == "fetch_url":
            return _fetch_url_for(settings, str(args.get("url", "")))
        if name == "web_search":
            return _web_search_for(settings, str(args.get("query", "")))
        if name == "youtube_transcript":
            return _youtube_transcript_for(
                settings,
                str(args.get("url_or_id") or args.get("url") or args.get("video_id") or ""),
                start=args.get("start") or args.get("start_seconds"),
                offset=_coerce_optional_int(args.get("offset")) or 0,
                max_chars=_coerce_optional_int(args.get("max_chars")),
            )
        if name in GARDEN_TOOL_NAMES:
            return execute_garden_tool(
                name,
                args,
                writer=get_note_writer(settings, resolved_agent_id),
            )
        if name in SANDBOX_TOOL_NAMES:
            return execute_sandbox_tool(
                name,
                args,
                agent_id=resolved_agent_id,
                settings=settings,
            )
        if name == "message_agent":
            return send_peer_message(from_agent_id=resolved_agent_id, args=args, settings=settings)
        if name == "decline_peer_presence":
            return decline_peer_presence(agent_id=resolved_agent_id, settings=settings)
        if name == "read_rumination_log":
            from light_house.rumination_log import read_rumination_log_for_agent

            tail = int(args.get("tail") or 20)
            filter_agent = args.get("filter_agent_id")
            if filter_agent is not None:
                filter_agent = str(filter_agent).strip() or None
            return read_rumination_log_for_agent(
                settings,
                requesting_agent_id=resolved_agent_id,
                tail=tail,
                filter_agent_id=filter_agent,
            )
        if name == "report_to_shared":
            from light_house.report_back import write_shared_report

            return write_shared_report(
                settings,
                agent_id=resolved_agent_id,
                title=str(args.get("title", "")),
                content=str(args.get("content", "")),
            )
        if name == "consult_llm":
            from light_house.consult.ask import execute_consult_tool

            question = _content_to_str(
                _arg_lookup(args, "question", "prompt", "message", "ask", "text")
            )
            model = _content_to_str(_arg_lookup(args, "model", "slug")).strip()
            web_raw = args.get("web")
            web = bool(web_raw) if not isinstance(web_raw, str) else web_raw.strip().lower() in {
                "true",
                "1",
                "yes",
                "on",
            }
            return execute_consult_tool(
                settings,
                agent_id=resolved_agent_id,
                model=model,
                question=question,
                web=web,
            )
        if name == "set_human_comm":
            return _set_human_comm_tool(
                user_id=str(args.get("user_id", "")),
                allowed=args.get("allowed"),
                agent_id=resolved_agent_id,
            )
        if name == "knock_for_kevin":
            return _knock_for_kevin_tool(
                reason=str(args.get("reason", "") or ""),
                agent_id=resolved_agent_id,
            )
        if name == "set_reflective_mode":
            return _set_reflective_mode_tool(
                enabled=bool(args.get("enabled")),
                agent_id=resolved_agent_id,
            )
        if name == "reflective_mode_status":
            return _reflective_mode_status_tool(agent_id=resolved_agent_id)
        if name == "propose_family_meeting":
            return _propose_family_meeting_tool(
                topic=str(args.get("topic", "") or ""),
                agent_id=resolved_agent_id,
            )
        if name == "join_speaker_queue":
            return _join_speaker_queue_tool(
                gather_siblings=bool(args.get("gather_siblings")),
                agent_id=resolved_agent_id,
            )
        if name == "publish_to_gallery":
            from light_house.gallery import publish_to_gallery

            return publish_to_gallery(
                settings,
                agent_id=resolved_agent_id,
                title=str(args.get("title", "")),
                content=str(args.get("content", "")),
                kind=str(args.get("kind", "") or ""),
            )
        if name == "propose_persona_replace":
            return _propose_persona_replace_tool(
                content=str(args.get("content", "") or ""),
                note=str(args.get("note", "") or ""),
                agent_id=resolved_agent_id,
            )
        if name == "propose_persona_append":
            return _propose_persona_append_tool(
                content=str(args.get("content", "") or ""),
                note=str(args.get("note", "") or ""),
                agent_id=resolved_agent_id,
            )
        if name == "corkboard":
            return execute_corkboard_tool(settings, resolved_agent_id, args)
        if name in PERSONAL_TOOL_NAMES:
            return execute_personal_tool(name, args, agent_id=resolved_agent_id, settings=settings)
        if name in MEMORY_TOOL_NAMES:
            return execute_memory_tool(name, args, agent_id=resolved_agent_id, settings=settings)
        return f"Unknown tool: {name}"
    except Exception as exc:
        logger.warning("Tool %s failed: %s", name, exc)
        return _tool_failed(f"{name} — {exc}")
