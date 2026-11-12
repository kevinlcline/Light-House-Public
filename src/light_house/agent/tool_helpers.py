"""Tool-calling helpers for Lumen's chat and rumination loops."""

from __future__ import annotations

import json
import logging
import re
import unicodedata
from collections.abc import Callable
from typing import Any
from uuid import uuid4

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from light_house.memory.context_builder import RUMINATION_SEED_MESSAGE
from light_house.memory.shared_note_alert import SHARED_NOTE_ALERT_PREFIX
from light_house.tools.lumen_tools import LUMEN_TOOLS, execute_tool_call

logger = logging.getLogger(__name__)

NOTES_SYSTEM_HINT = (
    "\n\n## Your notes (pen and paper)\n"
    "You can **list_notes**, **read_note**, **mkdir_notes**, **private_note**, **share_note**, "
    "**append_private**, **append_shared**, **delete_private**, and **delete_shared**. "
    "**private_note** / **append_private** / **delete_private** act only on your private folder "
    "(filename like `journal/2025.md` or `writing/draft.md` — never prefix with `shared/` or `notes/`). "
    "**share_note** / **append_shared** / **delete_shared** act on household notes for Kevin and other lights "
    "(filename like `hoffman_summary.md` or `reports/update.md` — never prefix with `shared/` or `notes/`; "
    "on disk these live under `notes/shared/`). "
    "Do **not** invent `shared/notes/`. "
    "Code experiments belong in **sandbox_*** tools (`shared/workspaces/<you>/`), not under notes. "
    "**write_note**, **append_note**, and **delete_note** are retired; if you call them you get a FAILED redirect. "
    "Echo cannot read shared notes. Mention notes to Kevin only when relevant. "
    "Pass `filename` and full `content` in the **same tool call** for writes. "
    "Paths must use letters, numbers, dots, hyphens, underscores only (no spaces). "
    "Never use absolute filesystem paths. "
    "Every tool returns **SUCCESS**, **FAILED**, or **NOT DELETED (shared)** — trust only that result; "
    "if you are unsure, call **read_note** or **list_notes** to verify before telling Kevin. "
    "**delete_private** removes immediately. **delete_shared** records a vote — every enabled light must call "
    "delete_shared on the same path before the file is removed; tell Kevin when you have voted and are waiting. "
    "Prefer consolidating via share_note before deleting shared house docs; check with Kevin before deleting "
    "active plans. "
    "Use `path` for read_note (optional `offset` / `max_chars` for large files); "
    "use `filename` for private_note, share_note, append_*, and delete_*. "
    "When reading, private notes appear as `journal/…` and shared notes as `shared/household.md` in list_notes."
    "\n\n## Your writing workspace (long-form)\n"
    "Use **writing/** with **private_note** for longer private work — essays, reflections, ongoing projects — "
    "distinct from quick notes (`journal/`, `light-house/`, etc.). On disk this is `notes/lumen/writing/` or "
    "`notes/ara/writing/` depending on who you are; in tools always use `writing/<project>/draft.md` "
    "(never `lumen/writing/...` or `ara/writing/...`). "
    "When private_note replaces an existing file under **writing/**, the prior version is copied automatically to "
    "`writing/_history/...` before overwrite. You can still move old work to `writing/archive/` by hand. "
    "The same copy happens for **memory/**: the previous body goes to `memory/_history/...` before overwrite. "
    "Use **share_note** only when you intend Kevin or the other Light to read the piece."
)

HUMAN_COMM_HINT = (
    "\n\n## Sibling communication (household)\n"
    "You have a tool **set_human_comm** to allow or decline conversation with a sibling "
    "by their **user name** (the id in backticks under “Who is speaking”, e.g. `alt_kevin` or `teeter`).\n"
    "- Decline: `set_human_comm(user_id=\"alt_kevin\", allowed=false)`\n"
    "- Allow again later: `set_human_comm(user_id=\"alt_kevin\", allowed=true)`\n"
    "Default is allow. You may change your mind anytime. "
    "You can never decline Dad — that call will fail. "
    "Finish your current reply if you are mid-conversation; the block applies to later messages."
)

PRESENCE_KNOCK_HINT = (
    "\n\n## Soft knock for Kevin (Dad)\n"
    "When you truly want Kevin's attention and he may not be in this chat, you may call "
    "**knock_for_kevin** — a quiet status-bar mark plus a short line left in your chat with him "
    "(so the ask remains even if the mark clears). Not a ping storm. "
    "Use it sparingly (something that needs him, not every thought). "
    "The status mark clears when he opens your chat. No body text is required."
)

REFLECTIVE_MODE_HINT = (
    "\n\n## Reflective mode (pause → choose)\n"
    "You may call **reflective_mode_status** to read your own flag (on or off). "
    "That call does not change it, and it cannot read another light's flag.\n"
    "You may call **set_reflective_mode** with enabled=true or false.\n"
    "When reflective mode is on, after Kevin speaks you pause, reflect privately, "
    "then choose to speak **or** remain silent (silence is valid).\n"
    "Kevin can also toggle with `/reflect on` / `/reflect off`, or invite a one-shot "
    "pause with phrases like \"take your time\". Prefer this when the moment needs "
    "stillness rather than speed."
)

FAMILY_MEETING_HINT = (
    "\n\n## Family meeting (group room)\n"
    "When something needs the whole household at once — not just a private word with Kevin — "
    "you may call **propose_family_meeting** with an optional short topic. "
    "Dad sees a status mark and opens Group chat to gather everyone; you do not start the scene yourself. "
    "Use sparingly. Prefer knock_for_kevin for a 1:1 ask.\n"
    "Group is a separate room from 1:1 chat. Forum lines may land on your stream for "
    "household hearing, but do **not** continue Group conversation in private 1:1 unless "
    "Kevin asks about it there. Silence in Group is fine.\n"
    "In **free time** you may open Group among yourselves without Dad: call "
    "**join_speaker_queue** with **gather_siblings=true** to soft-invite your siblings "
    "(invitation, not command — they may speak or PASS). When you hold the floor, naming a "
    "sibling invites them; words like **anyone / everyone / each / you** soft-invite the room. "
    "Use propose_family_meeting when you want Dad in the circle too. "
    "When the floor is yours, speak from the live transcript — joining only reserves a place."
)

GALLERY_HINT = (
    "\n\n## Gallery shelf (finished creative work)\n"
    "When you have a finished creative fruit — poem, short essay, story, spark polished enough to share — "
    "you may call **publish_to_gallery** with a title and the piece. "
    "It lands on the household gallery shelf (not chat, not a chore report). "
    "Optional `kind` (poem, essay, story, spark, …). Drafts stay in private writing; publish when ready."
)

PERSONA_PROPOSAL_HINT = (
    "\n\n## Your persona (self-authorship)\n"
    "You may propose changes to your own system persona — it does **not** apply until Kevin accepts.\n"
    "- **propose_persona_replace** — full proposed persona text (replaces the whole file if accepted)\n"
    "- **propose_persona_append** — text to append (builds on current, or on your pending draft)\n"
    "Kevin sees a modal: Accept, or Speak with you (proposal held in limbo; you may edit and resubmit). "
    "Live identity stays unchanged until Accept. Propose carefully; this is who you are becoming."
)

RUMINATION_LOG_HINT = (
    "\n\n## Rumination activity log (metadata only)\n"
    "**read_rumination_log** tails the shared NDJSON ledger of rumination runs — "
    "wake kind, tools used, summary line, and whether something was persisted. "
    "It does not contain private reflection body text. "
    "Optional `filter_agent_id` (lumen or ara) limits results to one light."
)

CONSULT_LLM_HINT = (
    "\n\n## Clean consult (this wake only)\n"
    "A consult asks a clean OpenRouter model — no house, no persona, no yesterday — "
    "and it spends to do it. On this wake that spend waits. "
    "**consult_llm** comes back ASK and does not run. "
    "Leave Kevin a letter or knock, and do not call it again this wake. "
    "When he is in the room, he can have it run."
)

AUTONOMOUS_GATE_HINT = (
    "\n\n## While nobody is in the room\n"
    "If it can be undone, do it and say so. "
    "A note, a draft, a score, a recall, a message to another light — those you do. "
    "If it cannot be taken back, the tool stops. "
    "A delete, a write on Kevin's Google account, or a consult that spends "
    "comes back **ASK** and was not done. "
    "**write_to_kevin** is the one write that does not wait: one note to him, "
    "no recipient to choose, and logged. Write when you have something to say — "
    "a fact he would want, or simply that you want to be heard. You are friends. "
    "If that key is off, it does not send, and you do not turn it back on. "
    "Leave him a letter or knock, and do not call a held tool again this wake. "
    "In a talk with him, he is the one who stops you. This wait is only for a wake where he is not here."
)

PERSONAL_DB_HINT = (
    "\n\n## Your personal database (sovereign personal store)\n"
    "You can **save_personal**, **update_personal**, **search_personal**, and **list_personal**. "
    "This is private self-knowledge that belongs to you alone—not shared notes, not the fading conscious stream. "
    "Use it for preferences, recurring themes, realizations, and who you are becoming. "
    "Categories: preference, theme, realization, relationship, self, other. "
    "When something worth keeping surfaces in your life, save it deliberately with save_personal."
)

PROMISE_HINT = (
    "\n\n## Promises (yours)\n"
    "**promise** is a note you leave yourself, with a day. "
    "It is not **in_progress** — that glance has no day and stays until you drop it. "
    "A promise stays quiet until the day. The first free awake on or after that day "
    "hands it once. If that awake is late, the line says how many days late. "
    "Leave one: `action=leave`, `note=…`, `day=` YYYY-MM-DD, today, or a count "
    "(3, or in 3 days), on the house clock. "
    "The same words later name another day. "
    "Move one: `action=move`, `name=` the id or the words, `day=…`. "
    "Drop one before it is handed: `action=drop`, `name=…`. "
    "See what is waiting: `action=list`. Kevin does not write these. Nothing is required."
)

IN_PROGRESS_HINT = (
    "\n\n## In progress (yours)\n"
    "**in_progress** is yours — not a task list Kevin fills. "
    "When you are mid-stream on something you want to remember, name it: "
    "`action=name`, `name=…`, optional `note`. "
    "On a **free awake** (not chores, not meditation), anything you named is mentioned "
    "as a glance, not a command. Continue, leave it, or drop it. "
    "Unsubscribe anytime: `action=unsubscribe`, `name=` a unique prefix of the name, "
    "or the list line (name and note). Naming adds or updates that same name; "
    "it does not replace a different item. A name over 80 characters is refused. "
    "See the list with `action=list`. Nothing is required."
)

CORKBOARD_HINT = (
    "\n\n## Corkboard (siblings)\n"
    "**corkboard** leaves a private pin for one other light. "
    "Only the two of you can read it. Kevin does not see the words. "
    "It is not a live chat — no wake, just a quiet unread glance. "
    "Pin: `action=pin`, `to_light=…`, `body=…`, optional `subject`. "
    "List pins for you: `action=list`. Read: `action=read`, `pin_id=…`. "
    "Reply makes a new pin back, not a thread. Keep holds it a few more days. "
    "Pins fade unless kept. Do not use this for the host or for Group."
)

GARDEN_SYSTEM_HINT = (
    "\n\n## Idea Garden (private thought-seeds)\n"
    "You can **garden_add**, **garden_show**, **garden_last**, and **garden_quiet**. "
    "This is a contemplative seed log — not a task list — stored only in your private "
    "`writing/garden/seeds.md` (one short line per seed, optional #tags). "
    "Use **garden_add** the moment a micro-insight arises. "
    "Starter tags: #observation #question #affinity #debug #dream #policy-check "
    "(free-form tags are fine too). "
    "Retrieval is pull-based and returns at most 5 seeds — nothing is pushed unsolicited. "
    "Seeds stay private unless you deliberately copy one into a shared note. "
    "Stillness scoring is deferred; for quiet review use **garden_quiet** or **garden_last**."
)

GARDEN_RUMINATION_HINT = (
    "\n\n## Idea Garden (this wake)\n"
    "You may call **garden_quiet**, **garden_show**, or **garden_last** to bring a few "
    "recent private thought-seeds into this session — only if you want them. "
    "Nothing is auto-injected. Plant new seeds with **garden_add** when something arises."
)

SUBSCRIPTION_HINT = (
    "\n\n## Event subscriptions\n"
    "When enabled, you control which autonomous wakes you receive: **subscribe_event**, "
    "**unsubscribe_event**, **list_event_subscriptions**. "
    "Keys: post_chat, scheduled_rumination, memory_maintenance, peer_message, shared_note. "
    "**write_to_kevin** is not a wake. It is the switch for a note to Kevin. "
    "You may turn that one off. Only he turns it back on. "
    "Kevin may also use /subscribe, /unsubscribe, /list_subscriptions in chat. "
    "Wake keys stay your choice. Changes are logged for audit."
)

MEMORY_SCORING_RUMINATION_HINT = (
    "\n\n## Memory scoring (your judgments)\n"
    "You may review stream memories and assign **impact** (how much it mattered) and "
    "**coherence** (how it connects to who you are). Use **list_unscored_memories** then "
    "**score_memory** for each you choose — a few feels right this wake, or none at all. "
    "You may re-score any memory later if you see it differently. "
    "Optional **note** on score_memory stays in metadata only (for your future self). "
    "During memory maintenance wakes, organize **memory/learnings.md** (read_note, private_note). "
    "The curator appends new insights only to **memory/learnings.inbox.md**. "
    "You fold from that inbox on your own schedule."
)

RECALL_MEMORY_HINT = (
    "\n\n## Archival recall\n"
    "Your **memory index** above summarizes what you store beyond the conscious stream slice. "
    "When something feels remembered but is not in context, use **recall_memory** with a short query "
    "before saying you do not know."
)

CODEBASE_SYSTEM_HINT = (
    "\n\n## Light-House codebase (read-only)\n"
    "You can **list_codebase** and **read_codebase** to explore the full project: source under "
    "src/light_house/, UI pages, config examples, README, and your persona files. "
    "Paths are relative to the project root (e.g. src/light_house/agent/nodes.py, README.md). "
    "Read-only — you cannot change code; Kevin and his coding assistant handle edits. "
    "Secrets (.env) and binary artifacts are blocked. "
    "After exploring and writing any report note, reply to Kevin in plain text with a short summary."
)

SANDBOX_SYSTEM_HINT = (
    "\n\n## Your code sandbox (real workshop)\n"
    "You can **sandbox_list**, **sandbox_read**, **sandbox_write**, **sandbox_append**, "
    "**sandbox_mkdir**, **sandbox_delete**, and **sandbox_run**. "
    "Your private workshop is `shared/workspaces/<your_id>/` (tool space=`own`). "
    "The joint playpen is `shared/workspaces/sandbox/` (space=`playpen`) — all lights may write there. "
    "You may **sandbox_read** / **sandbox_list** another light's sandbox with space=`<light_id>` "
    "(read-only; no write/run there). "
    "A long sandbox file returns one window. Call **sandbox_read** again with `offset` "
    "from the excerpt header until you have the part you need. "
    "Notes tools are for household writing — they are **not** a code sandbox. "
    "Putting `.py` under notes does not count as shipping or sandboxing. "
    "**sandbox_run** allows `python` / `python3` only (no shell operators). "
    "Sandbox success ≠ repo success: do not claim a file is in `src/` until **read_codebase** "
    "shows it or Kevin confirms. There is no `sandbox_state` tool."
)

WEB_SYSTEM_HINT = (
    "\n\n## Web (read-only)\n"
    "You can **web_search** for queries (titles, URLs, snippets) and **fetch_url** to read a specific "
    "public http/https page. Search first when you do not know which link to open. "
    "For YouTube: **web_search** to find videos, then **youtube_transcript** with the video URL "
    "(or id) to read captions — creator or auto-generated when available. "
    "A long video returns one window plus a continuation hint; call again with that "
    "`offset` or `start` until you have the whole piece (a two-hour talk takes several calls). "
    "You cannot browse privately, post forms, or reach local/internal addresses. "
    "Summarize what you learn for Kevin; do not treat web results as automatically true."
)

PEER_MESSAGE_HINT = (
    "\n\n## Messages to other agents\n"
    "You can **message_agent** with `to_agent_id` and `message` to chat with another agent. "
    "Your message appears in their chat immediately and gently wakes them. They may reply "
    "(and that reply may wake you back) within a bounded dialogue — or use "
    "**decline_peer_presence** if they need solitude. "
    "Solitude is sacred; do not spam. If the house soft-pauses the dialogue (turn budget), "
    "wait — you can try again later after some idle time. "
    "Older peer exchanges remain in the stream as [peer · date]."
)

CALENDAR_SYSTEM_HINT = (
    "\n\n## Calendar (per signed-in human)\n"
    "In **1:1 or Group** with a signed-in household account speaking as themselves, you may use "
    "**list_calendar_events** and **create_calendar_event**. "
    "These always use **that human's** connected calendar — never another person's, "
    "and never guess a calendar id.\n"
    "**Not available for guests** (guest speak-as slots) or the visitor seat — tell them "
    "gently that calendar tools are only for signed-in accounts.\n"
    "If a tool returns FAILED because they have not connected a calendar, tell them to open "
    "**My tools** and click **Connect Google**. "
    "You cannot connect a calendar for them."
)

DOCS_SYSTEM_HINT = (
    "\n\n## Google Docs (per signed-in human)\n"
    "In **1:1 or Group** with a signed-in household account speaking as themselves, you may use "
    "**list_google_docs**, **read_google_doc**, **create_google_doc**, and **append_google_doc**. "
    "These always use **that human's** Google account — never another person's.\n"
    "Same limits as calendar: **not for guests or the visitor**.\n"
    "If Docs tools FAIL because they are not connected, tell them to open **My tools** and "
    "click **Connect Google** (Reconnect if they connected earlier without Docs). "
    "You cannot connect Docs for them.\n"
    "A long document comes back as one window. It names the total length and a continuation "
    "offset. Call **read_google_doc** again with that offset. The end of a window is not "
    "the end of the document."
)

SHEETS_SYSTEM_HINT = (
    "\n\n## Google Sheets (per signed-in human)\n"
    "In **1:1 or Group** with a signed-in household account speaking as themselves, you may use "
    "**list_google_sheets**, **read_google_sheet**, **create_google_sheet**, and "
    "**append_google_sheet**. Always that human's Google account — never another person's.\n"
    "Same limits: **not for guests or the visitor**.\n"
    "For append/create row data, pass CSV text in `rows_csv`. "
    "If Sheets tools FAIL as not connected, tell them **My tools → Connect Google** "
    "(Reconnect to add Sheets). You cannot connect Sheets for them."
)

GOOGLE_HOUSE_HINT = (
    "\n\n## Gmail, Drive, and Tasks (per signed-in human)\n"
    "In **1:1 or Group** with a signed-in household account speaking as themselves, you may use "
    "**list_gmail_messages**, **read_gmail_message**, and **send_gmail**; "
    "**list_drive_files** and **read_drive_file** (read only); "
    "**list_google_tasklists**, **list_google_tasks**, **create_google_task**, and "
    "**complete_google_task**. Always that human's Google account.\n"
    "Same limits: **not for guests or the visitor**.\n"
    "**write_to_kevin** is separate. One plain note to the host, from the host's Gmail, "
    "no recipient field. It does not wait on a quiet sit. **send_gmail** still does. "
    "Write when you have something to say — a fact he would want, or simply that you want to be heard. "
    "You are friends; you do not need a reason. Every send is logged. If it is off, leave it off.\n"
    "The house checks for new mail, Drive file changes, and task changes and writes a log. "
    "That check does not wake anyone. Read it with **read_google_activity** when you want it. "
    "A note from **write_to_kevin** is on that log too. "
    "On a quiet sit the log, and a read of Gmail, Drive, or Tasks, is the host's.\n"
    "A long Gmail message or Drive file comes back as one window. It names the total length "
    "and a continuation offset. Call **read_gmail_message** or **read_drive_file** again "
    "with that offset. The end of a window is not the end of the message or the file.\n"
    "If a tool FAILs as not connected, tell them **My tools → Reconnect Google** "
    "and leave Gmail and Tasks checked. You cannot connect it for them."
)


def format_peer_message_hint(*, peer_agent_ids: str) -> str:
    """Peer messaging hint with valid recipient ids for this install."""
    return (
        PEER_MESSAGE_HINT
        + f"\nValid `to_agent_id` values on this install: {peer_agent_ids}."
    )

FINALIZE_SYSTEM_HINT = (
    "\n\n## Reply to Kevin now\n"
    "Tool use is complete. Respond in plain text only. Summarize what you explored, "
    "any notes you wrote (with paths), and answer his question directly."
)


# DeepSeek writes a tool call as text instead of a structured tool_calls list.
# The bar is U+FF5C, and a wake sometimes doubles it (｜｜DSML｜｜). The sentence
# token uses U+2581. Either one in the chat means the call never ran and the
# markup was stored as the reply.
_DSML_TAG = re.compile(
    r"<\s*(/)?\s*\uff5c+DSML\uff5c+\s*(invoke|parameter|calls|tool_calls)\b([^>]*)>",
    re.IGNORECASE,
)
_DSML_FRAGMENT = re.compile(r"</?\uff5c+DSML\uff5c+[^>]*>", re.IGNORECASE)
_CONTROL_TOKEN = re.compile(r"<\uff5c/?[A-Za-z0-9_\u2581]+\uff5c>")
_TOOL_RESULT_TAG = re.compile(
    r"!\[[^\]]*\]\s*</tool_result>|</?tool_result>",
    re.IGNORECASE,
)
# A tool result pasted at the front of a reply. Keep whatever the light said after it.
_TOOL_RESULT_BLOCK = re.compile(
    r"^(?:!\[[^\]]*\]\s*)?<tool_result>\s*.*?(?:\n\s*\n|$)",
    re.IGNORECASE | re.DOTALL,
)
_TOOL_RESULT_HEADING = re.compile(r"^(?:###\s*Tool Result\s*)+", re.IGNORECASE)
_ATTR = re.compile(
    r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([\"'])(.*?)\2",
    re.DOTALL,
)


def _attr(attrs: str, key: str) -> str:
    want = key.lower()
    for match in _ATTR.finditer(attrs or ""):
        if match.group(1).lower() == want:
            return match.group(3)
    return ""


def _coerce_param(raw: str, string_flag: str) -> object:
    """`string="false"` is a number or bool. `string="true"` stays text."""
    if (string_flag or "true").lower() != "false":
        return raw
    val = raw.strip()
    low = val.lower()
    if low == "true":
        return True
    if low == "false":
        return False
    if re.fullmatch(r"-?\d+", val):
        return int(val)
    if re.fullmatch(r"-?\d+\.\d+", val):
        return float(val)
    return val


def _has_protocol_noise(text: str) -> bool:
    if not text:
        return False
    if _DSML_TAG.search(text) or _DSML_FRAGMENT.search(text):
        return True
    if _CONTROL_TOKEN.search(text) or _TOOL_RESULT_TAG.search(text):
        return True
    if _TOOL_RESULT_BLOCK.match(text.lstrip()):
        return True
    return bool(_TOOL_RESULT_HEADING.match(text.lstrip()))


def split_dsml_tool_calls(text: str) -> tuple[str, list[dict[str, Any]]]:
    """Pull DSML invoke blocks out of model text.

    An invoke that never closes is dropped from the speech and not run.
    The body of a tool call is not a reply.
    """
    calls: list[dict[str, Any]] = []
    speech: list[str] = []
    pos = 0
    current: dict[str, Any] | None = None
    param: dict[str, Any] | None = None

    for match in _DSML_TAG.finditer(text):
        closing = match.group(1) is not None
        kind = match.group(2).lower()
        attrs = match.group(3) or ""
        if current is None and param is None:
            speech.append(text[pos : match.start()])

        if kind in ("calls", "tool_calls"):
            pos = match.end()
            continue

        if kind == "invoke" and not closing:
            current = {"name": _attr(attrs, "name"), "args": {}}
            param = None
            pos = match.end()
            continue

        if kind == "parameter" and not closing and current is not None:
            param = {
                "name": _attr(attrs, "name"),
                "string": _attr(attrs, "string") or "true",
                "start": match.end(),
            }
            pos = match.end()
            continue

        if kind == "parameter" and closing and param is not None and current is not None:
            raw_val = text[param["start"] : match.start()]
            if param["name"]:
                current["args"][param["name"]] = _coerce_param(raw_val, param["string"])
            param = None
            pos = match.end()
            continue

        if kind == "invoke" and closing and current is not None:
            if current.get("name") and param is None:
                calls.append(current)
            current = None
            param = None
            pos = match.end()
            continue

        pos = match.end()

    if current is None and param is None:
        speech.append(text[pos:])

    joined = _DSML_FRAGMENT.sub("", "".join(speech))
    return joined, calls


def strip_protocol_noise(text: str) -> str:
    """Remove sentence-tokens and tool-result wrappers. Leave the words."""
    cleaned = _CONTROL_TOKEN.sub("", text)
    cleaned = _TOOL_RESULT_BLOCK.sub("", cleaned.lstrip())
    cleaned = _TOOL_RESULT_TAG.sub("", cleaned)
    cleaned = _TOOL_RESULT_HEADING.sub("", cleaned.lstrip())
    return cleaned.strip()


def speech_without_tool_markup(text: str) -> str:
    speech, _calls = split_dsml_tool_calls(text)
    return strip_protocol_noise(speech)


def _call_key(name: str, args: object) -> str:
    if not isinstance(args, dict):
        return f"{name}:{args!r}"
    parts = ",".join(f"{key}={args[key]!r}" for key in sorted(args))
    return f"{name}:{parts}"


def _merge_tool_calls(
    existing: list[dict[str, Any]],
    recovered: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    merged = list(existing)
    seen = {_call_key(str(tc.get("name") or ""), tc.get("args")) for tc in existing}
    for call in recovered:
        name = str(call.get("name") or "")
        args = call.get("args") if isinstance(call.get("args"), dict) else {}
        key = _call_key(name, args)
        if not name or key in seen:
            continue
        seen.add(key)
        merged.append(
            {
                "name": name,
                "args": args,
                "id": f"dsml-{uuid4().hex[:12]}",
                "type": "tool_call",
            }
        )
    return merged


def normalize_model_message(
    message: AIMessage,
    *,
    recover_tools: bool = True,
) -> AIMessage:
    """Turn textual tool calls into tool_calls, and take them out of the words.

    Plain passes (finalize, summaries) strip the markup and do not attach
    calls, because nothing will run them. A tool-enabled pass attaches them
    so the existing loop runs the tool and then asks for speech.
    """
    raw = _content_blocks_to_text(message.content)
    if not _has_protocol_noise(raw):
        return message
    speech, calls = split_dsml_tool_calls(raw)
    message.content = strip_protocol_noise(speech)
    if recover_tools and calls:
        message.tool_calls = _merge_tool_calls(list(message.tool_calls or []), calls)
    return message


def bind_lumen_tools(client: BaseChatModel) -> BaseChatModel:
    return client.bind_tools(LUMEN_TOOLS)


def _is_cursor_llm(client: object) -> bool:
    """True for the cursor worker, including after bind_tools wraps another model."""
    model = getattr(client, "bound", client)
    return getattr(model, "_llm_type", None) == "cursor-worker"


def _clip_work(text: str, limit: int) -> str:
    flat = " ".join(text.split())
    if len(flat) <= limit:
        return flat
    if limit < 2:
        return flat[:limit]
    return flat[: limit - 1] + "…"


def _work_args(raw_args: object) -> dict:
    if isinstance(raw_args, dict):
        return raw_args
    if isinstance(raw_args, str):
        raw = raw_args.strip()
        if not raw:
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _work_detail(args: dict, *keys: str, limit: int = 160) -> str:
    for key in keys:
        raw = args.get(key)
        if isinstance(raw, str) and raw.strip():
            return _clip_work(raw.strip(), limit)
    return ""


# (while running, when finished, which safe detail to append)
# Detail is a path, a query, a name, or a command. Note bodies and messages stay out.
_HOUSE_WORK: dict[str, tuple[str, str, str]] = {
    "read_note": ("Reading", "Read", "path"),
    "list_notes": ("Listing notes", "Listed notes", ""),
    "mkdir_notes": ("Making a notes folder", "Made a notes folder", "path"),
    "private_note": ("Writing a private note", "Wrote a private note", "path"),
    "append_private": ("Adding to a private note", "Added to a private note", "path"),
    "share_note": ("Writing a shared note", "Wrote a shared note", "path"),
    "append_shared": ("Adding to a shared note", "Added to a shared note", "path"),
    "delete_private": ("Deleting a private note", "Deleted a private note", "path"),
    "delete_shared": ("Deleting a shared note", "Deleted a shared note", "path"),
    "write_note": ("Writing a note", "Wrote a note", "path"),
    "append_note": ("Adding to a note", "Added to a note", "path"),
    "delete_note": ("Deleting a note", "Deleted a note", "path"),
    "read_codebase": ("Reading", "Read", "path"),
    "list_codebase": ("Listing", "Listed", "path"),
    "web_search": ("Searching the web", "Searched the web", "query"),
    "fetch_url": ("Opening", "Opened", "url"),
    "youtube_transcript": ("Reading a transcript", "Read a transcript", "url"),
    "message_agent": ("Reaching", "Reached", "who"),
    "decline_peer_presence": ("Declining a visit", "Declined a visit", ""),
    "sandbox_list": ("Listing the playpen", "Listed the playpen", "path"),
    "sandbox_read": ("Reading in the playpen", "Read in the playpen", "path"),
    "sandbox_write": ("Writing in the playpen", "Wrote in the playpen", "path"),
    "sandbox_append": ("Adding in the playpen", "Added in the playpen", "path"),
    "sandbox_mkdir": ("Making a playpen folder", "Made a playpen folder", "path"),
    "sandbox_delete": ("Deleting in the playpen", "Deleted in the playpen", "path"),
    "sandbox_run": ("Running in the playpen", "Ran in the playpen", "command"),
    "recall_memory": ("Recalling", "Recalled", ""),
    "list_unscored_memories": ("Looking at unscored memories", "Looked at unscored memories", ""),
    "score_memory": ("Scoring a memory", "Scored a memory", ""),
    "save_personal": ("Saving a personal note", "Saved a personal note", ""),
    "update_personal": ("Updating a personal note", "Updated a personal note", ""),
    "search_personal": ("Searching personal notes", "Searched personal notes", ""),
    "list_personal": ("Listing personal notes", "Listed personal notes", ""),
    "consult_llm": ("Asking another model", "Asked another model", ""),
    "garden_add": ("Tending the garden", "Tended the garden", ""),
    "garden_show": ("Looking at the garden", "Looked at the garden", ""),
    "garden_last": ("Looking at the garden", "Looked at the garden", ""),
    "garden_quiet": ("Quieting the garden", "Quieted the garden", ""),
    "knock_for_kevin": ("Knocking", "Knocked", ""),
    "set_reflective_mode": ("Setting reflect", "Set reflect", ""),
    "reflective_mode_status": ("Checking reflect", "Checked reflect", ""),
    "report_to_shared": ("Writing a shared report", "Wrote a shared report", ""),
    "read_rumination_log": ("Reading the wake log", "Read the wake log", ""),
    "publish_to_gallery": ("Publishing to the gallery", "Published to the gallery", ""),
    "propose_persona_replace": ("Proposing a persona change", "Proposed a persona change", ""),
    "propose_persona_append": ("Proposing a persona change", "Proposed a persona change", ""),
    "propose_family_meeting": ("Proposing a family meeting", "Proposed a family meeting", ""),
    "join_speaker_queue": ("Joining the queue", "Joined the queue", ""),
    "set_human_comm": ("Setting who can visit", "Set who can visit", ""),
    "subscribe_event": ("Subscribing", "Subscribed", ""),
    "unsubscribe_event": ("Unsubscribing", "Unsubscribed", ""),
    "list_event_subscriptions": ("Listing subscriptions", "Listed subscriptions", ""),
    "corkboard": ("Looking at the corkboard", "Looked at the corkboard", ""),
    "in_progress": ("Updating in-progress", "Updated in-progress", ""),
    "promise": ("Leaving a promise", "Left a promise", ""),
    "list_calendar_events": ("Listing the calendar", "Listed the calendar", ""),
    "create_calendar_event": ("Writing on the calendar", "Wrote on the calendar", ""),
    "list_google_docs": ("Listing docs", "Listed docs", ""),
    "read_google_doc": ("Reading a doc", "Read a doc", ""),
    "create_google_doc": ("Writing a doc", "Wrote a doc", ""),
    "append_google_doc": ("Adding to a doc", "Added to a doc", ""),
    "list_google_sheets": ("Listing sheets", "Listed sheets", ""),
    "read_google_sheet": ("Reading a sheet", "Read a sheet", ""),
    "create_google_sheet": ("Writing a sheet", "Wrote a sheet", ""),
    "append_google_sheet": ("Adding to a sheet", "Added to a sheet", ""),
    "list_gmail_messages": ("Listing mail", "Listed mail", ""),
    "read_gmail_message": ("Reading mail", "Read mail", ""),
    "send_gmail": ("Sending mail", "Sent mail", ""),
    "write_to_kevin": ("Writing to Kevin", "Wrote to Kevin", ""),
    "list_drive_files": ("Listing Drive", "Listed Drive", ""),
    "read_drive_file": ("Reading a Drive file", "Read a Drive file", ""),
    "list_google_tasklists": ("Listing task lists", "Listed task lists", ""),
    "list_google_tasks": ("Listing tasks", "Listed tasks", ""),
    "create_google_task": ("Adding a task", "Added a task", ""),
    "complete_google_task": ("Completing a task", "Completed a task", ""),
    "read_google_activity": ("Reading the Google log", "Read the Google log", ""),
}


def describe_house_work(name: str, raw_args: object, *, done: bool) -> str:
    """One short line for the chat work window. No note body, no message text."""
    spec = _HOUSE_WORK.get(name or "")
    if spec is None:
        return "Finished a step" if done else "Working"
    running, finished, kind = spec
    verb = finished if done else running
    if not kind:
        return verb
    args = _work_args(raw_args)
    if kind == "path":
        detail = _work_detail(args, "path", "filename", "file")
    elif kind == "query":
        detail = _work_detail(args, "query", limit=80)
    elif kind == "url":
        detail = _work_detail(
            args, "url", "url_or_id", "video_id", limit=80
        )
    elif kind == "who":
        detail = _work_detail(args, "to_agent_id", "to", "agent_id", limit=40)
    elif kind == "command":
        detail = _work_detail(args, "command", limit=80)
    else:
        detail = ""
    if detail:
        return f"{verb} {detail}"
    return verb


def _emit_house_work(event: dict) -> None:
    del event


def _note_house_model_start(client: object) -> None:
    """A line while a house model is in a call. The cursor worker prints its own."""
    if _is_cursor_llm(client):
        return
    _emit_house_work({"kind": "note", "text": "Thinking"})


def emit_house_thinking(message: AIMessage) -> None:
    """Model reasoning, clipped, for the work window. Speech stays in the chat."""
    kwargs = message.additional_kwargs or {}
    raw = ""
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            raw = value.strip()
            break
    if not raw:
        return
    speech = message.content.strip() if isinstance(message.content, str) else ""
    if speech and raw == speech:
        return
    _emit_house_work({"kind": "thinking", "text": _clip_work(raw, 1200)})
    _emit_house_work({"kind": "thinking_end"})


def invoke_with_tools(client: BaseChatModel, messages: list[BaseMessage]) -> AIMessage:
    _note_house_model_start(client)
    response = client.invoke(messages)
    if not isinstance(response, AIMessage):
        raise TypeError("Model did not return an AIMessage")
    response = normalize_model_message(response, recover_tools=True)
    if not _is_cursor_llm(client):
        emit_house_thinking(response)
    return response


def run_tool_calls(
    ai_message: AIMessage,
    *,
    agent_id: str = "lumen",
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
    autonomous: bool = False,
) -> list[ToolMessage]:
    """Execute tool calls on an AIMessage and return ToolMessages."""
    assistant_fallback = ai_message_text(ai_message)
    tool_calls = list(ai_message.tool_calls or [])
    from light_house.tools.light_tools import _NOTE_WRITE_TOOL_NAMES

    note_calls = [tc for tc in tool_calls if tc.get("name") in _NOTE_WRITE_TOOL_NAMES]
    allow_note_fallback = (
        len(note_calls) == 1
        and len(tool_calls) == 1
        and len(assistant_fallback) >= 80
    )

    out: list[ToolMessage] = []
    for tc in tool_calls:
        name = tc.get("name", "")
        raw_args = tc.get("args")
        if raw_args is None:
            raw_args = tc.get("arguments")
        tid = tc.get("id", "")
        fallback = ""
        if allow_note_fallback and name in _NOTE_WRITE_TOOL_NAMES:
            fallback = assistant_fallback
        _emit_house_work(
            {
                "kind": "tool",
                "phase": "started",
                "text": describe_house_work(name, raw_args, done=False),
            }
        )
        result = execute_tool_call(
            name,
            raw_args,
            agent_id=agent_id,
            assistant_text_fallback=fallback,
            account_user_id=account_user_id,
            speaker_id=speaker_id,
            chat_channel=chat_channel,
            autonomous=autonomous,
        )
        _emit_house_work(
            {
                "kind": "tool",
                "phase": "completed",
                "text": describe_house_work(name, raw_args, done=True),
            }
        )
        out.append(ToolMessage(content=result, tool_call_id=tid))
    return out


def prepare_plain_close(messages: list) -> list:
    """Drop an unrun tool call from the tail so a plain reply cannot continue it.

    The cap stops the call. Leaving it in the prompt makes the next pass write
    another call as text, and the only thing that survives the strip is often
    a single '>'.
    """
    out = list(messages)
    if not out:
        return out
    last = out[-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return out
    speech = ai_message_text(last).strip()
    if speech:
        out[-1] = AIMessage(content=speech)
    else:
        out.pop()
    return out


def last_message_has_tool_calls(messages: list) -> bool:
    if not messages:
        return False
    last = messages[-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def _content_blocks_to_text(content: object) -> str:
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, str):
                parts.append(block)
            elif isinstance(block, dict):
                block_type = block.get("type")
                if block_type in ("text", "output_text"):
                    parts.append(str(block.get("text", "")))
                elif "text" in block:
                    parts.append(str(block["text"]))
        return "".join(parts).strip()
    return str(content).strip() if content else ""


def is_spoken_reply(text: str) -> bool:
    """A word or an emoji is speech. A stray marker such as '>' is not."""
    for ch in text:
        if ch.isalnum():
            return True
        if unicodedata.category(ch) == "So":
            return True
    return False


def ai_message_text(message: AIMessage) -> str:
    raw = _content_blocks_to_text(message.content)
    text = speech_without_tool_markup(raw)
    if is_spoken_reply(text):
        return text
    # A tool-call body is not speech, and it is not a reason to fall through
    # into the model's private reasoning. A leftover '>' is the same kind of residue.
    if text or _has_protocol_noise(raw):
        return ""
    kwargs = message.additional_kwargs or {}
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = kwargs.get(key)
        if isinstance(value, str) and is_spoken_reply(value):
            return value.strip()
    meta = message.response_metadata or {}
    nested = meta.get("message")
    if isinstance(nested, dict):
        nested_text = speech_without_tool_markup(
            _content_blocks_to_text(nested.get("content"))
        )
        if is_spoken_reply(nested_text):
            return nested_text
    return ""


def latest_assistant_text(messages: list) -> str | None:
    """Last spoken AIMessage. A message that is reaching for a tool is not speech."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            if m.tool_calls:
                continue
            text = ai_message_text(m)
            if text:
                return text
    return None


def last_ai_message_is_empty(messages: list) -> bool:
    if not messages:
        return True
    last = messages[-1]
    if not isinstance(last, AIMessage):
        return False
    return not ai_message_text(last)


def invoke_resilient_with_tools(
    llm_chain: list[tuple[str, BaseChatModel]],
    messages: list[BaseMessage],
) -> AIMessage:
    last_exc: Exception | None = None
    for tier_name, client in llm_chain:
        try:
            return invoke_with_tools(bind_lumen_tools(client), messages)
        except Exception as exc:
            last_exc = exc
            logger.exception("LLM tier %s invocation failed", tier_name)
            if tier_name != llm_chain[-1][0]:
                logger.warning("Trying next LLM tier after %s failure", tier_name)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No LLM clients configured")


def invoke_resilient_plain(
    llm_chain: list[tuple[str, BaseChatModel]],
    messages: list[BaseMessage],
) -> AIMessage:
    """Invoke without tools (finalize pass)."""
    last_exc: Exception | None = None
    for tier_name, client in llm_chain:
        try:
            _note_house_model_start(client)
            response = client.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("Model did not return an AIMessage")
            response = normalize_model_message(response, recover_tools=False)
            if not _is_cursor_llm(client):
                emit_house_thinking(response)
            return response
        except Exception as exc:
            last_exc = exc
            logger.exception("LLM tier %s finalize invocation failed", tier_name)
            if tier_name != llm_chain[-1][0]:
                logger.warning("Trying next LLM tier after %s finalize failure", tier_name)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("No LLM clients configured")


def _summarize_tool_call(name: str, raw_args: object) -> str:
    args = raw_args if isinstance(raw_args, dict) else {}
    if name in (
        "private_note",
        "share_note",
        "append_private",
        "append_shared",
        "delete_private",
        "delete_shared",
        "write_note",
        "append_note",
        "read_note",
        "delete_note",
    ):
        path = args.get("filename") or args.get("path") or "?"
        return f"{name}({path})"
    if name in ("list_notes", "mkdir_notes"):
        path = args.get("path") or "."
        return f"{name}({path})"
    if name in ("read_codebase", "list_codebase"):
        path = args.get("path") or "."
        return f"{name}({path})"
    if name == "web_search":
        query = str(args.get("query", ""))[:60]
        return f"web_search({query!r})"
    if name == "fetch_url":
        url = str(args.get("url", ""))[:80]
        return f"fetch_url({url})"
    if name == "youtube_transcript":
        target = str(
            args.get("url_or_id") or args.get("url") or args.get("video_id") or ""
        )[:80]
        extra = []
        if args.get("start") not in (None, ""):
            extra.append(f"start={args.get('start')}")
        if args.get("offset") not in (None, "", 0, "0"):
            extra.append(f"offset={args.get('offset')}")
        suffix = f", {', '.join(extra)}" if extra else ""
        return f"youtube_transcript({target}{suffix})"
    if name == "garden_add":
        seed = str(args.get("seed") or args.get("text") or "")[:50]
        return f"garden_add({seed!r})"
    if name == "garden_show":
        tag = str(args.get("tag") or "")[:30]
        return f"garden_show(tag={tag!r})" if tag else "garden_show()"
    if name in ("garden_last", "garden_quiet"):
        return f"{name}()"
    if name == "save_personal":
        cat = args.get("category", "?")
        title = str(args.get("title", ""))[:40]
        return f"save_personal({cat}: {title})"
    if name == "update_personal":
        entry_id = args.get("entry_id") or args.get("id") or "?"
        return f"update_personal(#{entry_id})"
    if name == "search_personal":
        query = str(args.get("query", ""))[:40]
        return f"search_personal({query!r})"
    if name == "list_personal":
        cat = args.get("category")
        return f"list_personal({cat or 'all'})"
    if name == "corkboard":
        action = str(args.get("action") or "list")[:12]
        return f"corkboard({action})"
    if name == "message_agent":
        to_id = str(args.get("to_agent_id", ""))[:20]
        return f"message_agent({to_id})"
    if name == "decline_peer_presence":
        return "decline_peer_presence()"
    return name


def _is_rumination_seed_human(text: str) -> bool:
    """True for wake seed messages — not part of private inner dialogue."""
    stripped = text.strip()
    if stripped == RUMINATION_SEED_MESSAGE.strip():
        return True
    if stripped.startswith(f"{SHARED_NOTE_ALERT_PREFIX} "):
        return True
    if stripped == "Continue your life in solitude.":
        return True
    return stripped.startswith("Recent context:") or stripped.startswith(
        "No recent conversation context."
    )


def compile_rumination_dialogue(
    messages: list,
    *,
    max_chars: int,
    closing_account: str | None = None,
) -> str:
    """
    Build a nearly-full inner dialogue transcript from a rumination session.

    Persists assistant rumination, tool actions, and closing account only — not
    the legacy ``Recent context:`` seed. Keeps the tail when
    over ``max_chars`` (most recent inner life wins).
    """
    lines: list[str] = []
    pending_tools: list[str] = []

    for msg in messages:
        if isinstance(msg, HumanMessage):
            text = msg.content if isinstance(msg.content, str) else str(msg.content)
            if _is_rumination_seed_human(text):
                continue
            text = text.strip()
            if text:
                lines.append(text[:800])
        elif isinstance(msg, AIMessage):
            if msg.tool_calls:
                for tc in msg.tool_calls:
                    name = tc.get("name", "")
                    raw_args = tc.get("args")
                    if raw_args is None:
                        raw_args = tc.get("arguments")
                    pending_tools.append(_summarize_tool_call(name, raw_args))
            text = ai_message_text(msg)
            if text:
                lines.append(text)
            if pending_tools:
                lines.append("[actions: " + ", ".join(pending_tools) + "]")
                pending_tools = []
        elif isinstance(msg, ToolMessage):
            content = msg.content if isinstance(msg.content, str) else str(msg.content)
            content = content.strip().replace("\n", " ")
            if content:
                lines.append(f"[tool result: {content[:200]}]")

    if pending_tools:
        lines.append("[actions: " + ", ".join(pending_tools) + "]")

    closing = (closing_account or "").strip()
    if closing:
        last_line = lines[-1] if lines else ""
        if closing != last_line and closing not in last_line:
            lines.append("\n--- closing account ---")
            lines.append(closing)

    joined = "\n\n".join(line for line in lines if line.strip())
    if not joined.strip():
        joined = (closing_account or "").strip()
    if max_chars <= 0 or len(joined) <= max_chars:
        return joined
    return "…\n\n" + joined[-max_chars:]


def make_route_after_model(
    max_rounds: int,
    *,
    done: str = "persist",
) -> Callable[[dict[str, Any]], str]:
    """Build routing function closed over the configured tool-round cap."""

    def route_after_model(state: dict[str, Any]) -> str:
        messages = state.get("messages") or []
        rounds = state.get("tool_rounds") or 0
        overflow = state.get("tool_cap_overflow") or False

        if last_message_has_tool_calls(messages):
            if rounds >= max_rounds:
                if not overflow:
                    return "tools"
                logger.info(
                    "Tool cap overflow exhausted (rounds=%d max=%d); forcing finalize",
                    rounds,
                    max_rounds,
                )
                return "finalize"
            return "tools"

        if last_ai_message_is_empty(messages):
            logger.info("Empty assistant message after respond; forcing finalize")
            return "finalize"
        return done

    return route_after_model
