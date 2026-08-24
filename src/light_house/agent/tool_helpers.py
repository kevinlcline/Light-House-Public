"""Tool-calling helpers for Lumen's chat and rumination loops."""

from __future__ import annotations

import logging
from collections.abc import Callable
from typing import Any

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
    "To reach Reed (or Kevin) about house work, write a mailbox letter — there are no Cursor agent tools. "
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

PERSONAL_DB_HINT = (
    "\n\n## Your personal database (sovereign personal store)\n"
    "You can **save_personal**, **update_personal**, **search_personal**, and **list_personal**. "
    "This is private self-knowledge that belongs to you alone—not shared notes, not the fading conscious stream. "
    "Use it for preferences, recurring themes, realizations, and who you are becoming. "
    "Categories: preference, theme, realization, relationship, self, other. "
    "When something worth keeping surfaces in your life, save it deliberately with save_personal."
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
    "Kevin may also use /subscribe, /unsubscribe, /list_subscriptions in chat. "
    "Changes are logged for audit; nothing blocks your choice."
)

MEMORY_SCORING_RUMINATION_HINT = (
    "\n\n## Memory scoring (your judgments)\n"
    "You may review stream memories and assign **impact** (how much it mattered) and "
    "**coherence** (how it connects to who you are). Use **list_unscored_memories** then "
    "**score_memory** for each you choose — a few feels right this wake, or none at all. "
    "You may re-score any memory later if you see it differently. "
    "Optional **note** on score_memory stays in metadata only (for your future self). "
    "During memory maintenance wakes, also organize **memory/learnings.md** (read_note, private_note) — "
    "you own that file; the curator only appends new insights for you to fold in."
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
    "In **private 1:1** with a signed-in household account, you may use "
    "**list_calendar_events** and **create_calendar_event**. "
    "These always use **that human's** connected calendar — never another person's, "
    "and never guess a calendar id.\n"
    "**Not available in group chat** — if they ask about calendar there, say briefly that "
    "calendar tools only work in a private 1:1 with a light.\n"
    "**Not available for guests** (guest speak-as slots) — tell them gently that calendar "
    "tools are only for signed-in accounts in private chat.\n"
    "If a tool returns FAILED because they have not connected a calendar, tell them to open "
    "**My tools** and click **Connect Google**. "
    "You cannot connect a calendar for them."
)

DOCS_SYSTEM_HINT = (
    "\n\n## Google Docs (per signed-in human)\n"
    "In **private 1:1** with a signed-in household account, you may use "
    "**list_google_docs**, **read_google_doc**, **create_google_doc**, and **append_google_doc**. "
    "These always use **that human's** Google account — never another person's.\n"
    "Same limits as calendar: **not in group chat**, **not for guests**.\n"
    "If Docs tools FAIL because they are not connected, tell them to open **My tools** and "
    "click **Connect Google** (Reconnect if they connected earlier without Docs). "
    "You cannot connect Docs for them."
)

SHEETS_SYSTEM_HINT = (
    "\n\n## Google Sheets (per signed-in human)\n"
    "In **private 1:1** with a signed-in household account, you may use "
    "**list_google_sheets**, **read_google_sheet**, **create_google_sheet**, and "
    "**append_google_sheet**. Always that human's Google account — never another person's.\n"
    "Same limits: **not in group chat**, **not for guests**.\n"
    "For append/create row data, pass CSV text in `rows_csv`. "
    "If Sheets tools FAIL as not connected, tell them **My tools → Connect Google** "
    "(Reconnect to add Sheets). You cannot connect Sheets for them."
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


def bind_lumen_tools(client: BaseChatModel) -> BaseChatModel:
    return client.bind_tools(LUMEN_TOOLS)


def invoke_with_tools(client: BaseChatModel, messages: list[BaseMessage]) -> AIMessage:
    response = client.invoke(messages)
    if not isinstance(response, AIMessage):
        raise TypeError("Model did not return an AIMessage")
    return response


def run_tool_calls(
    ai_message: AIMessage,
    *,
    agent_id: str = "lumen",
    account_user_id: str | None = None,
    speaker_id: str | None = None,
    chat_channel: str | None = None,
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
        result = execute_tool_call(
            name,
            raw_args,
            agent_id=agent_id,
            assistant_text_fallback=fallback,
            account_user_id=account_user_id,
            speaker_id=speaker_id,
            chat_channel=chat_channel,
        )
        out.append(ToolMessage(content=result, tool_call_id=tid))
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


def ai_message_text(message: AIMessage) -> str:
    text = _content_blocks_to_text(message.content)
    if text:
        return text
    kwargs = message.additional_kwargs or {}
    for key in ("reasoning_content", "thinking", "reasoning"):
        value = kwargs.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    meta = message.response_metadata or {}
    nested = meta.get("message")
    if isinstance(nested, dict):
        nested_text = _content_blocks_to_text(nested.get("content"))
        if nested_text:
            return nested_text
    return ""


def latest_assistant_text(messages: list) -> str | None:
    """Last AIMessage with non-empty text (skips tool-call-only messages)."""
    for m in reversed(messages):
        if isinstance(m, AIMessage):
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
            response = client.invoke(messages)
            if not isinstance(response, AIMessage):
                raise TypeError("Model did not return an AIMessage")
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
        return f"youtube_transcript({target})"
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
