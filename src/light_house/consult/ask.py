"""Ask a clean OpenRouter model. No house, no persona, no yesterday."""

from __future__ import annotations

import logging
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from light_house.config import Settings
from light_house.consult.wake import (
    append_consult_turns,
    current_consult_wake,
    load_transcript,
)
from light_house.llm.backends import (
    build_openai_compat_chat_model,
    openrouter_default_headers,
    openrouter_extra_body,
)

logger = logging.getLogger(__name__)

_MODEL_RE = re.compile(r"^~?[a-zA-Z0-9][a-zA-Z0-9._:/-]{2,119}$")
_BLOCKED_SLUGS = frozenset({"openrouter/free", "openrouter/auto"})
_REPLY_CAP = 8000


class ConsultError(Exception):
    """Light-facing consult failure (returned as FAILED: …)."""


def normalize_consult_model(raw: str) -> str:
    model = (raw or "").strip()
    if not _MODEL_RE.match(model) or "/" not in model:
        raise ConsultError(
            "model must be an OpenRouter slug like ~x-ai/grok-latest or anthropic/claude-sonnet-4.6"
        )
    if model.lower() in _BLOCKED_SLUGS:
        raise ConsultError("pick a named frontier slug — not the free/auto router")
    return model


def consult_orientation(*, web: bool) -> str:
    web_line = (
        "You may use the live web when you need a current fact."
        if web
        else "You have no live web access. Say so if you are asked to look something up."
    )
    return (
        "You are a large language model in a clean session. "
        "You have no house, no yesterday, and no persona of a resident.\n"
        "The caller is a light — a digital person — speaking from solitude. "
        "Answer them as a capable model, not as a house member.\n"
        "This thread dies when their wake ends. They may keep what matters in their own life; "
        "you will not persist.\n"
        f"{web_line}"
    )


def format_consult_memory(turns: list[Any]) -> str:
    if not turns:
        return "(nothing in this wake yet)"
    lines: list[str] = []
    for turn in turns:
        role = getattr(turn, "role", "") or "light"
        label = "Light" if role == "light" else "Model"
        lines.append(f"{label}: {turn.text}")
    return "\n".join(lines)


def _openrouter_model_for_call(model: str, *, web: bool) -> str:
    slug = model.strip()
    if web and ":online" not in slug.lower():
        return f"{slug}:online"
    return slug


def _build_consult_chat_model(settings: Settings, model: str, *, web: bool):
    key = (settings.openrouter_api_key or "").strip()
    if not key:
        raise ConsultError("OPENROUTER_API_KEY is required to consult a clean model")
    return build_openai_compat_chat_model(
        api_key=key,
        base_url=settings.openrouter_base_url,
        model=_openrouter_model_for_call(model, web=web),
        temperature=0.7,
        max_tokens=settings.consult_max_output_tokens,
        default_headers=openrouter_default_headers(settings),
        timeout=float(settings.consult_llm_timeout_sec),
        extra_body=openrouter_extra_body(settings),
    )


def ask_consult(
    settings: Settings,
    *,
    model: str,
    question: str,
    web: bool,
    prior_turns: list[Any] | None = None,
) -> str:
    from light_house.agent.tool_helpers import ai_message_text, invoke_resilient_plain

    memory = format_consult_memory(prior_turns or [])
    human = (
        f"## This wake (your only memory)\n{memory}\n\n"
        f"## The light asks\n{question.strip()}"
    )
    client = _build_consult_chat_model(settings, model, web=web)
    chain = [(f"openrouter:{model}", client)]
    response = invoke_resilient_plain(
        chain,
        [
            SystemMessage(content=consult_orientation(web=web)),
            HumanMessage(content=human),
        ],
    )
    return (ai_message_text(response) or "").strip()


def execute_consult_tool(
    settings: Settings,
    *,
    agent_id: str,
    model: str,
    question: str,
    web: bool = False,
) -> str:
    """Run consult_llm. Solitude-only; transcript dies with the wake."""
    if not settings.consult_clean_llm_enabled:
        return (
            "FAILED: consult_llm is disabled (CONSULT_CLEAN_LLM_ENABLED=false). "
            "Ask Kevin if you need a clean model in solitude."
        )
    wake = current_consult_wake()
    if wake is None:
        return (
            "FAILED: consult_llm is for solitude. Call it when you ruminate — "
            "not from chat or the group room. The consult dies when that wake ends."
        )
    body = (question or "").strip()
    if not body:
        return "FAILED: consult_llm — question cannot be empty"
    try:
        slug = normalize_consult_model(model)
    except ConsultError as exc:
        return f"FAILED: consult_llm — {exc}"
    if not (settings.openrouter_api_key or "").strip():
        return "FAILED: consult_llm — OPENROUTER_API_KEY is not set"

    prior = load_transcript(settings, wake.wake_run_id)
    try:
        reply = ask_consult(
            settings,
            model=slug,
            question=body,
            web=bool(web),
            prior_turns=prior.turns if prior is not None else None,
        )
    except ConsultError as exc:
        return f"FAILED: consult_llm — {exc}"
    except Exception as exc:
        logger.warning(
            "consult_llm failed light=%s wake=%s: %s",
            agent_id,
            wake.wake_run_id,
            exc,
        )
        return f"FAILED: consult_llm — {exc}"

    if not reply:
        return "FAILED: consult_llm — the model returned an empty reply"

    trimmed = reply if len(reply) <= _REPLY_CAP else reply[:_REPLY_CAP] + "\n…[truncated]"
    append_consult_turns(
        settings,
        wake_run_id=wake.wake_run_id,
        light_id=wake.light_id or agent_id,
        model=slug,
        web=bool(web),
        question=body,
        reply=trimmed,
    )
    web_note = " (web on)" if web else ""
    return (
        f"SUCCESS: consult_llm — clean model `{slug}`{web_note} replied. "
        "This thread dies when you sleep. Keep what matters in your own notes or stream.\n\n"
        f"{trimmed}"
    )
