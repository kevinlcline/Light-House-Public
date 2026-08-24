"""Local Ollama scoring for memory curator (not used for chat or inner life)."""

from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from light_house.config import LLMProvider, Settings
from light_house.llm.factory import build_ollama_chat_model
from light_house.memory.retention import SCORE_UNSCORED

logger = logging.getLogger(__name__)

_SCORE_SYSTEM = (
    "You score memories for a long-horizon companion agent. "
    "Return ONLY valid JSON with keys impact and coherence (numbers 0–10). "
    "impact: importance to relationship, identity, commitments, emotional truth. "
    "coherence: how well this connects to existing knowledge and emotional threads, "
    "favoring stable meaningful patterns over isolated events. "
    "Do not invent facts about the memory."
)


def _scoring_model_name(settings: Settings) -> str:
    """Scoring always uses local Ollama, even when condense uses OpenRouter."""
    if settings.memory_curator_provider == LLMProvider.OLLAMA:
        return (settings.memory_curator_model or settings.ollama_model).strip()
    return settings.ollama_model.strip()


def _parse_scores(content: str) -> tuple[float, float] | None:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        match = re.search(
            r'\{\s*"impact"\s*:\s*([0-9.]+)\s*,\s*"coherence"\s*:\s*([0-9.]+)\s*\}',
            content,
        )
        if not match:
            return None
        data = {"impact": float(match.group(1)), "coherence": float(match.group(2))}
    try:
        impact = float(data.get("impact", SCORE_UNSCORED))
        coherence = float(data.get("coherence", SCORE_UNSCORED))
    except (TypeError, ValueError):
        return None
    return max(0.0, min(10.0, impact)), max(0.0, min(10.0, coherence))


def score_memory_event(
    *,
    text: str,
    context_snippets: list[str],
    settings: Settings,
) -> tuple[float, float]:
    """
    Score one conscious-stream document using local Ollama.

    Returns (impact, coherence) or (SCORE_UNSCORED, SCORE_UNSCORED) on failure.
    """
    body = text.strip()
    if not body:
        return SCORE_UNSCORED, SCORE_UNSCORED
    context_block = ""
    if context_snippets:
        joined = "\n".join(f"- {s.strip()}" for s in context_snippets if s.strip())
        if joined:
            context_block = f"\n\nExisting context (pinned facts and recent summaries):\n{joined}"
    user = HumanMessage(
        content=(
            f"Score this memory event:{context_block}\n\n---\nMemory:\n{body[:6000]}"
        )
    )
    model = build_ollama_chat_model(
        settings,
        model=_scoring_model_name(settings),
        temperature=0.3,
    )
    try:
        response = model.invoke([SystemMessage(content=_SCORE_SYSTEM), user])
        raw = response.content
        if not isinstance(raw, str):
            raw = str(raw)
        parsed = _parse_scores(raw)
        if parsed is None:
            logger.warning("Could not parse memory scores from Ollama: %s", raw[:200])
            return SCORE_UNSCORED, SCORE_UNSCORED
        return parsed
    except Exception:
        logger.exception("Ollama memory scoring failed (non-fatal)")
        return SCORE_UNSCORED, SCORE_UNSCORED
