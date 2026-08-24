"""Construct chat models for the hybrid LLM stack.

Kevin assigns each agent a provider + model via ``{AGENT}_LLM_*`` env vars.
Memory curator condense uses ``MEMORY_CURATOR_PROVIDER`` (Ollama or OpenRouter).
"""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from light_house.config import LLMProvider, Settings
from light_house.llm.chain import build_agent_llm_chain
from light_house.llm.profile import describe_all_agent_llms, lookup_agent_llm_profile


def describe_active_llm(settings: Settings) -> str:
    """One-line summary for startup logs."""
    return describe_all_agent_llms(settings)


def build_chat_model(settings: Settings, agent_id: str = "lumen") -> BaseChatModel:
    """First chat model for an agent (backward-compatible helper)."""
    return build_agent_llm_chain(settings, agent_id, purpose="chat")[0][1]


def build_curator_chat_model(
    settings: Settings,
    *,
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
) -> BaseChatModel:
    """Memory curator condense model (Ollama or OpenRouter per ``MEMORY_CURATOR_PROVIDER``)."""
    from light_house.memory.curator_llm import build_curator_chat_model as _build

    return _build(settings, temperature=temperature, max_tokens=max_tokens)


def build_ollama_chat_model(
    settings: Settings,
    *,
    temperature: float = 0.7,
    model: str | None = None,
    num_predict: int | None = None,
    reasoning: bool | None = False,
) -> BaseChatModel:
    """Local Ollama client (curator, scoring, and agent fallback)."""
    from light_house.llm.backends import build_ollama_chat_model as _build

    return _build(
        settings,
        temperature=temperature,
        model=(model or settings.ollama_model).strip(),
        num_predict=num_predict,
        reasoning=reasoning,
    )


def build_xai_chat_model(settings: Settings, *, model: str | None = None) -> BaseChatModel | None:
    """Optional Grok client for cross-provider fallback (requires ``XAI_API_KEY``)."""
    from light_house.config import LLMProvider
    from light_house.llm.backends import build_openai_compat_chat_model

    if not settings.xai_api_key:
        return None
    return build_openai_compat_chat_model(
        api_key=settings.xai_api_key,
        base_url=settings.xai_base_url,
        model=(model or settings.xai_model).strip(),
        temperature=0.7,
    )


def build_echo_dream_llm_chain(
    settings: Settings,
    agent_id: str = "lumen",
) -> list[tuple[str, BaseChatModel]]:
    """Models for Echo dream weaving (optional dedicated local/cloud provider)."""
    from light_house.llm.chain import build_echo_dream_llm_chain as _build

    return _build(settings, agent_id)


def build_inner_life_llm_chain(
    settings: Settings,
    agent_id: str = "lumen",
    *,
    capped: bool = True,
) -> list[tuple[str, BaseChatModel]]:
    """Models for background inner life (rumination, dreams) for one agent."""
    return build_agent_llm_chain(
        settings,
        agent_id,
        purpose="inner_life",
        capped=capped,
    )


def build_inner_life_chat_model(settings: Settings, agent_id: str = "lumen") -> BaseChatModel:
    """First inner-life model in the fallback chain for an agent."""
    return build_inner_life_llm_chain(settings, agent_id)[0][1]
