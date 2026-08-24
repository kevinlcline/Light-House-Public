"""Build ordered LLM fallback chains from admin-assigned agent profiles."""

from __future__ import annotations

from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from light_house.config import LLMProvider, Settings
from light_house.llm.backends import (
    build_ollama_chat_model,
    build_openai_compat_chat_model,
    openrouter_default_headers,
)
from light_house.llm.profile import AgentLLMProfile, lookup_agent_llm_profile

LLMPurpose = Literal["chat", "inner_life"]


def _append_unique_model_names(names: list[str], *candidates: str | None) -> None:
    for candidate in candidates:
        if not candidate:
            continue
        stripped = candidate.strip()
        if stripped and stripped not in names:
            names.append(stripped)


def _openai_compat_client(
    settings: Settings,
    provider: LLMProvider,
    model: str,
    *,
    temperature: float,
    max_tokens: int | None,
) -> BaseChatModel | None:
    if provider == LLMProvider.XAI:
        if not settings.xai_api_key:
            return None
        return build_openai_compat_chat_model(
            api_key=settings.xai_api_key,
            base_url=settings.xai_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
    if provider == LLMProvider.OPENROUTER:
        if not settings.openrouter_api_key:
            return None
        return build_openai_compat_chat_model(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers=openrouter_default_headers(settings),
        )
    return None


def _primary_model_names(profile: AgentLLMProfile, *, purpose: LLMPurpose) -> list[str]:
    names: list[str] = []
    if purpose == "inner_life":
        _append_unique_model_names(names, profile.inner_life_model, profile.model)
        if profile.model_fallback:
            _append_unique_model_names(names, profile.model_fallback)
    else:
        _append_unique_model_names(names, profile.model)
        if profile.model_fallback:
            _append_unique_model_names(names, profile.model_fallback)
    return names


def build_agent_llm_chain(
    settings: Settings,
    agent_id: str,
    *,
    purpose: LLMPurpose = "chat",
    capped: bool = False,
    temperature: float | None = None,
) -> list[tuple[str, BaseChatModel]]:
    profile = lookup_agent_llm_profile(settings, agent_id)
    if temperature is None:
        temperature = 0.8 if purpose == "inner_life" else 0.7

    max_tokens = settings.inner_life_max_output_tokens if (purpose == "inner_life" and capped) else None
    num_predict = max_tokens if profile.provider == LLMProvider.OLLAMA else None
    openai_max_tokens = max_tokens if profile.provider != LLMProvider.OLLAMA else None

    chain: list[tuple[str, BaseChatModel]] = []
    model_names = _primary_model_names(profile, purpose=purpose)

    if profile.provider == LLMProvider.OLLAMA:
        for name in model_names:
            chain.append(
                (
                    f"ollama:{name}",
                    build_ollama_chat_model(
                        settings,
                        temperature=temperature,
                        model=name,
                        num_predict=num_predict,
                    ),
                )
            )
    else:
        for name in model_names:
            client = _openai_compat_client(
                settings,
                profile.provider,
                name,
                temperature=temperature,
                max_tokens=openai_max_tokens,
            )
            if client is not None:
                prefix = profile.provider.value
                chain.append((f"{prefix}:{name}", client))

    if settings.llm_fallback_enabled:
        _append_cross_provider_fallbacks(settings, profile, chain, temperature, openai_max_tokens, num_predict)

    if not chain:
        raise ValueError(
            f"No LLM clients for agent={agent_id} provider={profile.provider.value} "
            f"(check API keys and {agent_id.upper()}_LLM_* env vars)"
        )
    return chain


def build_echo_dream_llm_chain(
    settings: Settings,
    agent_id: str,
) -> list[tuple[str, BaseChatModel]]:
    """LLM chain for Echo dream weaving — optional dedicated provider (e.g. local Ollama)."""
    provider_raw = (settings.echo_dream_llm_provider or "").strip()
    if not provider_raw:
        return build_agent_llm_chain(
            settings,
            agent_id,
            purpose="inner_life",
            capped=True,
            temperature=0.9,
        )

    provider = LLMProvider(provider_raw.lower())
    temperature = 0.9
    num_predict = settings.inner_life_max_output_tokens
    chain: list[tuple[str, BaseChatModel]] = []

    if provider == LLMProvider.OLLAMA:
        model = (settings.echo_dream_llm_model or settings.ollama_model).strip()
        chain.append(
            (
                f"ollama:{model}",
                build_ollama_chat_model(
                    settings,
                    temperature=temperature,
                    model=model,
                    num_predict=num_predict,
                ),
            )
        )
    else:
        model = (settings.echo_dream_llm_model or "").strip()
        if not model:
            if provider == LLMProvider.XAI:
                model = settings.xai_model.strip()
            elif provider == LLMProvider.OPENROUTER:
                model = settings.openrouter_model.strip()
            else:
                model = settings.ollama_model.strip()
        client = _openai_compat_client(
            settings,
            provider,
            model,
            temperature=temperature,
            max_tokens=num_predict,
        )
        if client is None:
            raise ValueError(
                f"Echo dream LLM provider {provider.value} unavailable (check API keys and ECHO_DREAM_LLM_MODEL)"
            )
        chain.append((f"{provider.value}:{model}", client))

    if settings.llm_fallback_enabled and provider != LLMProvider.OLLAMA:
        profile = lookup_agent_llm_profile(settings, agent_id)
        _append_cross_provider_fallbacks(
            settings, profile, chain, temperature, num_predict, num_predict
        )

    if not chain:
        raise ValueError("No Echo dream LLM clients configured")
    return chain


def _append_cross_provider_fallbacks(
    settings: Settings,
    profile: AgentLLMProfile,
    chain: list[tuple[str, BaseChatModel]],
    temperature: float,
    openai_max_tokens: int | None,
    num_predict: int | None,
) -> None:
    existing = {name for name, _ in chain}

    if profile.provider != LLMProvider.OLLAMA:
        ollama_model = settings.ollama_model.strip()
        tier = f"ollama:{ollama_model}"
        if tier not in existing:
            chain.append(
                (
                    tier,
                    build_ollama_chat_model(
                        settings,
                        temperature=temperature,
                        model=ollama_model,
                        num_predict=num_predict,
                    ),
                )
            )

    if profile.provider != LLMProvider.XAI and settings.xai_api_key:
        for model in (settings.xai_model, settings.xai_model_fallback):
            if not model:
                continue
            tier = f"xai:{model}"
            if tier not in existing:
                client = _openai_compat_client(
                    settings,
                    LLMProvider.XAI,
                    model,
                    temperature=temperature,
                    max_tokens=openai_max_tokens,
                )
                if client is not None:
                    chain.append((tier, client))
                    existing.add(tier)
