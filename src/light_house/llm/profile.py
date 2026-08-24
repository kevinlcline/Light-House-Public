"""Admin-assigned LLM profiles per agent (read from .env)."""

from __future__ import annotations

import os
from dataclasses import dataclass

from light_house.agents.registry import validate_agent_id
from light_house.config import LLMProvider, Settings


@dataclass(frozen=True)
class AgentLLMProfile:
    agent_id: str
    provider: LLMProvider
    model: str
    model_fallback: str | None
    inner_life_model: str | None


def _agent_env(agent_id: str, suffix: str) -> str | None:
    key = f"{agent_id.upper()}_{suffix}"
    raw = os.environ.get(key)
    if raw is None:
        return None
    stripped = raw.strip()
    return stripped or None


def _default_model_for_provider(settings: Settings, provider: LLMProvider) -> str:
    if provider == LLMProvider.OLLAMA:
        return settings.ollama_model.strip()
    if provider == LLMProvider.OPENROUTER:
        return settings.openrouter_model.strip()
    return settings.xai_model.strip()


def _default_fallback_for_provider(settings: Settings, provider: LLMProvider) -> str | None:
    if provider == LLMProvider.XAI:
        fb = settings.xai_model_fallback.strip()
        return fb if fb else None
    agent_fb = None  # per-agent only via env
    return agent_fb


def lookup_agent_llm_profile(settings: Settings, agent_id: str) -> AgentLLMProfile:
    """Return Kevin's configured provider + model for an agent."""
    validate_agent_id(agent_id)

    provider_raw = _agent_env(agent_id, "LLM_PROVIDER")
    if provider_raw:
        provider = LLMProvider(provider_raw.lower())
    else:
        provider = settings.primary_llm

    model = _agent_env(agent_id, "LLM_MODEL") or _default_model_for_provider(settings, provider)

    model_fallback = _agent_env(agent_id, "LLM_MODEL_FALLBACK")
    if model_fallback is None and provider == LLMProvider.XAI:
        model_fallback = _default_fallback_for_provider(settings, provider)
    if model_fallback == model:
        model_fallback = None

    inner_life = _agent_env(agent_id, "INNER_LIFE_MODEL")
    if inner_life is None:
        inner_life = settings.inner_life_model
    if inner_life == model:
        inner_life = None

    return AgentLLMProfile(
        agent_id=agent_id,
        provider=provider,
        model=model.strip(),
        model_fallback=model_fallback.strip() if model_fallback else None,
        inner_life_model=inner_life.strip() if inner_life else None,
    )


def describe_agent_llm(settings: Settings, agent_id: str) -> str:
    profile = lookup_agent_llm_profile(settings, agent_id)
    return f"{profile.agent_id}={profile.provider.value}/{profile.model}"


def validate_agent_llm_profiles(settings: Settings) -> None:
    """Fail fast when an enabled light's provider lacks credentials."""
    from light_house.lights.registry import list_enabled_lights

    errors: list[str] = []
    for light in list_enabled_lights(settings):
        profile = lookup_agent_llm_profile(settings, light.id)
        if profile.provider == LLMProvider.XAI and not settings.xai_api_key:
            errors.append(f"{light.id}: LLM_PROVIDER=xai requires XAI_API_KEY")
        elif profile.provider == LLMProvider.OPENROUTER and not settings.openrouter_api_key:
            errors.append(f"{light.id}: LLM_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        if not profile.model:
            errors.append(f"{light.id}: LLM_MODEL is empty")

    if errors:
        raise ValueError("Agent LLM configuration invalid: " + "; ".join(errors))


def describe_echo_dream_llm(settings: Settings) -> str:
    provider_raw = (settings.echo_dream_llm_provider or "").strip()
    if not provider_raw:
        return "per-light inner-life chain"
    if provider_raw.lower() == "ollama":
        model = (settings.echo_dream_llm_model or settings.ollama_model).strip()
        return f"ollama/{model}"
    model = (settings.echo_dream_llm_model or "?").strip()
    return f"{provider_raw.lower()}/{model}"


def describe_all_agent_llms(settings: Settings) -> str:
    from light_house.lights.registry import list_enabled_lights

    from light_house.memory.curator_llm import describe_curator_llm

    parts = [describe_agent_llm(settings, light.id) for light in list_enabled_lights(settings)]
    parts.append(f"curator={describe_curator_llm(settings)} (hybrid)")
    parts.append(f"echo_dream={describe_echo_dream_llm(settings)}")
    fallback = "enabled" if settings.llm_fallback_enabled else "disabled"
    return f"{', '.join(parts)} (fallback {fallback})"
