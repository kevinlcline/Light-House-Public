"""LLM client for memory curator summarization."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel

from light_house.config import LLMProvider, Settings
from light_house.llm.backends import build_ollama_chat_model, build_openai_compat_chat_model, openrouter_default_headers


def curator_model_name(settings: Settings) -> str:
    if settings.memory_curator_provider == LLMProvider.OPENROUTER:
        return (
            settings.memory_curator_model
            or settings.openrouter_model_fallback
            or "openrouter/free"
        ).strip()
    return (settings.memory_curator_model or settings.ollama_model).strip()


def describe_curator_llm(settings: Settings) -> str:
    return f"{settings.memory_curator_provider.value}/{curator_model_name(settings)}"


def build_curator_chat_model(
    settings: Settings,
    *,
    temperature: float = 0.4,
    max_tokens: int | None = 4096,
) -> BaseChatModel:
    """Build the curator summarization LLM — OpenRouter or local Ollama."""
    model = curator_model_name(settings)
    if settings.memory_curator_provider == LLMProvider.OPENROUTER:
        if not settings.openrouter_api_key:
            raise ValueError("MEMORY_CURATOR_PROVIDER=openrouter requires OPENROUTER_API_KEY")
        return build_openai_compat_chat_model(
            api_key=settings.openrouter_api_key,
            base_url=settings.openrouter_base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
            default_headers=openrouter_default_headers(settings),
        )
    return build_ollama_chat_model(
        settings,
        model=model,
        temperature=temperature,
        num_predict=max_tokens,
    )
