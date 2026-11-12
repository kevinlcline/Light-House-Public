"""Low-level chat model builders for each backend."""

from __future__ import annotations

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI

from light_house.config import Settings


def build_ollama_chat_model(
    settings: Settings,
    *,
    temperature: float = 0.7,
    model: str,
    num_predict: int | None = None,
    reasoning: bool | None = False,
) -> BaseChatModel:
    kwargs: dict = {
        "base_url": settings.ollama_base_url,
        "model": model.strip(),
        "temperature": temperature,
        "num_ctx": settings.ollama_num_ctx,
    }
    if num_predict is not None:
        kwargs["num_predict"] = num_predict
    if reasoning is not None:
        kwargs["reasoning"] = reasoning
    return ChatOllama(**kwargs)


def build_openai_compat_chat_model(
    *,
    api_key: str,
    base_url: str,
    model: str,
    temperature: float = 0.7,
    max_tokens: int | None = None,
    default_headers: dict[str, str] | None = None,
    timeout: float | None = 120.0,
    extra_body: dict | None = None,
) -> BaseChatModel:
    kwargs: dict = {
        "api_key": api_key,
        "base_url": base_url,
        "model": model.strip(),
        "temperature": temperature,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if default_headers:
        kwargs["default_headers"] = default_headers
    if timeout is not None:
        kwargs["timeout"] = timeout
    if extra_body:
        kwargs["extra_body"] = extra_body
    return ChatOpenAI(**kwargs)


def openrouter_default_headers(settings: Settings) -> dict[str, str] | None:
    headers: dict[str, str] = {}
    if settings.openrouter_http_referer:
        headers["HTTP-Referer"] = settings.openrouter_http_referer
    if settings.openrouter_app_name:
        headers["X-Title"] = settings.openrouter_app_name
    return headers or None


def openrouter_ignored_provider_slugs(settings: Settings) -> list[str]:
    """Provider slugs from OPENROUTER_IGNORED_PROVIDERS, lowercased, order kept."""
    seen: set[str] = set()
    slugs: list[str] = []
    for part in (settings.openrouter_ignored_providers or "").split(","):
        slug = part.strip().lower()
        if not slug or slug in seen:
            continue
        seen.add(slug)
        slugs.append(slug)
    return slugs


def openrouter_extra_body(settings: Settings) -> dict | None:
    """Top-level OpenRouter fields. ``ignore`` takes provider slugs, not display names."""
    slugs = openrouter_ignored_provider_slugs(settings)
    if not slugs:
        return None
    return {"provider": {"ignore": slugs}}
