"""Memory curator LLM provider routing."""

from __future__ import annotations

from light_house.config import LLMProvider, Settings
from light_house.memory.curator_llm import build_curator_chat_model
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI


def test_build_curator_chat_model_uses_ollama_by_default():
    settings = Settings(
        _env_file=None,
        MEMORY_CURATOR_PROVIDER="ollama",
        MEMORY_CURATOR_MODEL="llama3.2",
        OLLAMA_MODEL="qwen2.5:7b",
    )
    model = build_curator_chat_model(settings)
    assert isinstance(model, ChatOllama)
    assert model.model == "llama3.2"


def test_build_curator_chat_model_uses_openrouter_when_configured():
    settings = Settings(
        _env_file=None,
        MEMORY_CURATOR_PROVIDER="openrouter",
        MEMORY_CURATOR_MODEL="openrouter/free",
        OPENROUTER_API_KEY="sk-test",
        OPENROUTER_BASE_URL="https://openrouter.ai/api/v1",
    )
    model = build_curator_chat_model(settings)
    assert isinstance(model, ChatOpenAI)
    assert model.model_name == "openrouter/free"
    assert settings.memory_curator_provider == LLMProvider.OPENROUTER
