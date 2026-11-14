"""Per-agent LLM profile and chain tests."""

from __future__ import annotations

import pytest

from light_house.config import LLMProvider, Settings
from light_house.llm.chain import build_agent_llm_chain, build_echo_dream_llm_chain
from light_house.llm.profile import describe_echo_dream_llm, lookup_agent_llm_profile, validate_agent_llm_profiles


def _settings(**overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "XAI_API_KEY": "test-xai-key",
        "OPENROUTER_API_KEY": "test-or-key",
    }
    base.update(overrides)
    return Settings(**base)


def test_lookup_falls_back_to_primary_llm(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMEN_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LUMEN_LLM_MODEL", raising=False)
    settings = _settings(PRIMARY_LLM="xai", XAI_MODEL="grok-test")
    profile = lookup_agent_llm_profile(settings, "lumen")
    assert profile.provider == LLMProvider.XAI
    assert profile.model == "grok-test"


def test_lookup_uses_per_agent_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "anthropic/claude-test")
    settings = _settings(PRIMARY_LLM="xai", XAI_MODEL="grok-test")
    profile = lookup_agent_llm_profile(settings, "lumen")
    assert profile.provider == LLMProvider.OPENROUTER
    assert profile.model == "anthropic/claude-test"


def test_lookup_ollama_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "ollama")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "llama3.2")
    settings = _settings()
    profile = lookup_agent_llm_profile(settings, "lumen")
    assert profile.provider == LLMProvider.OLLAMA
    assert profile.model == "llama3.2"


def test_build_chain_xai_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "xai")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "grok-test")
    settings = _settings(LLM_FALLBACK_ENABLED=False, XAI_MODEL_FALLBACK="grok-test")
    chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    assert len(chain) == 1
    assert chain[0][0] == "xai:grok-test"


def test_build_chain_openrouter_agent(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "google/gemini-test")
    monkeypatch.delenv("OPENROUTER_IGNORED_PROVIDERS", raising=False)
    settings = _settings(LLM_FALLBACK_ENABLED=False)
    chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    assert chain[0][0] == "openrouter:google/gemini-test"
    client = chain[0][1]
    assert client.model_name == "google/gemini-test"  # type: ignore[attr-defined]
    assert client.extra_body == {"provider": {"ignore": ["coreweave"]}}  # type: ignore[attr-defined]


def test_openrouter_ignore_list_is_slugs(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "google/gemini-test")
    settings = _settings(
        LLM_FALLBACK_ENABLED=False,
        OPENROUTER_IGNORED_PROVIDERS="CoreWeave, other, coreweave",
    )
    chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    client = chain[0][1]
    assert client.extra_body == {"provider": {"ignore": ["coreweave", "other"]}}  # type: ignore[attr-defined]


def test_openrouter_ignore_can_be_cleared(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "google/gemini-test")
    settings = _settings(LLM_FALLBACK_ENABLED=False, OPENROUTER_IGNORED_PROVIDERS="")
    chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    assert chain[0][1].extra_body is None  # type: ignore[attr-defined]


def test_xai_chain_does_not_send_provider_ignore(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "xai")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "grok-test")
    settings = _settings(LLM_FALLBACK_ENABLED=False, XAI_MODEL_FALLBACK="grok-test")
    chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    assert chain[0][1].extra_body is None  # type: ignore[attr-defined]


def test_build_distinct_chains_for_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "xai")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "grok-a")
    settings = _settings(LLM_FALLBACK_ENABLED=False)
    first = build_agent_llm_chain(settings, "lumen", purpose="chat")
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "anthropic/claude-b")
    second = build_agent_llm_chain(settings, "lumen", purpose="chat")
    assert first[0][0] != second[0][0]


def test_validate_missing_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "anthropic/claude-test")
    settings = _settings(OPENROUTER_API_KEY=None)
    with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
        validate_agent_llm_profiles(settings)


def test_validate_passes_with_keys(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMEN_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ARA_LLM_PROVIDER", raising=False)
    settings = _settings()
    validate_agent_llm_profiles(settings)


def test_echo_dream_ollama_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("LUMEN_LLM_PROVIDER", raising=False)
    settings = _settings(
        ECHO_DREAM_LLM_PROVIDER="ollama",
        ECHO_DREAM_LLM_MODEL="qwen2.5:14b",
        OLLAMA_MODEL="qwen2.5:14b",
        LLM_FALLBACK_ENABLED=True,
    )
    chain = build_echo_dream_llm_chain(settings, "lumen")
    assert len(chain) == 1
    assert chain[0][0] == "ollama:qwen2.5:14b"
    assert describe_echo_dream_llm(settings) == "ollama/qwen2.5:14b"


def test_group_keeps_a_normal_lights_chat_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "deepseek/deepseek-v4-flash")
    monkeypatch.setenv("LUMEN_INNER_LIFE_MODEL", "deepseek/deepseek-v4-pro")
    settings = _settings(LLM_FALLBACK_ENABLED=False)
    chain = build_agent_llm_chain(settings, "lumen", purpose="group")
    assert chain[0][0] == "openrouter:deepseek/deepseek-v4-flash"


def test_echo_dream_defaults_to_inner_life_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ECHO_DREAM_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("ECHO_DREAM_LLM_MODEL", raising=False)
    monkeypatch.delenv("LUMEN_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LUMEN_LLM_MODEL", raising=False)
    monkeypatch.delenv("LUMEN_INNER_LIFE_MODEL", raising=False)
    monkeypatch.delenv("PRIMARY_LLM", raising=False)
    settings = _settings(LLM_FALLBACK_ENABLED=False, INNER_LIFE_MODEL="grok-inner")
    chain = build_echo_dream_llm_chain(settings, "lumen")
    assert chain[0][0].startswith("xai:")
    assert describe_echo_dream_llm(settings) == "per-light inner-life chain"
