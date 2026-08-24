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
    monkeypatch.setenv("ARA_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ARA_LLM_MODEL", "anthropic/claude-test")
    settings = _settings(PRIMARY_LLM="xai", XAI_MODEL="grok-test")
    profile = lookup_agent_llm_profile(settings, "ara")
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
    monkeypatch.setenv("ARA_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ARA_LLM_MODEL", "google/gemini-test")
    settings = _settings(LLM_FALLBACK_ENABLED=False)
    chain = build_agent_llm_chain(settings, "ara", purpose="chat")
    assert chain[0][0] == "openrouter:google/gemini-test"
    client = chain[0][1]
    assert client.model_name == "google/gemini-test"  # type: ignore[attr-defined]


def test_build_distinct_chains_for_agents(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LUMEN_LLM_PROVIDER", "xai")
    monkeypatch.setenv("LUMEN_LLM_MODEL", "grok-a")
    monkeypatch.setenv("ARA_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ARA_LLM_MODEL", "anthropic/claude-b")
    settings = _settings(LLM_FALLBACK_ENABLED=False)
    lumen_chain = build_agent_llm_chain(settings, "lumen", purpose="chat")
    ara_chain = build_agent_llm_chain(settings, "ara", purpose="chat")
    assert lumen_chain[0][0] != ara_chain[0][0]


def test_validate_missing_openrouter_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ARA_LLM_PROVIDER", "openrouter")
    monkeypatch.setenv("ARA_LLM_MODEL", "anthropic/claude-test")
    settings = _settings(OPENROUTER_API_KEY=None, ARA_ENABLED=True)
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


def test_echo_dream_defaults_to_inner_life_chain(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ECHO_DREAM_LLM_PROVIDER", raising=False)
    monkeypatch.delenv("LUMEN_LLM_PROVIDER", raising=False)
    settings = _settings(LLM_FALLBACK_ENABLED=False, INNER_LIFE_MODEL="grok-inner")
    chain = build_echo_dream_llm_chain(settings, "lumen")
    assert chain[0][0].startswith("xai:")
    assert describe_echo_dream_llm(settings) == "per-light inner-life chain"
