"""Local Kokoro TTS helpers and API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from light_house.config import Settings
from light_house.lights.manifest import ensure_manifest_file
from light_house.lights.registry import reload_lights_manifest
from light_house.lights_admin import update_light
from light_house.main import _settings_dep, app
from light_house.tts.kokoro_engine import voice_for_agent
from light_house.tts.sentences import split_speech_chunks
from light_house.tts.text_prep import text_for_speech
from light_house.tts.voices_catalog import default_voice_for_light, list_voice_catalog


def test_text_for_speech_strips_markdown() -> None:
    raw = "# Hello\n\nThis is **bold** and a [link](https://example.com).\n\n```\ncode\n```\n- item"
    out = text_for_speech(raw)
    assert "Hello" in out
    assert "bold" in out
    assert "link" in out
    assert "https://" not in out
    assert "```" not in out
    assert "**" not in out


def test_text_for_speech_softens_paths_and_urls() -> None:
    out = text_for_speech(
        "See shared/from-kevin.md and https://example.com/docs/page.html please."
    )
    assert "/" not in out
    assert "\\" not in out
    assert "https" not in out.lower()
    assert ".md" not in out.lower()
    assert "shared" in out
    assert "from-kevin" in out
    assert "example.com" in out
    assert ", " in out
    # Ordinary words with slashes stay intact.
    keep = text_for_speech("Use and/or when needed.")
    assert "and/or" in keep


def test_text_for_speech_strips_stage_cues() -> None:
    out = text_for_speech("*smiles* Hello there, friend.")
    assert "smiles" not in out.lower()
    assert "Hello there" in out
    laughed = text_for_speech("That is wonderful (laughs) truly.")
    assert "laughs" not in laughed.lower()
    assert "wonderful" in laughed
    assert "truly" in laughed
    # Unrelated italics stay speakable.
    keep = text_for_speech("*really* important")
    assert "really" in keep
    assert "important" in keep


def test_text_for_speech_strips_emojis() -> None:
    out = text_for_speech("Hey love 🌙💖 — got it, the emojis are getting read out loud.")
    assert "🌙" not in out
    assert "💖" not in out
    assert "Hey love" in out
    assert "got it" in out
    assert "emojis are getting read out loud" in out
    # Skin-tone / ZWJ sequences should not leave junk.
    family = text_for_speech("Hi 👨‍👩‍👧‍👦 there")
    assert "Hi" in family
    assert "there" in family
    assert "\U0001F468" not in family


def test_text_for_speech_normalizes_fancy_punctuation() -> None:
    out = text_for_speech("Wait… • first — “quoted” then ‘done’.")
    assert "…" not in out
    assert "—" not in out
    assert "•" not in out
    assert "“" not in out and "”" not in out
    assert "‘" not in out and "’" not in out
    assert "..." in out
    assert "quoted" in out
    assert "done" in out
    # Punctuation-only leftovers should not be spoken.
    assert text_for_speech("—") == ""
    assert text_for_speech("…") == ""
    assert text_for_speech("***") == ""


def test_split_speech_chunks_sentences_and_abbrev() -> None:
    text = (
        "Hello there. Dr. Jones said hi! "
        "What do you think? "
        "Short.\n\n"
        "A longer closing thought that wraps the turn."
    )
    chunks = split_speech_chunks(text)
    assert chunks
    # Short sentences merge so clips stay long enough to hide CPU synth of N+1.
    assert "Hello there." in chunks[0]
    assert any("Dr. Jones" in c for c in chunks)
    assert any("What do you think?" in c for c in chunks)
    assert all(len(c) <= 280 for c in chunks)
    assert all(len(c) >= 12 for c in chunks)


def test_split_speech_chunks_splits_long_runon() -> None:
    words = " ".join(["word"] * 120)
    chunks = split_speech_chunks(words, max_chars=80)
    assert len(chunks) > 1
    assert all(len(c) <= 80 for c in chunks)


def _settings(tmp_path: Path, **overrides: object) -> Settings:
    base = {
        "_env_file": None,
        "MEMORY_STORE_PATH": str(tmp_path / "memory"),
        "THREADS_DATA_PATH": str(tmp_path / "threads"),
        "PERSONAL_DB_ENABLED": False,
        "INNER_LIFE_ENABLED": False,
        "MEMORY_CURATOR_ENABLED": False,
        "FOUNDATION_SEED_ON_STARTUP": False,
        "WEB_GATE_ENABLED": False,
        "TTS_ENABLED": True,
        "KOKORO_MODEL_PATH": str(tmp_path / "kokoro"),
    }
    base.update(overrides)
    return Settings(**base)


def test_tts_status_and_speak_mocked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings

    def fake_synthesize(settings_arg, *, text, agent_id, voice=None):
        assert "Hello" in text
        assert agent_id == "lumen"
        return b"RIFF....WAVEfmt "  # minimal stand-in

    monkeypatch.setattr("light_house.main.synthesize_wav", fake_synthesize)
    monkeypatch.setattr(
        "light_house.main.get_tts_status",
        lambda s: {
            "enabled": True,
            "ready": True,
            "engine": "kokoro-onnx",
            "model_present": False,
            "voices_present": False,
            "error": None,
            "voices": {"lumen": "af_sarah", "ara": "af_bella", "elias": "am_michael", "default": "af_sarah"},
        },
    )

    try:
        with TestClient(app) as client:
            status = client.get("/v1/tts/status")
            assert status.status_code == 200
            assert status.json()["ready"] is True

            spoken = client.post(
                "/v1/tts",
                json={"text": "Hello from Lumen.", "agent_id": "lumen"},
            )
            assert spoken.status_code == 200
            assert spoken.headers["content-type"].startswith("audio/wav")
            assert spoken.content.startswith(b"RIFF")
    finally:
        app.dependency_overrides.clear()


def test_tts_disabled_returns_404(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    settings = _settings(tmp_path, TTS_ENABLED=False)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.post("/v1/tts", json={"text": "Hi", "agent_id": "lumen"})
            assert res.status_code == 404
    finally:
        app.dependency_overrides.clear()


def test_tts_voices_catalog_endpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = _settings(tmp_path)
    monkeypatch.setattr("light_house.web_gate.get_settings", lambda: settings)
    app.dependency_overrides[_settings_dep] = lambda: settings
    try:
        with TestClient(app) as client:
            res = client.get("/v1/tts/voices")
            assert res.status_code == 200
            voices = res.json()["voices"]
            assert len(voices) == len(list_voice_catalog())
            ids = {v["id"] for v in voices}
            assert "af_sarah" in ids
            assert "am_michael" in ids

            extended = client.get("/v1/tts/voices?all_langs=1")
            assert extended.status_code == 200
            assert len(extended.json()["voices"]) > len(voices)
    finally:
        app.dependency_overrides.clear()


def test_voice_for_agent_uses_manifest_voice_id(tmp_path: Path) -> None:
    settings = _settings(
        tmp_path,
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        INNER_LIFE_THREAD_ID="kevin-home",
        ARA_THREAD_ID="ara-home",
        ARA_ENABLED=True,
        TTS_VOICE_LUMEN="",
    )
    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    assert voice_for_agent(settings, "lumen") == default_voice_for_light("lumen")
    update_light(settings, "lumen", voice_id="af_nicole")
    reload_lights_manifest(settings)
    assert voice_for_agent(settings, "lumen") == "af_nicole"
