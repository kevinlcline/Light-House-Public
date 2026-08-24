"""Contract tests for shared static UI assets."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
UI_ROOT = REPO_ROOT / "static" / "ui"

REQUIRED_ASSETS = [
    "theme-init.js",
    "theme.css",
    "base.css",
    "layout.css",
    "chat.css",
    "markdown.css",
    "notes.css",
    "faces.css",
    "theme.js",
    "lights.js",
    "markdown.js",
    "faces.js",
    "menu.js",
    "admin.js",
]

PAGES = {
    "index.html": {
        "assets": [
            "/static/ui/theme-init.js",
            "/static/ui/theme.css",
            "/static/ui/chat.css",
            "/static/ui/faces.css",
            "/static/ui/faces.js",
            "/static/ui/lights.js",
            "/static/ui/menu.js",
        ],
        "markers": ['id="agent-select"', 'id="theme-toggle"', 'class="chat-container"', 'id="face-stage"'],
    },
    "notes.html": {
        "assets": [
            "/static/ui/theme-init.js",
            "/static/ui/notes.css",
            "/static/ui/lights.js",
            "/static/ui/menu.js",
        ],
        "markers": ['id="toggle-tree-btn"', 'class="notes-layout"', 'id="theme-toggle"'],
    },
    "group.html": {
        "assets": [
            "/static/ui/theme-init.js",
            "/static/ui/chat.css",
            "/static/ui/theme.js",
            "/static/ui/menu.js",
            "/static/ui/faces.js",
            "/static/ui/faces.css",
            "/static/ui/markdown.js",
            "/static/ui/markdown.css",
            "/static/ui/tts.js",
            "/static/ui/bubble-actions.js",
        ],
        "markers": [
            'class="chat-container"',
            'id="theme-toggle"',
            'id="menu-trigger"',
            'id="join-queue"',
            "I want to speak",
            'id="face-stage"',
            "attachBubbleActions",
            "loadHumanVoices",
            'id="voice-toggle"',
            "Voice on",
        ],
    },
}


@pytest.mark.parametrize("name", REQUIRED_ASSETS)
def test_shared_ui_asset_exists(name: str) -> None:
    path = UI_ROOT / name
    assert path.is_file(), f"missing static/ui/{name}"


def test_lights_module_exports_api() -> None:
    text = (UI_ROOT / "lights.js").read_text(encoding="utf-8")
    assert "global.LightHouse" in text
    assert "fetchEnabled" in text
    assert "populateSelect" in text


def test_theme_module_exports_api() -> None:
    text = (UI_ROOT / "theme.js").read_text(encoding="utf-8")
    assert "setupThemeToggle" in text


@pytest.mark.parametrize("page_name,spec", PAGES.items())
def test_page_links_shared_assets(page_name: str, spec: dict) -> None:
    html = (REPO_ROOT / page_name).read_text(encoding="utf-8")
    for asset in spec["assets"]:
        assert asset in html, f"{page_name} should reference {asset}"
    for marker in spec["markers"]:
        assert marker in html, f"{page_name} should contain {marker}"


def test_index_does_not_duplicate_large_style_block() -> None:
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    assert "<style>" not in html
    assert 'href="/static/ui/theme.css"' in html


def test_notes_tree_toggle_not_only_in_menu() -> None:
    html = (REPO_ROOT / "notes.html").read_text(encoding="utf-8")
    assert 'id="toggle-tree-btn"' in html
    assert 'id="toggle-list"' not in html


@pytest.mark.parametrize("page_name", ["group.html", "index.html"])
def test_voice_toggle_is_visible_in_chat_menus(page_name: str) -> None:
    html = (REPO_ROOT / page_name).read_text(encoding="utf-8")
    match = re.search(r"<button[^>]*id=\"voice-toggle\"[^>]*>", html)
    assert match, f"{page_name} should include #voice-toggle"
    assert "hidden" not in match.group(0), (
        f"{page_name} #voice-toggle must not start hidden"
    )
    assert "setupVoiceToggle('#voice-toggle')" in html


def test_tts_menu_toggle_stays_visible_before_server_ready() -> None:
    text = (UI_ROOT / "tts.js").read_text(encoding="utf-8")
    fn = re.search(r"function syncToggleLabels\(\) \{.*?\n    \}", text, re.S)
    assert fn, "syncToggleLabels should exist"
    toggle_loop = re.search(
        r"querySelectorAll\('\[data-voice-toggle\]'\).*?\n        \}\);",
        fn.group(0),
        re.S,
    )
    assert toggle_loop, "syncToggleLabels should update [data-voice-toggle]"
    assert "el.hidden = !serverReady" not in toggle_loop.group(0)
    assert "Voice on" in fn.group(0)
    assert "Voice off" in fn.group(0)


def test_faces_emote_from_stage_cues() -> None:
    text = (UI_ROOT / "faces.js").read_text(encoding="utf-8")
    css = (UI_ROOT / "faces.css").read_text(encoding="utf-8")
    tts = (UI_ROOT / "tts.js").read_text(encoding="utf-8")
    assert "emoteFromText" in text
    assert "emotionTimeline" in text
    assert "syncSpeakingProgress" in text
    assert "pruneTo" in text
    assert "forgetAgent" in text
    assert "genderFromVoice" in text
    assert "face-bow-hair" in text
    assert "face-bow-tie" in text
    assert "#f0a0b8" in text
    assert "#d32f2f" in text
    assert "52,12" in text
    assert "32,64" in text
    assert "setVoices" in text
    assert "is-idle-bob" in text or "is-idle-bob" in css
    assert "face-bow-hair" in css
    assert "face-gender-girl" in css
    assert "face-gender-boy" in css
    assert "setVoices" in tts
    assert "emo-smile" in css
    assert "emo-bright" in css
    assert "emo-anger" in css
    assert "emo-kiss" in css
    assert "emo-pause" in css
    assert "emo-pause-smile" in css
    assert "face-eye-closed" in css
    assert "PAUSE_RE" in text
    assert "pause_smile" in text
    assert "face-cheek-arc" in css
    assert "chunkText" in tts
    assert "syncSpeakingProgress" in tts
    assert "😊" in text
    assert "EMOJI_MAP" in text
    assert "😘" in text
    assert "😠" in text
