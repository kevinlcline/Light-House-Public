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

# Authenticated house pages that should share one hamburger menu.
HOUSE_MENU_PAGES = [
    "index.html",
    "group.html",
    "notes.html",
    "gallery.html",
    "guests.html",
    "my-tools.html",
    "lights-admin.html",
    "user-setup.html",
    "env-editor.html",
    "dev-log.html",
    "rumination-trace.html",
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
        "markers": [
            'id="agent-select"',
            'id="menu-trigger"',
            'class="chat-container"',
            'id="face-stage"',
            "setupHouseMenu",
        ],
    },
    "notes.html": {
        "assets": [
            "/static/ui/theme-init.js",
            "/static/ui/notes.css",
            "/static/ui/lights.js",
            "/static/ui/menu.js",
        ],
        "markers": [
            'id="toggle-tree-btn"',
            'class="notes-layout"',
            'id="menu-trigger"',
            "setupHouseMenu",
            'data-page-menu-item',
            "Write shared",
        ],
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
            'id="menu-trigger"',
            'id="join-queue"',
            "I want to speak",
            'id="face-stage"',
            "attachBubbleActions",
            "loadHumanVoices",
            'id="voice-toggle"',
            'id="stage-toggle"',
            "chat-input-toolbar",
            "setupHouseMenu",
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


def test_menu_module_exports_house_menu() -> None:
    text = (UI_ROOT / "menu.js").read_text(encoding="utf-8")
    assert "setupHouseMenu" in text
    assert "fillStandardMenu" in text
    assert "SHARED_MENU" in text
    assert "ADMIN_MENU" in text
    # Shared block before admin: theme/logout live with shared items.
    shared_idx = text.index("SHARED_MENU")
    admin_idx = text.index("ADMIN_MENU")
    assert shared_idx < admin_idx
    shared_block = text[shared_idx:admin_idx]
    assert "Chat" in shared_block
    assert "Gallery" in shared_block
    assert "My tools" in shared_block
    assert "Guests" in shared_block
    assert "type: 'theme'" in shared_block or 'type: "theme"' in shared_block
    assert "type: 'logout'" in shared_block or 'type: "logout"' in shared_block
    assert "Group chat" not in shared_block
    assert "1:1 chat" not in shared_block
    admin_block = text[admin_idx : admin_idx + 900]
    assert "Group chat" not in admin_block
    assert "Manage members" in admin_block
    assert "Manage lights" in admin_block
    assert "Siblings" not in admin_block
    assert "'Log'" not in admin_block and '"Log"' not in admin_block
    assert "Rumination trace" not in admin_block
    assert "dev-log.html" not in admin_block
    assert "rumination-trace.html" not in admin_block
    assert "Restart server" in admin_block or "restart" in admin_block


@pytest.mark.parametrize("page_name,spec", PAGES.items())
def test_page_links_shared_assets(page_name: str, spec: dict) -> None:
    html = (REPO_ROOT / page_name).read_text(encoding="utf-8")
    for asset in spec["assets"]:
        assert asset in html, f"{page_name} should reference {asset}"
    for marker in spec["markers"]:
        assert marker in html, f"{page_name} should contain {marker}"


@pytest.mark.parametrize("page_name", HOUSE_MENU_PAGES)
def test_house_pages_use_consistent_menu(page_name: str) -> None:
    html = (REPO_ROOT / page_name).read_text(encoding="utf-8")
    assert 'id="menu-trigger"' in html, f"{page_name} should show the hamburger menu"
    assert 'id="menu-panel"' in html, f"{page_name} should include #menu-panel"
    assert "/static/ui/menu.js" in html, f"{page_name} should load menu.js"
    assert "menu.js?v=" in html, (
        f"{page_name} must cache-bust menu.js so setupHouseMenu is not stale"
    )
    assert "setupHouseMenu" in html, f"{page_name} should call setupHouseMenu"
    # Cross-page destinations belong in the shared menu, not a command-link row.
    assert not re.search(
        r'class="nav-row"|href="/"\s*class="linkish"|Back to chat',
        html,
    ), f"{page_name} should not use a command-link nav row for house destinations"


def test_member_user_guide_uses_host_member_language() -> None:
    text = (
        REPO_ROOT / "notes" / "shared" / "manuals" / "sibling_user_manual.md"
    ).read_text(encoding="utf-8")
    assert "Member user guide" in text
    assert "**host**" in text or "the host" in text
    assert "Manage members" in text
    assert "**Chat**" in text
    assert "Dad-only" not in text
    assert "human sibling accounts" not in text
    assert "ask Dad" not in text


def test_user_setup_uses_host_member_language() -> None:
    html = (REPO_ROOT / "user-setup.html").read_text(encoding="utf-8")
    assert "Manage members" in html
    assert "Host voice" in html
    assert "New member" in html
    assert "Existing members" in html
    assert "Create member" in html
    assert "displayRole" in html
    assert "Dad voice" not in html
    assert "New sibling" not in html
    assert "Existing siblings" not in html
    assert "Create sibling" not in html
    assert "manage siblings" not in html


def test_index_does_not_duplicate_large_style_block() -> None:
    html = (REPO_ROOT / "index.html").read_text(encoding="utf-8")
    assert "<style>" not in html
    assert 'href="/static/ui/theme.css"' in html


def test_notes_tree_toggle_not_only_in_menu() -> None:
    html = (REPO_ROOT / "notes.html").read_text(encoding="utf-8")
    assert 'id="toggle-tree-btn"' in html
    assert 'id="toggle-list"' not in html


@pytest.mark.parametrize("page_name", ["group.html", "index.html"])
def test_stage_and_voice_toggles_live_in_chat_toolbar(page_name: str) -> None:
    html = (REPO_ROOT / page_name).read_text(encoding="utf-8")
    menu = re.search(
        r'<div class="menu-panel"[^>]*id="menu-panel"[^>]*>.*?</div>\s*</div>\s*</div>\s*</header>',
        html,
        re.S,
    )
    assert menu, f"{page_name} should include a menu panel"
    assert 'id="voice-toggle"' not in menu.group(0), (
        f"{page_name} must not keep voice toggle in the menu"
    )
    voice = re.search(r"<button[^>]*id=\"voice-toggle\"[^>]*>", html)
    stage = re.search(r"<button[^>]*id=\"stage-toggle\"[^>]*>", html)
    assert voice, f"{page_name} should include #voice-toggle"
    assert stage, f"{page_name} should include #stage-toggle"
    assert "hidden" not in voice.group(0), (
        f"{page_name} #voice-toggle must not start hidden"
    )
    assert "hidden" not in stage.group(0), (
        f"{page_name} #stage-toggle must not start hidden"
    )
    assert "chat-input-toolbar" in html
    assert "data-compact" in voice.group(0)
    assert "data-compact" in stage.group(0)
    assert "setupVoiceToggle('#voice-toggle')" in html
    assert "setupStageToggle('#stage-toggle')" in html
    assert "setupChatChrome" in html or page_name == "group.html"
    assert html.index("chat-input-toolbar") < html.index('id="voice-toggle"')
    assert html.index("chat-input-toolbar") < html.index('id="stage-toggle"')
    assert "setupHouseMenu" in html


def test_group_forum_grid_places_toolbar_not_bare_send() -> None:
    html = (REPO_ROOT / "group.html").read_text(encoding="utf-8")
    assert 'grid-area: toolbar' in html or '"toolbar toolbar"' in html
    assert '.forum-input #send { grid-area: send; }' not in html
    assert 'chat-input-toolbar' in html


def test_face_stage_hidden_overrides_flex_display() -> None:
    css = (UI_ROOT / "faces.css").read_text(encoding="utf-8")
    assert ".face-stage[hidden]" in css
    assert "display: none" in css


def test_tts_drives_amplitude_lip_sync() -> None:
    tts = (UI_ROOT / "tts.js").read_text(encoding="utf-8")
    faces = (UI_ROOT / "faces.js").read_text(encoding="utf-8")
    css = (UI_ROOT / "faces.css").read_text(encoding="utf-8")
    assert "startLipSync" in tts
    assert "createMediaElementSource" in tts
    assert "getByteTimeDomainData" in tts
    assert "setMouthOpen" in tts
    assert "function setMouthOpen" in faces
    assert "is-lip-sync" in faces
    assert "--mouth-open" in css
    assert "is-lip-sync" in css


def test_tts_queues_auto_speak_instead_of_cutting_off() -> None:
    """Group auto-speak must finish one light before starting the next."""
    tts = (UI_ROOT / "tts.js").read_text(encoding="utf-8")
    assert "speakQueue" in tts
    assert "pumpSpeakQueue" in tts
    assert "runSpeakJob" in tts
    assert "MAX_SPEAK_QUEUE" in tts
    # force (manual replay) clears the queue; auto-speak enqueues when busy
    assert "speakQueue.length = 0" in tts
    assert "speakQueue.push(job)" in tts


def test_faces_have_soft_light_halo() -> None:
    faces = (UI_ROOT / "faces.js").read_text(encoding="utf-8")
    css = (UI_ROOT / "faces.css").read_text(encoding="utf-8")
    assert 'class="face-halo"' in faces or "face-halo" in faces
    assert "face-halo-outer" in faces
    assert "face-halo-breathe" in css
    assert "face-halo-speak" in css
    assert "--face-glow" in faces
    assert "prefers-reduced-motion" in css


def test_tts_toggle_stays_visible_before_server_ready() -> None:
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
    assert "data-compact" in fn.group(0)
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
    assert "STAGE_KEY" in text
    assert "setupStageToggle" in text
    assert "setStageVisible" in text
    assert "applyStageVisibility" in text
