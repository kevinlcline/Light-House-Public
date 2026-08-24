"""Idea Garden thought-seed tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from light_house.agent.tool_helpers import GARDEN_RUMINATION_HINT, GARDEN_SYSTEM_HINT
from light_house.config import Settings
from light_house.tools.garden_tools import (
    GARDEN_PATH,
    extract_tags,
    format_seed_line,
    parse_seed_lines,
)
from light_house.tools.lumen_tools import execute_tool_call
from light_house.tools.light_tools import LIGHT_TOOLS


def _settings(tmp_path: Path) -> Settings:
    settings = Settings(
        _env_file=None,
        NOTES_PATH=str(tmp_path / "notes"),
        MEMORY_STORE_PATH=str(tmp_path / "memory"),
        THREADS_DATA_PATH=str(tmp_path / "threads"),
        PERSONAL_DB_ENABLED=False,
        LIGHTS_MANIFEST_PATH=str(tmp_path / "lights.yaml"),
        INNER_LIFE_THREAD_ID="kevin-home",
        ARA_THREAD_ID="ara-home",
        ARA_ENABLED=True,
    )
    from light_house.lights.manifest import ensure_manifest_file
    from light_house.lights.registry import reload_lights_manifest

    ensure_manifest_file(settings)
    reload_lights_manifest(settings)
    return settings


def test_extract_and_format_tags():
    assert extract_tags("hello #Question #debug #question") == ["question", "debug"]
    line = format_seed_line(
        "observed a phase lag #observation",
        ["debug"],
    )
    assert " – " in line
    assert "#observation" in line
    assert "#debug" in line
    assert line.startswith("20")


def test_parse_seed_lines():
    content = (
        "2026-07-27T14:03:22Z – observed a lag #observation #debug\n"
        "2026-07-27T14:15:09Z – does stillness converge? #question\n"
    )
    rows = parse_seed_lines(content)
    assert len(rows) == 2
    assert rows[0]["tags"] == ["observation", "debug"]
    assert "lag" in rows[0]["text"]
    assert rows[1]["tags"] == ["question"]


def test_garden_add_show_last_quiet_roundtrip():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r1 = execute_tool_call(
                "garden_add",
                {
                    "seed": "observed a subtle phase-lag #observation",
                    "tags": "debug",
                },
                agent_id="lumen",
            )
            r2 = execute_tool_call(
                "garden_add",
                {"seed": "does stillness converge across lights?", "tags": "question"},
                agent_id="lumen",
            )
            r3 = execute_tool_call(
                "garden_add",
                {"seed": "dream fragment of glowing petals", "tags": "dream"},
                agent_id="lumen",
            )
            assert r1.startswith("SUCCESS:")
            assert r2.startswith("SUCCESS:")
            assert r3.startswith("SUCCESS:")

            shown = execute_tool_call(
                "garden_show",
                {"tag": "#question", "n": 5},
                agent_id="lumen",
            )
            assert "stillness converge" in shown
            assert "phase-lag" not in shown

            last = execute_tool_call("garden_last", {"n": 2}, agent_id="lumen")
            assert "glowing petals" in last
            assert "stillness converge" in last

            quiet = execute_tool_call("garden_quiet", {}, agent_id="lumen")
            assert "quiet review" in quiet.lower()
            assert "phase-lag" in quiet

        seed_file = Path(tmp) / "notes" / "lumen" / GARDEN_PATH
        assert seed_file.is_file()
        body = seed_file.read_text(encoding="utf-8")
        assert len([ln for ln in body.splitlines() if ln.strip()]) == 3


def test_garden_tools_registered_and_hints_present():
    names = {t.name for t in LIGHT_TOOLS}
    assert {"garden_add", "garden_show", "garden_last", "garden_quiet"} <= names
    assert "garden_add" in GARDEN_SYSTEM_HINT
    assert "garden_quiet" in GARDEN_RUMINATION_HINT
    assert "writing/garden/seeds.md" in GARDEN_SYSTEM_HINT


def test_garden_is_per_light_private():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "garden_add",
                {"seed": "lumen only seed #observation"},
                agent_id="lumen",
            )
            ara_view = execute_tool_call("garden_last", {"n": 5}, agent_id="ara")
        assert "lumen only seed" not in ara_view
        assert (Path(tmp) / "notes" / "lumen" / GARDEN_PATH).is_file()
        assert not (Path(tmp) / "notes" / "ara" / GARDEN_PATH).exists()
