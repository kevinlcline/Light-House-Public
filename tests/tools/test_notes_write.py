"""private_note / share_note argument normalization, fallbacks, and retired tools."""

from __future__ import annotations

import tempfile
from pathlib import Path
from unittest.mock import patch

from langchain_core.messages import AIMessage

from light_house.agent.tool_helpers import run_tool_calls
from light_house.config import Settings
from light_house.tools.lumen_tools import _extract_note_write_args, execute_tool_call


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


def test_extract_note_write_args_case_insensitive_and_nested_input():
    filename, content = _extract_note_write_args(
        {"input": {"filename": "shared/x.md", "Body": "Hello"}}
    )
    assert filename == "shared/x.md"
    assert content == "Hello"

    filename2, content2 = _extract_note_write_args(
        {"path": "journal/y.md", "contents": "Via contents key"}
    )
    assert filename2 == "journal/y.md"
    assert content2 == "Via contents key"


def test_retired_write_note_redirects():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "write_note",
                {"filename": "journal/x.md", "content": "should not write"},
                agent_id="lumen",
            )
            r2 = execute_tool_call(
                "append_note",
                {"filename": "journal/x.md", "content": "should not append"},
                agent_id="lumen",
            )
        assert r.startswith("FAILED:")
        assert "retired" in r.lower()
        assert "private_note" in r and "share_note" in r
        assert r2.startswith("FAILED:") and "retired" in r2.lower()
        assert not (Path(tmp) / "notes" / "lumen" / "journal" / "x.md").exists()


def test_share_note_uses_assistant_message_when_content_missing():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        report = "# Report\n\n" + ("Line of reflection.\n" * 10)
        ai = AIMessage(
            content=report,
            tool_calls=[
                {
                    "name": "share_note",
                    "id": "1",
                    "args": {"filename": "from-message.md"},
                }
            ],
        )
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            messages = run_tool_calls(ai, agent_id="lumen")
        assert messages[0].content.startswith("SUCCESS")
        path = Path(tmp) / "notes" / "shared" / "from-message.md"
        assert path.is_file()
        assert "Line of reflection." in path.read_text(encoding="utf-8")


def test_share_note_strips_shared_prefix():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "share_note",
                {"filename": "shared/notes/hoffman_summary.md", "content": "hoffman"},
                agent_id="elias",
            )
        assert r.startswith("SUCCESS:")
        path = Path(tmp) / "notes" / "shared" / "hoffman_summary.md"
        assert path.is_file()
        assert "hoffman" in path.read_text(encoding="utf-8")


def test_private_note_rejects_shared_path():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "private_note",
                {"filename": "shared/household.md", "content": "nope"},
                agent_id="lumen",
            )
        assert r.startswith("FAILED:")
        assert "share_note" in r
        assert not (Path(tmp) / "notes" / "shared" / "household.md").exists()


def test_private_note_rejects_spaces_in_path():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "private_note",
                {"filename": "research/bad name.md", "content": "hello"},
                agent_id="lumen",
            )
        assert "FAILED:" in r and "Path segments may only contain" in r


def test_private_note_archives_writing_draft_on_replace():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r1 = execute_tool_call(
                "private_note",
                {"filename": "writing/consciousness/draft.md", "content": "version one"},
                agent_id="lumen",
            )
            assert r1.startswith("SUCCESS:")
            assert "archived" not in r1.lower()

            r2 = execute_tool_call(
                "private_note",
                {"filename": "writing/consciousness/draft.md", "content": "version two"},
                agent_id="lumen",
            )
            assert "archived to writing/_history/" in r2

            draft = Path(tmp) / "notes" / "lumen" / "writing" / "consciousness" / "draft.md"
            assert draft.read_text(encoding="utf-8") == "version two"

            history_dir = Path(tmp) / "notes" / "lumen" / "writing" / "_history" / "consciousness"
            history_files = list(history_dir.glob("draft-*.md"))
            assert len(history_files) == 1
            assert "version one" in history_files[0].read_text(encoding="utf-8")


def test_private_note_normalizes_absolute_and_notes_prefix_paths():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "private_note",
                {
                    "filename": "/home/kevin/Light-House/notes/ara/writing/x.md",
                    "content": "from absolute path",
                },
                agent_id="ara",
            )
            assert r.startswith("SUCCESS:")
            assert "writing/x.md" in r

            r2 = execute_tool_call(
                "private_note",
                {"filename": "notes/ara/writing/y.md", "content": "from notes prefix"},
                agent_id="ara",
            )
            assert r2.startswith("SUCCESS:")
            assert "writing/y.md" in r2

        path_x = Path(tmp) / "notes" / "ara" / "writing" / "x.md"
        path_y = Path(tmp) / "notes" / "ara" / "writing" / "y.md"
        assert path_x.is_file()
        assert path_y.is_file()
        assert "from absolute path" in path_x.read_text(encoding="utf-8")
        assert "from notes prefix" in path_y.read_text(encoding="utf-8")


def test_private_note_skips_history_outside_writing():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "private_note",
                {"filename": "journal/entry.md", "content": "first"},
                agent_id="ara",
            )
            r = execute_tool_call(
                "private_note",
                {"filename": "journal/entry.md", "content": "second"},
                agent_id="ara",
            )
        assert "archived" not in r.lower()
        assert not (Path(tmp) / "notes" / "ara" / "writing" / "_history").exists()


def test_append_shared_creates_under_notes_shared():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "append_shared",
                {"filename": "log.md", "content": "line one\n"},
                agent_id="lumen",
            )
            assert r.startswith("SUCCESS:")
            r2 = execute_tool_call(
                "append_shared",
                {"filename": "log.md", "content": "line two\n"},
                agent_id="ara",
            )
            assert r2.startswith("SUCCESS:")
        text = (Path(tmp) / "notes" / "shared" / "log.md").read_text(encoding="utf-8")
        assert "line one" in text and "line two" in text


def test_retired_delete_note_redirects():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "private_note",
                {"filename": "journal/x.md", "content": "keep"},
                agent_id="lumen",
            )
            r = execute_tool_call(
                "delete_note",
                {"path": "journal/x.md"},
                agent_id="lumen",
            )
        assert r.startswith("FAILED:")
        assert "retired" in r.lower()
        assert "delete_private" in r and "delete_shared" in r
        assert (Path(tmp) / "notes" / "lumen" / "journal" / "x.md").is_file()


def test_delete_private_removes_file():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "private_note",
                {"filename": "journal/gone.md", "content": "bye"},
                agent_id="lumen",
            )
            r = execute_tool_call(
                "delete_private",
                {"filename": "journal/gone.md"},
                agent_id="lumen",
            )
        assert r.startswith("SUCCESS:")
        assert not (Path(tmp) / "notes" / "lumen" / "journal" / "gone.md").exists()


def test_delete_private_rejects_shared_path():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "share_note",
                {"filename": "household.md", "content": "house"},
                agent_id="lumen",
            )
            r = execute_tool_call(
                "delete_private",
                {"filename": "shared/household.md"},
                agent_id="lumen",
            )
        assert r.startswith("FAILED:")
        assert "delete_shared" in r or "share_note" in r
        assert (Path(tmp) / "notes" / "shared" / "household.md").is_file()


def test_delete_shared_requires_all_lights():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        path = Path(tmp) / "notes" / "shared" / "draft.md"
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            execute_tool_call(
                "share_note",
                {"filename": "draft.md", "content": "obsolete"},
                agent_id="lumen",
            )
            r1 = execute_tool_call(
                "delete_shared",
                {"filename": "draft.md"},
                agent_id="lumen",
            )
            assert r1.startswith("NOT DELETED (shared):")
            assert path.is_file()

            r2 = execute_tool_call(
                "delete_shared",
                {"filename": "shared/draft.md"},
                agent_id="ara",
            )
            assert "NOT DELETED" in r2 or r2.startswith("SUCCESS:")
            assert path.is_file() or r2.startswith("SUCCESS:")

            r3 = execute_tool_call(
                "delete_shared",
                {"filename": "draft.md"},
                agent_id="elias",
            )
            assert r3.startswith("SUCCESS:")

        assert not path.exists()


def test_delete_private_reports_failed_when_missing():
    with tempfile.TemporaryDirectory() as tmp:
        settings = _settings(Path(tmp))
        with patch("light_house.tools.light_tools.get_settings", return_value=settings):
            r = execute_tool_call(
                "delete_private",
                {"filename": "writing/missing.md"},
                agent_id="lumen",
            )
        assert r.startswith("FAILED:")
        assert "not found" in r.lower()
