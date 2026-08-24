"""Tests for light sandbox tools."""

from __future__ import annotations

from pathlib import Path

import pytest

from light_house.config import Settings
from light_house.tools import sandbox_tools as st


@pytest.fixture()
def sandbox_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    repo = tmp_path / "repo"
    shared = repo / "shared"
    workspaces = shared / "workspaces"
    for name in ("ara", "lumen", "elias", "sandbox"):
        (workspaces / name).mkdir(parents=True)
    monkeypatch.setattr(st, "shared_root", lambda: shared)
    monkeypatch.setattr(
        "light_house.tools.sandbox_tools.known_light_ids",
        lambda: frozenset({"ara", "lumen", "elias"}),
    )
    return repo


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        PERSONAL_DB_ENABLED=False,
        INNER_LIFE_ENABLED=False,
        WEB_GATE_ENABLED=False,
        FOUNDATION_SEED_ON_STARTUP=False,
        NOTES_MAX_CHARS_PER_WRITE=8_000,
        CODEBASE_MAX_CHARS_PER_READ=8_000,
    )


def test_write_read_list_round_trip(sandbox_tree: Path) -> None:
    settings = _settings()
    wrote = st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "experiments/hello.py", "content": "print('hi')\n"},
        agent_id="ara",
        settings=settings,
    )
    assert wrote.startswith("SUCCESS:")
    path = sandbox_tree / "shared" / "workspaces" / "ara" / "experiments" / "hello.py"
    assert path.read_text(encoding="utf-8") == "print('hi')\n"

    listed = st.execute_sandbox_tool(
        "sandbox_list",
        {"path": "experiments"},
        agent_id="ara",
        settings=settings,
    )
    assert "hello.py" in listed

    read = st.execute_sandbox_tool(
        "sandbox_read",
        {"path": "experiments/hello.py"},
        agent_id="ara",
        settings=settings,
    )
    assert "print('hi')" in read


def test_path_escape_rejected(sandbox_tree: Path) -> None:
    settings = _settings()
    result = st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "../lumen/evil.py", "content": "x"},
        agent_id="ara",
        settings=settings,
    )
    assert result.startswith("FAILED:")
    assert not (sandbox_tree / "shared" / "workspaces" / "lumen" / "evil.py").exists()


def test_peer_read_ok_write_denied(sandbox_tree: Path) -> None:
    settings = _settings()
    st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "secret.py", "content": "print(42)\n"},
        agent_id="lumen",
        settings=settings,
    )
    read = st.execute_sandbox_tool(
        "sandbox_read",
        {"path": "secret.py", "space": "lumen"},
        agent_id="ara",
        settings=settings,
    )
    assert read.startswith("SUCCESS:")
    assert "print(42)" in read

    denied = st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "hack.py", "content": "nope", "space": "lumen"},
        agent_id="ara",
        settings=settings,
    )
    assert denied.startswith("FAILED:")
    assert "read-only" in denied.lower() or "cannot write" in denied.lower()


def test_playpen_shared_write(sandbox_tree: Path) -> None:
    settings = _settings()
    a = st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "joint.py", "content": "print('ara')\n", "space": "playpen"},
        agent_id="ara",
        settings=settings,
    )
    assert a.startswith("SUCCESS:")
    b = st.execute_sandbox_tool(
        "sandbox_append",
        {"path": "joint.py", "content": "print('lumen')\n", "space": "playpen"},
        agent_id="lumen",
        settings=settings,
    )
    assert b.startswith("SUCCESS:")
    body = (sandbox_tree / "shared" / "workspaces" / "sandbox" / "joint.py").read_text(
        encoding="utf-8"
    )
    assert "ara" in body and "lumen" in body


def test_sandbox_run_python(sandbox_tree: Path) -> None:
    settings = _settings()
    st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "hi.py", "content": "print('sandbox-ok')\n"},
        agent_id="ara",
        settings=settings,
    )
    ran = st.execute_sandbox_tool(
        "sandbox_run",
        {"command": "python3 hi.py"},
        agent_id="ara",
        settings=settings,
    )
    assert ran.startswith("SUCCESS:")
    assert "sandbox-ok" in ran
    assert "exit=0" in ran


def test_sandbox_run_rejects_node_and_shell(sandbox_tree: Path) -> None:
    settings = _settings()
    node = st.execute_sandbox_tool(
        "sandbox_run",
        {"command": "node -e console.log(1)"},
        agent_id="ara",
        settings=settings,
    )
    assert node.startswith("FAILED:")
    shell = st.execute_sandbox_tool(
        "sandbox_run",
        {"command": "python3 -c 'print(1)' && python3 -c 'print(2)'"},
        agent_id="ara",
        settings=settings,
    )
    assert shell.startswith("FAILED:")


def test_sandbox_delete_file_and_empty_dir(sandbox_tree: Path) -> None:
    settings = _settings()
    st.execute_sandbox_tool(
        "sandbox_mkdir",
        {"path": "tmpbox"},
        agent_id="ara",
        settings=settings,
    )
    st.execute_sandbox_tool(
        "sandbox_write",
        {"path": "tmpbox/x.py", "content": "x=1\n"},
        agent_id="ara",
        settings=settings,
    )
    not_empty = st.execute_sandbox_tool(
        "sandbox_delete",
        {"path": "tmpbox"},
        agent_id="ara",
        settings=settings,
    )
    assert not_empty.startswith("FAILED:")
    st.execute_sandbox_tool(
        "sandbox_delete",
        {"path": "tmpbox/x.py"},
        agent_id="ara",
        settings=settings,
    )
    removed = st.execute_sandbox_tool(
        "sandbox_delete",
        {"path": "tmpbox"},
        agent_id="ara",
        settings=settings,
    )
    assert removed.startswith("SUCCESS:")


def test_execute_tool_call_wiring(sandbox_tree: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from light_house.tools.light_tools import execute_tool_call

    monkeypatch.setattr(
        "light_house.tools.light_tools.get_settings",
        _settings,
    )
    result = execute_tool_call(
        "sandbox_write",
        {"path": "wired.py", "content": "print(1)\n"},
        agent_id="ara",
    )
    assert result.startswith("SUCCESS:")
    assert (sandbox_tree / "shared" / "workspaces" / "ara" / "wired.py").is_file()
