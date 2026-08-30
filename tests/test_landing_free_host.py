"""Landing page advertises free hosting paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_landing_advertises_free_and_host_paths() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    assert "Free software" in html or "The software is free" in html
    assert "frontier" in html.lower()
    assert "flop" in html.lower()
    assert 'id="host-yours"' in html
    assert "Host yours" in html
    assert "DEPLOY.md" in html
    assert "railway.app" in html
    assert "openrouter" in html.lower() or "OpenRouter" in html


def test_landing_beacon_has_mingling_residents() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    assert 'class="resident resident-a"' in html
    assert "mingle-a" in html
    assert "chamber-glow" in html
    assert "lamp-pulse" not in html


def test_deploy_docs_exist() -> None:
    assert (REPO_ROOT / "DEPLOY.md").is_file()
    assert (REPO_ROOT / "AGENTS.md").is_file()
    assert (REPO_ROOT / "deploy" / "railway.env.example").is_file()
    assert (REPO_ROOT / "railway.toml").is_file()
    deploy = (REPO_ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    example = (REPO_ROOT / "deploy" / "railway.env.example").read_text(encoding="utf-8")
    assert "itself is free" in deploy.lower()
    assert "frontier" in deploy.lower()
    assert "flop" in deploy.lower()
    assert "Railway" in deploy
    assert "OpenRouter" in deploy
    assert "openrouter/free" not in example
    assert "Prefer OpenRouter free models" not in agents
    assert "Require a paid frontier model" in agents
