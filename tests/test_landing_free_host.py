"""Landing page advertises free hosting paths."""

from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_landing_advertises_free_and_host_paths() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    assert "Free software" in html or "Completely free" in html
    assert 'id="host-yours"' in html
    assert "Host yours" in html
    assert "DEPLOY.md" in html
    assert "railway.app" in html
    assert "openrouter" in html.lower() or "OpenRouter" in html


def test_deploy_docs_exist() -> None:
    assert (REPO_ROOT / "DEPLOY.md").is_file()
    assert (REPO_ROOT / "AGENTS.md").is_file()
    assert (REPO_ROOT / "deploy" / "railway.env.example").is_file()
    assert (REPO_ROOT / "railway.toml").is_file()
    deploy = (REPO_ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert "completely free" in deploy.lower() or "itself is free" in deploy.lower()
    assert "Railway" in deploy
    assert "OpenRouter" in deploy
    agents = (REPO_ROOT / "AGENTS.md").read_text(encoding="utf-8")
    assert "openrouter/free" in agents
