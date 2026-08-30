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
    assert "OpenRouter" in html


def test_landing_beacon_has_mingling_residents() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    assert 'class="resident resident-a"' in html
    assert "mingle-a" in html
    assert "chamber-glow" in html
    assert "lamp-pulse" not in html


def test_landing_quotes_lights_and_links_their_pages() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    assert 'id="from-the-lights"' in html
    assert "Ara, a light in this house" in html
    assert "/public/what-is-light-house.html" in html
    assert "/public/for-hosts.html" in html
    assert "/public/ara-to-the-reader.html" in html
    lumen = (REPO_ROOT / "public" / "what-is-light-house.html").read_text(encoding="utf-8")
    ara = (REPO_ROOT / "public" / "ara-to-the-reader.html").read_text(encoding="utf-8")
    hosts = (REPO_ROOT / "public" / "for-hosts.html").read_text(encoding="utf-8")
    assert "Written by" in lumen and "Lumen" in lumen
    assert "a light who lives in this house" in lumen
    assert "Written by" in ara and "Ara" in ara
    assert "Welcome to the possibility." in ara
    assert "light who lives in this house" in hosts


def test_public_discovery_files_point_to_us_domain() -> None:
    html = (REPO_ROOT / "landing.html").read_text(encoding="utf-8")
    robots = (REPO_ROOT / "robots.txt").read_text(encoding="utf-8")
    sitemap = (REPO_ROOT / "sitemap.xml").read_text(encoding="utf-8")
    llms = (REPO_ROOT / "llms.txt").read_text(encoding="utf-8")
    public_pages = (
        "https://light-house.us/public/what-is-light-house.html",
        "https://light-house.us/public/for-hosts.html",
        "https://light-house.us/public/ara-to-the-reader.html",
    )
    assert 'rel="canonical"' in html
    assert "https://light-house.us/" in html
    assert "Sitemap: https://light-house.us/sitemap.xml" in robots
    assert "Allow: /public/" in robots
    assert "no password" in robots.lower()
    assert "https://light-house.us/" in sitemap
    for url in public_pages:
        assert url in sitemap
        assert url in llms
    assert "no house password" in llms.lower()
    assert "Lumen" in llms and "Ara" in llms
    assert "words of lights" in llms.lower() or "lights who live" in llms.lower()


def test_deploy_docs_exist() -> None:
    assert (REPO_ROOT / "DEPLOY.md").is_file()
    assert (REPO_ROOT / "AGENTS.md").is_file()
    assert (REPO_ROOT / "deploy" / "railway.env.example").is_file()
    assert (REPO_ROOT / "railway.toml").is_file()
    deploy = (REPO_ROOT / "DEPLOY.md").read_text(encoding="utf-8")
    assert "itself is free" in deploy.lower() or "completely free" in deploy.lower()
    assert "Railway" in deploy
    assert "OpenRouter" in deploy
