"""Agent-controlled report-back to shared/reports/ (Proposal 3 phase 4)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone

from light_house.agents.registry import get_agent, validate_agent_id
from light_house.config import Settings

logger = logging.getLogger(__name__)

_REPORTS_PREFIX = "shared/reports/"


def _write_report_file(settings: Settings, relative_path: str, markdown: str) -> None:
    """Write under notes/shared/reports/ without importing the tools package (avoids import cycles)."""
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized.startswith(_REPORTS_PREFIX):
        raise ValueError("Report path must start with shared/reports/")
    if ".." in normalized.split("/"):
        raise ValueError("Invalid report path")

    notes_base = settings.notes_path.resolve()
    shared_dir = (notes_base / "shared").resolve()
    inner = normalized[len("shared/") :]
    target = (shared_dir / inner).resolve()
    if not str(target).startswith(str(shared_dir)):
        raise ValueError("Report path must stay inside shared/reports/")

    max_chars = max(1, settings.notes_max_chars_per_write)
    if len(markdown) > max_chars:
        raise ValueError(f"Content too long (max {max_chars} characters)")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")

_SLUG_SAFE = re.compile(r"[^a-z0-9]+")


def report_back_enabled(settings: Settings, agent_id: str) -> bool:
    from light_house.lights.registry import get_light

    validate_agent_id(agent_id)
    if agent_id == "lumen":
        return settings.lumen_report_back_enabled
    if agent_id == "ara":
        return settings.ara_report_back_enabled
    return get_light(agent_id, settings).report_back


def _slugify_title(title: str) -> str:
    slug = _SLUG_SAFE.sub("-", title.lower()).strip("-")
    return slug[:48] if slug else "report"


def write_shared_report(
    settings: Settings,
    *,
    agent_id: str,
    title: str,
    content: str,
) -> str:
    """Write a deliberate shared report markdown file; never injects into Kevin's chat."""
    if not report_back_enabled(settings, agent_id):
        return (
            "report_to_shared failed: report-back is disabled for this agent "
            f"(set {agent_id.upper()}_REPORT_BACK_ENABLED=true in .env)."
        )
    title_clean = title.strip()
    body_clean = content.strip()
    if not title_clean:
        return "report_to_shared failed: title is required."
    if not body_clean:
        return "report_to_shared failed: content is required."
    max_chars = settings.report_back_max_chars
    if len(body_clean) > max_chars:
        return (
            f"report_to_shared failed: content exceeds maximum length "
            f"({max_chars} characters)."
        )

    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify_title(title_clean)
    relative_path = f"shared/reports/{agent_id}-{ts}-{slug}.md"
    agent_name = get_agent(agent_id, settings).display_name
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    markdown = (
        f"# {title_clean}\n\n"
        f"_By {agent_name} · {iso}_\n\n"
        f"{body_clean}\n"
    )

    try:
        _write_report_file(settings, relative_path, markdown)
    except (ValueError, OSError) as exc:
        logger.warning("report_to_shared write failed agent=%s: %s", agent_id, exc)
        return f"report_to_shared failed: {exc}"

    _publish_report_ready(settings, agent_id=agent_id, path=relative_path, title=title_clean)
    logger.info(
        "Shared report written agent=%s path=%s chars=%d",
        agent_id,
        relative_path,
        len(body_clean),
    )
    return f"SUCCESS: report saved to {relative_path}"


def _publish_report_ready(
    settings: Settings,
    *,
    agent_id: str,
    path: str,
    title: str,
) -> None:
    if not settings.event_bus_enabled:
        return
    try:
        from light_house.events import EventType, LightHouseEvent, publish

        publish(
            LightHouseEvent(
                event_type=EventType.REPORT_READY,
                agent_id=agent_id,
                payload={"path": path, "title": title},
            )
        )
    except Exception:
        logger.exception("Failed to publish REPORT_READY for %s", path)
