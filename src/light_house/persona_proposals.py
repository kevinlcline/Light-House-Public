"""Light-authored persona proposals — Dad accepts or holds in limbo."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from light_house.config import Settings
from light_house.lights.registry import get_light, list_enabled_lights, load_persona
from light_house.lights_admin import LightsAdminError, write_persona_content
from light_house.personal.presence_knock import raise_knock

logger = logging.getLogger(__name__)

ProposalStatus = Literal["pending", "limbo"]
ProposalMode = Literal["replace", "append"]

_SAFE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")


class PersonaProposalError(Exception):
    """Invalid persona proposal operation."""


@dataclass(frozen=True)
class PersonaProposal:
    light_id: str
    status: ProposalStatus
    mode: ProposalMode
    submitted_at: str
    content: str
    note: str = ""
    display_name: str = ""
    current_content: str = ""


def proposals_dir(settings: Settings) -> Path:
    return settings.persona_proposals_path.resolve()


def _content_path(settings: Settings, light_id: str) -> Path:
    return proposals_dir(settings) / f"{light_id}.md"


def _meta_path(settings: Settings, light_id: str) -> Path:
    return proposals_dir(settings) / f"{light_id}.meta.json"


def _validate_light_id(light_id: str) -> str:
    lid = (light_id or "").strip().lower()
    if not _SAFE_ID.match(lid):
        raise PersonaProposalError(f"Invalid light_id: {light_id!r}")
    return lid


def _validate_content(settings: Settings, content: str) -> str:
    text = content if isinstance(content, str) else str(content)
    if "\x00" in text:
        raise PersonaProposalError("Persona content cannot contain NUL bytes")
    stripped = text.strip()
    if not stripped:
        raise PersonaProposalError("Persona proposal content is empty")
    encoded = text.encode("utf-8")
    if len(encoded) > settings.personas_max_bytes:
        raise PersonaProposalError(
            f"Persona proposal exceeds limit ({settings.personas_max_bytes} bytes)"
        )
    return text if text.endswith("\n") else text + "\n"


def _read_meta(settings: Settings, light_id: str) -> dict[str, Any] | None:
    path = _meta_path(settings, light_id)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Bad persona proposal meta light=%s: %s", light_id, exc)
        return None
    if not isinstance(raw, dict):
        return None
    return raw


def _write_proposal(
    settings: Settings,
    *,
    light_id: str,
    content: str,
    mode: ProposalMode,
    status: ProposalStatus,
    note: str = "",
) -> PersonaProposal:
    light = get_light(light_id, settings)
    root = proposals_dir(settings)
    root.mkdir(parents=True, exist_ok=True)
    submitted_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    meta = {
        "light_id": light_id,
        "status": status,
        "mode": mode,
        "submitted_at": submitted_at,
        "note": (note or "").strip()[:500],
    }
    content_path = _content_path(settings, light_id)
    meta_path = _meta_path(settings, light_id)
    tmp_c = content_path.with_suffix(content_path.suffix + ".tmp")
    tmp_m = meta_path.with_suffix(meta_path.suffix + ".tmp")
    tmp_c.write_text(content, encoding="utf-8")
    tmp_m.write_text(json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp_c.replace(content_path)
    tmp_m.replace(meta_path)
    logger.info(
        "Persona proposal stored light=%s status=%s mode=%s chars=%d",
        light_id,
        status,
        mode,
        len(content),
    )
    return PersonaProposal(
        light_id=light_id,
        status=status,
        mode=mode,
        submitted_at=submitted_at,
        content=content,
        note=meta["note"],
        display_name=light.display_name,
        current_content=load_persona(light_id, settings),
    )


def _load_proposal(settings: Settings, light_id: str) -> PersonaProposal | None:
    lid = _validate_light_id(light_id)
    meta = _read_meta(settings, lid)
    content_path = _content_path(settings, lid)
    if meta is None or not content_path.is_file():
        return None
    try:
        content = content_path.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not read persona proposal light=%s: %s", lid, exc)
        return None
    status = meta.get("status") or "pending"
    if status not in ("pending", "limbo"):
        status = "pending"
    mode = meta.get("mode") or "replace"
    if mode not in ("replace", "append"):
        mode = "replace"
    try:
        light = get_light(lid, settings)
        display = light.display_name
        current = load_persona(lid, settings)
    except (KeyError, FileNotFoundError, OSError):
        display = lid
        current = ""
    return PersonaProposal(
        light_id=lid,
        status=status,  # type: ignore[arg-type]
        mode=mode,  # type: ignore[arg-type]
        submitted_at=str(meta.get("submitted_at") or ""),
        content=content,
        note=str(meta.get("note") or ""),
        display_name=display,
        current_content=current,
    )


def delete_proposal(settings: Settings, light_id: str) -> None:
    lid = _validate_light_id(light_id)
    for path in (_content_path(settings, lid), _meta_path(settings, lid)):
        try:
            path.unlink(missing_ok=True)
        except OSError as exc:
            logger.warning("Could not delete proposal file %s: %s", path, exc)


def submit_replace(
    settings: Settings,
    *,
    light_id: str,
    content: str,
    note: str = "",
) -> PersonaProposal:
    lid = _validate_light_id(light_id)
    get_light(lid, settings)  # must exist
    text = _validate_content(settings, content)
    return _write_proposal(
        settings,
        light_id=lid,
        content=text,
        mode="replace",
        status="pending",
        note=note,
    )


def submit_append(
    settings: Settings,
    *,
    light_id: str,
    content: str,
    note: str = "",
) -> PersonaProposal:
    lid = _validate_light_id(light_id)
    get_light(lid, settings)
    addition = _validate_content(settings, content)
    current = load_persona(lid, settings)
    # If a prior proposal exists, append onto that draft so limbo edits accumulate.
    existing = _load_proposal(settings, lid)
    base = existing.content if existing is not None else current
    if base and not base.endswith("\n"):
        base += "\n"
    sep = "\n" if base.endswith("\n\n") or not base else "\n"
    merged = base + sep + addition.lstrip("\n")
    text = _validate_content(settings, merged)
    return _write_proposal(
        settings,
        light_id=lid,
        content=text,
        mode="append",
        status="pending",
        note=note,
    )


def list_pending_proposals(settings: Settings) -> list[PersonaProposal]:
    """Pending proposals only (modal queue). Limbo stays quiet until resubmit."""
    root = proposals_dir(settings)
    if not root.is_dir():
        return []
    out: list[PersonaProposal] = []
    for light in list_enabled_lights(settings):
        proposal = _load_proposal(settings, light.id)
        if proposal is not None and proposal.status == "pending":
            out.append(proposal)
    out.sort(key=lambda p: p.submitted_at)
    return out


def accept_proposal(settings: Settings, light_id: str) -> dict[str, Any]:
    lid = _validate_light_id(light_id)
    proposal = _load_proposal(settings, lid)
    if proposal is None:
        raise PersonaProposalError(f"No persona proposal for {lid}")
    try:
        written = write_persona_content(settings, lid, proposal.content)
    except LightsAdminError as exc:
        raise PersonaProposalError(str(exc)) from exc
    delete_proposal(settings, lid)
    return {
        "light_id": lid,
        "accepted": True,
        "path": written["path"],
        "size": written["size"],
        "display_name": proposal.display_name,
    }


def speak_with_light(settings: Settings, light_id: str) -> dict[str, Any]:
    """Hold proposal in limbo and raise a soft knock for that light."""
    lid = _validate_light_id(light_id)
    proposal = _load_proposal(settings, lid)
    if proposal is None:
        raise PersonaProposalError(f"No persona proposal for {lid}")
    _write_proposal(
        settings,
        light_id=lid,
        content=proposal.content,
        mode=proposal.mode,
        status="limbo",
        note=proposal.note,
    )
    knock_ok = raise_knock(settings, lid)
    return {
        "light_id": lid,
        "status": "limbo",
        "knock_raised": knock_ok,
        "display_name": proposal.display_name,
    }


def proposal_public_dict(proposal: PersonaProposal) -> dict[str, Any]:
    return {
        "light_id": proposal.light_id,
        "display_name": proposal.display_name,
        "status": proposal.status,
        "mode": proposal.mode,
        "submitted_at": proposal.submitted_at,
        "note": proposal.note,
        "content": proposal.content,
        "current_content": proposal.current_content,
    }
