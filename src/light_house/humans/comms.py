"""Per-light allow switches for siblings (Dad is never deniable)."""

from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any

from light_house.config import Settings
from light_house.humans.identity import dad_user_id
from light_house.humans.store import get_human, list_humans

logger = logging.getLogger(__name__)
_LOCK = threading.Lock()


def _path(settings: Settings) -> Path:
    return settings.humans_comms_path.resolve()


def _read(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"version": 1, "allows": {}}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read comms allow map %s: %s", path, exc)
        return {"version": 1, "allows": {}}
    if not isinstance(data, dict):
        return {"version": 1, "allows": {}}
    allows = data.get("allows")
    if not isinstance(allows, dict):
        data["allows"] = {}
    data.setdefault("version", 1)
    return data


def _write(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def resolve_sibling_user_id(settings: Settings, raw: str) -> str | None:
    """
    Map a tool argument to a canonical sibling user_id.

    Accepts user_id or display_name (e.g. alt_kevin or Moose).
    """
    token = (raw or "").strip().lower()
    if not token:
        return None
    if token == dad_user_id(settings):
        return None
    human = get_human(settings, token)
    if human is not None and human.enabled:
        return human.user_id
    for sibling in list_humans(settings):
        if not sibling.enabled:
            continue
        if sibling.user_id == token:
            return sibling.user_id
        if (sibling.display_name or "").strip().lower() == token:
            return sibling.user_id
    return None


def sibling_alias_keys(settings: Settings, user_id: str) -> set[str]:
    """Keys that may appear in the allow map for this sibling (id + display name)."""
    keys = {user_id.strip().lower()}
    human = get_human(settings, user_id)
    if human is not None:
        name = (human.display_name or "").strip().lower()
        if name:
            keys.add(name)
    return keys


def light_allows_human(settings: Settings, *, light_id: str, user_id: str) -> bool:
    """True if light will speak with this human. Dad always True."""
    lid = (light_id or "").strip().lower()
    uid = (user_id or "").strip().lower()
    if not lid or not uid:
        return False
    if uid == dad_user_id(settings):
        return True
    aliases = sibling_alias_keys(settings, uid)
    path = _path(settings)
    with _LOCK:
        data = _read(path)
        by_light = data.get("allows") or {}
        light_map = by_light.get(lid) if isinstance(by_light, dict) else None
        if not isinstance(light_map, dict):
            return True  # default allow
        # Prefer canonical user_id; otherwise any alias key.
        if uid in light_map:
            return bool(light_map[uid])
        for alias in aliases:
            if alias in light_map:
                return bool(light_map[alias])
        return True


def set_light_allows_human(
    settings: Settings,
    *,
    light_id: str,
    user_id: str,
    allowed: bool,
) -> dict[str, bool]:
    """Set allow flag for a sibling. Refusing Dad raises ValueError."""
    lid = (light_id or "").strip().lower()
    raw = (user_id or "").strip()
    if not lid or not raw:
        raise ValueError("light_id and user_id are required")

    resolved = resolve_sibling_user_id(settings, raw)
    if resolved is None:
        if raw.strip().lower() == dad_user_id(settings):
            raise ValueError("Lights cannot opt out of Dad")
        raise ValueError(
            f"Unknown sibling '{raw}'. Use their user name (e.g. alt_kevin), "
            "not a nickname unless it matches their display name exactly."
        )
    uid = resolved
    if uid == dad_user_id(settings):
        raise ValueError("Lights cannot opt out of Dad")

    aliases = sibling_alias_keys(settings, uid)
    path = _path(settings)
    with _LOCK:
        data = _read(path)
        allows = data.setdefault("allows", {})
        if not isinstance(allows, dict):
            allows = {}
            data["allows"] = allows
        light_map = allows.setdefault(lid, {})
        if not isinstance(light_map, dict):
            light_map = {}
            allows[lid] = light_map
        # Drop stale nickname keys so display-name mistakes don't linger.
        for alias in list(aliases):
            if alias != uid and alias in light_map:
                del light_map[alias]
        light_map[uid] = bool(allowed)
        _write(path, data)
        return {k: bool(v) for k, v in light_map.items()}


def list_allows_for_light(settings: Settings, *, light_id: str) -> dict[str, bool]:
    lid = (light_id or "").strip().lower()
    path = _path(settings)
    with _LOCK:
        data = _read(path)
        by_light = data.get("allows") or {}
        light_map = by_light.get(lid) if isinstance(by_light, dict) else None
        if not isinstance(light_map, dict):
            return {}
        return {str(k): bool(v) for k, v in light_map.items()}


def purge_allows_for_human(
    settings: Settings,
    *,
    user_id: str,
    extra_keys: list[str] | None = None,
) -> int:
    """Remove allow-map keys for a deleted sibling. Pass display_name via extra_keys if already deleted."""
    uid = (user_id or "").strip().lower()
    if not uid:
        return 0
    keys = sibling_alias_keys(settings, uid)
    for raw in extra_keys or []:
        token = (raw or "").strip().lower()
        if token:
            keys.add(token)
    keys.add(uid)
    path = _path(settings)
    removed = 0
    with _LOCK:
        data = _read(path)
        allows = data.get("allows") or {}
        if not isinstance(allows, dict):
            return 0
        for lid, light_map in list(allows.items()):
            if not isinstance(light_map, dict):
                continue
            for key in list(keys):
                if key in light_map:
                    del light_map[key]
                    removed += 1
            if not light_map:
                del allows[lid]
        if removed:
            _write(path, data)
            logger.info("Purged %d comms allow entries for human=%s", removed, uid)
        return removed
