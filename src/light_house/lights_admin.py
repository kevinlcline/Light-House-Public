"""Admin CRUD for lights manifest, personas, and per-light LLM env keys."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from light_house.config import LLMProvider, Settings
from light_house.env_admin import EnvAdminError, merge_env_keys
from light_house.lights.manifest import (
    LightEntry,
    default_color_for_light,
    ensure_manifest_file,
    light_entry_to_dict,
    manifest_to_dict,
    normalize_light_color,
    parse_manifest_dict,
    write_manifest_dict,
    _parse_light,
)
from light_house.lights.registry import (
    get_light,
    load_persona,
    persona_write_path,
    reload_lights_manifest,
    resolve_persona_file,
)
from light_house.llm.profile import lookup_agent_llm_profile

logger = logging.getLogger(__name__)


class LightsAdminError(Exception):
    """Invalid lights admin operation."""


def _read_raw_manifest(settings: Settings) -> dict:
    path = ensure_manifest_file(settings)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise LightsAdminError("Invalid lights manifest: expected mapping")
    return raw


def read_manifest_full(settings: Settings) -> dict:
    reload_lights_manifest(settings)
    manifest = parse_manifest_dict(_read_raw_manifest(settings), settings)
    return manifest_to_dict(manifest)


def _write_raw_manifest(settings: Settings, raw: dict) -> Path:
    try:
        path = write_manifest_dict(settings, raw)
    except ValueError as exc:
        raise LightsAdminError(str(exc)) from exc
    reload_lights_manifest(settings)
    return path


def _light_dict_from_entry(entry: LightEntry) -> dict:
    return light_entry_to_dict(entry)


def read_light_detail(settings: Settings, light_id: str) -> dict:
    reload_lights_manifest(settings)
    light = get_light(light_id, settings)
    return _light_dict_from_entry(light)


def persona_stub(display_name: str) -> str:
    return f"""# {display_name}

You are {display_name}, a sovereign light in this household.

## Role
(Describe who you are and how you relate to the household.)

## Tone
(Warm, direct, curious — whatever fits.)

## Boundaries
(What you will and won't do.)
"""


def read_persona_detail(settings: Settings, light_id: str) -> dict[str, Any]:
    light = get_light(light_id, settings)
    path, source = resolve_persona_file(settings, light)
    content = load_persona(light_id, settings)
    return {
        "light_id": light_id,
        "persona_file": light.persona_file,
        "path": str(path),
        "source": source,
        "content": content,
        "size": len(content.encode("utf-8")),
    }


def write_persona_content(settings: Settings, light_id: str, content: str) -> dict[str, Any]:
    if "\x00" in content:
        raise LightsAdminError("Persona content cannot contain NUL bytes")
    encoded = content.encode("utf-8")
    if len(encoded) > settings.personas_max_bytes:
        raise LightsAdminError(
            f"Persona exceeds limit ({settings.personas_max_bytes} bytes)"
        )
    light = get_light(light_id, settings)
    path = persona_write_path(settings, light)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        backup = path.with_name(path.name + ".bak")
        backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)
    logger.info("Persona updated for light %s at %s", light_id, path)
    return {
        "light_id": light_id,
        "path": str(path),
        "source": "data",
        "size": len(encoded),
    }


@dataclass(frozen=True)
class LightLlmDto:
    provider: str
    model: str
    model_fallback: str | None
    inner_life_model: str | None


def read_light_llm(settings: Settings, light_id: str) -> LightLlmDto:
    get_light(light_id, settings)
    profile = lookup_agent_llm_profile(settings, light_id)
    return LightLlmDto(
        provider=profile.provider.value,
        model=profile.model,
        model_fallback=profile.model_fallback,
        inner_life_model=profile.inner_life_model,
    )


def write_light_llm(
    settings: Settings,
    repo_root: Path,
    light_id: str,
    *,
    provider: str,
    model: str,
    model_fallback: str | None = None,
    inner_life_model: str | None = None,
) -> dict[str, str]:
    get_light(light_id, settings)
    prefix = light_id.upper()
    try:
        LLMProvider(provider.lower())
    except ValueError as exc:
        raise LightsAdminError(f"Invalid provider {provider!r}") from exc
    if not model.strip():
        raise LightsAdminError("model is required")
    updates: dict[str, str | None] = {
        f"{prefix}_LLM_PROVIDER": provider.lower(),
        f"{prefix}_LLM_MODEL": model.strip(),
        f"{prefix}_LLM_MODEL_FALLBACK": model_fallback.strip() if model_fallback else None,
        f"{prefix}_INNER_LIFE_MODEL": inner_life_model.strip() if inner_life_model else None,
    }
    try:
        path, size = merge_env_keys(settings, repo_root, updates)
    except EnvAdminError as exc:
        raise LightsAdminError(str(exc)) from exc
    return {"path": str(path), "size": str(size)}


def create_light(
    settings: Settings,
    repo_root: Path,
    *,
    light_id: str,
    display_name: str,
    thread_id: str | None = None,
    enabled: bool = True,
    persona_file: str | None = None,
    notes_dir: str | None = None,
    inner_life: bool = True,
    dreams: bool = True,
    report_back: bool = False,
    voice_id: str | None = None,
    color: str | None = None,
    persona_content: str | None = None,
    set_primary: bool = False,
    llm_provider: str | None = None,
    llm_model: str | None = None,
    llm_model_fallback: str | None = None,
    llm_inner_life_model: str | None = None,
) -> dict:
    raw = _read_raw_manifest(settings)
    lights_raw = raw.get("lights")
    if not isinstance(lights_raw, list):
        raise LightsAdminError("Invalid manifest: lights must be a list")
    if any(isinstance(item, dict) and item.get("id") == light_id for item in lights_raw):
        raise LightsAdminError(f"Light id {light_id!r} already exists")

    from light_house.tts.voices_catalog import default_voice_for_light, normalize_voice_id

    try:
        color_norm = normalize_light_color(
            color if color is not None else default_color_for_light(light_id),
            required=True,
        )
    except ValueError as exc:
        raise LightsAdminError(str(exc)) from exc

    entry = {
        "id": light_id,
        "display_name": display_name,
        "thread_id": thread_id or f"{light_id}-home",
        "enabled": enabled,
        "persona_file": persona_file or f"{light_id}_system.md",
        "notes_dir": notes_dir or light_id,
        "inner_life": inner_life,
        "dreams": dreams,
        "report_back": report_back,
        "voice_id": normalize_voice_id(
            voice_id or default_voice_for_light(light_id),
            light_id=light_id,
        ),
        "color": color_norm,
    }
    stub = persona_content if persona_content is not None else persona_stub(display_name)
    light_entry = _parse_light(entry, settings=settings)
    write_path = persona_write_path(settings, light_entry)
    write_path.parent.mkdir(parents=True, exist_ok=True)
    write_path.write_text(stub, encoding="utf-8")
    (settings.notes_path.resolve() / light_entry.notes_dir).mkdir(parents=True, exist_ok=True)

    lights_raw.append(entry)
    if set_primary:
        raw["primary_light_id"] = light_id
    path = _write_raw_manifest(settings, raw)

    if llm_provider and llm_model:
        write_light_llm(
            settings,
            repo_root,
            light_id,
            provider=llm_provider,
            model=llm_model,
            model_fallback=llm_model_fallback,
            inner_life_model=llm_inner_life_model,
        )

    return {"path": str(path), "light": read_light_detail(settings, light_id)}


def update_light(
    settings: Settings,
    light_id: str,
    *,
    display_name: str | None = None,
    thread_id: str | None = None,
    enabled: bool | None = None,
    inner_life: bool | None = None,
    dreams: bool | None = None,
    report_back: bool | None = None,
    voice_id: str | None = None,
    color: str | None = None,
    set_primary: bool | None = None,
) -> dict:
    raw = _read_raw_manifest(settings)
    lights_raw = raw.get("lights")
    if not isinstance(lights_raw, list):
        raise LightsAdminError("Invalid manifest: lights must be a list")
    found = False
    for item in lights_raw:
        if not isinstance(item, dict) or item.get("id") != light_id:
            continue
        found = True
        if display_name is not None:
            item["display_name"] = display_name
        if thread_id is not None:
            item["thread_id"] = thread_id
        if enabled is not None:
            item["enabled"] = enabled
        if inner_life is not None:
            item["inner_life"] = inner_life
        if dreams is not None:
            item["dreams"] = dreams
        if report_back is not None:
            item["report_back"] = report_back
        if voice_id is not None:
            from light_house.tts.voices_catalog import normalize_voice_id

            item["voice_id"] = normalize_voice_id(voice_id, light_id=light_id)
        if color is not None:
            try:
                item["color"] = normalize_light_color(color, required=True)
            except ValueError as exc:
                raise LightsAdminError(str(exc)) from exc
        break
    if not found:
        raise LightsAdminError(f"Unknown light id: {light_id}")
    if set_primary:
        raw["primary_light_id"] = light_id
    path = _write_raw_manifest(settings, raw)
    return {"path": str(path), "light": read_light_detail(settings, light_id)}


def delete_light(settings: Settings, light_id: str) -> dict:
    raw = _read_raw_manifest(settings)
    lights_raw = raw.get("lights")
    if not isinstance(lights_raw, list):
        raise LightsAdminError("Invalid manifest: lights must be a list")
    remaining = [item for item in lights_raw if not (isinstance(item, dict) and item.get("id") == light_id)]
    if len(remaining) == len(lights_raw):
        raise LightsAdminError(f"Unknown light id: {light_id}")
    if len(remaining) == 0:
        raise LightsAdminError("Cannot delete the last light")
    if raw.get("primary_light_id") == light_id:
        raise LightsAdminError(
            "Cannot delete primary light — set another light as primary first"
        )
    raw["lights"] = remaining
    path = _write_raw_manifest(settings, raw)
    return {"path": str(path), "deleted": light_id}
