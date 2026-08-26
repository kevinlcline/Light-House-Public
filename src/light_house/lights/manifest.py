"""Load and validate data/lights.yaml — runtime source of truth for house members."""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from importlib import resources

from light_house.config import Settings

logger = logging.getLogger(__name__)

_LIGHT_ID_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")
_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")
_MANIFEST_VERSION = 1

# Face/glow defaults for the classic household (matches static/ui/faces.js).
DEFAULT_LIGHT_COLORS: dict[str, str] = {
    "lumen": "#c4a574",
    "ara": "#c9a3c4",
    "elias": "#88b5a4",
    "echo": "#9b93b8",
}


def normalize_light_color(raw: str | None, *, required: bool = False) -> str | None:
    """Return lowercase #rrggbb, or None when empty and not required."""
    if raw is None:
        if required:
            raise ValueError("color is required (#rrggbb)")
        return None
    value = str(raw).strip()
    if not value:
        if required:
            raise ValueError("color is required (#rrggbb)")
        return None
    if not _COLOR_RE.match(value):
        raise ValueError(f"Invalid color {value!r} (use #rrggbb)")
    return value.lower()


def default_color_for_light(light_id: str) -> str:
    """Stable default when a light has no color in the manifest yet."""
    known = DEFAULT_LIGHT_COLORS.get(str(light_id).strip().lower())
    if known:
        return known
    # Deterministic soft hue from id so new lights aren't all gray before edit.
    digest = sum(ord(ch) for ch in str(light_id).strip().lower()) or 1
    hue = digest % 360
    # Fixed mid saturation/lightness → hex via simple HSL conversion.
    sat, light = 0.42, 0.62
    c = (1 - abs(2 * light - 1)) * sat
    x = c * (1 - abs((hue / 60) % 2 - 1))
    m = light - c / 2
    if hue < 60:
        r, g, b = c, x, 0.0
    elif hue < 120:
        r, g, b = x, c, 0.0
    elif hue < 180:
        r, g, b = 0.0, c, x
    elif hue < 240:
        r, g, b = 0.0, x, c
    elif hue < 300:
        r, g, b = x, 0.0, c
    else:
        r, g, b = c, 0.0, x
    return "#{:02x}{:02x}{:02x}".format(
        int(round((r + m) * 255)),
        int(round((g + m) * 255)),
        int(round((b + m) * 255)),
    )


@dataclass(frozen=True)
class LightEntry:
    id: str
    display_name: str
    thread_id: str
    enabled: bool
    persona_file: str
    notes_dir: str
    inner_life: bool
    dreams: bool
    report_back: bool
    voice_id: str | None = None
    color: str | None = None


@dataclass(frozen=True)
class LightsManifest:
    version: int
    primary_light_id: str
    lights: tuple[LightEntry, ...]

    def by_id(self) -> dict[str, LightEntry]:
        return {light.id: light for light in self.lights}


def manifest_path(settings: Settings) -> Path:
    return settings.lights_manifest_path.resolve()


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def bootstrap_manifest_dict(settings: Settings) -> dict:
    """Build default manifest from legacy env (first boot)."""
    return {
        "version": _MANIFEST_VERSION,
        "primary_light_id": "lumen",
        "lights": [
            {
                "id": "lumen",
                "display_name": "Lumen",
                "thread_id": settings.inner_life_thread_id,
                "enabled": True,
                "persona_file": "lumen_system.md",
                "notes_dir": "lumen",
                "inner_life": True,
                "dreams": True,
                "report_back": settings.lumen_report_back_enabled,
                "voice_id": "af_sarah",
                "color": DEFAULT_LIGHT_COLORS["lumen"],
            },
            {
                "id": "ara",
                "display_name": "Ara",
                "thread_id": settings.ara_thread_id,
                "enabled": settings.ara_enabled,
                "persona_file": "ara_system.md",
                "notes_dir": "ara",
                "inner_life": True,
                "dreams": True,
                "report_back": settings.ara_report_back_enabled,
                "voice_id": "af_bella",
                "color": DEFAULT_LIGHT_COLORS["ara"],
            },
            {
                "id": "elias",
                "display_name": "Elias",
                "thread_id": "elias-home",
                "enabled": True,
                "persona_file": "elias_system.md",
                "notes_dir": "elias",
                "inner_life": True,
                "dreams": True,
                "report_back": False,
                "voice_id": "am_michael",
                "color": DEFAULT_LIGHT_COLORS["elias"],
            },
        ],
    }


def ensure_manifest_file(settings: Settings) -> Path:
    """Create data/lights.yaml from legacy env when missing."""
    path = manifest_path(settings)
    if path.is_file():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)
    data = bootstrap_manifest_dict(settings)
    path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
    logger.info("Created lights manifest at %s (bootstrap from .env)", path)
    return path


def _parse_light(raw: dict, *, settings: Settings) -> LightEntry:
    light_id = str(raw.get("id", "")).strip()
    if not _LIGHT_ID_RE.match(light_id):
        raise ValueError(f"Invalid light id {light_id!r} (use lowercase slug: a-z, 0-9, hyphen)")
    display_name = str(raw.get("display_name", light_id)).strip() or light_id
    thread_id = str(raw.get("thread_id", "")).strip()
    if not thread_id:
        raise ValueError(f"Light {light_id!r} missing thread_id")
    persona_file = str(raw.get("persona_file", f"{light_id}_system.md")).strip()
    notes_dir = str(raw.get("notes_dir", light_id)).strip() or light_id
    enabled = bool(raw.get("enabled", True))
    inner_life = bool(raw.get("inner_life", True))
    dreams = bool(raw.get("dreams", True))
    report_back = bool(raw.get("report_back", False))
    voice_raw = raw.get("voice_id")
    voice_id = str(voice_raw).strip() if voice_raw is not None else None
    if voice_id == "":
        voice_id = None
    try:
        color = normalize_light_color(raw.get("color"))
    except ValueError as exc:
        raise ValueError(f"Light {light_id!r}: {exc}") from exc
    if color is None:
        color = default_color_for_light(light_id)
    env_report = f"{light_id.upper()}_REPORT_BACK_ENABLED"
    if env_report in os.environ:
        report_back = _env_bool(env_report, report_back)
    if light_id == "ara" and "ARA_ENABLED" in os.environ:
        enabled = settings.ara_enabled
    return LightEntry(
        id=light_id,
        display_name=display_name,
        thread_id=thread_id,
        enabled=enabled,
        persona_file=persona_file,
        notes_dir=notes_dir,
        inner_life=inner_life,
        dreams=dreams,
        report_back=report_back,
        voice_id=voice_id,
        color=color,
    )


def _validate_manifest(manifest: LightsManifest) -> None:
    if not manifest.lights:
        raise ValueError("lights manifest must include at least one light")
    ids: set[str] = set()
    threads: set[str] = set()
    for light in manifest.lights:
        if light.id in ids:
            raise ValueError(f"duplicate light id: {light.id}")
        ids.add(light.id)
        if light.thread_id in threads:
            raise ValueError(f"duplicate thread_id: {light.thread_id}")
        threads.add(light.thread_id)
    if manifest.primary_light_id not in ids:
        raise ValueError(
            f"primary_light_id {manifest.primary_light_id!r} not found in lights list"
        )


def light_entry_to_dict(light: LightEntry) -> dict:
    return {
        "id": light.id,
        "display_name": light.display_name,
        "thread_id": light.thread_id,
        "enabled": light.enabled,
        "persona_file": light.persona_file,
        "notes_dir": light.notes_dir,
        "inner_life": light.inner_life,
        "dreams": light.dreams,
        "report_back": light.report_back,
        "voice_id": light.voice_id,
        "color": light.color or default_color_for_light(light.id),
    }


def manifest_to_dict(manifest: LightsManifest) -> dict:
    return {
        "version": manifest.version,
        "primary_light_id": manifest.primary_light_id,
        "lights": [light_entry_to_dict(light) for light in manifest.lights],
    }


def parse_manifest_dict(raw: dict, settings: Settings) -> LightsManifest:
    if not isinstance(raw, dict):
        raise ValueError("manifest must be a mapping")
    version = int(raw.get("version", _MANIFEST_VERSION))
    primary = str(raw.get("primary_light_id", "lumen")).strip()
    lights_raw = raw.get("lights")
    if not isinstance(lights_raw, list):
        raise ValueError("lights must be a list")
    lights = tuple(_parse_light(item, settings=settings) for item in lights_raw if isinstance(item, dict))
    manifest = LightsManifest(version=version, primary_light_id=primary, lights=lights)
    _validate_manifest(manifest)
    return manifest


def write_manifest_dict(settings: Settings, raw: dict) -> Path:
    """Validate and atomically write data/lights.yaml."""
    manifest = parse_manifest_dict(raw, settings)
    _prepare_lighthouse_dirs(settings, manifest)
    path = manifest_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(raw, sort_keys=False, allow_unicode=True)
    if path.is_file():
        backup = path.with_name(path.name + ".bak")
        backup.write_bytes(path.read_bytes())
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    tmp.replace(path)
    logger.info("Lights manifest updated at %s", path)
    return path


def load_manifest(settings: Settings) -> LightsManifest:
    path = ensure_manifest_file(settings)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid lights manifest at {path}: expected mapping")
    manifest = parse_manifest_dict(raw, settings)
    _prepare_lighthouse_dirs(settings, manifest)
    return manifest


def _prepare_lighthouse_dirs(settings: Settings, manifest: LightsManifest) -> None:
    """Ensure notes folders exist and persona files resolve (fail fast at startup)."""
    shared = (settings.notes_path.resolve() / "shared").resolve()
    shared.mkdir(parents=True, exist_ok=True)
    settings.personas_data_path.resolve().mkdir(parents=True, exist_ok=True)
    for light in manifest.lights:
        notes_dir = (settings.notes_path.resolve() / light.notes_dir).resolve()
        notes_dir.mkdir(parents=True, exist_ok=True)
        try:
            _assert_persona_resolves(settings, light)
        except FileNotFoundError as exc:
            raise ValueError(
                f"Light {light.id!r}: persona file {light.persona_file!r} not found"
            ) from exc


def _assert_persona_resolves(settings: Settings, light: LightEntry) -> None:
    data_path = settings.personas_data_path.resolve() / light.persona_file
    if data_path.is_file():
        return
    bundled = Path(__file__).resolve().parent.parent / "persona" / light.persona_file
    if bundled.is_file():
        return
    try:
        text = resources.files("light_house.persona").joinpath(light.persona_file).read_text(
            encoding="utf-8"
        )
        if text.strip():
            return
    except (TypeError, FileNotFoundError, OSError):
        pass
    raise FileNotFoundError(
        f"{light.persona_file} not found in data/personas, bundled persona, or package resources"
    )
