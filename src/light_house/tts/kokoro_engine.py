"""Warm-loaded Kokoro ONNX TTS for Light-House."""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from typing import Any

import soundfile as sf

from light_house.config import Settings
from light_house.tts.text_prep import text_for_speech
from light_house.tts.voices_catalog import default_voice_for_light, normalize_voice_id

logger = logging.getLogger(__name__)

_LOCK = threading.Lock()
_ENGINE: Any | None = None
_ENGINE_PATHS: tuple[Path, Path] | None = None
_WARM_ERROR: str | None = None


class KokoroTtsError(RuntimeError):
    """TTS unavailable or synthesis failed."""


def _model_files(settings: Settings) -> tuple[Path, Path]:
    root = settings.kokoro_model_path.resolve()
    model = root / settings.kokoro_model_filename
    voices = root / settings.kokoro_voices_filename
    return model, voices


def voice_for_agent(settings: Settings, agent_id: str) -> str:
    """Resolve Kokoro voice for a light: env override → lights.yaml → defaults."""
    aid = (agent_id or "").strip().lower()
    env_overrides = {
        "lumen": settings.tts_voice_lumen,
        "ara": settings.tts_voice_ara,
        "elias": settings.tts_voice_elias,
    }
    if aid in env_overrides and env_overrides[aid]:
        return normalize_voice_id(env_overrides[aid], light_id=aid)
    try:
        from light_house.lights.registry import get_light

        light = get_light(aid, settings)
        if light.voice_id:
            return normalize_voice_id(light.voice_id, light_id=aid)
    except Exception:
        pass
    if settings.tts_default_voice:
        return normalize_voice_id(settings.tts_default_voice, light_id=aid)
    return default_voice_for_light(aid)


def get_tts_status(settings: Settings) -> dict[str, object]:
    model, voices = _model_files(settings)
    with _LOCK:
        ready = _ENGINE is not None
        err = _WARM_ERROR
    per_light: dict[str, str] = {"default": normalize_voice_id(settings.tts_default_voice)}
    try:
        from light_house.lights.registry import list_lights

        for light in list_lights(settings):
            per_light[light.id] = voice_for_agent(settings, light.id)
    except Exception:
        for lid in ("lumen", "ara", "elias"):
            per_light[lid] = voice_for_agent(settings, lid)
    return {
        "enabled": bool(settings.tts_enabled),
        "ready": ready,
        "engine": "kokoro-onnx",
        "model_path": str(model),
        "voices_path": str(voices),
        "model_present": model.is_file(),
        "voices_present": voices.is_file(),
        "error": err,
        "voices": per_light,
    }


def warm_tts(settings: Settings) -> bool:
    """Load Kokoro into memory. Returns True if ready."""
    global _ENGINE, _ENGINE_PATHS, _WARM_ERROR
    if not settings.tts_enabled:
        with _LOCK:
            _ENGINE = None
            _ENGINE_PATHS = None
            _WARM_ERROR = None
        logger.info("TTS disabled (TTS_ENABLED=false)")
        return False

    model, voices = _model_files(settings)
    if not model.is_file() or not voices.is_file():
        msg = (
            f"Kokoro model files missing under {settings.kokoro_model_path}. "
            "Download kokoro-v1.0.onnx and voices-v1.0.bin "
            "(see models/kokoro/README.md)."
        )
        with _LOCK:
            _ENGINE = None
            _ENGINE_PATHS = None
            _WARM_ERROR = msg
        logger.warning(msg)
        return False

    with _LOCK:
        if _ENGINE is not None and _ENGINE_PATHS == (model, voices):
            return True
        try:
            from kokoro_onnx import Kokoro

            logger.info("Loading Kokoro TTS model from %s …", model)
            engine = Kokoro(str(model), str(voices))
            # Touch once so first user request isn't the cold phonemizer hit.
            engine.create("Warm.", voice=voice_for_agent(settings, "lumen"), speed=1.0, lang="en-us")
            _ENGINE = engine
            _ENGINE_PATHS = (model, voices)
            _WARM_ERROR = None
            logger.info("Kokoro TTS warm and ready")
            return True
        except Exception as exc:  # noqa: BLE001 — surface in status
            _ENGINE = None
            _ENGINE_PATHS = None
            _WARM_ERROR = str(exc)
            logger.exception("Kokoro TTS failed to load")
            return False


def shutdown_tts() -> None:
    global _ENGINE, _ENGINE_PATHS, _WARM_ERROR
    with _LOCK:
        _ENGINE = None
        _ENGINE_PATHS = None
        _WARM_ERROR = None


def synthesize_wav(
    settings: Settings,
    *,
    text: str,
    agent_id: str,
    voice: str | None = None,
) -> bytes:
    """Return WAV bytes for speakable text. Raises KokoroTtsError on failure."""
    speak = text_for_speech(text, max_chars=settings.tts_max_chars)
    if not speak:
        raise KokoroTtsError("Nothing to speak")

    if not settings.tts_enabled:
        raise KokoroTtsError("TTS is disabled")

    chosen = (voice or "").strip() or voice_for_agent(settings, agent_id)
    speed = float(settings.tts_speed)

    with _LOCK:
        if _ENGINE is None:
            # Lazy warm if lifespan skipped or first call after enable.
            pass
    if _ENGINE is None:
        if not warm_tts(settings):
            raise KokoroTtsError(_WARM_ERROR or "TTS engine not ready")

    with _LOCK:
        engine = _ENGINE
        if engine is None:
            raise KokoroTtsError("TTS engine not ready")
        try:
            samples, sample_rate = engine.create(
                speak,
                voice=chosen,
                speed=speed,
                lang="en-us",
            )
        except Exception as exc:  # noqa: BLE001
            logger.exception("Kokoro synthesize failed voice=%s", chosen)
            raise KokoroTtsError(f"Synthesis failed: {exc}") from exc

    buf = io.BytesIO()
    sf.write(buf, samples, sample_rate, format="WAV")
    return buf.getvalue()
