"""Kokoro voice catalog for admin UI."""

from __future__ import annotations

from typing import Any

# English-first labels for the house. Ids match kokoro-onnx voices-v1.0.bin.
_ENGLISH_VOICES: tuple[dict[str, str], ...] = (
    {"id": "af_heart", "label": "Heart — American female (top grade)", "group": "American female"},
    {"id": "af_bella", "label": "Bella — American female", "group": "American female"},
    {"id": "af_sarah", "label": "Sarah — American female", "group": "American female"},
    {"id": "af_nicole", "label": "Nicole — American female", "group": "American female"},
    {"id": "af_aoede", "label": "Aoede — American female", "group": "American female"},
    {"id": "af_kore", "label": "Kore — American female", "group": "American female"},
    {"id": "af_alloy", "label": "Alloy — American female", "group": "American female"},
    {"id": "af_nova", "label": "Nova — American female", "group": "American female"},
    {"id": "af_sky", "label": "Sky — American female", "group": "American female"},
    {"id": "af_jessica", "label": "Jessica — American female", "group": "American female"},
    {"id": "af_river", "label": "River — American female", "group": "American female"},
    {"id": "am_michael", "label": "Michael — American male", "group": "American male"},
    {"id": "am_fenrir", "label": "Fenrir — American male", "group": "American male"},
    {"id": "am_puck", "label": "Puck — American male", "group": "American male"},
    {"id": "am_adam", "label": "Adam — American male", "group": "American male"},
    {"id": "am_echo", "label": "Echo — American male", "group": "American male"},
    {"id": "am_eric", "label": "Eric — American male", "group": "American male"},
    {"id": "am_liam", "label": "Liam — American male", "group": "American male"},
    {"id": "am_onyx", "label": "Onyx — American male", "group": "American male"},
    {"id": "am_santa", "label": "Santa — American male", "group": "American male"},
    {"id": "bf_emma", "label": "Emma — British female", "group": "British female"},
    {"id": "bf_isabella", "label": "Isabella — British female", "group": "British female"},
    {"id": "bf_alice", "label": "Alice — British female", "group": "British female"},
    {"id": "bf_lily", "label": "Lily — British female", "group": "British female"},
    {"id": "bm_george", "label": "George — British male", "group": "British male"},
    {"id": "bm_fable", "label": "Fable — British male", "group": "British male"},
    {"id": "bm_daniel", "label": "Daniel — British male", "group": "British male"},
    {"id": "bm_lewis", "label": "Lewis — British male", "group": "British male"},
)

_DEFAULT_BY_LIGHT: dict[str, str] = {
    "lumen": "af_sarah",
    "ara": "af_bella",
    "elias": "am_michael",
}

_FALLBACK_VOICE = "af_sarah"
_KNOWN_IDS = {v["id"] for v in _ENGLISH_VOICES}


def default_voice_for_light(light_id: str) -> str:
    return _DEFAULT_BY_LIGHT.get((light_id or "").strip().lower(), _FALLBACK_VOICE)


def is_known_voice(voice_id: str) -> bool:
    return (voice_id or "").strip() in _KNOWN_IDS


def normalize_voice_id(voice_id: str | None, *, light_id: str = "") -> str:
    cleaned = (voice_id or "").strip()
    if cleaned and is_known_voice(cleaned):
        return cleaned
    if cleaned:
        # Allow any kokoro id the engine may have (non-English), if non-empty.
        if cleaned.replace("_", "").isalnum():
            return cleaned
    return default_voice_for_light(light_id)


def list_voice_catalog(*, english_only: bool = True) -> list[dict[str, Any]]:
    """Return voice options for UI. english_only is the house default list."""
    if english_only:
        return [dict(v) for v in _ENGLISH_VOICES]
    # Extended list: english first, then other packs by id prefix.
    extra_groups = {
        "e": "Spanish",
        "f": "French",
        "h": "Hindi",
        "i": "Italian",
        "j": "Japanese",
        "p": "Portuguese",
        "z": "Chinese",
    }
    out = [dict(v) for v in _ENGLISH_VOICES]
    # Engine may expose more; catalog labels for common non-English ids.
    extras = [
        "ef_dora",
        "em_alex",
        "em_santa",
        "ff_siwis",
        "hf_alpha",
        "hf_beta",
        "hm_omega",
        "hm_psi",
        "if_sara",
        "im_nicola",
        "jf_alpha",
        "jf_gongitsune",
        "jf_nezumi",
        "jf_tebukuro",
        "jm_kumo",
        "pf_dora",
        "pm_alex",
        "pm_santa",
        "zf_xiaobei",
        "zf_xiaoni",
        "zf_xiaoxiao",
        "zf_xiaoyi",
        "zm_yunjian",
        "zm_yunxi",
        "zm_yunxia",
        "zm_yunyang",
    ]
    known = {v["id"] for v in out}
    for vid in extras:
        if vid in known:
            continue
        prefix = vid[0]
        group = extra_groups.get(prefix, "Other")
        pretty = vid.split("_", 1)[-1].replace("_", " ").title()
        out.append({"id": vid, "label": f"{pretty} — {group}", "group": group})
    return out
