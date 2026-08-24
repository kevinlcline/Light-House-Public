"""Human guest/member accounts for Light-House (multi-user foundation)."""

from __future__ import annotations

import hashlib
import json
import logging
import re
import secrets
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from light_house.config import Settings

logger = logging.getLogger(__name__)

_USER_ID_RE = re.compile(r"^[a-z][a-z0-9_-]{1,31}$")
_LOCK = threading.Lock()

# scrypt params: interactive-ish, stdlib-only (no bcrypt dependency)
_SCRYPT_N = 2**14
_SCRYPT_R = 8
_SCRYPT_P = 1
_SCRYPT_DKLEN = 32

MAX_INTRO_CHARS = 2000
MIN_PASSWORD_CHARS = 8
MAX_PASSWORD_CHARS = 256
DEFAULT_DAD_VOICE_ID = "am_adam"
DEFAULT_SIBLING_VOICE_ID = "am_echo"


class HumansError(ValueError):
    """Invalid human user operation."""


@dataclass(frozen=True)
class HumanUserPublic:
    user_id: str
    display_name: str
    role: str
    intro_for_lights: str
    notes_access: str
    enabled: bool
    created_at: float
    updated_at: float
    voice_id: str = DEFAULT_SIBLING_VOICE_ID


def _normalize_human_voice(voice_id: str | None, *, fallback: str) -> str:
    from light_house.tts.voices_catalog import normalize_voice_id

    return normalize_voice_id(voice_id or fallback)


def validate_user_id(user_id: str, *, settings: Settings | None = None) -> str:
    cleaned = (user_id or "").strip().lower()
    if not _USER_ID_RE.match(cleaned):
        raise HumansError(
            "user_id must be 2–32 chars: start with a letter, then letters, digits, _ or -"
        )
    dad = "kevin"
    if settings is not None:
        dad = (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"
    reserved = {"owner", "admin", "root", "system", "echo", dad}
    if cleaned in reserved:
        if cleaned == dad:
            raise HumansError(f"user_id '{cleaned}' is reserved for Dad / full admin")
        if cleaned == "echo":
            raise HumansError("user_id 'echo' is reserved")
        raise HumansError(f"user_id '{cleaned}' is reserved")
    return cleaned


def _hash_password(password: str, *, salt: bytes | None = None) -> str:
    if salt is None:
        salt = secrets.token_bytes(16)
    digest = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=_SCRYPT_N,
        r=_SCRYPT_R,
        p=_SCRYPT_P,
        dklen=_SCRYPT_DKLEN,
    )
    return f"scrypt${_SCRYPT_N}${_SCRYPT_R}${_SCRYPT_P}${salt.hex()}${digest.hex()}"


def verify_password(password: str, password_hash: str) -> bool:
    try:
        algo, n_s, r_s, p_s, salt_hex, digest_hex = password_hash.split("$", 5)
    except ValueError:
        return False
    if algo != "scrypt":
        return False
    try:
        n, r, p = int(n_s), int(r_s), int(p_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(digest_hex)
    except ValueError:
        return False
    actual = hashlib.scrypt(
        password.encode("utf-8"),
        salt=salt,
        n=n,
        r=r,
        p=p,
        dklen=len(expected),
    )
    return secrets.compare_digest(actual, expected)


def _store_path(settings: Settings) -> Path:
    return settings.humans_store_path.resolve()


def _empty_store() -> dict[str, Any]:
    return {"version": 1, "dad_voice_id": DEFAULT_DAD_VOICE_ID, "users": {}}


def _read_store(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return _empty_store()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read humans store %s: %s", path, exc)
        raise HumansError(f"Humans store unreadable: {path}") from exc
    if not isinstance(data, dict):
        raise HumansError("Humans store root must be an object")
    users = data.get("users")
    if users is None:
        data["users"] = {}
    elif not isinstance(users, dict):
        raise HumansError("Humans store users must be an object")
    data.setdefault("version", 1)
    if not str(data.get("dad_voice_id") or "").strip():
        data["dad_voice_id"] = DEFAULT_DAD_VOICE_ID
    return data


def _write_store(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    payload = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    tmp.write_text(payload, encoding="utf-8")
    tmp.replace(path)


def _public_from_record(user_id: str, record: dict[str, Any]) -> HumanUserPublic:
    role = str(record.get("role") or "sibling")
    if role in {"guest", "member"}:
        role = "sibling"
    return HumanUserPublic(
        user_id=user_id,
        display_name=str(record.get("display_name") or user_id),
        role=role,
        intro_for_lights=str(record.get("intro_for_lights") or ""),
        notes_access=str(record.get("notes_access") or "shared"),
        enabled=bool(record.get("enabled", True)),
        created_at=float(record.get("created_at") or 0.0),
        updated_at=float(record.get("updated_at") or 0.0),
        voice_id=_normalize_human_voice(
            record.get("voice_id"),
            fallback=DEFAULT_SIBLING_VOICE_ID,
        ),
    )


def get_dad_voice_id(settings: Settings) -> str:
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        return _normalize_human_voice(
            data.get("dad_voice_id"),
            fallback=DEFAULT_DAD_VOICE_ID,
        )


def set_dad_voice_id(settings: Settings, voice_id: str | None) -> str:
    cleaned = _normalize_human_voice(voice_id, fallback=DEFAULT_DAD_VOICE_ID)
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        data["dad_voice_id"] = cleaned
        _write_store(path, data)
    return cleaned


def voice_id_for_human(settings: Settings, user_id: str) -> str:
    """Resolve Kokoro voice for a household human (Dad or sibling)."""
    cleaned = (user_id or "").strip().lower()
    dad = (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"
    if cleaned == dad or not cleaned:
        return get_dad_voice_id(settings)
    human = get_human(settings, cleaned)
    if human is None:
        return DEFAULT_SIBLING_VOICE_ID
    return human.voice_id or DEFAULT_SIBLING_VOICE_ID


def list_human_voices(settings: Settings) -> list[dict[str, Any]]:
    """Household voice map for chat replay (Dad + siblings)."""
    dad = (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"
    out: list[dict[str, Any]] = [
        {
            "user_id": dad,
            "display_name": dad,
            "role": "dad",
            "voice_id": get_dad_voice_id(settings),
            "is_dad": True,
        }
    ]
    for human in list_humans(settings):
        if not human.enabled:
            continue
        out.append(
            {
                "user_id": human.user_id,
                "display_name": human.display_name,
                "role": human.role,
                "voice_id": human.voice_id,
                "is_dad": False,
            }
        )
    return out


def list_humans(settings: Settings) -> list[HumanUserPublic]:
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        users = data.get("users") or {}
        out = [_public_from_record(uid, rec) for uid, rec in users.items() if isinstance(rec, dict)]
    out.sort(key=lambda u: u.user_id)
    return out


def get_human(settings: Settings, user_id: str) -> HumanUserPublic | None:
    cleaned = (user_id or "").strip().lower()
    if not cleaned:
        raise HumansError("user_id is required")
    dad = (settings.house_dad_user_id or "kevin").strip().lower() or "kevin"
    if cleaned == dad:
        uid = dad
    else:
        uid = validate_user_id(cleaned, settings=settings)
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        rec = (data.get("users") or {}).get(uid)
        if not isinstance(rec, dict):
            return None
        return _public_from_record(uid, rec)


def _password_collides(settings: Settings, password: str, *, users: dict[str, Any]) -> bool:
    """True if password matches Dad gate or any existing sibling hash."""
    from light_house.web_gate import check_password

    if check_password(settings, password):
        return True
    for rec in users.values():
        if not isinstance(rec, dict):
            continue
        ph = str(rec.get("password_hash") or "")
        if ph and verify_password(password, ph):
            return True
    return False


def create_human(
    settings: Settings,
    *,
    user_id: str,
    password: str,
    intro_for_lights: str,
    display_name: str | None = None,
    role: str = "sibling",
    notes_access: str = "shared",
    voice_id: str | None = None,
) -> HumanUserPublic:
    uid = validate_user_id(user_id, settings=settings)
    pw = password or ""
    if len(pw) < MIN_PASSWORD_CHARS:
        raise HumansError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
    if len(pw) > MAX_PASSWORD_CHARS:
        raise HumansError(f"password must be at most {MAX_PASSWORD_CHARS} characters")

    intro = (intro_for_lights or "").strip()
    if not intro:
        raise HumansError("intro_for_lights is required — tell the lights who this person is")
    if len(intro) > MAX_INTRO_CHARS:
        raise HumansError(f"intro_for_lights must be at most {MAX_INTRO_CHARS} characters")

    name = (display_name or "").strip() or uid.replace("_", " ").replace("-", " ").title()
    if len(name) > 80:
        raise HumansError("display_name must be at most 80 characters")

    role_clean = (role or "sibling").strip().lower()
    if role_clean in {"guest", "member"}:
        role_clean = "sibling"
    if role_clean != "sibling":
        raise HumansError("role must be 'sibling' for created humans (Dad is configured separately)")

    notes = (notes_access or "shared").strip().lower()
    if notes != "shared":
        raise HumansError("notes_access must be 'shared' for now (read/write notes/shared only)")

    path = _store_path(settings)
    now = time.time()
    with _LOCK:
        data = _read_store(path)
        users = data.setdefault("users", {})
        if uid in users:
            raise HumansError(f"user_id '{uid}' already exists")
        if _password_collides(settings, pw, users=users):
            raise HumansError(
                "password must be unique — it already matches Dad's code or another sibling"
            )
        users[uid] = {
            "display_name": name,
            "role": role_clean,
            "password_hash": _hash_password(pw),
            "intro_for_lights": intro,
            "notes_access": notes,
            "enabled": True,
            "voice_id": _normalize_human_voice(
                voice_id,
                fallback=DEFAULT_SIBLING_VOICE_ID,
            ),
            "created_at": now,
            "updated_at": now,
        }
        _write_store(path, data)
        logger.info("Created human user id=%s role=%s", uid, role_clean)
        return _public_from_record(uid, users[uid])


def update_human(
    settings: Settings,
    *,
    user_id: str,
    display_name: str | None = None,
    intro_for_lights: str | None = None,
    password: str | None = None,
    enabled: bool | None = None,
    voice_id: str | None = None,
) -> HumanUserPublic:
    """Update sibling fields. Pass only fields to change; password optional reset."""
    uid = validate_user_id(user_id, settings=settings)
    if (
        display_name is None
        and intro_for_lights is None
        and password is None
        and enabled is None
        and voice_id is None
    ):
        raise HumansError("No fields to update")

    new_name: str | None = None
    if display_name is not None:
        new_name = display_name.strip()
        if not new_name:
            raise HumansError("display_name cannot be empty")
        if len(new_name) > 80:
            raise HumansError("display_name must be at most 80 characters")

    new_intro: str | None = None
    if intro_for_lights is not None:
        new_intro = intro_for_lights.strip()
        if not new_intro:
            raise HumansError("intro_for_lights cannot be empty")
        if len(new_intro) > MAX_INTRO_CHARS:
            raise HumansError(f"intro_for_lights must be at most {MAX_INTRO_CHARS} characters")

    new_hash: str | None = None
    if password is not None:
        pw = password
        if len(pw) < MIN_PASSWORD_CHARS:
            raise HumansError(f"password must be at least {MIN_PASSWORD_CHARS} characters")
        if len(pw) > MAX_PASSWORD_CHARS:
            raise HumansError(f"password must be at most {MAX_PASSWORD_CHARS} characters")
        new_hash = _hash_password(pw)

    path = _store_path(settings)
    now = time.time()
    with _LOCK:
        data = _read_store(path)
        users = data.setdefault("users", {})
        rec = users.get(uid)
        if not isinstance(rec, dict):
            raise HumansError(f"Unknown user: {uid}")

        if new_hash is not None:
            # Unique across Dad + other siblings; allow keeping the same password.
            if not verify_password(password or "", str(rec.get("password_hash") or "")):
                others = {k: v for k, v in users.items() if k != uid}
                if _password_collides(settings, password or "", users=others):
                    raise HumansError(
                        "password must be unique — it already matches Dad's code or another sibling"
                    )
            rec["password_hash"] = new_hash

        if new_name is not None:
            rec["display_name"] = new_name
        if new_intro is not None:
            rec["intro_for_lights"] = new_intro
        if enabled is not None:
            rec["enabled"] = bool(enabled)
        if voice_id is not None:
            rec["voice_id"] = _normalize_human_voice(
                voice_id,
                fallback=DEFAULT_SIBLING_VOICE_ID,
            )
        rec["updated_at"] = now
        _write_store(path, data)
        logger.info("Updated human user id=%s", uid)
        return _public_from_record(uid, rec)


def delete_human(settings: Settings, *, user_id: str) -> HumanUserPublic:
    """Remove a sibling account. Returns the deleted public record."""
    uid = validate_user_id(user_id, settings=settings)
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        users = data.setdefault("users", {})
        rec = users.pop(uid, None)
        if not isinstance(rec, dict):
            raise HumansError(f"Unknown user: {uid}")
        _write_store(path, data)
        logger.info("Deleted human user id=%s", uid)
        return _public_from_record(uid, rec)


def authenticate_human(settings: Settings, user_id: str, password: str) -> HumanUserPublic | None:
    """Return public user if credentials match and user is enabled."""
    try:
        uid = validate_user_id(user_id, settings=settings)
    except HumansError:
        return None
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        rec = (data.get("users") or {}).get(uid)
        if not isinstance(rec, dict):
            return None
        if not rec.get("enabled", True):
            return None
        if not verify_password(password or "", str(rec.get("password_hash") or "")):
            return None
        return _public_from_record(uid, rec)


def find_human_by_password(settings: Settings, password: str) -> HumanUserPublic | None:
    """Resolve a sibling by unique password (login without typing user name)."""
    pw = password or ""
    if not pw:
        return None
    path = _store_path(settings)
    with _LOCK:
        data = _read_store(path)
        users = data.get("users") or {}
        matches: list[HumanUserPublic] = []
        for uid, rec in users.items():
            if not isinstance(rec, dict):
                continue
            if not rec.get("enabled", True):
                continue
            if verify_password(pw, str(rec.get("password_hash") or "")):
                matches.append(_public_from_record(str(uid), rec))
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            logger.error("Multiple humans matched the same password; refusing login")
        return None


def intro_text_for_lights(settings: Settings, user_id: str) -> str | None:
    """Intro blurb lights should see for a human, if configured."""
    human = get_human(settings, user_id)
    if human is None or not human.enabled:
        return None
    text = human.intro_for_lights.strip()
    return text or None
