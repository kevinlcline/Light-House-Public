"""Shared helpers for long-term memory stores."""

from __future__ import annotations

import hashlib


def turn_dedup_key(user_text: str, assistant_text: str) -> str:
    raw = f"{user_text.strip()}\n{assistant_text.strip()}".encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def body_dedup_key(body: str) -> str:
    return hashlib.sha256(body.strip().encode("utf-8")).hexdigest()
