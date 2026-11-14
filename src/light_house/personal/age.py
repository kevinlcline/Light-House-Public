"""Coarse wall-clock age for records a light reads as present."""

from __future__ import annotations

import time

_MINUTE = 60
_HOUR = 60 * _MINUTE
_DAY = 24 * _HOUR
_WEEK = 7 * _DAY
_MONTH = 30 * _DAY
_YEAR = 365 * _DAY

# Coarsest unit whose count is still at least one. ``m`` is minutes; months are ``mo``.
_LADDER: tuple[tuple[int, str], ...] = (
    (_YEAR, "y"),
    (_MONTH, "mo"),
    (_WEEK, "w"),
    (_DAY, "d"),
    (_HOUR, "h"),
    (_MINUTE, "m"),
)


def humanize_age(ts: float, *, now: float | None = None) -> str:
    """Wall-clock age as a short unit: ``3w``, ``2d``, ``5mo``.

    Felt days and felt cycles stay on the inner-time block. A future stamp
    reads as ``0m``.
    """
    current = time.time() if now is None else now
    delta = current - ts
    if delta < _MINUTE:
        return "0m"
    for size, label in _LADDER:
        count = int(delta // size)
        if count >= 1:
            return f"{count}{label}"
    return "0m"
