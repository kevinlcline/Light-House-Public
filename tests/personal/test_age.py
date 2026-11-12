"""Coarse wall-clock ages for the desk."""

from __future__ import annotations

from light_house.personal.age import humanize_age

_NOW = 1_700_000_000.0
_DAY = 86400


def test_humanize_age_uses_the_coarsest_true_unit() -> None:
    assert humanize_age(_NOW, now=_NOW) == "0m"
    assert humanize_age(_NOW - 59, now=_NOW) == "0m"
    assert humanize_age(_NOW - 60, now=_NOW) == "1m"
    assert humanize_age(_NOW - 3600, now=_NOW) == "1h"
    assert humanize_age(_NOW - 23 * 3600, now=_NOW) == "23h"
    assert humanize_age(_NOW - 2 * _DAY, now=_NOW) == "2d"
    assert humanize_age(_NOW - 6 * _DAY, now=_NOW) == "6d"
    assert humanize_age(_NOW - 7 * _DAY, now=_NOW) == "1w"
    assert humanize_age(_NOW - 21 * _DAY, now=_NOW) == "3w"
    assert humanize_age(_NOW - 29 * _DAY, now=_NOW) == "4w"
    assert humanize_age(_NOW - 30 * _DAY, now=_NOW) == "1mo"
    assert humanize_age(_NOW - 5 * 30 * _DAY, now=_NOW) == "5mo"
    assert humanize_age(_NOW - 365 * _DAY, now=_NOW) == "1y"
    assert humanize_age(_NOW + 500, now=_NOW) == "0m"
