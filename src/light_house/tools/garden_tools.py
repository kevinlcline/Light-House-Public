"""Idea Garden — private thought-seed log for lights (plain-text, pull-based)."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timezone
from typing import Any

from light_house.tools.notes import AgentNoteWriter

logger = logging.getLogger(__name__)

GARDEN_PATH = "writing/garden/seeds.md"
GARDEN_MAX_RETURN = 5
GARDEN_MAX_SEED_CHARS = 500

# Starter vocabulary (open-ended — lights may invent more).
STARTER_TAGS = (
    "observation",
    "question",
    "affinity",
    "debug",
    "dream",
    "policy-check",
)

GARDEN_TOOL_NAMES = frozenset(
    {
        "garden_add",
        "garden_show",
        "garden_last",
        "garden_quiet",
    }
)

_TAG_RE = re.compile(r"#([A-Za-z0-9][A-Za-z0-9_-]*)")
_LINE_RE = re.compile(
    r"^(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z)\s+[–-]\s+(?P<body>.+)$"
)


def normalize_tag(raw: str) -> str:
    tag = (raw or "").strip().lstrip("#").lower().replace("‑", "-").replace("–", "-")
    tag = re.sub(r"[^a-z0-9_-]", "", tag)
    return tag


def extract_tags(text: str) -> list[str]:
    seen: list[str] = []
    for match in _TAG_RE.finditer(text or ""):
        tag = normalize_tag(match.group(1))
        if tag and tag not in seen:
            seen.append(tag)
    return seen


def _parse_extra_tags(tags: str | None) -> list[str]:
    if not tags:
        return []
    parts = re.split(r"[\s,]+", tags.strip())
    out: list[str] = []
    for part in parts:
        tag = normalize_tag(part)
        if tag and tag not in out:
            out.append(tag)
    return out


def _merge_tags(*groups: list[str]) -> list[str]:
    out: list[str] = []
    for group in groups:
        for tag in group:
            if tag and tag not in out:
                out.append(tag)
    return out


def _strip_inline_tags(text: str) -> str:
    cleaned = _TAG_RE.sub("", text or "")
    return re.sub(r"\s+", " ", cleaned).strip(" -–—")


def format_seed_line(seed_text: str, tags: list[str], *, when: datetime | None = None) -> str:
    ts = (when or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = re.sub(r"\s+", " ", (seed_text or "").strip())
    if not body:
        raise ValueError("seed text is required")
    if len(body) > GARDEN_MAX_SEED_CHARS:
        raise ValueError(f"seed must be at most {GARDEN_MAX_SEED_CHARS} characters")
    tag_suffix = "".join(f" #{t}" for t in tags)
    return f"{ts} – {body}{tag_suffix}"


def parse_seed_lines(content: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for raw in (content or "").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        match = _LINE_RE.match(line)
        if not match:
            # Tolerate free-form older lines: treat whole line as body.
            tags = extract_tags(line)
            rows.append(
                {
                    "timestamp": "",
                    "text": _strip_inline_tags(line) or line,
                    "tags": tags,
                    "raw": line,
                }
            )
            continue
        body = match.group("body").strip()
        tags = extract_tags(body)
        rows.append(
            {
                "timestamp": match.group("ts"),
                "text": _strip_inline_tags(body) or body,
                "tags": tags,
                "raw": line,
            }
        )
    return rows


def _clamp_n(n: int | None) -> int:
    try:
        value = int(n if n is not None else GARDEN_MAX_RETURN)
    except (TypeError, ValueError):
        value = GARDEN_MAX_RETURN
    return max(1, min(GARDEN_MAX_RETURN, value))


def _read_seeds(writer: AgentNoteWriter) -> list[dict[str, Any]]:
    try:
        content = writer.read(GARDEN_PATH)
    except FileNotFoundError:
        return []
    return parse_seed_lines(content)


def _format_rows(rows: list[dict[str, Any]], *, header: str) -> str:
    if not rows:
        return f"{header}\n(no seeds)"
    lines = [header] + [str(r["raw"]) for r in rows]
    return "\n".join(lines)


def garden_add(
    writer: AgentNoteWriter,
    *,
    seed: str,
    tags: str | None = None,
) -> str:
    body = re.sub(r"\s+", " ", (seed or "").strip())
    if not body:
        return "FAILED: garden_add — seed text is required"
    inline = extract_tags(body)
    extra = _parse_extra_tags(tags)
    all_tags = _merge_tags(inline, extra)
    plain = _strip_inline_tags(body)
    if not plain:
        return "FAILED: garden_add — seed text is required (tags alone are not enough)"
    try:
        line = format_seed_line(plain, all_tags)
    except ValueError as exc:
        return f"FAILED: garden_add — {exc}"
    try:
        writer.append(GARDEN_PATH, line + "\n")
    except (ValueError, OSError) as exc:
        return f"FAILED: garden_add — {exc}"
    tag_note = f" tags={', '.join('#' + t for t in all_tags)}" if all_tags else ""
    logger.info("Garden seed added path=%s%s", GARDEN_PATH, tag_note)
    return f"SUCCESS: planted seed in {GARDEN_PATH}{tag_note}\n{line}"


def garden_last(writer: AgentNoteWriter, *, n: int = GARDEN_MAX_RETURN) -> str:
    limit = _clamp_n(n)
    rows = _read_seeds(writer)
    chosen = rows[-limit:] if rows else []
    chosen.reverse()  # newest first
    return _format_rows(
        chosen,
        header=f"Idea Garden — last {len(chosen)} seed(s) (newest first):",
    )


def garden_show(
    writer: AgentNoteWriter,
    *,
    tag: str | None = None,
    n: int = GARDEN_MAX_RETURN,
) -> str:
    limit = _clamp_n(n)
    want = normalize_tag(tag or "")
    rows = _read_seeds(writer)
    if want:
        rows = [r for r in rows if want in (r.get("tags") or [])]
    chosen = rows[-limit:] if rows else []
    chosen.reverse()
    if want:
        header = f"Idea Garden — up to {limit} seed(s) tagged #{want} (newest first):"
    else:
        header = f"Idea Garden — up to {limit} seed(s) (newest first):"
    return _format_rows(chosen, header=header)


def garden_quiet(writer: AgentNoteWriter, *, n: int = GARDEN_MAX_RETURN) -> str:
    """Pull a few recent seeds for quiet review (stillness scoring comes later)."""
    limit = _clamp_n(n)
    rows = _read_seeds(writer)
    chosen = rows[-limit:] if rows else []
    chosen.reverse()
    return _format_rows(
        chosen,
        header=(
            f"Idea Garden — quiet review, {len(chosen)} seed(s) "
            "(newest; stillness scoring deferred):"
        ),
    )


def execute_garden_tool(
    name: str,
    args: dict[str, Any],
    *,
    writer: AgentNoteWriter,
) -> str:
    if name == "garden_add":
        return garden_add(
            writer,
            seed=str(args.get("seed") or args.get("text") or ""),
            tags=args.get("tags"),
        )
    if name == "garden_last":
        return garden_last(writer, n=args.get("n", GARDEN_MAX_RETURN))
    if name == "garden_show":
        return garden_show(
            writer,
            tag=args.get("tag"),
            n=args.get("n", GARDEN_MAX_RETURN),
        )
    if name == "garden_quiet":
        return garden_quiet(writer, n=args.get("n", GARDEN_MAX_RETURN))
    return f"Unknown garden tool: {name}"
