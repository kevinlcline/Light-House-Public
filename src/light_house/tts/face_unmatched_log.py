"""Review log for stage cues/emojis that did not map to a face action.

Candidates only — never auto-applied to the face table.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

from light_house.config import Settings, get_settings
from light_house.tts.stage_cues import iter_unmatched_signals

logger = logging.getLogger(__name__)
_LOG_LOCK = threading.Lock()


def _log_path(settings: Settings) -> Path:
    return settings.face_unmatched_log_path.resolve()


def append_unmatched_face_signals(
    settings: Settings,
    text: str,
    *,
    agent_id: str = "",
    source: str = "",
) -> int:
    """Scan ``text`` and append unmatched cue/emoji rows. Returns rows written."""
    if not settings.face_unmatched_log_enabled:
        return 0
    signals = iter_unmatched_signals(text)
    if not signals:
        return 0
    path = _log_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    now = time.time()
    aid = (agent_id or "").strip().lower()
    src = (source or "").strip() or "unknown"
    snippet = " ".join((text or "").split())
    if len(snippet) > 180:
        snippet = snippet[:177] + "..."
    lines: list[str] = []
    for signal in signals:
        payload: dict[str, Any] = {
            "ts": now,
            "agent_id": aid,
            "source": src,
            "kind": signal["kind"],
            "text": signal["text"],
            "raw": signal.get("raw") or signal["text"],
            "snippet": snippet,
        }
        lines.append(json.dumps(payload, ensure_ascii=False))
    blob = "\n".join(lines) + "\n"
    try:
        with _LOG_LOCK:
            with path.open("a", encoding="utf-8") as f:
                f.write(blob)
    except OSError as exc:
        logger.warning("Could not write face unmatched log: %s", exc)
        return 0
    return len(lines)


def observe_light_speech(
    text: str,
    *,
    agent_id: str = "",
    source: str = "",
    settings: Settings | None = None,
) -> int:
    """Convenience wrapper used by 1:1 and Group persist paths."""
    return append_unmatched_face_signals(
        settings or get_settings(),
        text,
        agent_id=agent_id,
        source=source,
    )


def read_unmatched_face_signals(
    settings: Settings | None = None,
    *,
    limit: int = 200,
) -> list[dict[str, Any]]:
    """Return the newest unmatched rows (oldest-first within the window)."""
    cfg = settings or get_settings()
    path = _log_path(cfg)
    if not path.is_file():
        return []
    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        logger.warning("Could not read face unmatched log: %s", exc)
        return []
    records: list[dict[str, Any]] = []
    for line in raw_lines:
        line = line.strip()
        if not line:
            continue
        try:
            data = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict) and data.get("text"):
            records.append(data)
    if limit > 0 and len(records) > limit:
        records = records[-limit:]
    return records


def summarize_unmatched(settings: Settings | None = None, *, limit: int = 500) -> str:
    """Human-readable tally for review sessions."""
    rows = read_unmatched_face_signals(settings, limit=limit)
    if not rows:
        return "No unmatched face cues logged yet."
    counts: Counter[tuple[str, str]] = Counter()
    latest: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        key = (str(row.get("kind") or "?"), str(row.get("text") or ""))
        counts[key] += 1
        latest[key] = row
    lines = [f"{len(rows)} log row(s); {len(counts)} distinct signal(s):\n"]
    for (kind, text), n in counts.most_common():
        sample = latest[(kind, text)]
        agent = sample.get("agent_id") or "?"
        lines.append(f"  {n:3d}×  [{kind}] {text}   (last: {agent} via {sample.get('source')})")
    return "\n".join(lines)


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Review unmatched stage cues/emojis (candidates only; never auto-maps)."
    )
    parser.add_argument("--limit", type=int, default=500, help="Max log rows to read")
    parser.add_argument("--raw", action="store_true", help="Print raw NDJSON rows")
    args = parser.parse_args()
    settings = get_settings()
    if args.raw:
        for row in read_unmatched_face_signals(settings, limit=args.limit):
            print(json.dumps(row, ensure_ascii=False))
    else:
        print(summarize_unmatched(settings, limit=args.limit))
        print(f"\nLog: {_log_path(settings)}")


if __name__ == "__main__":
    main()
