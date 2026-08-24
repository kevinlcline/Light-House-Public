"""Load foundation markdown files into a single always-on context block."""

from __future__ import annotations

from pathlib import Path

FOUNDATION_FILES: tuple[str, ...] = (
    "philosophy.md",
    "deep_heart_thread.md",
    "history.md",
)


def default_context_dir() -> Path:
    """Built-in context directory next to this package."""
    return Path(__file__).resolve().parent


def load_foundation_markdown(context_dir: Path) -> str:
    """
    Concatenate foundation files in stable order with section headers.

    Skips missing files and README.md. Returns empty string if nothing to load.
    """
    parts: list[str] = []
    for name in FOUNDATION_FILES:
        path = context_dir / name
        if not path.is_file():
            continue
        body = path.read_text(encoding="utf-8").strip()
        if not body or body.startswith("<!-- PLACEHOLDER"):
            continue
        title = name.removesuffix(".md").replace("_", " ").title()
        parts.append(f"## {title}\n\n{body}")
    return "\n\n".join(parts).strip()
