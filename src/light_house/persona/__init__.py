"""Persona assets (system prompt core, future tone packs)."""

from importlib import resources
from pathlib import Path


def load_lumen_system_prompt() -> str:
    """
    Load the canonical Lumen system prompt from ``lumen_system.md``.

    **Resolution order** (so refined markdown on disk is picked up reliably in dev):

    1. Filesystem path next to this module (always re-reads from disk; supports hot edits).
    2. Packaged resource via ``importlib.resources`` (installed wheel / zip-safe layouts).
    """
    here = Path(__file__).resolve().parent / "lumen_system.md"
    if here.is_file():
        return here.read_text(encoding="utf-8").strip()

    try:
        text = resources.files(__package__).joinpath("lumen_system.md").read_text(encoding="utf-8")
        if text.strip():
            return text.strip()
    except (TypeError, FileNotFoundError, OSError):
        pass

    raise FileNotFoundError(f"lumen_system.md not found at {here} and not in package resources")
