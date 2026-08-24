"""Load Echo's subconscious dream persona."""

from importlib import resources
from pathlib import Path


def load_echo_persona() -> str:
    """
    Load Echo's system prompt from ``persona.md``.

    Resolution order matches Lumen's persona loader: filesystem first, then package resources.
    """
    here = Path(__file__).resolve().parent / "persona.md"
    if here.is_file():
        return here.read_text(encoding="utf-8").strip()

    try:
        text = resources.files(__package__).joinpath("persona.md").read_text(encoding="utf-8")
        if text.strip():
            return text.strip()
    except (TypeError, FileNotFoundError, OSError):
        pass

    raise FileNotFoundError(f"persona.md not found at {here} and not in package resources")
