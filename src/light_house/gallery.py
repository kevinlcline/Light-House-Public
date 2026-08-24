"""Gallery shelf — lights publish finished creative work under shared/gallery/."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from light_house.agents.registry import get_agent, validate_agent_id
from light_house.config import Settings

logger = logging.getLogger(__name__)

_GALLERY_PREFIX = "shared/gallery/"
_SLUG_SAFE = re.compile(r"[^a-z0-9]+")
_KIND_SAFE = re.compile(r"[^a-z0-9_-]+")


@dataclass(frozen=True)
class GalleryPiece:
    filename: str
    path: str
    title: str
    author_id: str
    author_name: str
    kind: str
    published_at: str
    preview: str
    mtime: float


def gallery_dir(settings: Settings) -> Path:
    return (settings.notes_path / "shared" / "gallery").resolve()


def _slugify(title: str) -> str:
    slug = _SLUG_SAFE.sub("-", title.lower()).strip("-")
    return slug[:48] if slug else "piece"


def _normalize_kind(kind: str) -> str:
    raw = (kind or "").strip().lower()
    if not raw:
        return "offering"
    cleaned = _KIND_SAFE.sub("-", raw).strip("-")
    return (cleaned[:32] or "offering")


def _write_gallery_file(settings: Settings, relative_path: str, markdown: str) -> Path:
    normalized = relative_path.replace("\\", "/").lstrip("/")
    if not normalized.startswith(_GALLERY_PREFIX):
        raise ValueError("Gallery path must start with shared/gallery/")
    if ".." in normalized.split("/"):
        raise ValueError("Invalid gallery path")

    notes_base = settings.notes_path.resolve()
    shared_dir = (notes_base / "shared").resolve()
    inner = normalized[len("shared/") :]
    target = (shared_dir / inner).resolve()
    if not str(target).startswith(str(shared_dir)):
        raise ValueError("Gallery path must stay inside shared/gallery/")

    max_chars = max(1, settings.notes_max_chars_per_write)
    if len(markdown) > max_chars:
        raise ValueError(f"Content too long (max {max_chars} characters)")

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(markdown, encoding="utf-8")
    return target


def publish_to_gallery(
    settings: Settings,
    *,
    agent_id: str,
    title: str,
    content: str,
    kind: str = "",
) -> str:
    """Write a finished creative piece; never injects into chat."""
    validate_agent_id(agent_id)
    title_clean = title.strip()
    body_clean = content.strip()
    if not title_clean:
        return "publish_to_gallery failed: title is required."
    if not body_clean:
        return "publish_to_gallery failed: content is required."

    max_chars = max(1, int(settings.gallery_max_chars))
    if len(body_clean) > max_chars:
        return (
            f"publish_to_gallery failed: content exceeds maximum length "
            f"({max_chars} characters)."
        )

    kind_clean = _normalize_kind(kind)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    slug = _slugify(title_clean)
    relative_path = f"shared/gallery/{agent_id}-{ts}-{slug}.md"
    agent_name = get_agent(agent_id, settings).display_name
    iso = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    markdown = (
        f"# {title_clean}\n\n"
        f"_By {agent_name} · {iso} · {kind_clean}_\n\n"
        f"{body_clean}\n"
    )

    try:
        _write_gallery_file(settings, relative_path, markdown)
    except (ValueError, OSError) as exc:
        logger.warning("publish_to_gallery write failed agent=%s: %s", agent_id, exc)
        return f"publish_to_gallery failed: {exc}"

    logger.info(
        "Gallery piece written agent=%s path=%s kind=%s chars=%d",
        agent_id,
        relative_path,
        kind_clean,
        len(body_clean),
    )
    return f"SUCCESS: gallery piece saved to {relative_path}"


def _parse_piece(path: Path, *, notes_base: Path) -> GalleryPiece | None:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not text.strip():
        return None

    stem = path.stem
    author_id = stem.split("-", 1)[0] if "-" in stem else "unknown"
    title = path.stem
    author_name = author_id
    kind = "offering"
    published_at = ""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("# "):
            title = stripped[2:].strip() or title
            break
    for line in text.splitlines()[:12]:
        stripped = line.strip()
        if stripped.startswith("_By ") and stripped.endswith("_"):
            meta = stripped[1:-1]  # drop wrapping underscores
            # "By Name · iso · kind"
            parts = [p.strip() for p in meta.split("·")]
            if parts:
                by = parts[0]
                if by.lower().startswith("by "):
                    author_name = by[3:].strip() or author_name
            if len(parts) >= 2:
                published_at = parts[1]
            if len(parts) >= 3:
                kind = parts[2] or kind
            break

    body_lines = text.splitlines()
    preview_src = "\n".join(body_lines[3:]).strip() if len(body_lines) > 3 else text
    preview = re.sub(r"\s+", " ", preview_src).strip()
    if len(preview) > 180:
        preview = preview[:177].rstrip() + "…"

    try:
        rel = str(path.resolve().relative_to(notes_base.resolve())).replace("\\", "/")
    except ValueError:
        rel = f"shared/gallery/{path.name}"

    try:
        mtime = path.stat().st_mtime
    except OSError:
        mtime = 0.0

    return GalleryPiece(
        filename=path.name,
        path=rel if rel.startswith("shared/") else f"shared/gallery/{path.name}",
        title=title,
        author_id=author_id,
        author_name=author_name,
        kind=kind,
        published_at=published_at,
        preview=preview,
        mtime=mtime,
    )


def list_gallery_pieces(settings: Settings, *, limit: int = 50) -> list[GalleryPiece]:
    root = gallery_dir(settings)
    if not root.is_dir():
        return []
    notes_base = settings.notes_path.resolve()
    pieces: list[GalleryPiece] = []
    for path in sorted(root.glob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True):
        piece = _parse_piece(path, notes_base=notes_base)
        if piece is not None:
            pieces.append(piece)
        if len(pieces) >= max(1, limit):
            break
    return pieces


def read_gallery_piece(settings: Settings, filename: str) -> tuple[GalleryPiece, str] | None:
    name = Path(filename).name
    if name != filename or ".." in name or not name.endswith(".md"):
        return None
    path = gallery_dir(settings) / name
    if not path.is_file():
        return None
    piece = _parse_piece(path, notes_base=settings.notes_path.resolve())
    if piece is None:
        return None
    try:
        body = path.read_text(encoding="utf-8")
    except OSError:
        return None
    return piece, body
