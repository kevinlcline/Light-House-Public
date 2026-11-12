"""Pictures a host attaches to a chat or group prompt.

The file is stored under ``data/prompt_images``. The model sees it as an
image part on that turn. The transcript keeps a short id, not the bytes.
"""

from __future__ import annotations

import base64
import binascii
import re
import uuid
from dataclasses import dataclass
from typing import Literal

from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field, field_validator

from light_house.config import Settings

IMAGE_ONLY_TEXT = "See the attached image."
MAX_IMAGE_BYTES = 4_000_000
MAX_BASE64_CHARS = 6_000_000
MAX_IMAGES_ON_PROMPT = 4

MediaType = Literal["image/jpeg", "image/png", "image/webp", "image/gif"]

_IMAGE_ID = re.compile(r"^[a-f0-9]{32}$")
_EXT = {
    "image/jpeg": ".jpg",
    "image/png": ".png",
    "image/webp": ".webp",
    "image/gif": ".gif",
}


class PromptImageError(ValueError):
    """The attached picture is missing, too large, or not a supported type."""


class PromptImageIn(BaseModel):
    """One picture on a prompt. ``data_base64`` may be raw or a data URL."""

    media_type: MediaType
    data_base64: str = Field(..., min_length=16, max_length=MAX_BASE64_CHARS)

    @field_validator("data_base64", mode="before")
    @classmethod
    def unwrap_data_url(cls, value: object) -> object:
        if not isinstance(value, str):
            return value
        text = value.strip()
        if text.startswith("data:"):
            comma = text.find(",")
            if comma < 0:
                raise ValueError("image data URL is missing the payload")
            text = text[comma + 1 :]
        return "".join(text.split())


@dataclass(frozen=True)
class StoredPromptImage:
    image_id: str
    media_type: str
    data_base64: str

    @property
    def data_url(self) -> str:
        return f"data:{self.media_type};base64,{self.data_base64}"


def caption_for_prompt(text: str, *, has_image: bool) -> str:
    """Use the typed words, or a short line when the prompt is only a picture."""
    stripped = (text or "").strip()
    if stripped:
        return stripped
    if has_image:
        return IMAGE_ONLY_TEXT
    raise PromptImageError("message cannot be empty")


def prompt_image_url(image_id: str) -> str:
    return f"/v1/prompt-images/{image_id}"


def save_prompt_image(settings: Settings, image: PromptImageIn) -> str:
    """Write a validated picture and return its id."""
    raw, media_type = _decode(image)
    image_id = uuid.uuid4().hex
    directory = settings.prompt_image_dir
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{image_id}{_EXT[media_type]}"
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(raw)
    tmp.replace(path)
    return image_id


def load_prompt_image(settings: Settings, image_id: str) -> StoredPromptImage | None:
    path = _path_for(settings, image_id)
    if path is None or not path.is_file():
        return None
    media_type = _media_for_suffix(path.suffix.lower())
    if media_type is None:
        return None
    try:
        raw = path.read_bytes()
    except OSError:
        return None
    if not raw or len(raw) > MAX_IMAGE_BYTES:
        return None
    return StoredPromptImage(
        image_id=image_id,
        media_type=media_type,
        data_base64=base64.b64encode(raw).decode("ascii"),
    )


def read_prompt_image_bytes(settings: Settings, image_id: str) -> tuple[bytes, str] | None:
    stored = load_prompt_image(settings, image_id)
    if stored is None:
        return None
    try:
        raw = base64.b64decode(stored.data_base64, validate=True)
    except binascii.Error:
        return None
    return raw, stored.media_type


def image_ids_from_rows(rows: list[dict] | list) -> list[str]:
    """Image ids in order, keeping the most recent ones that fit on a prompt."""
    found: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        image_id = row.get("image_id")
        if isinstance(image_id, str) and _IMAGE_ID.fullmatch(image_id):
            found.append(image_id)
    if len(found) > MAX_IMAGES_ON_PROMPT:
        found = found[-MAX_IMAGES_ON_PROMPT:]
    return found


def human_message(text: str, settings: Settings, image_ids: list[str]) -> HumanMessage:
    """Text, plus picture parts for ids that still exist on disk."""
    parts: list[dict[str, object]] = [{"type": "text", "text": text}]
    attached = False
    for image_id in image_ids:
        stored = load_prompt_image(settings, image_id)
        if stored is None:
            continue
        parts.append(
            {
                "type": "image_url",
                "image_url": {"url": stored.data_url},
            }
        )
        attached = True
    if not attached:
        return HumanMessage(content=text)
    return HumanMessage(content=parts)


def message_text(content: object) -> str:
    """Plain words from a human message, including a vision content list."""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        chunks: list[str] = []
        for block in content:
            if isinstance(block, str):
                chunks.append(block)
            elif isinstance(block, dict) and block.get("type") == "text":
                chunks.append(str(block.get("text") or ""))
        return "\n".join(chunk.strip() for chunk in chunks if chunk and chunk.strip()).strip()
    return str(content or "").strip()


def transcript_line(name: str, text: str, *, image_id: str | None) -> str:
    line = f"{name}: {text}"
    if image_id:
        line += " [image attached]"
    return line


def _decode(image: PromptImageIn) -> tuple[bytes, str]:
    try:
        raw = base64.b64decode(image.data_base64, validate=True)
    except binascii.Error as exc:
        raise PromptImageError("image data is not valid base64") from exc
    if not raw:
        raise PromptImageError("image is empty")
    if len(raw) > MAX_IMAGE_BYTES:
        raise PromptImageError("image is too large (max 4 MB)")
    sniffed = _sniff_media_type(raw)
    if sniffed is None:
        raise PromptImageError("image must be JPEG, PNG, WebP, or GIF")
    if sniffed != image.media_type:
        raise PromptImageError("image type does not match the file")
    return raw, sniffed


def _sniff_media_type(raw: bytes) -> str | None:
    if raw.startswith(b"\xff\xd8\xff"):
        return "image/jpeg"
    if raw.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if raw.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if len(raw) >= 12 and raw.startswith(b"RIFF") and raw[8:12] == b"WEBP":
        return "image/webp"
    return None


def _media_for_suffix(suffix: str) -> str | None:
    for media_type, ext in _EXT.items():
        if ext == suffix:
            return media_type
    return None


def _path_for(settings: Settings, image_id: str):
    if not _IMAGE_ID.fullmatch(image_id or ""):
        return None
    directory = settings.prompt_image_dir.resolve()
    if not directory.is_dir():
        return None
    matches = []
    for path in directory.glob(f"{image_id}.*"):
        resolved = path.resolve()
        if (
            resolved.is_file()
            and resolved.parent == directory
            and resolved.suffix.lower() in _EXT.values()
        ):
            matches.append(resolved)
    if len(matches) != 1:
        return None
    return matches[0]
