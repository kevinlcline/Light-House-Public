"""Host pictures on a chat or group prompt."""

from __future__ import annotations

import base64

import pytest
from pydantic import ValidationError

from light_house.config import Settings
from light_house.main import ChatRequest, GroupUtterRequest
from light_house.prompt_image import (
    IMAGE_ONLY_TEXT,
    MAX_IMAGE_BYTES,
    PromptImageError,
    PromptImageIn,
    human_message,
    load_prompt_image,
    message_text,
    save_prompt_image,
    transcript_line,
)

JPEG = b"\xff\xd8\xff" + b"\x00" * 32
PNG = b"\x89PNG\r\n\x1a\n" + b"png-body"


def _settings(tmp_path) -> Settings:
    return Settings(
        _env_file=None,
        LIGHT_HOUSE_ENV="test",
        WEB_GATE_ENABLED=False,
        PROMPT_IMAGE_DIR=str(tmp_path / "images"),
    )


def test_save_and_load_round_trip(tmp_path) -> None:
    settings = _settings(tmp_path)
    image = PromptImageIn(
        media_type="image/jpeg",
        data_base64=base64.b64encode(JPEG).decode("ascii"),
    )
    image_id = save_prompt_image(settings, image)
    stored = load_prompt_image(settings, image_id)
    assert stored is not None
    assert stored.media_type == "image/jpeg"
    assert base64.b64decode(stored.data_base64) == JPEG


def test_rejects_type_mismatch(tmp_path) -> None:
    settings = _settings(tmp_path)
    image = PromptImageIn(
        media_type="image/jpeg",
        data_base64=base64.b64encode(PNG).decode("ascii"),
    )
    with pytest.raises(PromptImageError, match="does not match"):
        save_prompt_image(settings, image)


def test_rejects_oversize(tmp_path) -> None:
    settings = _settings(tmp_path)
    raw = b"\xff\xd8\xff" + b"x" * MAX_IMAGE_BYTES
    image = PromptImageIn(
        media_type="image/jpeg",
        data_base64=base64.b64encode(raw).decode("ascii"),
    )
    with pytest.raises(PromptImageError, match="too large"):
        save_prompt_image(settings, image)


def test_data_url_prefix_is_unwrapped() -> None:
    payload = base64.b64encode(JPEG).decode("ascii")
    image = PromptImageIn(
        media_type="image/jpeg",
        data_base64=f"data:image/jpeg;base64,{payload}",
    )
    assert image.data_base64 == payload


def test_human_message_includes_image_part(tmp_path) -> None:
    settings = _settings(tmp_path)
    image_id = save_prompt_image(
        settings,
        PromptImageIn(
            media_type="image/png",
            data_base64=base64.b64encode(PNG).decode("ascii"),
        ),
    )
    message = human_message("Look at this.", settings, [image_id])
    content = message.content
    assert isinstance(content, list)
    assert content[0]["type"] == "text"
    assert content[0]["text"] == "Look at this."
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert message_text(content) == "Look at this."


def test_chat_request_allows_image_only() -> None:
    payload = base64.b64encode(JPEG).decode("ascii")
    req = ChatRequest(
        message="",
        image={"media_type": "image/jpeg", "data_base64": payload},
    )
    assert req.message == IMAGE_ONLY_TEXT
    utter = GroupUtterRequest(image={"media_type": "image/jpeg", "data_base64": payload})
    assert utter.message == IMAGE_ONLY_TEXT


def test_chat_request_rejects_empty() -> None:
    with pytest.raises(ValidationError):
        ChatRequest(message="   ")


def test_transcript_line_marks_image() -> None:
    assert transcript_line("Kevin", "hi", image_id=None) == "Kevin: hi"
    assert transcript_line("Kevin", "hi", image_id="abc") == "Kevin: hi [image attached]"
