"""Group speak-as: account vs present guest attribution."""

from __future__ import annotations

import pytest

from light_house.group_chat.speaker import (
    format_current_speaker_for_prompt,
    format_presence_for_prompt,
    normalize_present_humans,
    resolve_group_utterance_speaker,
)


def test_defaults_to_account():
    sid, name = resolve_group_utterance_speaker(
        account_user_id="kevin",
        account_display_name="Kevin",
    )
    assert sid == "kevin"
    assert name == "Kevin"


def test_explicit_account_speaker():
    sid, name = resolve_group_utterance_speaker(
        account_user_id="kevin",
        account_display_name="Kevin",
        speaker_id="kevin",
        display_name="Ignored",
    )
    assert sid == "kevin"
    assert name == "Kevin"


def test_guest_speaker():
    sid, name = resolve_group_utterance_speaker(
        account_user_id="kevin",
        account_display_name="Kevin",
        speaker_id="guest-1",
        display_name="Joey",
    )
    assert sid == "guest-1"
    assert name == "Joey"


def test_guest_requires_display_name():
    with pytest.raises(ValueError, match="display_name"):
        resolve_group_utterance_speaker(
            account_user_id="kevin",
            account_display_name="Kevin",
            speaker_id="guest-2",
        )


def test_rejects_spoofed_sibling_id():
    with pytest.raises(ValueError, match="guest"):
        resolve_group_utterance_speaker(
            account_user_id="kevin",
            account_display_name="Kevin",
            speaker_id="teeter",
            display_name="Teeter",
        )


def test_rejects_bad_display_name():
    with pytest.raises(ValueError, match="display_name"):
        resolve_group_utterance_speaker(
            account_user_id="kevin",
            account_display_name="Kevin",
            speaker_id="guest-1",
            display_name="bad/name",
        )


def test_normalize_present_includes_account_and_guests():
    roster = normalize_present_humans(
        account_user_id="kevin",
        account_display_name="Kevin",
        present=[
            {
                "speaker_id": "guest-1",
                "display_name": "Joey",
                "description": "Kevin's friend visiting from Austin",
            },
            {"speaker_id": "teeter", "display_name": "Nope"},
        ],
    )
    assert roster == [
        {"speaker_id": "kevin", "display_name": "Kevin", "description": ""},
        {
            "speaker_id": "guest-1",
            "display_name": "Joey",
            "description": "Kevin's friend visiting from Austin",
        },
    ]


def test_presence_prompt_mentions_guests():
    text = format_presence_for_prompt(
        [
            {"speaker_id": "kevin", "display_name": "Kevin"},
            {
                "speaker_id": "guest-1",
                "display_name": "Joey",
                "description": "Kevin's friend visiting from Austin",
            },
        ]
    )
    assert "Joey" in text
    assert "Kevin's friend visiting from Austin" in text
    assert "About the guests" in text
    assert "not necessarily who just spoke" in text
    assert "muted" in text.lower() or "affection" in text.lower()


def test_current_speaker_prompt_guest_not_host():
    present = [
        {"speaker_id": "alt_kevin", "display_name": "Moose", "description": ""},
        {
            "speaker_id": "guest-1",
            "display_name": "Pinky pick pocket",
            "description": "He's a bad guy. Try to ignore him",
        },
    ]
    text = format_current_speaker_for_prompt(
        human_id="guest-1",
        human_display_name="Pinky pick pocket",
        present_humans=present,
    )
    assert "Who is speaking" in text
    assert "Pinky pick pocket" in text
    assert "guest-1" in text
    assert "not the host account" in text
    assert "He's a bad guy" in text
    assert "still address them as **Pinky pick pocket**" in text


def test_current_speaker_prompt_host_among_guests():
    present = [
        {"speaker_id": "alt_kevin", "display_name": "Moose", "description": ""},
        {"speaker_id": "guest-1", "display_name": "Pinky pick pocket", "description": ""},
    ]
    text = format_current_speaker_for_prompt(
        human_id="alt_kevin",
        human_display_name="Moose",
        present_humans=present,
    )
    assert "Moose" in text
    assert "did not say this line" in text
