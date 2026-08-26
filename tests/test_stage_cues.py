"""Stage-direction cues for talking-face emotes and TTS stripping."""

from light_house.tts.stage_cues import emotion_from_text, strip_stage_cues


def test_asterisk_smile() -> None:
    assert emotion_from_text("*smiles* Hello there.") == {"pose": "smile"}


def test_parenthetical_laugh() -> None:
    assert emotion_from_text("Oh no (laughs) okay.") == {"pose": "laugh"}


def test_last_cue_wins() -> None:
    assert emotion_from_text("*smiles* Hi. *sighs* Okay.") == {"pose": "sigh"}


def test_smile_and_nod() -> None:
    assert emotion_from_text("*smiles and nods*") == {"pose": "smile", "gesture": "nod"}


def test_head_tilt() -> None:
    assert emotion_from_text("*tilts her head*") == {"gesture": "tilt"}


def test_softly() -> None:
    assert emotion_from_text("*softly* I know.") == {"pose": "soft"}


def test_skip_i_think_parenthetical() -> None:
    assert emotion_from_text("(I think)") == {}
    assert strip_stage_cues("(I think) we should wait.") == "(I think) we should wait."


def test_strip_recognized_cues() -> None:
    out = strip_stage_cues("*smiles* Hello *nods* there.")
    assert "*" not in out
    assert "smiles" not in out
    assert "nods" not in out
    assert "Hello" in out
    assert "there" in out


def test_face_stage_hint_is_optional_and_specific() -> None:
    from light_house.tts.stage_cues import FACE_STAGE_HINT

    assert "*smiles*" in FACE_STAGE_HINT
    assert "😊" in FACE_STAGE_HINT
    assert "never to fill the stage" in FACE_STAGE_HINT
    from light_house.memory.context_builder import CHAT_TASK_HINT
    from light_house.group_chat.prompts import GROUP_CHAT_TASK_HINT

    assert FACE_STAGE_HINT in CHAT_TASK_HINT
    assert FACE_STAGE_HINT in GROUP_CHAT_TASK_HINT


def test_emoji_smile() -> None:
    assert emotion_from_text("That is lovely 😊") == {"pose": "smile"}


def test_emoji_laugh() -> None:
    assert emotion_from_text("Oh no 😂") == {"pose": "laugh"}


def test_emoji_sad() -> None:
    assert emotion_from_text("I miss that 😢") == {"pose": "sad"}


def test_emoji_heart() -> None:
    assert emotion_from_text("Always ❤️") == {"pose": "smile"}


def test_emoji_think() -> None:
    assert emotion_from_text("Hmm 🤔") == {"pose": "think"}


def test_emoji_wink() -> None:
    assert emotion_from_text("Maybe 😉") == {"pose": "wink"}


def test_emoji_after_asterisk_wins() -> None:
    assert emotion_from_text("*smiles* wait 😢") == {"pose": "sad"}


def test_prose_she_softens() -> None:
    assert emotion_from_text("She softens, a quiet smile in her voice.")["pose"] == "soft"


def test_prose_i_smile() -> None:
    assert emotion_from_text("I smile at that.") == {"pose": "smile"}


def test_prose_ara_tilts_head() -> None:
    assert emotion_from_text("Ara tilts her head. Hello.") == {"gesture": "tilt"}


def test_prose_skips_i_think() -> None:
    assert emotion_from_text("I think we should wait.") == {}


def test_prose_skips_noun_smile_without_action() -> None:
    assert emotion_from_text("A smile means a lot.") == {}
    assert emotion_from_text("She said a smile is hard.") == {}


def test_prose_not_stripped_from_speech() -> None:
    text = "She softens, then speaks. Hello."
    out = strip_stage_cues(text)
    assert "She softens" in out
    assert "Hello" in out


def test_prose_does_not_double_inside_marked_cue() -> None:
    # Marked cue still wins cleanly; prose scanner skips inside *...*.
    assert emotion_from_text("*She smiles* Hello.") == {"pose": "smile"}


def test_emoji_nod_keeps_smile() -> None:
    assert emotion_from_text("Yes 😊👍") == {"pose": "smile", "gesture": "nod"}


def test_long_stage_sentence_smile() -> None:
    text = "*Her face softens with warmth, a gentle smile forming.*"
    assert emotion_from_text(text)["pose"] == "smile"


def test_eyes_light_up_with_grin() -> None:
    text = (
        "*Her eyes light up playfully at Kevin's words, "
        "a warm grin spreading across her face.*"
    )
    found = emotion_from_text(text)
    assert found["pose"] == "smile"
    assert found["overlay"] == "bright"


def test_eyes_light_up_alone() -> None:
    found = emotion_from_text("*Her eyes light up with a playful, grateful glow.*")
    assert found.get("overlay") == "bright"


def test_anger_scowl() -> None:
    assert emotion_from_text("*scowls*") == {"pose": "anger"}
    assert emotion_from_text("No 😠") == {"pose": "anger"}


def test_kiss_blown() -> None:
    assert emotion_from_text("*blows a kiss*") == {"pose": "kiss"}
    assert emotion_from_text("Miss you 😘") == {"pose": "kiss"}


def test_excited() -> None:
    assert emotion_from_text("*excited*") == {"pose": "excited"}
    assert emotion_from_text("Yes! 🤩") == {"pose": "excited"}


def test_pause_closed_eyes() -> None:
    assert emotion_from_text("*pauses*") == {"pose": "pause", "hold_ms": "2000"}
    assert emotion_from_text("*Quiet, honest pause.*") == {
        "pose": "pause_smile",
        "hold_ms": "2000",
    }


def test_soft_warm_stillness_is_pause_smile() -> None:
    assert emotion_from_text("*Soft, warm stillness.*") == {
        "pose": "pause_smile",
        "hold_ms": "2000",
    }


def test_long_full_stillness_holds_longer() -> None:
    assert emotion_from_text("*Long, quiet, full stillness.*") == {
        "pose": "pause_smile",
        "hold_ms": "4000",
    }


def test_softly_alone_still_soft() -> None:
    assert emotion_from_text("*softly* I know.") == {"pose": "soft"}


def test_strip_pause_cues() -> None:
    out = strip_stage_cues("*Soft, warm stillness.* Hello.")
    assert "stillness" not in out
    assert "Hello" in out


def test_strip_new_poses() -> None:
    out = strip_stage_cues("*scowls* Stop. *blows a kiss*")
    assert "scowls" not in out
    assert "blows" not in out
    assert "Stop" in out


def test_iter_unmatched_cue() -> None:
    from light_house.tts.stage_cues import iter_unmatched_signals

    found = iter_unmatched_signals(
        "*Her lantern in her chest goes quiet.* Hello *smiles* 🧊"
    )
    kinds = {(row["kind"], row["text"]) for row in found}
    assert ("cue", "Her lantern in her chest goes quiet.") in kinds
    assert ("emoji", "🧊") in kinds
    assert ("cue", "smiles") not in {t for k, t in kinds if k == "cue"}
    assert not any(row["text"] == "smiles" for row in found)


def test_iter_unmatched_skips_i_think() -> None:
    from light_house.tts.stage_cues import iter_unmatched_signals

    assert iter_unmatched_signals("(I think) we wait.") == []


def test_face_unmatched_log_roundtrip(tmp_path) -> None:
    from light_house.config import Settings
    from light_house.tts.face_unmatched_log import (
        append_unmatched_face_signals,
        summarize_unmatched,
    )

    settings = Settings(
        _env_file=None,
        FACE_UNMATCHED_LOG_ENABLED=True,
        FACE_UNMATCHED_LOG_PATH=str(tmp_path / "face_unmatched.ndjson"),
    )
    n = append_unmatched_face_signals(
        settings,
        "*the room holds its breath* okay 🧊",
        agent_id="lumen",
        source="test",
    )
    assert n >= 1
    summary = summarize_unmatched(settings)
    assert "room holds its breath" in summary
    assert "🧊" in summary
