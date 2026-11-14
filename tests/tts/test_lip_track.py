"""Cartoon lip-track: five shapes stretched across a known duration."""

from __future__ import annotations

from light_house.tts.lip_track import build_lip_track, duration_from_wav


def _shapes(cues: list[dict]) -> set[str]:
    return {str(cue["shape"]) for cue in cues}


def test_empty_text_is_closed() -> None:
    cues = build_lip_track("", 1.0)
    assert cues[0]["shape"] == "closed"
    assert cues[-1]["shape"] == "closed"


def test_mmm_is_one_closed_hold() -> None:
    cues = build_lip_track("Mmm.", 1.2)
    shapes = [str(cue["shape"]) for cue in cues]
    assert shapes[0] == "closed"
    assert shapes[-1] == "closed"
    # Letter-M spelling would insert "wide" for the E in "em".
    assert "wide" not in shapes
    assert sum(1 for shape in shapes if shape == "closed") <= 3


def test_mama_uses_closed() -> None:
    cues = build_lip_track("mama", 1.2)
    assert "closed" in _shapes(cues)


def test_see_uses_wide() -> None:
    cues = build_lip_track("see", 0.8)
    assert "wide" in _shapes(cues)


def test_boot_uses_round() -> None:
    cues = build_lip_track("boot", 0.9)
    assert "round" in _shapes(cues)


def test_hot_uses_open() -> None:
    cues = build_lip_track("hot", 0.8)
    assert "open" in _shapes(cues)


def test_open_amounts_stay_cartoon() -> None:
    cues = build_lip_track("see hot boot", 1.6)
    by_shape = {str(cue["shape"]): float(cue["open"]) for cue in cues}
    assert by_shape.get("wide", 0.0) >= 0.45
    assert by_shape.get("open", 0.0) >= 0.78
    assert by_shape.get("round", 0.0) >= 0.50


def test_short_line_does_not_chew() -> None:
    cues = build_lip_track("Hello there friend.", 2.4)
    spoken = [cue for cue in cues if cue["shape"] != "closed"]
    assert 1 <= len(spoken) <= 8
    # Word gaps must not snap shut — only start/end and M/B/P.
    assert sum(1 for cue in cues if cue["shape"] == "closed") <= 3


def test_track_covers_duration_and_stays_ordered() -> None:
    cues = build_lip_track("Hello there friend.", 2.4)
    times = [float(cue["t"]) for cue in cues]
    assert times == sorted(times)
    assert times[0] == 0.0
    assert times[-1] == 2.4
    assert cues[0]["shape"] == "closed"
    assert cues[-1]["shape"] == "closed"
    assert all(cue["shape"] in {"closed", "smile", "wide", "round", "open"} for cue in cues)


def test_duration_from_wav_reads_pcm_header() -> None:
    # 0.5s mono 16-bit 24kHz silence
    samples = 12000
    data = samples * 2
    wav = bytearray(44 + data)
    wav[0:4] = b"RIFF"
    wav[4:8] = (36 + data).to_bytes(4, "little")
    wav[8:12] = b"WAVE"
    wav[12:16] = b"fmt "
    wav[16:20] = (16).to_bytes(4, "little")
    wav[20:22] = (1).to_bytes(2, "little")
    wav[22:24] = (1).to_bytes(2, "little")
    wav[24:28] = (24000).to_bytes(4, "little")
    wav[28:32] = (48000).to_bytes(4, "little")
    wav[32:34] = (2).to_bytes(2, "little")
    wav[34:36] = (16).to_bytes(2, "little")
    wav[36:40] = b"data"
    wav[40:44] = data.to_bytes(4, "little")
    assert abs(duration_from_wav(bytes(wav)) - 0.5) < 0.001
