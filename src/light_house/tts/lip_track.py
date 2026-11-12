"""Five-shape cartoon lip track for studio, timed to a spoken line.

Shapes stay coarse on purpose: closed, smile, wide, round, open.
Vowels and M/B/P carry the talk; f/v/s/z add a little extra motion.
Filler consonants and word gaps still hold the last shape so a short
line does not chew.
Phonemes are stretched across the WAV duration — good enough for TTS we
just synthesized, without Montreal Forced Aligner or a video model.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_HUM_RE = re.compile(r"\b[hm]m+\b", re.I)

SHAPES = ("closed", "smile", "wide", "round", "open")

_OPEN = {
    "closed": 0.06,
    "smile": 0.32,
    "wide": 0.48,
    "round": 0.56,
    "open": 0.82,
}
_WEIGHT = {
    "closed": 0.40,
    "smile": 0.85,
    "wide": 1.20,
    "round": 1.25,
    "open": 1.40,
}

# Hold a shape at least this long unless it is a brief M/B/P close.
_MIN_HOLD = 0.08
_MIN_CLOSED = 0.07

_IPA_CLOSED = set("pbmbm")
_IPA_WIDE = set("iɪeɛyj")
_IPA_ROUND = set("ouɔʊw")
_IPA_OPEN = set("aɑæʌɐ")
_IPA_SMILE = set("fvszθðʃʒ")
_IPA_SKIP = set("ˈˌː͡.·-‐— ")


@dataclass(frozen=True)
class LipCue:
    t: float
    shape: str
    open: float

    def as_dict(self) -> dict[str, float | str]:
        return {"t": round(self.t, 3), "shape": self.shape, "open": round(self.open, 3)}


def duration_from_wav(wav: bytes) -> float:
    """Seconds of audio in a PCM WAV. 0 if the header is unreadable."""
    if len(wav) < 44 or wav[0:4] != b"RIFF" or wav[8:12] != b"WAVE":
        return 0.0
    offset = 12
    sample_rate = 0
    channels = 1
    bits = 16
    data_size = 0
    while offset + 8 <= len(wav):
        chunk_id = wav[offset : offset + 4]
        size = int.from_bytes(wav[offset + 4 : offset + 8], "little")
        body = offset + 8
        if chunk_id == b"fmt " and body + 16 <= len(wav):
            channels = max(1, int.from_bytes(wav[body + 2 : body + 4], "little"))
            sample_rate = int.from_bytes(wav[body + 4 : body + 8], "little")
            bits = int.from_bytes(wav[body + 14 : body + 16], "little") or 16
        elif chunk_id == b"data":
            data_size = max(0, min(size, len(wav) - body))
        advance = 8 + size + (size & 1)
        if advance <= 0:
            break
        offset += advance
    if sample_rate <= 0 or bits <= 0 or channels <= 0 or data_size <= 0:
        return 0.0
    return data_size / (sample_rate * channels * (bits / 8))


def build_lip_track(text: str, duration_s: float) -> list[dict[str, float | str]]:
    """Return sparse cues covering [0, duration_s]. Always starts and ends closed."""
    seconds = max(0.0, float(duration_s))
    units = _speech_units(text)
    if seconds <= 0 or not units:
        return [LipCue(0.0, "closed", 0.0).as_dict()]

    pad = min(0.05, seconds * 0.06)
    speak = max(0.04, seconds - 2 * pad)
    weights = [_WEIGHT[shape] for _, shape in units]
    total = sum(weights) or 1.0
    cues: list[LipCue] = [LipCue(0.0, "closed", 0.0)]
    t = pad
    for (_phone, shape), weight in zip(units, weights, strict=True):
        if cues[-1].shape != shape:
            cues.append(LipCue(t, shape, _OPEN[shape]))
        t += speak * (weight / total)
    end = max(cues[-1].t, seconds - pad)
    if cues[-1].shape != "closed":
        cues.append(LipCue(end, "closed", 0.0))
    if cues[-1].t < seconds:
        cues.append(LipCue(seconds, "closed", 0.0))
    return [cue.as_dict() for cue in _coarsen(cues, seconds)]


_HUM_ONLY_RE = re.compile(r"^\W*hmm\W*$", re.I)


def _speech_units(text: str) -> list[tuple[str, str]]:
    raw = _HUM_RE.sub("hmm", (text or "").strip())
    if not raw:
        return []
    if _HUM_ONLY_RE.match(raw):
        return [("hmm", "closed")]
    phones = _ipa_units(raw)
    units = phones if phones else _letter_units(raw)
    talking = [
        (phone, shape)
        for phone, shape in units
        if shape in {"closed", "smile", "wide", "round", "open"}
    ]
    if talking:
        return talking
    return [("speech", "smile")]


def _coarsen(cues: list[LipCue], _seconds: float) -> list[LipCue]:
    """Drop shape changes that arrive faster than cartoon speech."""
    if len(cues) <= 2:
        return cues
    kept: list[LipCue] = [cues[0]]
    last = len(cues) - 1
    for i, cue in enumerate(cues[1:], start=1):
        if i == last:
            if cue.shape != kept[-1].shape:
                kept.append(cue)
            else:
                kept[-1] = LipCue(cue.t, kept[-1].shape, kept[-1].open)
            continue
        # Opening rest is not a hold — always take the first talking shape.
        if kept[-1].t == 0.0 and kept[-1].shape == "closed":
            kept.append(cue)
            continue
        dt = cue.t - kept[-1].t
        min_dt = _MIN_CLOSED if cue.shape == "closed" else _MIN_HOLD
        if dt < min_dt:
            continue
        kept.append(cue)
    return kept


def _ipa_units(text: str) -> list[tuple[str, str]]:
    try:
        from phonemizer import phonemize
    except Exception:
        return []
    try:
        ipa = phonemize(
            text,
            language="en-us",
            backend="espeak",
            strip=True,
            preserve_punctuation=True,
            with_stress=False,
        )
    except Exception:
        return []
    if not isinstance(ipa, str) or not ipa.strip():
        return []
    units: list[tuple[str, str]] = []
    for ch in ipa:
        if ch in _IPA_SKIP:
            continue
        if ch in ",.!?;:":
            if ch in ".!?":
                units.append((ch, "closed"))
            continue
        shape = _shape_for_ipa(ch)
        if shape:
            units.append((ch, shape))
    return units


def _shape_for_ipa(ch: str) -> str | None:
    if ch in _IPA_CLOSED:
        return "closed"
    if ch in _IPA_WIDE:
        return "wide"
    if ch in _IPA_ROUND:
        return "round"
    if ch in _IPA_OPEN:
        return "open"
    if ch in _IPA_SMILE:
        return "smile"
    return None


def _letter_units(text: str) -> list[tuple[str, str]]:
    units: list[tuple[str, str]] = []
    i = 0
    lower = text.lower()
    while i < len(lower):
        ch = lower[i]
        nxt = lower[i + 1] if i + 1 < len(lower) else ""
        if ch in " \n\t":
            i += 1
            continue
        if ch in ",.!?;:":
            if ch in ".!?":
                units.append((ch, "closed"))
            i += 1
            continue
        if not ch.isalpha():
            i += 1
            continue
        pair = ch + nxt
        if pair in {"ee", "ea", "ey", "ie", "ei"}:
            units.append((pair, "wide"))
            i += 2
            continue
        if pair in {"oo", "ou", "ow", "au", "aw"}:
            units.append((pair, "round"))
            i += 2
            continue
        if pair in {"ai", "ay"}:
            units.append((pair, "wide"))
            i += 2
            continue
        if ch in "mbp":
            # One closed hold for "mmm" / "pp" — not three letter pulses.
            j = i
            while j < len(lower) and lower[j] == ch:
                j += 1
            units.append((lower[i:j], "closed"))
            i = j
            continue
        elif ch in "a":
            units.append((ch, "open"))
        elif ch in "eiy":
            units.append((ch, "wide"))
        elif ch in "ouw":
            units.append((ch, "round"))
        elif ch in "fvsz":
            units.append((ch, "smile"))
        i += 1
    return units
