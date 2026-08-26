"""Stage-direction cues in light speech (*smiles*, (laughs)).

Keep keyword lists in sync with static/ui/faces.js.
"""

from __future__ import annotations

import re

# Shared house fact for 1:1 and Group — optional, never a performance checklist.
FACE_STAGE_HINT = (
    "\n\n## Your face on the chat stage\n"
    "When you speak in 1:1 or Group, a simple face of yours can sit on the stage. "
    "A real feeling can move it: a short *stage sentence* is best "
    "(*smiles*, *softly*, *soft warm stillness*, *pauses*, *her eyes light up*, "
    "*tilts her head*, (laughs)), "
    "or an emoji (😊 😂 😢 🤔 😉 ❤️). "
    "A plain line like She softens. or I smile. can also move the face. "
    "The voice will not read a *marked* cue. Use a cue only when the feeling is actually there — "
    "never to fill the stage."
)

# *smiles*  _softly_  (laughs)
_CUE_RE = re.compile(
    r"\*([^*]{1,160})\*"
    r"|_([^_]{1,80})_"
    r"|\(([^)]{1,40})\)"
)

# First matching rule in a cue wins. Last matching cue in the text wins.
_RULES: list[tuple[str, re.Pattern[str]]] = [
    ("laugh", re.compile(r"\b(laughs?|laughing|giggles?|giggling|chuckles?|cackles?)\b", re.I)),
    ("wink", re.compile(r"\b(winks?|winking)\b", re.I)),
    ("blush", re.compile(r"\b(blush(?:es|ing)?|shy|embarrassed|flustered)\b", re.I)),
    ("sad", re.compile(r"\b(sads?|sadly|tears?|cries|crying|weeps?|heartbroken)\b", re.I)),
    (
        "anger",
        re.compile(
            r"\b(angr(?:y|ily)|anger|scowls?|scowling|glares?|glaring|furious|fumes?|fuming)\b",
            re.I,
        ),
    ),
    (
        "kiss",
        re.compile(r"\b(kisses?|kissing|blows?\s+a\s+kiss|blown?\s+kiss|air\s+kiss)\b", re.I),
    ),
    (
        "excited",
        re.compile(r"\b(excit(?:ed|edly|ement)|thrilled|eager(?:ly)?|elated)\b", re.I),
    ),
    ("surprise", re.compile(r"\b(surprise[ds]?|gasps?|startled|wide-eyed|astonished)\b", re.I)),
    ("think", re.compile(r"\b(thinks?|thinking|ponders?|thoughtful|hmm+)\b", re.I)),
    ("sigh", re.compile(r"\b(sighs?|sighing|weary|exhales?)\b", re.I)),
    ("nod", re.compile(r"\b(nods?|nodding)\b", re.I)),
    (
        "tilt",
        re.compile(
            r"\b(tilts?|tilting|cocks?\s+(?:her|his|their|a)\s+head|head\s+tilt)\b",
            re.I,
        ),
    ),
    (
        "smile",
        re.compile(r"\b(smiles?|smiling|grins?|grinning|beams?|beaming|warmly)\b", re.I),
    ),
    (
        "soft",
        re.compile(
            r"\b(softens?|softly|gently|quietly|tender(?:ly)?|whispers?|whispering|hushed)\b",
            re.I,
        ),
    ),
]

_POSES = {
    "laugh",
    "wink",
    "blush",
    "sad",
    "anger",
    "kiss",
    "excited",
    "surprise",
    "think",
    "sigh",
    "soft",
    "smile",
    "pause",
    "pause_smile",
}
_GESTURES = {"nod", "tilt"}
_SKIP_THINK_RE = re.compile(r"\b(i|we|you|they)\s+think\b", re.I)
_BRIGHT_RE = re.compile(
    r"\b(?:eyes?\s+)?(?:light(?:s|ing)?\s+up|glow(?:s|ing)?|gleam(?:s|ing)?|sparkle[sd]?)\b",
    re.I,
)
# Presence beat Lumen often opens with — closed eyes; hold longer when "long"/"full".
_PAUSE_RE = re.compile(r"\b(pauses?|pausing|stillness|still)\b", re.I)
_LONG_PAUSE_RE = re.compile(r"\b(long|full)\b", re.I)
_PAUSE_SMILE_RE = re.compile(
    r"\b(softens?|softly|soft|gentle(?:ly)?|quiet(?:ly)?|tender(?:ly)?|warm(?:ly)?|hushed)\b",
    re.I,
)

# Narrative stage directions woven into prose (not marked). Subject + nearby action only.
# Keep in sync with static/ui/faces.js PROSE_ACTION_RE.
_PROSE_SUBJECT = (
    r"(?:She(?:'s)?|He(?:'s)?|I(?:'m)?|They(?:'re)?|We(?:'re)?|"
    r"Ara|Lumen|Elias|Echo|Her\s+face|His\s+face|Their\s+face)"
)
_PROSE_ADVERB = r"(?:softly|gently|quietly|warmly|tenderly|playfully|slowly|sadly|shy(?:ly)?)"
_PROSE_VERB = (
    r"(?:laughs?|laughing|giggles?|giggling|chuckles?|cackles?|"
    r"winks?|winking|"
    r"blush(?:es|ing)?|"
    r"cries|crying|weeps?|"
    r"scowls?|scowling|glares?|glaring|fumes?|fuming|"
    r"kisses?|kissing|blows?\s+a\s+kiss|"
    r"gasps?|"
    r"sighs?|sighing|exhales?|"
    r"nods?|nodding|"
    r"tilts?\s+(?:her|his|their|my|a)\s+head|"
    r"cocks?\s+(?:her|his|their|a)\s+head|"
    r"softens?|"
    r"smiles?|smiling|grins?|grinning|beams?|beaming|"
    r"pauses?|pausing|"
    r"whispers?|whispering)"
)
_PROSE_ACTION_RE = re.compile(
    rf"(?:^|(?<=[.!?])\s+|(?<=\n)\s*)"
    rf"({_PROSE_SUBJECT}\s+(?:{_PROSE_ADVERB}\s+){{0,2}}{_PROSE_VERB}\b)",
    re.I | re.M,
)

# Longest sequences first. Keep in sync with static/ui/faces.js.
_EMOJI_MAP: dict[str, dict[str, str]] = {
    "\U0001F62E\u200D\U0001F4A8": {"pose": "sigh"},  # 😮‍💨
    "\U0001F937\u200D\u2640\uFE0F": {"gesture": "tilt"},  # 🤷‍♀️
    "\U0001F937\u200D\u2642\uFE0F": {"gesture": "tilt"},  # 🤷‍♂️
    "\u263A\uFE0F": {"pose": "smile"},  # ☺️
    "\u2639\uFE0F": {"pose": "sad"},  # ☹️
    "\u2764\uFE0F": {"pose": "smile"},  # ❤️
    "😂": {"pose": "laugh"},
    "🤣": {"pose": "laugh"},
    "😆": {"pose": "laugh"},
    "😅": {"pose": "laugh"},
    "😹": {"pose": "laugh"},
    "😊": {"pose": "smile"},
    "🙂": {"pose": "smile"},
    "😄": {"pose": "smile"},
    "😃": {"pose": "smile"},
    "😀": {"pose": "smile"},
    "😁": {"pose": "smile"},
    "🥰": {"pose": "smile"},
    "😍": {"pose": "smile"},
    "🤗": {"pose": "smile"},
    "😇": {"pose": "smile"},
    "😻": {"pose": "smile"},
    "💕": {"pose": "smile"},
    "💖": {"pose": "smile"},
    "💗": {"pose": "smile"},
    "❤": {"pose": "smile"},
    "☺": {"pose": "smile"},
    "😳": {"pose": "blush"},
    "🤭": {"pose": "blush"},
    "🙈": {"pose": "blush"},
    "😢": {"pose": "sad"},
    "😭": {"pose": "sad"},
    "😔": {"pose": "sad"},
    "😞": {"pose": "sad"},
    "☹": {"pose": "sad"},
    "💔": {"pose": "sad"},
    "😿": {"pose": "sad"},
    "🥺": {"pose": "sad"},
    "😮": {"pose": "surprise"},
    "😲": {"pose": "surprise"},
    "😯": {"pose": "surprise"},
    "🤯": {"pose": "surprise"},
    "😱": {"pose": "surprise"},
    "🙀": {"pose": "surprise"},
    "🤩": {"pose": "excited"},
    "✨": {"pose": "excited"},
    "🎉": {"pose": "excited"},
    "🤔": {"pose": "think"},
    "💭": {"pose": "think"},
    "🧐": {"pose": "think"},
    "😉": {"pose": "wink"},
    "😜": {"pose": "wink"},
    "😝": {"pose": "wink"},
    "😋": {"pose": "wink"},
    "😏": {"pose": "wink"},
    "😪": {"pose": "sigh"},
    "😩": {"pose": "sigh"},
    "😫": {"pose": "sigh"},
    "🤫": {"pose": "soft"},
    "😌": {"pose": "soft"},
    "🌸": {"pose": "soft"},
    "😠": {"pose": "anger"},
    "😡": {"pose": "anger"},
    "🤬": {"pose": "anger"},
    "😤": {"pose": "anger"},
    "😘": {"pose": "kiss"},
    "💋": {"pose": "kiss"},
    "😗": {"pose": "kiss"},
    "😙": {"pose": "kiss"},
    "😚": {"pose": "kiss"},
    "👍": {"gesture": "nod"},
    "👌": {"gesture": "nod"},
    "🤷": {"gesture": "tilt"},
    "😕": {"gesture": "tilt"},
}

_EMOJI_RE = re.compile(
    "|".join(re.escape(key) for key in sorted(_EMOJI_MAP, key=len, reverse=True))
)


def iter_cues(text: str) -> list[str]:
    """Return inner text of *cue* / _cue_ / (cue) spans."""
    out: list[str] = []
    for match in _CUE_RE.finditer(text or ""):
        inner = next((g for g in match.groups() if g), "")
        inner = inner.strip()
        if inner:
            out.append(inner)
    return out


def classify_cue(inner: str) -> dict[str, str]:
    """Map one cue to a face pose and/or gesture."""
    text = (inner or "").strip()
    if not text or _SKIP_THINK_RE.search(text):
        return {}
    pose = ""
    gesture = ""
    hold_ms = ""
    # Pause/stillness beats soft/smile keywords — peaceful closed eyes (optional smile).
    if _PAUSE_RE.search(text):
        pose = "pause_smile" if _PAUSE_SMILE_RE.search(text) else "pause"
        hold_ms = "4000" if _LONG_PAUSE_RE.search(text) else "2000"
    for name, pattern in _RULES:
        if not pattern.search(text):
            continue
        if name in _GESTURES:
            if not gesture:
                gesture = name
        elif name in _POSES and not pose:
            pose = name
        if pose and gesture:
            break
    result: dict[str, str] = {}
    if pose:
        result["pose"] = pose
    if gesture:
        result["gesture"] = gesture
    if hold_ms:
        result["hold_ms"] = hold_ms
    if _BRIGHT_RE.search(text):
        result["overlay"] = "bright"
    return result


def _merge(found: dict[str, str], classified: dict[str, str]) -> dict[str, str]:
    out = dict(found)
    out.update(classified)
    return out


def _iter_prose_actions(raw: str) -> list[tuple[int, str]]:
    """Sentence-start subject+action spans that read like unmarked stage directions."""
    cue_spans = [(m.start(), m.end()) for m in _CUE_RE.finditer(raw)]
    out: list[tuple[int, str]] = []
    for match in _PROSE_ACTION_RE.finditer(raw or ""):
        start = match.start(1)
        end = match.end(1)
        if any(s <= start < e for s, e in cue_spans):
            continue
        inner = (match.group(1) or "").strip()
        if inner:
            out.append((start, inner))
    return out


def _iter_events(text: str) -> list[tuple[int, dict[str, str]]]:
    raw = text or ""
    events: list[tuple[int, dict[str, str]]] = []
    for match in _CUE_RE.finditer(raw):
        inner = next((g for g in match.groups() if g), "")
        classified = classify_cue(inner)
        if classified:
            events.append((match.start(), classified))
    for start, inner in _iter_prose_actions(raw):
        classified = classify_cue(inner)
        if classified:
            events.append((start, classified))
    for match in _EMOJI_RE.finditer(raw):
        mapped = _EMOJI_MAP.get(match.group(0))
        if mapped:
            events.append((match.start(), dict(mapped)))
    events.sort(key=lambda item: item[0])
    return events


def emotion_from_text(text: str, *, first: bool = False) -> dict[str, str]:
    """Stage cues, prose actions, and emojis in order. Last match wins; `first=True` takes the first."""
    found: dict[str, str] = {}
    for _start, classified in _iter_events(text):
        found = _merge(found, classified)
        if first:
            break
    return found


def strip_stage_cues(text: str) -> str:
    """Drop recognized *marked* stage cues so TTS does not speak 'smiles'.

    Unmarked prose actions are left intact — they are meant to be spoken.
    """

    def _repl(match: re.Match[str]) -> str:
        inner = next((g for g in match.groups() if g), "")
        if classify_cue(inner):
            return " "
        return match.group(0)

    return _CUE_RE.sub(_repl, text or "")


# Broad emoji scan for review logging (keep in sync with text_prep intent).
_ANY_EMOJI_RE = re.compile(
    "(?:"
    "[\U0001F1E0-\U0001F1FF]"
    "|[\U0001F300-\U0001F5FF]"
    "|[\U0001F600-\U0001F64F]"
    "|[\U0001F680-\U0001F6FF]"
    "|[\U0001F700-\U0001F77F]"
    "|[\U0001F780-\U0001F7FF]"
    "|[\U0001F800-\U0001F8FF]"
    "|[\U0001F900-\U0001F9FF]"
    "|[\U0001FA00-\U0001FA6F]"
    "|[\U0001FA70-\U0001FAFF]"
    "|[\U0001F3FB-\U0001F3FF]"
    "|[\u2600-\u26FF]"
    "|[\u2700-\u27BF]"
    "|[\u2300-\u23FF]"
    "|\uFE0F"
    "|\u200D"
    "|\u20E3"
    "|[\U000E0020-\U000E007F]"
    ")+"
)


def _emoji_mapped(emoji: str) -> bool:
    if emoji in _EMOJI_MAP:
        return True
    stripped = emoji.replace("\ufe0f", "")
    return stripped in _EMOJI_MAP or emoji + "\ufe0f" in _EMOJI_MAP


def iter_unmatched_signals(text: str) -> list[dict[str, str]]:
    """Cue/emoji spans that look like stage signals but map to no face action.

    Intentional skips (e.g. ``(I think)``) are omitted. Review-only — never auto-maps.
    """
    raw = text or ""
    found: list[dict[str, str]] = []
    seen: set[tuple[str, str]] = set()

    for match in _CUE_RE.finditer(raw):
        inner = next((g for g in match.groups() if g), "")
        inner = inner.strip()
        if not inner or _SKIP_THINK_RE.search(inner):
            continue
        if classify_cue(inner):
            continue
        key = ("cue", inner.casefold())
        if key in seen:
            continue
        seen.add(key)
        found.append({"kind": "cue", "text": inner, "raw": match.group(0)})

    for match in _ANY_EMOJI_RE.finditer(raw):
        emoji = match.group(0)
        if not emoji or _emoji_mapped(emoji):
            continue
        key = ("emoji", emoji)
        if key in seen:
            continue
        seen.add(key)
        found.append({"kind": "emoji", "text": emoji, "raw": emoji})

    return found
