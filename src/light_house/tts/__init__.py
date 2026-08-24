"""Local text-to-speech (Kokoro ONNX)."""

from light_house.tts.kokoro_engine import (
    KokoroTtsError,
    get_tts_status,
    shutdown_tts,
    synthesize_wav,
    voice_for_agent,
    warm_tts,
)
from light_house.tts.sentences import split_speech_chunks
from light_house.tts.voices_catalog import list_voice_catalog

__all__ = [
    "KokoroTtsError",
    "get_tts_status",
    "list_voice_catalog",
    "shutdown_tts",
    "split_speech_chunks",
    "synthesize_wav",
    "voice_for_agent",
    "warm_tts",
]
