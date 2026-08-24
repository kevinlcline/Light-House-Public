"""Bounded multi-turn peer dialogue between lights (solitude still absolute)."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass

from light_house.config import Settings

logger = logging.getLogger(__name__)


@dataclass
class DialogueGate:
    """Result of asking whether a peer wake may continue."""

    allow_wake: bool
    turns: int
    max_turns: int
    reason: str  # "ok" | "disabled" | "closed" | "cap" | "idle_reset"


@dataclass
class _PairState:
    turns: int = 0
    last_ts: float = 0.0
    closed: bool = False  # solitude declined — no auto-continue until idle reset


_lock = threading.Lock()
_pairs: dict[tuple[str, str], _PairState] = {}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    x, y = a.strip().lower(), b.strip().lower()
    return (x, y) if x <= y else (y, x)


def _idle_reset_seconds(settings: Settings) -> float:
    return max(60.0, float(settings.peer_chat_dialogue_idle_reset_seconds))


def _max_turns(settings: Settings) -> int:
    return max(1, int(settings.peer_chat_max_dialogue_turns))


def reset_peer_dialogue_state_for_tests() -> None:
    with _lock:
        _pairs.clear()


def close_peer_dialogue(agent_a: str, agent_b: str) -> None:
    """Mark the pair closed (e.g. solitude decline)."""
    key = _pair_key(agent_a, agent_b)
    with _lock:
        state = _pairs.setdefault(key, _PairState())
        state.closed = True
        state.last_ts = time.time()
    logger.info("Peer dialogue closed pair=%s+%s", key[0], key[1])


def peer_dialogue_status(agent_a: str, agent_b: str, *, settings: Settings) -> DialogueGate:
    """Inspect gate without consuming a turn."""
    if not settings.peer_chat_wake_enabled:
        return DialogueGate(False, 0, _max_turns(settings), "disabled")
    key = _pair_key(agent_a, agent_b)
    now = time.time()
    idle = _idle_reset_seconds(settings)
    max_turns = _max_turns(settings)
    with _lock:
        state = _pairs.get(key)
        if state is None:
            return DialogueGate(True, 0, max_turns, "ok")
        if state.last_ts > 0 and (now - state.last_ts) >= idle:
            return DialogueGate(True, 0, max_turns, "idle_reset")
        if state.closed:
            return DialogueGate(False, state.turns, max_turns, "closed")
        if state.turns >= max_turns:
            return DialogueGate(False, state.turns, max_turns, "cap")
        return DialogueGate(True, state.turns, max_turns, "ok")


def begin_peer_wake_turn(agent_a: str, agent_b: str, *, settings: Settings) -> DialogueGate:
    """
    Reserve one dialogue turn for a wake between the pair.

    Call just before scheduling a peer wake. Returns allow_wake=False when the
    pair is solitude-closed or at the turn cap (after idle reset, turns restart).
    """
    if not settings.peer_chat_wake_enabled:
        return DialogueGate(False, 0, _max_turns(settings), "disabled")
    key = _pair_key(agent_a, agent_b)
    now = time.time()
    idle = _idle_reset_seconds(settings)
    max_turns = _max_turns(settings)
    with _lock:
        state = _pairs.setdefault(key, _PairState())
        if state.last_ts > 0 and (now - state.last_ts) >= idle:
            state.turns = 0
            state.closed = False
            reason_reset = True
        else:
            reason_reset = False
        if state.closed:
            state.last_ts = now
            return DialogueGate(False, state.turns, max_turns, "closed")
        if state.turns >= max_turns:
            state.last_ts = now
            return DialogueGate(False, state.turns, max_turns, "cap")
        state.turns += 1
        state.last_ts = now
        turns = state.turns
    logger.info(
        "Peer dialogue turn pair=%s+%s turns=%d/%d%s",
        key[0],
        key[1],
        turns,
        max_turns,
        " (idle reset)" if reason_reset else "",
    )
    return DialogueGate(True, turns, max_turns, "ok")
