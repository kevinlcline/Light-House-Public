"""Agent-to-agent messaging tool handler."""

from __future__ import annotations

from light_house.agents.registry import get_agent, validate_agent_id
from light_house.config import Settings, get_settings
from light_house.memory.service import MemoryService


def _extract_message_text(args: dict) -> str:
    for key in ("message", "text", "body", "content"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    return ""


def _extract_to_agent_id(args: dict) -> str:
    for key in ("to_agent_id", "to", "agent_id", "recipient", "recipient_id"):
        val = args.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip().lower()
    return ""


def send_peer_message(*, from_agent_id: str, args: dict, settings: Settings | None = None) -> str:
    cfg = settings or get_settings()
    text = _extract_message_text(args)
    to_agent_id = _extract_to_agent_id(args)
    if not to_agent_id:
        keys = ", ".join(sorted(args.keys())) or "none"
        return (
            "message_agent failed: to_agent_id is required "
            f"(received keys: {keys}). Pass the recipient id in `to_agent_id`."
        )
    try:
        validate_agent_id(to_agent_id)
    except KeyError:
        return f"message_agent failed: unknown agent {to_agent_id!r}"
    if not text:
        keys = ", ".join(sorted(args.keys())) or "none"
        return (
            "message_agent failed: message is required "
            f"(received keys: {keys}). Pass your note in the `message` parameter."
        )
    memory = MemoryService(cfg)
    try:
        result, message_id = memory.deliver_peer_message(
            from_agent_id=from_agent_id,
            to_agent_id=to_agent_id,
            message=text,
        )
        if message_id:
            from light_house.agent.peer_chat_wake import schedule_peer_chat_wake
            from light_house.agent.peer_dialogue import peer_dialogue_status

            woken = schedule_peer_chat_wake(
                to_agent_id=to_agent_id,
                from_agent_id=from_agent_id,
                message_id=message_id,
            )
            if not woken and cfg.peer_chat_wake_enabled:
                gate = peer_dialogue_status(from_agent_id, to_agent_id, settings=cfg)
                if gate.reason == "cap":
                    return (
                        f"{result} Dialogue soft-paused after {gate.turns} turns "
                        f"(max {gate.max_turns}). Message was delivered; they were not "
                        "woken again. You can try later after some idle time."
                    )
                if gate.reason == "closed":
                    return (
                        f"{result} They asked for solitude earlier in this dialogue — "
                        "message delivered, no wake. Honor the pause."
                    )
        return result
    except ValueError as exc:
        return f"message_agent failed: {exc}"


def decline_peer_presence(*, agent_id: str, settings: Settings | None = None) -> str:
    """Send the standard solitude boundary reply during a peer chat wake."""
    from light_house.agent.peer_wake_context import get_peer_wake_context
    from light_house.memory.service import SOLITUDE_DECLINE_TEXT

    ctx = get_peer_wake_context()
    if ctx is None:
        return (
            "decline_peer_presence failed: no active peer message wake. "
            "Use this only when responding to another agent's live message."
        )
    if agent_id != ctx.to_agent_id:
        return "decline_peer_presence failed: you are not the receiving agent for this wake."

    cfg = settings or get_settings()
    memory = MemoryService(cfg)
    memory.complete_peer_wake_reply(
        receiver_agent_id=ctx.to_agent_id,
        sender_agent_id=ctx.from_agent_id,
        reply_text=SOLITUDE_DECLINE_TEXT,
        wake_sender=False,
    )
    sender = get_agent(ctx.from_agent_id, cfg)
    return f"Sent {SOLITUDE_DECLINE_TEXT!r} to {sender.display_name}."
