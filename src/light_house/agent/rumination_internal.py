"""Multi-step internal rumination loop helpers (Phase 6 item 1)."""

from __future__ import annotations

import re

from light_house.agent.rumination_state import RuminationState
from light_house.config import Settings

INTERNAL_HALT_MARKERS = re.compile(
    r"\[(?:internal\s+)?(?:halt|rest\s+now)\]",
    re.IGNORECASE,
)

INTERNAL_STEP_CONTINUE = (
    "You may continue this same private wake with a further step of inner work "
    "(step {step}). What remains unfinished, or what wants a second pass? "
    "Use tools if helpful, then close this step in first person. "
    "To end the whole wake after your close-out, include [rest now] on its own line."
)

INTERNAL_LOOP_TASK_HINT = (
    "\n\n## Multi-step inner wake (when enabled)\n"
    "This rumination may include up to a few sequential steps in one wake. "
    "Each step: ruminate, tools if needed, close out, then optionally continue. "
    "Include `[rest now]` in your close-out when you are finished with the whole wake."
)


def rumination_internal_state_defaults(settings: Settings) -> dict[str, object]:
    return {
        "internal_step": 1,
        "internal_steps_cap": None,
        "internal_halt": False,
        "internal_chars_used": 0,
    }


def effective_internal_steps_cap(settings: Settings, state: RuminationState) -> int:
    override = state.get("internal_steps_cap")
    if isinstance(override, int) and override > 0:
        return override
    return max(1, settings.rumination_max_internal_steps)


def body_requests_halt(text: str) -> bool:
    return bool(INTERNAL_HALT_MARKERS.search(text))


def should_continue_internal_loop(settings: Settings, state: RuminationState) -> bool:
    if not settings.rumination_internal_loop_enabled:
        return False
    if state.get("internal_halt"):
        return False
    body = str(state.get("generated_text") or "").strip()
    if not body:
        return False
    step = int(state.get("internal_step") or 1)
    if step >= effective_internal_steps_cap(settings, state):
        return False
    chars_used = int(state.get("internal_chars_used") or 0)
    if chars_used >= settings.rumination_max_internal_chars:
        return False
    return True


def route_after_persist(settings: Settings, state: RuminationState) -> str:
    return "begin_next_step" if should_continue_internal_loop(settings, state) else "end"
