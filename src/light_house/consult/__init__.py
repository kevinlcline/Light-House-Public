"""Clean-model consult for rumination (Door 2).

A light may ask a frontier OpenRouter model that has no house, no persona,
and no yesterday. The thread lives only for the current wake and is deleted
when that wake ends.
"""

from light_house.consult.ask import ConsultError, execute_consult_tool
from light_house.consult.wake import (
    ConsultWake,
    begin_consult_wake,
    clear_consult_wake,
    current_consult_wake,
    end_wake,
    invoke_rumination_graph,
)

__all__ = [
    "ConsultError",
    "ConsultWake",
    "begin_consult_wake",
    "clear_consult_wake",
    "current_consult_wake",
    "end_wake",
    "execute_consult_tool",
    "invoke_rumination_graph",
]
