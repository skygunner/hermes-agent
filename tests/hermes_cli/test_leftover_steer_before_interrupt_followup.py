"""#30323: a leftover /steer is guidance for the next turn, so it must reach ``_pending_input``
before an interrupt follow-up; the interrupt batch itself must keep its own order."""

import queue
from unittest.mock import MagicMock

from hermes_cli.cli_chat_turn_mixin import CLIChatTurnMixin


class _Stub(CLIChatTurnMixin):
    def __init__(self):
        self._interrupt_queue = queue.Queue()
        self._pending_input = queue.Queue()
        self._voice_tts = None
        self._voice_continuous = False
        self.agent = MagicMock(max_iterations=500)
        self._chat_print_reasoning_box = lambda turn: None
        self._chat_print_response_panel = lambda turn, response: None
        self._emit_focus_recovery_line = lambda: None
        self._ring_bell = lambda **kwargs: None


def _interrupted_turn(result):
    turn = MagicMock()
    turn.mute_notification_reply = False
    turn.use_streaming_tts = False
    turn.result = result
    return turn


def test_leftover_steer_is_queued_before_the_interrupt_followup():
    cli = _Stub()
    cli._chat_render_turn(
        _interrupted_turn({"final_response": "answer", "interrupted": True,
                           "interrupt_message": "follow-up",
                           "pending_steer": "prefer the stdlib"}),
        MagicMock(), None,
    )

    assert cli._pending_input.get_nowait() == "prefer the stdlib"
    assert cli._pending_input.get_nowait() == "follow-up"
    assert cli._pending_input.empty()


def test_interrupt_batch_keeps_its_own_order_behind_the_steer():
    cli = _Stub()
    cli._interrupt_queue.put("second")
    cli._interrupt_queue.put("third")

    cli._chat_render_turn(
        _interrupted_turn({"final_response": "answer", "interrupted": True,
                           "interrupt_message": "first",
                           "pending_steer": "prefer the stdlib"}),
        MagicMock(), None,
    )

    assert cli._pending_input.get_nowait() == "prefer the stdlib"
    assert cli._pending_input.get_nowait() == "first\nsecond\nthird"
    assert cli._pending_input.empty()
