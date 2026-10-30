"""Regression tests for #30323: leftover /steer ordering vs interrupt follow-up.

``_pending_input`` is a FIFO feeding the next turn. When a run ends with BOTH
leftover ``pending_steer`` (the agent finished before absorbing the /steer) and
a re-queued interrupt follow-up, the leftover steer must land in the queue
FIRST, otherwise the next turn processes the normal follow-up text before the
steer guidance and the steer appears ineffective.
"""

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


def _turn(result):
    turn = MagicMock()
    turn.mute_notification_reply = False
    turn.result = result
    turn.use_streaming_tts = False
    return turn


def _drain(q):
    out = []
    while not q.empty():
        out.append(q.get_nowait())
    return out


def test_leftover_steer_is_enqueued_before_interrupt_followup():
    cli = _Stub()
    result = {
        "interrupted": True,
        "interrupt_message": "please continue with the refactor",
        "pending_steer": "actually focus on the bug first",
        "final_response": "",
    }

    cli._chat_render_turn(_turn(result), MagicMock(), None)

    assert _drain(cli._pending_input) == [
        "actually focus on the bug first",
        "please continue with the refactor",
    ]


def test_interrupt_batch_follows_steer_and_preserves_arrival_order():
    cli = _Stub()
    cli._interrupt_queue.put("second interrupt")
    cli._interrupt_queue.put("third interrupt")
    result = {
        "interrupted": True,
        "interrupt_message": "first interrupt",
        "pending_steer": "steer guidance",
        "final_response": "",
    }

    cli._chat_render_turn(_turn(result), MagicMock(), None)

    drained = _drain(cli._pending_input)
    assert drained[0] == "steer guidance"
    # Combined interrupt batch keeps its internal arrival order (#110737 invariant).
    assert drained[1] == "first interrupt\nsecond interrupt\nthird interrupt"


def test_steer_only_when_no_interrupt_followup():
    cli = _Stub()
    result = {"pending_steer": "use rg not grep", "final_response": ""}

    cli._chat_render_turn(_turn(result), MagicMock(), None)

    assert _drain(cli._pending_input) == ["use rg not grep"]
