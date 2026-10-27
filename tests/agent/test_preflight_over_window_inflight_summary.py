"""An over-window turn never rides out an in-flight summary to the provider (#145249).

Gateway hygiene summarises a session in a detached worker. While that summary is running, its
compression lock makes the turn's own preflight a no-op, and the turn used to send its full
transcript anyway — a request the provider rejects with HTTP 400 even when the transcript itself is
under the raw window, because the output reservation comes out of the same window. Both gates must
instead (a) wait, bounded, for the holder's fenced commit and compress/adopt, or (b) end the turn as
``compression_deferred`` before the provider call.
"""

from types import SimpleNamespace

import pytest

from agent import conversation_compression_deadline as ccd
from agent import conversation_compression_inflight as cci
from agent import turn_context as tc
from agent.turn_context_compaction import run_turn_start_compaction
from agent.turn_preflight import PreflightGateVerdict, run_preflight_compression

WINDOW = 262_144
OUTPUT_RESERVATION = 32_000


class _LockBoard:
    """SessionDB stand-in whose compression lock the test owns: ``releases`` = reads before it frees."""

    def __init__(self, releases=None):
        self.reads = 0
        self.releases = releases

    def get_compression_lock_holder(self, session_id):
        self.reads += 1
        if self.releases is not None and self.reads > self.releases:
            return None
        return "pid=1:tid=1:agent=hygiene:nonce=0"


@pytest.fixture(autouse=True)
def _fast_lock_poll(monkeypatch):
    monkeypatch.setattr(cci, "_COMPRESSION_LOCK_POLL_SECONDS", 0.01, raising=False)
    monkeypatch.setattr(ccd, "resolve_context_compression_timeouts", lambda *a, **k: (0.05, 0.1))


def _compressor(*, skip_due_to_lock: bool) -> SimpleNamespace:
    return SimpleNamespace(
        context_length=WINDOW, max_tokens=OUTPUT_RESERVATION, threshold_tokens=222_822,
        protect_first_n=1, protect_last_n=1, summary_target_ratio=0.2,
        last_prompt_tokens=224_522, last_real_prompt_tokens=0,
        should_compress=lambda _tokens: True,
        should_defer_preflight_to_real_usage=lambda _tokens: True,
        get_active_compression_failure_cooldown=lambda: None,
    )


def _pre_api_agent(board, calls):
    agent = SimpleNamespace(
        compression_enabled=True, model="m", session_id="s1", max_compression_attempts=3,
        _compression_feasibility_checked=True, _session_db=board, _api_call_count=1,
        iteration_budget=SimpleNamespace(refund=lambda: None),
        _emit_status=lambda _msg: None, _safe_print=lambda *_a: None,
    )
    agent.context_compressor = _compressor(skip_due_to_lock=True)

    def _compress_context(messages, system, **_kwargs):
        agent._compression_skipped_due_to_lock = "pid=1:tid=1:agent=hygiene:nonce=0"
        calls.append("compress")
        return messages, system

    agent._compress_context = _compress_context
    return agent


def _pre_api_verdict(messages) -> PreflightGateVerdict:
    return PreflightGateVerdict(
        action="fallthrough", pending_moa_prepared_request=None, messages=messages,
        active_system_prompt="sys", conversation_history=None, api_call_count=1,
        compression_attempts=0, final_response=None, failed=False, _turn_exit_reason=None,
        _compression_timeout_exhausted=False, _preflight_compression_blocked=False,
        _provider_overflow_recovery_pending=False, _last_preflight_pressure=0,
    )


def _drive_pre_api(agent, tokens: int):
    messages = [
        {"role": "user", "content": "x" * 8_000},
        {"role": "assistant", "content": "y" * 8_000},
        {"role": "user", "content": "current ask"},
    ]
    return run_preflight_compression(
        agent, _pre_api_verdict(messages), compressor=agent.context_compressor,
        request_pressure_tokens=tokens, provider_overflow_preflight=False,
        defer_preflight=lambda _tokens: False, moa_prepared_request=None,
        system_message="sys", user_message="current ask", max_compression_attempts=3,
        effective_task_id="task",
    )


def test_pre_api_lock_skip_defers_an_over_window_request_without_a_provider_call():
    """257,249 tokens fits the raw 262,144 window but not the 230,144 input budget; the holder never
    commits, so the turn must end as compression_deferred instead of sending the payload."""
    agent = _pre_api_agent(_LockBoard(releases=None), calls=[])
    verdict = _drive_pre_api(agent, 257_249)

    assert verdict.action == "return"
    assert verdict.result["compression_deferred"] is True and verdict.result["failed"] is False
    assert agent._compression_lock_waited is True


def test_pre_api_lock_skip_resumes_compression_once_the_summary_commits():
    """The holder commits inside the budget: the gate re-enters the loop (action ``continue``) so the
    freed lock compresses, instead of sending the unchanged request."""
    agent = _pre_api_agent(_LockBoard(releases=1), calls=[])
    verdict = _drive_pre_api(agent, 257_249)

    assert verdict.action == "continue"
    assert verdict.result is None


def test_pre_api_lock_skip_still_sends_a_request_that_fits_the_budget():
    """Below the input budget the lock-hold stays a DEFER: today's send-as-is behaviour is kept."""
    agent = _pre_api_agent(_LockBoard(releases=None), calls=[])
    verdict = _drive_pre_api(agent, 200_000)

    assert verdict.action == "fallthrough"


def _turn_start_agent(board, calls):
    agent = SimpleNamespace(
        compression_enabled=True, model="m", session_id="s1", max_compression_attempts=3,
        _session_db=board, _emit_status=lambda _msg: None,
        _clear_context_overflow_warn=lambda: None,
    )
    agent.context_compressor = _compressor(skip_due_to_lock=False)

    def _compress_context(messages, system, **_kwargs):
        calls.append("compress")
        return list(messages), system

    agent._compress_context = _compress_context
    return agent


def _drive_turn_start(agent) -> None:
    messages = [
        {"role": "user", "content": "a"},
        {"role": "assistant", "content": "b"},
        {"role": "user", "content": "c"},
        {"role": "user", "content": "current ask"},
    ]
    run_turn_start_compaction(
        agent, messages=messages, system_message="sys", active_system_prompt="sys",
        conversation_history=None, current_turn_user_idx=3, user_message="current ask",
        effective_task_id="task",
    )


def test_turn_start_defers_an_unanchored_over_window_request_until_the_summary_commits(monkeypatch):
    """The rough estimate 249,853 is not anchored on real usage, so preflight normally skips — but the
    request cannot fit while the holder is summarising, so the turn waits for the commit and
    compresses instead of sending it."""
    monkeypatch.setattr(tc, "_preflight_request_tokens", lambda *_a, **_k: 249_853)
    calls: list = []
    _drive_turn_start(_turn_start_agent(_LockBoard(releases=1), calls))

    assert calls == ["compress"]


def test_turn_start_ends_as_deferred_when_the_in_flight_summary_never_commits(monkeypatch):
    monkeypatch.setattr(tc, "_preflight_request_tokens", lambda *_a, **_k: 249_853)
    calls: list = []

    with pytest.raises(tc.PreflightCompressionDeferred, match="does not fit the model's input budget"):
        _drive_turn_start(_turn_start_agent(_LockBoard(releases=None), calls))

    assert calls == []
