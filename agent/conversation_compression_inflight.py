"""An over-window request behind another path's in-flight summary (#145249).

Gateway session hygiene summarises a session in a detached worker. Its per-session compression lock
makes every other path's preflight a no-op while it streams, and the summary's commit cannot be
cancelled — so a turn that would otherwise send its full transcript has exactly two safe outcomes:
wait for the commit, or end as ``compression_deferred`` before the provider call. These are the
shared questions that decision needs; compression itself stays in ``conversation_compression``.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Optional

logger = logging.getLogger("agent.conversation_compression")

_COMPRESSION_LOCK_POLL_SECONDS = 0.25


def request_exceeds_input_budget(agent: Any, request_tokens: Any) -> Optional[bool]:
    """``request_exceeds_model_window`` counted against the provider's usable INPUT budget
    (``context_length - max_tokens``). The provider carves the output reservation out of the same
    window, so a prompt between the budget and the raw window is still rejected as over-window.
    ``None`` when either side is unknown, like its sibling."""
    window = getattr(getattr(agent, "context_compressor", None), "context_length", None)
    if isinstance(window, bool) or not isinstance(window, int) or window <= 0:
        return None
    if isinstance(request_tokens, bool) or not isinstance(request_tokens, int):
        return None
    reservation = getattr(agent.context_compressor, "max_tokens", None)
    if isinstance(reservation, int) and not isinstance(reservation, bool) and 0 < reservation < window:
        window -= reservation
    return request_tokens > window


def compression_lock_held(agent: Any) -> bool:
    """Whether another path holds this session's compression lock. SessionDB is the cross-process
    witness; an unreadable board is never taken as "held"."""
    session_db = getattr(agent, "_session_db", None)
    session_id = getattr(agent, "session_id", None) or ""
    getter = getattr(type(session_db), "get_compression_lock_holder", None) if session_db is not None else None
    if not callable(getter) or not session_id:
        return False
    try:
        return bool(getter(session_db, session_id))
    except Exception:
        logger.debug("compression lock read failed (session=%s)", session_id, exc_info=True)
        return False


def wait_for_compression_lock_release(agent: Any, *, timeout_seconds: Optional[float] = None) -> bool:
    """Wait, bounded, for the in-flight summary of this session to commit; True once the lock is free.
    A hygiene summary runs in a detached worker whose commit cannot be cancelled, so a turn holding an
    unsendable request waits here instead of probing the provider. One wait per turn: once
    ``_compression_lock_waited`` is set, a repeat call is a plain read, so a lock that is re-taken
    immediately cannot wedge the turn in wait/re-measure cycles."""
    if getattr(agent, "_compression_lock_waited", False):
        return not compression_lock_held(agent)
    if not compression_lock_held(agent):
        return True
    agent._compression_lock_waited = True
    if timeout_seconds is None:
        from agent.conversation_compression_deadline import resolve_context_compression_timeouts

        timeout_seconds, _ = resolve_context_compression_timeouts()
    try:
        budget = max(float(timeout_seconds), 0.0)
    except (TypeError, ValueError):
        budget = 0.0
    deadline = time.monotonic() + budget
    while time.monotonic() < deadline:
        time.sleep(min(_COMPRESSION_LOCK_POLL_SECONDS, max(deadline - time.monotonic(), 0.01)))
        if not compression_lock_held(agent):
            return True
    return not compression_lock_held(agent)
