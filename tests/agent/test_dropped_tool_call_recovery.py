"""Regression tests for dropped tool-call recovery.

Some providers (observed: claude-opus-4.8 / claude-sonnet-4.5 on GitHub
Copilot, ~2026-07) return ``finish_reason="tool_calls"`` while the parsed
``tool_calls`` array is empty — the model signalled it wanted to act but the
payload shipped no call. Before the fix, the conversation loop took the
no-tool-calls ``else`` branch, treated the turn's narration as the final
answer, and exited with the task unstarted. On an unattended multi-step job
(e.g. a scheduled PR reviewer) this silently did nothing.

The fix keys on the provider contract violation itself
(``finish_reason == "tool_calls"`` with zero ``tool_calls``) and re-prompts,
bounded to 3 consecutive stalls, with the budget resetting after any
successful tool round so it guards each stall rather than the whole run. A
genuine ``finish_reason="stop"`` text turn is unaffected.

Regression for #143040: the same contract violation is also reported by
providers that DID finish the turn (the visible answer shipped, only the
finish_reason lies). Re-prompting that answer discards it and a compliant
model then fabricates no-op tool calls to fill the re-opened turn, so the
recovery only applies to a response with no visible text at all.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture()
def loop_agent():
    """AIAgent with a mocked OpenAI client (mirrors test_run_agent's fixture)
    so we can stage a dropped-tool-call response + continuation pair on
    ``.chat.completions.create``."""
    from run_agent import AIAgent
    with (
        patch("model_tools.get_tool_definitions", return_value=[]),
        patch("model_tools.check_toolset_requirements", return_value={}),
        patch("agent.process_bootstrap.OpenAI"),
    ):
        agent = AIAgent(
            api_key="test-key-1234567890",
            base_url="https://openrouter.ai/api/v1",
            quiet_mode=True,
            skip_context_files=True,
            skip_memory=True,
        )
        agent.client = MagicMock()
        agent._cached_system_prompt = "You are helpful."
        agent._use_prompt_caching = False
        agent.tool_delay = 0
        agent.compression_enabled = False
        agent.save_trajectories = False
        return agent


def _dropped_tool_call_response(content: str):
    """A response whose finish_reason claims a tool call, but tool_calls is
    empty — the provider contract violation this fix recovers from."""
    from tests.agent.test_run_agent import _mock_assistant_msg
    return SimpleNamespace(
        id="chatcmpl-dropped",
        model="test/model",
        choices=[SimpleNamespace(
            index=0,
            message=_mock_assistant_msg(content=content, tool_calls=None),
            finish_reason="tool_calls",
        )],
        usage=None,
    )


def _request_messages(call) -> list:
    return call.kwargs.get("messages") or call.args[0].get("messages")


def _nudge_rows_sent(agent) -> int:
    """How many dropped-tool-call nudge rows the model saw on its last request."""
    from agent.conversation_loop import _DROPPED_TOOLCALL_NUDGE_CONTENT
    last = _request_messages(agent.client.chat.completions.create.call_args_list[-1])
    return sum(
        1 for m in last
        if m.get("role") == "user" and m.get("content") == _DROPPED_TOOLCALL_NUDGE_CONTENT
    )


class TestDroppedToolCallRecovery:
    def test_dropped_tool_call_reprompts_instead_of_exiting(self, loop_agent):
        """finish_reason=tool_calls with an empty tool_calls array and no
        visible text must re-prompt the model to emit the call rather than
        exiting the loop with nothing delivered."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response(""),
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert loop_agent.client.chat.completions.create.call_count == 2, (
            "A dropped tool call must trigger a re-prompt (second API call), "
            "not exit the loop after one call."
        )

        # The loop must have injected a nudge user-message telling the model to
        # issue the actual tool call.
        second_call = loop_agent.client.chat.completions.create.call_args_list[1]
        msgs = _request_messages(second_call)
        last_user = next(
            (m for m in reversed(msgs) if m.get("role") == "user"), None,
        )
        assert last_user is not None
        assert "tool call" in (last_user.get("content") or "").lower(), (
            "The nudge must explicitly ask the model to issue the tool call."
        )
        assert "All checks pass" in result["final_response"]


    def test_clean_stop_text_turn_is_unaffected(self, loop_agent):
        """A genuine finish_reason=stop text response must exit normally — the
        recovery path must not fire on ordinary final answers."""
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _mock_response(content="Here is your answer.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("hello")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A clean finish_reason=stop turn must not trigger a re-prompt."
        )
        assert "Here is your answer." in result["final_response"]

    def test_answer_with_dropped_tool_call_finish_is_delivered(self, loop_agent):
        """Regression for #143040: a response that carries the visible answer
        but reports finish_reason=tool_calls with no calls has finished the
        turn. Re-prompting it throws the answer away and the model answers the
        nudge with fabricated no-op tool calls, so the turn must end here."""
        answer = "The migration is finished. Nothing left to run — waiting for your review."

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response(answer),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("finish the migration")

        assert loop_agent.client.chat.completions.create.call_count == 1, (
            "A visible answer must end the turn, not be re-opened by a nudge."
        )
        assert result["completed"] is True
        assert result["final_response"] == answer

    def test_persistent_dropped_tool_calls_are_bounded(self, loop_agent):
        """If the model never emits a call, the recovery must give up after a
        bounded number of consecutive stalls instead of looping forever."""
        from tests.agent.test_run_agent import _mock_response

        # Four empty dropped-tool-call responses then a clean stop: the first
        # three exhaust the nudge bound, the fourth must fall to the empty-
        # response ladder instead of nudging again.
        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response("") for _ in range(4)
        ] + [_mock_response(content="done", finish_reason="stop")]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
            # Past the bound the empty-response ladder owns the blank reply; its
            # paid backoff wait is not what this test measures.
            patch("agent.turn_empty_response.interruptible_backoff_sleep", return_value=None),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert _nudge_rows_sent(loop_agent) == 3, (
            "Consecutive dropped tool calls must be bounded to 3 re-prompts."
        )
        assert result["completed"] is True
        assert "done" in result["final_response"]

    def test_nudge_pair_is_ephemeral_scaffolding(self, loop_agent):
        """The re-prompt pair (interim assistant turn + synthetic user nudge)
        must be flagged as ephemeral scaffolding so persistence never writes
        it to the durable transcript — a resumed session must not replay the
        internal "issue the actual tool call now" instruction as user-authored
        context (#69630 review follow-up)."""
        from agent.session_persistence import _is_ephemeral_scaffolding
        from tests.agent.test_run_agent import _mock_response

        loop_agent.client.chat.completions.create.side_effect = [
            _dropped_tool_call_response(""),
            _mock_response(content="All checks pass. Approved.", finish_reason="stop"),
        ]

        with (
            patch.object(loop_agent, "_persist_session"),
            patch.object(loop_agent, "_save_trajectory"),
            patch.object(loop_agent, "_cleanup_task_resources"),
        ):
            result = loop_agent.run_conversation("review the PR")

        assert result["completed"] is True
        # The finalization pop strips the answered pair from the live list —
        # no flagged scaffolding may survive into the returned transcript.
        leftover = [
            m for m in result["messages"]
            if isinstance(m, dict) and m.get("_dropped_toolcall_nudge")
        ]
        assert not leftover, (
            "The re-prompt pair must be stripped at finalization, not kept "
            "in the returned transcript."
        )
        # And the persistence filter must classify the flag as ephemeral so a
        # mid-turn flush can never write the pair to the durable store either.
        assert _is_ephemeral_scaffolding(
            {"role": "user", "content": "nudge", "_dropped_toolcall_nudge": True}
        ), (
            "_dropped_toolcall_nudge messages must be classified as "
            "ephemeral scaffolding so they are never persisted."
        )
