"""Matrix exec-approval cards: reaction resolution and prompt lifetime (regression for #144083)."""

import time
import types

import pytest
from unittest.mock import AsyncMock, patch

from gateway.config import PlatformConfig
from gateway.platforms.base import ExecApprovalPrompt, SendResult


class TestMatrixExecApprovalReactions:


    @pytest.mark.asyncio
    async def test_reaction_resolves_pending_approval(self, monkeypatch):
        monkeypatch.setenv("MATRIX_ALLOWED_USERS", "@liizfq:liizfq.top")
        from plugins.platforms.matrix.adapter import MatrixAdapter, _MatrixApprovalPrompt

        adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
        # Resolve user_id so _is_self_sender doesn't defensively drop all traffic (#15763).
        adapter._user_id = "@bot:example.org"
        adapter._approval_prompts_by_event["$target"] = _MatrixApprovalPrompt(
            session_key="sess-1", chat_id="!room:example.org", message_id="$target"
        )
        adapter._approval_prompt_by_session["sess-1"] = "$target"

        content = {"m.relates_to": {"event_id": "$target", "key": "✅"}}
        event = types.SimpleNamespace(
            sender="@liizfq:liizfq.top",
            event_id="$react1",
            room_id="!room:example.org",
            content=content,
        )

        with patch("tools.approval.resolve_gateway_approval", return_value=1) as mock_resolve:
            await adapter._on_reaction(event)

        mock_resolve.assert_called_once_with("sess-1", "once")
        assert "$target" not in adapter._approval_prompts_by_event
        assert "sess-1" not in adapter._approval_prompt_by_session


async def _approval_card_lifetime() -> float:
    """Send an exec-approval card through the real path and return its lifetime, in seconds."""
    from plugins.platforms.matrix.adapter import MatrixAdapter

    adapter = MatrixAdapter(PlatformConfig(enabled=True, token="tok", extra={"homeserver": "https://matrix.example.org"}))
    adapter._client = object()
    adapter.send = AsyncMock(return_value=SendResult(success=True, message_id="$card"))
    adapter._send_reaction = AsyncMock(return_value="$react")
    await adapter._send_exec_approval_prompt(ExecApprovalPrompt(
        chat_id="!room:example.org", session_key="sess-1", text="run it",
        actions=[("OK", "once", "primary"), ("No", "deny", "danger")],
        command="ls", description="list", smart_denied=False))
    return adapter._approval_prompts_by_event["$card"].expires_at - time.monotonic()


@pytest.mark.asyncio
async def test_card_lifetime_follows_approvals_timeout(monkeypatch):
    """#144083: the reaction map must stay alive as long as the turn waits on approvals.timeout."""
    monkeypatch.delenv("MATRIX_APPROVAL_TIMEOUT_SECONDS", raising=False)
    monkeypatch.setattr("plugins.platforms.matrix.adapter.approval_timeout_seconds", lambda: 86400, raising=False)

    assert await _approval_card_lifetime() == pytest.approx(86400, abs=5)


@pytest.mark.asyncio
async def test_env_override_still_wins_over_approvals_timeout(monkeypatch):
    monkeypatch.setenv("MATRIX_APPROVAL_TIMEOUT_SECONDS", "120")
    monkeypatch.setattr("plugins.platforms.matrix.adapter.approval_timeout_seconds", lambda: 86400, raising=False)

    assert await _approval_card_lifetime() == pytest.approx(120, abs=5)
