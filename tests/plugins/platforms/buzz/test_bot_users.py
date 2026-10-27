"""Buzz sender classification feeds the existing gateway loop budget (#145270)."""

from unittest.mock import AsyncMock

import pytest

from gateway.config import PlatformConfig
from gateway.run import GatewayRunner
from tests.gateway._plugin_adapter_loader import load_plugin_adapter

buzz = load_plugin_adapter("buzz")
BOT = "b" * 64
HUMAN = "a" * 64


@pytest.fixture
def adapter(monkeypatch):
    for name in ("BUZZ_BOT_USERS", "BUZZ_ALLOWED_USERS", "BUZZ_ALLOW_ALL_USERS",
                 "GATEWAY_ALLOWED_USERS", "GATEWAY_ALLOW_ALL_USERS"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(buzz, "_DEFAULT_CREDENTIALS_DIR", "/nonexistent")

    def make(extra):
        instance = buzz.BuzzAdapter(PlatformConfig(enabled=True, extra=extra))
        instance._message_handler = AsyncMock()
        instance.handle_message = AsyncMock()
        instance.send_reaction = AsyncMock()
        return instance

    return make


async def dispatch(instance, sender=BOT, chat="test-chat", message="message"):
    await instance._dispatch_message(
        text="hello", chat_id=chat, chat_type="group", user_id=sender,
        user_name="sender", message_id=message, created_at=1000,
    )
    return instance.handle_message.call_args.args[0].source


@pytest.mark.asyncio
async def test_bot_users_marks_only_listed_senders(adapter):
    instance = adapter({"bot_users": [BOT]})
    assert (await dispatch(instance)).is_bot is True
    assert (await dispatch(instance, HUMAN)).is_bot is False
    assert (await dispatch(adapter({}))).is_bot is False


@pytest.mark.asyncio
@pytest.mark.parametrize("encoding", ["hex", "npub", "env"])
async def test_bot_identity_normalization(adapter, monkeypatch, encoding):
    value = buzz.hex_to_npub(BOT) if encoding != "hex" else BOT.upper()
    extra = {"bot_users": [value]}
    if encoding == "env":
        monkeypatch.setenv("BUZZ_BOT_USERS", f" {value}, {HUMAN} ")
        extra = {}
    assert (await dispatch(adapter(extra))).is_bot is True


@pytest.mark.asyncio
async def test_multiplex_bot_users_do_not_borrow_process_env(adapter, monkeypatch):
    from agent.secret_scope import reset_secret_scope, set_multiplex_active, set_secret_scope

    monkeypatch.setenv("BUZZ_BOT_USERS", BOT)
    set_multiplex_active(True)
    token = set_secret_scope({})
    try:
        assert (await dispatch(adapter({}))).is_bot is False
        assert (await dispatch(adapter({"bot_users": [BOT]}))).is_bot is True
    finally:
        reset_secret_scope(token)
        set_multiplex_active(False)


@pytest.mark.asyncio
async def test_buzz_bot_budget_preserves_authorization(adapter):
    from gateway.bot_loop_guard import BotLoopGuard, BotLoopGuardSettings
    from gateway.config import GatewayConfig

    instance = adapter({"allowed_users": [BOT, HUMAN], "bot_users": [BOT]})
    runner = object.__new__(GatewayRunner)
    runner.config = GatewayConfig()
    runner.pairing_store = None
    runner.adapters = {instance.platform: instance}
    runner._bot_loop_guard = BotLoopGuard(
        settings=lambda: BotLoopGuardSettings(max_events=6), clock=lambda: 1000.0,
    )
    for i in range(6):
        source = await dispatch(instance, message=str(i))
        assert runner._is_user_authorized(source) is True
        assert runner._admit_bot_message(source) is True
    source = await dispatch(instance, message="overflow")
    assert runner._admit_bot_message(source) is False
    assert runner._is_user_authorized(source) is False
    assert runner._is_user_authorized(await dispatch(instance, HUMAN)) is True
    assert runner._admit_bot_message(await dispatch(instance, chat="other-chat")) is True

    # Classification must not grant access to a sender absent from the allowlist.
    denied = adapter({"allowed_users": [HUMAN], "bot_users": [BOT]})
    runner.adapters = {denied.platform: denied}
    source = await dispatch(denied, chat="unapproved-chat")
    assert source.is_bot is True
    assert runner._is_user_authorized(source) is False
