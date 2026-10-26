"""Regression for #145151: auto voice replies synthesize inside the served profile's scope.

With ``gateway.multiplex_profiles`` on, both auto voice reply lanes run after the turn's secret
scope was reset: the runner's text reply for ``/voice`` chats and the base adapter's auto-TTS for
voice input. The Gemini provider's optional ``GEMINI_BASE_URL`` read then raised
``UnscopedSecretError``, so every voice reply was dropped while the text reply arrived.
"""

import asyncio
import base64
import struct
from unittest.mock import AsyncMock, patch

import pytest

from agent import secret_scope
from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.platforms.base import BasePlatformAdapter, SendResult
from gateway.platforms.event import MessageEvent, MessageType
from gateway.run import GatewayRunner
from gateway.session import SessionSource, build_session_key
from tools.tts_tool_providers import DEFAULT_GEMINI_TTS_BASE_URL

_PROFILE_KEY = "profile-scoped-key"


class _GeminiResponse:
    status_code = 200

    def json(self):
        pcm = base64.b64encode(struct.pack("<h", 0) * 64).decode()
        return {"candidates": [{"content": {"parts": [{"inlineData": {"data": pcm}}]}}]}


class _DummyAdapter(BasePlatformAdapter):
    def __init__(self, platform: Platform):
        super().__init__(PlatformConfig(enabled=True, token="fake-token"), platform)

    async def connect(self, *, is_reconnect: bool = False) -> bool:
        return True

    async def disconnect(self) -> None:
        return None

    async def send(self, chat_id, content, reply_to=None, metadata=None) -> SendResult:
        return SendResult(success=True, message_id="1")

    async def send_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def stop_typing(self, chat_id: str, metadata=None) -> None:
        return None

    async def get_chat_info(self, chat_id: str):
        return {"id": chat_id}


@pytest.fixture
def multiplexed_profile(tmp_path, monkeypatch):
    home = tmp_path / ".hermes"
    home.mkdir()
    (home / "config.yaml").write_text("tts:\n  provider: gemini\n", encoding="utf-8")
    (home / ".env").write_text(f"GEMINI_API_KEY={_PROFILE_KEY}\n", encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(home))
    for name in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GEMINI_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    secret_scope.set_multiplex_active(True)
    try:
        yield home
    finally:
        secret_scope.set_multiplex_active(False)


def _make_runner() -> GatewayRunner:
    runner = GatewayRunner.__new__(GatewayRunner)
    runner.config = GatewayConfig(multiplex_profiles=True)
    runner._voice_mode = {}
    runner.adapters = {}
    return runner


def _make_source(message_type: MessageType = MessageType.TEXT) -> SessionSource:
    return SessionSource(platform=Platform.TELEGRAM, chat_id="123", user_id="u1", user_name="User")


def _make_event(message_type: MessageType = MessageType.TEXT) -> MessageEvent:
    return MessageEvent(
        text="trigger", source=_make_source(), message_type=message_type, message_id="456"
    )


def _hold_typing():
    async def hold(*_args, **_kwargs):
        await asyncio.Event().wait()

    return hold


@pytest.mark.asyncio
async def test_auto_voice_reply_synthesizes_on_the_served_profile(multiplexed_profile, monkeypatch):
    posted = []

    def fake_post(url, payload, headers):
        posted.append((url, headers))
        return _GeminiResponse()

    monkeypatch.setattr("tools.tts_tool_providers._post_json", fake_post)

    runner = _make_runner()
    adapter = _DummyAdapter(Platform.TELEGRAM)
    adapter.gateway_runner = runner
    adapter.send_voice = AsyncMock()
    runner.adapters[Platform.TELEGRAM] = adapter

    event = _make_event()
    runner._voice_mode[runner._voice_key_for_source(event.source)] = "all"
    await runner._send_voice_reply(event, "hello there")

    assert posted, "runner voice reply never reached the TTS provider"
    assert posted[0][0].startswith(DEFAULT_GEMINI_TTS_BASE_URL)
    assert posted[0][1]["x-goog-api-key"] == _PROFILE_KEY
    adapter.send_voice.assert_awaited_once()

    posted.clear()
    adapter._keep_typing = _hold_typing()
    adapter._should_auto_tts_for_chat = lambda _chat_id: True
    adapter.play_tts = AsyncMock(return_value=SendResult(success=True, message_id="tts-1"))
    adapter.set_message_handler(lambda _event: asyncio.sleep(0, result="reply text"))
    voice_event = _make_event(MessageType.VOICE)
    await adapter._process_message_background(voice_event, build_session_key(voice_event.source))

    assert posted, "base-adapter auto-TTS never reached the TTS provider"
    assert posted[0][0].startswith(DEFAULT_GEMINI_TTS_BASE_URL)
    assert posted[0][1]["x-goog-api-key"] == _PROFILE_KEY
    adapter.play_tts.assert_awaited_once()
