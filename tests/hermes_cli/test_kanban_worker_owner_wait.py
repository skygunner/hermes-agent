"""A Kanban worker spawned while `hermes update` drains the owner waits for it instead of failing.

The dispatcher keeps spawning workers during an update window; the owner answers ``draining``
until the update releases the install. Exiting 1 there books a crash against the card, and two
of those trip the breaker and park a card that never ran (the in-flight handoff cell). The wait
must never submit the claim twice, and a non-transient refusal still fails at once.

Regression for #145237: a cold profile's activation blocks the owner loop that answers the session
ticket and answers the websocket upgrade, so those connect-phase failures (nothing submitted yet)
are retried inside the same wait, while anything after ``kanban.run`` still fails at once.
"""

import asyncio
from contextlib import asynccontextmanager
from types import SimpleNamespace

import pytest
import websockets.asyncio.client

from hermes_cli import gateway_client, kanban_worker_client


def _owner(states, submitted):
    @asynccontextmanager
    async def connect():
        state = states.pop(0) if states else None
        if isinstance(state, Exception):
            raise state
        if state:
            # The exact refusal connect_gateway raises for a draining/starting owner.
            unavailable = getattr(gateway_client, "GatewayUnavailableError", None)
            if unavailable is None:  # pre-fix tree: the same message as a plain client error
                raise gateway_client.GatewayClientError(f"Gateway {state}: update_paused")
            raise unavailable(f"Gateway {state}: update_paused", state)

        class _Client:
            events = asyncio.Queue()

            async def rpc(self, method, **params):
                submitted.append(method)
                return {"session_id": "s1", "receipt": {"status": "terminal", "admission_id": "a1"}}

        yield _Client()

    return connect


@pytest.mark.asyncio
async def test_worker_waits_out_a_draining_owner_and_submits_once(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["draining", "starting"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_worker_fails_at_once_when_the_owner_is_not_transiently_down(monkeypatch, tmp_path):
    submitted = []
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner(["inaccessible"], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert submitted == []


@pytest.mark.asyncio
async def test_worker_retries_a_cold_owner_connect_failure_and_submits_once(monkeypatch, tmp_path):
    """A ticket timeout and a failed upgrade both happen before any claim is sent."""
    submitted = []
    connect_error = getattr(gateway_client, "GatewayConnectError", gateway_client.GatewayClientError)
    monkeypatch.setattr(gateway_client, "connect_gateway", _owner([
        connect_error("Gateway ticket request failed: timed out"),
        connect_error("Gateway connection failed; no local fallback")], submitted))
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    from gateway import session_kanban
    monkeypatch.setattr(session_kanban, "worker_result", lambda *_: {"exit_code": 0, "last_output": "card closed"})
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    assert await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db")) == 0
    assert submitted == ["kanban.run"]


@pytest.mark.asyncio
async def test_worker_does_not_retry_a_failure_after_the_claim_was_sent(monkeypatch, tmp_path):
    """The upgrade succeeded, so the claim may already be with the owner: never send it again."""
    submitted = []

    class _Client:
        events = asyncio.Queue()

        async def rpc(self, method, **params):
            submitted.append(method)
            raise gateway_client.GatewayClientError("Gateway disconnected; turn outcome is unknown.")

    @asynccontextmanager
    async def connect():
        yield _Client()

    monkeypatch.setattr(gateway_client, "connect_gateway", connect)
    monkeypatch.setattr(kanban_worker_client.asyncio, "sleep", _no_sleep)
    monkeypatch.delenv("HERMES_TUI_GATEWAY_URL", raising=False)

    with pytest.raises(gateway_client.GatewayClientError):
        await kanban_worker_client.run({"task_id": "t1"}, str(tmp_path / "kanban.db"))
    assert submitted == ["kanban.run"]


class _TimedOutUpgrade:
    """An upgrade that outlasts ``open_timeout``, for either way the dial is awaited."""

    def __await__(self):
        return self._fail().__await__()

    async def __aenter__(self):
        return await self._fail()

    async def __aexit__(self, *_exc):
        return None

    async def _fail(self):
        raise TimeoutError("timed out during opening handshake")


class _SilentSocket:
    """A connected owner that never answers, usable as ``async with connect(...)`` or awaited."""

    subprotocol = gateway_client.GATEWAY_WS_PROTOCOL
    parked = asyncio.Event()

    async def send(self, _payload):
        return None

    async def close(self):
        return None

    def __aiter__(self):
        return self

    async def __anext__(self):
        await self.parked.wait()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_exc):
        return None

    async def _awaited(self):
        return self

    def __await__(self):
        return self._awaited().__await__()


def _endpoint():
    return SimpleNamespace(profile_id="p1", instance_id="i1", control_home=None,
                           api_origin="http://127.0.0.1:8765")


def _ticket_grant(endpoint):
    import json
    return json.dumps({"ok": True, "protocol": 1, "id": 1, "result": {
        "profile_id": endpoint.profile_id, "instance_id": endpoint.instance_id,
        "runtime_protocol": 1, "ticket": "t1"}}).encode() + b"\n"


def test_a_ticket_request_that_times_out_is_a_typed_connect_failure(monkeypatch):
    """What ``run()`` retries must be a ``GatewayConnectError``, never a string-matched message."""
    def timed_out(*_args, **_kwargs):
        raise TimeoutError("timed out")

    monkeypatch.setattr(gateway_client, "private_control_query", timed_out)

    with pytest.raises(gateway_client.GatewayConnectError):
        gateway_client._session_ticket(_endpoint().control_home or "/nonexistent", _endpoint())


@pytest.mark.asyncio
async def test_a_slow_websocket_upgrade_is_a_typed_connect_failure(monkeypatch):
    endpoint = _endpoint()
    monkeypatch.setattr(gateway_client, "private_control_query",
                        lambda *_args, **_kwargs: _ticket_grant(endpoint))
    monkeypatch.setattr(websockets.asyncio.client, "connect", lambda *_args, **_kwargs: _TimedOutUpgrade())

    with pytest.raises(gateway_client.GatewayConnectError):
        async with gateway_client.connect_gateway(endpoint):
            pass


@pytest.mark.asyncio
async def test_a_lost_answer_after_the_upgrade_is_not_a_connect_failure(monkeypatch):
    """Past the upgrade a turn may be running: its failure keeps the ambiguous report, unretried."""
    endpoint = _endpoint()
    monkeypatch.setattr(gateway_client, "private_control_query",
                        lambda *_args, **_kwargs: _ticket_grant(endpoint))
    monkeypatch.setattr(websockets.asyncio.client, "connect", lambda *_args, **_kwargs: _SilentSocket())

    with pytest.raises(gateway_client.GatewayClientError) as caught:
        async with gateway_client.connect_gateway(endpoint) as client:
            await client.rpc("kanban.run", _timeout=0.05)
    assert not isinstance(caught.value, getattr(gateway_client, "GatewayConnectError", ()))


async def _no_sleep(_seconds):
    return None
