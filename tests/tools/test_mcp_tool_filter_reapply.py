"""Regression for #145265: ``tools.include``/``exclude`` selects which of a server's tools a profile
registers, so it is not part of the connection's identity. Editing it on disk while the connection is
live must re-apply the filter — widening it registers the newly selected tools, narrowing it
deregisters the dropped ones — instead of detaching the profile's whole overlay and keeping the
connection up, which is the state nobody logged."""

from __future__ import annotations

import logging
from types import SimpleNamespace

import pytest

_TOOLSET = "mcp-x"
_UTILITIES = {"mcp__x__list_resources", "mcp__x__read_resource", "mcp__x__list_prompts", "mcp__x__get_prompt"}


def _tool(name):
    return SimpleNamespace(name=name, description="d", inputSchema={"type": "object", "properties": {}},
                           annotations=None)


def _server(name, tools, cfg, session=None):
    return SimpleNamespace(name=name, session=session if session is not None else object(), _config=dict(cfg),
                           _tools=[_tool(t) for t in tools], tool_timeout=30, initialize_result=None,
                           _registered_tool_names=[], _sampling=None)


def _connect(tools=("t", "t2"), cfg=None, session=None):
    """Publish a live connection and register its tools exactly as discovery does."""
    from tools import mcp_tool_discovery as disc
    from tools import mcp_tool_registration as reg

    cfg = dict(cfg or {"url": "https://mcp.example/x"})
    server = _server("x", tools, cfg, session=session)
    disc._adopt_server("x", server)
    server._registered_tool_names = reg._register_server_tools("x", server, cfg)
    return server


def _utility_session():
    return SimpleNamespace(**{name: (lambda *args, **kwargs: None) for name in
                              ("list_resources", "read_resource", "list_prompts", "get_prompt")})


@pytest.fixture
def mcp_profile(tmp_path, monkeypatch):
    """A served profile's MCP process state: clean ledgers, its own registry overlay, restored after."""
    import tools.mcp_tool as core
    from hermes_constants import hermes_home_key, reset_hermes_home_override, set_hermes_home_override
    from tools.registry import registry

    home = tmp_path / "profiles" / "a"
    home.mkdir(parents=True)
    monkeypatch.setattr("agent.secret_scope.is_multiplex_active", lambda: True)
    token = set_hermes_home_override(home)
    scope = hermes_home_key(home)
    ledgers = ("_servers", "_server_scope_keys", "_server_tool_scopes", "_server_connecting",
               "_server_connect_errors", "_server_connect_retry_after", "_server_connect_failures",
               "_server_error_counts", "_server_breaker_opened_at", "_lazy_server_configs",
               "_lazy_server_fingerprints", "_lazy_server_tool_names", "_mcp_tool_server_names",
               "_orphaned_adopters", "_parallel_safe_servers", "_server_trust_levels",
               "_tool_read_only_hints", "_server_cost_policies")
    saved = {name: type(getattr(core, name))(getattr(core, name)) for name in ledgers}
    for name in ledgers:
        getattr(core, name).clear()
    yield scope
    for tool_name in list(registry.get_tool_names_for_toolset(_TOOLSET)):
        registry.deregister(tool_name, scope=scope)
    for name in ledgers:
        getattr(core, name).clear()
        getattr(core, name).update(saved[name])
    reset_hermes_home_override(token)


def test_widening_include_reapplies_to_the_live_connection(mcp_profile, monkeypatch):
    """The reported repro: add a name to ``tools.include`` while the gateway is up and let the
    housekeeping tick run. The connection stays, and the newly selected tool appears."""
    from tools import mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_scope import _server_key
    from tools.registry import registry

    server = _connect(cfg={"url": "https://mcp.example/x", "tools": {"include": ["t"]}})
    assert registry.get_tool_names_for_toolset(_TOOLSET) == ["mcp__x__t"]

    widened = {"url": "https://mcp.example/x", "tools": {"include": ["t", "t2"]}}
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"x": widened})
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: [])

    assert disc.reconcile_mcp_servers_with_config() == {"removed": [], "added": [], "pending": []}
    assert registry.get_tool_names_for_toolset(_TOOLSET) == ["mcp__x__t", "mcp__x__t2"]
    assert core._servers[_server_key("x")] is server, "a filter edit must not cost the connection"


def test_narrowing_include_deregisters_the_dropped_tool(mcp_profile, monkeypatch):
    """The mirror case: narrow ``tools.include`` and the dropped tool stops answering immediately,
    without waiting for a restart."""
    from tools import mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools.mcp_tool_scope import _server_key
    from tools.registry import registry

    server = _connect(cfg={"url": "https://mcp.example/x", "tools": {"include": ["t", "t2"]}})
    assert registry.get_tool_names_for_toolset(_TOOLSET) == ["mcp__x__t", "mcp__x__t2"]

    narrowed = {"url": "https://mcp.example/x", "tools": {"include": ["t"]}}
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"x": narrowed})
    monkeypatch.setattr(_config, "_filter_suspicious_mcp_servers", lambda servers: servers)
    monkeypatch.setattr(core, "_ensure_mcp_sdk", lambda: True)
    monkeypatch.setattr(disc, "_run_discovery_pass", lambda servers: None)

    disc.register_mcp_servers({"x": narrowed})
    assert registry.get_tool_names_for_toolset(_TOOLSET) == ["mcp__x__t"]
    assert core._servers[_server_key("x")] is server


def test_filter_edit_never_touches_generated_utilities(mcp_profile, monkeypatch):
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "tools": {"include": ["t", "t2"]}}
    _connect(cfg=cfg, session=_utility_session())
    assert _UTILITIES <= set(registry.get_tool_names_for_toolset(_TOOLSET))

    narrowed = {**cfg, "tools": {"include": ["t"]}}
    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"x": narrowed})
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: [])

    disc.reconcile_mcp_servers_with_config()
    assert set(registry.get_tool_names_for_toolset(_TOOLSET)) == _UTILITIES | {"mcp__x__t"}


def test_connected_connection_that_lost_its_tools_warns(mcp_profile, monkeypatch, caplog):
    """A connection that is up while this profile's registry holds none of its tools used to be
    completely silent: it must name the server, what it lost, and the repair."""
    from tools import mcp_tool as core
    from tools import mcp_tool_config as _config
    from tools import mcp_tool_discovery as disc
    from tools.registry import registry

    cfg = {"url": "https://mcp.example/x", "tools": {"include": ["t"]}}
    _connect(cfg=cfg)
    registry.deregister("mcp__x__t", scope=core._mcp_registry_scope())
    assert registry.get_tool_names_for_toolset(_TOOLSET) == []

    monkeypatch.setattr(_config, "_load_mcp_config", lambda: {"x": dict(cfg)})
    monkeypatch.setattr(disc, "discover_mcp_tools", lambda *a, **k: [])
    caplog.set_level(logging.WARNING, logger="tools.mcp_tool")

    disc.reconcile_mcp_servers_with_config()
    messages = [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING]
    assert any("holds none of the 1 tool(s)" in message and "mcp__x__t" in message and "reload-mcp" in message
               for message in messages), messages
