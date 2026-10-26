"""`hermes debug` must not report a provider authenticated from the credential pool as "not set".

``hermes auth add <provider>`` and every OAuth login (Nous Portal, Codex, …) keep the credential in
``auth.json``'s ``credential_pool``, not in an env var, so the dump's env-only API-key scan called a
working login "not set" and sent support chasing a nonexistent auth problem (#20675).
"""

import json
from types import SimpleNamespace


def _api_key_line(out: str, label: str) -> str:
    for line in out.splitlines():
        if line.strip().startswith(f"{label} "):
            return line
    raise AssertionError(f"no '{label}' api_keys line in dump output:\n{out}")


def _write_pool_token(home, provider: str, token: str) -> None:
    (home / "auth.json").write_text(json.dumps({"credential_pool": {provider: [{"id": "abc123", "access_token": token}]}}))


def test_dump_reports_oauth_pool_token_as_set(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    _write_pool_token(home, "nous", "nous-portal-access-token")

    dump.run_dump(SimpleNamespace(show_keys=False))

    assert "set (auth pool)" in _api_key_line(capsys.readouterr().out, "nous")


def test_dump_reports_api_key_pool_token_as_set(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    _write_pool_token(home, "openrouter", "sk-or-pooled")

    dump.run_dump(SimpleNamespace(show_keys=False))

    assert "set (auth pool)" in _api_key_line(capsys.readouterr().out, "openrouter")


def test_dump_keeps_provider_without_any_credential_not_set(monkeypatch, capsys, tmp_path):
    from hermes_cli import dump
    from hermes_cli.config import get_hermes_home

    monkeypatch.setattr(dump, "get_project_root", lambda: tmp_path / "noproject")
    monkeypatch.delenv("NOUS_API_KEY", raising=False)
    home = get_hermes_home()
    home.mkdir(parents=True, exist_ok=True)
    _write_pool_token(home, "openrouter", "sk-or-pooled")

    dump.run_dump(SimpleNamespace(show_keys=False))

    assert "not set" in _api_key_line(capsys.readouterr().out, "nous")
