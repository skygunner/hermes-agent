"""launchd plist refresh must not report success when launchd never registered the service (#12866)."""
import subprocess
from unittest.mock import MagicMock

from hermes_cli import gateway as gw
from hermes_cli import gateway_service_owner

def _stale_plist(tmp_path, monkeypatch, *, registered: bool):
    plist_path = tmp_path / "com.hermes.plist"
    plist_path.write_text("<old/>", encoding="utf-8")
    monkeypatch.setattr(gw, "get_launchd_plist_path", lambda: plist_path)
    monkeypatch.setattr(gw, "launchd_plist_is_current", lambda: False)
    monkeypatch.setattr(gw, "generate_launchd_plist", lambda: "<new/>")
    monkeypatch.setattr(gateway_service_owner, "refuse_temp_home_service_write", lambda *a: False)
    monkeypatch.setattr(gw, "get_launchd_label", lambda: "com.hermes.agent")
    monkeypatch.setattr(gw, "_launchd_domain", lambda: "gui/501")
    monkeypatch.setattr(gw, "_append_launchd_reload_log", lambda msg: None)
    monkeypatch.setattr("gateway.status.get_running_pid", lambda: None)
    monkeypatch.setattr(gw, "_retry_launchctl_bootstrap_until_registered", lambda *a, **k: registered)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: MagicMock(returncode=0))

def test_refresh_reports_registration_outcome(tmp_path, monkeypatch, capsys):
    _stale_plist(tmp_path, monkeypatch, registered=False)
    assert gw.refresh_launchd_plist_if_needed() is False
    assert "Updated" not in capsys.readouterr().out

    _stale_plist(tmp_path, monkeypatch, registered=True)
    assert gw.refresh_launchd_plist_if_needed() is True
    assert "Updated" in capsys.readouterr().out


def test_a_binary_plist_is_not_current(tmp_path, monkeypatch):
    """Regression #142999: launchd may hand us a bplist00 plist, and reading it as text raised
    UnicodeDecodeError out of the refresh path."""
    from hermes_cli.gateway_launchd import launchd_plist_is_current

    plist_path = tmp_path / 'com.hermes.agent.plist'
    plist_path.write_bytes(b'bplist00\xd4\xd4\xd4\xd4')
    monkeypatch.setattr('hermes_cli.gateway_launchd._gw', lambda: MagicMock(
        get_launchd_plist_path=lambda: plist_path,
        generate_launchd_plist=lambda: '<?xml version="1.0"?><plist/>'))
    assert launchd_plist_is_current() is False
