"""Unit tests for gateway.slash_access — per-platform slash command access control.

Tests the pure policy resolver (no gateway plumbing). Integration tests that
exercise the dispatch site live in test_slash_access_dispatch.py.

Regression for #144428: WhatsApp delivers one operator as a bare number, a phone JID or a LID
while ``allow_admin_from`` was compared by exact string, so a phone-listed admin fell to the
non-admin floor whenever the message arrived as a LID.
"""
from __future__ import annotations

import json

import pytest

from gateway.config import GatewayConfig, Platform, PlatformConfig
from gateway.session import SessionSource
from gateway.slash_access import (
    policy_for_source,
    policy_from_extra,
)
from hermes_constants import get_hermes_home


# ---------------------------------------------------------------------------
# policy_from_extra — input normalization + scope resolution
# ---------------------------------------------------------------------------


class TestPolicyFromExtra:

    def test_disabled_policy_treats_anyone_as_admin(self):
        # When gating is off, downstream code uses is_admin/can_run uniformly.
        # Both must short-circuit to True so existing behavior is preserved.
        p = policy_from_extra({}, "dm")
        assert p.is_admin("anyone") is True
        assert p.can_run("anyone", "stop") is True


    def test_id_coercion_ints_become_strings(self):
        # YAML often loads numeric IDs as ints; we stringify on ingest.
        p = policy_from_extra({"allow_admin_from": [12345, 67890]}, "dm")
        assert p.admin_user_ids == frozenset({"12345", "67890"})
        assert p.is_admin("12345") is True
        assert p.is_admin(12345) is True  # is_admin also stringifies


    def test_command_coercion_strips_leading_slash_and_lowercases(self):
        p = policy_from_extra(
            {
                "allow_admin_from": ["111"],
                "user_allowed_commands": ["/Status", "MODEL", "/help"],
            },
            "dm",
        )
        assert p.user_allowed_commands == frozenset({"status", "model", "help"})


    def test_dm_admin_does_not_imply_group_admin(self):
        # Admin lists are scope-specific. DM admin must not auto-promote in groups.
        extra = {"allow_admin_from": ["111"]}
        dm = policy_from_extra(extra, "dm")
        assert dm.is_admin("111") is True
        # Group has no admin list set → gating disabled in groups → "111"
        # gets unrestricted access, but that's the backward-compat fallback,
        # not implicit admin promotion. The distinction matters when the
        # group DOES have an admin list set:
        extra2 = {
            "allow_admin_from": ["111"],
            "group_allow_admin_from": ["222"],
        }
        gp2 = policy_from_extra(extra2, "group")
        assert gp2.is_admin("111") is False
        assert gp2.is_admin("222") is True


# ---------------------------------------------------------------------------
# policy_for_source — wires GatewayConfig + SessionSource together
# ---------------------------------------------------------------------------


class TestPolicyForSource:


    def test_dm_chat_type_resolves_to_dm_scope(self):
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={
                        "allow_admin_from": ["111"],
                        "user_allowed_commands": ["status"],
                        "group_allow_admin_from": ["222"],
                        "group_user_allowed_commands": ["help"],
                    },
                )
            }
        )
        dm_src = SessionSource(
            platform=Platform.DISCORD, chat_id="A", chat_type="dm", user_id="111"
        )
        p = policy_for_source(cfg, dm_src)
        assert p.is_admin("111") is True
        assert p.can_run("999", "status") is True
        assert p.can_run("999", "help") is True  # always-allowed floor
        assert p.can_run("999", "kanban") is False


    def test_no_admin_list_for_dm_means_unrestricted_in_dm(self):
        # Group has admin list, DM does not → DM gating disabled, group active.
        cfg = GatewayConfig(
            platforms={
                Platform.DISCORD: PlatformConfig(
                    enabled=True,
                    extra={"group_allow_admin_from": ["222"]},
                )
            }
        )
        dm_src = SessionSource(
            platform=Platform.DISCORD, chat_id="A", chat_type="dm", user_id="999"
        )
        grp_src = SessionSource(
            platform=Platform.DISCORD, chat_id="G", chat_type="group", user_id="999"
        )
        dm_p = policy_for_source(cfg, dm_src)
        grp_p = policy_for_source(cfg, grp_src)
        assert dm_p.enabled is False
        assert dm_p.can_run("999", "stop") is True  # backward compat
        assert grp_p.enabled is True
        assert grp_p.can_run("999", "stop") is False  # gated


@pytest.mark.parametrize(
    ("extra", "chat_type", "admin", "gated"),
    [
        # DM-only gated: blank used to fall into the ungated group scope (#116282).
        ({"allow_admin_from": ["111"]}, "", "111", True),
        ({"allow_admin_from": ["111"]}, None, "111", True),
        # Mirror image and tie keep the historical group scope; ungated stays ungated.
        ({"group_allow_admin_from": ["222"]}, "", "222", True),
        ({"allow_admin_from": ["111"], "group_allow_admin_from": ["222"]}, None, "222", True),
        ({}, "", None, False),
    ],
)
def test_blank_chat_type_resolves_to_the_gated_scope(extra, chat_type, admin, gated):
    """A blank/None chat_type (relay frames may carry ""/null, restored rows keep a
    stored empty value) must resolve to whichever scope is gated, never to an
    ungated allow-everything scope (#116282)."""
    cfg = GatewayConfig(platforms={Platform.DISCORD: PlatformConfig(enabled=True, extra=extra)})
    src = SessionSource(platform=Platform.DISCORD, chat_id="A", chat_type=chat_type, user_id="999")
    p = policy_for_source(cfg, src)
    assert p.enabled is gated
    assert p.can_run("999", "stop") is (not gated)
    if admin is not None:
        assert p.is_admin(admin) is True


# ---------------------------------------------------------------------------
# WhatsApp identity aliases — allow_admin_from matches every form of one operator
# ---------------------------------------------------------------------------

_PHONE = "351912345678"
_LID = "77214955630717"


def _write_lid_mapping():
    """Mirror what the JS bridge writes: phone→LID and LID→phone (reverse)."""
    session_dir = get_hermes_home() / "whatsapp" / "session"
    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / f"lid-mapping-{_PHONE}.json").write_text(json.dumps(_LID), encoding="utf-8")
    (session_dir / f"lid-mapping-{_LID}_reverse.json").write_text(
        json.dumps(_PHONE), encoding="utf-8"
    )


def _policy(extra, *, platform=Platform.WHATSAPP, chat_type="dm", user_id=f"{_LID}@lid"):
    cfg = GatewayConfig(platforms={platform: PlatformConfig(enabled=True, extra=extra)})
    src = SessionSource(platform=platform, chat_id="c1", chat_type=chat_type, user_id=user_id)
    return policy_for_source(cfg, src)


class TestWhatsAppAliasedAdmin:

    def test_admin_listed_as_phone_number_matches_lid_sender(self):
        _write_lid_mapping()
        p = _policy({"allow_admin_from": [_PHONE]})
        assert p.is_admin(f"{_LID}@lid") is True
        assert p.can_run(f"{_LID}@lid", "model") is True

    def test_admin_listed_as_phone_number_matches_phone_jid_sender(self):
        _write_lid_mapping()
        p = _policy({"allow_admin_from": [_PHONE]})
        assert p.is_admin(f"{_PHONE}@s.whatsapp.net") is True

    def test_group_admin_listed_as_phone_number_matches_lid_sender(self):
        _write_lid_mapping()
        p = _policy({"group_allow_admin_from": [_PHONE]}, chat_type="group")
        assert p.is_admin(f"{_LID}@lid") is True
        assert p.can_run(f"{_LID}@lid", "model") is True

    def test_whatsapp_cloud_uses_the_same_aliases(self):
        _write_lid_mapping()
        p = _policy({"allow_admin_from": [_PHONE]}, platform=Platform.WHATSAPP_CLOUD)
        assert p.is_admin(f"{_LID}@lid") is True

    def test_unrelated_sender_stays_on_the_non_admin_floor(self):
        _write_lid_mapping()
        p = _policy({"allow_admin_from": [_PHONE]})
        assert p.is_admin("999999999999999@lid") is False
        assert p.can_run("999999999999999@lid", "model") is False
        assert p.can_run("999999999999999@lid", "help") is True

    def test_other_platforms_keep_exact_matching(self):
        _write_lid_mapping()
        p = _policy({"allow_admin_from": [_PHONE]}, platform=Platform.DISCORD)
        assert p.is_admin(f"{_PHONE}@s.whatsapp.net") is False
        assert p.is_admin(_PHONE) is True
