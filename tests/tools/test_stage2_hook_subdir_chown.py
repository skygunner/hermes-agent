"""Regression tests for the Docker stage2 data-volume ownership gate (#41699).

`usermod -u` already re-owns the hermes home directory and `groupmod` re-owns
nothing, so gating the whole data-volume repair on the top-level `$HERMES_HOME`
UID left stale GIDs and root-owned subtrees behind after a `HERMES_UID` /
`HERMES_GID` remap. Each hermes-owned subtree must be probed on its own, for
both UID and GID.

The repair is POSIX shell, so the block is sliced out of the hook and executed
under `sh` with the OS ownership probes stubbed — a test cannot create foreign
uids without root, and rebuilding the image is the docker suite's job.
"""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
STAGE2_HOOK = REPO_ROOT / "docker" / "stage2-hook.sh"

HERMES_UID = "4242"
HERMES_GID = "4343"


@pytest.fixture(scope="module")
def repair_block() -> str:
    if not STAGE2_HOOK.exists():
        pytest.skip("docker/stage2-hook.sh not present in this checkout")
    text = STAGE2_HOOK.read_text()
    start = text.index("actual_hermes_uid=$(id -u hermes)")
    end = text.index("\n# --- Immutable install tree ---", start)
    return text[start:end]


def _run_repair(
    block: str,
    hermes_home: Path,
    log_path: Path,
    *,
    top_owner: str,
    mismatched_tree: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    """Run the real repair block with stubbed ownership probes.

    ``stat`` answers both the old ``%u`` and the UID:GID query, so the same
    harness drives the unfixed hook (which must fail these tests) and the fix.
    """
    shell = shutil.which("sh")
    if shell is None:
        pytest.skip("sh not available")
    harness = (
        "set -eu\n"
        f'HERMES_HOME="{hermes_home}"\n'
        f'FAKE_HERMES_UID="{HERMES_UID}"\n'
        f'FAKE_HERMES_GID="{HERMES_GID}"\n'
        f'TOP_OWNER="{top_owner}"\n'
        f'MISMATCHED_TREE="{mismatched_tree or ""}"\n'
        f'CHOWN_LOG="{log_path}"\n'
        "id() {\n"
        '    case "$1" in\n'
        "        -u) printf '%s\\n' \"$FAKE_HERMES_UID\" ;;\n"
        "        -g) printf '%s\\n' \"$FAKE_HERMES_GID\" ;;\n"
        "        *) printf '%s\\n' 0 ;;\n"
        "    esac\n"
        "}\n"
        "stat() {\n"
        '    case "$2" in\n'
        "        *:*) printf '%s\\n' \"$TOP_OWNER\" ;;\n"
        '        *) printf \'%s\\n\' "${TOP_OWNER%%:*}" ;;\n'
        "    esac\n"
        "}\n"
        "find() {\n"
        '    if [ -n "$MISMATCHED_TREE" ] && [ "$1" = "$MISMATCHED_TREE" ]; then\n'
        '        printf \'%s\\n\' "$1/.mismatched"\n'
        "    fi\n"
        "}\n"
        'chown() { printf \'%s\\n\' "$*" >> "$CHOWN_LOG"; }\n'
        f"{block}\n"
    )
    return subprocess.run([shell, "-c", harness], capture_output=True, text=True, check=False)


def _chowned(log_path: Path) -> list[str]:
    if not log_path.exists():
        return []
    return log_path.read_text().splitlines()


def test_stale_gid_alone_repairs_top_level(
    repair_block: str, tmp_path: Path,
) -> None:
    """A UID-correct but GID-stale $HERMES_HOME must still be repaired."""
    home = tmp_path / "home"
    home.mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_repair(repair_block, home, log_path, top_owner=f"{HERMES_UID}:9999")

    assert proc.returncode == 0, proc.stderr
    assert _chowned(log_path) == [f"hermes:hermes {home}"]


def test_mismatched_subdir_repaired_under_correct_top_level(
    repair_block: str, tmp_path: Path,
) -> None:
    """A subtree must be repaired even when $HERMES_HOME itself is correct.

    This is the reported failure: after the remap the top-level directory
    already matched, so the per-subtree repair never ran and `sessions/` kept
    its stale ownership.
    """
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    (home / "logs").mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_repair(
        repair_block, home, log_path,
        top_owner=f"{HERMES_UID}:{HERMES_GID}",
        mismatched_tree=home / "sessions",
    )

    assert proc.returncode == 0, proc.stderr
    assert _chowned(log_path) == [f"-R hermes:hermes {home / 'sessions'}"]


def test_correctly_owned_volume_is_not_walked(
    repair_block: str, tmp_path: Path,
) -> None:
    """A volume that already matches hermes UID:GID must not be chowned."""
    home = tmp_path / "home"
    (home / "sessions").mkdir(parents=True)
    (home / "logs").mkdir()
    log_path = tmp_path / "chown.log"

    proc = _run_repair(repair_block, home, log_path, top_owner=f"{HERMES_UID}:{HERMES_GID}")

    assert proc.returncode == 0, proc.stderr
    assert _chowned(log_path) == []
    assert "Fixing ownership" not in proc.stdout