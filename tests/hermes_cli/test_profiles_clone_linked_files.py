"""``--clone-all`` keeps a linked root file by resolving it where the SOURCE resolves it.

Regression for #142943. A Stow-managed profile keeps ``config.yaml`` / ``.env`` / ``SOUL.md`` as
relative links (``../dotfiles/hermes/<name>``, written for the profile folder's own location).
``copytree(symlinks=True)`` copies the link text verbatim into ``profiles/.<name>.staging-<pid>/``,
which sits at another depth, so materializing the link from the staging copy makes it point
somewhere else: the clone lost the file and silently fell back to built-in defaults, the
placeholder ``.env`` and the default persona.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from hermes_cli.profiles import create_profile

_LINKED_FILES = {
    ".env": "OPENAI_API_KEY=sk-model-key\n",
    "config.yaml": "model: gpt-5\nprovider: openai\n",
    "SOUL.md": "# Be concise.\n",
}


@pytest.fixture()
def profile_env(tmp_path, monkeypatch):
    """``Path.home()`` + ``HERMES_HOME`` inside ``tmp_path``, so ``_get_profiles_root()`` stays there."""
    monkeypatch.setattr(Path, "home", lambda: tmp_path)
    default_home = tmp_path / ".hermes"
    default_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(default_home))
    return tmp_path


def _stow_links(source_home: Path, dotfiles: Path) -> None:
    """Link every root file the way GNU Stow does: relative text, target beside ``source_home``'s
    own location (a profile folder that is itself a symlink resolves from its physical one)."""
    dotfiles.mkdir(parents=True)
    for name, content in _LINKED_FILES.items():
        (dotfiles / name).write_text(content, encoding="utf-8")
        (source_home / name).symlink_to(os.path.relpath(dotfiles / name, source_home))


@pytest.mark.require_symlinks
@pytest.mark.parametrize("name", sorted(_LINKED_FILES))
def test_clone_all_keeps_a_linked_root_file_of_the_default_profile(profile_env, name):
    default_home = profile_env / ".hermes"
    _stow_links(default_home, profile_env / "dotfiles" / "hermes")

    profile_dir = create_profile("coder", clone_all=True, no_alias=True)

    assert (profile_dir / name).read_text(encoding="utf-8-sig") == _LINKED_FILES[name]
    assert (default_home / name).is_symlink()  # the clone copies, never writes through the link


@pytest.mark.require_symlinks
@pytest.mark.parametrize("name", sorted(_LINKED_FILES))
def test_clone_all_keeps_a_linked_root_file_of_a_profile_folder_that_is_a_symlink(profile_env, name):
    """A profile kept elsewhere (profiles/stowed -> ../other-disk/stowed) resolves its links from
    its physical location; the staging copy sits under profiles/, so it cannot resolve them."""
    default_home = profile_env / ".hermes"
    (default_home / "profiles").mkdir()
    physical_home = profile_env / "other-disk" / "stowed"
    physical_home.mkdir(parents=True)
    (default_home / "profiles" / "stowed").symlink_to(physical_home, target_is_directory=True)
    _stow_links(physical_home, profile_env / "other-disk" / "dotfiles")

    profile_dir = create_profile("coder", clone_from="stowed", clone_all=True, no_alias=True)

    assert (profile_dir / name).read_text(encoding="utf-8-sig") == _LINKED_FILES[name]
