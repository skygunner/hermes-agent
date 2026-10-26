"""Auto voice reply synthesis for the platform adapters (#145151).

Sibling of ``gateway.platforms.base``: base imports this module, so the platform-aware output path
is imported late. Synthesis runs after the turn's profile scope was reset, so it binds the routed
profile's scope itself — the TTS provider reads profile-level env (``GEMINI_BASE_URL``) and the
provider key through ``get_secret``, which fails closed while the gateway multiplexes profiles.
"""

import asyncio
import contextlib
import json
import logging
from pathlib import Path
from typing import Any, Optional

from gateway.session import SessionSource

logger = logging.getLogger(__name__)


def _existing_paths(raw_paths: list) -> list[str]:
    return [str(path) for path in raw_paths if path and Path(path).exists()]


async def synthesize_auto_tts(
    adapter: Any, text_content: str, source: Optional[SessionSource]
) -> tuple[list[str], Optional[str]]:
    """Synthesize auto-TTS audio -> ``(existing_paths, requested_path)``; empty/None on failure
    (logged, never raised). Path built platform-aware HERE: HERMES_SESSION_PLATFORM is cleared
    post-handler."""
    from gateway.platforms.base import build_auto_tts_output_path

    paths: list[str] = []
    requested_path = None
    try:
        from tools.tts_tool import text_to_speech_tool, check_tts_requirements
        resolve_scope = getattr(adapter.gateway_runner, "_async_profile_scope_for_source", None)
        scope = (resolve_scope(source) if callable(resolve_scope) and source is not None
                 else contextlib.nullcontext())
        async with scope:
            if check_tts_requirements():
                speech_text = adapter.prepare_tts_text(text_content)
                if not speech_text:
                    raise ValueError("Empty text after markdown cleanup")
                requested_path = build_auto_tts_output_path(adapter.platform)
                tts_data = json.loads(await asyncio.to_thread(
                    text_to_speech_tool, text=speech_text, output_path=requested_path))
                if tts_data.get("success", True):
                    raw_tts_paths = tts_data.get("file_paths") or [tts_data.get("file_path")]
                    paths = await asyncio.to_thread(_existing_paths, raw_tts_paths)
    # health: allow BLE001 -- best-effort reply: the TTS tool logs its own traceback
    except Exception as tts_err:
        logger.warning("[%s] Auto-TTS failed: %s", adapter.name, tts_err)
    return paths, requested_path
