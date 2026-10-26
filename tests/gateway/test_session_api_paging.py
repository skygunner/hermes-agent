"""``GET /api/sessions`` pagination truthfulness (#144349): ``has_more`` describes the recency
window, never the pinned rows ``include_pinned`` back-fills past it. A pinned row that is recent
enough to sit inside the window fills the page like any other row, so a client paging on
``has_more`` must still see the rest; a pin back-filled onto a short page must not claim a page
that does not exist."""

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter
from hermes_state import SessionDB


@pytest.fixture
def session_db(tmp_path):
    db = SessionDB(tmp_path / "state.db")
    try:
        yield db
    finally:
        close = getattr(db, "close", None)
        if callable(close):
            close()


@pytest.fixture
def adapter(session_db):
    adapter = APIServerAdapter(PlatformConfig(enabled=True))
    adapter._session_db = session_db
    return adapter


def _sessions_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application()
    app.router.add_get("/api/sessions", adapter._handle_list_sessions)
    return app


def _seed(session_db: SessionDB, ids: list[str]) -> None:
    """Create *ids* with distinct, increasing recency (oldest first) so page composition is fixed."""
    for i, session_id in enumerate(ids):
        session_db.create_session(session_id, "api_server", model="test/model")
        session_db._write_sql(
            "UPDATE sessions SET started_at = ?, last_activity_at = ? WHERE id = ?",
            (1_700_000_000 + i, 1_700_000_000 + i, session_id))


@pytest.mark.asyncio
async def test_has_more_with_a_pin_inside_the_page(adapter, session_db):
    ids = [f"sess-{i:02d}" for i in range(55)]
    _seed(session_db, ids)
    session_db.set_session_pinned(ids[-1], True)  # newest row: pinned inside the first page

    async with TestClient(TestServer(_sessions_app(adapter))) as cli:
        first = await (await cli.get("/api/sessions?limit=50")).json()
        rest = await (await cli.get("/api/sessions?limit=50&offset=50")).json()

    page_ids = [row["id"] for row in first["data"]]
    assert len(page_ids) == 50
    assert ids[-1] in page_ids and first["data"][page_ids.index(ids[-1])]["pinned"] is True
    assert first["has_more"] is True
    # offset=limit reaches the rest; the pin rides along on every page by include_pinned contract.
    assert sorted(r["id"] for r in rest["data"] if not r["pinned"]) == sorted(set(ids) - set(page_ids))


@pytest.mark.asyncio
async def test_has_more_ignores_pins_backfilled_past_a_short_page(adapter, session_db):
    ids = [f"sess-{i:02d}" for i in range(5)]
    _seed(session_db, ids)
    for archived_pin in ("pinned-archived-a", "pinned-archived-b"):
        session_db.create_session(archived_pin, "api_server", model="test/model")
        session_db.set_session_pinned(archived_pin, True)
        session_db.set_session_archived(archived_pin, True)

    async with TestClient(TestServer(_sessions_app(adapter))) as cli:
        page = await (await cli.get("/api/sessions?limit=10")).json()

    # The pins still extend the page (they ignore the archived filter), but the window is short.
    assert sorted(row["id"] for row in page["data"]) == sorted(
        ids + ["pinned-archived-a", "pinned-archived-b"])
    assert page["has_more"] is False
