"""Unit tests for agent/plugins/periodic_snapshot_plugin.py.

Loads the plugin by file path (the same way OpenSage's loader does) and mocks the OpenSage writer +
session accessor, so we test the throttle, arg-wiring, and never-break-the-run behavior without a
live Neo4j/session. Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

import opensage.cli.opensage_cli as cli
import opensage.session.opensage_session as sess

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "periodic_snapshot_plugin", _REPO / "agent" / "plugins" / "periodic_snapshot_plugin.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
PeriodicSnapshotPlugin = _mod.PeriodicSnapshotPlugin


def _ctx() -> SimpleNamespace:
    session = SimpleNamespace(id="sess-1", app_name="agent", user_id="user")
    return SimpleNamespace(session=session, session_service="SVC")


def _patch(monkeypatch, persist=None) -> AsyncMock:
    persist = persist or AsyncMock()
    monkeypatch.setattr(cli, "_persist_web_session_snapshot_async", persist)
    monkeypatch.setattr(sess, "get_opensage_session", lambda sid, *a, **k: f"OS:{sid}")
    return persist


def _fire(p, ctx: SimpleNamespace) -> None:
    asyncio.run(p.on_event_callback(invocation_context=ctx, event=object()))


def test_throttle_every_n(monkeypatch):
    monkeypatch.delenv("SNAPSHOT_EVERY_N", raising=False)
    persist = _patch(monkeypatch)
    p = PeriodicSnapshotPlugin(every_n_events=3)
    ctx = _ctx()
    _fire(p, ctx)
    _fire(p, ctx)
    assert persist.await_count == 0  # not yet (events 1,2)
    _fire(p, ctx)
    assert persist.await_count == 1  # fires on the 3rd event


def test_args_wired(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_EVERY_N", "1")
    persist = _patch(monkeypatch)
    _fire(PeriodicSnapshotPlugin(), _ctx())
    persist.assert_awaited_once()
    kw = persist.await_args.kwargs
    assert kw["session_id"] == "sess-1"
    assert kw["app_name"] == "agent"
    assert kw["user_id"] == "user"
    assert kw["session_service"] == "SVC"
    assert kw["opensage_session"] == "OS:sess-1"
    assert kw["agent_dir"].endswith("/agent")


def test_exception_is_swallowed(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_EVERY_N", "1")
    boom = _patch(monkeypatch, persist=AsyncMock(side_effect=RuntimeError("disk full")))
    # must NOT raise — a snapshot failure can't break the run
    _fire(PeriodicSnapshotPlugin(), _ctx())
    assert boom.await_count == 1


def test_env_overrides_param(monkeypatch):
    monkeypatch.setenv("SNAPSHOT_EVERY_N", "2")
    persist = _patch(monkeypatch)
    p = PeriodicSnapshotPlugin(every_n_events=1)  # env (2) wins over the param (1)
    ctx = _ctx()
    _fire(p, ctx)
    assert persist.await_count == 0
    _fire(p, ctx)
    assert persist.await_count == 1
