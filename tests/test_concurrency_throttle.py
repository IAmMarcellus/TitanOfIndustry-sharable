"""Unit tests for agent/plugins/concurrency_throttle_plugin.py — gating, release, and the cap.

Loaded by file path (as OpenSage's loader does). Async callbacks are driven with asyncio.run, like
tests/test_periodic_snapshot.py. Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location(
    "concurrency_throttle_plugin", _REPO / "agent" / "plugins" / "concurrency_throttle_plugin.py"
)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ConcurrencyThrottlePlugin = _mod.ConcurrencyThrottlePlugin


def _heavy() -> SimpleNamespace:
    return SimpleNamespace(name="call_subagent_as_tool")


def _light() -> SimpleNamespace:
    return SimpleNamespace(name="think")


def test_light_tool_not_gated() -> None:
    async def go() -> None:
        p = ConcurrencyThrottlePlugin(max_active=1)
        await p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None)  # exhaust the permit
        # a non-gated tool must pass straight through even with the semaphore empty
        await asyncio.wait_for(
            p.before_tool_callback(tool=_light(), tool_args={}, tool_context=None), timeout=1
        )

    asyncio.run(go())


def test_blocks_when_full_then_releases() -> None:
    async def go() -> None:
        p = ConcurrencyThrottlePlugin(max_active=1)
        await p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None)
        t = asyncio.create_task(p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None))
        await asyncio.sleep(0.05)
        assert not t.done()  # second heavy call blocked at the cap
        await p.after_tool_callback(tool=_heavy(), tool_args={}, tool_context=None, result={})
        await asyncio.wait_for(t, timeout=1)  # release lets it through
        assert t.done()

    asyncio.run(go())


def test_over_release_guard() -> None:
    async def go() -> None:
        p = ConcurrencyThrottlePlugin(max_active=1)
        # spurious after with nothing held must NOT inflate the permit count
        await p.after_tool_callback(tool=_heavy(), tool_args={}, tool_context=None, result={})
        await p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None)
        t = asyncio.create_task(p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None))
        await asyncio.sleep(0.05)
        assert not t.done()  # still capped at 1 despite the spurious release
        t.cancel()

    asyncio.run(go())


def test_env_overrides_param(monkeypatch) -> None:
    monkeypatch.setenv("MAX_ACTIVE_SUBAGENTS", "3")
    assert ConcurrencyThrottlePlugin(max_active=1)._max == 3


def test_cap_bounds_parallel_heavy_calls() -> None:
    async def go() -> None:
        p = ConcurrencyThrottlePlugin(max_active=2)
        active = 0
        peak = 0

        async def call() -> None:
            nonlocal active, peak
            await p.before_tool_callback(tool=_heavy(), tool_args={}, tool_context=None)
            active += 1
            peak = max(peak, active)
            await asyncio.sleep(0.03)
            active -= 1
            await p.after_tool_callback(tool=_heavy(), tool_args={}, tool_context=None, result={})

        await asyncio.gather(*(call() for _ in range(6)))
        assert peak <= 2

    asyncio.run(go())
