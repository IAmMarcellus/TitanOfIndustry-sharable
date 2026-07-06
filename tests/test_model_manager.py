"""Unit tests for scripts/model_manager.py — the C3a/C3b state machine, with mocked engines.

The engine subprocesses and vLLM HTTP control endpoints are faked, so we exercise the mode
transitions (boot / ensure-vision / idle-release) without a GPU. Run:
uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
import os
from pathlib import Path

os.environ.setdefault("MODEL_MANAGER_PORT", "1")
os.environ.setdefault("VLLM_TEXT_BASE_URL", "https://text-engine.test/v1")
os.environ.setdefault("VLLM_VISION_BASE_URL", "https://vision-engine.test/v1")

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("model_manager", _REPO / "scripts" / "model_manager.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
ModelManager = _mod.ModelManager
Mode = _mod.Mode


class FakeProc:
    def __init__(self) -> None:
        self._alive = True

    def terminate(self) -> None:
        self._alive = False  # so poll() reports exited immediately (no real sleep in _stop)

    def kill(self) -> None:
        self._alive = False

    def poll(self):
        return None if self._alive else 0


class FakeResp:
    def __init__(self, status_code: int = 200, text: str = "") -> None:
        self.status_code = status_code
        self.text = text


class FakeClient:
    """Health is always 200; text /metrics reports drained (0), vision /metrics reports vision_running."""

    def __init__(self, vision_running: str = "0.0") -> None:
        self.vision_running = vision_running
        self.calls: list = []

    async def get(self, url: str, **kw):
        self.calls.append(("GET", url))
        if url.endswith("/metrics"):
            running = self.vision_running if url.startswith("https://vision-engine.test") else "0.0"
            return FakeResp(200, f"vllm:num_requests_running {running}\n")
        return FakeResp(200, "")

    async def post(self, url: str, **kw):
        self.calls.append(("POST", url, kw.get("params")))
        return FakeResp(200, "")

    async def aclose(self) -> None:
        pass


class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def _mgr(strategy: str, client: FakeClient | None = None):
    spawned: list = []

    def spawn(cmd, env):
        spawned.append((cmd, dict(env)))
        return FakeProc()

    clock = Clock()
    mgr = ModelManager(spawn=spawn, client=client or FakeClient(), clock=clock, strategy=strategy)
    return mgr, spawned, clock


def _is_vision(cmd) -> bool:
    return any("serve-vllm-engine.sh" in part for part in cmd) and "vision" in cmd


def _is_text(cmd) -> bool:
    return any("serve-vllm-engine.sh" in part for part in cmd) and "text" in cmd


def test_start_c3b_is_text_full_only() -> None:
    async def go() -> None:
        mgr, spawned, _ = _mgr("c3b")
        assert await mgr.start() == Mode.TEXT_FULL
        assert len(spawned) == 1
        cmd, env = spawned[0]
        assert _is_text(cmd)
        assert env["VLLM_TEXT_GPU_UTIL"] == str(_mod.TEXT_UTIL_FULL)
        assert env["VLLM_TEXT_MAX_SEQS"] == str(_mod.TEXT_SEQS_FULL)

    asyncio.run(go())


def test_start_c3a_is_steady_with_slept_vision() -> None:
    async def go() -> None:
        client = FakeClient()
        mgr, spawned, _ = _mgr("c3a", client=client)
        assert await mgr.start() == Mode.STEADY
        assert _is_text(spawned[0][0])
        assert spawned[0][1]["VLLM_TEXT_GPU_UTIL"] == str(_mod.TEXT_UTIL_MIXED)
        assert any(_is_vision(c) for c, _ in spawned)
        assert any(m[0] == "POST" and m[1].endswith("/sleep") for m in client.calls)

    asyncio.run(go())


def test_ensure_vision_transitions_text_full_to_mixed() -> None:
    async def go() -> None:
        mgr, spawned, _ = _mgr("c3b")
        await mgr.start()
        assert await mgr.ensure_vision() == Mode.MIXED
        text_launches = [env for c, env in spawned if _is_text(c)]
        assert len(text_launches) == 2  # original FULL + relaunched MIXED
        assert text_launches[1]["VLLM_TEXT_GPU_UTIL"] == str(_mod.TEXT_UTIL_MIXED)
        assert any(_is_vision(c) for c, _ in spawned)

    asyncio.run(go())


def test_ensure_vision_when_mixed_only_wakes() -> None:
    async def go() -> None:
        client = FakeClient()
        mgr, spawned, _ = _mgr("c3b", client=client)
        await mgr.start()
        await mgr.ensure_vision()  # -> MIXED
        n = len(spawned)
        client.calls.clear()
        assert await mgr.ensure_vision() == Mode.MIXED
        assert len(spawned) == n  # no new engine launches
        assert any(m[0] == "POST" and m[1].endswith("/wake_up") for m in client.calls)

    asyncio.run(go())


def test_idle_returns_c3b_to_text_full() -> None:
    async def go() -> None:
        client = FakeClient(vision_running="0.0")
        mgr, spawned, clock = _mgr("c3b", client=client)
        await mgr.start()
        await mgr.ensure_vision()  # MIXED
        clock.t += _mod.VISION_IDLE_S + 1
        assert await mgr.maybe_release_vision() == Mode.TEXT_FULL
        last_text = [env for c, env in spawned if _is_text(c)][-1]
        assert last_text["VLLM_TEXT_GPU_UTIL"] == str(_mod.TEXT_UTIL_FULL)
        assert any(m[0] == "POST" and m[1].endswith("/sleep") for m in client.calls)

    asyncio.run(go())


def test_idle_release_skips_when_vision_busy() -> None:
    async def go() -> None:
        client = FakeClient(vision_running="2.0")  # vision still serving
        mgr, _, clock = _mgr("c3b", client=client)
        await mgr.start()
        await mgr.ensure_vision()  # MIXED
        clock.t += _mod.VISION_IDLE_S + 1
        assert await mgr.maybe_release_vision() == Mode.MIXED  # stays

    asyncio.run(go())


def test_no_release_before_idle_timeout() -> None:
    async def go() -> None:
        mgr, _, _ = _mgr("c3b")
        await mgr.start()
        await mgr.ensure_vision()  # MIXED
        assert await mgr.maybe_release_vision() == Mode.MIXED  # clock not advanced -> not idle

    asyncio.run(go())
