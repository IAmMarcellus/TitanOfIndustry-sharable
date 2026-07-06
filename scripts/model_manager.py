#!/usr/bin/env python3
"""TitanOfIndustry model-manager — VRAM scheduler for the dual-model (C3a/C3b) setup on one 3090.

Owns the lifecycle of the two vLLM engines and flips between modes so the VISION model is resident
only when needed:

  TEXT_FULL : text engine at FULL util (4 seqs), vision STOPPED.            (C3b default / idle)
  MIXED     : text engine CAPPED (2 seqs) + vision resident (sleep/wake).   (during vision work)
  STEADY    : text CAPPED always + vision resident (sleep/wake), no restarts. (C3a toggle)

vLLM can't live-resize its VRAM pool, so changing text's slice means a RESTART — that only happens at
TEXT_FULL<->MIXED boundaries (C3b), each preceded by a graceful (capped) drain of in-flight text so a
restart doesn't kill running code-gen. Within MIXED, vision is cheaply slept/woken via vLLM sleep
mode (/sleep, /wake_up).

The LiteLLM proxy calls POST /ensure-vision (via proxy/hooks.py) before forwarding a qwen-vl request;
a background watchdog returns to TEXT_FULL after VISION_IDLE_S of no vision activity (C3b) or just
sleeps vision (C3a/STEADY).

Run: `make model-stack` (or `python scripts/model_manager.py`). FastAPI binds to MODEL_MANAGER_PORT.
The engine lifecycle / sleep-wake is GPU-gated; the state machine is unit-tested with mocks
(tests/test_model_manager.py).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import os
import subprocess
import time
from collections.abc import Callable
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import httpx
from fastapi import FastAPI

logger = logging.getLogger("model_manager")

_REPO_ROOT = Path(__file__).resolve().parents[1]
_ENGINE_SCRIPT = str(_REPO_ROOT / "scripts" / "serve-vllm-engine.sh")  # usage: serve-vllm-engine.sh text|vision


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return float(default)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return int(default)


TEXT_BASE_URL = os.environ.get("VLLM_TEXT_BASE_URL", "")
VISION_BASE_URL = os.environ.get("VLLM_VISION_BASE_URL", "")
TEXT_UTIL_FULL = _env_float("VLLM_TEXT_GPU_UTIL_FULL", 0.90)
TEXT_UTIL_MIXED = _env_float("VLLM_TEXT_GPU_UTIL_MIXED", 0.58)
TEXT_SEQS_FULL = _env_int("VLLM_TEXT_MAX_SEQS_FULL", 4)
TEXT_SEQS_MIXED = _env_int("VLLM_TEXT_MAX_SEQS_MIXED", 2)
VISION_IDLE_S = _env_float("VISION_IDLE_S", 180)
STRATEGY = os.environ.get("VISION_STRATEGY", "c3b").lower()  # "c3b" (default) or "c3a"
DRAIN_TIMEOUT_S = _env_float("MODEL_MANAGER_DRAIN_TIMEOUT_S", 90)
ENGINE_START_TIMEOUT_S = _env_float("MODEL_MANAGER_ENGINE_START_TIMEOUT_S", 300)
WATCHDOG_S = _env_float("MODEL_MANAGER_WATCHDOG_S", 10)
PORT = int(os.environ["MODEL_MANAGER_PORT"])


def _origin(base_url: str) -> str:
    """vLLM control endpoints (/health, /sleep, /wake_up, /metrics) live at the root, not under /v1."""
    base_url = base_url.rstrip("/")
    return base_url[: -len("/v1")] if base_url.endswith("/v1") else base_url


class Mode(str, Enum):
    STOPPED = "stopped"
    TEXT_FULL = "text_full"
    MIXED = "mixed"
    STEADY = "steady"


# Spawner indirection so tests can inject a fake instead of launching real vLLM.
Spawn = Callable[[list[str], dict[str, str]], "subprocess.Popen[bytes]"]


def _default_spawn(cmd: list[str], env: dict[str, str]) -> "subprocess.Popen[bytes]":
    return subprocess.Popen(cmd, env=env)


class ModelManager:
    """Serialized state machine over the two vLLM engines. All transitions hold `self._lock`."""

    def __init__(
        self,
        *,
        spawn: Spawn = _default_spawn,
        client: Optional[httpx.AsyncClient] = None,
        clock: Callable[[], float] = time.monotonic,
        strategy: Optional[str] = None,
    ) -> None:
        self._spawn = spawn
        self._client = client  # lazily created on first use if not injected (see _http)
        self._clock = clock
        self._strategy = (strategy or STRATEGY).lower()
        self._lock = asyncio.Lock()
        self._mode = Mode.STOPPED
        self._text_proc: Optional[Any] = None
        self._vision_proc: Optional[Any] = None
        self._last_vision = 0.0

    def _http(self) -> httpx.AsyncClient:
        if self._client is None:
            self._client = httpx.AsyncClient(timeout=30.0)
        return self._client

    @property
    def mode(self) -> Mode:
        return self._mode

    @property
    def strategy(self) -> str:
        return self._strategy

    # ---- engine lifecycle ----
    async def _launch_text(self, util: float, seqs: int) -> None:
        env = dict(os.environ)
        env["VLLM_TEXT_GPU_UTIL"] = str(util)
        env["VLLM_TEXT_MAX_SEQS"] = str(seqs)
        self._text_proc = self._spawn(["bash", _ENGINE_SCRIPT, "text"], env)
        await self._await_health(TEXT_BASE_URL)

    async def _launch_vision(self) -> None:
        self._vision_proc = self._spawn(["bash", _ENGINE_SCRIPT, "vision"], dict(os.environ))
        await self._await_health(VISION_BASE_URL)

    async def _restart_text(self, util: float, seqs: int) -> None:
        """Drain in-flight text, stop the engine, relaunch it at a new util/seqs (the C3b restart)."""
        await self._drain_text()
        await self._stop(self._text_proc)
        await self._launch_text(util, seqs)

    async def _stop(self, proc: Optional[Any]) -> None:
        if proc is None:
            return
        with contextlib.suppress(ProcessLookupError):
            proc.terminate()
        for _ in range(50):  # ~5s grace, then hard-kill
            if proc.poll() is not None:
                return
            await asyncio.sleep(0.1)
        with contextlib.suppress(ProcessLookupError):
            proc.kill()

    async def _await_health(self, base_url: str, timeout: Optional[float] = None) -> None:
        deadline = self._clock() + (timeout if timeout is not None else ENGINE_START_TIMEOUT_S)
        url = _origin(base_url) + "/health"
        while self._clock() < deadline:
            with contextlib.suppress(Exception):
                resp = await self._http().get(url)
                if resp.status_code == 200:
                    return
            await asyncio.sleep(1.0)
        raise TimeoutError(f"engine did not become healthy: {url}")

    async def _num_running(self, base_url: str) -> float:
        """Parse vllm:num_requests_running from /metrics; 0 on any error (treat as drained)."""
        url = _origin(base_url) + "/metrics"
        with contextlib.suppress(Exception):
            resp = await self._http().get(url)
            for line in resp.text.splitlines():
                if line.startswith("vllm:num_requests_running"):
                    return float(line.rsplit(" ", 1)[-1])
        return 0.0

    async def _drain_text(self) -> None:
        deadline = self._clock() + DRAIN_TIMEOUT_S
        while self._clock() < deadline:
            if await self._num_running(TEXT_BASE_URL) <= 0:
                return
            await asyncio.sleep(1.0)
        logger.warning("text drain timed out after %.0fs; restarting with requests in flight", DRAIN_TIMEOUT_S)

    # ---- vision sleep/wake (vLLM dev endpoints) ----
    async def _wake_vision(self) -> None:
        with contextlib.suppress(Exception):
            await self._http().post(_origin(VISION_BASE_URL) + "/wake_up")

    async def _sleep_vision(self) -> None:
        with contextlib.suppress(Exception):
            await self._http().post(_origin(VISION_BASE_URL) + "/sleep", params={"level": 1})

    # ---- public API ----
    async def start(self) -> Mode:
        """Boot the initial engines: C3b -> TEXT_FULL; C3a/STEADY -> capped text + slept vision."""
        async with self._lock:
            if self._strategy == "c3a":
                await self._launch_text(TEXT_UTIL_MIXED, TEXT_SEQS_MIXED)
                await self._launch_vision()
                await self._sleep_vision()
                self._mode = Mode.STEADY
            else:
                await self._launch_text(TEXT_UTIL_FULL, TEXT_SEQS_FULL)
                self._mode = Mode.TEXT_FULL
            return self._mode

    async def ensure_vision(self) -> Mode:
        """Make vision serviceable for an incoming request; transitions TEXT_FULL->MIXED if needed."""
        async with self._lock:
            self._last_vision = self._clock()
            if self._mode in (Mode.MIXED, Mode.STEADY):
                await self._wake_vision()
                return self._mode
            # TEXT_FULL -> MIXED: cap text (restart) and bring vision up alongside it.
            await self._restart_text(TEXT_UTIL_MIXED, TEXT_SEQS_MIXED)
            await self._launch_vision()
            self._mode = Mode.MIXED
            return self._mode

    async def maybe_release_vision(self) -> Mode:
        """Watchdog tick: after VISION_IDLE_S idle, sleep vision (STEADY) or return to TEXT_FULL (C3b)."""
        if self._clock() - self._last_vision <= VISION_IDLE_S:
            return self._mode
        if self._mode == Mode.STEADY:
            await self._sleep_vision()
            return self._mode
        if self._mode != Mode.MIXED:
            return self._mode
        async with self._lock:
            # Re-check under the lock; don't tear down if a request snuck in or vision is still busy.
            if self._mode != Mode.MIXED or self._clock() - self._last_vision <= VISION_IDLE_S:
                return self._mode
            if await self._num_running(VISION_BASE_URL) > 0:
                return self._mode
            await self._sleep_vision()
            await self._stop(self._vision_proc)
            self._vision_proc = None
            await self._restart_text(TEXT_UTIL_FULL, TEXT_SEQS_FULL)
            self._mode = Mode.TEXT_FULL
            return self._mode

    async def aclose(self) -> None:
        await self._stop(self._vision_proc)
        await self._stop(self._text_proc)
        if self._client is not None:
            await self._client.aclose()


# ---- FastAPI app (module singleton; tests construct ModelManager directly) ----
manager = ModelManager()


async def _watchdog() -> None:
    while True:
        await asyncio.sleep(WATCHDOG_S)
        with contextlib.suppress(Exception):
            await manager.maybe_release_vision()


@contextlib.asynccontextmanager
async def lifespan(_app: FastAPI):  # noqa: ANN201 (FastAPI lifespan)
    await manager.start()
    task = asyncio.create_task(_watchdog())
    try:
        yield
    finally:
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        await manager.aclose()


app = FastAPI(title="TitanOfIndustry model-manager", lifespan=lifespan)


@app.post("/ensure-vision")
async def ensure_vision() -> dict[str, str]:
    return {"mode": (await manager.ensure_vision()).value}


@app.post("/sleep-vision")
async def sleep_vision() -> dict[str, str]:
    return {"mode": (await manager.maybe_release_vision()).value}


@app.get("/status")
async def status() -> dict[str, str]:
    return {"mode": manager.mode.value, "strategy": manager.strategy}


@app.get("/healthz")
async def healthz() -> dict[str, bool]:
    return {"ok": True}


if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host="127.0.0.1", port=PORT)
