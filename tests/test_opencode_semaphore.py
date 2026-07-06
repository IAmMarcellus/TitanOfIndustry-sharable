"""agent/opencode_tool.py — the module-level semaphore bounds concurrent opencode subprocesses.

The subprocess spawn is faked, so we just assert that no more than OPENCODE_MAX_PARALLEL `communicate`
bodies run at once. Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("opencode_tool", _REPO / "agent" / "opencode_tool.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


def test_subprocess_concurrency_capped(tmp_path, monkeypatch) -> None:
    async def go() -> None:
        _mod._SEM = asyncio.Semaphore(2)  # force a known cap regardless of env
        active = 0
        peak = 0

        class FakeProc:
            returncode = 0

            async def communicate(self):
                nonlocal active, peak
                active += 1
                peak = max(peak, active)
                await asyncio.sleep(0.05)
                active -= 1
                return (b"out", b"err")

            def kill(self) -> None:
                pass

            async def wait(self) -> None:
                pass

        async def fake_exec(*a, **k):
            return FakeProc()

        monkeypatch.setattr(_mod.asyncio, "create_subprocess_exec", fake_exec)
        results = await asyncio.gather(
            *(_mod.opencode_run("do x", cwd=str(tmp_path)) for _ in range(6))
        )
        assert all(r["success"] for r in results)
        assert peak <= 2

    asyncio.run(go())
