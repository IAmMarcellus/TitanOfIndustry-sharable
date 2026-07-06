"""proxy/hooks.py — the LiteLLM pre-call hook wakes vision only for the qwen-vl model, and never raises.

Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("proxy_hooks", _REPO / "proxy" / "hooks.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)


class _FakeClient:
    def __init__(self) -> None:
        self.calls: list[str] = []

    async def post(self, url: str, **k):
        self.calls.append(url)
        return None


def _run(model: str, fc) -> dict:
    """Drive the hook for one request with the reused client patched to `fc`."""
    hook = _mod.VisionWakeHook()
    return asyncio.run(hook.async_pre_call_hook(None, None, {"model": model}, "completion"))


def test_vision_call_triggers_ensure(monkeypatch) -> None:
    fc = _FakeClient()
    monkeypatch.setattr(_mod, "_http", lambda: fc)
    out = _run("qwen-vl", fc)
    assert out == {"model": "qwen-vl"}
    assert any(u.endswith("/ensure-vision") for u in fc.calls)


def test_openai_prefixed_vision_model_matches(monkeypatch) -> None:
    fc = _FakeClient()
    monkeypatch.setattr(_mod, "_http", lambda: fc)
    _run("openai/qwen-vl", fc)  # last path segment == qwen-vl
    assert len(fc.calls) == 1


def test_text_call_does_not_trigger(monkeypatch) -> None:
    fc = _FakeClient()
    monkeypatch.setattr(_mod, "_http", lambda: fc)
    _run("qwen-codex", fc)
    assert fc.calls == []


def test_superstring_model_does_not_falsely_match(monkeypatch) -> None:
    fc = _FakeClient()
    monkeypatch.setattr(_mod, "_http", lambda: fc)
    _run("qwen-vl-7b", fc)  # exact segment match, not substring
    assert fc.calls == []


def test_manager_error_is_swallowed(monkeypatch) -> None:
    class Boom:
        async def post(self, *a, **k):
            raise RuntimeError("manager down")

    monkeypatch.setattr(_mod, "_http", lambda: Boom())
    out = _run("qwen-vl", Boom())
    assert out == {"model": "qwen-vl"}  # must not raise