"""agent/paperclip_tool.py — control-plane tools read per-run creds from session state and call
the Paperclip API with the right auth headers; missing creds no-op gracefully (no HTTP).

httpx is faked, so we assert URL/method/headers/body construction and the no-creds short-circuit
without any running services. Run: uv run --project vendor/opensage-adk --with pytest pytest tests/ -q
"""

from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("paperclip_tool", _REPO / "agent" / "paperclip_tool.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

_FULL_STATE = {
    "paperclip_api_token": "tok-123",
    "paperclip_run_id": "run-abc",
    "paperclip_base_url": "https://paperclip.test",
    "paperclip_agent_id": "agent-9",
    "paperclip_company_id": "co-7",
}


class _Ctx:
    """Minimal stand-in for ADK ToolContext: just needs a `.state` mapping with `.get`."""

    def __init__(self, state: dict) -> None:
        self.state = dict(state)


class _FakeResp:
    status_code = 200
    text = "ok"

    def json(self) -> dict:
        return {"ok": True}


def _install_fake_httpx(monkeypatch) -> list[dict]:
    """Replace httpx.AsyncClient with a recorder; return the list of captured requests."""
    calls: list[dict] = []

    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc) -> bool:
            return False

        async def request(self, method, url, json=None, params=None, headers=None):
            calls.append(
                {"method": method, "url": url, "json": json, "params": params, "headers": headers}
            )
            return _FakeResp()

    monkeypatch.setattr(_mod.httpx, "AsyncClient", _FakeClient)
    return calls


def test_post_comment_builds_request_and_auth(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        res = await _mod.paperclip_post_comment("ISS1", "hello", _Ctx(_FULL_STATE))
        assert res["success"] is True and res["status_code"] == 200
        assert len(calls) == 1
        c = calls[0]
        assert c["method"] == "POST"
        assert c["url"] == "https://paperclip.test/api/issues/ISS1/comments"
        assert c["json"] == {"body": "hello"}
        assert c["headers"]["authorization"] == "Bearer tok-123"
        assert c["headers"]["x-paperclip-run-id"] == "run-abc"

    asyncio.run(go())


def test_checkout_sends_agent_id_and_statuses(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        res = await _mod.paperclip_checkout("ISS1", _Ctx(_FULL_STATE))
        assert res["success"] is True
        c = calls[0]
        assert c["url"].endswith("/api/issues/ISS1/checkout")
        assert c["json"]["agentId"] == "agent-9"
        assert "todo" in c["json"]["expectedStatuses"]

    asyncio.run(go())


def test_update_issue_only_sends_provided_fields(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        res = await _mod.paperclip_update_issue("ISS1", _Ctx(_FULL_STATE), status="done", comment="ok")
        assert res["success"] is True
        assert calls[0]["json"] == {"status": "done", "comment": "ok"}

    asyncio.run(go())


def test_update_issue_empty_is_noop_no_http(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        res = await _mod.paperclip_update_issue("ISS1", _Ctx(_FULL_STATE))
        assert res["success"] is False and "nothing to update" in res["error"]
        assert len(calls) == 0

    asyncio.run(go())


def test_create_subtask_targets_company(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        await _mod.paperclip_create_subtask("T", "D", _Ctx(_FULL_STATE), parent_id="P", assignee_agent_id="a2")
        c = calls[0]
        assert c["url"].endswith("/api/companies/co-7/issues")
        assert c["json"] == {"title": "T", "description": "D", "parentId": "P", "assigneeAgentId": "a2"}

    asyncio.run(go())


def test_no_credentials_short_circuits_without_http(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)

    async def go() -> None:
        # state with no token (e.g. JWT secret unset on the Paperclip server)
        res = await _mod.paperclip_post_comment("ISS1", "hi", _Ctx({}))
        assert res["success"] is False and "no paperclip credentials" in res["error"]
        assert len(calls) == 0

    asyncio.run(go())


def test_base_url_falls_back_to_default(monkeypatch) -> None:
    calls = _install_fake_httpx(monkeypatch)
    state = {k: v for k, v in _FULL_STATE.items() if k != "paperclip_base_url"}

    async def go() -> None:
        await _mod.paperclip_get_issue("ISS1", _Ctx(state))
        assert calls[0]["url"] == _mod._DEFAULT_BASE_URL.rstrip("/") + "/api/issues/ISS1"

    asyncio.run(go())
