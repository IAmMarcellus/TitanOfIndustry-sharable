"""Thin async client for the Paperclip control plane.

The voice sidecar runs on the same host as Paperclip and calls it over loopback with NO
Authorization header: in `local_trusted` deployment mode the server elevates such requests to an
implicit instance-admin actor (see vendor/paperclip/server/src/middleware/auth.ts), so no credential
minting is needed. Every helper degrades gracefully — a dead control plane must never crash a call.
"""

import asyncio
import os

import httpx
from loguru import logger

API_BASE = (os.environ.get("PAPERCLIP_API_URL", "").rstrip("/")) + "/api"

_client: httpx.AsyncClient | None = None


def client() -> httpx.AsyncClient:
    global _client
    if _client is None:
        _client = httpx.AsyncClient(base_url=API_BASE, timeout=httpx.Timeout(15.0, connect=4.0))
    return _client


async def get_json(path: str, params: dict | None = None) -> dict | list | None:
    """GET a Paperclip endpoint; returns parsed JSON or None on any failure."""
    try:
        resp = await client().get(path, params=params)
        resp.raise_for_status()
        return resp.json()
    except Exception as err:
        logger.warning(f"paperclip GET {path} failed: {err}")
        return None


async def send_json(method: str, path: str, body: dict) -> tuple[dict | list | None, str | None]:
    """POST/PATCH a Paperclip endpoint. Returns (parsed JSON, None) on success or (None, error
    message) on failure — write tools must be able to TELL the operator a write didn't land,
    not silently degrade like reads."""
    try:
        resp = await client().request(method, path, json=body)
        resp.raise_for_status()
        return resp.json() if resp.content else {}, None
    except httpx.HTTPStatusError as err:
        detail = ""
        try:
            detail = str(err.response.json().get("error", ""))[:120]
        except Exception:
            pass
        logger.warning(f"paperclip {method} {path} failed: {err} {detail}")
        return None, detail or f"request failed ({err.response.status_code})"
    except Exception as err:
        logger.warning(f"paperclip {method} {path} failed: {err}")
        return None, "the control plane didn't respond"


async def fetch_digest() -> str:
    """Fetch the live cross-company status snapshot (server-side cached ~15s)."""
    data = await get_json("/board/oversight/voice/digest")
    return data.get("digest", "") if isinstance(data, dict) else ""


async def persist_turn(role: str, body: str) -> None:
    """Append a finalized spoken turn to the durable Conference Room thread. Best effort."""
    if not body.strip():
        return
    try:
        await client().post("/board/oversight/voice/messages", json={"role": role, "body": body})
    except Exception as err:
        logger.warning(f"paperclip persist ({role}) failed: {err}")


def persist_turn_bg(role: str, body: str) -> None:
    """Fire-and-forget persist_turn for event handlers that must not block the pipeline."""
    task = asyncio.create_task(persist_turn(role, body))
    task.add_done_callback(lambda t: t.exception() and logger.warning(t.exception()))
