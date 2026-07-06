"""LiteLLM proxy callbacks. Two CustomLogger handlers are defined here:

- UsageLogger (`usage_logger_instance`): log per-request + cumulative token usage (input/output/
  cached) so cloud-planner (chatgpt-codex) subscription burn is observable in the proxy window.
  This is the handler registered in proxy/litellm-proxy.yaml (`litellm_settings: callbacks`).
- VisionWakeHook (`proxy_handler_instance`): on a qwen-vl call, ensure the model-manager has
  transitioned so vision can serve it (TEXT_FULL->MIXED + wake) BEFORE the proxy forwards (C3b).
  Relevant only on the --with-vllm/model-manager path and NOT currently registered; text calls pass
  through untouched and it never blocks routing on its own failure (manager unreachable -> just
  forward and let the proxy's retries/timeout cover a cold engine).

litellm's get_instance_fn resolves the callback module RELATIVE to the config file's directory, so
the yaml registers it as `hooks.usage_logger_instance` (= proxy/hooks.py), not `proxy.hooks.…`.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import Any, Optional

import httpx
from litellm.integrations.custom_logger import CustomLogger

_MANAGER_URL = os.environ.get("MODEL_MANAGER_URL", "").rstrip("/")
_VISION_MODEL = os.environ.get("VLLM_VISION_SERVED_NAME", "qwen-vl")
# The transition can take tens of seconds (text restart) — allow a generous wait for /ensure-vision.
_ENSURE_TIMEOUT_S = float(os.environ.get("MODEL_MANAGER_ENSURE_TIMEOUT_S", "600"))

_client: Optional[httpx.AsyncClient] = None


def _http() -> httpx.AsyncClient:
    """One reused client (keep-alive to the local manager) instead of one per vision call."""
    global _client
    if _client is None:
        _client = httpx.AsyncClient(timeout=_ENSURE_TIMEOUT_S)
    return _client


class VisionWakeHook(CustomLogger):
    """On a qwen-vl call, block until the model-manager reports vision is serviceable."""

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: dict[str, Any],
        call_type: str,
    ) -> Optional[dict[str, Any]]:
        # Match the model's last path segment exactly so "openai/qwen-vl" hits but "qwen-vl-7b" doesn't.
        model = str((data or {}).get("model", ""))
        if model.rsplit("/", 1)[-1] == _VISION_MODEL:
            try:
                await _http().post(_MANAGER_URL + "/ensure-vision")
            except Exception:
                # Never block routing on the hook's own failure; forward and let retries/timeout cover it.
                pass
        return data


proxy_handler_instance = VisionWakeHook()


def _get(obj: Any, key: str) -> Any:
    """Read ``key`` whether ``obj`` is a pydantic/usage object (attr) or a plain dict."""
    if obj is None:
        return None
    val = getattr(obj, key, None)
    if val is None and isinstance(obj, dict):
        val = obj.get(key)
    return val


def _cached_tokens(usage: Any) -> int:
    """Cached-prompt-token count, robust to the two usage shapes we see on this proxy.

    The chatgpt-codex route runs litellm's chat->Responses bridge: OpenAI returns cache hits under
    ``usage.input_tokens_details.cached_tokens`` (Responses shape); the bridge normally remaps that to
    ``usage.prompt_tokens_details.cached_tokens`` (Chat shape). Reading ONLY the Chat shape (the old
    behaviour) silently reported 0 whenever the response_obj still carried the Responses shape — which
    made the planner look uncached. Check both, accept the first positive value.
    """
    for attr in ("prompt_tokens_details", "input_tokens_details"):
        v = _get(_get(usage, attr), "cached_tokens")
        if isinstance(v, int) and v > 0:
            return v
    return 0


class UsageLogger(CustomLogger):
    """Log token usage per completion + a running per-model total, so cloud-planner (chatgpt-codex)
    subscription burn — and prompt-cache hit rate — is observable in the proxy's tmux window/logfile.

    Emits one grep-friendly ``[usage]`` line per successful call via ``print(flush=True)`` (rather than
    a logger) so it lands in stdout regardless of litellm's log level. ``cached`` is the prompt-cache
    hit count (read from either ``prompt_tokens_details`` or ``input_tokens_details``; see
    ``_cached_tokens``). ``cache%`` is cached/input for the call and ``cum`` is the running total — a
    steady ``cache%=0`` on warm consecutive planner turns means caching genuinely isn't firing (e.g.
    an unstable prompt prefix), now that the field-mismatch blind spot is fixed.
    """

    def __init__(self) -> None:
        # model -> cumulative {in, out, cached, calls} since proxy start.
        self._totals: dict[str, dict[str, int]] = defaultdict(
            lambda: {"in": 0, "out": 0, "cached": 0, "calls": 0}
        )

    @staticmethod
    def _usage_of(kwargs: dict[str, Any], response_obj: Any) -> Any:
        """The usage object. Prefer the response; fall back to litellm's aggregated streaming response,
        since the chat->Responses bridge yields per-chunk ``usage=None`` and fills usage only on the
        aggregate — which litellm sets on ``response_obj`` / ``complete_streaming_response`` before the
        success callback fires."""
        usage = getattr(response_obj, "usage", None)
        if usage is not None:
            return usage
        agg = kwargs.get("complete_streaming_response")
        return getattr(agg, "usage", None) if agg is not None else None

    async def async_log_success_event(
        self, kwargs: dict[str, Any], response_obj: Any, start_time: Any, end_time: Any
    ) -> None:
        usage = self._usage_of(kwargs, response_obj)
        if usage is None:
            return
        in_tok = _get(usage, "prompt_tokens") or 0
        out_tok = _get(usage, "completion_tokens") or 0
        cached = _cached_tokens(usage)
        model = str(kwargs.get("model") or "?")
        t = self._totals[model]
        t["in"] += in_tok
        t["out"] += out_tok
        t["cached"] += cached
        t["calls"] += 1
        pct = (100 * cached // in_tok) if in_tok else 0
        cum_pct = (100 * t["cached"] // t["in"]) if t["in"] else 0
        print(
            f"[usage] model={model} in={in_tok} out={out_tok} cached={cached} cache%={pct} "
            f"| cum in={t['in']} out={t['out']} cached={t['cached']} cache%={cum_pct} calls={t['calls']}",
            flush=True,
        )


usage_logger_instance = UsageLogger()
