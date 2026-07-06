"""Concurrency throttle for heavy tool calls (agent-local OpenSage plugin).

OpenSage runs subagents in parallel with NO built-in cap, so a single orchestrator turn can fan out
into many concurrent subagent runs — blowing up host RAM/process count and stacking up behind the
GPU. This plugin bounds the number of concurrent HEAVY tool calls (by default the subagent dispatcher
`call_subagent_as_tool`) with a process-wide semaphore.

The LiteLLM proxy's `global_max_parallel_requests` is the AUTHORITATIVE GPU cap (A3); this plugin is
the host-side bound (A2) so the orchestrator can't spawn an unbounded number of subagent runs at once.

Robustness invariants (why the simple acquire-in-before / release-in-after pairing is safe here):
  1. The gated tool (`call_subagent_as_tool`) NEVER raises — it returns an error dict on failure — so
     `after_tool_callback` always fires when `before_tool_callback` did. (Don't gate tools that can
     raise without an equivalent guarantee.)
  2. This plugin MUST be enabled LAST in `[plugins] enabled`: a later plugin returning non-None from
     `before_tool_callback` would short-circuit the tool and skip our `after_tool_callback`, leaking a
     permit. Being last means nothing runs after us.
  3. `_held` guards against over-release (asyncio.Semaphore is unbounded on release()).

Cap is set via the `MAX_ACTIVE_SUBAGENTS` env var (default 2). No vendored code is changed: this file
is discovered from `{agent_dir}/plugins/` and enabled by name. See ../../CLAUDE.md and the plan's
"Concurrency governance" addendum.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any, Optional

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)


class ConcurrencyThrottlePlugin(BasePlugin):
    """Bound concurrent heavy (subagent) tool calls with a process-wide semaphore."""

    def __init__(self, max_active: int = 2, gated_tools: Optional[list[str]] = None) -> None:
        super().__init__(name="concurrency_throttle")
        try:
            self._max = max(1, int(os.environ.get("MAX_ACTIVE_SUBAGENTS", max_active)))
        except (TypeError, ValueError):
            self._max = max(1, max_active)
        self._gated = frozenset(gated_tools or ("call_subagent_as_tool",))
        self._sem = asyncio.Semaphore(self._max)
        self._held = 0  # permits currently held; guards against over-release (asyncio.Semaphore is unbounded)
        logger.info("concurrency throttle: max %d concurrent calls to %s", self._max, sorted(self._gated))

    @staticmethod
    def _tool_name(tool: Any) -> Optional[str]:
        name = getattr(tool, "name", None)
        if isinstance(name, str) and name:
            return name
        name = getattr(tool, "__name__", None)
        return name if isinstance(name, str) and name else None

    async def before_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any
    ) -> None:
        # Block until a permit is free, then let the tool proceed (never short-circuit — we only pace).
        if self._tool_name(tool) in self._gated:
            await self._sem.acquire()
            self._held += 1
        return None

    async def after_tool_callback(
        self, *, tool: Any, tool_args: dict[str, Any], tool_context: Any, result: Any
    ) -> None:
        if self._tool_name(tool) in self._gated and self._held > 0:
            self._held -= 1
            self._sem.release()
        return None
