"""Periodic / crash-resilient session snapshotting (agent-local OpenSage plugin).

OpenSage only snapshots the ADK session on a *graceful* exit; a crash or `kill -9` loses the live
session. This plugin reuses OpenSage's OWN snapshot writer (`_persist_web_session_snapshot_async`)
on every Nth event, so a crash loses at most the last few events. The snapshot is byte-identical to
the graceful-exit one, hence fully compatible with `make opensage RESUME=last` / `RESUME=<id>`.

No vendored code is changed: this file is discovered from `{agent_dir}/plugins/` and enabled by name
in `[plugins] enabled`. Cadence is tuned with the `SNAPSHOT_EVERY_N` env var (default 1 = every event;
raise it for very long sessions). See ../../CLAUDE.md and the plan's "Periodic snapshotting" addendum.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

from google.adk.plugins.base_plugin import BasePlugin

logger = logging.getLogger(__name__)

# The OpenSage agent dir (the `--agent` path) = the parent of this `plugins/` dir. The snapshot
# writer derives the store dir name (`<agent_name>_<session_id>`) from it.
_AGENT_DIR = str(Path(__file__).resolve().parents[1])


class PeriodicSnapshotPlugin(BasePlugin):
    """Persist the live ADK session every Nth event so a crash is recoverable via `--resume`."""

    def __init__(self, every_n_events: int = 1) -> None:
        super().__init__(name="periodic_snapshot")
        try:
            self._n = max(1, int(os.environ.get("SNAPSHOT_EVERY_N", every_n_events)))
        except (TypeError, ValueError):
            self._n = 1
        self._count = 0

    async def on_event_callback(self, *, invocation_context: Any, event: Any) -> None:
        self._count += 1
        if self._count % self._n != 0:
            return None
        try:
            # Lazy import so an upstream API change degrades to "no periodic snapshot", not a crash.
            from opensage.cli.opensage_cli import _persist_web_session_snapshot_async
            from opensage.session.opensage_session import get_opensage_session

            s = invocation_context.session
            await _persist_web_session_snapshot_async(
                session_id=s.id,
                app_name=s.app_name,
                user_id=s.user_id,
                agent_dir=_AGENT_DIR,
                session_service=invocation_context.session_service,
                opensage_session=get_opensage_session(s.id),
            )
        except Exception as exc:  # snapshotting must NEVER break the agent run
            logger.warning("periodic snapshot failed (continuing): %s", exc)
        return None
