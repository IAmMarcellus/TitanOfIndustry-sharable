"""Approved MCP toolsets exposed by the local OpenSage app."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Mapping

from google.adk.tools.mcp_tool.mcp_toolset import (
    StdioConnectionParams,
    StdioServerParameters,
)

from opensage.agents.opensage_agent import OpenSageMCPToolset


@dataclass(frozen=True)
class ApprovedMCPServer:
    name: str
    command: str
    args: tuple[str, ...]


_APPROVED_MCP_REGISTRY: dict[str, ApprovedMCPServer] = {
    "playwright": ApprovedMCPServer(
        name="playwright",
        command="npx",
        args=("-y", "@playwright/mcp@latest"),
    )
}
# Playwright is OPT-IN, not default: its ~21 browser tool schemas are re-sent on every (paid) cloud
# planner turn but used by almost no coding task, so defaulting it off shrinks the planner's prompt.
# Browser-driving agents must set PAPERCLIP_APPROVED_MCP_SERVERS=playwright (the root tool list is also
# what create_subagent(tools_list=["playwright"]) resolves from, so the opt-in re-enables that path).
_DEFAULT_APPROVED_MCP_SERVERS: tuple[str, ...] = ()
_DISABLED_VALUES = {"none", "false", "off"}


def resolve_approved_mcp_servers(
    env: Mapping[str, str] | None = None,
) -> list[ApprovedMCPServer]:
    """Resolve the approved MCP allowlist from environment configuration."""
    source = os.environ if env is None else env
    raw = source.get("PAPERCLIP_APPROVED_MCP_SERVERS")
    value = (raw or "").strip()
    if not value:
        names = list(_DEFAULT_APPROVED_MCP_SERVERS)
    elif value.lower() in _DISABLED_VALUES:
        names = []
    else:
        names = [entry.strip().lower() for entry in value.split(",") if entry.strip()]

    servers: list[ApprovedMCPServer] = []
    seen: set[str] = set()
    for name in names:
        if name in seen:
            continue
        seen.add(name)
        server = _APPROVED_MCP_REGISTRY.get(name)
        if server is not None:
            servers.append(server)
    return servers


def build_playwright_toolset() -> OpenSageMCPToolset:
    """Build the approved Playwright MCP toolset."""
    server = _APPROVED_MCP_REGISTRY["playwright"]
    return OpenSageMCPToolset(
        name=server.name,
        connection_params=StdioConnectionParams(
            server_params=StdioServerParameters(
                command=server.command,
                args=list(server.args),
            )
        ),
        tool_name_prefix=server.name,
    )


def build_approved_mcp_toolsets(
    env: Mapping[str, str] | None = None,
) -> list[OpenSageMCPToolset]:
    toolsets: list[OpenSageMCPToolset] = []
    for server in resolve_approved_mcp_servers(env):
        if server.name == "playwright":
            toolsets.append(build_playwright_toolset())
    return toolsets
