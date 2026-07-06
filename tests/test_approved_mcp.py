"""Unit tests for agent/approved_mcp.py.

Run from repo root:
    uv run --project vendor/opensage-adk --with pytest pytest tests/test_approved_mcp.py -q
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from google.adk.tools.mcp_tool.mcp_toolset import StdioServerParameters

from agent.approved_mcp import build_approved_mcp_toolsets, resolve_approved_mcp_servers
from opensage.agents.opensage_agent import OpenSageMCPToolset


def test_resolve_approved_mcp_servers_defaults_to_empty() -> None:
    # Playwright is opt-in now (see approved_mcp.py): an unset/empty allowlist yields no servers.
    assert resolve_approved_mcp_servers({}) == []


def test_resolve_approved_mcp_servers_opt_in_playwright() -> None:
    servers = resolve_approved_mcp_servers({"PAPERCLIP_APPROVED_MCP_SERVERS": "playwright"})
    assert [server.name for server in servers] == ["playwright"]


def test_resolve_approved_mcp_servers_can_disable() -> None:
    assert resolve_approved_mcp_servers({"PAPERCLIP_APPROVED_MCP_SERVERS": "none"}) == []
    assert resolve_approved_mcp_servers({"PAPERCLIP_APPROVED_MCP_SERVERS": "off"}) == []
    assert resolve_approved_mcp_servers({"PAPERCLIP_APPROVED_MCP_SERVERS": "false"}) == []


def test_build_approved_mcp_toolsets_uses_stdio_playwright() -> None:
    toolsets = build_approved_mcp_toolsets({"PAPERCLIP_APPROVED_MCP_SERVERS": "playwright"})
    assert len(toolsets) == 1
    toolset = toolsets[0]
    assert isinstance(toolset, OpenSageMCPToolset)
    assert toolset.name == "playwright"
    assert toolset.tool_name_prefix == "playwright"

    server_params = toolset._connection_params.server_params
    assert isinstance(server_params, StdioServerParameters)
    assert server_params.command == "npx"
    assert server_params.args == ["-y", "@playwright/mcp@latest"]


def test_build_approved_mcp_toolsets_defaults_to_empty() -> None:
    assert build_approved_mcp_toolsets({}) == []


def test_build_approved_mcp_toolsets_omits_disabled_toolsets() -> None:
    assert build_approved_mcp_toolsets({"PAPERCLIP_APPROVED_MCP_SERVERS": "none"}) == []
