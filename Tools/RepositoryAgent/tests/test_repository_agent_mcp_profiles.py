#!/usr/bin/env python3
"""Profile-admission tests for the RepositoryAgent MCP adapter."""
from __future__ import annotations

import sys
import types

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: "stub source"
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.repository_agent_mcp import (
    TOOL_PROFILES,
    TOOLS,
    StdioMCPServer,
)


class RecordingInterface:
    def __init__(self) -> None:
        self.requests: list[dict[str, object]] = []

    def dispatch(self, request: dict[str, object]) -> dict[str, object]:
        self.requests.append(request)
        return {"operation": request["op"]}


def call(server: StdioMCPServer, request_id: int, name: str,
         arguments: dict[str, object]) -> dict[str, object]:
    response = server._handle_request({
        "jsonrpc": "2.0", "id": request_id, "method": "tools/call",
        "params": {"name": name, "arguments": arguments},
    })
    assert response is not None
    return response


def listed_names(server: StdioMCPServer, request_id: int, params: dict[str, object] | None = None) -> list[str]:
    response = server._handle_request({
        "jsonrpc": "2.0", "id": request_id, "method": "tools/list",
        "params": params or {},
    })
    assert response is not None
    return [tool["name"] for tool in response["result"]["tools"]]


def main() -> None:
    # The default remains the full pre-profile surface, and a raw tool dispatches.
    full = StdioMCPServer()
    full_names = listed_names(full, 1)
    assert full_names == [tool["name"] for tool in TOOLS]
    assert {"list_files", "source_excerpt", "resolve_symbol", "function_for_line"} <= set(full_names)
    full.iface = RecordingInterface()
    raw = call(full, 2, "list_files", {"prefix": "Tools/RepositoryAgent"})
    assert raw["result"]["isError"] is False
    assert full.iface.requests == [{"op": "list_files", "prefix": "Tools/RepositoryAgent"}]

    # Advertisement and dispatch both use the same single profile-policy entry.
    assisted = StdioMCPServer(profile="codex_assisted")
    assisted_names = listed_names(assisted, 3)
    assert assisted_names == list(TOOL_PROFILES["codex_assisted"])
    assert assisted_names == [
        "investigate_source_claim", "investigate_source_bundle_claim",
        "investigate_relationship_claim", "investigate_relationship_chain_claim",
        "investigate_relationship_set_claim", "investigate_symbol",
        "investigate_subsystem",
    ]
    for hidden in ("list_files", "source_excerpt", "resolve_symbol", "function_for_line"):
        assert hidden not in assisted_names
        rejected = call(assisted, 4, hidden, {})
        assert rejected["result"]["isError"] is True
        assert rejected["result"]["content"][0]["text"] == f"unsupported tool: {hidden}"

    assisted.iface = RecordingInterface()
    bounded = call(assisted, 5, "investigate_symbol", {
        "topic_id": "profile-test", "topic": "MCP profile", "symbol": "Demo::symbol",
    })
    assert bounded["result"]["isError"] is False
    assert assisted.iface.requests == [{
        "op": "investigate_symbol", "topic_id": "profile-test",
        "topic": "MCP profile", "symbol": "Demo::symbol",
    }]

    # Request fields cannot elevate the process-selected profile.
    assert listed_names(assisted, 6, {"profile": "full"}) == assisted_names
    elevation = call(assisted, 7, "list_files", {"profile": "full"})
    assert elevation["result"]["isError"] is True
    assert elevation["result"]["content"][0]["text"] == "unsupported tool: list_files"
    assert assisted.tool_profile == "codex_assisted"

    try:
        StdioMCPServer(profile="not-a-profile")
    except ValueError as exc:
        assert "invalid MCP tool profile" in str(exc)
    else:
        raise AssertionError("invalid MCP profile must fail closed during construction")

    print("test_repository_agent_mcp_profiles: PASS")


if __name__ == "__main__":
    main()
