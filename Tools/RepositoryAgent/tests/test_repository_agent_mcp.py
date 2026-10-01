#!/usr/bin/env python3
"""Deterministic tests for the RepositoryAgent MCP adapter."""
from __future__ import annotations
import json
from Tools.RepositoryAgent.repository_agent_mcp import StdioMCPServer, TOOLS

def main() -> None:
    server = StdioMCPServer()
    init = server._handle_request({
        "jsonrpc":"2.0","id":1,"method":"initialize",
        "params":{"protocolVersion":"2024-11-05","capabilities":{},
                  "clientInfo":{"name":"test","version":"1"}}
    })
    assert init["result"]["serverInfo"]["name"] == "expertadvisor-repository-agent"
    listed = server._handle_request({"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}})
    names = [x["name"] for x in listed["result"]["tools"]]
    assert names == [x["name"] for x in TOOLS]
    caps = server._handle_request({
        "jsonrpc":"2.0","id":3,"method":"tools/call",
        "params":{"name":"capabilities","arguments":{}}
    })
    payload = json.loads(caps["result"]["content"][0]["text"])
    assert payload["read_only"] is True
    assert "repository_write" in payload["forbidden_capabilities"]
    assert "shell" in payload["forbidden_capabilities"]
    bad = server._handle_request({
        "jsonrpc":"2.0","id":4,"method":"tools/call",
        "params":{"name":"shell","arguments":{"command":"pwd"}}
    })
    assert bad["result"]["isError"] is True
    assert server._handle_request({
        "jsonrpc":"2.0","method":"notifications/initialized","params":{}
    }) is None
    print("REPOSITORY AGENT MCP ADAPTER TEST: PASS")

if __name__ == "__main__":
    main()
