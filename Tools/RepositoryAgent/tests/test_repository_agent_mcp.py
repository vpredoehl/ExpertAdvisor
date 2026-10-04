#!/usr/bin/env python3
"""Deterministic tests for the RepositoryAgent MCP adapter."""
from __future__ import annotations
import json
import sys
import types

stub=types.ModuleType("expertadvisor_agent")
stub.list_files=lambda prefix="": []
stub.search=lambda pattern,max_results=100: ""
stub.read_file=lambda name,start=1,end=200: "stub source"
sys.modules.setdefault("expertadvisor_agent",stub)

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
    assert "investigate_relationship_claim" in payload["operations"]
    assert "investigate_relationship_chain_claim" in payload["operations"]
    bad = server._handle_request({
        "jsonrpc":"2.0","id":4,"method":"tools/call",
        "params":{"name":"shell","arguments":{"command":"pwd"}}
    })
    assert bad["result"]["isError"] is True
    closed = server._handle_request({
        "jsonrpc":"2.0","id":5,"method":"tools/call",
        "params":{"name":"investigate_relationship_claim","arguments":{
            "topic_id":"t","topic":"topic","claim":"claim",
            "caller":"caller","callee":"callee","query":"forbidden"
        }}
    })
    assert closed["result"]["isError"] is True
    assert "unexpected tool arguments" in closed["result"]["content"][0]["text"]
    chain_closed = server._handle_request({
        "jsonrpc":"2.0","id":6,"method":"tools/call",
        "params":{"name":"investigate_relationship_chain_claim","arguments":{
            "topic_id":"t","topic":"topic","claim":"claim",
            "path":["A","B","C"],"caller":"forbidden"
        }}
    })
    assert chain_closed["result"]["isError"] is True
    assert "unexpected tool arguments" in chain_closed["result"]["content"][0]["text"]
    missing_path = server._handle_request({
        "jsonrpc":"2.0","id":7,"method":"tools/call",
        "params":{"name":"investigate_relationship_chain_claim","arguments":{
            "topic_id":"t","topic":"topic","claim":"claim"
        }}
    })
    assert missing_path["result"]["isError"] is True
    assert "missing required tool arguments: path" in missing_path["result"]["content"][0]["text"]
    assert server._handle_request({
        "jsonrpc":"2.0","method":"notifications/initialized","params":{}
    }) is None
    print("REPOSITORY AGENT MCP ADAPTER TEST: PASS")

if __name__ == "__main__":
    main()
