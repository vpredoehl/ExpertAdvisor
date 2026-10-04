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
    assert "investigate_relationship_set_claim" in payload["operations"]
    assert "discover_catalog_targets" in payload["operations"]
    assert "investigate_symbol" in payload["operations"]
    assert "investigate_subsystem" in payload["operations"]
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
    set_closed = server._handle_request({
        "jsonrpc":"2.0","id":8,"method":"tools/call",
        "params":{"name":"investigate_relationship_set_claim","arguments":{
            "topic_id":"t","topic":"topic","claim":"claim",
            "relationships":[
                {"caller":"A","callee":"B","extra":"forbidden"},
                {"caller":"C","callee":"D"},
            ]
        }}
    })
    assert set_closed["result"]["isError"] is True
    assert "unexpected tool arguments.relationships[0] fields: extra" in set_closed["result"]["content"][0]["text"]
    set_top_closed = server._handle_request({
        "jsonrpc":"2.0","id":9,"method":"tools/call",
        "params":{"name":"investigate_relationship_set_claim","arguments":{
            "topic_id":"t","topic":"topic","claim":"claim",
            "relationships":[{"caller":"A","callee":"B"},{"caller":"C","callee":"D"}],
            "query":"forbidden"
        }}
    })
    assert set_top_closed["result"]["isError"] is True
    assert "unexpected tool arguments: query" in set_top_closed["result"]["content"][0]["text"]
    symbol_closed = server._handle_request({
        "jsonrpc":"2.0","id":12,"method":"tools/call",
        "params":{"name":"investigate_symbol","arguments":{
            "topic_id":"t","topic":"topic","symbol":"Demo::f","claim":"forbidden"
        }}
    })
    assert symbol_closed["result"]["isError"] is True
    assert "unexpected tool arguments: claim" in symbol_closed["result"]["content"][0]["text"]
    symbol_empty = server._handle_request({
        "jsonrpc":"2.0","id":13,"method":"tools/call",
        "params":{"name":"investigate_symbol","arguments":{
            "topic_id":"t","topic":"topic","symbol":""
        }}
    })
    assert symbol_empty["result"]["isError"] is True
    assert "must contain at least 1 characters" in symbol_empty["result"]["content"][0]["text"]
    subsystem_closed = server._handle_request({
        "jsonrpc":"2.0","id":14,"method":"tools/call",
        "params":{"name":"investigate_subsystem","arguments":{
            "topic_id":"t","topic":"topic","subsystem":"Sources/SchedulerCore","symbol":"forbidden"
        }}
    })
    assert subsystem_closed["result"]["isError"] is True
    assert "unexpected tool arguments: symbol" in subsystem_closed["result"]["content"][0]["text"]
    subsystem_missing_files = server._handle_request({
        "jsonrpc":"2.0","id":15,"method":"tools/call",
        "params":{"name":"investigate_subsystem","arguments":{
            "topic_id":"t","topic":"topic","subsystem":"Sources/SchedulerCore"
        }}
    })
    assert subsystem_missing_files["result"]["isError"] is True
    assert "missing required tool arguments: files" in subsystem_missing_files["result"]["content"][0]["text"]
    for request_id, relationships in ((10, [{"caller":"A","callee":"B"}]),
                                      (11, [{"caller":str(n),"callee":"B"} for n in range(6)])):
        count_error = server._handle_request({
            "jsonrpc":"2.0","id":request_id,"method":"tools/call",
            "params":{"name":"investigate_relationship_set_claim","arguments":{
                "topic_id":"t","topic":"topic","claim":"claim","relationships":relationships
            }}
        })
        assert count_error["result"]["isError"] is True
        assert "tool arguments.relationships must contain" in count_error["result"]["content"][0]["text"]
    assert server._handle_request({
        "jsonrpc":"2.0","method":"notifications/initialized","params":{}
    }) is None
    print("REPOSITORY AGENT MCP ADAPTER TEST: PASS")

if __name__ == "__main__":
    main()
