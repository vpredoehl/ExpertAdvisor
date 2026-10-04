#!/usr/bin/env python3
import sys
import types

stub=types.ModuleType("expertadvisor_agent")
stub.list_files=lambda prefix="": []
stub.search=lambda pattern,max_results=100: ""
stub.read_file=lambda name,start=1,end=200: "stub source"
sys.modules.setdefault("expertadvisor_agent",stub)

from ..repository_agent_mcp import SERVER_INFO, TOOLS

def main():
    names={x["name"] for x in TOOLS}
    assert SERVER_INFO["version"]=="1.4.0"
    assert {"verify_source_claim","verify_source_bundle_claim","verified_claims","investigate_source_claim","investigate_source_bundle_claim","investigate_relationship_claim"} <= names
    single=next(x for x in TOOLS if x["name"]=="verify_source_claim")
    assert "excerpt" not in single["inputSchema"]["properties"]
    bundle=next(x for x in TOOLS if x["name"]=="verify_source_bundle_claim")
    assert bundle["inputSchema"]["properties"]["ranges"]["maxItems"]==8
    targeted=next(x for x in TOOLS if x["name"]=="investigate_source_claim")
    assert targeted["inputSchema"]["additionalProperties"] is False
    assert set(targeted["inputSchema"]["required"]) == {"topic_id","topic","claim","file","start","end"}
    targeted_bundle=next(x for x in TOOLS if x["name"]=="investigate_source_bundle_claim")
    schema=targeted_bundle["inputSchema"]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"topic_id","topic","claim","ranges"}
    assert schema["properties"]["ranges"]["minItems"]==2
    assert schema["properties"]["ranges"]["maxItems"]==8
    assert schema["properties"]["ranges"]["items"]["additionalProperties"] is False
    relationship=next(x for x in TOOLS if x["name"]=="investigate_relationship_claim")
    relationship_schema=relationship["inputSchema"]
    assert relationship_schema["additionalProperties"] is False
    assert set(relationship_schema["required"]) == {"topic_id","topic","claim","caller","callee"}
    assert set(relationship_schema["properties"]) == {"topic_id","topic","claim","caller","callee"}
    print("test_repository_agent_mcp_claims: PASS")

if __name__ == "__main__": main()
