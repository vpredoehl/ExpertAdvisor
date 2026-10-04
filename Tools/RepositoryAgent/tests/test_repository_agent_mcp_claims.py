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
    assert SERVER_INFO["version"]=="1.7.0"
    assert {"verify_source_claim","verify_source_bundle_claim","verified_claims","investigate_source_claim","investigate_source_bundle_claim","investigate_relationship_claim","investigate_relationship_chain_claim","investigate_relationship_set_claim","investigate_symbol"} <= names
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
    chain=next(x for x in TOOLS if x["name"]=="investigate_relationship_chain_claim")
    chain_schema=chain["inputSchema"]
    assert chain_schema["additionalProperties"] is False
    assert set(chain_schema["required"]) == {"topic_id","topic","claim","path"}
    assert set(chain_schema["properties"]) == {"topic_id","topic","claim","path"}
    assert chain_schema["properties"]["path"] == {
        "type":"array", "minItems":3, "maxItems":5, "items":{"type":"string"}
    }
    relationship_set=next(x for x in TOOLS if x["name"]=="investigate_relationship_set_claim")
    set_schema=relationship_set["inputSchema"]
    assert set_schema["additionalProperties"] is False
    assert set(set_schema["required"]) == {"topic_id","topic","claim","relationships"}
    relationships=set_schema["properties"]["relationships"]
    assert relationships["minItems"]==2 and relationships["maxItems"]==5
    assert relationships["items"] == {
        "type":"object",
        "properties":{"caller":{"type":"string"},"callee":{"type":"string"}},
        "required":["caller","callee"],
        "additionalProperties":False,
    }
    symbol=next(x for x in TOOLS if x["name"]=="investigate_symbol")
    symbol_schema=symbol["inputSchema"]
    assert symbol_schema["additionalProperties"] is False
    assert set(symbol_schema["required"]) == {"topic_id","topic","symbol"}
    assert set(symbol_schema["properties"]) == {"topic_id","topic","symbol"}
    assert all(item["minLength"] == 1 for item in symbol_schema["properties"].values())
    print("test_repository_agent_mcp_claims: PASS")

if __name__ == "__main__": main()
