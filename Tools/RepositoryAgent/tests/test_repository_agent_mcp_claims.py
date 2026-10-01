#!/usr/bin/env python3
from ..repository_agent_mcp import SERVER_INFO, TOOLS

def main():
    names={x["name"] for x in TOOLS}
    assert SERVER_INFO["version"]=="1.1.0"
    assert {"verify_source_claim","verify_source_bundle_claim","verified_claims"} <= names
    single=next(x for x in TOOLS if x["name"]=="verify_source_claim")
    assert "excerpt" not in single["inputSchema"]["properties"]
    bundle=next(x for x in TOOLS if x["name"]=="verify_source_bundle_claim")
    assert bundle["inputSchema"]["properties"]["ranges"]["maxItems"]==8
    print("test_repository_agent_mcp_claims: PASS")

if __name__ == "__main__": main()
