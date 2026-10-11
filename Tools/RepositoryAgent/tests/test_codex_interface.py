#!/usr/bin/env python3
import json, sys, tempfile, types
from pathlib import Path
stub=types.ModuleType("expertadvisor_agent")
stub.list_files=lambda prefix="": ["Sources/A.cpp"]
stub.search=lambda pattern,max_results=100: f"Sources/A.cpp:2: {pattern}"
stub.read_file=lambda name,start=1,end=200: "\n".join(f"{n}: line{n}" for n in range(start,end+1))
sys.modules["expertadvisor_agent"]=stub
from Tools.RepositoryAgent import codex_interface
codex_interface.list_files = stub.list_files
codex_interface.search = stub.search
codex_interface.read_file = stub.read_file
CodexRepositoryInterface = codex_interface.CodexRepositoryInterface
iface=CodexRepositoryInterface()
cap=iface.dispatch({"op":"capabilities"})
assert cap["read_only"] is True
assert "shell" in cap["forbidden_capabilities"]
assert iface.dispatch({"op":"search","pattern":"needle"})["pattern"]=="needle"
r=iface.dispatch({"op":"read","file":"Sources/A.cpp","start":2,"end":3})
assert r["start"]==2 and r["end"]==3
# Targeted claim verification must fetch only the requested allowed range; it
# must not construct the repository-wide structural index just for provenance.
iface._idx=lambda: (_ for _ in ()).throw(AssertionError("targeted claim built index"))
item=iface._claim_item({"file":"Sources/A.cpp","start":2,"end":3})
assert item["file"]=="Sources/A.cpp" and "2: line2" in item["excerpt"]
for bad in ("/tmp/outside.cpp", "../outside.cpp", "Sources/../outside.cpp", "Sources\\outside.cpp"):
    try: iface._claim_item({"file":bad,"start":1,"end":1}); raise AssertionError("malformed claim path accepted")
    except ValueError: pass
for bad_start, bad_end in (("not-a-line", 1), (2, 1), (1, 501)):
    try: iface._claim_item({"file":"Sources/A.cpp","start":bad_start,"end":bad_end}); raise AssertionError("invalid claim range accepted")
    except ValueError: pass
try: iface.dispatch({"op":"shell","command":"rm -rf /"}); raise AssertionError("shell accepted")
except ValueError: pass
try: iface.dispatch({"op":"read","file":"Sources/A.cpp","start":1,"end":999}); raise AssertionError("oversized read accepted")
except Exception: pass
with tempfile.TemporaryDirectory() as td:
    class Runtime:
        def __init__(self): self.calls=0
        def verify_bundle_claim(self, topic, claim, items):
            self.calls += 1
            return {"supports":True,"establishes":"visible cross-file handoff","reason":"test","model_turns":1}

    runtime=Runtime()
    claim_ledger=Path(td)/"claims.json"
    direct=CodexRepositoryInterface(claim_ledger_path=str(claim_ledger),claim_runtime=runtime)
    direct._idx=lambda: (_ for _ in ()).throw(AssertionError("bundle claim built index"))
    bundle=direct.dispatch({"op":"investigate_source_bundle_claim","topic_id":"topic","topic":"topic","claim":"cross-file handoff","ranges":[{"file":"Sources/B.cpp","start":4,"end":4},{"file":"Sources/A.cpp","start":2,"end":2}]})
    assert bundle["manifest"]["metrics"]["repository_read_count"]==2
    assert runtime.calls==1

    p=Path(td)/"ledger.json"
    p.write_text(json.dumps({"schema_version":2,"verification_identity":{"cache_namespace":"ExpertAdvisor","verifier_identity":"expertadvisor_semantic_verifier_v1","verifier_schema_version":1},"records":{},"bundle_records":{}}))
    li=CodexRepositoryInterface(ledger_path=str(p))
    assert li.dispatch({"op":"ledger_records"})["matched"]==0
print("REPOSITORY AGENT CODEX INTERFACE TEST: PASS")
