#!/usr/bin/env python3
import json, sys, tempfile, types
from pathlib import Path
stub=types.ModuleType("expertadvisor_agent")
stub.list_files=lambda prefix="": ["Sources/A.cpp"]
stub.search=lambda pattern,max_results=100: f"Sources/A.cpp:2: {pattern}"
stub.read_file=lambda name,start=1,end=200: "\n".join(f"{n}: line{n}" for n in range(start,end+1))
sys.modules["expertadvisor_agent"]=stub
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
iface=CodexRepositoryInterface()
cap=iface.dispatch({"op":"capabilities"})
assert cap["read_only"] is True
assert "shell" in cap["forbidden_capabilities"]
assert iface.dispatch({"op":"search","pattern":"needle"})["pattern"]=="needle"
r=iface.dispatch({"op":"read","file":"Sources/A.cpp","start":2,"end":3})
assert r["start"]==2 and r["end"]==3
try: iface.dispatch({"op":"shell","command":"rm -rf /"}); raise AssertionError("shell accepted")
except ValueError: pass
try: iface.dispatch({"op":"read","file":"Sources/A.cpp","start":1,"end":999}); raise AssertionError("oversized read accepted")
except Exception: pass
with tempfile.TemporaryDirectory() as td:
    p=Path(td)/"ledger.json"
    p.write_text(json.dumps({"schema_version":2,"verification_identity":{"cache_namespace":"ExpertAdvisor","verifier_identity":"expertadvisor_semantic_verifier_v1","verifier_schema_version":1},"records":{},"bundle_records":{}}))
    li=CodexRepositoryInterface(ledger_path=str(p))
    assert li.dispatch({"op":"ledger_records"})["matched"]==0
print("REPOSITORY AGENT CODEX INTERFACE TEST: PASS")
