#!/usr/bin/env python3
import sys, types
_stub = types.ModuleType("expertadvisor_agent")
_stub.list_files = lambda prefix="": []
_stub.search = lambda pattern, max_results=100: ""
_stub.read_file = lambda name, start=1, end=200: ""
sys.modules.setdefault("expertadvisor_agent", _stub)
from . import retrieval as r
assert hasattr(r, "RETRIEVED_LINE_RE")
assert isinstance(r.TOPIC_NAVIGATION, dict)
assert r.MAX_TOOL_OUTPUT == 30000
assert r.MAX_READ_LINES == 500
assert r.execute_tool({"tool":"search","pattern":"spawn"}) == "TOOL RESULT: no matches"
assert r.execute_tool({"tool":"list_files","prefix":"Sources"}) == "TOOL RESULT: no matches"
assert r.execute_tool({"tool":"read","file":"Sources/T.cpp","start":1,"end":2}) == "TOOL RESULT: no matches"

# record_retrieved_lines / retrieved_evidence_excerpt preserve exact line provenance.
retrieved={}
result="  10 | alpha\n  11 | beta\n  12 | gamma"
r.record_retrieved_lines(retrieved,"Sources/T.cpp",result)
x=r.retrieved_evidence_excerpt(retrieved,"Sources/T.cpp",10,12)
assert x is not None
assert "10 | alpha" in x and "12 | gamma" in x
assert r.retrieved_evidence_excerpt(retrieved,"Sources/T.cpp",9,12) is None

# normalize_call keeps allowed tool shape deterministic.
assert r.normalize_call({"tool":"read","file":"Sources/T.cpp","start":10,"end":12}) == (
    "read", "Sources/T.cpp", 10, 12
)
assert r.normalize_call({"tool":"search","pattern":"spawn"}) == ("search", "spawn")
assert r.normalize_call({"tool":"list_files","prefix":"Sources"}) == ("list_files", "Sources")

# Generic relationship assembly remains scheduler-agnostic in its own module.
text=__import__("pathlib").Path(r.__file__).read_text()
for forbidden in ("ProductionSchedulerDaemon.cpp","SchedulerPolicy.cpp","GlobalExperimentControl.cpp"):
    assert forbidden not in text

print("REPOSITORY AGENT M3 RETRIEVAL TEST: PASS")
