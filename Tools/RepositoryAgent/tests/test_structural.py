#!/usr/bin/env python3
import ast, hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent
ORACLE=HERE.parent/"expertadvisor_repository_agent_ledger_hardened.py"
def defs(p):
    t=ast.parse(Path(p).read_text())
    return {(type(n).__name__,n.name) for n in t.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef,ast.ClassDef))}
o=defs(ORACLE); c=defs(HERE/"repository_agent.py"); e=defs(HERE/"evidence.py"); v=defs(HERE/"verifier.py"); r=defs(HERE/"retrieval.py")
moved=(e|v|r)
assert (c|moved)==o, ("package/oracle behavior surface differs",sorted(o-(c|moved)),sorted((c|moved)-o))
assert not (c & moved), ("moved definitions remain in controller",sorted(c&moved))
cs=(HERE/"repository_agent.py").read_text()
assert "PHASE2D_METRICS" not in cs
assert "REPOSITORY_AGENT_METRICS = {" in cs
for x in ("from .evidence import (","from .verifier import (","from .retrieval import (","from .scheduler_architecture_benchmark import ("):
    assert x in cs
rs=(HERE/"retrieval.py").read_text()
assert "MAX_READ_LINES" in cs
assert "from expertadvisor_agent import list_files, search, read_file" in rs
assert "from .scheduler_architecture_benchmark import TOPIC_NAVIGATION" in rs
assert "MAX_TOOL_OUTPUT = 30000" in rs
assert "MAX_READ_LINES = 500" in rs
for x in ("def topic_index_navigation(","def record_retrieved_lines(","def retrieved_evidence_excerpt(","def generic_relationship_bundle_candidates("):
    assert x in rs
print("REPOSITORY AGENT M3 STRUCTURAL TEST: PASS")
