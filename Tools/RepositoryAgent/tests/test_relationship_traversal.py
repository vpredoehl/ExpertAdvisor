#!/usr/bin/env python3
"""Phase 5D: bounded generic multi-hop relationship traversal."""
from Tools.RepositoryAgent import retrieval

class FakeIndex:
    def resolve_symbol(self, symbol):
        facts = {
            "Bridge": {
                "definitions": [{"file": "Sources/Bridge.cpp", "line": 20}],
                "references": [],
                "callers": [],
                "callees": [],
            },
            "ConsumeResult": {
                "definitions": [{"file": "Sources/Consumer.cpp", "line": 50}],
                "references": [],
                "callers": [],
                "callees": [],
            },
        }
        return facts.get(symbol, {"definitions": [], "references": [], "callers": [], "callees": []})
    def function_for_line(self, filename, line): return None
    def callees_of(self, function): return []

retrieved = {
    "Sources/App.cpp": {
        n: ("Bridge(value);" if n == 10 else "// app")
        for n in range(8, 13)
    },
}

def fake_execute(call):
    filename = call["file"]
    start, end = call["start"], call["end"]
    rows = []
    for n in range(start, end + 1):
        if filename == "Sources/Bridge.cpp" and n == 20:
            text = "void Bridge(Value value) { ConsumeResult(value); }"
        elif filename == "Sources/Consumer.cpp" and n == 50:
            text = "void ConsumeResult(Value value) { durable_use(value); }"
        else:
            text = "// source"
        rows.append(f"{n:6d} | {text}")
    return "\n".join(rows)

old_index = retrieval.get_repository_index
old_execute = retrieval.execute_tool
try:
    retrieval.get_repository_index = lambda: FakeIndex()
    retrieval.execute_tool = fake_execute
    admitted = retrieval._bounded_relationship_traversal(
        {
            "id": "generic",
            "title": "result consumption",
            "question": "follow a produced result to the consumer",
            "hints": "trace the relationship",
        },
        "result_consumption",
        retrieved,
        max_depth=2,
        max_reads=2,
        max_frontier=8,
    )
finally:
    retrieval.get_repository_index = old_index
    retrieval.execute_tool = old_execute

assert len(admitted) == 2, admitted
assert any(x["file"] == "Sources/Bridge.cpp" for x in admitted), admitted
assert any(x["file"] == "Sources/Consumer.cpp" for x in admitted), admitted
assert 20 in retrieved["Sources/Bridge.cpp"]
assert 50 in retrieved["Sources/Consumer.cpp"]
assert all(x["origin"].startswith("traversal_") for x in admitted), admitted

source = open(retrieval.__file__).read()
for forbidden in ("PgModelIO", "PersistObservationIdempotently", "Tensor::Add", "profitability_persistence", "model_persistence_read"):
    assert forbidden not in source, forbidden

print("PHASE 5D BOUNDED RELATIONSHIP TRAVERSAL TEST: PASS")
