#!/usr/bin/env python3
"""Structural/unit regression for Phase 5C generic relationship completion."""
from pathlib import Path
from Tools.RepositoryAgent import retrieval

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

class FakeIndex:
    def resolve_symbol(self, symbol):
        if symbol == "PersistThing":
            return {
                "definitions": [{"file": "Sources/Repo.cpp", "line": 20}],
                "references": [],
                "callers": [{"file": "Sources/App.cpp", "line": 10, "caller": "Run"}],
                "callees": [],
            }
        return {"definitions": [], "references": [], "callers": [], "callees": []}
    def function_for_line(self, filename, line):
        return None
    def callees_of(self, function):
        return []

retrieved = {
    "Sources/App.cpp": {n: ("PersistThing(value);" if n == 10 else "// app") for n in range(8, 13)},
    "Sources/Repo.cpp": {n: ("void PersistThing(Value value) {" if n == 20 else
                              'db.exec("INSERT INTO durable_table ...");' if n == 21 else "// repo")
                         for n in range(18, 24)},
}
category_item = {
    "file": "Sources/App.cpp", "start": 8, "end": 12,
    "excerpt": retrieval.retrieved_evidence_excerpt(retrieved, "Sources/App.cpp", 8, 12),
    "extractor_claim": "caller passes the value to PersistThing",
}
old_get = retrieval.get_repository_index
old_resolved = retrieval._topic_resolved_symbols
try:
    retrieval.get_repository_index = lambda: FakeIndex()
    retrieval._topic_resolved_symbols = lambda topic: []
    candidates = retrieval.generic_relationship_bundle_candidates(
        {"id": "generic", "question": "persist a value", "hints": ""},
        "durable_persistence",
        retrieved,
        {"durable_persistence": [category_item]},
    )
finally:
    retrieval.get_repository_index = old_get
    retrieval._topic_resolved_symbols = old_resolved

assert any(x["file"] == "Sources/App.cpp" for x in candidates), candidates
assert any(x["file"] == "Sources/Repo.cpp" and x["origin"] == "evidence_definition:PersistThing" for x in candidates), candidates

controller = (ROOT / "repository_agent.py").read_text()
assert "bundle_categories =" not in controller
assert "if category in individually_covered:" in controller

retrieval_source = (ROOT / "retrieval.py").read_text()
for forbidden in ("PgModelIO", "PersistObservationIdempotently", "Tensor::Add", "profitability_persistence", "model_persistence_read"):
    assert forbidden not in retrieval_source, forbidden

print("PHASE 5C GENERIC RELATIONSHIP COMPLETION TEST: PASS")
