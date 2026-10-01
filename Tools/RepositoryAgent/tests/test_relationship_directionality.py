#!/usr/bin/env python3
"""Phase 5C.1: category-directed relationship candidate ranking."""
from Tools.RepositoryAgent import retrieval

class FakeIndex:
    def resolve_symbol(self, symbol):
        facts = {
            "CreateModel": ({"file": "Sources/Model.cpp", "line": 10}, {"file": "Sources/App.cpp", "line": 5, "caller": "Run", "callee": "CreateModel"}),
            "LoadModel": ({"file": "Sources/Store.cpp", "line": 30}, {"file": "Sources/App.cpp", "line": 40, "caller": "Resume", "callee": "LoadModel"}),
            "ConsumeTrainingInput": ({"file": "Sources/Train.cpp", "line": 70}, {"file": "Sources/Train.cpp", "line": 60, "caller": "Train", "callee": "ConsumeTrainingInput"}),
        }
        if symbol not in facts:
            return {"definitions": [], "references": [], "callers": [], "callees": []}
        definition, caller = facts[symbol]
        return {"definitions": [definition], "references": [], "callers": [caller], "callees": []}
    def function_for_line(self, filename, line): return None
    def callees_of(self, function): return []

retrieved = {
    "Sources/Model.cpp": {n: ("Model CreateModel() {" if n == 10 else "// model") for n in range(8, 14)},
    "Sources/Store.cpp": {n: ("Model LoadModel() {" if n == 30 else "// durable read") for n in range(28, 34)},
    "Sources/App.cpp": {n: ("CreateModel();" if n == 5 else "LoadModel();" if n == 40 else "// app") for n in list(range(3, 8))+list(range(38, 43))},
    "Sources/Train.cpp": {n: ("ConsumeTrainingInput(batch);" if n == 60 else "void ConsumeTrainingInput(Batch batch) {" if n == 70 else "// train") for n in list(range(58, 63))+list(range(68, 74))},
}

old_index = retrieval.get_repository_index
old_topic = retrieval._topic_resolved_symbols
try:
    retrieval.get_repository_index = lambda: FakeIndex()
    retrieval._topic_resolved_symbols = lambda topic: []
    ranked = retrieval._retrieved_category_symbols(
        {"id":"p", "title":"model persistence", "question":"load persisted model for use", "hints":"find the read reconstruction path"},
        "model_persistence_read", retrieved,
    )
    assert ranked[0][0] == "LoadModel", ranked

    candidates = retrieval.generic_relationship_bundle_candidates(
        {"id":"p", "title":"model persistence", "question":"load persisted model for use", "hints":"find the read reconstruction path"},
        "model_persistence_read", retrieved, {}, max_items=4,
    )
    assert any(x["origin"] == "category_definition:LoadModel" for x in candidates), candidates

    ranked_train = retrieval._retrieved_category_symbols(
        {"id":"t", "title":"training input construction", "question":"constructed training input consumed by training model", "hints":"follow input to model consumer"},
        "training_model_consumption", retrieved,
    )
    assert ranked_train[0][0] == "ConsumeTrainingInput", ranked_train
finally:
    retrieval.get_repository_index = old_index
    retrieval._topic_resolved_symbols = old_topic

source = open(retrieval.__file__).read()
for forbidden in ("PgModelIO", "PersistObservationIdempotently", "Tensor::Add", "profitability_persistence", "model_persistence_read"):
    assert forbidden not in source, forbidden
print("PHASE 5C.1 RELATIONSHIP DIRECTIONALITY TEST: PASS")
