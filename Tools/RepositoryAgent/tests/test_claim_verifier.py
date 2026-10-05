#!/usr/bin/env python3
from .. import claim_verifier as v

orig = v.run_generation
try:
    outputs = iter(["not json", '{"supports":true,"establishes":"visible handoff","reason":"source shows it"}'])
    v.run_generation = lambda *a, **k: next(outputs)
    item = {"file": "x.cpp", "start": 10, "end": 20, "excerpt": "10: a();\n20: b();"}
    verdict = v.verify_source_claim_semantics(None, None, "topic", "a reaches b", item)
    assert verdict["supports"] is True
    assert verdict["establishes"] == "visible handoff"

    # The verifier prompt must distinguish visible lexical data flow from
    # prohibited inference of a missing link.
    assert "Directly visible syntactic relationships are evidence" in v.CLAIM_VERIFY_SYSTEM
    assert "assigns a call result to a local variable" in v.CLAIM_VERIFY_SYSTEM
    assert "passes that same local variable to later calls" in v.CLAIM_VERIFY_SYSTEM

    # Bundle bounds are deterministic and do not invoke the model.
    one = v.verify_source_bundle_claim_semantics(None, None, "topic", "x", [item])
    assert one["supports"] is False and "2-8" in one["reason"]

    # Relationship mode is deliberately separate from the generic bundle rules.
    assert "Verify every listed caller->callee call independently" in v.RELATIONSHIP_CLAIM_VERIFY_SYSTEM
    assert "Do not require an additional data-flow or control-flow handoff" in v.RELATIONSHIP_CLAIM_VERIFY_SYSTEM

    # The relationship mode retains the same bounded retry and fail-closed
    # handling for malformed verifier output.
    outputs = iter(["not json", "still not json"])
    v.run_generation = lambda *a, **k: next(outputs)
    relationship = v.verify_relationship_bundle_claim_semantics(
        None, None, "topic", "a calls b", [{"caller": "A", "callee": "B", "evidence": [item]}]
    )
    assert relationship["supports"] is False
    assert relationship["verifier_error"] is True
    assert relationship["model_turns"] == 2
finally:
    v.run_generation = orig

print("REPOSITORY AGENT PHASE 6C.1 CLAIM VERIFIER TEST: PASS")
