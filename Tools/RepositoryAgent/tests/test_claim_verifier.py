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

    # Bundle bounds are deterministic and do not invoke the model.
    one = v.verify_source_bundle_claim_semantics(None, None, "topic", "x", [item])
    assert one["supports"] is False and "2-8" in one["reason"]
finally:
    v.run_generation = orig

print("REPOSITORY AGENT PHASE 6C.1 CLAIM VERIFIER TEST: PASS")
