#!/usr/bin/env python3
from . import verifier as v
orig=v.run_generation
try:
    it=iter(["bad",'{"supports":true,"establishes":"fact","reason":"ok"}'])
    v.run_generation=lambda *a,**k: next(it)
    z=v.verify_evidence_semantics(None,None,{"title":"t"},"capacity_or_dispatch_gate","x","1 | x")
    assert z["supports"] is True
finally: v.run_generation=orig
print("REPOSITORY AGENT M3 VERIFIER TEST: PASS")
