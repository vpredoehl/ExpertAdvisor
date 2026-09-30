#!/usr/bin/env python3
import tempfile
from pathlib import Path
from . import evidence as e
with tempfile.TemporaryDirectory() as td:
    l=e.VerifiedEvidenceLedger(Path(td)/"l.json"); x="x\n"
    l.record_decision("t","c","T",1,1,x,{"supports":True,"establishes":"fact","reason":"ok"})
    assert l.lookup("t","c","T",1,1,x)["supports"] is True
    l.record_decision("t","c","T",1,1,x,{"supports":False,"verifier_error":True,"reason":"bad"})
    assert next(iter(l.records.values()))["status"]==e.EVIDENCE_ACCEPTED
print("REPOSITORY AGENT M3 LEDGER TEST: PASS")
