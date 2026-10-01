#!/usr/bin/env python3
import tempfile
from pathlib import Path

from ..claim_evidence import VerifiedClaimLedger, CLAIM_ACCEPTED

with tempfile.TemporaryDirectory() as td:
    path = Path(td) / "claims.json"
    ledger = VerifiedClaimLedger(path)
    excerpt = "1: alpha();\n2: beta();"
    accepted = {"supports": True, "establishes": "alpha precedes beta", "reason": "visible"}

    ledger.record_decision("topic", "alpha reaches beta", "x.cpp", 1, 2, excerpt, accepted)
    hit = ledger.lookup("topic", "alpha reaches beta", "x.cpp", 1, 2, excerpt)
    assert hit and hit["supports"] is True and hit["ledger_status"] == CLAIM_ACCEPTED

    # Whitespace-only claim differences intentionally share identity.
    hit2 = ledger.lookup("topic", "  alpha   reaches\n beta ", "x.cpp", 1, 2, excerpt)
    assert hit2 and hit2["supports"] is True

    # Semantically different claim against identical source must not collide.
    assert ledger.lookup("topic", "beta reaches alpha", "x.cpp", 1, 2, excerpt) is None

    # Changed source invalidates the cached decision.
    assert ledger.lookup("topic", "alpha reaches beta", "x.cpp", 1, 2, excerpt + "\n3: gamma();") is None

    # Retryable failure cannot erase an accepted decision.
    ledger.record_decision(
        "topic", "alpha reaches beta", "x.cpp", 1, 2, excerpt,
        {"supports": False, "establishes": "", "reason": "bad json", "verifier_error": True},
    )
    assert ledger.lookup("topic", "alpha reaches beta", "x.cpp", 1, 2, excerpt)["supports"] is True

    bundle = [
        {"file": "x.cpp", "start": 1, "end": 2, "excerpt": excerpt},
        {"file": "y.cpp", "start": 8, "end": 9, "excerpt": "8: beta();\n9: gamma();"},
    ]
    ledger.record_bundle_decision("topic", "alpha reaches gamma", bundle, {
        "supports": True, "establishes": "bundle connects alpha to gamma", "reason": "visible handoff"
    })
    bhit = ledger.lookup_bundle("topic", "alpha reaches gamma", bundle)
    assert bhit and bhit["supports"] is True
    assert ledger.lookup_bundle("topic", "gamma reaches alpha", bundle) is None

print("REPOSITORY AGENT PHASE 6C.1 CLAIM LEDGER TEST: PASS")
