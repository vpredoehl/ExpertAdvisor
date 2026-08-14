#!/usr/bin/env python3
"""Focused fail-closed checks for the post-064 H1 ACL composition authority."""

from __future__ import annotations

import csv
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "Scripts/CampaignOperationsH1Post064AclProvenance.py"
CAPTURE = ROOT / "Artifacts/campaign_operations_h1_post064_pre_correction_audit.log"
CLASSIFICATION = ROOT / "LSTM_CampaignOperations_PreH1_Post064_136Tuple_FullProvenance_Classification.tsv"
AUTHORITY = ROOT / "Database/manifests/post064_campaign_operations_h1_acl_composition_authority.tsv"


def run(*args: str, input_text: str | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(TOOL), *args], input=input_text,
                          text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          check=True)


def raw_tuples() -> list[str]:
    return [line for line in CAPTURE.read_text().splitlines()
            if line.startswith("H1A006|explicit_acl|")]


def main() -> None:
    tuples = raw_tuples()
    assert len(tuples) == 136 and len(set(tuples)) == 136
    with tempfile.TemporaryDirectory() as temporary:
        generated = Path(temporary) / "classification.tsv"
        generated_authority = Path(temporary) / "authority.tsv"
        run("generate", str(CAPTURE), str(generated), str(ROOT))
        assert generated.read_bytes() == CLASSIFICATION.read_bytes()
        run("emit-authority", str(generated), str(generated_authority))
        assert generated_authority.read_bytes() == AUTHORITY.read_bytes()

    with CLASSIFICATION.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    assert len(rows) == 136
    counts = {kind: sum(row["classification"] == kind for row in rows)
              for kind in ("A", "B1", "B2", "C")}
    assert counts == {"A": 83, "B1": 35, "B2": 0, "C": 18}

    findings = "\n".join(tuples) + "\n"
    after_full_composition = run("filter-findings", str(AUTHORITY), "--allow-b1",
                                 input_text=findings).stdout.splitlines()
    assert len(after_full_composition) == 18
    assert all("|pqxx|" in line or "|expected_minus_actual|" in line
               for line in after_full_composition)

    # B1 is conditional on later ledger/H2 validation and cannot be accepted
    # by the predecessor-only path.
    predecessor_only = run("filter-findings", str(AUTHORITY), input_text=findings).stdout
    assert "campaign_operations_production_phase5_transactional" in predecessor_only

    extra = ("H1A006|explicit_acl|actual_minus_expected|H1-ACL-UNEXPECTED-OBJECT|"
             "table|LSTM|public|public.unapproved_acl_fixture|"
             "campaign_operations_h1_boundary_authority|pqxx|SELECT|f|f|r\n")
    assert extra in run("filter-findings", str(AUTHORITY), "--allow-b1",
                        input_text=findings + extra).stdout

    audit_source = (ROOT / "Scripts/CampaignOperationsH1DeploymentAudit.sh").read_text()
    assert audit_source.index("CampaignOperationsH2DeploymentAudit.sh") < audit_source.index("filter-findings")
    assert "validate_pre_phase_h_acl_overlay" in audit_source
    print("H1_POST064_ACL_PROVENANCE_TESTS_OK tuples=136 accept=118 reject=18 fail_closed=PASS")


if __name__ == "__main__":
    main()
