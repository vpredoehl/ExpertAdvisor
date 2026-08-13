#!/usr/bin/env python3
"""Generate the H1 legacy trust-path inventory from executable source checks."""
from __future__ import annotations

import ast
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "Tests/fixtures/CampaignOperationsH1LegacyPathInventory.tsv"
FIELDS = ("entry_id", "legacy_class", "entry_point", "disposition", "replacement", "diagnostic")

STATIC_REMOVALS = (
    ("LEGACY-AUTHORITY-READER", "authority", "CampaignOperationsH1EvidenceGraph.load_graph", "replaced", "CampaignOperationsH1EvidenceAuthority.validate_normative_authority-v2", "H1A102"),
    ("LEGACY-AUTHORITY-VERSION", "authority", "h1-normative-authority-v1", "fail_closed", "h1-normative-authority-v2", "H1A102"),
    ("LEGACY-RECEIPT-EMITTER", "validator_receipt", "CampaignOperationsH1EvidenceAuthority.emit_execution_receipts", "fail_closed", "CampaignOperationsH1TrustedRunner.TrustedValidatorRunner.execute", "H1A510"),
    ("LEGACY-RECEIPT-VALIDATOR", "validator_receipt", "CampaignOperationsH1EvidenceAuthority.validate_execution_receipts", "fail_closed", "CampaignOperationsH1TrustedRunner.TrustedValidatorRunner.validate", "H1A511"),
    ("LEGACY-RAW-CONTRACT", "raw_evidence", "marker-and-ad-hoc-log-readers", "fail_closed", "h1-raw-execution-evidence-v2", "H1T102"),
    ("LEGACY-RUNTIME-V1", "runtime_record", "CampaignOperationsPhaseH1TraceabilityTests.sh", "fail_closed", "h1-runtime-result-v2", "H1T006"),
    ("LEGACY-ACL-COAUTHORED", "acl_catalog", "h1-acl-catalog-runtime-v2", "fail_closed", "h1-acl-catalog-runtime-v3", "H1V402"),
    ("LEGACY-PATH-REOPEN", "artifact", "CampaignOperationsH1RestoreEvidence.pathname-cli", "fail_closed", "CampaignOperationsH1RestoreEvidence.validate_snapshot_bytes", "H1R002"),
    ("LEGACY-GRAPH-SHORTCUT", "graph", "seven-node-readiness-graph", "replaced", "full-provenance-boundary-status", "H1T108"),
    ("LEGACY-READINESS-SHORTCUT", "readiness", "legacy-counts-ready-predicate", "replaced", "CampaignOperationsH1TrustModel.evaluate_readiness", "H1T001"),
    ("LEGACY-REPORT-GENERATION", "report_generation", "path-and-marker-report-writer", "replaced", "CampaignOperationsH1EvidenceGraph.materialize_provenance", "H1R006"),
    ("LEGACY-ACL-DEFAULT-GENERATION", "acl_catalog", "expected-derived-acl-observation", "replaced", "CampaignOperationsH1AclCatalogGenerator.direct-catalog-query", "H1V403"),
    ("LEGACY-RESTORE-VALIDATION", "restore_validation", "pathname-restore-validator", "replaced", "CampaignOperationsH1RestoreEvidence.validate_snapshot_bytes", "H1V801"),
    ("LEGACY-DIRECT-VALIDATOR", "validator", "CampaignOperationsH1EvidenceValidator.generate-or-validate", "fail_closed", "CampaignOperationsH1TrustedRunner.TrustedValidatorRunner.execute", "H1V010"),
)


def source_contract(path: Path, required_names: set[str], forbidden_calls: set[str]) -> bool:
    source = path.read_text()
    tree = ast.parse(source)
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    calls = {node.func.attr for node in ast.walk(tree)
             if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)}
    return required_names <= names and not (calls & forbidden_calls)


def rows() -> list[tuple[str, ...]]:
    graph_source = (ROOT / "Scripts/CampaignOperationsH1EvidenceGraph.py").read_text()
    authority_source = (ROOT / "Scripts/CampaignOperationsH1EvidenceAuthority.py").read_text()
    validator_source = (ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py").read_text()
    catalog_source = (ROOT / "Scripts/CampaignOperationsH1AclCatalogGenerator.py").read_text()
    restore_source = (ROOT / "Scripts/CampaignOperationsH1RestoreEvidence.py").read_text()
    trust_source = (ROOT / "Scripts/CampaignOperationsH1TrustModel.py").read_text()
    legacy_tests_source = (ROOT / "Tests/CampaignOperationsPhaseH1LegacyReachabilityTests.py").read_text()
    raw_contract = (ROOT / "Tests/fixtures/CampaignOperationsH1RawEvidenceContracts.tsv").read_text()
    probes = {
        "LEGACY-AUTHORITY-READER": "validate_normative_authority" in graph_source,
        "LEGACY-AUTHORITY-VERSION": 'AUTHORITY_VERSION = "h1-normative-authority-v2"' in authority_source,
        "LEGACY-RECEIPT-EMITTER": "metadata-only-receipt-emission-prohibited-use-trusted-runner" in authority_source,
        "LEGACY-RECEIPT-VALIDATOR": "legacy-receipt-validation-prohibited-use-trusted-runner" in authority_source,
        "LEGACY-RAW-CONTRACT": all(line.startswith("h1-raw-contract-v2\t")
                                   for line in raw_contract.splitlines()[1:] if line),
        "LEGACY-RUNTIME-V1": "test_traceability_parser_rejects_v1_runtime_records" in legacy_tests_source,
        "LEGACY-ACL-COAUTHORED": "legacy-coauthored-expected-observed-acl-record" in validator_source,
        "LEGACY-PATH-REOPEN": "validate_snapshot_bytes" in restore_source and "Path.read_" not in restore_source,
        "LEGACY-GRAPH-SHORTCUT": "materialize_provenance" in graph_source,
        "LEGACY-READINESS-SHORTCUT": "def evaluate_readiness" in trust_source,
        "LEGACY-REPORT-GENERATION": all(value in graph_source for value in
                                         ("h1-provenance-chains.tsv", "h1-provenance-edges.tsv", "report-provenance-freshness")),
        "LEGACY-ACL-DEFAULT-GENERATION": all(value in catalog_source for value in
                                              ("FROM pg_", "H1_TRUSTED_GENERATOR_EXECUTION_ID")),
        "LEGACY-RESTORE-VALIDATION": all(value in restore_source for value in
                                          ("validate_snapshot_bytes", "registered-artifact-path-reopen-prohibited-use-snapshot-bundle")),
        "LEGACY-DIRECT-VALIDATOR": "direct-validator-execution-prohibited-use-trusted-runner" in validator_source,
    }
    result = []
    for row in STATIC_REMOVALS:
        if probes.get(row[0], False):
            result.append(row)
        else:
            result.append((*row[:3], "still_reachable", "unresolved-source-probe", row[5]))
    consumer_contracts = (
        ("LEGACY-RAW-LOCK-READER", "raw_evidence", "CampaignOperationsH1LockEvidence.py:path-consumer",
         ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py"),
        ("LEGACY-RAW-ACL-ORIGIN-READER", "raw_evidence", "CampaignOperationsH1AclEvidence.py:path-consumer",
         ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py"),
        ("LEGACY-GRAPH-PATH-READ", "artifact", "CampaignOperationsH1EvidenceGraph.py:path-consumer",
         ROOT / "Scripts/CampaignOperationsH1EvidenceGraph.py"),
        ("LEGACY-VALIDATOR-PATH-READ", "artifact", "CampaignOperationsH1EvidenceValidator.py:path-consumer",
         ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py"),
    )
    for identifier, legacy_class, entry_point, path in consumer_contracts:
        mediated = source_contract(path, {"evidence_snapshot", "evidence_exists"}, set())
        disposition = "replaced" if mediated else "still_reachable"
        replacement = "ArtifactSnapshotSet.snapshot-identity-consumer" if mediated else "unresolved-path-consumer"
        result.append((identifier, legacy_class, entry_point, disposition, replacement, "H1T109"))
    return sorted(result)


def main() -> None:
    if len(sys.argv) != 2 or sys.argv[1] not in {"--write", "--check"}:
        raise SystemExit("usage: CampaignOperationsH1LegacyInventory.py --write|--check")
    target_rows = rows()
    from io import StringIO
    stream = StringIO(newline="")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
    writer.writerow(FIELDS); writer.writerows(target_rows)
    content = stream.getvalue()
    if sys.argv[1] == "--write":
        OUTPUT.write_text(content)
    elif not OUTPUT.is_file() or OUTPUT.read_text() != content:
        raise SystemExit("H1T205 stale-machine-generated-legacy-inventory")
    if any(row[3] == "still_reachable" for row in target_rows):
        raise SystemExit("H1T203 legacy-path-still-reachable")
    print(f"H1_LEGACY_INVENTORY_OK entries={len(target_rows)} reachable=0")


if __name__ == "__main__":
    main()
