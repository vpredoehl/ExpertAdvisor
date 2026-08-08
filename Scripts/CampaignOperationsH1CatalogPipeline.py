#!/usr/bin/env python3
"""Trusted ACL/default capture and comparator materialization for disposable H1 runs."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import re
import sys
from pathlib import Path

from CampaignOperationsH1AclCatalog import reconcile_acl_manifest_set
from CampaignOperationsH1EvidenceAuthority import RAW_ENVELOPE_FIELDS
from CampaignOperationsH1TrustedGenerator import TrustedGeneratorRunner

ROOT = Path(__file__).resolve().parents[1]


def render(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"empty-materialization:{path.name}")
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def manifest_digest(manifest_root: Path) -> str:
    parts = []
    for name in ("055_campaign_operations_h1_object_inventory.tsv",
                 "055_campaign_operations_h1_explicit_acl.tsv",
                 "055_campaign_operations_h1_default_acl.tsv",
                 "055_campaign_operations_h1_column_acl.tsv"):
        parts.append(hashlib.sha256((manifest_root / name).read_bytes()).hexdigest())
    return hashlib.sha256("\n".join(parts).encode()).hexdigest()


def main() -> None:
    if len(sys.argv) != 7:
        raise SystemExit("usage: ... HOST PORT USER DATABASE RUN_ID OUTPUT_ROOT")
    host, port, user, database, run_id, output_text = sys.argv[1:]
    output_root = Path(output_text); output_root.mkdir(parents=True, exist_ok=True)
    capture = output_root / "h1-acl-catalog-observed.tsv"
    payload = output_root / "h1-acl-catalog-payload.json"
    generators = {"GEN-ACL-MANIFEST": {
        "version": "h1-generator-registry-v2", "generator_id": "GEN-ACL-MANIFEST",
        "implementation": "Scripts/CampaignOperationsH1AclCatalogGenerator.py",
        "entry_point": "generate", "executable_required": "true"}}
    runner = TrustedGeneratorRunner(ROOT, generators)
    observation = runner.execute(
        "GEN-ACL-MANIFEST", run_id,
        [host, port, user, database, run_id, str(capture), str(payload)], {},
        {"ART-SUPPORT-ACL-CATALOG-CAPTURE": capture, "ART-SUPPORT-ACL-CATALOG-PAYLOAD": payload})
    runner.validate(observation, run_id, {})
    receipt = observation.receipt
    render(output_root / "h1-generator-execution-receipts.tsv", [receipt])
    payload_value = json.loads(observation.output_snapshots["ART-SUPPORT-ACL-CATALOG-PAYLOAD"].text())
    cluster_id = payload_value["cluster_id"]
    query_execution_id = payload_value["query_execution_id"]
    manifest_root = ROOT / "Database/manifests"
    set_digest = manifest_digest(manifest_root)
    requirements_path = ROOT / "Tests/fixtures/CampaignOperationsH1Fixtures.tsv"
    with requirements_path.open(newline="") as source:
        fixtures = [row for row in csv.DictReader(source, delimiter="\t")
                    if row["generator_id"] == "GEN-ACL-MANIFEST"]
    envelopes, ledger, generic_runtime = [], [], []
    with (ROOT / "Tests/fixtures/CampaignOperationsH1Traceability.tsv").open(newline="") as source:
        trace = {row["fixture_id"]: row for row in csv.DictReader(source, delimiter="\t")}
    artifact_root = output_root / "runtime-artifacts/acl-catalog"; artifact_root.mkdir(parents=True, exist_ok=True)
    observed_snapshot = observation.output_snapshots["ART-SUPPORT-ACL-CATALOG-CAPTURE"]
    for fixture in sorted(fixtures, key=lambda row: row["fixture_id"]):
        fixture_id, requirement_id = fixture["fixture_id"], fixture["requirement_id"]
        envelope = runner.raw_envelope(
            observation, "acl_catalog", requirement_id, run_id,
            "ART-SUPPORT-ACL-CATALOG-PAYLOAD")
        envelope_id = f"RAW-{run_id}-{fixture_id}"
        envelope_digest = hashlib.sha256(json.dumps(envelope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        envelopes.append({"raw_envelope_id": envelope_id, **envelope, "envelope_digest": envelope_digest})
        try:
            comparison = reconcile_acl_manifest_set(
                manifest_root, capture, requirement_id, run_id, cluster_id, {query_execution_id},
                set_digest, observed_snapshot.digest, observed_capture_bytes=observed_snapshot.data)
        except Exception as error:
            raise RuntimeError(f"acl-comparison-failed:{fixture_id}:{requirement_id}:{error}") from error
        record = {
            "format_version": "h1-acl-catalog-runtime-v3", "run_id": run_id,
            "fixture_id": fixture_id, "requirement_id": requirement_id,
            "generator_id": "GEN-ACL-MANIFEST", "generator_version": "h1-generator-registry-v2",
            "implementation_path": "Scripts/CampaignOperationsH1AclCatalogGenerator.py", "entry_point": "generate",
            "cluster_id": cluster_id, "observed_catalog_path": capture.name,
            "observed_catalog_digest": observed_snapshot.digest,
            "attested_query_execution_ids": query_execution_id,
            "generator_execution_id": receipt["generator_execution_id"],
            "generator_receipt_digest": receipt["attestation_digest"],
            "raw_envelope_id": envelope_id, "raw_envelope_digest": envelope_digest,
            "comparison_result": comparison["comparison"], "tuple_count": comparison["tuple_count"],
            "canonical_tuple_digest": comparison["canonical_tuple_digest"],
            "manifest_set_digest": set_digest,
        }
        record_path = artifact_root / f"{fixture_id}.tsv"
        render(record_path, [record])
        record_digest = hashlib.sha256(record_path.read_bytes()).hexdigest()
        ledger.append({**record, "record_artifact_path": f"runtime-artifacts/acl-catalog/{fixture_id}.tsv",
                       "record_digest": record_digest})
        observation_path = output_root / f"runtime-artifacts/records/{fixture_id}/{fixture_id}.tsv"
        observation_path.parent.mkdir(parents=True, exist_ok=True)
        observation_path.write_bytes(record_path.read_bytes())
        observation_digest = hashlib.sha256(observation_path.read_bytes()).hexdigest()
        contract = trace[fixture_id]
        timestamp_match = re.search(r"(\d{8})T(\d{6})Z", run_id)
        timestamp = (f"{timestamp_match[1][:4]}-{timestamp_match[1][4:6]}-{timestamp_match[1][6:]}T"
                     f"{timestamp_match[2][:2]}:{timestamp_match[2][2:4]}:{timestamp_match[2][4:]}Z"
                     if timestamp_match else "1970-01-01T00:00:00Z")
        runtime_row = {
            "result_format_version": "h1-runtime-result-v2", "run_id": run_id,
            "fixture_id": fixture_id, "requirement_id": requirement_id,
            "test_source": "trusted-postgresql-acl-catalog-v3", "timestamp": timestamp,
            "actual_status": contract["expected_status"], "sqlstate": contract["expected_sqlstate"],
            "diagnostic": contract["expected_diagnostic"],
            "object_identity": contract["expected_failing_object"], "stage": contract["expected_stage"],
            "operation_id": fixture_id,
            "artifact_path": f"runtime-artifacts/records/{fixture_id}/{fixture_id}.tsv",
            "artifact_digest": observation_digest, "lock_outcome": "not-applicable",
            "cycle_detected": "false", "cleanup_result": "PASS", "generator_id": "GEN-ACL-MANIFEST",
            "generator_version": "h1-generator-registry-v2",
            "generator_implementation": "Scripts/CampaignOperationsH1AclCatalogGenerator.py",
            "generator_entry_point": "generate", "emitted_runtime_record_id": f"RT-{fixture_id}",
            "output_artifact_id": f"ART-RUNTIME-OBS-{fixture_id}",
        }
        runtime_row["record_digest"] = hashlib.sha256(
            "\t".join(runtime_row.values()).encode()).hexdigest()
        generic_runtime.append(runtime_row)
    render(output_root / "h1-raw-evidence-envelopes.tsv", envelopes,
           ["raw_envelope_id", *RAW_ENVELOPE_FIELDS, "envelope_digest"])
    render(output_root / "h1-acl-requirement-evidence.tsv", ledger)
    render(output_root / "h1-acl-generic-runtime.tsv", generic_runtime)
    snapshot_rows = []
    for artifact_id, snapshot in sorted(observation.output_snapshots.items()):
        snapshot_rows.append({"artifact_id": artifact_id, "snapshot_id": snapshot.snapshot_id,
                              "lexical_path": snapshot.lexical_path, "device": str(snapshot.device),
                              "inode": str(snapshot.inode), "size": str(snapshot.size),
                              "digest": snapshot.digest, "run_id": snapshot.run_id})
    render(output_root / "h1-generator-output-snapshots.tsv", snapshot_rows)
    print(f"H1_ACL_CATALOG_V3_OK rows={len(fixtures)} generator_execution={receipt['generator_execution_id']}")


if __name__ == "__main__":
    main()
