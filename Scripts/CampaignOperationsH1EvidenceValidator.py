#!/usr/bin/env python3
"""Authenticate H1 runtime semantics and emit first-class validator results."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
from pathlib import Path

from CampaignOperationsH1EvidenceAuthority import (
    validate_filesystem,
)
from CampaignOperationsH1AclCatalog import AclCatalogError, reconcile_acl_manifest_set
from CampaignOperationsH1ArtifactSnapshot import capture_regular_file
from CampaignOperationsH1RestoreEvidence import validate_snapshot_bytes as validate_restore_snapshot_bytes

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = Path(os.environ.get("H1_REGISTRY_ROOT", ROOT / "Tests/fixtures"))
RESULT_VERSION = "h1-validator-result-v2"
VALIDATOR_VERSION = "h1-validator-registry-v2"
IMPLEMENTATION = "Scripts/CampaignOperationsH1EvidenceValidator.py"
EVIDENCE_ROOT: Path | None = None
SNAPSHOTS = None


def fail(code: str, key: str, detail: str, stage: str = "authentic-runtime-validation") -> None:
    print(f"H1V{code} key={key} stage={stage} detail={detail}", file=sys.stderr)
    raise SystemExit(1)


def sha(path: Path) -> str:
    snapshot = evidence_snapshot(path)
    if snapshot is not None:
        return snapshot.digest
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        fail("001", str(path), "missing-artifact")


def read_tsv(path: Path, expected_header: list[str] | None = None) -> tuple[list[str], list[dict[str, str]]]:
    try:
        snapshot = evidence_snapshot(path)
        if snapshot is not None:
            data = list(csv.reader(io.StringIO(snapshot.text()), delimiter="\t"))
        else:
            with path.open(newline="") as source:
                data = list(csv.reader(source, delimiter="\t"))
    except OSError:
        fail("001", str(path), "missing-input")
    if not data:
        fail("002", str(path), "empty-input")
    if expected_header is not None and data[0] != expected_header:
        fail("002", path.name, "invalid-header")
    if len(set(data[0])) != len(data[0]):
        fail("002", path.name, "duplicate-header")
    rows = []
    for number, values in enumerate(data[1:], 2):
        if len(values) != len(data[0]):
            fail("002", f"{path.name}:{number}", "field-count")
        rows.append(dict(zip(data[0], values)))
    return data[0], rows


def evidence_snapshot(path: Path):
    if EVIDENCE_ROOT is None or SNAPSHOTS is None:
        return None
    try:
        relative = path.absolute().relative_to(EVIDENCE_ROOT.absolute()).as_posix()
    except ValueError:
        return None
    return SNAPSHOTS.by_path(relative)


def evidence_exists(path: Path) -> bool:
    if EVIDENCE_ROOT is None or SNAPSHOTS is None:
        return path.is_file()
    try:
        relative = path.absolute().relative_to(EVIDENCE_ROOT.absolute()).as_posix()
    except ValueError:
        return path.is_file()
    return relative in SNAPSHOTS


def evidence_text(path: Path, errors: str = "replace") -> str:
    snapshot = evidence_snapshot(path)
    if snapshot is not None:
        return snapshot.text(errors=errors)
    return path.read_text(errors=errors)


def keyed(name: str, key: str) -> dict[str, dict[str, str]]:
    _, rows = read_tsv(REGISTRY_ROOT / name)
    result = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in result:
            fail("003", value or name, f"invalid-key:{name}", "registry-semantic-validation")
        result[value] = row
    return result


def split_ids(value: str) -> list[str]:
    return [] if not value else value.split(",")


def canonical(row: dict[str, str], fields: list[str] | None = None) -> str:
    chosen = fields or list(row)
    return "\t".join(row[field] for field in chosen)


def json_value(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def timestamp_from(run_id: str, source: dict[str, str]) -> str:
    value = source.get("timestamp", "")
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z", value):
        return value
    match = re.search(r"(\d{8})T(\d{6})Z", run_id)
    if match:
        date, clock = match.groups()
        return f"{date[:4]}-{date[4:6]}-{date[6:]}T{clock[:2]}:{clock[2:4]}:{clock[4:]}Z"
    return "1970-01-01T00:00:00Z"


def validate_implementations(generators: dict[str, dict[str, str]], validators: dict[str, dict[str, str]]) -> None:
    allowed_generator_entries = {"emit_runtime_result", "generate", "emit_acl_catalog_runtime",
                                 "emit_pre_enablement_runtime", "emit_invariant_runtime", "record"}
    allowed_validator_entries = {"validate_generic", "validate_lock", "validate_acl_origin",
                                 "validate_pre_enablement", "validate_invariant", "validate_final_assurance",
                                 "validate_mutation"}
    for node_type, rows, version, entries in [
        ("generator", generators, "h1-generator-registry-v2", allowed_generator_entries),
        ("validator", validators, "h1-validator-registry-v2", allowed_validator_entries),
    ]:
        for identifier, row in rows.items():
            if row.get("version") != version:
                fail("004", identifier, f"stale-version:{row.get('version')}", "registry-semantic-validation")
            path_text = row.get("implementation", "")
            if not path_text or Path(path_text).is_absolute() or ".." in Path(path_text).parts:
                fail("004", identifier, "invalid-implementation-path", "registry-semantic-validation")
            path = ROOT / path_text
            if not path.is_file():
                fail("004", identifier, "nonexistent-implementation", "registry-semantic-validation")
            if row.get("entry_point") not in entries:
                fail("004", identifier, "invalid-entry-point", "registry-semantic-validation")
            if row.get("executable_required") != "true" or not (path.stat().st_mode & stat.S_IXUSR):
                fail("004", identifier, "implementation-not-executable", "registry-semantic-validation")


def validate_generic(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                     artifact: dict[str, str], root: Path, run_id: str,
                     trace: dict[str, dict[str, str]]) -> tuple[dict, dict, str, str, str]:
    fixture_id = fixture["fixture_id"]
    contract = trace.get(expected["requirement_id"])
    if contract is None:
        fail("101", fixture_id, "missing-trace-contract")
    wanted = {
        "requirement_id": contract["requirement_id"], "fixture_id": contract["fixture_id"],
        "actual_status": contract["expected_status"],
        "sqlstate": contract["expected_sqlstate"], "diagnostic": contract["expected_diagnostic"],
        "object_identity": contract["expected_failing_object"], "stage": contract["expected_stage"],
        "artifact_name": contract["artifact"], "generator_id": expected["generator_id"],
        "generator_version": "h1-generator-registry-v2", "emitted_runtime_record_id": expected["runtime_record_id"],
        "output_artifact_id": expected["artifact_id"],
        "generator_implementation": "Tests/CampaignOperationsPhaseH1MigrationTests.sh",
        "generator_entry_point": "emit_runtime_result",
        "test_source_policy": "nonempty",
    }
    actual = {key: source.get(key, "") for key in wanted if key not in {"artifact_name", "test_source_policy"}}
    actual["artifact_name"] = Path(source.get("artifact_path", "")).name
    actual["test_source_policy"] = "nonempty" if source.get("test_source") else "empty"
    if source.get("result_format_version") != "h1-runtime-result-v2" or source.get("run_id") != run_id:
        fail("102", fixture_id, "stale-or-invalid-generic-runtime")
    evidence = root / artifact["path"]
    aggregate_digest = sha(evidence)
    if source.get("artifact_path") != artifact["path"] or source.get("artifact_digest") != aggregate_digest:
        fail("103", fixture_id, "artifact-provenance-mismatch")
    fields = list(source)
    recorded_digest = source.get("record_digest", "")
    recomputed = hashlib.sha256(canonical(source, fields[:-1]).encode()).hexdigest()
    if recorded_digest != recomputed:
        fail("104", fixture_id, "stale-record-digest")
    if actual != wanted:
        differing = next(key for key in wanted if actual.get(key) != wanted[key])
        fail("105", fixture_id, f"semantic-mismatch:{differing}")
    if fixture_id == "H1CPP002":
        contents = evidence_text(evidence)
        if re.search(r"H[234].*(entry points?|mutation).*(present|reachable)", contents, re.I):
            fail("106", fixture_id, "prohibited-h2-h3-h4-entry-point")
        for marker in ["command=rg", "scanned_paths=Sources", "patterns=H2,H3,H4",
                       "result_rows=0", "exit_status=1", "scan_result=PASS"]:
            if marker not in contents:
                fail("106", fixture_id, f"insufficient-exclusion-authenticity:{marker.split('=')[0]}")
    return wanted, actual, source["actual_status"], source["diagnostic"], recomputed


def validate_lock(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                  artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    matrix = keyed("CampaignOperationsH1LockPathMatrix.tsv", "test_id")
    contract = matrix.get(identifier)
    if contract is None:
        fail("200", identifier, "missing-independent-lock-contract")
    wanted = {
        "format_version": "h1-lock-runtime-v3", "run_id": run_id,
        "h1lock_id": identifier, "generator_id": "GEN-LOCK",
        "generator_version": "h1-generator-registry-v2",
        "implementation_path": "Scripts/CampaignOperationsH1LockEvidence.py", "entry_point": "generate",
        "emitted_runtime_record_id": expected["runtime_record_id"],
        "output_artifact_id": expected["artifact_id"], "raw_artifact_path": artifact["path"],
        "cycle_detected": "false", "reverse_wait_observed": "false",
        "lock_release_verified": "true", "partial_evidence_count": "0", "cleanup_result": "PASS",
    }
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted:
        differing = next(key for key in wanted if actual.get(key) != wanted[key])
        fail("201", identifier, f"lock-semantic-mismatch:{differing}")
    independent = {
        "requirement_id": expected["requirement_id"],
        "first_operation": contract["first_workflow"],
        "second_operation": contract["second_workflow"],
        "workflow_classification": contract["expected_classification"],
        "permitted_wait_direction": contract["permitted_wait"],
        "prohibited_reverse_direction": contract["prohibited_wait"],
    }
    if contract["requirement_id"] != expected["requirement_id"]:
        fail("200", identifier, "lock-contract-requirement-mismatch")
    for field, value in independent.items():
        if field in source and source.get(field) != value:
            fail("201", identifier, f"lock-authority-mismatch:{field}")
    if source.get("observed_wait_direction") != contract["permitted_wait"]:
        fail("202", identifier, "prohibited-wait-direction")
    if (not source.get("first_pid", "").isdigit() or not source.get("second_pid", "").isdigit() or
            source["first_pid"] == source["second_pid"] or
            not source.get("first_application") or not source.get("second_application") or
            source["first_application"] == source["second_application"]):
        fail("203", identifier, "invalid-backend-identity")
    if sha(root / artifact["path"]) != source.get("raw_artifact_digest"):
        fail("204", identifier, "stale-raw-artifact")
    recomputed = hashlib.sha256(canonical(source, list(source)[:-1]).encode()).hexdigest()
    if recomputed != source.get("record_digest"):
        fail("205", identifier, "stale-record-digest")
    semantic_fields = ["first_pid", "second_pid", "first_application", "second_application",
        "first_held_lock_set", "second_held_lock_set", "requested_lock", "lock_type", "lock_identity",
        "lock_mode", "first_granted", "second_waiting", "blocking_pids_first", "blocking_pids_second",
        "complete_wait_for_graph", "permitted_wait_direction", "observed_wait_direction",
        "prohibited_reverse_direction", "reverse_wait_observed", "cycle_detected",
        "first_transaction_outcome", "second_transaction_outcome", "lock_release_verified",
        "partial_evidence_count"]
    parsed_actual = {**actual, "parsed_lock_semantics": {key: source.get(key, "") for key in semantic_fields}}
    parsed_wanted = {**wanted, "lock_semantic_policy": {
        "backend_identity": "numeric-distinct-pids-and-distinct-applications",
        "observed_direction": contract["permitted_wait"], "prohibited_reverse_observed": "false",
        "cycle_detected": "false", "release_verified": "true", "partial_evidence_count": "0"}}
    return parsed_wanted, parsed_actual, "SUCCESS", "lock-semantics-reconciled", recomputed


def validate_acl_origin(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                        artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    acl_contracts = keyed("CampaignOperationsH1AclOriginFixtures.tsv", "fixture_id")
    contract = acl_contracts.get(identifier)
    if contract is None:
        fail("300", identifier, "missing-independent-acl-contract")
    wanted = {
        "runtime_format_version": "h1-acl-origin-runtime-v3", "run_id": run_id,
        "fixture_id": identifier, "requirement_id": expected["requirement_id"],
        "generator_id": "GEN-ACL-ORIGIN", "generator_version": "h1-generator-registry-v2",
        "implementation_path": "Scripts/CampaignOperationsH1AclEvidence.py", "entry_point": "generate",
        "emitted_runtime_record_id": expected["runtime_record_id"], "output_artifact_id": expected["artifact_id"],
        "raw_artifact_path": artifact["path"], "expected_sqlstate": "42501", "actual_sqlstate": "42501",
        "expected_diagnostic": "H1A006", "expected_stage": "database-audit",
        "actual_stage": "database-audit", "cleanup_result": "PASS",
    }
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted or not source.get("actual_diagnostic", "").startswith("H1A006"):
        differing = next((key for key in wanted if actual.get(key) != wanted[key]), "actual_diagnostic")
        fail("301", identifier, f"acl-origin-semantic-mismatch:{differing}")
    if source.get("expected_origin") not in {"null", "explicit"} or source.get("actual_origin") not in {"null", "explicit"}:
        fail("302", identifier, "invalid-acl-origin")
    for field in ["requirement_id", "object_class", "schema_name", "object_identity",
                  "catalog_acl_column", "expected_origin", "actual_origin", "direction"]:
        if source.get(field) != contract.get(field):
            fail("301", identifier, f"acl-authority-mismatch:{field}")
    if not source.get("expanded_acl_count", "").isdigit() or not re.fullmatch(r"[0-9a-f]{64}", source.get("effective_acl_expansion_digest", "")):
        fail("303", identifier, "invalid-canonical-expansion")
    if sha(root / artifact["path"]) != source.get("raw_artifact_digest"):
        fail("304", identifier, "stale-raw-artifact")
    recomputed = hashlib.sha256(canonical(source, list(source)[:-1]).encode()).hexdigest()
    if recomputed != source.get("record_digest"):
        fail("305", identifier, "stale-record-digest")
    semantic_fields = ["object_class", "schema_name", "object_identity", "catalog_acl_column",
        "expected_origin", "actual_origin", "raw_acl_is_null", "raw_acl_text", "owner", "acldefault_type",
        "expanded_acl_count", "canonical_acl_expansion", "effective_acl_expansion_digest",
        "expected_sqlstate", "actual_sqlstate", "expected_diagnostic", "actual_diagnostic",
        "expected_stage", "actual_stage"]
    parsed_actual = {**actual, "parsed_acl_origin_semantics": {key: source.get(key, "") for key in semantic_fields}}
    parsed_wanted = {**wanted, "acl_origin_policy": {
        "origin_enum": "null-or-explicit", "sqlstate": "42501", "diagnostic_prefix": "H1A006",
        "stage": "database-audit", "canonical_expansion_digest": "sha256"}}
    return parsed_wanted, parsed_actual, "SUCCESS", source["actual_diagnostic"], recomputed


def validate_acl_catalog(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                         artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    _, record_rows = read_tsv(root / artifact["path"])
    if len(record_rows) != 1:
        fail("401", identifier, "acl-catalog-record-cardinality")
    record = record_rows[0]
    if record.get("format_version") != "h1-acl-catalog-runtime-v3":
        fail("402", identifier, "legacy-coauthored-expected-observed-acl-record")
    wanted = {"format_version": "h1-acl-catalog-runtime-v3", "run_id": run_id,
              "fixture_id": identifier, "requirement_id": expected["requirement_id"],
              "generator_id": "GEN-ACL-MANIFEST", "generator_version": "h1-generator-registry-v2",
              "implementation_path": "Scripts/CampaignOperationsH1AclCatalogGenerator.py",
              "entry_point": "generate", "comparison_result": "equal"}
    actual = {key: record.get(key, "") for key in wanted}
    if actual != wanted:
        differing = next(key for key in wanted if actual.get(key) != wanted[key])
        fail("402", identifier, f"acl-catalog-semantic-mismatch:{differing}")
    observed_path = root / record.get("observed_catalog_path", "")
    attested = set(filter(None, record.get("attested_query_execution_ids", "").split(",")))
    receipt_path = root / "h1-generator-execution-receipts.tsv"
    envelope_path = root / "h1-raw-evidence-envelopes.tsv"
    if not evidence_exists(receipt_path) or not evidence_exists(envelope_path):
        fail("403", identifier, "missing-generator-receipt-or-raw-envelope")
    _, receipt_rows = read_tsv(receipt_path)
    receipts = {row.get("generator_execution_id", ""): row for row in receipt_rows}
    receipt = receipts.get(record.get("generator_execution_id", ""))
    if (receipt is None or receipt.get("version") != "h1-generator-execution-receipt-v2" or
            receipt.get("run_id") != run_id or receipt.get("actual_exit_status") != "0" or
            receipt.get("execution_status") != "completed" or
            receipt.get("attestation_digest") != record.get("generator_receipt_digest")):
        fail("403", identifier, "invalid-generator-execution-receipt-binding")
    _, envelope_rows = read_tsv(envelope_path)
    envelopes = {row.get("raw_envelope_id", ""): row for row in envelope_rows}
    envelope = envelopes.get(record.get("raw_envelope_id", ""))
    if envelope is None:
        fail("403", identifier, "missing-raw-v2-envelope")
    envelope_payload = {key: envelope[key] for key in envelope
                        if key not in {"raw_envelope_id", "envelope_digest"}}
    envelope_digest = hashlib.sha256(json.dumps(
        envelope_payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    if (envelope.get("evidence_version") != "h1-raw-execution-evidence-v2" or
            envelope.get("evidence_class") != "acl_catalog" or
            envelope.get("requirement_id") != expected["requirement_id"] or
            envelope.get("run_id") != run_id or
            envelope.get("generator_execution_id") != record.get("generator_execution_id") or
            envelope.get("envelope_digest") != envelope_digest or
            envelope_digest != record.get("raw_envelope_digest")):
        fail("403", identifier, "raw-v2-envelope-binding-mismatch")
    observed_snapshot = evidence_snapshot(observed_path)
    if observed_snapshot is None:
        fail("403", identifier, "observed-catalog-not-descriptor-snapshotted")
    try:
        manifest_root = ROOT / "Database/manifests"
        manifest_names = ["055_campaign_operations_h1_object_inventory.tsv",
                          "055_campaign_operations_h1_explicit_acl.tsv",
                          "055_campaign_operations_h1_default_acl.tsv",
                          "055_campaign_operations_h1_column_acl.tsv"]
        manifest_digests = [capture_regular_file(manifest_root / name, f"ACL-EXPECTED-{name}", run_id,
                                                f"Database/manifests/{name}").digest
                            for name in manifest_names]
        manifest_set_digest = hashlib.sha256("\n".join(manifest_digests).encode()).hexdigest()
        if record.get("manifest_set_digest") != manifest_set_digest:
            fail("403", identifier, "acl-manifest-set-digest-mismatch")
        comparison = reconcile_acl_manifest_set(
            manifest_root, observed_path, expected["requirement_id"], run_id,
            record.get("cluster_id", ""), attested, manifest_set_digest,
            record.get("observed_catalog_digest", ""),
            observed_capture_bytes=observed_snapshot.data)
    except (AclCatalogError, OSError) as error:
        fail("403", identifier, f"acl-catalog-independent-comparison:{error}")
    record_digest = sha(root / artifact["path"])
    if source.get("record_digest") != record_digest or source.get("record_artifact_path") != artifact["path"]:
        fail("404", identifier, "stale-acl-record-digest")
    return {**wanted, "comparison_policy": "independent-two-way-relational"}, \
        {**actual, **comparison}, "SUCCESS", "catalog-tuples-equal", record_digest


def validate_pre_enablement(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                            artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    wanted = {"version": "h1-pre-enablement-runtime-v1", "run_id": run_id,
              "evidence_id": identifier, "requirement_id": expected["requirement_id"],
              "executable_workflow": "false", "disposition": "pre-enablement-blocker", "cleanup_result": "PASS"}
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted or not source.get("architectural_reason"):
        fail("501", identifier, "pre-enablement-semantic-mismatch")
    _, retained = read_tsv(root / artifact["path"])
    if len(retained) != 1 or retained[0] != source:
        fail("502", identifier, "pre-enablement-raw-record-mismatch")
    digest = sha(root / artifact["path"])
    return wanted, actual, "BLOCKED_PRE_ENABLEMENT", source["architectural_reason"], digest


def validate_invariant(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                       artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    wanted = {"version": "h1-uniqueness-invariant-runtime-v1", "run_id": run_id,
              "invariant_id": identifier, "requirement_id": expected["requirement_id"],
              "uniqueness_expression": "UNIQUE (operational_campaign_id, action_kind, action_contract_version)",
              "disposition": "accepted_non_executable_invariant", "cleanup_result": "PASS"}
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted:
        fail("601", identifier, "invariant-semantic-mismatch")
    _, retained = read_tsv(root / artifact["path"])
    if len(retained) != 1 or retained[0] != source:
        fail("602", identifier, "invariant-raw-record-mismatch")
    digest = sha(root / artifact["path"])
    return wanted, actual, "SUCCESS", "catalog-uniqueness-reconciled", digest


def validate_final_assurance(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                             artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    identifier = fixture["fixture_id"]
    evidence_id = artifact["record_key_value"]
    wanted = {"version": "h1-final-assurance-result-v2", "run_id": run_id,
              "evidence_id": evidence_id, "generator_id": "GEN-FINAL-ASSURANCE",
              "generator_version": "h1-generator-registry-v2", "status": "PASS",
              "artifact_id": expected["artifact_id"], "artifact_path": artifact["path"]}
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted or not source.get("case_count", "").isdigit() or int(source["case_count"]) < 1:
        fail("701", evidence_id, "invalid-final-assurance-record")
    log_path = root / artifact["path"]
    if sha(log_path) != source.get("artifact_digest"):
        fail("702", evidence_id, "stale-final-assurance-artifact")
    log = evidence_text(log_path)
    parsed_semantics: dict[str, str] = {}
    pairs = dict(re.findall(r"\b([a-z0-9_]+)=([^\s]+)", log))
    if evidence_id == "RELEASE_BUILD":
        if "** BUILD SUCCEEDED **" not in log or "** BUILD FAILED **" in log:
            fail("703", evidence_id, "release-build-not-succeeded")
        for required in ["configuration=Release", "product_path=", "derived_data_path=", "exit_status=0", "warning_count="]:
            if required not in log:
                fail("703", evidence_id, f"missing-build-field:{required.rstrip('=')}")
        for required in ["Command line invocation:", "Build description signature:", "xcodebuild_version="]:
            if required not in log:
                fail("703", evidence_id, "insufficient-build-execution-authenticity")
        parsed_semantics = {key: pairs.get(key, "") for key in
                            ["exit_status", "configuration", "product_path", "derived_data_path",
                             "warning_count", "warning_classification"]}
        parsed_semantics["build_succeeded_marker"] = "true"
        parsed_semantics["build_failed_marker"] = "false"
    elif evidence_id == "STRICT_COMPILE":
        for required in ["command=clang++", "translation_units=", "exit_status=0", "warning_count=0", "error_count=0"]:
            if required not in log:
                fail("704", evidence_id, f"missing-compile-field:{required.rstrip('=')}")
        parsed_semantics = {key: pairs.get(key, "") for key in
                            ["command", "flags", "translation_units", "exit_status", "warning_count", "error_count"]}
    elif evidence_id == "CHECKSUM":
        for required in ["migration_sha256=", "embedded_checksum=", "manifest_digest=", "ledger_result=PASS", "replay_result=PASS"]:
            if required not in log:
                fail("705", evidence_id, f"missing-checksum-field:{required.rstrip('=')}")
        parsed_semantics = {key: pairs.get(key, "") for key in
                            ["migration_sha256", "embedded_checksum", "manifest_digest", "ledger_result", "replay_result"]}
        if parsed_semantics["migration_sha256"] != parsed_semantics["embedded_checksum"]:
            fail("705", evidence_id, "migration-embedded-checksum-mismatch")
    elif evidence_id == "WORKTREE_STATUS":
        for required in ["tracked_count=", "untracked_count=", "policy=no-commit"]:
            if required not in log:
                fail("706", evidence_id, f"missing-worktree-field:{required.rstrip('=')}")
        parsed_semantics = {key: pairs.get(key, "") for key in ["tracked_count", "untracked_count", "policy"]}
    elif evidence_id in {"LOCK_MUTATIONS", "ACL_MUTATIONS", "GRAPH_MUTATIONS", "MANIFEST_MUTATIONS"}:
        if "passed cases=" not in log or "unexpectedly passed" in log:
            fail("707", evidence_id, "mutation-suite-semantic-failure")
        parsed_semantics = {"case_count": source["case_count"], "machine_case_rows": str(log.count("H1_MUTATION_CASE\t"))}
    elif evidence_id == "TRACE_PARSER":
        if "acceptance_evidence=false" not in log:
            fail("708", evidence_id, "parser-classification-missing")
        parsed_semantics = {"case_count": source["case_count"], "acceptance_evidence": "false",
                            "machine_case_rows": str(log.count("H1_MUTATION_CASE\t"))}
    elif evidence_id == "RESTORE_ARTIFACT":
        if "scenarios=10" not in log:
            fail("709", evidence_id, "restore-scenario-count")
        parsed_semantics = {"scenario_count": "10", "semantic_result": "PASS"}
    elif evidence_id == "DETERMINISTIC_REGENERATION":
        if "byte_identical=true" not in log or "semantic_validation=PASS" not in log:
            fail("710", evidence_id, "determinism-without-correctness")
        parsed_semantics = {"byte_identical": pairs.get("byte_identical", ""),
                            "semantic_validation": pairs.get("semantic_validation", "")}
    elif evidence_id == "FULL_PIPELINE":
        if "H1_REFERENCE_GRAPH" not in log or "semantic_validation=PASS" not in log:
            fail("711", evidence_id, "pipeline-semantic-result-missing")
        parsed_semantics = {"semantic_validation": pairs.get("semantic_validation", ""),
                            "defects": pairs.get("defects", ""), "ready": pairs.get("ready", "")}
    record_digest = hashlib.sha256(canonical(source, list(source)[:-1]).encode()).hexdigest()
    if source.get("record_digest") != record_digest:
        fail("712", evidence_id, "stale-final-record-digest")
    return {**wanted, "semantic_policy": evidence_id}, {**actual, "parsed_semantics": parsed_semantics}, \
        "PASS", source.get("diagnostic", evidence_id), record_digest


def validate_mutation(source: dict[str, str], expected: dict[str, str], fixture: dict[str, str],
                      artifact: dict[str, str], root: Path, run_id: str) -> tuple[dict, dict, str, str, str]:
    case_id = artifact["record_key_value"]
    wanted = {
        "version": "h1-mutation-result-v1", "run_id": run_id, "mutation_case_id": case_id,
        "status": "PASS", "fixture_id": fixture["fixture_id"],
        "runtime_record_id": expected["runtime_record_id"],
        "validator_result_id": expected["validator_result_ids"],
        "report_entry_id": expected["report_entry_ids"], "artifact_id": expected["artifact_id"],
        "artifact_path": artifact["path"], "generator_id": "GEN-MUTATION",
        "generator_version": "h1-generator-registry-v2",
        "implementation_path": "Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh", "entry_point": "record",
    }
    actual = {key: source.get(key, "") for key in wanted}
    if actual != wanted:
        differing = next(key for key in wanted if actual.get(key) != wanted[key])
        fail("901", case_id, f"mutation-identity-mismatch:{differing}")
    if (not source.get("mutated_field_or_artifact") or not source.get("expected_failure_code") or
            source.get("expected_failure_code") != source.get("actual_failure_code") or
            source.get("expected_stage") != source.get("actual_stage")):
        fail("902", case_id, "mutation-diagnostic-or-stage-mismatch")
    path = root / artifact["path"]
    if not evidence_exists(path) or sha(path) != source.get("artifact_digest") or source.get("record_digest") != source.get("artifact_digest"):
        fail("903", case_id, "stale-mutation-record")
    _, record_rows = read_tsv(path)
    if len(record_rows) != 1:
        fail("904", case_id, "mutation-record-cardinality")
    record = record_rows[0]
    for key in ["run_id", "mutation_case_id", "suite", "mutated_field_or_artifact",
                "expected_failure_code", "expected_stage", "actual_failure_code", "actual_stage", "status",
                "fixture_id", "runtime_record_id", "validator_result_id", "report_entry_id", "artifact_id",
                "generator_id", "generator_version", "implementation_path", "entry_point"]:
        if record.get(key) != source.get(key):
            fail("905", case_id, f"mutation-record-mismatch:{key}")
    parsed_wanted = {**wanted, "expected_failure_code": source["expected_failure_code"],
                     "expected_stage": source["expected_stage"]}
    parsed_actual = {**actual, "actual_failure_code": source["actual_failure_code"],
                     "actual_stage": source["actual_stage"],
                     "mutated_field_or_artifact": source["mutated_field_or_artifact"]}
    return parsed_wanted, parsed_actual, "PASS", source["actual_failure_code"], source["record_digest"]


def main() -> None:
    global EVIDENCE_ROOT, SNAPSHOTS
    trusted_entries = {"validate_generic", "validate_lock", "validate_acl_origin",
                       "validate_pre_enablement", "validate_invariant",
                       "validate_final_assurance", "validate_mutation"}
    trusted_worker = len(sys.argv) == 7 and sys.argv[1] in trusted_entries
    if trusted_worker:
        entry_point, root, run_id, output, runtime_worker_output, worker_validator_id = (
            sys.argv[1], Path(sys.argv[2]), sys.argv[3], Path(sys.argv[4]), Path(sys.argv[5]), sys.argv[6])
        mode = "generate"
    elif len(sys.argv) == 5 and sys.argv[1] in {"generate", "validate"}:
        fail("010", "legacy-validator-entry", "direct-validator-execution-prohibited-use-trusted-runner",
             "validator-execution-binding")
    else:
        raise SystemExit("usage: CampaignOperationsH1EvidenceValidator.py ENTRY ROOT RUN_ID RESULT_OUTPUT RUNTIME_OUTPUT VALIDATOR_ID")
    requirements = keyed("CampaignOperationsH1Requirements.tsv", "requirement_id")
    fixtures = keyed("CampaignOperationsH1Fixtures.tsv", "fixture_id")
    generators = keyed("CampaignOperationsH1Generators.tsv", "generator_id")
    runtime = keyed("CampaignOperationsH1RuntimeRecords.tsv", "runtime_record_id")
    validators = keyed("CampaignOperationsH1Validators.tsv", "validator_id")
    reports = keyed("CampaignOperationsH1ReportEntries.tsv", "report_entry_id")
    artifacts = keyed("CampaignOperationsH1Artifacts.tsv", "artifact_id")
    validate_implementations(generators, validators)
    worker_validator = validators.get(worker_validator_id)
    if worker_validator is None or worker_validator.get("entry_point") != entry_point:
        fail("010", worker_validator_id, "trusted-worker-registry-entry-point-mismatch",
             "validator-execution-binding")
    execution_id = os.environ.get("H1_TRUSTED_VALIDATOR_EXECUTION_ID", "")
    if not re.fullmatch(r"VEXEC-.+", execution_id):
        fail("010", worker_validator_id, "missing-trusted-runner-execution-identity",
             "validator-execution-binding")
    EVIDENCE_ROOT = root
    SNAPSHOTS = validate_filesystem(root, artifacts, mode == "generate", run_id)
    SNAPSHOTS.assert_run_id(run_id)
    restore_runtime = root / "h1-restore-runtime.tsv"
    if evidence_exists(restore_runtime):
        restore_snapshot = evidence_snapshot(restore_runtime)
        if restore_snapshot is None:
            fail("801", "h1-restore-runtime.tsv", "restore-runtime-not-snapshotted")
        try:
            _, restore_rows = restore_snapshot.tsv()
            restore_artifacts = {
                row["artifact_path"]: evidence_snapshot(root / row["artifact_path"]).data
                for row in restore_rows
            }
            validate_restore_snapshot_bytes(restore_snapshot.data, restore_artifacts, run_id)
        except (KeyError, SystemExit):
            fail("801", "h1-restore-runtime.tsv", "restore-semantic-validation-failed")
    _, trace_rows = read_tsv(REGISTRY_ROOT / "CampaignOperationsH1Traceability.tsv")
    trace = {row["requirement_id"]: row for row in trace_rows}

    sources: dict[str, dict[str, dict[str, str]]] = {}
    for name, key in [("h1-runtime-results.tsv", "fixture_id"), ("h1-lock-runtime.tsv", "h1lock_id"),
                      ("h1-acl-origin-runtime.tsv", "fixture_id"), ("h1-acl-requirement-evidence.tsv", "fixture_id"),
                      ("h1-pre-enablement-runtime.tsv", "evidence_id"),
                      ("h1-uniqueness-invariant-runtime.tsv", "invariant_id")]:
        path = root / name
        if evidence_exists(path):
            _, rows = read_tsv(path)
            mapping = {}
            for row in rows:
                value = row.get(key, "")
                if not value or value in mapping:
                    fail("005", value or name, f"duplicate-or-empty-runtime-key:{name}")
                mapping[value] = row
            sources[name] = mapping
    final_rows: dict[str, dict[str, str]] = {}
    final_path = root / "h1-final-assurance-results.tsv"
    if evidence_exists(final_path):
        _, rows = read_tsv(final_path)
        final_rows = {row.get("evidence_id", ""): row for row in rows}
    mutation_rows: dict[str, dict[str, str]] = {}
    mutation_path = root / "h1-mutation-results.tsv"
    if evidence_exists(mutation_path):
        _, rows = read_tsv(mutation_path)
        mutation_rows = {row.get("mutation_case_id", ""): row for row in rows}
        if len(mutation_rows) != len(rows):
            fail("906", "h1-mutation-results.tsv", "duplicate-mutation-case")

    results = []
    actual_runtime = []
    for runtime_id, expected in sorted(runtime.items()):
        fixture = fixtures[expected["fixture_id"]]
        artifact = artifacts[expected["artifact_id"]]
        generator_id = expected["generator_id"]
        if generator_id == "GEN-MUTATION":
            source = mutation_rows.get(artifact["record_key_value"])
            if source is None:
                continue
            parsed = validate_mutation(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = sha(root / artifact["path"])
        elif artifact["required_phase"] == "final":
            source = final_rows.get(artifact["record_key_value"])
            if source is None:
                continue
            parsed = validate_final_assurance(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = sha(root / artifact["path"])
        elif generator_id == "GEN-LOCK":
            source = sources.get("h1-lock-runtime.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-lock-runtime")
            parsed = validate_lock(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = sha(root / artifact["path"])
        elif generator_id == "GEN-ACL-ORIGIN":
            source = sources.get("h1-acl-origin-runtime.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-acl-origin-runtime")
            parsed = validate_acl_origin(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = sha(root / artifact["path"])
        elif generator_id == "GEN-ACL-MANIFEST":
            source = sources.get("h1-acl-requirement-evidence.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-acl-catalog-runtime")
            parsed = validate_acl_catalog(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = sha(root / artifact["path"])
        elif generator_id == "GEN-PRE":
            source = sources.get("h1-pre-enablement-runtime.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-pre-enablement-runtime")
            parsed = validate_pre_enablement(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = parsed[4]
        elif generator_id == "GEN-INVARIANT":
            source = sources.get("h1-uniqueness-invariant-runtime.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-invariant-runtime")
            parsed = validate_invariant(source, expected, fixture, artifact, root, run_id)
            aggregate_digest = parsed[4]
        else:
            source = sources.get("h1-runtime-results.tsv", {}).get(fixture["fixture_id"])
            if source is None: fail("006", runtime_id, "missing-generic-runtime")
            parsed = validate_generic(source, expected, fixture, artifact, root, run_id, trace)
            aggregate_digest = sha(root / artifact["path"])
        expected_values, actual_values, status, diagnostic, record_digest = parsed
        validator = validators[expected["validator_id"]]
        report_id = expected["report_entry_ids"]
        validator_result_id = expected["validator_result_ids"]
        row = {
            "version": RESULT_VERSION, "validator_result_id": validator_result_id,
            "validator_execution_id": "",
            "validator_id": validator["validator_id"], "validator_version": validator["version"],
            "implementation_path": validator["implementation"], "entry_point": validator["entry_point"],
            "run_id": run_id, "runtime_record_ids": runtime_id,
            "requirement_ids": expected["requirement_id"], "fixture_ids": fixture["fixture_id"],
            "artifact_ids": expected["artifact_id"], "expected_values": json_value(expected_values),
            "actual_parsed_values": json_value(actual_values), "comparison_result": "equal",
            "status": status, "diagnostic": diagnostic, "stage": "authentic-runtime-validation",
            "timestamp": timestamp_from(run_id, source), "output_artifact_id": "ART-GENERATED-VALIDATORS",
            "output_artifact_digest": record_digest, "runtime_record_digest": record_digest,
            "aggregate_artifact_digest": aggregate_digest, "report_entry_ids": report_id,
        }
        results.append(row)
        actual_runtime.append({
            "version": "h1-reconciled-runtime-v2", "run_id": run_id,
            "runtime_record_id": runtime_id, "requirement_id": expected["requirement_id"],
            "fixture_id": fixture["fixture_id"], "generator_id": generator_id,
            "generator_version": generators[generator_id]["version"],
            "implementation_path": generators[generator_id]["implementation"],
            "entry_point": generators[generator_id]["entry_point"],
            "validator_result_ids": validator_result_id, "report_entry_ids": report_id,
            "artifact_ids": expected["artifact_id"], "canonical_record": json_value(source),
            "record_digest": record_digest, "aggregate_digest": aggregate_digest,
            "record_index_key": artifact["record_key_value"], "status": status,
        })

    def render(rows: list[dict[str, str]]) -> str:
        from io import StringIO
        target = StringIO(newline="")
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
        return target.getvalue()

    results = [row for row in results if row["validator_id"] == worker_validator_id]
    runtime_ids = {row["runtime_record_ids"] for row in results}
    actual_runtime = [row for row in actual_runtime if row["runtime_record_id"] in runtime_ids]
    for result in results:
        result["validator_execution_id"] = execution_id
    if not results or not actual_runtime:
        fail("010", worker_validator_id, "trusted-validator-produced-no-results",
             "validator-execution-binding")
    output.write_text(render(results))
    runtime_worker_output.write_text(render(actual_runtime))
    print(f"H1_AUTHENTIC_VALIDATOR_RESULTS_OK validator={worker_validator_id} "
          f"results={len(results)} semantic_validation=PASS")


if __name__ == "__main__":
    main()
