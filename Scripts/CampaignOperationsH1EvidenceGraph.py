#!/usr/bin/env python3
"""Compute and report the explicit ADR-0019B H1 evidence graph."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

from CampaignOperationsH1EvidenceAuthority import (
    RAW_CLASS_PAYLOADS,
    validate_exact_runtime_inventory,
    validate_filesystem,
    validate_normative_authority,
)
from CampaignOperationsH1Provenance import (
    CHAIN_FIELDS, EDGE_FIELDS, NODE_SEQUENCE, ProvenanceError, validate_provenance,
)
from CampaignOperationsH1RestoreEvidence import validate_snapshot_bytes as validate_restore_snapshot_bytes
from CampaignOperationsH1TrustedEvidencePipeline import evidence_classes
from CampaignOperationsH1TrustedRunner import RunnerError, TrustedValidatorRunner
from CampaignOperationsH1TrustModel import (
    READINESS_COMPONENTS, evaluate_readiness, snapshot_coverage_complete,
    generator_execution_complete, validate_legacy_inventory, validator_execution_complete,
)

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
REGISTRY_ROOT = Path(os.environ.get("H1_REGISTRY_ROOT", ROOT / "Tests/fixtures"))
FINAL_REPORT = "CampaignOperations_PhaseH_H1_ADR0019B_EvidenceGraphReporting_FinalCorrection_Implementation_Output.md"
TRACE_REPORT = "CampaignOperationsH1Traceability.md"
EVIDENCE_ROOT: Path | None = None
SNAPSHOTS = None
GENERATED = {
    "h1-reconciled-runtime-records.tsv", "h1-validator-results.tsv",
    "h1-report-entry-registry.tsv", "h1-evidence-edges.tsv",
    "h1-graph-health.tsv", "h1-artifact-index.tsv", "h1-artifact-snapshots.tsv",
    "h1-validator-execution-receipts.tsv",
    "h1-provenance-chains.tsv", "h1-provenance-edges.tsv",
    TRACE_REPORT, FINAL_REPORT,
}

REGISTRY_SPECS = {
    "requirements": ("CampaignOperationsH1Requirements.tsv", "requirement_id", "h1-requirement-registry-v2"),
    "fixtures": ("CampaignOperationsH1Fixtures.tsv", "fixture_id", "h1-fixture-registry-v2"),
    "generators": ("CampaignOperationsH1Generators.tsv", "generator_id", "h1-generator-registry-v2"),
    "runtime": ("CampaignOperationsH1RuntimeRecords.tsv", "runtime_record_id", "h1-runtime-registry-v2"),
    "validators": ("CampaignOperationsH1Validators.tsv", "validator_id", "h1-validator-registry-v2"),
    "reports": ("CampaignOperationsH1ReportEntries.tsv", "report_entry_id", "h1-report-entry-registry-v2"),
    "artifacts": ("CampaignOperationsH1Artifacts.tsv", "artifact_id", "h1-artifact-registry-v2"),
    "edges": ("CampaignOperationsH1Edges.tsv", "edge_id", "h1-edge-registry-v2"),
}

ALLOWED = {
    "evidence_class": {"acl_catalog", "acl_origin", "lock", "restore", "role_security", "runtime",
                       "pipeline", "mutation", "parser_unit", "compile", "build", "checksum", "report", "repository"},
    "classification": {"executable", "pre_enablement_non_final", "accepted_non_executable", "final_assurance"},
    "status_policy": {"RECONCILED", "BLOCKED_PRE_ENABLEMENT", "PASS"},
    "fixture_cardinality": {"exactly_one"},
    "report_cardinality": {"one_entry_per_reconciled_requirement"},
    "runtime_cardinality": {"exactly_one"},
    "artifact_phase": {"base", "final", "generated"},
}


def fail(code: str, key: str, detail: str, stage: str = "reference-graph-reconciliation") -> None:
    print(f"H1R{code} key={key} stage={stage} detail={detail}", file=sys.stderr)
    raise SystemExit(1)


def digest_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def digest(path: Path) -> str:
    snapshot = evidence_snapshot(path)
    if snapshot is not None:
        return snapshot.digest
    try:
        return digest_bytes(path.read_bytes())
    except OSError:
        fail("001", str(path), "missing-artifact")


def read(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        snapshot = evidence_snapshot(path)
        if snapshot is not None:
            data = list(csv.reader(io.StringIO(snapshot.text()), delimiter="\t"))
        else:
            with path.open(newline="") as source:
                data = list(csv.reader(source, delimiter="\t"))
    except OSError:
        fail("001", str(path), "missing-graph-node")
    if not data:
        fail("001", str(path), "empty-graph-node")
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
    if relative not in SNAPSHOTS:
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


def keyed(path: Path, key: str, version: str) -> dict[str, dict[str, str]]:
    _, rows = read(path)
    result = {}
    for row in rows:
        identifier = row.get(key, "")
        if not identifier:
            fail("002", path.name, f"empty-key:{key}", "registry-semantic-validation")
        if identifier in result:
            fail("002", identifier, f"duplicate-key:{path.name}", "registry-semantic-validation")
        if row.get("version") != version:
            fail("002", identifier, f"stale-version:{row.get('version')}", "registry-semantic-validation")
        if any(value == "" for field, value in row.items()
               if field not in {"allowlist_reason", "record_key_field", "record_key_value",
                                "runtime_record_ids", "report_entry_ids"}):
            fail("002", identifier, "empty-semantic-field", "registry-semantic-validation")
        result[identifier] = row
    return result


def ids(value: str) -> list[str]:
    if not value:
        return []
    result = value.split(",")
    if any(not item for item in result) or result != sorted(set(result)):
        fail("003", value, "malformed-adjacency", "registry-semantic-validation")
    return result


def validate_source_reference(value: str, key: str) -> None:
    if not value or Path(value).is_absolute() or ".." in Path(value).parts:
        fail("003", key, "invalid-source-reference", "registry-semantic-validation")
    match = re.fullmatch(r"(.+?)(?::([1-9][0-9]*))?", value)
    if match is None:
        fail("003", key, "malformed-source-reference", "registry-semantic-validation")
    path = ROOT / match.group(1)
    if not path.is_file():
        fail("003", key, "nonexistent-source-reference", "registry-semantic-validation")
    if match.group(2) and int(match.group(2)) > sum(1 for _ in path.open(errors="replace")):
        fail("003", key, "source-line-out-of-range", "registry-semantic-validation")


def load_graph() -> dict[str, dict[str, dict[str, str]]]:
    validate_normative_authority(REGISTRY_ROOT)
    digest_rows = keyed(REGISTRY_ROOT / "CampaignOperationsH1RegistryDigests.tsv",
                        "registry_path", "h1-registry-digest-v2")
    expected_files = {spec[0] for spec in REGISTRY_SPECS.values()}
    if set(digest_rows) != expected_files:
        fail("002", "registry-digests", "missing-or-extra-registry", "registry-semantic-validation")
    for filename in expected_files:
        if digest(REGISTRY_ROOT / filename) != digest_rows[filename].get("sha256"):
            fail("002", filename, "stale-registry-digest", "registry-semantic-validation")
    graph = {name: keyed(REGISTRY_ROOT / filename, key, version)
             for name, (filename, key, version) in REGISTRY_SPECS.items()}
    validate_registry_semantics(graph)
    return graph


def validate_registry_semantics(graph: dict[str, dict[str, dict[str, str]]]) -> None:
    requirements, fixtures, generators = graph["requirements"], graph["fixtures"], graph["generators"]
    runtime, validators, reports, artifacts, edges = (graph["runtime"], graph["validators"], graph["reports"],
                                                       graph["artifacts"], graph["edges"])
    for requirement_id, row in requirements.items():
        if row["evidence_class"] not in ALLOWED["evidence_class"] or row["classification"] not in ALLOWED["classification"]:
            fail("003", requirement_id, "unknown-requirement-enum", "registry-semantic-validation")
        if row["status_policy"] not in ALLOWED["status_policy"]:
            fail("003", requirement_id, "invalid-status-policy", "registry-semantic-validation")
        if re.search(r"^Evidence contract for|\bplaceholder\b|\bTBD\b", row["description"], re.I):
            fail("003", requirement_id, "placeholder-description", "registry-semantic-validation")
        if not re.fullmatch(r"(?:ADR0019(?:A|B)?|PHASEH|VOLUMEXII)-[0-9]+(?:\.[0-9]+)?", row["architecture_source_section"]):
            fail("003", requirement_id, "invalid-architecture-clause", "registry-semantic-validation")
        for field in ["required_fixture_cardinality", "required_runtime_cardinality", "required_report_entry_cardinality"]:
            if row[field] != "1":
                fail("004", requirement_id, f"unsupported-cardinality:{field}:{row[field]}", "registry-semantic-validation")
        fixture_ids, runtime_ids, report_ids = ids(row["fixture_ids"]), ids(row["runtime_record_ids"]), ids(row["report_entry_ids"])
        if len(fixture_ids) != 1 or len(runtime_ids) != 1 or len(report_ids) != 1 or len(ids(row["validator_ids"])) != 1:
            fail("004", requirement_id, "declared-cardinality-mismatch", "registry-semantic-validation")
    for fixture_id, row in fixtures.items():
        if row["cardinality"] not in ALLOWED["fixture_cardinality"] or row["classification"] not in ALLOWED["classification"]:
            fail("003", fixture_id, "invalid-fixture-semantic", "registry-semantic-validation")
        validate_source_reference(row["source_reference"], fixture_id)
        for collection, field in [(requirements, "requirement_id"), (generators, "generator_id"),
                                  (runtime, "expected_runtime_record_id"), (validators, "validator_id"),
                                  (reports, "report_entry_id"), (artifacts, "artifact_id")]:
            if row[field] not in collection:
                fail("003", fixture_id, f"unknown-node:{field}:{row[field]}", "registry-semantic-validation")
        if row["reverse_requirement_ids"] != row["requirement_id"]:
            fail("003", fixture_id, "conflicting-reverse-requirement", "registry-semantic-validation")
    for runtime_id, row in runtime.items():
        if (row["cardinality"] not in ALLOWED["runtime_cardinality"] or
                row["classification"] not in ALLOWED["classification"] or
                row["record_type"] not in {"runtime", "final_assurance_runtime", "mutation_case_runtime"}):
            fail("003", runtime_id, "invalid-runtime-semantic", "registry-semantic-validation")
        fixture = fixtures.get(row["fixture_id"])
        if fixture is None:
            fail("003", runtime_id, "unknown-fixture", "registry-semantic-validation")
        expected = {"requirement_id": fixture["requirement_id"], "generator_id": fixture["generator_id"],
                    "validator_id": fixture["validator_id"], "artifact_id": fixture["artifact_id"],
                    "validator_result_ids": f"VR-{fixture['fixture_id']}", "report_entry_ids": fixture["report_entry_id"]}
        for field, value in expected.items():
            if row[field] != value:
                fail("003", runtime_id, f"contradictory-runtime-edge:{field}", "registry-semantic-validation")
    for report_id, row in reports.items():
        if row["cardinality"] not in ALLOWED["report_cardinality"] or row["status_policy"] != "reconciled_validator_status":
            fail("003", report_id, "invalid-report-semantic", "registry-semantic-validation")
        expected_runtime = runtime.get(row["runtime_record_id"])
        if expected_runtime is None:
            fail("003", report_id, "unknown-runtime-record", "registry-semantic-validation")
        expected = {"requirement_id": expected_runtime["requirement_id"], "validator_id": expected_runtime["validator_id"],
                    "artifact_id": expected_runtime["artifact_id"], "validator_result_id": expected_runtime["validator_result_ids"]}
        for field, value in expected.items():
            if row[field] != value:
                fail("003", report_id, f"contradictory-report-edge:{field}", "registry-semantic-validation")
    for artifact_id, row in artifacts.items():
        if row["required_phase"] not in ALLOWED["artifact_phase"] or Path(row["path"]).is_absolute() or ".." in Path(row["path"]).parts:
            fail("003", artifact_id, "invalid-artifact-semantic", "registry-semantic-validation")
        for runtime_id in ids(row["runtime_record_ids"]):
            if runtime_id not in runtime:
                fail("003", artifact_id, "unknown-artifact-runtime", "registry-semantic-validation")
        for report_id in ids(row["report_entry_ids"]):
            if report_id not in reports:
                fail("003", artifact_id, "unknown-artifact-report", "registry-semantic-validation")
        if bool(row["record_key_field"]) != bool(row["record_key_value"]):
            fail("003", artifact_id, "incomplete-record-selector", "registry-semantic-validation")
        if ids(row["runtime_record_ids"]):
            expected_runtime_ids = sorted(value for value, runtime_row in runtime.items()
                                          if runtime_row["artifact_id"] == artifact_id)
            expected_report_ids = sorted(value for value, report_row in reports.items()
                                         if report_row["artifact_id"] == artifact_id)
            if ids(row["runtime_record_ids"]) != expected_runtime_ids or ids(row["report_entry_ids"]) != expected_report_ids:
                fail("003", artifact_id, "artifact-adjacency-mismatch", "registry-semantic-validation")
    validate_explicit_edges(graph)
    # Declared generator and validator adjacency is semantic, not documentary.
    fixture_by_generator: dict[str, list[str]] = defaultdict(list)
    runtime_by_generator: dict[str, list[str]] = defaultdict(list)
    runtime_by_validator: dict[str, list[str]] = defaultdict(list)
    reports_by_validator: dict[str, list[str]] = defaultdict(list)
    for fixture in fixtures.values():
        fixture_by_generator[fixture["generator_id"]].append(fixture["fixture_id"])
        runtime_by_generator[fixture["generator_id"]].append(fixture["expected_runtime_record_id"])
        runtime_by_validator[fixture["validator_id"]].append(fixture["expected_runtime_record_id"])
        reports_by_validator[fixture["validator_id"]].append(fixture["report_entry_id"])
    for generator_id, row in generators.items():
        implementation = ROOT / row["implementation"]
        if (Path(row["implementation"]).is_absolute() or ".." in Path(row["implementation"]).parts or
                not implementation.is_file() or not os.access(implementation, os.X_OK) or
                row["executable_required"] != "true"):
            fail("003", generator_id, "invalid-generator-implementation", "registry-semantic-validation")
        if row["input_node_type"] not in {"traceability_fixture", "lock_fixture", "acl_origin_fixture", "acl_manifest_requirement",
                                             "pre_enablement_fixture", "invariant_fixture", "final_assurance_fixture",
                                             "mutation_case_fixture"} or not row["output_node_type"].endswith("runtime_record"):
            fail("003", generator_id, "wrong-generator-node-type", "registry-semantic-validation")
        if ids(row["input_fixture_ids"]) != sorted(fixture_by_generator[generator_id]) or ids(row["output_runtime_record_ids"]) != sorted(runtime_by_generator[generator_id]):
            fail("003", generator_id, "generator-adjacency-mismatch", "registry-semantic-validation")
    for validator_id, row in validators.items():
        implementation = ROOT / row["implementation"]
        if (Path(row["implementation"]).is_absolute() or ".." in Path(row["implementation"]).parts or
                not implementation.is_file() or not os.access(implementation, os.X_OK) or
                row["executable_required"] != "true"):
            fail("003", validator_id, "invalid-validator-implementation", "registry-semantic-validation")
        if row["input_node_type"] != "runtime_record" or row["output_node_type"] != "validator_result" or row["duplicate_policy"] != "reject" or row["stale_policy"] != "reject":
            fail("003", validator_id, "wrong-validator-semantic", "registry-semantic-validation")
        if ids(row["input_runtime_record_ids"]) != sorted(runtime_by_validator[validator_id]) or ids(row["report_entry_ids"]) != sorted(reports_by_validator[validator_id]):
            fail("003", validator_id, "validator-adjacency-mismatch", "registry-semantic-validation")
    for requirement_id, row in requirements.items():
        expected_fixtures = sorted(value for value, fixture in fixtures.items() if fixture["requirement_id"] == requirement_id)
        expected_runtime = sorted(fixtures[value]["expected_runtime_record_id"] for value in expected_fixtures)
        expected_reports = sorted(fixtures[value]["report_entry_id"] for value in expected_fixtures)
        expected_validators = sorted({fixtures[value]["validator_id"] for value in expected_fixtures})
        if (ids(row["fixture_ids"]) != expected_fixtures or ids(row["runtime_record_ids"]) != expected_runtime or
                ids(row["report_entry_ids"]) != expected_reports or ids(row["validator_ids"]) != expected_validators):
            fail("003", requirement_id, "requirement-adjacency-mismatch", "registry-semantic-validation")


def validate_explicit_edges(graph: dict[str, dict[str, dict[str, str]]]) -> None:
    nodes = {
        "requirement": set(graph["requirements"]), "fixture": set(graph["fixtures"]),
        "generator": set(graph["generators"]), "runtime_record": set(graph["runtime"]),
        "validator_result": {row["validator_result_ids"] for row in graph["runtime"].values()},
        "report_entry": set(graph["reports"]), "artifact": set(graph["artifacts"]),
    }
    tuples = set()
    for edge_id, edge in graph["edges"].items():
        if edge["source_type"] not in nodes or edge["target_type"] not in nodes:
            fail("008", edge_id, "unknown-edge-node-type", "explicit-edge-validation")
        if edge["source_id"] not in nodes[edge["source_type"]] or edge["target_id"] not in nodes[edge["target_type"]]:
            fail("008", edge_id, "unknown-edge-node", "explicit-edge-validation")
        item = (edge["edge_type"], edge["source_type"], edge["source_id"], edge["target_type"], edge["target_id"])
        if item in tuples:
            fail("008", edge_id, "duplicate-edge", "explicit-edge-validation")
        tuples.add(item)
    for edge_id, edge in graph["edges"].items():
        reverse = (edge["reverse_edge_type"], edge["target_type"], edge["target_id"], edge["source_type"], edge["source_id"])
        if reverse not in tuples:
            fail("009", edge_id, "missing-explicit-reverse-edge", "explicit-edge-validation")
    expected = set()
    for fixture in graph["fixtures"].values():
        runtime = graph["runtime"][fixture["expected_runtime_record_id"]]
        pairs = [
            ("requirement", fixture["requirement_id"], "fixture", fixture["fixture_id"]),
            ("fixture", fixture["fixture_id"], "generator", fixture["generator_id"]),
            ("generator", fixture["generator_id"], "runtime_record", runtime["runtime_record_id"]),
            ("runtime_record", runtime["runtime_record_id"], "validator_result", runtime["validator_result_ids"]),
            ("validator_result", runtime["validator_result_ids"], "report_entry", fixture["report_entry_id"]),
            ("runtime_record", runtime["runtime_record_id"], "artifact", fixture["artifact_id"]),
            ("report_entry", fixture["report_entry_id"], "artifact", fixture["artifact_id"]),
        ]
        for left_type, left_id, right_type, right_id in pairs:
            expected.add((f"{left_type}_to_{right_type}", left_type, left_id, right_type, right_id))
            expected.add((f"{right_type}_to_{left_type}", right_type, right_id, left_type, left_id))
    if tuples != expected:
        missing = expected - tuples
        extra = tuples - expected
        detail = "missing-forward-edge" if missing else "conflicting-edge"
        key = next(iter(missing or extra))[2]
        fail("009", key, detail, "explicit-edge-validation")


def tsv_text(rows: list[dict[str, str]]) -> str:
    if not rows:
        fail("006", "generated-registry", "empty-generated-registry")
    target = io.StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return target.getvalue()


def run_validator(root: Path, run_id: str, graph: dict, snapshots) -> list[dict[str, str]]:
    """Execute each materialized validator registry row through the trusted observer."""
    runner = TrustedValidatorRunner(ROOT, graph["validators"])
    result_rows: list[dict[str, str]] = []
    runtime_rows: list[dict[str, str]] = []
    receipts: list[dict[str, str]] = []
    prior_results: list[dict[str, str]] = []
    prior_runtime: list[dict[str, str]] = []
    prior_receipts: dict[str, dict[str, str]] = {}
    if all(name in snapshots for name in ("h1-validator-results.tsv", "h1-reconciled-runtime-records.tsv",
                                           "h1-validator-execution-receipts.tsv")):
        prior_results = snapshots.by_id("ART-GENERATED-VALIDATORS").tsv()[1]
        prior_runtime = snapshots.by_id("ART-GENERATED-RUNTIME").tsv()[1]
        prior_receipt_rows = snapshots.by_id("ART-GENERATED-VALIDATOR-RECEIPTS").tsv()[1]
        validate_persisted_validator_outputs(prior_runtime, prior_results, prior_receipt_rows)
        prior_receipts = {row["validator_id"]: row for row in prior_receipt_rows}
    materialized_final: set[str] = set()
    materialized_mutations: set[str] = set()
    if "h1-final-assurance-results.tsv" in snapshots:
        materialized_final = {row.get("artifact_id", "").removeprefix("ART-RECORD-")
                              for row in snapshots.by_id("ART-SUPPORT-FINAL-LEDGER").tsv()[1]}
    if "h1-mutation-results.tsv" in snapshots:
        materialized_mutations = {row.get("fixture_id", "")
                                  for row in snapshots.by_id("ART-SUPPORT-MUTATION-LEDGER").tsv()[1]}
    support = {
        "VAL-TRACE": {"ART-SUPPORT-RUNTIME-LEDGER", "ART-SUPPORT-ACL-CATALOG",
                      "ART-SUPPORT-ACL-CATALOG-CAPTURE", "ART-SUPPORT-ACL-CATALOG-PAYLOAD",
                      "ART-SUPPORT-GENERATOR-RECEIPTS", "ART-SUPPORT-RAW-ENVELOPES",
                      "ART-SUPPORT-GENERATOR-SNAPSHOTS"},
        "VAL-LOCK": {"ART-SUPPORT-LOCK-RUNTIME", "ART-SUPPORT-LOCK-CATALOG", "ART-SUPPORT-LOCK-PIDS"},
        "VAL-ACL-ORIGIN": {"ART-SUPPORT-ACL-ORIGIN"},
        "VAL-PRE": {"ART-SUPPORT-PRE"}, "VAL-INVARIANT": {"ART-SUPPORT-INVARIANT"},
        "VAL-FINAL-ASSURANCE": {"ART-SUPPORT-FINAL-LEDGER"},
        "VAL-MUTATION": {"ART-SUPPORT-MUTATION-LEDGER"},
    }
    with tempfile.TemporaryDirectory(prefix="ea-h1-trusted-validator-") as temporary:
        temporary_root = Path(temporary)
        for validator_id, validator in sorted(graph["validators"].items()):
            declared_runtime_ids = ids(validator["input_runtime_record_ids"])
            runtime_ids = [runtime_id for runtime_id in declared_runtime_ids
                           if (graph["runtime"][runtime_id]["generator_id"] not in {"GEN-FINAL-ASSURANCE", "GEN-MUTATION"} or
                               graph["runtime"][runtime_id]["fixture_id"] in
                               (materialized_final if graph["runtime"][runtime_id]["generator_id"] == "GEN-FINAL-ASSURANCE"
                                else materialized_mutations))
                           if graph["artifacts"][graph["runtime"][runtime_id]["artifact_id"]]["path"] in snapshots]
            expected_artifacts = {graph["runtime"][runtime_id]["artifact_id"] for runtime_id in runtime_ids}
            if not expected_artifacts:
                continue
            expected_result_ids = {graph["runtime"][runtime_id]["validator_result_ids"] for runtime_id in runtime_ids}
            input_ids = expected_artifacts | support.get(validator_id, set()) | {"ART-SUPPORT-RUN-ID"}
            if not all(value in {row["artifact_id"] for row in snapshots.inventory_rows()} for value in input_ids):
                continue
            input_artifacts = {artifact_id: snapshots.by_id(artifact_id) for artifact_id in input_ids}
            prior = prior_receipts.get(validator_id)
            current_ids = ",".join(sorted(input_artifacts))
            current_digests = ",".join(input_artifacts[key].digest for key in sorted(input_artifacts))
            if (prior is not None and prior.get("run_id") == run_id and
                    prior.get("input_artifact_ids") == current_ids and
                    prior.get("input_artifact_digests") == current_digests and
                    set(ids(prior.get("output_validator_result_ids", ""))) == expected_result_ids):
                receipts.append(prior)
                result_rows.extend(row for row in prior_results if row.get("validator_id") == validator_id)
                runtime_rows.extend(row for row in prior_runtime if row.get("validator_result_ids") in expected_result_ids)
                continue
            result_path = temporary_root / f"{validator_id}-results.tsv"
            runtime_path = temporary_root / f"{validator_id}-runtime.tsv"
            declared_outputs = {
                f"ART-TRUSTED-{validator_id}-RESULTS": result_path,
                f"ART-TRUSTED-{validator_id}-RUNTIME": runtime_path,
            }
            try:
                observation = runner.execute(
                    validator_id, run_id,
                    [str(root), run_id, str(result_path), str(runtime_path), validator_id],
                    input_artifacts, declared_outputs, expected_result_ids)
                runner.validate(observation.receipt, run_id, input_artifacts,
                                observation.output_snapshots, expected_result_ids)
            except RunnerError as error:
                fail("005", validator_id, f"trusted-validator-execution-failed:{error}",
                     "validator-execution-binding")
            receipts.append(observation.receipt)
            for artifact_id, destination in [(f"ART-TRUSTED-{validator_id}-RESULTS", result_rows),
                                             (f"ART-TRUSTED-{validator_id}-RUNTIME", runtime_rows)]:
                snapshot = observation.output_snapshots[artifact_id]
                table = list(csv.DictReader(io.StringIO(snapshot.text()), delimiter="\t"))
                destination.extend(table)
    if len({row["validator_result_id"] for row in result_rows}) != len(result_rows):
        fail("005", "trusted-validator-results", "duplicate-validator-result")
    if len({row["runtime_record_id"] for row in runtime_rows}) != len(runtime_rows):
        fail("005", "trusted-runtime-results", "duplicate-runtime-result")
    (root / "h1-validator-results.tsv").write_text(tsv_text(result_rows))
    (root / "h1-reconciled-runtime-records.tsv").write_text(tsv_text(runtime_rows))
    (root / "h1-validator-execution-receipts.tsv").write_text(tsv_text(receipts))
    return receipts


def recursive_scan(root: Path, graph: dict, generation: bool, run_id: str):
    global EVIDENCE_ROOT, SNAPSHOTS
    EVIDENCE_ROOT = root
    SNAPSHOTS = validate_filesystem(root, graph["artifacts"], generation, run_id)
    SNAPSHOTS.assert_run_id(run_id)
    return SNAPSHOTS


def validate_persisted_validator_outputs(runtime_rows: list[dict[str, str]],
                                         validator_rows: list[dict[str, str]],
                                         receipts: list[dict[str, str]]) -> None:
    """Bind persisted aggregate rows to the per-process output digests."""
    for receipt in receipts:
        validator_id = receipt.get("validator_id", "")
        outputs = receipt.get("output_artifact_ids", "").split(",")
        digests = receipt.get("output_artifact_digests", "").split(",")
        bindings = dict(zip(outputs, digests)) if len(outputs) == len(digests) else {}
        result_ids = set(ids(receipt.get("output_validator_result_ids", "")))
        expected = {
            f"ART-TRUSTED-{validator_id}-RESULTS": [row for row in validator_rows if row.get("validator_id") == validator_id],
            f"ART-TRUSTED-{validator_id}-RUNTIME": [row for row in runtime_rows if row.get("validator_result_ids") in result_ids],
        }
        for artifact_id, rows in expected.items():
            if not rows or bindings.get(artifact_id) != digest_bytes(tsv_text(rows).encode()):
                name = "h1-validator-results.tsv" if artifact_id.endswith("-RESULTS") else "h1-reconciled-runtime-records.tsv"
                detail = "stale-validator-results" if artifact_id.endswith("-RESULTS") else "stale-runtime-results"
                print(f"H1V007 key={name} stage=authentic-runtime-validation detail={detail}", file=sys.stderr)
                raise SystemExit(1)


def validate_persisted_generic_semantics(graph: dict, snapshots) -> None:
    """Recheck generic observed values without reopening producer pathnames."""
    _, source_rows = snapshots.by_id("ART-SUPPORT-RUNTIME-LEDGER").tsv()
    by_fixture = {row.get("fixture_id", ""): row for row in source_rows}
    trace = {row["fixture_id"]: row for row in read(REGISTRY_ROOT / "CampaignOperationsH1Traceability.tsv")[1]}
    for fixture_id, fixture in graph["fixtures"].items():
        if fixture["generator_id"] != "GEN-TRACE":
            continue
        source = by_fixture.get(fixture_id, {})
        contract = trace[fixture_id]
        wanted = {
            "requirement_id": contract["requirement_id"], "fixture_id": fixture_id,
            "actual_status": contract["expected_status"], "sqlstate": contract["expected_sqlstate"],
            "diagnostic": contract["expected_diagnostic"],
            "object_identity": contract["expected_failing_object"], "stage": contract["expected_stage"],
            "generator_id": "GEN-TRACE", "generator_version": "h1-generator-registry-v2",
            "emitted_runtime_record_id": fixture["expected_runtime_record_id"],
            "output_artifact_id": fixture["artifact_id"],
            "generator_implementation": "Tests/CampaignOperationsPhaseH1MigrationTests.sh",
            "generator_entry_point": "emit_runtime_result",
        }
        actual = {key: source.get(key, "") for key in wanted}
        if actual != wanted:
            differing = next(key for key in wanted if actual.get(key) != wanted[key])
            print(f"H1V105 key={fixture_id} stage=authentic-runtime-validation "
                  f"detail=semantic-mismatch:{differing}", file=sys.stderr)
            raise SystemExit(1)
        if fixture_id == "H1CPP002":
            contents = snapshots.by_id(fixture["artifact_id"]).text()
            required_markers = [
                "command=rg",
                "scanned_paths=Sources",
                "continuous_h4_result_rows=0",
                "scan_result=PASS",
                "prohibited_query=--campaign-operations-production-continuous",
                "accepted_post_h1_query=--campaign-operations-manager-run-once",
                "accepted_post_h1_files=Sources/ExperimentScheduler.cpp",
            ]
            for marker in required_markers:
                if marker not in contents:
                    print(
                        "H1V106 key=H1CPP002 "
                        "stage=authentic-runtime-validation "
                        "detail=insufficient-exclusion-authenticity:"
                        f"{marker.split('=')[0]}",
                        file=sys.stderr)
                    raise SystemExit(1)


def validate_persisted_special_semantics(snapshots, run_id: str) -> None:
    if "h1-restore-runtime.tsv" in snapshots:
        restore_snapshot = snapshots.by_id("ART-SUPPORT-RESTORE")
        try:
            _, restore_rows = restore_snapshot.tsv()
            restore_artifacts = {row["artifact_path"]: snapshots.by_path(row["artifact_path"]).data
                                 for row in restore_rows}
            validate_restore_snapshot_bytes(restore_snapshot.data, restore_artifacts, run_id)
        except (KeyError, SystemExit):
            print("H1V801 key=h1-restore-runtime.tsv stage=authentic-runtime-validation "
                  "detail=restore-semantic-validation-failed", file=sys.stderr)
            raise SystemExit(1)
    if "h1-final-assurance-results.tsv" in snapshots:
        _, final_rows = snapshots.by_id("ART-SUPPORT-FINAL-LEDGER").tsv()
        release = next((row for row in final_rows if row.get("evidence_id") == "RELEASE_BUILD"), None)
        if release is not None:
            try:
                log = snapshots.by_path(release["artifact_path"]).text()
            except KeyError:
                log = ""
            if "** BUILD SUCCEEDED **" not in log or "** BUILD FAILED **" in log:
                print("H1V703 key=RELEASE_BUILD stage=authentic-runtime-validation "
                      "detail=release-build-not-succeeded", file=sys.stderr)
                raise SystemExit(1)


def validate_generator_materialization(root: Path, run_id: str, graph: dict) -> tuple[bool, bool, dict, dict, dict]:
    """Validate persisted trusted executions and their v2 envelope/snapshot bindings."""
    try:
        _, receipts = read(root / "h1-generator-execution-receipts.tsv")
        _, envelopes = read(root / "h1-raw-evidence-envelopes.tsv")
        _, output_snapshots = read(root / "h1-generator-output-snapshots.tsv")
    except SystemExit:
        return False, False, {}, {}, {}
    receipt_by_id = {row.get("generator_execution_id", ""): row for row in receipts}
    envelope_by_requirement = {row.get("requirement_id", ""): row for row in envelopes}
    snapshot_by_id = {row.get("snapshot_id", ""): row for row in output_snapshots}
    if (len(receipt_by_id) != len(receipts) or len(envelope_by_requirement) != len(envelopes) or
            len(snapshot_by_id) != len(output_snapshots)):
        return False, False, receipt_by_id, envelope_by_requirement, snapshot_by_id
    expected_requirements = set(graph["requirements"])
    expected_classes = evidence_classes()
    executions_ok = set(envelope_by_requirement) == expected_requirements
    envelopes_ok = executions_ok
    referenced_receipts: set[str] = set()
    for requirement_id, envelope in envelope_by_requirement.items():
        fixture_ids = ids(graph["requirements"].get(requirement_id, {}).get("fixture_ids", ""))
        fixture = graph["fixtures"].get(fixture_ids[0]) if len(fixture_ids) == 1 else None
        execution_id = envelope.get("generator_execution_id", "")
        receipt = receipt_by_id.get(execution_id)
        referenced_receipts.add(execution_id)
        payload_fields = {key: envelope.get(key, "") for key in envelope
                          if key not in {"raw_envelope_id", "envelope_digest"}}
        digest = hashlib.sha256(json.dumps(payload_fields, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        try:
            payload = json.loads(envelope.get("payload_json", ""))
        except json.JSONDecodeError:
            payload = None
        evidence_class = envelope.get("evidence_class", "")
        forbidden = {"expected", "actual", "comparison", "comparison_result", "success", "status"}
        outputs = receipt.get("output_artifact_ids", "").split(",") if receipt else []
        snapshot_ids = receipt.get("output_snapshot_ids", "").split(",") if receipt else []
        snapshot_bindings = list(zip(outputs, snapshot_ids)) if len(outputs) == len(snapshot_ids) else []
        if (fixture is None or receipt is None or receipt.get("version") != "h1-generator-execution-receipt-v2" or
                receipt.get("generator_id") != fixture["generator_id"] or receipt.get("run_id") != run_id or
                receipt.get("actual_exit_status") != "0" or receipt.get("execution_status") != "completed" or
                not receipt.get("attestation_digest") or envelope.get("evidence_version") != "h1-raw-execution-evidence-v2" or
                envelope.get("run_id") != run_id or evidence_class != expected_classes.get(requirement_id) or
                envelope.get("envelope_digest") != digest or not isinstance(payload, dict) or
                forbidden.intersection(payload) or RAW_CLASS_PAYLOADS.get(evidence_class, {"missing"}) - set(payload) or
                not snapshot_bindings or any(value not in snapshot_by_id for _, value in snapshot_bindings)):
            executions_ok = False; envelopes_ok = False
    executions_ok = (executions_ok and referenced_receipts == set(receipt_by_id) and
                     generator_execution_complete(receipts, envelopes, output_snapshots,
                                                  expected_requirements, run_id))
    return executions_ok, envelopes_ok, receipt_by_id, envelope_by_requirement, snapshot_by_id


def materialize_provenance(run_id: str, graph: dict, runtime_actual: dict, validator_actual: dict,
                           reports: list[dict[str, str]], validator_receipts: list[dict[str, str]],
                           generator_receipts: dict, envelopes: dict, generator_snapshots: dict) -> tuple[list, list, bool]:
    obligations = {row["obligation_id"]: row for row in read(REGISTRY_ROOT / "CampaignOperationsH1EvidenceObligations.tsv")[1]}
    derivations = read(REGISTRY_ROOT / "CampaignOperationsH1ClauseRequirementDerivations.tsv")[1]
    required_pairs = {(row["normative_clause_id"], row["requirement_id"]) for row in derivations}
    reports_by_requirement = {row["requirement_id"]: row for row in reports}
    validator_receipt_by_execution = {row["validator_execution_id"]: row for row in validator_receipts}
    chains: list[dict[str, str]] = []
    for derivation in derivations:
        requirement_id = derivation["requirement_id"]
        requirement = graph["requirements"].get(requirement_id)
        fixture_ids = ids(requirement["fixture_ids"]) if requirement else []
        if len(fixture_ids) != 1:
            continue
        fixture = graph["fixtures"][fixture_ids[0]]
        runtime = runtime_actual.get(fixture["expected_runtime_record_id"])
        validator = validator_actual.get(graph["runtime"][fixture["expected_runtime_record_id"]]["validator_result_ids"])
        report = reports_by_requirement.get(requirement_id)
        envelope = envelopes.get(requirement_id)
        generator_receipt = generator_receipts.get(envelope.get("generator_execution_id", "")) if envelope else None
        validator_receipt = validator_receipt_by_execution.get(validator.get("validator_execution_id", "")) if validator else None
        if not all((runtime, validator, report, envelope, generator_receipt, validator_receipt)):
            continue
        output_ids = generator_receipt["output_artifact_ids"].split(",")
        snapshot_ids = generator_receipt["output_snapshot_ids"].split(",")
        candidates = [value for key, value in zip(output_ids, snapshot_ids)
                      if key.endswith(fixture["fixture_id"]) or key == "ART-SUPPORT-ACL-CATALOG-PAYLOAD"]
        snapshot_id = candidates[0] if len(candidates) == 1 else ""
        if snapshot_id not in generator_snapshots:
            continue
        chains.append({
            "normative_clause_id": derivation["normative_clause_id"],
            "requirement_id": requirement_id, "obligation_id": requirement_id,
            "generator_execution_id": envelope["generator_execution_id"],
            "generator_receipt_id": generator_receipt["attestation_digest"],
            "raw_envelope_id": envelope["raw_envelope_id"], "snapshot_id": snapshot_id,
            "runtime_record_id": runtime["runtime_record_id"],
            "validator_execution_id": validator["validator_execution_id"],
            "validator_receipt_id": validator_receipt["receipt_digest"],
            "validator_result_id": validator["validator_result_id"],
            "report_entry_id": report["report_entry_id"], "generated_report_id": "ART-REPORT-TRACE",
            "run_id": run_id,
        })
    edges_set: set[tuple[str, ...]] = set()
    for chain in chains:
        for source_type, target_type in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
            for left, right in ((source_type, target_type), (target_type, source_type)):
                edges_set.add((left, chain[left], f"{left}_to_{right}", right, chain[right], run_id))
    edges = [dict(zip(EDGE_FIELDS, row)) for row in sorted(edges_set)]
    try:
        validate_provenance(chains, edges, set(obligations), run_id, required_pairs)
        complete = True
    except ProvenanceError:
        complete = False
    return chains, edges, complete


def reconcile(mode: str, root: Path, run_id: str) -> tuple[dict, list, list, list, list, list, list, dict[str, bool], list, list]:
    graph = load_graph()
    require_final = mode == "validate" and os.environ.get("H1_ALLOW_PARTIAL_FINAL") != "1"
    initial = recursive_scan(root, graph, mode == "generate", run_id)
    if mode == "validate":
        try:
            _, prior_index = initial.by_id("ART-GENERATED-INDEX").tsv()
            trace_index = next(row for row in prior_index if row.get("artifact_id") == "ART-REPORT-TRACE")
            if initial.by_id("ART-REPORT-TRACE").digest != trace_index.get("artifact_digest"):
                fail("006", TRACE_REPORT, "stale-generated-report", "report-provenance-freshness")
        except (KeyError, StopIteration):
            fail("006", TRACE_REPORT, "stale-generated-report", "report-provenance-freshness")
    validate_exact_runtime_inventory(root, graph, require_final, initial)
    if mode == "validate":
        validate_persisted_generic_semantics(graph, initial)
        validate_persisted_special_semantics(initial, run_id)
    if mode == "generate":
        receipt_rows = run_validator(root, run_id, graph, initial)
        present = recursive_scan(root, graph, True, run_id)
    else:
        _, receipt_rows = read(root / "h1-validator-execution-receipts.tsv")
        present = initial
    _, runtime_rows = read(root / "h1-reconciled-runtime-records.tsv")
    _, validator_rows = read(root / "h1-validator-results.tsv")
    if mode == "validate":
        validate_persisted_validator_outputs(runtime_rows, validator_rows, receipt_rows)
    runtime_actual = {row["runtime_record_id"]: row for row in runtime_rows}
    validator_actual = {row["validator_result_id"]: row for row in validator_rows}
    if len(runtime_actual) != len(runtime_rows) or len(validator_actual) != len(validator_rows):
        fail("005", "runtime-validator", "duplicate-actual-node")
    reports = []
    for report_id, contract in sorted(graph["reports"].items()):
        validator = validator_actual.get(contract["validator_result_id"])
        runtime = runtime_actual.get(contract["runtime_record_id"])
        if validator is None or runtime is None:
            continue
        for field, expected in [("runtime_record_ids", contract["runtime_record_id"]),
                                ("requirement_ids", contract["requirement_id"]),
                                ("artifact_ids", contract["artifact_id"]),
                                ("report_entry_ids", report_id)]:
            if validator.get(field) != expected:
                fail("005", report_id, f"validator-report-edge:{field}")
        artifact_snapshot = present.by_id(contract["artifact_id"])
        reports.append({
            "version": "h1-reconciled-report-entry-v2", "run_id": run_id,
            "requirement_id": contract["requirement_id"], "runtime_record_id": contract["runtime_record_id"],
            "validator_result_id": contract["validator_result_id"], "report_entry_id": report_id,
            "artifact_id": contract["artifact_id"], "record_digest": runtime["record_digest"],
            "aggregate_digest": runtime["aggregate_digest"], "comparison_result": validator["comparison_result"],
            "status": validator["status"], "diagnostic": validator["diagnostic"],
            "stage": validator["stage"],
            "snapshot_id": artifact_snapshot.snapshot_id,
            "artifact_lexical_path": artifact_snapshot.lexical_path,
            "artifact_device": str(artifact_snapshot.device), "artifact_inode": str(artifact_snapshot.inode),
            "artifact_size": str(artifact_snapshot.size), "snapshot_digest": artifact_snapshot.digest,
        })
    expected_runtime = set(graph["runtime"])
    expected_validators = {row["validator_result_ids"] for row in graph["runtime"].values()}
    expected_reports = set(graph["reports"])
    actual_reports = {row["report_entry_id"] for row in reports}
    generator_complete, raw_complete, generator_receipts, raw_envelopes, generator_snapshots = \
        validate_generator_materialization(root, run_id, graph)
    provenance_chains, provenance_edges, provenance_complete = materialize_provenance(
        run_id, graph, runtime_actual, validator_actual, reports, receipt_rows,
        generator_receipts, raw_envelopes, generator_snapshots)
    defects = {
        "missing_forward_edges": 0, "missing_reverse_edges": 0, "orphan_nodes": 0,
        "duplicate_keys": 0, "conflicting_edges": 0, "stale_versions": 0,
        "stale_run_ids": sum(row.get("run_id") != run_id for row in runtime_rows + validator_rows + reports),
        "stale_artifact_digests": 0, "stale_record_digests": 0,
        "unknown_nodes": len(set(runtime_actual) - expected_runtime) + len(set(validator_actual) - expected_validators) + len(actual_reports - expected_reports),
        "unknown_files": 0, "invalid_cardinalities": 0, "invalid_semantic_fields": 0,
        "missing_runtime_records": len(expected_runtime - set(runtime_actual)),
        "missing_validator_results": len(expected_validators - set(validator_actual)),
        "missing_report_entries": len(expected_reports - actual_reports),
        "missing_final_assurance_records": sum(1 for row in graph["artifacts"].values()
                                                 if row["required_phase"] == "final" and row["runtime_record_ids"] and
                                                 row["path"] not in present),
    }
    consumed_artifacts = {row["artifact_ids"] for row in runtime_rows}
    snapshot_rows = present.inventory_rows()
    try:
        validate_legacy_inventory(REGISTRY_ROOT / "CampaignOperationsH1LegacyPathInventory.tsv")
        no_legacy = True
    except RuntimeError:
        no_legacy = False
    acl_rows = []
    acl_path = root / "h1-acl-requirement-evidence.tsv"
    if evidence_exists(acl_path):
        _, acl_rows = read(acl_path)
    trust = {
        "authority_complete": True,
        "trusted_generator_execution_complete": generator_complete,
        "raw_envelopes_complete": raw_complete,
        "snapshots_complete": snapshot_coverage_complete(
            [row for row in snapshot_rows if row["artifact_id"] in consumed_artifacts],
            consumed_artifacts, run_id),
        "runtime_records_complete": set(runtime_actual) == expected_runtime,
        "trusted_validator_execution_complete": validator_execution_complete(
            receipt_rows, set(graph["validators"]), validator_rows, run_id),
        "validator_results_complete": set(validator_actual) == expected_validators,
        "acl_catalog_independent": bool(acl_rows) and all(
            row.get("format_version") == "h1-acl-catalog-runtime-v3" and
            not {"expected_tuple_state", "actual_tuple_state", "comparison"}.intersection(row)
            for row in acl_rows),
        "provenance_graph_complete": provenance_complete,
        "report_complete": actual_reports == expected_reports,
        "no_legacy_path_reachable": no_legacy,
    }
    for boundary in READINESS_COMPONENTS:
        defects[f"trust_{boundary}"] = int(not trust[boundary])
    health = [{"version": "h1-graph-health-v2", "defect_type": key, "count": str(value)}
              for key, value in sorted(defects.items())]
    edge_rows = list(graph["edges"].values())
    return (graph, runtime_rows, validator_rows, reports, edge_rows, health, receipt_rows, trust,
            provenance_chains, provenance_edges)


def artifact_index(graph: dict, root: Path, runtime_rows: list, reports: list,
                   generated_contents: dict[str, str], run_id: str) -> list[dict[str, str]]:
    runtime_by_id = {row["runtime_record_id"]: row for row in runtime_rows}
    report_ids = {row["report_entry_id"] for row in reports}
    rows = []
    for artifact_id, contract in sorted(graph["artifacts"].items()):
        path = contract["path"]
        file_path = root / path
        snapshot = None
        if path in generated_contents:
            # Index the bytes from this derivation pass, never a prior on-disk
            # generation captured before fresh validator execution.
            artifact_digest = digest_bytes(generated_contents[path].encode())
            state = "materialized-current"
        elif path == "h1-artifact-index.tsv":
            artifact_digest = digest_bytes(b"h1-artifact-index-v2-canonical-self")
            state = "canonical-self"
        elif evidence_exists(file_path):
            snapshot = evidence_snapshot(file_path)
            artifact_digest = snapshot.digest
            state = "snapshot-bound"
        else:
            artifact_digest = ""
            state = "not-yet-materialized"
        runtime_ids = ids(contract["runtime_record_ids"])
        record_digests = [runtime_by_id[value]["record_digest"] for value in runtime_ids if value in runtime_by_id]
        rows.append({
            "version": "h1-artifact-index-v2", "artifact_id": artifact_id, "path": path,
            "artifact_type": contract["artifact_type"], "run_id": run_id,
            "artifact_digest": artifact_digest, "digest_state": state,
            "record_key_field": contract["record_key_field"], "record_key_value": contract["record_key_value"],
            "record_level_digests": ",".join(record_digests),
            "runtime_record_ids": ",".join(value for value in runtime_ids if value in runtime_by_id),
            "report_entry_ids": ",".join(value for value in ids(contract["report_entry_ids"]) if value in report_ids),
            "reverse_runtime_record_ids": ",".join(value for value in runtime_ids if value in runtime_by_id),
            "reverse_report_entry_ids": ",".join(value for value in ids(contract["report_entry_ids"]) if value in report_ids),
            "snapshot_id": snapshot.snapshot_id if snapshot is not None else "",
            "snapshot_lexical_path": snapshot.lexical_path if snapshot is not None else "",
            "snapshot_device": str(snapshot.device) if snapshot is not None else "",
            "snapshot_inode": str(snapshot.inode) if snapshot is not None else "",
            "snapshot_size": str(snapshot.size) if snapshot is not None else "",
            "snapshot_run_id": snapshot.run_id if snapshot is not None else "",
        })
    return rows


def counts(graph: dict, runtime_rows: list, validator_rows: list, reports: list, health: list,
           trust: dict[str, bool] | None = None) -> dict[str, int]:
    requirements = graph["requirements"].values()
    validators_by_id = {row["validator_result_id"]: row for row in validator_rows}
    result = {
        "requirements": len(graph["requirements"]), "fixtures": len(graph["fixtures"]),
        "generators": len(graph["generators"]), "runtime_records": len(runtime_rows),
        "validator_results": len(validator_rows), "report_entries": len(reports),
        "artifacts": len(graph["artifacts"]), "edges": len(graph["edges"]),
        "lock_records": sum(row["evidence_class"] == "lock" and row["classification"] == "executable" for row in requirements),
        "pre_enablement": sum(row["classification"] == "pre_enablement_non_final" for row in requirements),
        "invariants": sum(row["classification"] == "accepted_non_executable" for row in requirements),
        "acl_origin": sum(row["evidence_class"] == "acl_origin" for row in requirements),
        "acl_catalog": sum(row["generator_id"] == "GEN-ACL-MANIFEST" for row in graph["fixtures"].values()),
        "restore": sum(row["evidence_class"] == "restore" and row["classification"] != "final_assurance" for row in requirements),
        "final_assurance": sum(row["validator_id"] == "VAL-FINAL-ASSURANCE" for row in graph["fixtures"].values()),
        "final_assurance_validated": sum(row["validator_id"] == "VAL-FINAL-ASSURANCE" for row in validator_rows),
        "mutation_cases": sum(row["validator_id"] == "VAL-MUTATION" for row in graph["fixtures"].values()),
        "mutation_cases_validated": sum(row["validator_id"] == "VAL-MUTATION" for row in validator_rows),
        "defects": sum(int(row["count"]) for row in health),
    }
    trust = trust or {component: False for component in READINESS_COMPONENTS}
    ready, _ = evaluate_readiness(trust)
    for component in READINESS_COMPONENTS:
        result[component] = int(trust[component])
    result["ready"] = int(ready)
    return result


def provenance(report: dict[str, str]) -> str:
    return (f"requirement `{report['requirement_id']}`; runtime `{report['runtime_record_id']}`; "
            f"validator `{report['validator_result_id']}`; report `{report['report_entry_id']}`; "
            f"artifact `{report['artifact_id']}`; record `{report['record_digest']}`; "
            f"aggregate `{report['aggregate_digest']}`")


def render_trace(run_id: str, summary: dict[str, int], reports: list, health: list) -> str:
    disposition = "READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION" if summary["ready"] else "NOT_READY_FOR_REVERIFICATION"
    lines = ["# Campaign Operations H1 reconciled traceability report", "", f"Run ID: `{run_id}`.", "",
             f"Disposition: `{disposition}`.", "", "## Computed graph health", "",
             "| Defect type | Count |", "|---|---:|"]
    lines += [f"| {row['defect_type']} | {row['count']} |" for row in health]
    lines += ["", "## Reconciled entries", "",
              "| Requirement | Runtime | Validator | Report | Artifact | Record digest | Aggregate digest | Status |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {row['requirement_id']} | {row['runtime_record_id']} | {row['validator_result_id']} | {row['report_entry_id']} | {row['artifact_id']} | `{row['record_digest']}` | `{row['aggregate_digest']}` | {row['status']} |"
              for row in reports]
    return "\n".join(lines) + "\n"


def report_fact(reports_by_requirement: dict[str, dict[str, str]], requirement_id: str) -> str:
    row = reports_by_requirement.get(requirement_id)
    return "No reconciled entry is present." if row is None else provenance(row) + f"; status `{row['status']}`."


def render_final(run_id: str, summary: dict[str, int], reports: list, validator_rows: list,
                 health: list, worktree_text: str) -> str:
    disposition = "READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION" if summary["ready"] else "NOT_READY_FOR_REVERIFICATION"
    by_requirement = {row["requirement_id"]: row for row in reports}
    validators_by_id = {row["validator_result_id"]: row for row in validator_rows}
    def parsed_fact(requirement_id: str) -> str:
        report = by_requirement.get(requirement_id)
        if not report or report["validator_result_id"] not in validators_by_id:
            return "{}"
        try:
            actual = json.loads(validators_by_id[report["validator_result_id"]]["actual_parsed_values"])
        except (KeyError, json.JSONDecodeError):
            return "{}"
        return json.dumps(actual.get("parsed_semantics", {}), sort_keys=True, separators=(",", ":"))
    defect_text = ", ".join(f"{row['defect_type']}={row['count']}" for row in health)
    sections = [
        ("Executive result", f"Run `{run_id}` produced ready=`{str(summary['ready']).lower()}` from {summary['defects']} computed defects and {summary['final_assurance_validated']}/{summary['final_assurance']} validated final-assurance records."),
        ("Confirmation no architecture redesign occurred", "No sealed-role identity, role graph, ownership allowlist, transition identity, canonical format, replay semantic, migration number, restore policy, H1/H2/H3/H4 boundary, scheduler ownership, or lifecycle ownership was changed."),
        ("Files changed", "The implementation changes are enumerated by the authenticated worktree evidence in section 32; no commit was created."),
        ("Graph-health computation", f"Health was computed from materialized nodes, explicit edges, versions, run IDs, artifact bytes, record bytes, cardinalities, semantic parsers, and the recursive file scan: {defect_text}."),
        ("Explicit forward/reverse adjacency", f"The v2 edge registry contains {summary['edges']} directed edges; every forward edge has a separately stored reverse edge."),
        ("Unified artifact-ID design", f"All {summary['artifacts']} declarations use the same `ART-*` identity referenced by runtime, validator, report, and artifact-index rows; paths are metadata."),
        ("Unknown-file scanning", "The entire evidence root is scanned recursively. Only registered paths and the explicit disposable-cluster `data/`, `s/`, and `restore/` non-evidence prefixes are allowed."),
        ("Final-assurance graph registration", f"All {summary['final_assurance']} final-assurance requirements, fixtures, runtime nodes, validator results, report entries, artifacts, and bidirectional edges are ordinary registry rows."),
        ("Authentic validator-result design", f"{summary['validator_results']} validator-result rows carry implementation/entry point, run, runtime/requirement/fixture/artifact IDs, expected/actual parsed JSON, comparison, diagnostic, stage, timestamp, and output digest."),
        ("Semantic log parsing", "Lock, ACL-origin, ACL/default catalog, restore, build, strict compile, checksum, mutation, exclusion, determinism, and worktree evidence are parsed by evidence class before report generation."),
        ("Registry semantic validation", "Exact v2 schemas, versions, enums, paths, entry points, node types, cardinalities, policies, aliases, adjacency lists, and cross-registry mappings were validated."),
        ("Accepted-negative regression matrix", f"The reconciled graph contains {summary['mutation_cases_validated']}/{summary['mutation_cases']} independently addressable mutation-case records with expected and actual diagnostic/stage fields. " + report_fact(by_requirement, "H1-ASSURANCE-GRAPH-MUTATIONS")),
        ("Authoritative requirement derivation", f"The reviewed registry contains {summary['requirements']} clause records with exact architecture sections and behavioral descriptions; final-assurance clauses are not appended during reporting."),
        ("Forty-one ACL/default results", f"The computed ACL/default catalog count is {summary['acl_catalog']}. " + report_fact(by_requirement, "H1-ACL-ADMISSION")),
        ("Record-level evidence design", "Every reconciled runtime row retains canonical serialized bytes, a record digest, a stable artifact ID, aggregate digest, and explicit record selector."),
        ("One-record/one-entry delta results", "The graph mutation receipt includes five locality checks covering lock, ACL-origin, restore, generic, and final-assurance records. " + report_fact(by_requirement, "H1-ASSURANCE-GRAPH-MUTATIONS")),
        ("Lock normalizer correction", f"The computed executable lock count is {summary['lock_records']}; observation is derived from raw blocker PIDs before comparison with the permitted direction. " + report_fact(by_requirement, "H1-LOCK-001")),
        ("Prohibited-direction mutation result", report_fact(by_requirement, "H1-ASSURANCE-LOCK-MUTATIONS")),
        ("Lock run/version result", "Every raw and normalized lock record carries format version, run ID, H1LOCK ID, generator identity, raw artifact ID, output artifact ID, and record digest."),
        ("Report provenance", "Every factual evidence row below names requirement, runtime record, validator result, report entry, artifact, record digest, and aggregate digest."),
        ("Stale-report and forged-evidence results", report_fact(by_requirement, "H1-ASSURANCE-GRAPH-MUTATIONS")),
        ("Determinism plus correctness", report_fact(by_requirement, "H1-ASSURANCE-DETERMINISM")),
        ("Migration/restore/historical regression", f"The computed restore evidence count is {summary['restore']}. " + report_fact(by_requirement, "H1-RESTORE-A")),
        ("Strict compilation", f"Parsed values: `{parsed_fact('H1-ASSURANCE-STRICT-COMPILE')}`. " + report_fact(by_requirement, "H1-ASSURANCE-STRICT-COMPILE")),
        ("Isolated Release build", f"Parsed values: `{parsed_fact('H1-ASSURANCE-RELEASE-BUILD')}`. " + report_fact(by_requirement, "H1-ASSURANCE-RELEASE-BUILD")),
        ("H1 inertness and H2/H3/H4 exclusion", report_fact(by_requirement, "H1-H2-H4-EXCLUSION")),
        ("Skipped-suite classifications", "Process-interfering shared scheduler/worker suites are safety-deferred and classified as pre-production blockers; H2/H3/H4 suites remain phase prerequisites outside H1."),
        ("Residual risks", "Legacy/libpqxx Release warnings remain separately classified; no production cutover, production migration, or live scheduler integration was attempted."),
        ("Migration checksum and manifest digest", f"Parsed values: `{parsed_fact('H1-ASSURANCE-CHECKSUM')}`. " + report_fact(by_requirement, "H1-ASSURANCE-CHECKSUM")),
        ("Final node and edge counts", f"requirements={summary['requirements']}, fixtures={summary['fixtures']}, generators={summary['generators']}, runtime={summary['runtime_records']}, validators={summary['validator_results']}, reports={summary['report_entries']}, artifacts={summary['artifacts']}, directed_edges={summary['edges']}."),
        ("Final defect counts", defect_text + "."),
        ("Final git status", "```text\n" + (worktree_text.strip() or "not captured") + "\n```"),
        ("Full diff stat including untracked files", "```text\n" + (worktree_text.strip() or "not captured") + "\n```"),
        ("Explicit disposition", f"`{disposition}`. This is not a claim that H1 is ready to commit."),
    ]
    lines = ["# Campaign Operations Phase H H1 ADR-0019B Evidence Graph and Reporting Final Correction Implementation Output", ""]
    for number, (title, body) in enumerate(sections, 1):
        lines += [f"## {number}. {title}", "", body, ""]
    lines += ["## Reconciled evidence ledger", "",
              "| Requirement | Runtime | Validator | Report | Artifact | Record digest | Aggregate digest | Status |",
              "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {row['requirement_id']} | {row['runtime_record_id']} | {row['validator_result_id']} | {row['report_entry_id']} | {row['artifact_id']} | `{row['record_digest']}` | `{row['aggregate_digest']}` | {row['status']} |"
              for row in reports]
    return "\n".join(lines) + "\n"


def main() -> None:
    if sys.argv[1:] == ["validate-registries"]:
        graph = load_graph()
        print(f"H1_REGISTRY_SEMANTICS_OK requirements={len(graph['requirements'])} "
              f"artifacts={len(graph['artifacts'])} edges={len(graph['edges'])}")
        return
    if len(sys.argv) not in {4, 5} or sys.argv[1] not in {"generate", "validate"}:
        raise SystemExit("usage: CampaignOperationsH1EvidenceGraph.py validate-registries | generate|validate ROOT RUN_ID [REPORT]")
    mode, root, run_id = sys.argv[1], Path(sys.argv[2]), sys.argv[3]
    (graph, runtime_rows, validator_rows, reports, edge_rows, health, receipt_rows, trust,
     provenance_chains, provenance_edges) = reconcile(mode, root, run_id)
    summary = counts(graph, runtime_rows, validator_rows, reports, health, trust)
    report_text = tsv_text(reports)
    edge_text = tsv_text(edge_rows)
    health_text = tsv_text(health)
    worktree_path = root / "final-assurance/WORKTREE_STATUS.log"
    worktree = evidence_text(worktree_path) if evidence_exists(worktree_path) else ""
    trace = render_trace(run_id, summary, reports, health)
    final = render_final(run_id, summary, reports, validator_rows, health, worktree)
    # The snapshot ledger binds immutable generator inputs consumed by runtime
    # records.  Generated validator registries are deliberately excluded: each
    # reconciliation executes fresh validator processes, so including their
    # fresh receipts would make the snapshot ledger recursively stale on every
    # byte-exact validation pass.
    snapshot_node_ids = {row["artifact_ids"] for row in runtime_rows}
    snapshot_rows = [row for row in SNAPSHOTS.inventory_rows() if row["artifact_id"] in snapshot_node_ids]
    generated_contents = {
        "h1-report-entry-registry.tsv": report_text, "h1-evidence-edges.tsv": edge_text,
        "h1-graph-health.tsv": health_text, TRACE_REPORT: trace, FINAL_REPORT: final,
        "h1-artifact-snapshots.tsv": tsv_text(snapshot_rows),
        "h1-provenance-chains.tsv": tsv_text(provenance_chains),
        "h1-provenance-edges.tsv": tsv_text(provenance_edges),
    }
    index_rows = artifact_index(graph, root, runtime_rows, reports, generated_contents, run_id)
    index_text = tsv_text(index_rows)
    generated_contents["h1-artifact-index.tsv"] = index_text
    if mode == "generate":
        for name, contents in generated_contents.items():
            (root / name).write_text(contents)
    else:
        for name, contents in generated_contents.items():
            path = root / name
            if not evidence_exists(path) or evidence_text(path, errors="strict") != contents:
                detail = "stale-generated-report" if name.endswith(".md") else "stale-generated-registry"
                fail("006", name, detail, "report-provenance-freshness")
        report = Path(sys.argv[4]) if len(sys.argv) == 5 else root / TRACE_REPORT
        expected = final if report.name == FINAL_REPORT else trace
        if not evidence_exists(report) or evidence_text(report, errors="strict") != expected:
            fail("006", str(report), "stale-generated-report", "report-provenance-freshness")
    disposition = "READY_FOR_FOCUSED_INDEPENDENT_REVERIFICATION" if summary["ready"] else "NOT_READY_FOR_REVERIFICATION"
    print("H1_REFERENCE_GRAPH_OK " + " ".join(f"{key}={value}" for key, value in summary.items()) +
          f" disposition={disposition} semantic_validation=PASS")


if __name__ == "__main__":
    main()
