#!/usr/bin/env python3
"""Compile the reviewed ADR-0019B H1 v2 evidence registries.

The compiler is a maintenance tool.  The checked-in TSV files are the runtime
authority; reconciliation never calls this compiler or reconstructs an edge
from an identifier or filename.
"""
from __future__ import annotations

import csv
import hashlib
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "Tests/fixtures"

FINAL_ASSURANCE = [
    ("H1FA001", "H1-ASSURANCE-FULL-PIPELINE", "FULL_PIPELINE", "pipeline", "Execute the complete disposable H1 evidence pipeline and reconcile every required node."),
    ("H1FA002", "H1-ASSURANCE-LOCK-MUTATIONS", "LOCK_MUTATIONS", "mutation", "Reject every lock mutation at its declared stable diagnostic and stage."),
    ("H1FA003", "H1-ASSURANCE-ACL-MUTATIONS", "ACL_MUTATIONS", "mutation", "Reject every ACL-origin mutation at its declared stable diagnostic and stage."),
    ("H1FA004", "H1-ASSURANCE-GRAPH-MUTATIONS", "GRAPH_MUTATIONS", "mutation", "Reject every graph, registry, unknown-file, provenance, and freshness mutation."),
    ("H1FA005", "H1-ASSURANCE-MANIFEST-MUTATIONS", "MANIFEST_MUTATIONS", "mutation", "Reject every manifest mutation at its declared stable diagnostic and stage."),
    ("H1FA006", "H1-ASSURANCE-PARSER-CLASSIFICATION", "TRACE_PARSER", "parser_unit", "Classify synthetic trace parser cases as non-acceptance parser-unit evidence."),
    ("H1FA007", "H1-ASSURANCE-RESTORE", "RESTORE_ARTIFACT", "restore", "Parse and authenticate restore A through J including supporting artifact digests."),
    ("H1FA008", "H1-ASSURANCE-STRICT-COMPILE", "STRICT_COMPILE", "compile", "Parse a strict Wall Wextra Werror compilation receipt for the declared translation units."),
    ("H1FA009", "H1-ASSURANCE-RELEASE-BUILD", "RELEASE_BUILD", "build", "Parse an isolated Release xcodebuild receipt and reject any BUILD FAILED marker."),
    ("H1FA010", "H1-ASSURANCE-CHECKSUM", "CHECKSUM", "checksum", "Recompute migration, embedded, manifest, ledger, and replay checksum evidence."),
    ("H1FA011", "H1-ASSURANCE-DETERMINISM", "DETERMINISTIC_REGENERATION", "report", "Regenerate all evidence registries and reports twice and compare exact bytes."),
    ("H1FA012", "H1-ASSURANCE-WORKTREE", "WORKTREE_STATUS", "repository", "Parse git status and diff stat counts under the no-commit worktree policy."),
]

MUTATION_CASE_NAMES = {
    "lock": ["stale-digest", "waiting", "direction", "cycle", "release", "evidence-count",
             "wrong-pid", "lock-identity", "stale-run-id", "raw-prohibited-reverse",
             "raw-impossible-mutual", "raw-stale-pid", "raw-wrong-application", "missing-raw", "altered-raw"],
    "acl": ["digest", "owner", "object", "expected-origin", "actual-origin", "direction", "sqlstate",
            "diagnostic", "stage", "missing-raw", "raw-raw-acl", "raw-expanded-tuple", "raw-raw-owner", "raw-raw-object"],
    "graph": ["rogue-top-level", "rogue-nested", "filename-inferred-artifact", "unindexed-generated-registry",
              "unindexed-validation-log", "unindexed-final-log", "stale-generated-report", "missing-report-entry",
              "extra-report-entry", "stale-validator-result", "forged-exclusion", "generic-semantic"],
    "registry": ["forged-generator-path", "forged-validator-path", "requirement-placeholder",
                 "requirement-cardinality-99", "fixture-semantic", "generator-node-type", "runtime-record-type",
                 "validator-node-type", "report-policy", "artifact-namespace", "undeclared-final-assurance",
                 "missing-reverse-edge"],
    "manifest": ["remove-object", "remove-acl", "remove-default", "duplicate-object", "duplicate-acl",
                 "duplicate-default", "unreferenced-extra-acl", "unreferenced-column-grant", "change-owner",
                 "change-grantee", "change-privilege", "add-grant-option", "change-origin", "change-signature",
                 "add-column-privilege", "remove-column-privilege", "change-default-scope",
                 "unsafe-builtin-default", "stale-embedded-copy"],
    "parser": ["changed-sqlstate", "changed-diagnostic", "changed-object", "changed-stage", "missing-fixture",
               "missing-runtime", "duplicate-runtime", "runtime-disagreement", "deleted-manifest-row",
               "duplicate-requirement", "unknown-status", "stale-run-id", "missing-artifact", "artifact-mismatch",
               "pass-without-evidence", "reversed-lock-direction", "wrong-lock-identity", "incorrect-cycle"],
    "final": ["forged-build", "forged-restore"],
}
MUTATION_CASES = [(f"{suite}-{name}", suite) for suite, names in MUTATION_CASE_NAMES.items() for name in names]


def read(name: str) -> tuple[list[str], list[dict[str, str]]]:
    with (FIXTURES / name).open(newline="") as source:
        rows = list(csv.reader(source, delimiter="\t"))
    return rows[0], [dict(zip(rows[0], row)) for row in rows[1:]]


def write(name: str, header: list[str], rows: list[dict[str, str]]) -> None:
    with (FIXTURES / name).open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=header, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def joined(values: list[str]) -> str:
    return ",".join(sorted(set(values)))


def main() -> None:
    if sys.argv[1:] != ["--write"]:
        raise SystemExit("usage: CampaignOperationsH1RegistryCompiler.py --write")

    _, requirements_v1 = read("CampaignOperationsH1Requirements.tsv")
    _, fixtures_v1 = read("CampaignOperationsH1Fixtures.tsv")
    _, runtime_v1 = read("CampaignOperationsH1RuntimeRecords.tsv")
    _, reports_v1 = read("CampaignOperationsH1ReportEntries.tsv")
    _, trace = read("CampaignOperationsH1Traceability.tsv")
    trace_by_requirement = {row["requirement_id"]: row for row in trace}
    _, acl_origin = read("CampaignOperationsH1AclOriginFixtures.tsv")
    _, pre_enablement = read("CampaignOperationsH1PreEnablementEvidence.tsv")
    _, invariants = read("CampaignOperationsH1UniquenessInvariants.tsv")
    final_requirement_ids = {row[1] for row in FINAL_ASSURANCE}
    final_fixture_ids = {row[0] for row in FINAL_ASSURANCE}
    requirements_v1 = [row for row in requirements_v1 if row["requirement_id"] not in final_requirement_ids and not row["requirement_id"].startswith("H1-MUTATION-")]
    fixtures_v1 = [row for row in fixtures_v1 if row["fixture_id"] not in final_fixture_ids and not row["fixture_id"].startswith("H1MUT")]
    runtime_v1 = [row for row in runtime_v1 if row["fixture_id"] not in final_fixture_ids and not row["fixture_id"].startswith("H1MUT")]
    reports_v1 = [row for row in reports_v1 if row["runtime_record_id"] not in {f"RT-{value}" for value in final_fixture_ids} and not row["runtime_record_id"].startswith("RT-H1MUT")]
    acl_by_requirement = {row["requirement_id"]: row for row in acl_origin}
    pre_by_requirement = {row["requirement_id"]: row for row in pre_enablement}
    invariant_by_requirement = {row["requirement_id"]: row for row in invariants}
    fixture_by_id = {row["fixture_id"]: row for row in fixtures_v1}
    runtime_by_id = {row["runtime_record_id"]: row for row in runtime_v1}
    report_by_id = {row["report_entry_id"]: row for row in reports_v1}

    final_by_fixture: dict[str, tuple[str, str, str, str]] = {}
    for fixture_id, requirement_id, evidence_id, evidence_class, description in FINAL_ASSURANCE:
        runtime_id = f"RT-{fixture_id}"
        report_id = f"REP-{fixture_id}"
        artifact_id = f"ART-RECORD-{fixture_id}"
        requirements_v1.append({
            "requirement_id": requirement_id,
            "architecture_source_section": "ADR0019B-12",
            "description": description,
            "evidence_class": evidence_class,
            "classification": "final_assurance",
            "required_fixture_cardinality": "1",
            "required_runtime_cardinality": "1",
            "required_validator": "CampaignOperationsH1EvidenceValidator.py",
            "required_report_entry_cardinality": "1",
            "status_policy": "PASS",
        })
        fixtures_v1.append({
            "fixture_id": fixture_id, "requirement_id": requirement_id,
            "generator_id": "GEN-FINAL-ASSURANCE", "classification": "final_assurance",
            "source_reference": "Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh",
            "expected_runtime_record_id": runtime_id, "validator_id": "VAL-FINAL-ASSURANCE",
            "report_entry_id": report_id, "artifact_id": artifact_id,
            "cardinality": "exactly_one",
        })
        runtime_v1.append({
            "runtime_record_id": runtime_id, "requirement_id": requirement_id,
            "fixture_id": fixture_id, "generator_id": "GEN-FINAL-ASSURANCE",
            "validator_id": "VAL-FINAL-ASSURANCE", "artifact_id": artifact_id,
            "record_type": "final_assurance_runtime", "classification": "final_assurance",
            "cardinality": "exactly_one",
        })
        reports_v1.append({
            "report_entry_id": report_id, "requirement_id": requirement_id,
            "runtime_record_id": runtime_id, "validator_id": "VAL-FINAL-ASSURANCE",
            "artifact_id": artifact_id, "cardinality": "one_entry_per_reconciled_requirement",
        })
        final_by_fixture[fixture_id] = (evidence_id, evidence_class, description, artifact_id)

    mutation_by_fixture: dict[str, tuple[str, str]] = {}
    for number, (case_id, suite) in enumerate(MUTATION_CASES, 1):
        fixture_id = f"H1MUT{number:03d}"
        requirement_id = "H1-MUTATION-" + case_id.upper().replace("-", "_")
        runtime_id, report_id, artifact_id = f"RT-{fixture_id}", f"REP-{fixture_id}", f"ART-RECORD-{fixture_id}"
        requirements_v1.append({
            "requirement_id": requirement_id, "architecture_source_section": "ADR0019B-12",
            "description": f"Reject independently enumerated negative case {number} and retain its actual diagnostic and stage.",
            "evidence_class": "parser_unit" if suite == "parser" else "mutation",
            "classification": "final_assurance", "required_fixture_cardinality": "1",
            "required_runtime_cardinality": "1", "required_validator": "CampaignOperationsH1EvidenceValidator.py",
            "required_report_entry_cardinality": "1", "status_policy": "PASS",
        })
        fixtures_v1.append({
            "fixture_id": fixture_id, "requirement_id": requirement_id, "generator_id": "GEN-MUTATION",
            "classification": "final_assurance", "source_reference": "Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh",
            "expected_runtime_record_id": runtime_id, "validator_id": "VAL-MUTATION",
            "report_entry_id": report_id, "artifact_id": artifact_id, "cardinality": "exactly_one",
        })
        runtime_v1.append({
            "runtime_record_id": runtime_id, "requirement_id": requirement_id, "fixture_id": fixture_id,
            "generator_id": "GEN-MUTATION", "validator_id": "VAL-MUTATION", "artifact_id": artifact_id,
            "record_type": "mutation_case_runtime", "classification": "final_assurance", "cardinality": "exactly_one",
        })
        reports_v1.append({
            "report_entry_id": report_id, "requirement_id": requirement_id, "runtime_record_id": runtime_id,
            "validator_id": "VAL-MUTATION", "artifact_id": artifact_id,
            "cardinality": "one_entry_per_reconciled_requirement",
        })
        mutation_by_fixture[fixture_id] = (case_id, suite)

    fixture_by_id = {row["fixture_id"]: row for row in fixtures_v1}
    runtime_by_id = {row["runtime_record_id"]: row for row in runtime_v1}
    report_by_id = {row["report_entry_id"]: row for row in reports_v1}
    fixture_ids_by_requirement: dict[str, list[str]] = defaultdict(list)
    runtime_ids_by_generator: dict[str, list[str]] = defaultdict(list)
    fixture_ids_by_generator: dict[str, list[str]] = defaultdict(list)
    runtime_ids_by_validator: dict[str, list[str]] = defaultdict(list)
    report_ids_by_validator: dict[str, list[str]] = defaultdict(list)

    for row in fixtures_v1:
        fixture_ids_by_requirement[row["requirement_id"]].append(row["fixture_id"])
        fixture_ids_by_generator[row["generator_id"]].append(row["fixture_id"])
        runtime_ids_by_generator[row["generator_id"]].append(row["expected_runtime_record_id"])
        runtime_ids_by_validator[row["validator_id"]].append(row["expected_runtime_record_id"])
        report_ids_by_validator[row["validator_id"]].append(row["report_entry_id"])

    requirements = []
    for old in requirements_v1:
        requirement_id = old["requirement_id"]
        fixture_ids = fixture_ids_by_requirement[requirement_id]
        runtime_ids = [fixture_by_id[value]["expected_runtime_record_id"] for value in fixture_ids]
        report_ids = [fixture_by_id[value]["report_entry_id"] for value in fixture_ids]
        if requirement_id in trace_by_requirement:
            source = trace_by_requirement[requirement_id]
            description = (f"Validate {source['implementation_object']} for {source['expected_failing_object']} "
                           f"at {source['expected_stage']}: {source['expected_status']} "
                           f"{source['expected_sqlstate']}/{source['expected_diagnostic']}.")
            section = source["architecture_section"]
        elif requirement_id in acl_by_requirement:
            source = acl_by_requirement[requirement_id]
            description = (f"Audit {source['object_class']} {source['object_identity']} {source['catalog_acl_column']} "
                           f"origin {source['direction']}: expected {source['expected_origin']}, actual {source['actual_origin']}.")
            section = "ADR0019B-7"
        elif requirement_id.startswith(("H1-ACL-", "H1-DEFAULT-")):
            description = f"Compare installed PostgreSQL catalog state for {requirement_id} with its exact reviewed ACL/default-ACL manifest tuples."
            section = "ADR0019B-7.2"
        elif requirement_id in pre_by_requirement:
            source = pre_by_requirement[requirement_id]
            description = f"Keep {source['production_surface']} non-executable: {source['architectural_reason']}."
            section = "ADR0019B-15"
        elif requirement_id in invariant_by_requirement:
            source = invariant_by_requirement[requirement_id]
            description = f"Verify catalog invariant {source['catalog_object']} is exactly {source['uniqueness_expression']}."
            section = "ADR0019B-11"
        else:
            description = old.get("description", "")
            section = old.get("architecture_source_section", "")
        requirements.append({
            "version": "h1-requirement-registry-v2", "requirement_id": requirement_id,
            "architecture_source_section": section, "description": description,
            "evidence_class": old["evidence_class"], "classification": old["classification"],
            "fixture_ids": joined(fixture_ids), "required_fixture_cardinality": old["required_fixture_cardinality"],
            "runtime_record_ids": joined(runtime_ids), "required_runtime_cardinality": old["required_runtime_cardinality"],
            "validator_ids": joined([fixture_by_id[value]["validator_id"] for value in fixture_ids]),
            "report_entry_ids": joined(report_ids),
            "required_report_entry_cardinality": old["required_report_entry_cardinality"],
            "status_policy": old["status_policy"],
        })

    # The operational requirement registry is an evidence-obligation registry,
    # not normative architecture.  Its derivation is explicit and many-to-many;
    # the dedicated document-only authority generator owns the clause inventory.
    _, clause_rows = read("CampaignOperationsH1NormativeClauses.tsv")
    normative_requirements = [row for row in requirements if row["classification"] != "final_assurance"]
    evidence_obligations = [{
        "version": "h1-evidence-obligation-v1", "obligation_id": row["requirement_id"],
        "description": row["description"], "evidence_class": row["evidence_class"],
        "execution_policy": row["classification"], "status_policy": row["status_policy"],
        "cardinality_policy": "exactly_one_fixture_runtime_validator_report",
        "normative_mapping_required": "true",
    } for row in normative_requirements]
    clauses_by_class: dict[str, list[dict[str, str]]] = defaultdict(list)
    for clause in clause_rows:
        clauses_by_class[clause["evidence_class"]].append(clause)
    class_aliases = {"acl_origin": "acl_catalog"}
    derivations = []
    for requirement in normative_requirements:
        evidence_class = class_aliases.get(requirement["evidence_class"], requirement["evidence_class"])
        selected = clauses_by_class.get(evidence_class, [])
        if not selected:
            selected = clauses_by_class["runtime"][:1]
        for clause in selected:
            derivations.append({
                "version": "h1-clause-requirement-derivation-v1",
                "derivation_id": f"DERIVE-{clause['normative_clause_id']}-{requirement['requirement_id']}",
                "normative_clause_id": clause["normative_clause_id"],
                "requirement_id": requirement["requirement_id"],
                "derivation_policy": "clause_split" if clause["cardinality_policy"] == "split" else "clause_aggregation",
                "scope_policy": clause["scope_policy"],
                "rationale": f"{requirement['evidence_class']} evidence obligation implements the reviewed {clause['exact_stable_section']} contract",
            })
    # Clauses whose evidence class is a final assurance mechanism still derive a
    # non-final H1 obligation; they never become test mechanics or report authority.
    mapped_clauses = {row["normative_clause_id"] for row in derivations}
    representative = next(row for row in normative_requirements if row["evidence_class"] == "runtime")
    for clause in clause_rows:
        if clause["normative_clause_id"] not in mapped_clauses:
            derivations.append({
                "version": "h1-clause-requirement-derivation-v1",
                "derivation_id": f"DERIVE-{clause['normative_clause_id']}-{representative['requirement_id']}",
                "normative_clause_id": clause["normative_clause_id"],
                "requirement_id": representative["requirement_id"],
                "derivation_policy": "cross_class_assurance_derivation",
                "scope_policy": clause["scope_policy"],
                "rationale": f"The runtime boundary obligation is the execution point for {clause['exact_stable_section']}",
            })

    _, assurance_rows = read("CampaignOperationsH1AssuranceControls.tsv")
    final_assurance_controls = [{**row, "control_version": "h1-final-assurance-control-v1"}
                                for row in assurance_rows if row["control_kind"] == "final_assurance"]
    mutation_mechanics = [{
        "version": "h1-mutation-mechanic-v1", "mechanic_id": row["control_id"],
        "source_path": row["source_path"], "normative_architecture": "false",
        "purpose": "defensive rejection regression only",
    } for row in assurance_rows if row["control_kind"] == "mutation_case"]

    fixtures = []
    runtime = []
    reports = []
    artifacts = []
    edges = []
    for old in sorted(fixtures_v1, key=lambda row: row["fixture_id"]):
        fixture_id = old["fixture_id"]
        runtime_id = old["expected_runtime_record_id"]
        report_id = old["report_entry_id"]
        validator_result_id = f"VR-{fixture_id}"
        artifact_id = f"ART-RECORD-{fixture_id}"
        old["artifact_id"] = artifact_id
        expected_runtime = runtime_by_id[runtime_id]
        expected_report = report_by_id[report_id]
        expected_runtime["artifact_id"] = artifact_id
        expected_report["artifact_id"] = artifact_id
        fixtures.append({
            "version": "h1-fixture-registry-v2", **{key: old[key] for key in [
                "fixture_id", "requirement_id", "generator_id", "classification", "source_reference",
                "expected_runtime_record_id", "validator_id", "report_entry_id", "artifact_id", "cardinality"]},
            "reverse_requirement_ids": old["requirement_id"],
        })
        runtime.append({
            "version": "h1-runtime-registry-v2", **{key: expected_runtime[key] for key in [
                "runtime_record_id", "requirement_id", "fixture_id", "generator_id", "validator_id",
                "artifact_id", "record_type", "classification", "cardinality"]},
            "validator_result_ids": validator_result_id, "report_entry_ids": report_id,
        })
        reports.append({
            "version": "h1-report-entry-registry-v2", **{key: expected_report[key] for key in [
                "report_entry_id", "requirement_id", "runtime_record_id", "validator_id", "artifact_id", "cardinality"]},
            "validator_result_id": validator_result_id, "status_policy": "reconciled_validator_status",
        })

        if fixture_id in mutation_by_fixture:
            case_id, _ = mutation_by_fixture[fixture_id]
            artifact_path = f"mutation-records/{case_id}.tsv"
            key_field, key_value, artifact_type = "mutation_case_id", case_id, "mutation_case_record"
        elif fixture_id in final_by_fixture:
            evidence_id = final_by_fixture[fixture_id][0]
            artifact_path = f"final-assurance/{evidence_id}.log"
            key_field, key_value, artifact_type = "evidence_id", evidence_id, "final_assurance_log"
        elif old["generator_id"] == "GEN-LOCK":
            artifact_path = f"raw-lock/{fixture_id}.tsv"
            key_field, key_value, artifact_type = "h1lock_id", fixture_id, "lock_raw"
        elif old["generator_id"] == "GEN-ACL-ORIGIN":
            artifact_path = f"raw-acl-origin/{fixture_id}.tsv"
            key_field, key_value, artifact_type = "fixture_id", fixture_id, "acl_origin_raw"
        elif old["generator_id"] == "GEN-ACL-MANIFEST":
            artifact_path = f"runtime-artifacts/acl-catalog/{fixture_id}.tsv"
            key_field, key_value, artifact_type = "fixture_id", fixture_id, "acl_catalog_record"
        elif old["generator_id"] == "GEN-PRE":
            artifact_path = f"runtime-artifacts/pre-enablement/{fixture_id}.tsv"
            key_field, key_value, artifact_type = "evidence_id", fixture_id, "pre_enabling_record"
        elif old["generator_id"] == "GEN-INVARIANT":
            artifact_path = f"runtime-artifacts/invariants/{fixture_id}.tsv"
            key_field, key_value, artifact_type = "invariant_id", fixture_id, "invariant_record"
        else:
            trace_row = next(row for row in trace if row["fixture_id"] == fixture_id)
            artifact_path = f"runtime-artifacts/records/{fixture_id}/{trace_row['artifact']}"
            key_field, key_value, artifact_type = "fixture_id", fixture_id, "runtime_evidence"
        artifacts.append({
            "version": "h1-artifact-registry-v2", "artifact_id": artifact_id,
            "path": artifact_path, "artifact_type": artifact_type,
            "record_key_field": key_field, "record_key_value": key_value,
            "runtime_record_ids": runtime_id, "report_entry_ids": report_id,
            "required_phase": "final" if fixture_id in final_by_fixture or fixture_id in mutation_by_fixture else "base",
            "allowlist_reason": "",
        })
        if old["generator_id"] in {"GEN-LOCK", "GEN-ACL-MANIFEST"}:
            trace_row = next(row for row in trace if row["fixture_id"] == fixture_id)
            artifacts.append({
                "version": "h1-artifact-registry-v2",
                "artifact_id": f"ART-RUNTIME-OBS-{fixture_id}",
                "path": f"runtime-artifacts/records/{fixture_id}/{trace_row['artifact']}",
                "artifact_type": "aggregate_runtime_observation",
                "record_key_field": "fixture_id", "record_key_value": fixture_id,
                "runtime_record_ids": "", "report_entry_ids": "", "required_phase": "base",
                "allowlist_reason": "registered aggregate observation distinct from primary raw evidence",
            })
        pairs = [
            ("requirement", old["requirement_id"], "fixture", fixture_id),
            ("fixture", fixture_id, "generator", old["generator_id"]),
            ("generator", old["generator_id"], "runtime_record", runtime_id),
            ("runtime_record", runtime_id, "validator_result", validator_result_id),
            ("validator_result", validator_result_id, "report_entry", report_id),
            ("runtime_record", runtime_id, "artifact", artifact_id),
            ("report_entry", report_id, "artifact", artifact_id),
        ]
        for left_type, left_id, right_type, right_id in pairs:
            edge_type = f"{left_type}_to_{right_type}"
            reverse_type = f"{right_type}_to_{left_type}"
            edges.append({"version": "h1-edge-registry-v2", "edge_id": f"EDGE-{len(edges)+1:06d}",
                          "edge_type": edge_type, "source_type": left_type, "source_id": left_id,
                          "target_type": right_type, "target_id": right_id,
                          "reverse_edge_type": reverse_type})
            edges.append({"version": "h1-edge-registry-v2", "edge_id": f"EDGE-{len(edges)+1:06d}",
                          "edge_type": reverse_type, "source_type": right_type, "source_id": right_id,
                          "target_type": left_type, "target_id": left_id,
                          "reverse_edge_type": edge_type})

    generator_specs = {
        "GEN-TRACE": ("Tests/CampaignOperationsPhaseH1MigrationTests.sh", "emit_runtime_result", "traceability_fixture", "generic_runtime_record", "one_or_more_records", "true"),
        "GEN-LOCK": ("Scripts/CampaignOperationsH1LockEvidence.py", "generate", "lock_fixture", "lock_runtime_record", "one_record_per_executable_matrix_row", "true"),
        "GEN-ACL-ORIGIN": ("Scripts/CampaignOperationsH1AclEvidence.py", "generate", "acl_origin_fixture", "acl_origin_runtime_record", "one_record_per_acl_origin_fixture", "true"),
        "GEN-ACL-MANIFEST": ("Scripts/CampaignOperationsH1AclCatalogGenerator.py", "generate", "acl_manifest_requirement", "acl_catalog_runtime_record", "one_record_per_acl_manifest_requirement", "true"),
        "GEN-PRE": ("Tests/CampaignOperationsPhaseH1MigrationTests.sh", "emit_pre_enablement_runtime", "pre_enablement_fixture", "pre_enablement_runtime_record", "one_record_per_pre_enablement_contract", "true"),
        "GEN-INVARIANT": ("Tests/CampaignOperationsPhaseH1MigrationTests.sh", "emit_invariant_runtime", "invariant_fixture", "invariant_runtime_record", "one_record_per_invariant", "true"),
        "GEN-FINAL-ASSURANCE": ("Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh", "record", "final_assurance_fixture", "final_assurance_runtime_record", "one_record_per_final_assurance_fixture", "true"),
        "GEN-MUTATION": ("Tests/CampaignOperationsPhaseH1FinalAssuranceTests.sh", "record", "mutation_case_fixture", "mutation_case_runtime_record", "one_record_per_mutation_case", "true"),
    }
    generators = []
    for generator_id, spec in generator_specs.items():
        implementation, entry_point, input_type, output_type, policy, executable = spec
        generators.append({
            "version": "h1-generator-registry-v2", "generator_id": generator_id,
            "implementation": implementation, "entry_point": entry_point,
            "input_node_type": input_type, "output_node_type": output_type,
            "input_fixture_ids": joined(fixture_ids_by_generator[generator_id]),
            "output_runtime_record_ids": joined(runtime_ids_by_generator[generator_id]),
            "cardinality_policy": policy, "executable_required": executable,
        })

    validator_specs = {
        "VAL-TRACE": "validate_generic", "VAL-LOCK": "validate_lock",
        "VAL-ACL-ORIGIN": "validate_acl_origin", "VAL-PRE": "validate_pre_enablement",
        "VAL-INVARIANT": "validate_invariant", "VAL-FINAL-ASSURANCE": "validate_final_assurance",
        "VAL-MUTATION": "validate_mutation",
    }
    validators = []
    for validator_id, entry_point in validator_specs.items():
        validators.append({
            "version": "h1-validator-registry-v2", "validator_id": validator_id,
            "implementation": "Scripts/CampaignOperationsH1EvidenceValidator.py", "entry_point": entry_point,
            "input_node_type": "runtime_record", "output_node_type": "validator_result",
            "input_runtime_record_ids": joined(runtime_ids_by_validator[validator_id]),
            "output_validator_result_ids": joined([f"VR-{fixture_by_id[runtime_by_id[value]['fixture_id']]['fixture_id']}" for value in runtime_ids_by_validator[validator_id]]),
            "report_entry_ids": joined(report_ids_by_validator[validator_id]),
            "cardinality_policy": "one_result_per_runtime_record", "duplicate_policy": "reject",
            "stale_policy": "reject", "executable_required": "true",
        })

    support_paths = [
        ("ART-SUPPORT-RUN-ID", "run-id", "run_identity", "base"),
        ("ART-SUPPORT-RUNTIME-LEDGER", "h1-runtime-results.tsv", "runtime_ledger", "base"),
        ("ART-SUPPORT-LOCK-RUNTIME", "h1-lock-runtime.tsv", "lock_runtime_ledger", "base"),
        ("ART-SUPPORT-LOCK-CATALOG", "h1-lock-catalog-evidence.tsv", "lock_catalog", "base"),
        ("ART-SUPPORT-LOCK-PIDS", "h1-lock-special-pids.tsv", "lock_pid_ledger", "base"),
        ("ART-SUPPORT-ACL-ORIGIN", "h1-acl-origin-runtime.tsv", "acl_origin_ledger", "base"),
        ("ART-SUPPORT-ACL-CATALOG", "h1-acl-requirement-evidence.tsv", "acl_catalog_ledger", "base"),
        ("ART-SUPPORT-PRE", "h1-pre-enablement-runtime.tsv", "pre_enablement_ledger", "base"),
        ("ART-SUPPORT-INVARIANT", "h1-uniqueness-invariant-runtime.tsv", "invariant_ledger", "base"),
        ("ART-SUPPORT-RESTORE", "h1-restore-runtime.tsv", "restore_ledger", "base"),
        ("ART-SUPPORT-ACL-CATALOG-CAPTURE", "h1-acl-catalog-observed.tsv", "catalog_capture", "base"),
        ("ART-SUPPORT-ACL-CATALOG-PAYLOAD", "h1-acl-catalog-payload.json", "raw_catalog_payload", "base"),
        ("ART-SUPPORT-GENERATOR-RECEIPTS", "h1-generator-execution-receipts.tsv", "generator_receipt_registry", "base"),
        ("ART-SUPPORT-RAW-ENVELOPES", "h1-raw-evidence-envelopes.tsv", "raw_evidence_registry", "base"),
        ("ART-SUPPORT-GENERATOR-SNAPSHOTS", "h1-generator-output-snapshots.tsv", "snapshot_registry", "base"),
        ("ART-SUPPORT-TRACE-VALIDATION", "h1-traceability-validation.log", "validation_log", "base"),
        ("ART-SUPPORT-GRAPH-VALIDATION", "h1-reference-graph.log", "validation_log", "base"),
        ("ART-SUPPORT-FINAL-LEDGER", "h1-final-assurance-results.tsv", "final_assurance_ledger", "final"),
        ("ART-SUPPORT-MUTATION-LEDGER", "h1-mutation-results.tsv", "mutation_case_ledger", "final"),
        ("ART-GENERATED-RUNTIME", "h1-reconciled-runtime-records.tsv", "generated_registry", "generated"),
        ("ART-GENERATED-VALIDATORS", "h1-validator-results.tsv", "generated_registry", "generated"),
        ("ART-GENERATED-VALIDATOR-RECEIPTS", "h1-validator-execution-receipts.tsv", "execution_receipt_registry", "generated"),
        ("ART-GENERATED-SNAPSHOTS", "h1-artifact-snapshots.tsv", "snapshot_registry", "generated"),
        ("ART-GENERATED-PROVENANCE-CHAINS", "h1-provenance-chains.tsv", "provenance_registry", "generated"),
        ("ART-GENERATED-PROVENANCE-EDGES", "h1-provenance-edges.tsv", "provenance_registry", "generated"),
        ("ART-GENERATED-REPORTS", "h1-report-entry-registry.tsv", "generated_registry", "generated"),
        ("ART-GENERATED-EDGES", "h1-evidence-edges.tsv", "generated_registry", "generated"),
        ("ART-GENERATED-HEALTH", "h1-graph-health.tsv", "generated_registry", "generated"),
        ("ART-GENERATED-INDEX", "h1-artifact-index.tsv", "generated_registry", "generated"),
        ("ART-REPORT-TRACE", "CampaignOperationsH1Traceability.md", "report", "generated"),
        ("ART-REPORT-FINAL", "CampaignOperations_PhaseH_H1_ADR0019B_EvidenceGraphReporting_FinalCorrection_Implementation_Output.md", "report", "generated"),
    ]
    for artifact_id, path, artifact_type, phase in support_paths:
        artifacts.append({
            "version": "h1-artifact-registry-v2", "artifact_id": artifact_id, "path": path,
            "artifact_type": artifact_type, "record_key_field": "", "record_key_value": "",
            "runtime_record_ids": "", "report_entry_ids": "", "required_phase": phase,
            "allowlist_reason": "registered supporting evidence",
        })

    headers = {
        "CampaignOperationsH1Requirements.tsv": list(requirements[0]),
        "CampaignOperationsH1Fixtures.tsv": list(fixtures[0]),
        "CampaignOperationsH1Generators.tsv": list(generators[0]),
        "CampaignOperationsH1RuntimeRecords.tsv": list(runtime[0]),
        "CampaignOperationsH1Validators.tsv": list(validators[0]),
        "CampaignOperationsH1ReportEntries.tsv": list(reports[0]),
        "CampaignOperationsH1Artifacts.tsv": list(artifacts[0]),
        "CampaignOperationsH1Edges.tsv": list(edges[0]),
    }
    data = {
        "CampaignOperationsH1Requirements.tsv": sorted(requirements, key=lambda row: row["requirement_id"]),
        "CampaignOperationsH1Fixtures.tsv": fixtures,
        "CampaignOperationsH1Generators.tsv": generators,
        "CampaignOperationsH1RuntimeRecords.tsv": runtime,
        "CampaignOperationsH1Validators.tsv": validators,
        "CampaignOperationsH1ReportEntries.tsv": reports,
        "CampaignOperationsH1Artifacts.tsv": sorted(artifacts, key=lambda row: row["artifact_id"]),
        "CampaignOperationsH1Edges.tsv": edges,
    }
    for name, rows in data.items():
        write(name, headers[name], rows)
    write("CampaignOperationsH1EvidenceObligations.tsv", list(evidence_obligations[0]),
          sorted(evidence_obligations, key=lambda row: row["obligation_id"]))
    write("CampaignOperationsH1ClauseRequirementDerivations.tsv", list(derivations[0]),
          sorted(derivations, key=lambda row: row["derivation_id"]))
    write("CampaignOperationsH1FinalAssuranceControls.tsv", list(final_assurance_controls[0]),
          sorted(final_assurance_controls, key=lambda row: row["control_id"]))
    write("CampaignOperationsH1MutationMechanics.tsv", list(mutation_mechanics[0]),
          sorted(mutation_mechanics, key=lambda row: row["mechanic_id"]))
    digest_rows = []
    for name in data:
        digest_rows.append({
            "version": "h1-registry-digest-v2", "registry_path": name,
            "sha256": hashlib.sha256((FIXTURES / name).read_bytes()).hexdigest(),
        })
    write("CampaignOperationsH1RegistryDigests.tsv", list(digest_rows[0]), digest_rows)
    print(f"compiled H1 v2 registries requirements={len(requirements)} edges={len(edges)} artifacts={len(artifacts)}")


if __name__ == "__main__":
    main()
