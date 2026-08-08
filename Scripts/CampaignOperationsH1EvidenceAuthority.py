#!/usr/bin/env python3
"""Independent ADR-0019B H1 evidence authority and representation checks.

This module intentionally does not derive expected requirements from runtime
evidence, graph edges, reports, generators, validators, or mutation output.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import stat
import sys
import unicodedata
from pathlib import Path, PurePosixPath

from CampaignOperationsH1ArtifactSnapshot import ArtifactSnapshot, ArtifactSnapshotSet, SnapshotError, capture_regular_file

csv.field_size_limit(sys.maxsize)

ROOT = Path(__file__).resolve().parents[1]
FROZEN_ROOT = ROOT / "Tests/fixtures"
AUTHORITY_VERSION = "h1-normative-authority-v2"
AUTHORITY_DIGEST = "34be85602ebf1c492530ebf17955aaaa2e407c1ee8d7cbc569627f6d8934df4f"
CONTROL_DIGEST = "0107496646e3473d24238dba27b65de2333bcf265b1528586548eee682261b99"
POLICY_DIGESTS = {
    "CampaignOperationsH1EvidenceObligations.tsv": "e2ad39bfecfec92a0297bedb505f1ff5dac1714be5c18acd80e6136717d224a2",
    "CampaignOperationsH1ClauseRequirementDerivations.tsv": "46355637cac0b024c8711d90dc6ce7462ff9df6d9578d26aaf290da4f2d598f0",
    "CampaignOperationsH1FinalAssuranceControls.tsv": "c44d8f7805682af844ec92f0a934ec76cabff861d8dca42b8d3fcc61284abaec",
    "CampaignOperationsH1MutationMechanics.tsv": "1a38d77cd18929b55a01b4edfefd9dd51a6a7d7e06e19e7089f86bd7cf8337f3",
    "CampaignOperationsH1ExpectedValueAuthorities.tsv": "1fa523696ad1bfa384bac18cea72ed2451fa7518500a711f4d5629420fcede88",
    "CampaignOperationsH1ObservedValueProvenance.tsv": "5c25e71c084377a2158c734cf106ba0bd8de1c46ae76fe8f469104268860b5fb",
    "CampaignOperationsH1RawEvidenceContracts.tsv": "16b4e64594ac94a54f0ad46a37412ebf7b6683d58b003727eecc9fe0b7a48128",
    "CampaignOperationsH1DirectoryPolicy.tsv": "257f0b40257c3feb1d115888a34e7c2b8b81426d0df37691ac94e31e0250f03e",
    "CampaignOperationsH1ReviewDisposition.tsv": "466012f1bccbac6384249fa594b6510096cd70ed7f03283ba7afc1d740b1394f",
}


def fail(code: str, key: str, detail: str, stage: str) -> None:
    print(f"H1A{code} key={key} stage={stage} detail={detail}", file=sys.stderr)
    raise SystemExit(1)


def sha(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()
    except OSError:
        fail("001", str(path), "missing-authority-input", "authority-integrity")


def read_tsv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open(newline="") as source:
            data = list(csv.reader(source, delimiter="\t"))
    except OSError:
        fail("001", str(path), "missing-input", "authority-integrity")
    if not data or len(set(data[0])) != len(data[0]):
        fail("002", path.name, "invalid-header", "authority-integrity")
    rows: list[dict[str, str]] = []
    for number, values in enumerate(data[1:], 2):
        if len(values) != len(data[0]):
            fail("002", f"{path.name}:{number}", "field-count", "authority-integrity")
        rows.append(dict(zip(data[0], values)))
    return data[0], rows


def keyed(path: Path, key: str, stage: str) -> dict[str, dict[str, str]]:
    _, rows = read_tsv(path)
    result: dict[str, dict[str, str]] = {}
    for row in rows:
        value = row.get(key, "")
        if not value or value in result:
            fail("003", value or path.name, "duplicate-or-empty-key", stage)
        result[value] = row
    return result


def _source_excerpt_digest(row: dict[str, str]) -> str:
    match = re.fullmatch(r"lines ([1-9][0-9]*)-([1-9][0-9]*)", row["exact_source_anchor"])
    if not match:
        fail("104", row["normative_clause_id"], "invalid-source-anchor", "normative-authority-validation")
    first, last = map(int, match.groups())
    path = ROOT / row["source_document"]
    try:
        lines = path.read_text().splitlines(keepends=True)
    except OSError:
        fail("104", row["normative_clause_id"], "missing-source-document", "normative-authority-validation")
    if first > last or last > len(lines):
        fail("104", row["normative_clause_id"], "source-anchor-out-of-range", "normative-authority-validation")
    raw = "".join(lines[first - 1:last])
    excerpt = re.sub(r"\s+", " ", raw).strip()
    if excerpt != row.get("canonical_excerpt"):
        fail("104", row["normative_clause_id"], "altered-canonical-excerpt", "normative-authority-validation")
    return hashlib.sha256(raw.encode()).hexdigest()


def validate_normative_authority(registry_root: Path, authority_root: Path | None = None) -> dict[str, dict[str, str]]:
    frozen = authority_root is None
    authority_root = authority_root or FROZEN_ROOT
    authority_path = authority_root / "CampaignOperationsH1NormativeClauses.tsv"
    control_path = authority_root / "CampaignOperationsH1AssuranceControls.tsv"
    if frozen:
        if sha(authority_path) != AUTHORITY_DIGEST:
            fail("101", authority_path.name, "frozen-authority-digest-mismatch", "normative-authority-validation")
        if sha(control_path) != CONTROL_DIGEST:
            fail("101", control_path.name, "frozen-control-digest-mismatch", "normative-authority-validation")
        for name, expected_digest in POLICY_DIGESTS.items():
            if sha(authority_root / name) != expected_digest:
                fail("101", name, "frozen-policy-digest-mismatch", "normative-authority-validation")
    clauses = keyed(authority_path, "normative_clause_id", "normative-authority-validation")
    expected_header = ["authority_version", "normative_clause_id", "source_document", "exact_stable_section",
                       "exact_source_anchor", "canonical_excerpt", "canonical_clause_digest",
                       "normalized_normative_requirement", "derived_h1_requirement_identity", "evidence_class",
                       "executable_policy", "status_policy", "cardinality_policy", "precedence_rank", "scope_policy"]
    header, _ = read_tsv(authority_path)
    if header != expected_header:
        fail("102", authority_path.name, "wrong-normative-authority-schema", "normative-authority-validation")
    source_contract = {
        "docs/architecture/adr/ADR-0019-campaign-operations-production-dispatch-admission-and-manager.md": (1, "ADR-0019 §"),
        "docs/architecture/adr/ADR-0019A-h1-owner-safe-transaction-authorization.md": (2, "ADR-0019A §"),
        "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md": (3, "ADR-0019B §"),
        "docs/architecture/CampaignOperations_PhaseH_Production_Dispatch_Admission_and_Manager.md": (4, "Corrected Phase H §"),
        "docs/architecture/Volume_XII_Database.md": (5, "Volume XII §"),
    }
    # The reviewed v2 inventory is the sole authority.  Do not import the
    # generator's in-memory clause list or reconstruct authority from any
    # subordinate registry.
    governing_ids = set(keyed(FROZEN_ROOT / "CampaignOperationsH1NormativeClauses.tsv",
                              "normative_clause_id", "governing-clause-reconciliation"))
    missing_inventory = sorted(governing_ids - set(clauses))
    if missing_inventory:
        fail("109", missing_inventory[0], "governing-clause-without-authority-row", "governing-clause-reconciliation")
    extra_inventory = sorted(set(clauses) - governing_ids)
    if extra_inventory:
        fail("110", extra_inventory[0], "authority-row-without-governing-clause", "governing-clause-reconciliation")
    architecture_ids: set[str] = set()
    for clause_id, row in clauses.items():
        if row.get("authority_version") != AUTHORITY_VERSION:
            fail("102", clause_id, "wrong-authority-version", "normative-authority-validation")
        contract = source_contract.get(row["source_document"])
        if contract is None or row["precedence_rank"] != str(contract[0]) or not row["exact_stable_section"].startswith(contract[1]):
            fail("104", clause_id, "wrong-governing-document-or-precedence", "normative-authority-validation")
        match = re.fullmatch(r"lines ([0-9]+)-([0-9]+)", row["exact_source_anchor"])
        if match is None or int(match.group(2)) - int(match.group(1)) + 1 > 32:
            fail("104", clause_id, "broad-or-invalid-source-anchor", "normative-authority-validation")
        if _source_excerpt_digest(row) != row["canonical_clause_digest"]:
            fail("104", clause_id, "altered-normative-text", "normative-authority-validation")
        architecture_id = row["derived_h1_requirement_identity"]
        if not architecture_id or architecture_id in architecture_ids:
            fail("103", architecture_id or clause_id, "duplicate-architectural-requirement", "normative-authority-validation")
        architecture_ids.add(architecture_id)

    requirements = keyed(registry_root / "CampaignOperationsH1Requirements.tsv", "requirement_id",
                         "normative-requirement-reconciliation")
    controls = keyed(control_path, "control_id", "normative-requirement-reconciliation")
    obligations = keyed(authority_root / "CampaignOperationsH1EvidenceObligations.tsv", "obligation_id",
                        "normative-requirement-reconciliation")
    derivations = keyed(authority_root / "CampaignOperationsH1ClauseRequirementDerivations.tsv", "derivation_id",
                        "normative-requirement-reconciliation")
    final_controls = keyed(authority_root / "CampaignOperationsH1FinalAssuranceControls.tsv", "control_id",
                           "normative-requirement-reconciliation")
    mutation_mechanics = keyed(authority_root / "CampaignOperationsH1MutationMechanics.tsv", "mechanic_id",
                               "normative-requirement-reconciliation")
    normative_rows = {key: value for key, value in requirements.items()
                      if value.get("classification") != "final_assurance"}
    control_rows = {key: value for key, value in requirements.items()
                    if value.get("classification") == "final_assurance"}
    mapped_clauses = {row.get("normative_clause_id", "") for row in derivations.values()}
    mapped_requirements = {row.get("requirement_id", "") for row in derivations.values()}
    missing_clauses = sorted(set(clauses) - mapped_clauses)
    if missing_clauses:
        fail("105", missing_clauses[0], "normative-clause-without-derived-requirement", "normative-requirement-reconciliation")
    extra_clauses = sorted(mapped_clauses - set(clauses))
    if extra_clauses:
        fail("106", extra_clauses[0], "derivation-without-normative-clause", "normative-requirement-reconciliation")
    missing = sorted(set(obligations) - set(normative_rows))
    if missing:
        fail("105", missing[0], "missing-normative-requirement", "normative-requirement-reconciliation")
    extra = sorted(set(normative_rows) - set(obligations))
    if extra:
        fail("106", extra[0], "non-normative-requirement", "normative-requirement-reconciliation")
    if mapped_requirements != set(obligations):
        offender = sorted(mapped_requirements ^ set(obligations))[0]
        detail = "requirement-without-normative-derivation" if offender in obligations else "derivation-without-evidence-obligation"
        fail("106", offender, detail, "normative-requirement-reconciliation")
    unexpected_controls = sorted(set(control_rows) - set(controls))
    if unexpected_controls:
        fail("106", unexpected_controls[0], "non-normative-requirement", "normative-requirement-reconciliation")
    missing_controls = sorted(set(controls) - set(control_rows))
    if missing_controls:
        fail("107", missing_controls[0], "missing-assurance-control", "normative-requirement-reconciliation")
    if set(final_controls) != {key for key, row in controls.items() if row["control_kind"] == "final_assurance"}:
        fail("107", "final-assurance-controls", "final-assurance-control-separation-mismatch", "normative-requirement-reconciliation")
    if set(mutation_mechanics) != {key for key, row in controls.items() if row["control_kind"] == "mutation_case"}:
        fail("107", "mutation-mechanics", "mutation-mechanic-separation-mismatch", "normative-requirement-reconciliation")
    if (set(controls) | set(mutation_mechanics)).intersection(architecture_ids | mapped_requirements):
        offender = sorted((set(controls) | set(mutation_mechanics)).intersection(architecture_ids | mapped_requirements))[0]
        fail("106", offender, "test-mechanic-labeled-normative", "normative-requirement-reconciliation")
    for requirement_id, contract in obligations.items():
        row = normative_rows[requirement_id]
        fields = {
            "description": contract["description"], "evidence_class": contract["evidence_class"],
            "classification": contract["execution_policy"], "status_policy": contract["status_policy"],
        }
        for field, expected in fields.items():
            if row.get(field) != expected:
                fail("108", requirement_id, f"authority-mismatch:{field}", "normative-requirement-reconciliation")
        if any(row.get(field) != "1" for field in
               ("required_fixture_cardinality", "required_runtime_cardinality", "required_report_entry_cardinality")):
            fail("108", requirement_id, "authority-mismatch:cardinality", "normative-requirement-reconciliation")
    return clauses


def validate_exact_runtime_inventory(root: Path, graph: dict[str, dict[str, dict[str, str]]],
                                     require_final: bool = True,
                                     snapshots: ArtifactSnapshotSet | None = None) -> None:
    fixtures = graph["fixtures"]
    artifacts = graph["artifacts"]
    def keys_for(generator_id: str) -> set[str]:
        return {artifacts[row["artifact_id"]]["record_key_value"] for row in fixtures.values()
                if row["generator_id"] == generator_id}
    source_specs = {
        "h1-runtime-results.tsv": ("fixture_id", {k for k, v in fixtures.items()
                                                     if v["generator_id"] in {"GEN-TRACE", "GEN-LOCK", "GEN-ACL-MANIFEST"}}),
        "h1-lock-runtime.tsv": ("h1lock_id", {k for k, v in fixtures.items() if v["generator_id"] == "GEN-LOCK"}),
        "h1-acl-origin-runtime.tsv": ("fixture_id", {k for k, v in fixtures.items() if v["generator_id"] == "GEN-ACL-ORIGIN"}),
        "h1-acl-requirement-evidence.tsv": ("fixture_id", {k for k, v in fixtures.items() if v["generator_id"] == "GEN-ACL-MANIFEST"}),
        "h1-pre-enablement-runtime.tsv": ("evidence_id", {k for k, v in fixtures.items() if v["generator_id"] == "GEN-PRE"}),
        "h1-uniqueness-invariant-runtime.tsv": ("invariant_id", {k for k, v in fixtures.items() if v["generator_id"] == "GEN-INVARIANT"}),
        "h1-final-assurance-results.tsv": ("evidence_id", keys_for("GEN-FINAL-ASSURANCE")),
        "h1-mutation-results.tsv": ("mutation_case_id", keys_for("GEN-MUTATION")),
        "h1-restore-runtime.tsv": ("scenario_id", {key.removeprefix("H1RESTORE") for key in fixtures
                                                       if re.fullmatch(r"H1RESTORE[A-J]", key)}),
    }
    for name, (key, expected) in source_specs.items():
        path = root / name
        final_source = name in {"h1-final-assurance-results.tsv", "h1-mutation-results.tsv"}
        if snapshots is not None:
            exists = name in snapshots
        else:
            exists = path.is_file()
        if not exists:
            if expected and (require_final or not final_source):
                fail("201", name, "missing-runtime-source", "exact-runtime-inventory")
            continue
        _, rows = snapshots.by_path(name).tsv() if snapshots is not None else read_tsv(path)
        values = [row.get(key, "") for row in rows]
        duplicates = sorted(value for value in set(values) if values.count(value) > 1)
        if duplicates:
            fail("202", duplicates[0], f"duplicate-runtime-row:{name}", "exact-runtime-inventory")
        unexpected = sorted(set(values) - expected)
        if unexpected:
            fail("203", unexpected[0] or name, f"unexpected-runtime-row:{name}", "exact-runtime-inventory")
        missing = sorted(expected - set(values)) if (require_final or not final_source) else []
        if missing:
            fail("204", missing[0], f"missing-runtime-row:{name}", "exact-runtime-inventory")
        for row in rows:
            value = row.get(key, "")
            fixture_id = (value if value in fixtures else
                          next((candidate for candidate, fixture in fixtures.items()
                                if artifacts[fixture["artifact_id"]]["record_key_value"] == value), ""))
            if not fixture_id and name == "h1-restore-runtime.tsv":
                fixture_id = "H1RESTORE" + value
            fixture = fixtures.get(fixture_id)
            if fixture is None:
                continue
            expected_fields = {
                "requirement_id": fixture["requirement_id"],
                "generator_id": fixture["generator_id"],
                "emitted_runtime_record_id": fixture["expected_runtime_record_id"],
                "runtime_record_id": fixture["expected_runtime_record_id"],
                "validator_result_id": "VR-" + fixture_id,
                "report_entry_id": fixture["report_entry_id"],
                "artifact_id": fixture["artifact_id"],
                "output_artifact_id": (f"ART-RUNTIME-OBS-{fixture_id}"
                                       if name == "h1-runtime-results.tsv" and fixture["generator_id"] != "GEN-TRACE"
                                       else fixture["artifact_id"]),
            }
            for field, expected_value in expected_fields.items():
                if field in row and row[field] != expected_value:
                    fail("205", value, f"wrong-runtime-reference:{field}", "exact-runtime-inventory")


def _canonical_relative(value: str, artifact_id: str) -> PurePosixPath:
    if (not value or value.startswith("/") or "\\" in value or "//" in value or
            unicodedata.normalize("NFC", value) != value):
        fail("301", artifact_id, "unsupported-path-normalization", "filesystem-representation-validation")
    pure = PurePosixPath(value)
    if str(pure) != value or any(part in {"", ".", ".."} for part in pure.parts):
        fail("301", artifact_id, "non-canonical-relative-path", "filesystem-representation-validation")
    if any(part.endswith((" ", ".")) for part in pure.parts):
        fail("301", artifact_id, "trailing-space-or-dot-alias", "filesystem-representation-validation")
    return pure


def classify_mode(mode: int) -> str:
    if stat.S_ISREG(mode): return "regular"
    if stat.S_ISDIR(mode): return "directory"
    if stat.S_ISLNK(mode): return "symbolic_link"
    return "unsupported"


def validate_filesystem(root: Path, artifacts: dict[str, dict[str, str]], generation: bool = False,
                        run_id: str = "representation-only",
                        before_read_hook=None) -> ArtifactSnapshotSet:
    policy_rows = keyed(FROZEN_ROOT / "CampaignOperationsH1DirectoryPolicy.tsv", "relative_path",
                        "directory-policy-validation")
    opaque = {key for key, row in policy_rows.items() if row["opaque"] == "true"}
    try:
        return ArtifactSnapshotSet(root, artifacts, run_id, generation, opaque, before_read_hook)
    except SnapshotError as error:
        fail(error.code, error.key, error.detail, error.stage)


RAW_ENVELOPE_FIELDS = [
    "evidence_version", "evidence_class", "requirement_id", "run_id", "implementation_version",
    "invocation_contract", "tool_identity", "tool_digest", "process_id", "monotonic_start_ns",
    "monotonic_completion_ns", "actual_exit_status", "input_snapshot_ids", "input_digests",
    "output_artifact_ids", "output_digests", "stdout", "stderr", "payload_json",
    "generator_execution_id",
]
RAW_CLASS_PAYLOADS = {
    "runtime": {"query_contract", "sqlstate", "diagnostic", "object_identity", "transaction_outcome", "observation_rows", "query_receipt"},
    "role_security": {"query_contract", "role_identity", "sqlstate", "diagnostic", "catalog_rows", "query_receipt"},
    "lock": {"pg_locks", "pg_blocking_pids", "pg_stat_activity", "backend_identities", "transaction_outcomes", "query_receipt"},
    "acl_origin": {"query_id", "catalog_rows", "owner", "raw_acl", "origin", "acldefault_source", "expanded_acl_rows", "deployment_audit"},
    "acl_catalog": {"query_id", "catalog_rows", "owner", "raw_acl", "origin", "default_acl_source", "expanded_acl_rows"},
    "restore": {"invocation", "source_cluster_id", "target_cluster_id", "dump_digest", "pre_audit", "post_audit", "historical_bytes"},
    "release_build": {"xcodebuild_identity", "command", "configuration", "product_identity", "derived_data_identity", "complete_log"},
    "strict_compile": {"compiler_identity", "command", "translation_units", "raw_diagnostics"},
    "checksum": {"migration_bytes_digest", "embedded_checksum_source", "hash_implementation", "ledger_output", "replay_output"},
    "worktree": {"git_identity", "command", "repository_identity", "status_output", "diff_output"},
    "mutation": {"command", "fixture_identity", "expected_failure", "actual_failure", "output_log"},
    "exclusion": {"scanner_identity", "command", "scanned_source_paths", "patterns", "result_rows"},
    "report_freshness": {"report_id", "input_artifact_ids", "input_digests", "output_digest"},
    "full_pipeline": {"command", "input_snapshot_ids", "validator_execution_ids", "graph_digest", "report_digest"},
}


def validate_raw_evidence(evidence_class: str, evidence: ArtifactSnapshot | Path,
                          expected_requirement_id: str | None = None,
                          expected_run_id: str | None = None,
                          attested_generator_receipts: dict[str, dict[str, str]] | None = None,
                          expected_semantics: dict[str, object] | None = None) -> dict[str, object]:
    if evidence_class not in RAW_CLASS_PAYLOADS:
        fail("401", evidence_class, "unknown-evidence-class", "raw-evidence-authenticity")
    if isinstance(evidence, Path):
        if not evidence.exists():
            fail("405", evidence_class, f"missing-raw-evidence:{evidence.name}", "raw-evidence-authenticity")
        fail("405", evidence_class, "raw-evidence-snapshot-required", "raw-evidence-authenticity")
    header, rows = evidence.tsv()
    if header != RAW_ENVELOPE_FIELDS or len(rows) != 1:
        fail("402", evidence_class, "invalid-class-specific-raw-envelope", "raw-evidence-authenticity")
    row = rows[0]
    if row["evidence_version"] != "h1-raw-execution-evidence-v2" or row["evidence_class"] != evidence_class:
        fail("403", evidence_class, "raw-evidence-class-or-version-mismatch", "raw-evidence-authenticity")
    if expected_requirement_id is not None and row["requirement_id"] != expected_requirement_id:
        fail("403", evidence_class, "raw-evidence-requirement-mismatch", "raw-evidence-authenticity")
    if expected_run_id is not None and row["run_id"] != expected_run_id:
        fail("403", evidence_class, "stale-raw-evidence-run", "raw-evidence-authenticity")
    receipt = ((attested_generator_receipts or {}).get(row["generator_execution_id"])
               if isinstance(attested_generator_receipts, dict) else None)
    if receipt is None:
        fail("403", evidence_class, "missing-authentic-generator-execution-receipt", "raw-evidence-authenticity")
    receipt_bindings = {
        "version": "h1-generator-execution-receipt-v2",
        "generator_execution_id": row["generator_execution_id"], "run_id": row["run_id"],
        "implementation_digest": row["tool_digest"], "actual_exit_status": "0",
        "execution_status": "completed", "output_artifact_ids": row["output_artifact_ids"],
        "output_artifact_digests": row["output_digests"],
    }
    if any(receipt.get(field) != expected for field, expected in receipt_bindings.items()):
        fail("403", evidence_class, "generator-receipt-envelope-binding-mismatch", "raw-evidence-authenticity")
    if (not row["process_id"].isdigit() or not row["monotonic_start_ns"].isdigit() or
            not row["monotonic_completion_ns"].isdigit() or
            int(row["monotonic_completion_ns"]) < int(row["monotonic_start_ns"]) or
            row["actual_exit_status"] != "0" or not re.fullmatch(r"[0-9a-f]{64}", row["tool_digest"])):
        fail("403", evidence_class, "invalid-observed-execution-semantics", "raw-evidence-authenticity")
    try:
        payload = json.loads(row["payload_json"])
    except json.JSONDecodeError:
        fail("404", evidence_class, "invalid-structured-raw-output", "raw-evidence-authenticity")
    if not isinstance(payload, dict):
        fail("404", evidence_class, "invalid-structured-raw-output", "raw-evidence-authenticity")
    forbidden = {"expected", "actual", "comparison", "comparison_result", "success", "status"}
    if forbidden.intersection(payload):
        fail("404", evidence_class, "self-asserted-comparison-or-success", "raw-evidence-authenticity")
    missing = sorted(RAW_CLASS_PAYLOADS[evidence_class] - set(payload))
    if missing:
        fail("404", evidence_class, f"missing-class-semantic:{missing[0]}", "raw-evidence-authenticity")
    if expected_semantics is not None:
        for key, expected in expected_semantics.items():
            if payload.get(key) != expected:
                fail("404", evidence_class, f"independent-semantic-mismatch:{key}", "raw-evidence-authenticity")
    return payload


def emit_execution_receipts(*_args, **_kwargs) -> list[dict[str, str]]:
    """Permanent fail-closed tombstone for the removed v1 metadata emitter."""
    fail("510", "legacy-receipt-emitter", "metadata-only-receipt-emission-prohibited-use-trusted-runner",
         "validator-execution-binding")


def validate_execution_receipts(*_args, **_kwargs) -> None:
    """Permanent fail-closed tombstone for v1 receipt validation."""
    fail("511", "legacy-receipt-validator", "legacy-receipt-validation-prohibited-use-trusted-runner",
         "validator-execution-binding")


def _render_tsv(rows: list[dict[str, str]], fields: list[str] | None = None) -> str:
    from io import StringIO
    if not rows:
        return ""
    target = StringIO(newline="")
    writer = csv.DictWriter(target, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n")
    writer.writeheader(); writer.writerows(rows)
    return target.getvalue()


def main() -> None:
    if len(sys.argv) < 2:
        raise SystemExit("usage: CampaignOperationsH1EvidenceAuthority.py validate-registries [REGISTRY_ROOT] | validate-raw CLASS PATH")
    if sys.argv[1] == "validate-registries" and len(sys.argv) in {2, 3}:
        validate_normative_authority(Path(sys.argv[2]) if len(sys.argv) == 3 else FROZEN_ROOT)
        print("H1_NORMATIVE_AUTHORITY_OK clauses=36 evidence_obligations=183 controls=104 sources=5 "
              "disposition=PHASE_H_H1_FINAL_FALSIFICATION_NOT_VERIFIED")
        return
    if sys.argv[1] == "validate-raw" and len(sys.argv) == 4:
        raw_snapshot = capture_regular_file(Path(sys.argv[3]), f"RAW-{sys.argv[2]}", "standalone-raw-validation")
        validate_raw_evidence(sys.argv[2], raw_snapshot)
        print(f"H1_RAW_EVIDENCE_OK class={sys.argv[2]}")
        return
    raise SystemExit("usage: CampaignOperationsH1EvidenceAuthority.py validate-registries [REGISTRY_ROOT] | validate-raw CLASS PATH")


if __name__ == "__main__":
    main()
