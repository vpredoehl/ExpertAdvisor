#!/usr/bin/env python3
"""Permanent regressions for ADR-0019B targeted defensive evidence authority."""
from __future__ import annotations

import contextlib
import csv
import io
import os
import shutil
import stat
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
import CampaignOperationsH1EvidenceAuthority as authority  # noqa: E402


class FailureMixin:
    def assert_failure(self, expected: str, function, *args):
        output = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stderr(output):
            function(*args)
        self.assertEqual(output.getvalue().strip(), expected)


class NormativeAuthorityTests(FailureMixin, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-authority-")
        self.root = Path(self.temp.name)
        shutil.copy(ROOT / "Tests/fixtures/CampaignOperationsH1Requirements.tsv", self.root)
        for name in ["CampaignOperationsH1NormativeClauses.tsv", "CampaignOperationsH1AssuranceControls.tsv",
                     "CampaignOperationsH1EvidenceObligations.tsv", "CampaignOperationsH1ClauseRequirementDerivations.tsv",
                     "CampaignOperationsH1FinalAssuranceControls.tsv", "CampaignOperationsH1MutationMechanics.tsv",
                     "CampaignOperationsH1ExpectedValueAuthorities.tsv", "CampaignOperationsH1ObservedValueProvenance.tsv",
                     "CampaignOperationsH1RawEvidenceContracts.tsv", "CampaignOperationsH1DirectoryPolicy.tsv",
                     "CampaignOperationsH1ReviewDisposition.tsv"]:
            shutil.copy(ROOT / "Tests/fixtures" / name, self.root)

    def tearDown(self): self.temp.cleanup()

    def rows(self):
        path = self.root / "CampaignOperationsH1Requirements.tsv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            return list(reader.fieldnames), list(reader)

    def write(self, fields, rows):
        with (self.root / "CampaignOperationsH1Requirements.tsv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)

    def test_synthetic_invalid_evidence_chain_is_non_normative(self):
        fields, rows = self.rows(); synthetic = dict(rows[0])
        synthetic["requirement_id"] = "H1-SYNTHETIC-INVALID-EVIDENCE"
        synthetic["fixture_ids"] = "H1SYNTH001"; synthetic["runtime_record_ids"] = "RT-H1SYNTH001"
        synthetic["report_entry_ids"] = "REP-H1SYNTH001"
        rows.append(synthetic); self.write(fields, rows)
        self.assert_failure(
            "H1A106 key=H1-SYNTHETIC-INVALID-EVIDENCE stage=normative-requirement-reconciliation detail=non-normative-requirement",
            authority.validate_normative_authority, self.root)

    def test_legitimate_chain_substitution_is_rejected(self):
        fields, rows = self.rows(); template = dict(rows[0])
        rows = [row for row in rows if row["requirement_id"] != "H1-ACL-ADMISSION"]
        template["requirement_id"] = "H1-SYNTHETIC-INVALID-EVIDENCE"; rows.append(template)
        self.write(fields, rows)
        self.assert_failure(
            "H1A105 key=H1-ACL-ADMISSION stage=normative-requirement-reconciliation detail=missing-normative-requirement",
            authority.validate_normative_authority, self.root)

    def test_missing_normative_requirement_is_rejected(self):
        fields, rows = self.rows(); rows = [row for row in rows if row["requirement_id"] != "H1-ACL-ADMISSION"]
        self.write(fields, rows)
        self.assert_failure(
            "H1A105 key=H1-ACL-ADMISSION stage=normative-requirement-reconciliation detail=missing-normative-requirement",
            authority.validate_normative_authority, self.root)

    def test_extra_non_normative_requirement_is_rejected_after_digest_refresh(self):
        self.test_synthetic_invalid_evidence_chain_is_non_normative()

    def authority_rows(self):
        path = self.root / "CampaignOperationsH1NormativeClauses.tsv"
        with path.open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            return list(reader.fieldnames), list(reader)

    def write_authority(self, fields, rows):
        with (self.root / "CampaignOperationsH1NormativeClauses.tsv").open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=fields, delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerows(rows)

    def assert_clause_omission(self, clause_id):
        fields, rows = self.authority_rows()
        rows = [row for row in rows if row["normative_clause_id"] != clause_id]
        self.write_authority(fields, rows)
        self.assert_failure(
            f"H1A109 key={clause_id} stage=governing-clause-reconciliation detail=governing-clause-without-authority-row",
            authority.validate_normative_authority, self.root, self.root)

    def test_omitted_adr0019_clause_is_rejected(self): self.assert_clause_omission("ADR19-AUTHORITY")
    def test_omitted_adr0019a_clause_is_rejected(self): self.assert_clause_omission("ADR19A-IDENTIFIER")
    def test_omitted_adr0019b_clause_is_rejected(self): self.assert_clause_omission("ADR19B-ACL")
    def test_omitted_corrected_architecture_clause_is_rejected(self): self.assert_clause_omission("PHASEH-INCREMENTS")
    def test_omitted_volume_xii_clause_is_rejected(self): self.assert_clause_omission("VOLUMEXII-TESTING")

    def test_valid_clause_mapped_to_wrong_governing_document_is_rejected(self):
        fields, rows = self.authority_rows()
        row = next(value for value in rows if value["normative_clause_id"] == "ADR19A-IDENTIFIER")
        row["source_document"] = "docs/architecture/adr/ADR-0019B-h1-sealed-role-deployment-contract.md"
        self.write_authority(fields, rows)
        self.assert_failure(
            "H1A104 key=ADR19A-IDENTIFIER stage=normative-authority-validation detail=wrong-governing-document-or-precedence",
            authority.validate_normative_authority, self.root, self.root)

    def test_broad_section_authority_is_rejected(self):
        fields, rows = self.authority_rows()
        row = next(value for value in rows if value["normative_clause_id"] == "ADR19-AUTHORITY")
        row["exact_source_anchor"] = "lines 1-100"
        self.write_authority(fields, rows)
        self.assert_failure(
            "H1A104 key=ADR19-AUTHORITY stage=normative-authority-validation detail=broad-or-invalid-source-anchor",
            authority.validate_normative_authority, self.root, self.root)

    def test_test_mechanic_labeled_normative_is_rejected(self):
        fields, rows = self.authority_rows()
        row = next(value for value in rows if value["normative_clause_id"] == "ADR19-AUTHORITY")
        row["derived_h1_requirement_identity"] = "H1-MUTATION-ACL_ACTUAL_ORIGIN"
        self.write_authority(fields, rows)
        self.assert_failure(
            "H1A106 key=H1-MUTATION-ACL_ACTUAL_ORIGIN stage=normative-requirement-reconciliation detail=test-mechanic-labeled-normative",
            authority.validate_normative_authority, self.root, self.root)


class RuntimeInventoryTests(FailureMixin, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-runtime-"); self.root = Path(self.temp.name)
        self.graph = {"fixtures": {"H1ONE": {"fixture_id": "H1ONE", "requirement_id": "REQ",
                      "generator_id": "GEN-TRACE", "expected_runtime_record_id": "RT-H1ONE",
                      "report_entry_id": "REP-H1ONE", "artifact_id": "ART-H1ONE"}},
                      "artifacts": {"ART-H1ONE": {"artifact_id": "ART-H1ONE", "record_key_value": "H1ONE"}}}
        fields = ["fixture_id", "requirement_id", "generator_id", "emitted_runtime_record_id",
                  "report_entry_id", "output_artifact_id"]
        rows = [["H1ONE", "REQ", "GEN-TRACE", "RT-H1ONE", "REP-H1ONE", "ART-H1ONE"],
                ["H1UNEXPECTED999", "UNKNOWN", "GEN-TRACE", "RT-UNKNOWN", "REP-UNKNOWN", "ART-UNKNOWN"]]
        with (self.root / "h1-runtime-results.tsv").open("w", newline="") as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n"); writer.writerow(fields); writer.writerows(rows)

    def tearDown(self): self.temp.cleanup()

    def test_unexpected_complete_runtime_row_is_rejected(self):
        self.assert_failure(
            "H1A203 key=H1UNEXPECTED999 stage=exact-runtime-inventory detail=unexpected-runtime-row:h1-runtime-results.tsv",
            authority.validate_exact_runtime_inventory, self.root, self.graph)


class FilesystemRepresentationTests(FailureMixin, unittest.TestCase):
    def setUp(self): self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-files-"); self.root = Path(self.temp.name)
    def tearDown(self): self.temp.cleanup()
    @staticmethod
    def artifact(identifier, path):
        return {"artifact_id": identifier, "path": path, "required_phase": "base"}

    def test_symbolic_link_artifact_outside_root_is_rejected(self):
        outside = Path(self.temp.name).parent / (Path(self.temp.name).name + "-outside")
        outside.write_text("raw\n"); os.symlink(outside, self.root / "evidence.tsv")
        try:
            self.assert_failure("H1A305 key=evidence.tsv stage=filesystem-representation-validation detail=symbolic-link-artifact",
                                authority.validate_filesystem, self.root, {"ART-A": self.artifact("ART-A", "evidence.tsv")})
        finally: outside.unlink()

    def test_unregistered_directory_is_rejected(self):
        (self.root / "rogue").mkdir()
        self.assert_failure("H1A306 key=rogue stage=directory-policy-validation detail=unregistered-directory",
                            authority.validate_filesystem, self.root, {})

    def test_symbolic_link_directory_is_rejected(self):
        target = self.root / "runtime-artifacts"; target.mkdir(); os.symlink(target, self.root / "linked")
        self.assert_failure("H1A305 key=linked stage=filesystem-representation-validation detail=symbolic-link-directory-or-entry",
                            authority.validate_filesystem, self.root,
                            {"ART-A": self.artifact("ART-A", "linked/evidence.tsv")})

    def test_duplicate_physical_file_identity_is_rejected(self):
        (self.root / "a.tsv").write_text("raw\n"); os.link(self.root / "a.tsv", self.root / "b.tsv")
        artifacts = {"ART-A": self.artifact("ART-A", "a.tsv"), "ART-B": self.artifact("ART-B", "b.tsv")}
        self.assert_failure("H1A310 key=ART-A stage=physical-file-identity-validation detail=unsupported-hard-link-count:2",
                            authority.validate_filesystem, self.root, artifacts)

    def test_path_escape_is_rejected_lexically(self):
        self.assert_failure("H1A301 key=ART-A stage=filesystem-representation-validation detail=non-canonical-relative-path",
                            authority.validate_filesystem, self.root,
                            {"ART-A": self.artifact("ART-A", "../outside.tsv")})

    def test_unsupported_metadata_classification_is_fail_closed(self):
        self.assertEqual(authority.classify_mode(stat.S_IFSOCK | 0o600), "unsupported")
        self.assertEqual(authority.classify_mode(stat.S_IFIFO | 0o600), "unsupported")

    def test_symbolic_link_replacement_before_descriptor_read_is_rejected(self):
        path = self.root / "evidence.tsv"; path.write_text("trusted\n")
        target = self.root / "replacement.tsv"; target.write_text("changed\n")
        def replace(_): path.unlink(); os.symlink(target.name, path)
        self.assert_failure(
            "H1A314 key=ART-A stage=snapshot-identity-validation detail=registered-path-identity-changed",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1", replace)

    def test_different_inode_replacement_before_descriptor_read_is_rejected(self):
        path = self.root / "evidence.tsv"; path.write_text("trusted\n")
        def replace(_):
            path.unlink(); path.write_text("changed\n")
        self.assert_failure(
            "H1A314 key=ART-A stage=snapshot-identity-validation detail=registered-path-identity-changed",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1", replace)

    def test_matching_content_inode_substitution_is_rejected(self):
        path = self.root / "evidence.tsv"; path.write_text("same\n")
        def replace(_):
            path.unlink(); path.write_text("same\n")
        self.assert_failure(
            "H1A314 key=ART-A stage=snapshot-identity-validation detail=registered-path-identity-changed",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1", replace)

    def test_parent_directory_replacement_is_rejected(self):
        directory = self.root / "nested"; directory.mkdir(); (directory / "evidence.tsv").write_text("trusted\n")
        def replace(_):
            directory.rename(self.root / "old-nested")
            directory.mkdir(); (directory / "evidence.tsv").write_text("changed\n")
        self.assert_failure(
            "H1A314 key=ART-A stage=snapshot-identity-validation detail=parent-directory-identity-changed",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "nested/evidence.tsv")}, False, "run-1", replace)

    def test_registered_path_moved_before_read_is_rejected(self):
        path = self.root / "evidence.tsv"; path.write_text("trusted\n")
        def replace(_): path.rename(self.root / "moved.tsv")
        self.assert_failure(
            "H1A314 key=ART-A stage=snapshot-identity-validation detail=registered-path-binding-lost",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1", replace)

    def test_changed_file_after_snapshot_cannot_change_snapshot_bytes(self):
        path = self.root / "evidence.tsv"; path.write_text("trusted\n")
        captured = authority.validate_filesystem(
            self.root, {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1")
        path.write_text("changed\n")
        self.assertEqual(captured.by_id("ART-A").data, b"trusted\n")
        self.assertEqual(captured.by_id("ART-A").digest,
                         __import__("hashlib").sha256(b"trusted\n").hexdigest())

    def test_stale_snapshot_run_is_rejected(self):
        (self.root / "evidence.tsv").write_text("trusted\n")
        captured = authority.validate_filesystem(
            self.root, {"ART-A": self.artifact("ART-A", "evidence.tsv")}, False, "run-1")
        with self.assertRaises(authority.SnapshotError) as raised:
            captured.assert_run_id("run-2")
        self.assertEqual(raised.exception.detail, "stale-snapshot-run-identity")

    def test_path_alias_is_rejected(self):
        self.assert_failure(
            "H1A301 key=ART-A stage=filesystem-representation-validation detail=non-canonical-relative-path",
            authority.validate_filesystem, self.root,
            {"ART-A": self.artifact("ART-A", "nested/../evidence.tsv")})

    def test_generate_mode_allows_absent_generated_artifact(self):
        artifact = self.artifact("ART-A", "generated.tsv"); artifact["required_phase"] = "generated"
        captured = authority.validate_filesystem(self.root, {"ART-A": artifact}, True, "run-1")
        self.assertNotIn("generated.tsv", captured)

    def test_validate_mode_requires_generated_artifact(self):
        artifact = self.artifact("ART-A", "generated.tsv"); artifact["required_phase"] = "generated"
        self.assert_failure(
            "H1A311 key=ART-A stage=filesystem-representation-validation detail=missing-registered-artifact:generated.tsv",
            authority.validate_filesystem, self.root, {"ART-A": artifact}, False, "run-1")


class ReceiptAndRawAuthenticityTests(FailureMixin, unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-receipt-"); self.root = Path(self.temp.name)
        self.key = self.root.parent / (self.root.name + ".key"); self.key.write_bytes(os.urandom(32)); self.key.chmod(0o600)
        self.validators = {"VAL-TRACE": {"validator_id": "VAL-TRACE", "version": "h1-validator-registry-v2",
            "implementation": "Scripts/CampaignOperationsH1EvidenceValidator.py", "entry_point": "validate_generic"}}
        self.results = [{"validator_result_id": "VR-H1ONE", "validator_execution_id": "",
            "validator_id": "VAL-TRACE", "artifact_ids": "ART-H1ONE", "aggregate_artifact_digest": "a" * 64}]
    def tearDown(self):
        if self.key.exists(): self.key.unlink()
        self.temp.cleanup()

    def test_hand_authored_validator_result_without_receipt_is_rejected(self):
        self.assert_failure("H1A511 key=legacy-receipt-validator stage=validator-execution-binding detail=legacy-receipt-validation-prohibited-use-trusted-runner",
                            authority.validate_execution_receipts, self.root, "run-1", self.results, self.validators, self.key)

    def assert_summary_rejected(self, evidence_class):
        path = self.root / (evidence_class + ".tsv"); path.write_text("claim\nPASS\n")
        expected = f"H1A402 key={evidence_class} stage=raw-evidence-authenticity detail=invalid-class-specific-raw-envelope"
        snapshot = authority.capture_regular_file(path, "RAW", "run-1")
        self.assert_failure(expected, authority.validate_raw_evidence, evidence_class, snapshot)

    def test_non_authentic_lock_pair_is_rejected(self): self.assert_summary_rejected("lock")
    def test_non_authentic_acl_pair_is_rejected(self): self.assert_summary_rejected("acl_origin")
    def test_non_authentic_restore_result_is_rejected(self): self.assert_summary_rejected("restore")
    def test_hand_authored_release_build_success_is_rejected(self): self.assert_summary_rejected("release_build")
    def test_hand_authored_exclusion_pass_is_rejected(self): self.assert_summary_rejected("exclusion")

    def test_deleted_raw_evidence_is_rejected(self):
        for evidence_class in ("lock", "acl_origin", "acl_catalog", "restore", "release_build", "mutation", "exclusion"):
            with self.subTest(evidence_class=evidence_class):
                path = self.root / ("deleted-" + evidence_class + ".tsv")
                self.assert_failure(
                    f"H1A405 key={evidence_class} stage=raw-evidence-authenticity detail=missing-raw-evidence:{path.name}",
                    authority.validate_raw_evidence, evidence_class, path)

    def raw_envelope(self, evidence_class, requirement="REQ-ONE", run_id="run-1", payload=None):
        required = {key: [] for key in authority.RAW_CLASS_PAYLOADS[evidence_class]}
        required.update(payload or {})
        row = {
            "evidence_version": "h1-raw-execution-evidence-v2", "evidence_class": evidence_class,
            "requirement_id": requirement, "run_id": run_id, "implementation_version": "v1",
            "invocation_contract": "[tool,--capture]", "tool_identity": "tool-v1", "tool_digest": "a" * 64,
            "process_id": "123", "monotonic_start_ns": "100", "monotonic_completion_ns": "200",
            "actual_exit_status": "0", "input_snapshot_ids": "SNAP-1", "input_digests": "b" * 64,
            "output_artifact_ids": "ART-ONE", "output_digests": "c" * 64,
            "stdout": "captured", "stderr": "", "payload_json": __import__("json").dumps(required),
            "generator_execution_id": "GEXEC-1",
        }
        path = self.root / (evidence_class + "-valid.tsv")
        with path.open("w", newline="") as target:
            writer = csv.DictWriter(target, fieldnames=authority.RAW_ENVELOPE_FIELDS,
                                    delimiter="\t", lineterminator="\n")
            writer.writeheader(); writer.writerow(row)
        return authority.capture_regular_file(path, f"RAW-{evidence_class}", run_id)

    @staticmethod
    def generator_receipts(run_id="run-1"):
        return {"GEXEC-1": {
            "version": "h1-generator-execution-receipt-v2", "generator_execution_id": "GEXEC-1",
            "run_id": run_id, "implementation_digest": "a" * 64, "actual_exit_status": "0",
            "execution_status": "completed", "output_artifact_ids": "ART-ONE",
            "output_artifact_digests": "c" * 64,
        }}

    def test_every_declared_raw_class_requires_attested_class_specific_semantics(self):
        for evidence_class in authority.RAW_CLASS_PAYLOADS:
            with self.subTest(evidence_class=evidence_class):
                path = self.raw_envelope(evidence_class)
                payload = authority.validate_raw_evidence(
                    evidence_class, path, "REQ-ONE", "run-1", self.generator_receipts())
                self.assertEqual(set(authority.RAW_CLASS_PAYLOADS[evidence_class]) - set(payload), set())

    def test_valid_markers_without_execution_receipt_are_rejected(self):
        path = self.raw_envelope("lock")
        self.assert_failure(
            "H1A403 key=lock stage=raw-evidence-authenticity detail=missing-authentic-generator-execution-receipt",
            authority.validate_raw_evidence, "lock", path, "REQ-ONE", "run-1", set())

    def test_raw_evidence_from_another_class_requirement_or_run_is_rejected(self):
        path = self.raw_envelope("lock")
        self.assert_failure("H1A403 key=acl_origin stage=raw-evidence-authenticity detail=raw-evidence-class-or-version-mismatch",
                            authority.validate_raw_evidence, "acl_origin", path, "REQ-ONE", "run-1", self.generator_receipts())
        self.assert_failure("H1A403 key=lock stage=raw-evidence-authenticity detail=raw-evidence-requirement-mismatch",
                            authority.validate_raw_evidence, "lock", path, "REQ-TWO", "run-1", self.generator_receipts())
        self.assert_failure("H1A403 key=lock stage=raw-evidence-authenticity detail=stale-raw-evidence-run",
                            authority.validate_raw_evidence, "lock", path, "REQ-ONE", "run-2", self.generator_receipts())

    def test_coauthored_expected_actual_and_precomputed_comparison_are_rejected(self):
        path = self.raw_envelope("acl_catalog", payload={"expected": "x", "actual": "x", "comparison": "equal"})
        self.assert_failure("H1A404 key=acl_catalog stage=raw-evidence-authenticity detail=self-asserted-comparison-or-success",
                            authority.validate_raw_evidence, "acl_catalog", path, "REQ-ONE", "run-1", self.generator_receipts())

    def test_independent_semantic_mismatch_is_rejected_even_after_envelope_regeneration(self):
        path = self.raw_envelope("exclusion", payload={"result_rows": ["false-success"]})
        self.assert_failure("H1A404 key=exclusion stage=raw-evidence-authenticity detail=independent-semantic-mismatch:result_rows",
                            authority.validate_raw_evidence, "exclusion", path, "REQ-ONE", "run-1", self.generator_receipts(),
                            {"result_rows": []})


if __name__ == "__main__":
    unittest.main(verbosity=2)
