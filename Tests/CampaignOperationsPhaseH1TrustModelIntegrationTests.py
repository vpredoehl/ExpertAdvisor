#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1TrustModel import (  # noqa: E402
    READINESS_COMPONENTS, TrustModelError, evaluate_readiness,
    generator_execution_complete, snapshot_coverage_complete, validate_legacy_inventory,
    validator_execution_complete,
)
from CampaignOperationsH1EvidenceGraph import counts  # noqa: E402


class TrustModelIntegrationTests(unittest.TestCase):
    def complete(self):
        return {key: True for key in READINESS_COMPONENTS}

    def test_exact_readiness_equation_accepts_only_all_boundaries(self):
        self.assertEqual(evaluate_readiness(self.complete()),
                         (True, "H1T199 all-trust-boundaries-complete"))

    def test_one_boundary_at_a_time_fails_closed(self):
        for index, boundary in enumerate(READINESS_COMPONENTS):
            with self.subTest(boundary=boundary):
                values = self.complete(); values[boundary] = False
                self.assertEqual(evaluate_readiness(values),
                                 (False, f"H1T1{index:02d} {boundary}=false"))

    def test_alternative_readiness_schema_is_rejected(self):
        values = self.complete(); values["legacy_ready"] = True
        with self.assertRaisesRegex(TrustModelError, "readiness-component-schema:legacy_ready"):
            evaluate_readiness(values)

    def test_legacy_graph_counts_cannot_report_ready(self):
        graph = {"requirements": {}, "fixtures": {}, "generators": {}, "artifacts": {}, "edges": {}}
        health = [{"count": "0"}]
        self.assertEqual(counts(graph, [], [], [], health)["ready"], 0)

    def test_machine_legacy_inventory_has_no_reachable_entry(self):
        rows = validate_legacy_inventory(
            ROOT / "Tests/fixtures/CampaignOperationsH1LegacyPathInventory.tsv")
        self.assertTrue(rows)
        self.assertNotIn("still_reachable", {row["disposition"] for row in rows})

    def test_v1_or_omitted_validator_receipt_is_incomplete(self):
        receipt = {"version": "h1-validator-execution-receipt-v2", "validator_execution_id": "VX1",
                   "validator_id": "VAL", "run_id": "run", "actual_exit_status": "0",
                   "execution_status": "completed", "output_validator_result_ids": "VR"}
        result = {"validator_result_id": "VR", "validator_execution_id": "VX1", "validator_id": "VAL"}
        self.assertTrue(validator_execution_complete([receipt], {"VAL"}, [result], "run"))
        receipt["version"] = "h1-validator-execution-receipt-v1"
        self.assertFalse(validator_execution_complete([receipt], {"VAL"}, [result], "run"))
        self.assertFalse(validator_execution_complete([], {"VAL"}, [result], "run"))

    def test_snapshot_identity_requires_exact_artifact_and_run_coverage(self):
        row = {"artifact_id": "ART", "snapshot_id": "S", "lexical_path": "a.tsv",
               "device": "1", "inode": "2", "size": "3", "digest": "a" * 64, "run_id": "run"}
        self.assertTrue(snapshot_coverage_complete([row], {"ART"}, "run"))
        self.assertFalse(snapshot_coverage_complete([row], {"ART", "MISSING"}, "run"))
        self.assertFalse(snapshot_coverage_complete([row], {"ART"}, "other-run"))

    def test_removing_generator_receipt_or_output_snapshot_is_incomplete(self):
        receipt = {"version": "h1-generator-execution-receipt-v2", "generator_execution_id": "GX1",
                   "run_id": "run", "actual_exit_status": "0", "execution_status": "completed",
                   "output_snapshot_ids": "SNAP1"}
        envelope = {"requirement_id": "REQ1", "generator_execution_id": "GX1"}
        snapshot = {"snapshot_id": "SNAP1"}
        self.assertTrue(generator_execution_complete([receipt], [envelope], [snapshot], {"REQ1"}, "run"))
        self.assertFalse(generator_execution_complete([], [envelope], [snapshot], {"REQ1"}, "run"))
        self.assertFalse(generator_execution_complete([receipt], [envelope], [], {"REQ1"}, "run"))


if __name__ == "__main__":
    unittest.main(verbosity=2)
