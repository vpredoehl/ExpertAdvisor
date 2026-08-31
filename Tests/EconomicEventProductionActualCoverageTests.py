#!/usr/bin/env python3

from __future__ import annotations

import hashlib
import json
import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
EVIDENCE = (
    ROOT / "AuditEvidence" / "AuthoritativeEconomicCalendar" /
    "Phase11" / "2026-08-30"
)


class EconomicEventProductionActualCoverageTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.manifest_bytes = (
            EVIDENCE / "pce-production-import-manifest.jsonl"
        ).read_bytes()
        cls.manifest = [
            json.loads(line)
            for line in cls.manifest_bytes.decode("utf-8").splitlines()
        ]
        cls.audit = json.loads(
            (EVIDENCE / "pce-production-readiness.json").read_text(
                encoding="utf-8"
            )
        )

    def test_manifest_identity_and_hash_are_deterministic(self) -> None:
        self.assertEqual(len(self.manifest), 183)
        self.assertEqual(self.audit["manifest_row_count"], 183)
        self.assertEqual(
            hashlib.sha256(self.manifest_bytes).hexdigest(),
            self.audit["manifest_sha256"],
        )
        identities = {
            (row["economic_event_id"], row["revision_sequence"])
            for row in self.manifest
        }
        self.assertEqual(len(identities), 183)

    def test_every_importable_pce_actual_matches_runtime_semantics(self) -> None:
        for row in self.manifest:
            self.assertEqual(row["source_agency"], "BEA")
            self.assertEqual(row["publication_state"], "initial")
            self.assertEqual(row["revision_sequence"], 0)
            self.assertEqual(row["actual_value_kind"], "scalar")
            self.assertEqual(row["actual_unit"], "percent")
            self.assertEqual(row["actual_qualifier"], "m/m")
            self.assertEqual(
                row["semantic_contract"],
                "bea_current_dollar_pce_mom_percent_v1",
            )

    def test_causal_availability_is_the_certified_release_instant(self) -> None:
        by_observation = {
            row["source_observation_id"]: row for row in self.manifest
        }
        self.assertEqual(len(by_observation), 183)
        for reconciliation in self.audit["reconciliation"]:
            actual = by_observation[reconciliation["source_observation_id"]]
            self.assertEqual(
                actual["available_at"],
                reconciliation["candidate_available_at"],
            )
            self.assertEqual(
                reconciliation["candidate_available_at"],
                reconciliation["catalog_event_timestamp_utc"],
            )

    def test_joint_feature_coverage_and_missing_consensus_fail_closed(self) -> None:
        self.assertEqual(
            self.audit["classification_counts"],
            {
                "CONFLICT_EXISTING": 0,
                "DUPLICATE_EXISTING": 0,
                "IDENTITY_MISMATCH": 0,
                "INCOMPATIBLE_CONSENSUS": 0,
                "INVALID_SEMANTICS": 0,
                "MISSING_CONSENSUS": 15,
                "MISSING_EVENT": 0,
                "READY": 168,
            },
        )
        ready = [
            row for row in self.audit["reconciliation"]
            if row["classification"] == "READY"
        ]
        missing = [
            row for row in self.audit["reconciliation"]
            if row["classification"] == "MISSING_CONSENSUS"
        ]
        self.assertEqual(len(ready), 168)
        self.assertEqual(len(missing), 15)
        for row in ready:
            self.assertTrue(row["consensus_actual_semantics_compatible"])
            self.assertEqual(row["selected_consensus_value_kind"], "scalar")
            self.assertEqual(row["selected_consensus_unit"], "percent")
            self.assertEqual(row["selected_consensus_qualifier"], "m/m")
        for row in missing:
            self.assertFalse(row["selected_consensus_present"])
            self.assertIsNone(row["consensus_actual_semantics_compatible"])

    def test_unsupported_source_values_remain_excluded(self) -> None:
        rejected = self.audit["rejected_source_observations"]
        self.assertEqual(len(rejected), 13)
        self.assertEqual(self.audit["rejected_source_observation_count"], 13)
        self.assertTrue(all(row["decision"] == "invalid_semantics"
                            for row in rejected))
        reasons = {row["rejection_reason"] for row in rejected}
        self.assertEqual(
            reasons,
            {
                "unsupported_semantics:pce_less_than_scalar_not_exact",
                "unsupported_semantics:pce_single_month_reference_not_proven",
            },
        )


if __name__ == "__main__":
    unittest.main()
