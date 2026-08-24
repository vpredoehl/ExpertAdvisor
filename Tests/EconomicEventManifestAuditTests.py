#!/usr/bin/env python3

import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "Scripts"))

from audit_economic_event_manifest import apply_batch_validation


def candidate(source_id: str, family: str, timestamp: str) -> dict[str, str]:
    return {
        "source_agency": "CENSUS",
        "source_event_id": source_id,
        "raw_event_family": family,
        "event_timestamp_unix_micros": timestamp,
        "acquisition_status": "succeeded",
        "provenance_status": "validated",
        "parse_status": "parsed",
        "validation_status": "validated",
        "collision_status": "not_run",
        "acceptance_status": "rejected",
        "collision_diagnostic": "",
        "timestamp_confidence": "exact",
        "source_release_date": "2020-02-03",
        "source_release_time": "10:00:00",
        "source_timezone": "America/New_York",
        "reference_period": "2020-01",
        "source_artifact_sha256": "a" * 64,
        "source_url": "https://www.census.gov/example_202001.pdf",
    }


class BatchValidationTests(unittest.TestCase):
    def test_marks_only_entries_participating_in_batch_collisions(self) -> None:
        duplicate_identity_a = candidate("census:duplicate", "FAMILY_A", "1")
        duplicate_identity_b = candidate("census:duplicate", "FAMILY_B", "2")
        duplicate_timestamp_a = candidate("census:a", "FAMILY_C", "3")
        duplicate_timestamp_b = candidate("census:b", "FAMILY_C", "3")
        accepted = candidate("census:accepted", "FAMILY_D", "4")
        rows = [
            duplicate_identity_a,
            duplicate_identity_b,
            duplicate_timestamp_a,
            duplicate_timestamp_b,
            accepted,
        ]

        apply_batch_validation(rows)

        self.assertEqual(accepted["collision_status"], "validated")
        self.assertEqual(accepted["acceptance_status"], "accepted")
        self.assertIn(
            "duplicate_source_event_id_in_batch",
            duplicate_identity_a["collision_diagnostic"],
        )
        self.assertIn(
            "duplicate_source_event_id_in_batch",
            duplicate_identity_b["collision_diagnostic"],
        )
        self.assertIn(
            "duplicate_agency_family_timestamp_in_batch",
            duplicate_timestamp_a["collision_diagnostic"],
        )
        self.assertIn(
            "duplicate_agency_family_timestamp_in_batch",
            duplicate_timestamp_b["collision_diagnostic"],
        )

    def test_invalid_candidate_still_blocks_colliding_valid_candidate(self) -> None:
        failed = candidate("census:duplicate", "FAMILY", "1")
        failed["validation_status"] = "failed"
        accepted = candidate("census:duplicate", "FAMILY", "1")

        apply_batch_validation([failed, accepted])

        self.assertEqual(failed["validation_status"], "failed")
        self.assertEqual(failed["collision_status"], "failed")
        self.assertEqual(accepted["collision_status"], "failed")
        self.assertEqual(accepted["validation_status"], "validated")
        self.assertEqual(accepted["acceptance_status"], "rejected")
        self.assertIn(
            "duplicate_source_event_id_in_batch",
            accepted["collision_diagnostic"],
        )

    def test_census_exact_alias_selects_reference_period_artifact(self) -> None:
        alias = candidate("census:family:cb20-1", "FAMILY", "1")
        alias["source_url"] = "https://www.census.gov/example_201912.pdf"
        canonical = candidate("census:family:cb20-1", "FAMILY", "1")

        apply_batch_validation([alias, canonical])

        self.assertEqual(alias["collision_status"], "duplicate_evidence")
        self.assertEqual(alias["acceptance_status"], "rejected")
        self.assertEqual(canonical["collision_status"], "validated")
        self.assertEqual(canonical["acceptance_status"], "accepted")

    def test_dol_exact_alias_selects_url_with_matching_directory_year(self) -> None:
        alias = candidate("dol_eta:usdl-12-2533-nat", "WEEKLY_CLAIMS", "1")
        canonical = candidate("dol_eta:usdl-12-2533-nat", "WEEKLY_CLAIMS", "1")
        for row in (alias, canonical):
            row["source_agency"] = "DOL_ETA"
            row["source_release_date"] = "2013-01-03"
            row["reference_period"] = "week ending 2012-12-29"
            row["artifact_sha256"] = "b" * 64
        alias["source_url"] = "https://oui.doleta.gov/press/2012/010313.asp"
        canonical["source_url"] = "https://oui.doleta.gov/press/2013/010313.asp"

        apply_batch_validation([alias, canonical])

        self.assertEqual(alias["collision_status"], "duplicate_evidence")
        self.assertEqual(alias["acceptance_status"], "rejected")
        self.assertEqual(canonical["collision_status"], "validated")
        self.assertEqual(canonical["acceptance_status"], "accepted")

    def test_evidenced_fomc_date_only_pair_is_not_a_collision(self) -> None:
        first = candidate(
            "federal_reserve:monetary20140917a", "FOMC_STATEMENT", "1"
        )
        second = candidate(
            "federal_reserve:monetary20140917c", "FOMC_STATEMENT", "1"
        )
        for row in (first, second):
            row["source_agency"] = "FEDERAL_RESERVE"
            row["timestamp_confidence"] = "date_only"
            row["source_release_date"] = "2014-09-17"

        apply_batch_validation([first, second])

        self.assertEqual(first["collision_status"], "validated")
        self.assertEqual(second["collision_status"], "validated")
        self.assertEqual(first["acceptance_status"], "accepted")
        self.assertEqual(second["acceptance_status"], "accepted")


if __name__ == "__main__":
    unittest.main()
