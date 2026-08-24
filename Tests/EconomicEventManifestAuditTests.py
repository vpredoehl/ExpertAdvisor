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
        "validation_status": "validated",
        "diagnostic": "",
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

        self.assertEqual(accepted["validation_status"], "validated")
        self.assertIn(
            "duplicate_source_event_id_in_batch",
            duplicate_identity_a["diagnostic"],
        )
        self.assertIn(
            "duplicate_source_event_id_in_batch",
            duplicate_identity_b["diagnostic"],
        )
        self.assertIn(
            "duplicate_agency_family_timestamp_in_batch",
            duplicate_timestamp_a["diagnostic"],
        )
        self.assertIn(
            "duplicate_agency_family_timestamp_in_batch",
            duplicate_timestamp_b["diagnostic"],
        )

    def test_ignores_entries_that_did_not_validate_individually(self) -> None:
        failed = candidate("census:duplicate", "FAMILY", "1")
        failed["validation_status"] = "failed"
        failed["diagnostic"] = "original failure"
        accepted = candidate("census:duplicate", "FAMILY", "1")

        apply_batch_validation([failed, accepted])

        self.assertEqual(failed["diagnostic"], "original failure")
        self.assertEqual(accepted["validation_status"], "validated")


if __name__ == "__main__":
    unittest.main()
