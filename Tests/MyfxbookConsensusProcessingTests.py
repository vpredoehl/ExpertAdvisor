#!/usr/bin/env python3

from __future__ import annotations

import csv
import hashlib
import pathlib
import sys
import tempfile
import unittest
from collections import Counter


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

import process_myfxbook_consensus as processor


RAW_CAPTURE = (
    ROOT / "EconomicCalendar/raw/myfxbook/myfxbook_consensus_history.json"
)
OANDA_MATCHES = ROOT / "EconomicCalendar/raw/oanda/oanda_consensus_matches.csv"


def read_csv(path: pathlib.Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def digest(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MyfxbookConsensusProcessingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = tempfile.TemporaryDirectory()
        cls.output = pathlib.Path(cls.temporary.name) / "first"
        cls.raw_digest_before = digest(RAW_CAPTURE)
        cls.summary = processor.process_inputs(
            RAW_CAPTURE, OANDA_MATCHES, cls.output
        )
        cls.normalized = read_csv(cls.output / processor.NORMALIZED_FILENAME)
        cls.gap_candidates = read_csv(cls.output / processor.GAP_FILL_FILENAME)
        cls.reconciliation = read_csv(
            cls.output / processor.OANDA_RECONCILIATION_FILENAME
        )
        cls.exclusions = read_csv(cls.output / processor.EXCLUSIONS_FILENAME)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_recognizes_nine_source_families_and_exact_mapping(self) -> None:
        observed = {
            row["myfxbook_source_family"] for row in self.normalized
        }
        self.assertEqual(observed, set(processor.SOURCE_TO_CANONICAL))
        observed_mapping = {
            row["myfxbook_source_family"]: row["event_family"]
            for row in self.normalized
        }
        self.assertEqual(observed_mapping, processor.SOURCE_TO_CANONICAL)
        observed_ids = {
            row["myfxbook_source_family"]: int(row["myfxbook_event_id"])
            for row in self.normalized
        }
        self.assertEqual(observed_ids, processor.EXPECTED_EVENT_IDS)

    def test_duplicate_dates_are_retained_and_excluded(self) -> None:
        expected_keys = {
            ("EMPLOYMENT", "2025-12-16"),
            ("JOLTS", "2025-12-09"),
            ("PCE", "2019-04-29"),
            ("PCE", "2026-01-22"),
            ("PPI", "2026-01-14"),
        }
        duplicate_rows = [
            row for row in self.normalized
            if row["family_date_observation_count"] != "1"
        ]
        self.assertEqual(len(duplicate_rows), 10)
        self.assertEqual(
            {(row["event_family"], row["release_date"]) for row in duplicate_rows},
            expected_keys,
        )
        self.assertTrue(
            all(row["classification"] == "ambiguous_duplicate" for row in duplicate_rows)
        )
        self.assertTrue(
            all(row["automatic_candidate_eligible"] == "0" for row in duplicate_rows)
        )
        gap_keys = {
            (row["event_family"], row["release_date"])
            for row in self.gap_candidates
        }
        self.assertTrue(expected_keys.isdisjoint(gap_keys))

    def test_pce_nulls_remain_row_level_null_evidence(self) -> None:
        nulls = [
            row for row in self.normalized
            if row["event_family"] == "PCE" and row["consensus_present"] == "0"
        ]
        self.assertEqual(len(nulls), 44)
        self.assertTrue(all(row["myfxbook_consensus"] == "" for row in nulls))
        self.assertEqual(
            sum(row["anomaly_type"] == "pce_null_consensus" for row in self.exclusions),
            44,
        )

    def test_known_cpi_and_gdp_anomalies_are_not_candidates(self) -> None:
        cpi = next(
            row for row in self.normalized
            if row["event_family"] == "CPI" and row["release_date"] == "2024-02-09"
        )
        self.assertEqual(cpi["classification"], "manual_review")
        self.assertEqual(cpi["consensus_present"], "0")
        self.assertEqual(cpi["automatic_candidate_eligible"], "0")
        gdp = next(
            row for row in self.normalized
            if row["event_family"] == "GDP" and row["release_date"] == "2013-03-28"
        )
        self.assertEqual(gdp["classification"], "unique_null")
        self.assertEqual(gdp["myfxbook_consensus"], "")

    def test_fomc_emergency_events_remain_null(self) -> None:
        dates = {"2020-03-03", "2020-03-15"}
        rows = [
            row for row in self.normalized
            if row["event_family"] == "FOMC" and row["release_date"] in dates
        ]
        self.assertEqual({row["release_date"] for row in rows}, dates)
        self.assertTrue(all(row["classification"] == "unique_null" for row in rows))
        self.assertTrue(all(row["myfxbook_consensus"] == "" for row in rows))
        reconciled = [
            row for row in self.reconciliation
            if row["event_family"] == "FOMC" and row["official_release_date"] in dates
        ]
        self.assertEqual(len(reconciled), 2)
        self.assertTrue(
            all(row["merge_classification"] == "myfxbook_null" for row in reconciled)
        )
        self.assertTrue(
            all(row["candidate_myfxbook_consensus"] == "" for row in reconciled)
        )

    def test_gap_fill_is_exactly_the_verified_jolts_sequence(self) -> None:
        expected_counts = {
            family: 0 for family in processor.OANDA_REMAINING_COVERAGE_WINDOWS
        }
        expected_counts["JOLTS"] = 113
        self.assertEqual(self.summary["gap_fill_candidates"]["by_family"], expected_counts)
        self.assertEqual(len(self.gap_candidates), 113)
        self.assertEqual(Counter(row["event_family"] for row in self.gap_candidates), Counter({"JOLTS": 113}))
        self.assertEqual(
            (self.gap_candidates[0]["release_date"], self.gap_candidates[0]["myfxbook_consensus"]),
            ("2014-07-08", "4530000"),
        )
        self.assertEqual(
            (self.gap_candidates[-1]["release_date"], self.gap_candidates[-1]["myfxbook_consensus"]),
            ("2023-11-01", "9250000"),
        )
        self.assertTrue(all(row["consensus_source"] == "myfxbook" for row in self.gap_candidates))

    def test_oanda_blank_reconciliation_has_exact_candidates(self) -> None:
        self.assertEqual(len(self.reconciliation), 11)
        counts = Counter(row["merge_classification"] for row in self.reconciliation)
        self.assertEqual(
            counts,
            Counter(
                {
                    "myfxbook_populated_candidate": 3,
                    "myfxbook_null": 2,
                    "myfxbook_no_match": 6,
                }
            ),
        )
        candidates = {
            (
                row["event_family"],
                row["official_release_date"],
                row["candidate_myfxbook_consensus"],
            )
            for row in self.reconciliation
            if row["merge_classification"] == "myfxbook_populated_candidate"
        }
        self.assertEqual(
            candidates,
            {
                ("PPI", "2013-12-13", "-0.1"),
                ("RETAIL_SALES", "2022-09-15", "0.0"),
                ("CPI", "2023-12-12", "0.0"),
            },
        )
        populated = [
            row for row in self.reconciliation
            if row["merge_classification"] == "myfxbook_populated_candidate"
        ]
        self.assertTrue(all(row["original_oanda_forecast"] == "" for row in populated))
        self.assertTrue(all(row["candidate_consensus_source"] == "myfxbook" for row in populated))

    def test_processing_is_byte_deterministic_and_does_not_change_raw_capture(self) -> None:
        second = pathlib.Path(self.temporary.name) / "second"
        processor.process_inputs(RAW_CAPTURE, OANDA_MATCHES, second)
        first_names = sorted(path.name for path in self.output.iterdir())
        second_names = sorted(path.name for path in second.iterdir())
        self.assertEqual(first_names, second_names)
        for name in first_names:
            self.assertEqual((self.output / name).read_bytes(), (second / name).read_bytes())
        self.assertEqual(digest(RAW_CAPTURE), self.raw_digest_before)


if __name__ == "__main__":
    unittest.main(verbosity=2)
