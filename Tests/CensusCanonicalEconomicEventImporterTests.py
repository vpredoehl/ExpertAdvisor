#!/usr/bin/env python3

from __future__ import annotations

import csv
import pathlib
import re
import subprocess
import sys
import unittest
from collections import Counter
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

import import_census_events as importer
import audit_economic_event_coverage as coverage_audit


CANONICAL = (
    ROOT / "EconomicCalendar/raw/census/census_events_extracted.csv"
)


class CensusCanonicalEconomicEventImporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        with CANONICAL.open(newline="", encoding="utf-8") as source:
            cls.raw_rows = list(csv.DictReader(source))
        cls.source_rows, cls.prepared = importer.load_and_prepare(CANONICAL)

    def test_csv_row_parsing_and_exact_counts(self) -> None:
        self.assertEqual(len(self.source_rows), 399)
        self.assertEqual(len(self.prepared), 399)
        self.assertEqual(
            Counter(row["event_family"] for row in self.prepared),
            Counter({"RETAIL_SALES": 200, "DURABLE_GOODS": 199}),
        )

    def test_canonical_field_mapping(self) -> None:
        mapped = importer.prepare_row(dict(self.raw_rows[0]))
        self.assertEqual(mapped["currency"], "USD")
        self.assertEqual(mapped["source_agency"], "CENSUS")
        self.assertEqual(mapped["event_family"], "RETAIL_SALES")
        self.assertEqual(mapped["event_timestamp_utc"], "2010-01-14T13:30:00+00:00")
        self.assertEqual(mapped["source_release_date"], "2010-01-14")
        self.assertEqual(mapped["source_release_time"], "08:30:00")
        self.assertEqual(mapped["source_timezone"], "America/New_York")
        self.assertEqual(mapped["reference_period"], "December 2009")
        self.assertEqual(mapped["event_importance"], 3)
        self.assertEqual(mapped["historical_time_confidence"], "exact")
        self.assertTrue(mapped["title"])
        self.assertTrue(mapped["timestamp_source_url"])

    def test_source_event_id_is_stable_and_unique(self) -> None:
        expected = "census:retail-sales-2009-12"
        self.assertEqual(
            importer.source_event_id("RETAIL_SALES", 2009, 12),
            expected,
        )
        self.assertEqual(importer.prepare_row(dict(self.raw_rows[0]))["source_event_id"], expected)
        self.assertEqual(importer.prepare_row(dict(self.raw_rows[0]))["source_event_id"], expected)
        identities = {row["source_event_id"] for row in self.prepared}
        self.assertEqual(len(identities), 399)

    def test_canonical_keys_are_stable_and_unique(self) -> None:
        first = self.prepared[0]
        self.assertEqual(
            importer.canonical_key(first),
            ("CENSUS", "RETAIL_SALES", "2010-01-14T13:30:00+00:00"),
        )
        self.assertEqual(
            len({importer.canonical_key(row) for row in self.prepared}),
            399,
        )

    def assert_row_rejected(self, field: str, value: str, diagnostic: str) -> None:
        row = dict(self.raw_rows[0])
        row[field] = value
        with self.assertRaisesRegex(ValueError, diagnostic):
            importer.prepare_row(row)

    def test_rejects_unexpected_source_agency(self) -> None:
        self.assert_row_rejected("source_agency", "BEA", "unexpected source_agency")

    def test_rejects_unexpected_event_family(self) -> None:
        self.assert_row_rejected("event_family", "GDP", "unexpected event_family")

    def test_rejects_missing_timestamp(self) -> None:
        self.assert_row_rejected("event_timestamp_utc", "", "missing event_timestamp_utc")

    def test_rejects_invalid_or_timezone_naive_timestamp(self) -> None:
        for value, diagnostic in (
            ("not-a-timestamp", "invalid event_timestamp_utc"),
            ("2010-01-14T13:30:00", "timezone-naive"),
        ):
            with self.subTest(value=value):
                self.assert_row_rejected("event_timestamp_utc", value, diagnostic)

    def test_rejects_missing_provenance_title_and_reference_fields(self) -> None:
        for field in (
            "title",
            "reference_period",
            "url",
            "timestamp_source_url",
            "source_local_date",
            "source_local_time",
        ):
            with self.subTest(field=field):
                self.assert_row_rejected(field, "", f"missing {field}")

    def test_rejects_unsupported_parser_and_provenance_states(self) -> None:
        self.assert_row_rejected(
            "timestamp_parser",
            "best_guess",
            "unexpected timestamp_parser",
        )
        self.assert_row_rejected(
            "discovery_source",
            "search_result",
            "unexpected discovery_source",
        )

    def test_dry_run_database_access_is_read_only(self) -> None:
        sql = importer.build_dry_run_sql()
        self.assertIn("BEGIN READ ONLY", sql)
        for statement in ("INSERT", "UPDATE", "DELETE", "CREATE", "ALTER", "DROP", "TRUNCATE"):
            self.assertIsNone(re.search(rf"\b{statement}\b", sql, re.IGNORECASE))

        completed = subprocess.CompletedProcess([], 0, "", "")
        config = importer.DatabaseConfig("test")
        with mock.patch.object(importer, "run_psql", return_value=completed) as run:
            self.assertEqual(importer.load_database_rows(config), [])
        self.assertTrue(run.call_args.kwargs["read_only"])

    def test_idempotent_database_dispositions(self) -> None:
        empty = importer.classify_database_effects(self.prepared, [])
        self.assertEqual(empty.would_insert, 399)
        self.assertEqual(empty.unchanged, 0)
        self.assertEqual(empty.would_update, 0)
        self.assertEqual(empty.rejected, 0)

        existing = [
            {field: row[field] for field in importer.TARGET_FIELDS}
            for row in self.prepared
        ]
        rerun = importer.classify_database_effects(self.prepared, existing)
        self.assertEqual(rerun.would_insert, 0)
        self.assertEqual(rerun.unchanged, 399)
        self.assertEqual(rerun.would_update, 0)
        self.assertEqual(rerun.rejected, 0)

    def test_immutable_existing_difference_is_rejected_not_updated(self) -> None:
        existing = {
            field: self.prepared[0][field]
            for field in importer.TARGET_FIELDS
        }
        existing["source_url"] = "https://www.census.gov/conflicting"
        effects = importer.classify_database_effects([self.prepared[0]], [existing])
        self.assertEqual(effects.rejected, 1)
        self.assertEqual(effects.would_update, 0)

    def test_commit_sql_is_atomic_and_uses_canonical_conflict_guards(self) -> None:
        sql = importer.build_commit_sql(self.prepared)
        self.assertRegex(sql, r"(?s)^\s*BEGIN;.*COMMIT;\s*$")
        self.assertIn("SET TRANSACTION ISOLATION LEVEL SERIALIZABLE", sql)
        self.assertIn("ON CONFLICT DO NOTHING", sql)
        self.assertNotRegex(sql, r"\bUPDATE\b")
        self.assertNotRegex(sql, r"\bDELETE\b")

    def test_combined_audit_includes_census_release_year_contract(self) -> None:
        self.assertIn(
            ("CENSUS", "RETAIL_SALES"),
            coverage_audit.EXPECTED_IMPORTED_FAMILIES,
        )
        self.assertIn(
            ("CENSUS", "DURABLE_GOODS"),
            coverage_audit.EXPECTED_IMPORTED_FAMILIES,
        )
        self.assertFalse(any(
            row["source_agency"] == "CENSUS"
            for row in coverage_audit.PLANNED_REMAINING_FAMILIES
        ))

        for family, partial_count in (
            ("RETAIL_SALES", 9),
            ("DURABLE_GOODS", 8),
        ):
            for year in range(2010, 2025):
                expected, _, _ = coverage_audit.expected_year_count(
                    "CENSUS", family, year
                )
                self.assertEqual(expected, 12)
            self.assertEqual(
                coverage_audit.expected_year_count("CENSUS", family, 2025)[0],
                11,
            )
            expected, status, note = coverage_audit.expected_year_count(
                "CENSUS", family, 2026
            )
            self.assertEqual(expected, partial_count)
            self.assertEqual(status, "documented_irregular")
            self.assertIn("Partial release year", note)

    def test_combined_audit_keeps_existing_source_expectations(self) -> None:
        self.assertEqual(
            coverage_audit.expected_year_count("BLS", "CPI", 2025)[0],
            11,
        )
        self.assertEqual(
            coverage_audit.expected_year_count("BEA", "GDP", 2019)[0],
            11,
        )
        self.assertEqual(
            coverage_audit.expected_year_count("FEDERAL_RESERVE", "FOMC", 2020)[0],
            10,
        )


if __name__ == "__main__":
    unittest.main()
