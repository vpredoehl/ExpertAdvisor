#!/usr/bin/env python3
"""Focused deterministic tests for Phase-8 Weekly Claims evidence."""

from __future__ import annotations

import dataclasses
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from myfxbook_weekly_claims_consensus import (  # noqa: E402
    CatalogEvent,
    ParsedForecast,
    SnapshotArtifact,
    SnapshotIdentity,
    normalize_forecast,
    parse_snapshot,
    prepare,
)


REAL_CAPTURE = (
    ROOT / "EconomicCalendar/raw/myfxbook/weekly_claims_wayback/"
    "20250605063750-category-initial-jobless-claims.html"
)
REAL_URL = (
    "https://www.myfxbook.com/forex-economic-calendar/category/"
    "initial-jobless-claims"
)


def event(event_id: int = 10, release: str = "2025-06-05T12:30:00.000000Z"):
    return CatalogEvent(
        event_id,
        release,
        "dol_eta:usdl-25-fixture-nat",
        "week ending 2025-05-31",
        release[:10],
        "https://oui.doleta.gov/press/2025/fixture.pdf",
    )


def artifact(
    capture: str = "20250605063750",
    forecast: str | None = "235K",
    actual: str | None = None,
    release: str = "2025-06-05T12:30:00.000000Z",
):
    identity = SnapshotIdentity(capture, REAL_URL)
    return SnapshotArtifact(
        identity,
        identity.available_at,
        "2026-09-05T17:00:00.000000Z",
        "EconomicCalendar/raw/myfxbook/weekly_claims_wayback/fixture.html",
        "a" * 64,
        (ParsedForecast(278697, release, "May/31", forecast, actual),),
    )


class MyfxbookWeeklyClaimsConsensusTests(unittest.TestCase):
    def test_real_pre_release_snapshot(self):
        parsed = parse_snapshot(
            REAL_CAPTURE.read_bytes(),
            SnapshotIdentity("20250605063750", REAL_URL),
            "2026-09-05T17:00:00.000000Z",
            REAL_CAPTURE.relative_to(ROOT).as_posix(),
        )
        self.assertEqual(parsed.provider_observed_at,
                         "2025-06-05T06:37:51.100000Z")
        self.assertEqual(parsed.forecast_available_at,
                         "2025-06-05T06:37:51.100000Z")
        self.assertEqual(len(parsed.rows), 7)
        row = parsed.rows[0]
        self.assertEqual(row.myfxbook_event_id, 278697)
        self.assertEqual(row.release_at, "2025-06-05T12:30:00.000000Z")
        self.assertEqual(row.reference_period, "May/31")
        self.assertEqual(row.forecast_raw, "235K")
        self.assertIsNone(row.actual_raw)

    def test_supported_count_spellings_normalize_to_scale_one_count(self):
        for raw in ("235K", "235k", "235,000", "235000"):
            with self.subTest(raw=raw):
                self.assertEqual(normalize_forecast(raw), ("235000", "235000"))
        with self.assertRaisesRegex(ValueError, "not_count"):
            normalize_forecast("235.5")
        with self.assertRaisesRegex(ValueError, "not_count"):
            normalize_forecast("235M")

    def test_snapshot_identity_and_clock_fail_closed(self):
        for url in (
            "https://example.com/forex-economic-calendar",
            "https://www.myfxbook.com/forex-economic-calendar/other",
            REAL_URL + "?mutable=true",
        ):
            with self.subTest(url=url), self.assertRaisesRegex(
                    ValueError, "original_url_invalid"):
                SnapshotIdentity("20250605063750", url)
        with self.assertRaises(ValueError):
            SnapshotIdentity("2025060506375", REAL_URL)

        html = REAL_CAPTURE.read_bytes().replace(
            b"1749105471100", b"1749305471100", 1
        )
        with self.assertRaisesRegex(ValueError, "provider_clock_conflict"):
            parse_snapshot(
                html, SnapshotIdentity("20250605063750", REAL_URL),
                "2026-09-05T17:00:00.000000Z", "fixture.html"
            )
        with self.assertRaisesRegex(ValueError, "retrieval_predates"):
            parse_snapshot(
                REAL_CAPTURE.read_bytes(),
                SnapshotIdentity("20250605063750", REAL_URL),
                "2025-06-05T06:00:00.000000Z", "fixture.html"
            )

    def test_pre_release_mapping_and_raw_retention(self):
        eligible, rejected = prepare([event()], [artifact()])
        self.assertEqual(rejected, [])
        self.assertEqual(len(eligible), 1)
        row = eligible[0]
        self.assertEqual(row["forecast_raw"], "235K")
        self.assertEqual(row["forecast_value_low"], "235000")
        self.assertEqual(row["forecast_canonical_value_low"], "235000")
        self.assertEqual(row["forecast_unit"], "count")
        self.assertEqual(row["forecast_scale"], "1")
        self.assertEqual(row["forecast_qualifier"], "")
        self.assertEqual(row["pre_release_actual_raw"], "")

    def test_post_release_missing_actual_and_mapping_fail_closed(self):
        cases = (
            (artifact(capture="20250605130000"), "archive_not_pre_release"),
            (artifact(forecast=None), "forecast_missing"),
            (artifact(actual="236K"), "post_release_actual_present"),
            (artifact(release="2025-06-12T12:30:00.000000Z"),
             "unmapped_release_timestamp"),
        )
        for source, decision in cases:
            with self.subTest(decision=decision):
                eligible, rejected = prepare([event()], [source])
                self.assertEqual(eligible, [])
                self.assertEqual([row["decision"] for row in rejected],
                                 [decision])

        duplicate = dataclasses.replace(event(), economic_event_id=11)
        eligible, rejected = prepare([event(), duplicate], [artifact()])
        self.assertEqual(eligible, [])
        self.assertEqual(rejected[0]["decision"],
                         "ambiguous_release_timestamp")

    def test_later_post_release_evidence_does_not_replace_forecast(self):
        eligible, rejected = prepare(
            [event()],
            [artifact(), artifact(capture="20250605130000", forecast="999K")],
        )
        self.assertEqual(eligible[0]["forecast_canonical_value_low"], "235000")
        self.assertEqual(rejected[0]["decision"], "archive_not_pre_release")


if __name__ == "__main__":
    unittest.main()
