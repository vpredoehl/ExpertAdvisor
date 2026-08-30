#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import unittest
from decimal import Decimal


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from import_economic_event_release_actual import build_insert_sql  # noqa: E402
from release_actual_ingestion import (  # noqa: E402
    CensusArtifact,
    Consensus,
    EconomicEvent,
    ReleaseActualCandidate,
    coverage_audit,
    deterministic_json_lines,
    extract_census_candidates,
    match_candidates,
    read_artifact_text,
    sha256_file,
)


FIXTURES = ROOT / "Tests" / "fixtures" / "economic_calendar" / "release_actual"
RETRIEVED_AT = "2026-08-25T04:55:48.000000Z"


def artifact(name: str, family: str, reference: str, available: str, source_id: str) -> CensusArtifact:
    path = FIXTURES / name
    return CensusArtifact(
        event_family=family,
        reference_period=reference,
        source_url="https://www.census.gov/economic-indicators/fixture.pdf",
        path=path,
        repository_path=path.relative_to(ROOT).as_posix(),
        source_event_id=source_id,
        available_at=available,
        retrieved_at=RETRIEVED_AT,
        archive_commit="8a740a78a8c8ff10c8afd04c2d1c6eb6dfbbbb86",
        sha256=sha256_file(path),
    )


def event(
    event_id: int = 1,
    family: str = "RETAIL_SALES",
    reference: str = "January 2010",
    source_id: str = "census:retail-sales-2010-01",
    agency: str = "CENSUS",
    timestamp: str = "2010-02-12T13:30:00Z",
) -> EconomicEvent:
    return EconomicEvent(
        event_id, agency, family, timestamp, source_id,
        "https://www.census.gov/economic-indicators/fixture.pdf", reference,
    )


class ReleaseActualImporterTests(unittest.TestCase):
    def setUp(self) -> None:
        self.retail_artifact = artifact(
            "census_retail_initial_revision.txt", "RETAIL_SALES", "January 2010",
            "2010-02-12T13:30:00Z", "census:retail-sales-2010-01",
        )
        source_ids = {
            ("RETAIL_SALES", "January 2010"): "census:retail-sales-2010-01",
            ("RETAIL_SALES", "December 2009"): "census:retail-sales-2009-12",
        }
        self.candidates, rejected = extract_census_candidates(
            self.retail_artifact, source_ids
        )
        self.assertEqual(rejected, [])
        self.initial = next(row for row in self.candidates if row.publication_state == "initial")
        self.revision = next(row for row in self.candidates if row.publication_state == "revision")

    def test_valid_initial_and_revision_contract(self) -> None:
        self.assertEqual(self.initial.revision_sequence, 0)
        self.assertEqual(self.initial.actual_value_low, "0.5")
        self.assertEqual(self.initial.actual_canonical_value_low, "0.5")
        self.assertEqual(self.initial.actual_unit, "percent")
        self.assertEqual(self.initial.actual_qualifier, "m/m")
        self.assertEqual(self.revision.revision_sequence, 1)
        self.assertEqual(self.revision.actual_value_low, "-0.1")
        self.assertEqual(self.revision.candidate_reference_period, "December 2009")

    def test_durable_card_initial_and_revision(self) -> None:
        item = artifact(
            "census_durable_initial_revision.txt", "DURABLE_GOODS", "January 2020",
            "2020-02-27T13:30:00Z", "census:durable-goods-2020-01",
        )
        rows, rejected = extract_census_candidates(item, {
            ("DURABLE_GOODS", "January 2020"): "census:durable-goods-2020-01",
            ("DURABLE_GOODS", "December 2019"): "census:durable-goods-2019-12",
        })
        self.assertEqual(rejected, [])
        self.assertEqual([(row.publication_state, row.actual_value_low) for row in rows], [
            ("initial", "-0.2"), ("revision", "2.9"),
        ])

    def test_revision_cannot_replace_initial(self) -> None:
        initial_event = event(
            event_id=2, reference="December 2009",
            source_id="census:retail-sales-2009-12",
            timestamp="2010-01-14T13:30:00Z",
        )
        persisted_initial = self.initial.persisted_values(2)
        persisted_initial.update({
            "economic_event_release_actual_id": 9,
            "source_observation_id": "preexisting-initial",
            "revision_sequence": 1,
            "publication_state": "revision",
        })
        decision = match_candidates([self.revision], [initial_event], [persisted_initial])[0]
        self.assertEqual(decision.decision, "conflict")

    def test_duplicate_identical_and_idempotent_repeat(self) -> None:
        matched = match_candidates([self.initial], [event()])[0]
        self.assertEqual(matched.decision, "matched")
        existing = self.initial.persisted_values(1)
        existing["economic_event_release_actual_id"] = 100
        repeated = match_candidates([self.initial], [event()], [existing])[0]
        self.assertEqual(repeated.decision, "duplicate_identical")
        self.assertEqual(build_insert_sql([repeated]), "BEGIN;\nCOMMIT;\n")

    def test_conflicting_duplicate(self) -> None:
        existing = self.initial.persisted_values(1)
        existing["economic_event_release_actual_id"] = 101
        existing["actual_raw"] = "0.6%"
        self.assertEqual(
            match_candidates([self.initial], [event()], [existing])[0].decision,
            "conflict",
        )

    def test_source_agency_mismatch(self) -> None:
        decision = match_candidates([self.initial], [event(agency="BEA")])[0]
        self.assertEqual(decision.decision, "invalid_semantics")
        self.assertEqual(decision.rejection_reason, "source_agency_mismatch")

    def test_unmatched_and_ambiguous(self) -> None:
        self.assertEqual(match_candidates([self.initial], [])[0].decision, "unmatched")
        duplicates = [event(1), event(2)]
        self.assertEqual(match_candidates([self.initial], duplicates)[0].decision, "ambiguous")

    def test_unsupported_document_and_semantics(self) -> None:
        unsupported = artifact(
            "unsupported.txt", "RETAIL_SALES", "January 2010",
            "2010-02-12T13:30:00Z", "census:retail-sales-2010-01",
        )
        rows, rejected = extract_census_candidates(unsupported, {})
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "missing_initial_provenance")
        with self.assertRaisesRegex(ValueError, "unsupported_document_type"):
            read_artifact_text(FIXTURES / "not-a-document.html")

    def test_unsupported_unit_scaling_and_qualifier_validation(self) -> None:
        with self.assertRaisesRegex(ValueError, "census_semantics_invalid"):
            dataclasses.replace(self.initial, actual_unit="count")
        with self.assertRaisesRegex(ValueError, "raw_canonical_scaling_invalid"):
            dataclasses.replace(self.initial, actual_scale="1000")
        with self.assertRaisesRegex(ValueError, "census_semantics_invalid"):
            dataclasses.replace(self.initial, actual_qualifier="y/y")

    def test_sha_availability_retrieval_and_qualifier_preserved(self) -> None:
        self.assertEqual(self.initial.source_artifact_sha256, self.retail_artifact.sha256)
        self.assertEqual(self.initial.available_at, "2010-02-12T13:30:00Z")
        self.assertNotEqual(self.initial.available_at, self.initial.retrieved_at)
        self.assertEqual(self.initial.source_provenance["retrieved_at_basis"],
                         "first_git_archive_admission_committer_timestamp")
        self.assertEqual(self.initial.actual_qualifier, "m/m")

    def test_causal_available_at_rejected_if_before_event(self) -> None:
        later_event = event(timestamp="2010-02-12T13:30:00.000001Z")
        decision = match_candidates([self.initial], [later_event])[0]
        self.assertEqual(decision.decision, "invalid_semantics")
        self.assertEqual(decision.rejection_reason, "available_at_predates_event")

    def test_deterministic_dry_run_and_coverage_without_database_or_network(self) -> None:
        decisions = match_candidates([self.initial], [event()])
        first = deterministic_json_lines(decisions, [])
        second = deterministic_json_lines(decisions, [])
        self.assertEqual(first, second)
        self.assertNotIn("urllib", first)

        consensus = [Consensus(1, "scalar", "percent", "m/m")]
        first_audit = coverage_audit([event()], consensus, decisions, [])
        second_audit = coverage_audit([event()], consensus, decisions, [])
        self.assertEqual(first_audit, second_audit)
        summary = first_audit["families"]["RETAIL_SALES"]
        self.assertEqual(summary["certified_initial_actuals"], 1)
        self.assertEqual(summary["jointly_usable_surprise_rows"], 1)
        self.assertEqual(summary["usable_surprise_coverage_percent"], 100.0)

    def test_insert_sql_is_append_only(self) -> None:
        decision = match_candidates([self.initial], [event()])[0]
        sql = build_insert_sql([decision])
        self.assertIn("INSERT INTO economic_event_release_actual", sql)
        self.assertNotIn("UPDATE", sql)
        self.assertNotIn("ON CONFLICT", sql)
        self.assertTrue(sql.startswith("BEGIN;"))
        self.assertTrue(sql.endswith("COMMIT;\n"))


if __name__ == "__main__":
    unittest.main()
