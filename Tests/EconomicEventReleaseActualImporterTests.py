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
    BeaArtifact,
    CensusArtifact,
    Consensus,
    EconomicEvent,
    ReleaseActualCandidate,
    coverage_audit,
    deterministic_json_lines,
    extract_bea_candidates,
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


def bea_artifact(
    name: str,
    family: str,
    reference: str,
    available: str,
    source_id: str,
) -> BeaArtifact:
    path = FIXTURES / name
    return BeaArtifact(
        event_family=family,
        reference_period=reference,
        source_url="https://www.bea.gov/news/fixture",
        path=path,
        repository_path=path.relative_to(ROOT).as_posix(),
        source_event_id=source_id,
        available_at=available,
        retrieved_at=RETRIEVED_AT,
        archive_commit="8a740a78a8c8ff10c8afd04c2d1c6eb6dfbbbb86",
        sha256=sha256_file(path),
        title=reference,
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

        observation = self.initial.provenance_observation_values(1)
        self.assertEqual(observation["source_publication_at"], self.initial.available_at)
        self.assertEqual(observation["observed_at"], self.initial.retrieved_at)
        self.assertEqual(observation["source_publication_time_status"], "exact")
        self.assertEqual(observation["availability_proof"], "source_publication")
        self.assertEqual(observation["source_native_event_id"],
                         self.initial.candidate_source_event_id)
        self.assertIn(self.initial.source_artifact_sha256,
                      str(observation["evidence_key"]))

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
        self.assertIn("INSERT INTO economic_event_actual_observation", sql)
        self.assertIn("source_publication_at", sql)
        self.assertIn("observed_at", sql)
        self.assertNotIn("UPDATE", sql)
        self.assertNotIn("ON CONFLICT", sql)
        self.assertTrue(sql.startswith("BEGIN;"))
        self.assertTrue(sql.endswith("COMMIT;\n"))


class BeaReleaseActualImporterTests(unittest.TestCase):
    def test_gdp_advance_is_initial_and_second_is_only_revision(self) -> None:
        initial_artifact = bea_artifact(
            "bea_gdp_advance.txt", "GDP", "Q4 2009 Advance",
            "2010-01-29T13:30:00Z", "bea:gdp:q4-2009-advance",
        )
        revision_artifact = bea_artifact(
            "bea_gdp_second.txt", "GDP", "Q4 2009 Second",
            "2010-02-26T13:30:00Z", "bea:gdp:q4-2009-second",
        )
        identity = {"Q4 2009": ("Q4 2009 Advance", "bea:gdp:q4-2009-advance")}
        initials, rejected = extract_bea_candidates(initial_artifact, identity)
        revisions, revision_rejected = extract_bea_candidates(revision_artifact, identity)
        self.assertEqual(rejected + revision_rejected, [])
        self.assertEqual(
            (initials[0].publication_state, initials[0].revision_sequence,
             initials[0].actual_value_low, initials[0].actual_qualifier),
            ("initial", 0, "5.7", None),
        )
        self.assertEqual(
            (revisions[0].publication_state, revisions[0].revision_sequence,
             revisions[0].candidate_source_event_id,
             revisions[0].candidate_reference_period,
             revisions[0].actual_value_low),
            ("revision", 1, "bea:gdp:q4-2009-advance", "Q4 2009 Advance", "5.9"),
        )

    def test_gdp_revision_without_advance_provenance_fails_closed(self) -> None:
        artifact = bea_artifact(
            "bea_gdp_second.txt", "GDP", "Q4 2009 Second",
            "2010-02-26T13:30:00Z", "bea:gdp:q4-2009-second",
        )
        rows, rejected = extract_bea_candidates(artifact, {})
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "missing_initial_provenance")

    def test_pce_selects_nominal_expenditures_not_price_index(self) -> None:
        artifact = bea_artifact(
            "bea_pce_initial.txt", "PCE", "December 2009",
            "2010-02-01T13:30:00Z", "bea:pce:december-2009",
        )
        rows, rejected = extract_bea_candidates(artifact, {})
        self.assertEqual(rejected, [])
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0].actual_value_low, "0.2")
        self.assertEqual(rows[0].actual_qualifier, "m/m")
        self.assertIn("current_dollar_personal_consumption_expenditures",
                      rows[0].source_provenance["headline_statistic"])

    def test_bea_wrong_unit_qualifier_and_authority_are_rejected(self) -> None:
        artifact = bea_artifact(
            "bea_pce_initial.txt", "PCE", "December 2009",
            "2010-02-01T13:30:00Z", "bea:pce:december-2009",
        )
        row = extract_bea_candidates(artifact, {})[0][0]
        with self.assertRaisesRegex(ValueError, "bea_semantics_invalid"):
            dataclasses.replace(row, actual_unit="count")
        with self.assertRaisesRegex(ValueError, "bea_semantics_invalid"):
            dataclasses.replace(row, actual_qualifier="y/y")
        with self.assertRaisesRegex(ValueError, "source_url_not_authoritative"):
            dataclasses.replace(row, source_url="https://example.com/release")

    def test_bea_matching_preserves_sha_causality_and_is_deterministic(self) -> None:
        artifact = bea_artifact(
            "bea_pce_initial.txt", "PCE", "December 2009",
            "2010-02-01T13:30:00Z", "bea:pce:december-2009",
        )
        row = extract_bea_candidates(artifact, {})[0][0]
        canonical = EconomicEvent(
            20, "BEA", "PCE", "2010-02-01T13:30:00Z",
            "bea:pce:december-2009", "https://www.bea.gov/news/fixture",
            "December 2009",
        )
        decisions = match_candidates([row], [canonical])
        self.assertEqual(decisions[0].decision, "matched")
        self.assertEqual(row.source_artifact_sha256, artifact.sha256)
        self.assertGreater(row.retrieved_at, row.available_at)
        self.assertEqual(
            deterministic_json_lines(decisions, []),
            deterministic_json_lines(decisions, []),
        )


if __name__ == "__main__":
    unittest.main()
