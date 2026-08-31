#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from prepare_multi_family_initial_actual_import import (  # noqa: E402
    CLASSIFICATIONS,
    _decision_classification,
    deterministic_json_lines,
)
from release_actual_ingestion import ImportDecision, ReleaseActualCandidate  # noqa: E402


def candidate(state: str = "initial", sequence: int = 0) -> ReleaseActualCandidate:
    return ReleaseActualCandidate(
        source_family="BEA",
        candidate_event_family="GDP",
        candidate_source_event_id="bea:gdp:q4-2009-advance",
        candidate_reference_period="Q4 2009 Advance",
        source_agency="BEA",
        source_observation_id=f"bea:gdp:q4-2009:{state}:{sequence}:fixture",
        publication_state=state,
        revision_sequence=sequence,
        available_at="2010-01-29T13:30:00Z",
        retrieved_at="2026-08-25T04:55:48Z",
        source_url="https://www.bea.gov/news/fixture",
        source_artifact_path="fixture/bea-gdp.html",
        source_artifact_sha256="a" * 64,
        semantic_contract="bea_real_gdp_annualized_quarterly_percent_v1",
        source_provenance={"provider": "BEA"},
        actual_raw="increased at an annual rate of 5.7 percent",
        actual_value_kind="scalar",
        actual_value_low="5.7",
        actual_value_high=None,
        actual_canonical_value_low="5.7",
        actual_canonical_value_high=None,
        actual_unit="percent",
        actual_scale="1",
        actual_qualifier=None,
    )


class MultiFamilyInitialActualReadinessTests(unittest.TestCase):
    def test_initial_revision_and_duplicate_classifications(self) -> None:
        self.assertEqual(
            _decision_classification(ImportDecision(candidate(), "matched", 1, None)),
            "importable_initial",
        )
        self.assertEqual(
            _decision_classification(
                ImportDecision(candidate("revision", 1), "matched", 1, None)
            ),
            "revision_only",
        )
        self.assertEqual(
            _decision_classification(
                ImportDecision(candidate(), "duplicate_identical", 1, None)
            ),
            "duplicate_identical",
        )

    def test_fail_closed_match_classifications(self) -> None:
        cases = (
            ("ambiguous", "duplicate_source_event_id", "ambiguous_match"),
            ("unmatched", "source_event_id_not_found", "unmatched"),
            ("invalid_semantics", "source_agency_mismatch", "source_agency_mismatch"),
            ("invalid_semantics", "available_at_predates_event", "causal_violation"),
            ("invalid_semantics", "canonical_event_identity_mismatch", "unsupported_semantics"),
        )
        for decision, reason, expected in cases:
            with self.subTest(expected=expected):
                self.assertEqual(
                    _decision_classification(
                        ImportDecision(candidate(), decision, None, reason)
                    ),
                    expected,
                )
                self.assertIn(expected, CLASSIFICATIONS)

    def test_conflicting_duplicate_requires_manual_resolution(self) -> None:
        with self.assertRaisesRegex(ValueError, "manual_resolution"):
            _decision_classification(
                ImportDecision(
                    candidate(), "conflict", 1, "immutable_identity_conflict"
                )
            )

    def test_json_lines_are_canonical(self) -> None:
        rows = [{"z": 1, "a": "x"}, {"z": 2, "a": "y"}]
        expected = '{"a":"x","z":1}\n{"a":"y","z":2}\n'
        self.assertEqual(deterministic_json_lines(rows), expected)
        self.assertEqual(deterministic_json_lines(rows), deterministic_json_lines(rows))


if __name__ == "__main__":
    unittest.main()
