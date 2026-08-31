#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import pathlib
import sys
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from import_economic_event_release_actual import build_insert_sql  # noqa: E402
from prepare_pce_production_import import (  # noqa: E402
    MANIFEST_FIELDS,
    deterministic_manifest,
    manifest_rows,
    manifest_sha256,
    reconciliation_rows,
)
from release_actual_ingestion import (  # noqa: E402
    Consensus,
    EconomicEvent,
    ImportDecision,
    ReleaseActualCandidate,
)


def candidate(
    source_id: str = "bea:pce:december-2009",
    available: str = "2010-02-01T13:30:00Z",
    value: str = "0.2",
) -> ReleaseActualCandidate:
    return ReleaseActualCandidate(
        source_family="BEA",
        candidate_event_family="PCE",
        candidate_source_event_id=source_id,
        candidate_reference_period="December 2009",
        source_agency="BEA",
        source_observation_id=f"bea:pce:2009-12:initial:{source_id[-4:]}",
        publication_state="initial",
        revision_sequence=0,
        available_at=available,
        retrieved_at="2026-08-25T04:55:48Z",
        source_url="https://www.bea.gov/news/fixture",
        source_artifact_path="fixture/bea-pce.html",
        source_artifact_sha256="a" * 64,
        semantic_contract="bea_current_dollar_pce_mom_percent_v1",
        source_provenance={"provider": "BEA"},
        actual_raw=f"increased {value} percent",
        actual_value_kind="scalar",
        actual_value_low=value,
        actual_value_high=None,
        actual_canonical_value_low=value,
        actual_canonical_value_high=None,
        actual_unit="percent",
        actual_scale="1",
        actual_qualifier="m/m",
    )


def event(event_id: int = 20) -> EconomicEvent:
    return EconomicEvent(
        event_id, "BEA", "PCE", "2010-02-01T13:30:00Z",
        "bea:pce:december-2009", "https://www.bea.gov/news/fixture",
        "December 2009",
    )


class PceProductionReadinessTests(unittest.TestCase):
    def test_manifest_has_exact_migration_fields_order_and_stable_hash(self) -> None:
        later = dataclasses.replace(
            candidate("bea:pce:january-2010", "2010-03-01T13:30:00Z", "0"),
            candidate_reference_period="January 2010",
            source_observation_id="bea:pce:2010-01:initial:2010",
        )
        decisions = [
            ImportDecision(later, "matched", 21, None),
            ImportDecision(candidate(), "matched", 20, None),
        ]
        rows = manifest_rows(decisions)
        self.assertEqual(tuple(rows[0]), MANIFEST_FIELDS)
        self.assertEqual([row["economic_event_id"] for row in rows], [20, 21])
        first = deterministic_manifest(rows)
        second = deterministic_manifest(manifest_rows(list(reversed(decisions))))
        self.assertEqual(first, second)
        self.assertEqual(
            manifest_sha256(first),
            "2795e2691116ae91b9bc8fee1d554b86ae1681d1b79d298b2b1073cba02257d9",
        )

    def test_only_importable_rows_enter_manifest(self) -> None:
        row = candidate()
        decisions = [
            ImportDecision(row, "matched", 20, None),
            ImportDecision(row, "invalid_semantics", None, "source_agency_mismatch"),
            ImportDecision(row, "conflict", 20, "immutable_identity_conflict"),
        ]
        self.assertEqual(len(manifest_rows(decisions)), 1)

    def test_catalog_classifications_fail_closed(self) -> None:
        row = candidate()
        canonical = event()
        cases = [
            (ImportDecision(row, "matched", 20, None), [], "MISSING_CONSENSUS"),
            (ImportDecision(row, "matched", 20, None),
             [Consensus(20, "scalar", "percent", "y/y")],
             "INCOMPATIBLE_CONSENSUS"),
            (ImportDecision(row, "matched", 20, None),
             [Consensus(20, "scalar", "percent", "m/m")], "READY"),
            (ImportDecision(row, "unmatched", None, "source_event_id_not_found"),
             [], "MISSING_EVENT"),
            (ImportDecision(row, "invalid_semantics", None, "source_agency_mismatch"),
             [], "IDENTITY_MISMATCH"),
        ]
        for decision, consensus, expected in cases:
            with self.subTest(expected=expected):
                actual = reconciliation_rows([decision], [canonical], consensus)
                self.assertEqual(actual[0]["classification"], expected)
                self.assertEqual(
                    actual[0]["import_eligible"],
                    decision.decision in {"matched", "duplicate_identical"},
                )

    def test_sql_is_exact_append_only_transaction(self) -> None:
        first = ImportDecision(candidate(), "matched", 20, None)
        later_candidate = dataclasses.replace(
            candidate("bea:pce:january-2010", "2010-03-01T13:30:00Z", "0"),
            candidate_reference_period="January 2010",
            source_observation_id="bea:pce:2010-01:initial:2010",
        )
        later = ImportDecision(later_candidate, "matched", 21, None)
        sql = build_insert_sql([later, first])
        self.assertEqual(sql, build_insert_sql([first, later]))
        upper = sql.upper()
        self.assertTrue(sql.startswith("BEGIN;\nINSERT INTO "))
        self.assertTrue(sql.endswith(";\nCOMMIT;\n"))
        self.assertNotIn(" UPDATE ", upper)
        self.assertNotIn(" DELETE ", upper)
        self.assertNotIn("ON CONFLICT", upper)
        self.assertNotIn("UPSERT", upper)


if __name__ == "__main__":
    unittest.main()
