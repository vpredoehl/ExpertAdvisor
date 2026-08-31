#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import json
import pathlib
import sys
import tempfile
import unittest


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))

from release_actual_ingestion import (  # noqa: E402
    BLS_CPI_SEMANTIC_CONTRACT,
    BLS_EMPLOYMENT_SEMANTIC_CONTRACT,
    BLS_JOLTS_SEMANTIC_CONTRACT,
    BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT,
    BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT,
    BlsArtifact,
    EconomicEvent,
    extract_bls_candidates,
    load_bls_artifacts,
    match_candidates,
    sha256_file,
)


FIXTURES = ROOT / "Tests/fixtures/economic_calendar/release_actual"
RETRIEVED_AT = "2026-08-31T12:00:00.000000Z"


def artifact(
    filename: str,
    family: str,
    reference: str,
    release_date: str,
    available_at: str,
) -> BlsArtifact:
    path = FIXTURES / filename
    stem = {
        "CPI": "cpi",
        "EMPLOYMENT": "empsit",
        "PPI": "ppi",
        "JOLTS": "jolts",
    }[family]
    identity = f"{stem}_{release_date.replace('-', '')}"
    return BlsArtifact(
        event_family=family,
        reference_period=reference,
        release_date=release_date,
        source_url=f"https://www.bls.gov/news.release/archives/{identity}.htm",
        canonical_source_url=f"https://www.bls.gov/schedule/{release_date[:4]}/home.htm",
        path=path,
        repository_path=path.relative_to(ROOT).as_posix(),
        source_event_id=f"bls:{family.lower()}:{release_date}:{reference.lower().replace(' ', '-')}",
        available_at=available_at,
        retrieved_at=RETRIEVED_AT,
        first_archive_admission_at=RETRIEVED_AT,
        sha256=sha256_file(path),
        source_release_identity=identity,
        release_timestamp_evidence={
            "CPI": "8:30 a.m. (EST) January 11, 2019",
            "EMPLOYMENT": "8:30 a.m. (EST) Friday, January 4, 2013",
            "PPI": (
                "8:30 a.m. (EST), Wednesday, February 19, 2014"
                if release_date == "2014-02-19"
                else "8:30 a.m. (EST), Wednesday, January 15, 2014"
            ),
            "JOLTS": "10:00 a.m. (ET) Wednesday, January 3, 2024",
        }[family],
    )


def event_for(
    item: BlsArtifact,
    event_id: int,
    *,
    reference: str | None = None,
    timestamp: str | None = None,
) -> EconomicEvent:
    return EconomicEvent(
        economic_event_id=event_id,
        source_agency="BLS",
        event_family=item.event_family,
        event_timestamp_utc=timestamp or item.available_at,
        source_event_id=item.source_event_id,
        source_url=item.canonical_source_url,
        reference_period=reference or item.reference_period,
    )


class BlsReleaseActualImporterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.artifacts = [
            artifact(
                "bls_cpi_initial.txt", "CPI", "December 2018",
                "2019-01-11", "2019-01-11T13:30:00Z",
            ),
            artifact(
                "bls_employment_initial.txt", "EMPLOYMENT", "December 2012",
                "2013-01-04", "2013-01-04T13:30:00Z",
            ),
            artifact(
                "bls_ppi_finished_goods_initial.txt", "PPI", "December 2013",
                "2014-01-15", "2014-01-15T13:30:00Z",
            ),
            artifact(
                "bls_ppi_final_demand_initial.txt", "PPI", "January 2014",
                "2014-02-19", "2014-02-19T13:30:00Z",
            ),
            artifact(
                "bls_jolts_initial.txt", "JOLTS", "November 2023",
                "2024-01-03", "2024-01-03T15:00:00Z",
            ),
        ]
        cls.candidates = []
        for item in cls.artifacts:
            rows, rejected = extract_bls_candidates(item)
            if rejected:
                raise AssertionError(rejected)
            if len(rows) != 1:
                raise AssertionError(rows)
            cls.candidates.append(rows[0])

    def test_valid_initial_statistic_units_scale_and_provenance(self) -> None:
        expected = [
            ("CPI", "-0.1", "-0.1", "percent", "1", "m/m", BLS_CPI_SEMANTIC_CONTRACT),
            ("EMPLOYMENT", "155", "155000", "count", "1000", None, BLS_EMPLOYMENT_SEMANTIC_CONTRACT),
            ("PPI", "-0.1", "-0.1", "percent", "1", "m/m", BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT),
            ("PPI", "0.2", "0.2", "percent", "1", "m/m", BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT),
            ("JOLTS", "8790", "8790000", "count", "1000", None, BLS_JOLTS_SEMANTIC_CONTRACT),
        ]
        for item, candidate, values in zip(self.artifacts, self.candidates, expected):
            with self.subTest(family=item.event_family, reference=item.reference_period):
                self.assertEqual(candidate.publication_state, "initial")
                self.assertEqual(candidate.revision_sequence, 0)
                self.assertEqual(
                    (
                        candidate.candidate_event_family,
                        candidate.actual_value_low,
                        candidate.actual_canonical_value_low,
                        candidate.actual_unit,
                        candidate.actual_scale,
                        candidate.actual_qualifier,
                        candidate.semantic_contract,
                    ),
                    values,
                )
                self.assertEqual(candidate.available_at, item.available_at)
                self.assertEqual(candidate.source_artifact_sha256, item.sha256)
                self.assertEqual(candidate.source_provenance["artifact_sha256"], item.sha256)
                self.assertEqual(
                    candidate.source_provenance["reference_period"],
                    item.reference_period,
                )
                self.assertIn("release_identity_evidence", candidate.source_provenance)

    def test_revision_values_do_not_leak_into_initials(self) -> None:
        values = [row.actual_value_low for row in self.candidates]
        self.assertEqual(values, ["-0.1", "155", "-0.1", "0.2", "8790"])
        self.assertTrue(all(row.publication_state == "initial" for row in self.candidates))
        self.assertTrue(all(row.revision_sequence == 0 for row in self.candidates))

    def test_release_timestamp_proof_is_required_for_each_family(self) -> None:
        for item in self.artifacts:
            with self.subTest(family=item.event_family, reference=item.reference_period):
                invalid = dataclasses.replace(
                    item,
                    release_timestamp_evidence="8:30 a.m. January 1, 1999",
                )
                rows, rejected = extract_bls_candidates(invalid)
                self.assertEqual(rows, [])
                self.assertEqual(rejected[0].decision, "missing_initial_provenance")
                self.assertIn("release_timestamp", rejected[0].rejection_reason)

    def test_malformed_and_ambiguous_inputs_fail_closed_for_each_family(self) -> None:
        for index, item in enumerate(self.artifacts, start=1):
            with self.subTest(family=item.event_family, reference=item.reference_period):
                identity_only = (
                    f"{item.release_timestamp_evidence} "
                    f"{item.reference_period} Bureau of Labor Statistics"
                )
                rows, rejected = extract_bls_candidates(
                    item,
                    text_reader=lambda _: (identity_only, "malformed_fixture_v1"),
                )
                self.assertEqual(rows, [])
                self.assertEqual(rejected[0].decision, "invalid_semantics")

                candidate = self.candidates[index - 1]
                decisions = match_candidates(
                    [candidate],
                    [event_for(item, index * 10), event_for(item, index * 10 + 1)],
                )
                self.assertEqual(decisions[0].decision, "ambiguous")

    def test_reference_identity_duplicate_conflict_and_causality_for_each_family(self) -> None:
        for index, (item, candidate) in enumerate(
            zip(self.artifacts, self.candidates), start=1
        ):
            with self.subTest(family=item.event_family, reference=item.reference_period):
                canonical = event_for(item, index)
                self.assertEqual(match_candidates([candidate], [canonical])[0].decision, "matched")

                wrong_reference = event_for(item, index, reference="January 1900")
                wrong = match_candidates([candidate], [wrong_reference])[0]
                self.assertEqual(wrong.decision, "invalid_semantics")
                self.assertEqual(wrong.rejection_reason, "canonical_event_identity_mismatch")

                existing = candidate.persisted_values(index)
                existing["economic_event_release_actual_id"] = index
                duplicate = match_candidates([candidate], [canonical], [existing])[0]
                self.assertEqual(duplicate.decision, "duplicate_identical")

                conflict_row = dict(existing)
                conflict_row["actual_raw"] = "different"
                conflict = match_candidates([candidate], [canonical], [conflict_row])[0]
                self.assertEqual(conflict.decision, "conflict")

                later = event_for(
                    item,
                    index,
                    timestamp=candidate.available_at.replace("Z", ".000001Z"),
                )
                causal = match_candidates([candidate], [later])[0]
                self.assertEqual(causal.decision, "invalid_semantics")
                self.assertEqual(causal.rejection_reason, "available_at_predates_event")

    def test_reissued_employment_release_is_not_treated_as_initial(self) -> None:
        item = self.artifacts[1]
        text = item.path.read_text(encoding="utf-8").replace(
            "THE EMPLOYMENT SITUATION -- DECEMBER 2012",
            "THE EMPLOYMENT SITUATION -- DECEMBER 2012 "
            "(NOTE: BLS reissued this news release to correct prior data.)",
        )
        rows, rejected = extract_bls_candidates(
            item, text_reader=lambda _: (text, "reissued_fixture_v1")
        )
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "missing_initial_provenance")
        self.assertEqual(
            rejected[0].rejection_reason,
            "missing_initial_provenance:employment_release_reissued",
        )

    def test_manifest_sha_validation_and_exact_catalog_identity(self) -> None:
        source = self.artifacts[0]
        with tempfile.TemporaryDirectory(prefix="ea-bls-manifest-") as directory:
            root = pathlib.Path(directory)
            local = root / "EconomicCalendar/raw/bls/releases/cpi/cpi_01112019.htm"
            local.parent.mkdir(parents=True)
            local.write_bytes(source.path.read_bytes())
            manifest = root / "EconomicCalendar/raw/bls/manifest.jsonl"
            row = {
                "artifact_kind": "release",
                "bls_family": "CPI",
                "first_archive_admission_at": RETRIEVED_AT,
                "immutable_local_path": local.relative_to(root).as_posix(),
                "reference_period": source.reference_period,
                "release_date": source.release_date,
                "release_timestamp_evidence": source.release_timestamp_evidence,
                "retrieved_at": RETRIEVED_AT,
                "sha256": "0" * 64,
                "source_release_identity": source.source_release_identity,
                "source_url": source.source_url,
            }
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "bls_artifact_hash_mismatch"):
                load_bls_artifacts(root, manifest, [event_for(source, 1)])

            row["sha256"] = sha256_file(local)
            manifest.write_text(json.dumps(row) + "\n", encoding="utf-8")
            loaded = load_bls_artifacts(root, manifest, [event_for(source, 1)])
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].source_event_id, source.source_event_id)


if __name__ == "__main__":
    unittest.main()
