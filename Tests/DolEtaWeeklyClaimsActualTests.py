#!/usr/bin/env python3

from __future__ import annotations

import dataclasses
import hashlib
import json
import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "EconomicCalendar"))
sys.path.insert(0, str(ROOT / "Scripts"))

from authoritative_acquisition import Download  # noqa: E402
from acquire_dol_eta_weekly_claims_actuals import (  # noqa: E402
    _release_date as acquisition_release_date,
    acquire,
)
from dol_eta_weekly_claims_actual import (  # noqa: E402
    PARSER_VERSION,
    DolEtaArtifact,
    _url_release_date as parser_release_date,
    extract_dol_eta_candidates,
    load_dol_eta_artifacts,
    parse_dol_eta_weekly_claims_actual,
)
from import_economic_event_release_actual import build_insert_sql  # noqa: E402
from release_actual_ingestion import (  # noqa: E402
    DOL_ETA_WEEKLY_CLAIMS_SEMANTIC_CONTRACT,
    EconomicEvent,
    deterministic_json_lines,
    match_candidates,
)


FIXTURES = ROOT / "Tests/fixtures/economic_calendar/dol_eta"
RETRIEVED_AT = "2026-09-05T12:00:00.000000Z"


def sha256(path: pathlib.Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def artifact(name: str, release_date: str, url: str) -> DolEtaArtifact:
    path = FIXTURES / name
    return DolEtaArtifact(
        release_date=release_date,
        source_url=url,
        source_artifact_identity="fixture:" + name,
        source_path=path,
        source_repository_path=path.relative_to(ROOT).as_posix(),
        source_sha256=sha256(path),
        parser_path=path,
        parser_repository_path=path.relative_to(ROOT).as_posix(),
        parser_sha256=sha256(path),
        extractor="first_party_excerpt_v1",
        retrieved_at=RETRIEVED_AT,
    )


def event(
    event_id: int,
    reference: str,
    timestamp: str,
    source_id: str,
    url: str,
) -> EconomicEvent:
    return EconomicEvent(
        event_id,
        "DOL_ETA",
        "WEEKLY_CLAIMS",
        timestamp,
        source_id,
        url,
        reference,
    )


class DolEtaWeeklyClaimsActualParserTests(unittest.TestCase):
    def setUp(self) -> None:
        self.url = "https://oui.doleta.gov/press/2010/072210.asp"
        self.artifact = artifact("2010-07-22.html", "2010-07-22", self.url)
        self.current = event(
            20, "week ending 2010-07-17", "2010-07-22T12:30:00Z",
            "dol_eta:usdl-10-990-nat", self.url,
        )
        self.prior = event(
            19, "week ending 2010-07-10", "2010-07-15T12:30:00Z",
            "dol_eta:usdl-10-950-nat",
            "https://oui.doleta.gov/press/2010/071510.asp",
        )

    def test_initial_and_prior_revision_are_distinct(self) -> None:
        parsed = parse_dol_eta_weekly_claims_actual(self.artifact)
        self.assertEqual(parsed.publication_at, "2010-07-22T12:30:00.000000Z")
        self.assertEqual(parsed.reference_period, "week ending 2010-07-17")
        self.assertEqual(parsed.initial_value, "464000")
        self.assertIn("advance figure", parsed.initial_raw)
        self.assertEqual(parsed.revision_reference_period, "week ending 2010-07-10")
        self.assertEqual(parsed.revision_value, "427000")
        self.assertIn("revised figure", parsed.revision_raw or "")

        rows, rejected = extract_dol_eta_candidates(
            self.artifact, [self.current, self.prior]
        )
        self.assertEqual(rejected, [])
        self.assertEqual(
            [(row.publication_state, row.revision_sequence,
              row.candidate_source_event_id, row.actual_value_low) for row in rows],
            [
                ("initial", 0, self.current.source_event_id, "464000"),
                ("revision", 1, self.prior.source_event_id, "427000"),
            ],
        )
        initial = rows[0]
        self.assertEqual(initial.actual_unit, "count")
        self.assertEqual(initial.actual_scale, "1")
        self.assertIsNone(initial.actual_qualifier)
        self.assertEqual(
            initial.semantic_contract,
            DOL_ETA_WEEKLY_CLAIMS_SEMANTIC_CONTRACT,
        )
        self.assertEqual(initial.source_artifact_sha256, self.artifact.source_sha256)
        self.assertEqual(
            initial.source_provenance["publication_timestamp_basis"],
            "embedded_release_embargo_date_time_zone",
        )
        observation = initial.provenance_observation_values(self.current.economic_event_id)
        self.assertEqual(observation["source_publication_time_status"], "exact")
        self.assertEqual(observation["availability_proof"], "source_publication")
        self.assertEqual(observation["observation_kind"], "initial")

    def test_modern_explicit_revision_uses_to_value(self) -> None:
        url = "https://oui.doleta.gov/press/2025/010225.pdf"
        item = artifact("2025-01-02.txt", "2025-01-02", url)
        parsed = parse_dol_eta_weekly_claims_actual(item)
        self.assertEqual(parsed.initial_value, "211000")
        self.assertEqual(parsed.reference_period, "week ending 2024-12-28")
        self.assertEqual(parsed.revision_value, "220000")
        self.assertNotEqual(parsed.revision_value, "219000")

    def test_publication_proof_missing_or_contradictory_fails_closed(self) -> None:
        missing = dataclasses.replace(
            self.artifact, parser_path=FIXTURES / "2025-01-02.txt",
            parser_repository_path="Tests/fixtures/economic_calendar/dol_eta/2025-01-02.txt",
        )
        text = missing.parser_path.read_text().replace(
            "8:30 A.M. (Eastern) Thursday, January 2, 2025", "Thursday, January 2, 2025"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "missing.txt"
            path.write_text(text)
            with self.assertRaisesRegex(ValueError, "publication_time_unproven"):
                parse_dol_eta_weekly_claims_actual(
                    dataclasses.replace(missing, parser_path=path)
                )

        contradictory = self.artifact.parser_path.read_text().replace("(EDT)", "(EST)")
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "contradictory.html"
            path.write_text(contradictory)
            with self.assertRaisesRegex(ValueError, "timezone_contradiction"):
                parse_dol_eta_weekly_claims_actual(
                    dataclasses.replace(self.artifact, parser_path=path)
                )

    def test_wrong_title_and_current_value_fail_closed(self) -> None:
        text = self.artifact.parser_path.read_text()
        cases = (
            (text.replace("UNEMPLOYMENT INSURANCE WEEKLY CLAIMS REPORT", "OTHER REPORT"),
             "title_missing"),
            (text.replace("advance figure", "latest figure"),
             "actual_value_unavailable"),
        )
        for value, expected in cases:
            with self.subTest(expected=expected), tempfile.TemporaryDirectory() as directory:
                path = pathlib.Path(directory) / "bad.html"
                path.write_text(value)
                with self.assertRaisesRegex(ValueError, expected):
                    parse_dol_eta_weekly_claims_actual(
                        dataclasses.replace(self.artifact, parser_path=path)
                    )

    def test_exact_mapping_missing_and_ambiguous(self) -> None:
        rows, rejected = extract_dol_eta_candidates(self.artifact, [])
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "missing_event_mapping")

        rows, rejected = extract_dol_eta_candidates(
            self.artifact, [self.current, dataclasses.replace(self.current, economic_event_id=21)]
        )
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "event_mapping_ambiguous")

        wrong_reference = dataclasses.replace(
            self.current, reference_period="week ending 2010-07-16"
        )
        rows, rejected = extract_dol_eta_candidates(
            self.artifact, [wrong_reference, self.prior]
        )
        self.assertEqual(rows, [])
        self.assertEqual(rejected[0].decision, "event_mapping_ambiguous")

    def test_idempotency_conflict_and_separate_revision(self) -> None:
        rows, _ = extract_dol_eta_candidates(
            self.artifact, [self.current, self.prior]
        )
        first = match_candidates(rows, [self.current, self.prior])
        self.assertEqual([row.decision for row in first], ["matched", "matched"])
        existing = []
        for index, row in enumerate(rows, 1):
            event_id = (
                self.current.economic_event_id
                if row.publication_state == "initial"
                else self.prior.economic_event_id
            )
            persisted = row.persisted_values(event_id)
            persisted["economic_event_release_actual_id"] = index
            existing.append(persisted)
        repeated = match_candidates(rows, [self.current, self.prior], existing)
        self.assertEqual(
            [row.decision for row in repeated],
            ["duplicate_identical", "duplicate_identical"],
        )
        self.assertEqual(build_insert_sql(repeated), "BEGIN;\nCOMMIT;\n")
        conflict = dict(existing[0])
        conflict["actual_raw"] = "conflicting payload"
        self.assertEqual(
            match_candidates([rows[0]], [self.current], [conflict])[0].decision,
            "conflict",
        )
        self.assertNotEqual(rows[0].source_observation_id, rows[1].source_observation_id)
        self.assertNotEqual(
            rows[0].provenance_observation_values(20)["evidence_key"],
            rows[1].provenance_observation_values(19)["evidence_key"],
        )
        self.assertEqual(
            deterministic_json_lines(first, []),
            deterministic_json_lines(first, []),
        )


class DolEtaAcquisitionManifestTests(unittest.TestCase):
    def test_only_documented_filename_year_exception_is_accepted(self) -> None:
        known = "https://oui.doleta.gov/press/2019/010318.pdf"
        self.assertEqual(acquisition_release_date(known), "2019-01-03")
        self.assertEqual(parser_release_date(known), "2019-01-03")
        unknown = "https://oui.doleta.gov/press/2020/010319.pdf"
        with self.assertRaisesRegex(ValueError, "artifact_identity_year_mismatch"):
            acquisition_release_date(unknown)
        with self.assertRaisesRegex(ValueError, "artifact_identity_year_mismatch"):
            parser_release_date(unknown)

    def test_resumable_manifest_is_byte_deterministic(self) -> None:
        url = "https://oui.doleta.gov/press/2010/072210.asp"
        source = (FIXTURES / "2010-07-22.html").read_bytes()
        with tempfile.TemporaryDirectory(prefix="ea-dol-acquisition-") as directory:
            repo = pathlib.Path(directory)
            output = repo / "EconomicCalendar/raw/dol_eta"
            fetcher = mock.Mock(return_value=Download(url, url, source))
            first = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=fetcher, observed_at=RETRIEVED_AT,
            )
            manifest = first[2]
            before = manifest.read_bytes()
            second = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=mock.Mock(side_effect=AssertionError("network retry")),
                observed_at="2099-01-01T00:00:00.000000Z",
            )
            self.assertEqual(first[:2], (1, 0))
            self.assertEqual(second[:2], (1, 0))
            self.assertEqual(before, manifest.read_bytes())
            self.assertEqual(fetcher.call_count, 1)
            loaded = load_dol_eta_artifacts(repo, manifest)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(loaded[0].source_sha256, hashlib.sha256(source).hexdigest())

    def test_failure_is_manifested_and_retryable(self) -> None:
        url = "https://oui.doleta.gov/press/2010/072210.asp"
        source = (FIXTURES / "2010-07-22.html").read_bytes()
        with tempfile.TemporaryDirectory(prefix="ea-dol-acquisition-") as directory:
            repo = pathlib.Path(directory)
            output = repo / "EconomicCalendar/raw/dol_eta"
            result = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=mock.Mock(side_effect=RuntimeError("rate limited")),
                observed_at=RETRIEVED_AT,
            )
            self.assertEqual(result[:2], (0, 1))
            failed = json.loads(result[2].read_text())
            self.assertEqual(failed["http_acquisition_result"], "failed")
            self.assertEqual(failed["import_eligibility_disposition"], "extraction_failed")

            retried = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=mock.Mock(return_value=Download(url, url, source)),
                observed_at=RETRIEVED_AT,
            )
            self.assertEqual(retried[:2], (1, 0))

    def test_manifest_hash_conflict_and_unknown_version_fail_closed(self) -> None:
        url = "https://oui.doleta.gov/press/2010/072210.asp"
        source = (FIXTURES / "2010-07-22.html").read_bytes()
        with tempfile.TemporaryDirectory(prefix="ea-dol-acquisition-") as directory:
            repo = pathlib.Path(directory)
            output = repo / "EconomicCalendar/raw/dol_eta"
            _, _, manifest = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=mock.Mock(return_value=Download(url, url, source)),
                observed_at=RETRIEVED_AT,
            )
            row = json.loads(manifest.read_text())
            row["parser_version"] = "unknown"
            manifest.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "parser_version_invalid"):
                load_dol_eta_artifacts(repo, manifest)

    def test_manifest_resource_identity_conflict_fails_closed(self) -> None:
        url = "https://oui.doleta.gov/press/2010/072210.asp"
        source = (FIXTURES / "2010-07-22.html").read_bytes()
        with tempfile.TemporaryDirectory(prefix="ea-dol-acquisition-") as directory:
            repo = pathlib.Path(directory)
            output = repo / "EconomicCalendar/raw/dol_eta"
            _, _, manifest = acquire(
                [url], output, repo_root=repo, pdftotext=None,
                fetcher=mock.Mock(return_value=Download(url, url, source)),
                observed_at=RETRIEVED_AT,
            )
            row = json.loads(manifest.read_text())
            row["source_artifact_identity"] = "dol_eta:press:wrong"
            manifest.write_text(json.dumps(row) + "\n")
            with self.assertRaisesRegex(ValueError, "artifact_identity_mismatch"):
                load_dol_eta_artifacts(repo, manifest)


if __name__ == "__main__":
    unittest.main()
