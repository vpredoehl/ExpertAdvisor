#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "fetch_federal_reserve_economic_releases.py"
SPEC = importlib.util.spec_from_file_location("fetch_federal_reserve", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fetch_federal_reserve = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_federal_reserve)


class FederalReserveAcquisitionTests(unittest.TestCase):
    def test_supported_first_party_occurrences(self) -> None:
        cases = [
            (
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20100810a.htm",
                "fomc_statement",
                "monetary20100810a",
            ),
            (
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240703a.htm",
                "fomc_minutes",
                "monetary20240703a",
            ),
            (
                "https://www.federalreserve.gov/fomc/beigebook/2010/20100113/"
                "fullreport20100113.pdf",
                "beige_book",
                "beigebook-20100113",
            ),
            (
                "https://www.federalreserve.gov/monetarypolicy/files/"
                "BeigeBook_20240417.pdf",
                "beige_book",
                "beigebook-20240417",
            ),
        ]
        for url, family, identity in cases:
            canonical, actual_identity = (
                fetch_federal_reserve.canonical_occurrence_url(url, family)
            )
            self.assertEqual(canonical, url)
            self.assertEqual(actual_identity, identity)

    def test_rejects_non_fed_redirectable_and_contradictory_urls(self) -> None:
        rejected = [
            (
                "http://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240612a.htm",
                "fomc_statement",
            ),
            (
                "https://example.com/newsevents/pressreleases/monetary20240612a.htm",
                "fomc_statement",
            ),
            (
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240612a.htm?source=calendar",
                "fomc_statement",
            ),
            (
                "https://www.federalreserve.gov/fomc/beigebook/2010/20110113/"
                "fullreport20110113.pdf",
                "beige_book",
            ),
            (
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240231a.htm",
                "fomc_statement",
            ),
        ]
        for url, family in rejected:
            with self.assertRaises(ValueError):
                fetch_federal_reserve.canonical_occurrence_url(url, family)

    def test_acquire_writes_deterministic_hashed_manifest(self) -> None:
        selections = [
            (
                "fomc_minutes",
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240703a.htm",
            ),
            (
                "fomc_statement",
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                "monetary20240612a.htm",
            ),
        ]

        def fake_fetch(url: str) -> bytes:
            return ("<html>first-party:" + url + "</html>").encode("utf-8")

        with tempfile.TemporaryDirectory(prefix="ea-fed-acquisition-test-") as directory:
            output = pathlib.Path(directory) / "output"
            with mock.patch.object(
                fetch_federal_reserve, "fetch", side_effect=fake_fetch
            ):
                fetch_federal_reserve.acquire(
                    list(reversed(selections)), output, None
                )

            manifest = (output / "manifest.tsv").read_text(encoding="utf-8")
            self.assertTrue(manifest.startswith(
                "manifest_version\t1\n"
                "parser_version\tfederal_reserve_economic_release_v1\n"
            ))
            entries = manifest.splitlines()[3:]
            self.assertEqual(entries, sorted(entries))
            self.assertEqual(len(entries), 2)
            self.assertIn("\tfomc_minutes_html\t", manifest)
            self.assertIn("\tfomc_statement_html\t", manifest)
            for entry in entries:
                fields = entry.split("\t")
                artifact = (output / fields[0]).read_bytes()
                self.assertEqual(fetch_federal_reserve.sha256(artifact), fields[1])
                self.assertEqual(fields[0], fields[4])
                self.assertEqual(fields[1], fields[5])
                self.assertEqual(fields[6], "none")

    def test_beige_book_retains_pdf_and_extractor_provenance(self) -> None:
        url = (
            "https://www.federalreserve.gov/monetarypolicy/files/"
            "BeigeBook_20240417.pdf"
        )

        def fake_extract(
            arguments: list[str], **_: object
        ) -> subprocess.CompletedProcess[str]:
            pathlib.Path(arguments[-1]).write_bytes(
                b"Beige Book - April 17, 2024\nFor use at 2:00 p.m. EDT\n"
            )
            return subprocess.CompletedProcess(arguments, 0, "", "")

        with tempfile.TemporaryDirectory(prefix="ea-fed-acquisition-test-") as directory:
            output = pathlib.Path(directory) / "output"
            with mock.patch.object(
                fetch_federal_reserve, "fetch", return_value=b"%PDF-test"
            ), mock.patch.object(
                fetch_federal_reserve,
                "extractor_identity",
                return_value="pdftotext-test-1",
            ), mock.patch.object(
                fetch_federal_reserve.subprocess, "run", side_effect=fake_extract
            ):
                fetch_federal_reserve.acquire(
                    [("beige_book", url)], output, "/test/pdftotext"
                )

            entry = (output / "manifest.tsv").read_text(encoding="utf-8").splitlines()[3]
            fields = entry.split("\t")
            self.assertEqual(fields[2], "beige_book_pdf_text")
            self.assertEqual(fields[6], "pdftotext-test-1")
            self.assertTrue((output / fields[0]).is_file())
            self.assertTrue((output / fields[4]).is_file())

    def test_bounded_selection_and_nonempty_output(self) -> None:
        base = (
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20240612a.htm"
        )
        too_many = [("fomc_statement", base)] * (
            fetch_federal_reserve.MAX_ARTIFACTS + 1
        )
        # Duplicate selections collapse deterministically before the cap.
        with tempfile.TemporaryDirectory(prefix="ea-fed-acquisition-test-") as directory:
            output = pathlib.Path(directory) / "output"
            with mock.patch.object(
                fetch_federal_reserve, "fetch", return_value=b"<html>test</html>"
            ):
                fetch_federal_reserve.acquire(too_many, output, None)
            self.assertEqual(
                len((output / "manifest.tsv").read_text().splitlines()[3:]), 1
            )

        unique = []
        for index in range(fetch_federal_reserve.MAX_ARTIFACTS + 1):
            day = index % 28 + 1
            suffix = chr(ord("a") + (index // 28))
            unique.append((
                "fomc_statement",
                "https://www.federalreserve.gov/newsevents/pressreleases/"
                f"monetary202401{day:02d}{suffix}.htm",
            ))
        with tempfile.TemporaryDirectory(prefix="ea-fed-acquisition-test-") as directory:
            with self.assertRaisesRegex(ValueError, "safety cap"):
                fetch_federal_reserve.acquire(
                    unique, pathlib.Path(directory) / "output", None
                )


if __name__ == "__main__":
    unittest.main()
