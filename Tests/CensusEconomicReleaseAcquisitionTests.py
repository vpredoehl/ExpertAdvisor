#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import pathlib
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "Scripts" / "fetch_census_economic_releases.py"
SPEC = importlib.util.spec_from_file_location("fetch_census", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
fetch_census = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(fetch_census)


class CensusAcquisitionTests(unittest.TestCase):
    def test_supported_first_party_occurrences(self) -> None:
        cases = [
            (
                "https://www2.census.gov/retail/releases/historical/marts/adv2402.pdf",
                "retail-sales-advance", 2024, 2,
            ),
            (
                "https://www.census.gov/construction/nrc/pdf/newresconst_202404.pdf",
                "new-residential-construction", 2024, 4,
            ),
            (
                "https://www.census.gov/construction/nrs/pdf/newressales_202412.pdf",
                "new-residential-sales", 2024, 12,
            ),
            (
                "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/prel/2024/feb24prel.pdf",
                "manufacturers-orders", 2024, 2,
            ),
            (
                "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2024/feb24adv.pdf",
                "durable-goods-advance", 2024, 2,
            ),
            (
                "https://www.census.gov/construction/c30/pdf/pr202402.pdf",
                "construction-spending", 2024, 2,
            ),
        ]
        for url, family, year, month in cases:
            canonical, actual_family, actual_year, actual_month, filename = (
                fetch_census.occurrence(url)
            )
            self.assertEqual(canonical, url)
            self.assertEqual((actual_family, actual_year, actual_month), (family, year, month))
            self.assertTrue(filename.endswith(".pdf"))

    def test_rejects_non_census_and_unsupported_joint_trade(self) -> None:
        rejected = [
            "http://www.census.gov/construction/c30/pdf/pr202402.pdf",
            "https://example.com/construction/c30/pdf/pr202402.pdf",
            "https://www.census.gov/foreign-trade/Press-Release/current_press_release/ft900.pdf",
            "https://www.census.gov/manufacturing/m3/historical_data/pressreleases/adv/2024/feb25adv.pdf",
        ]
        for url in rejected:
            with self.assertRaises(ValueError):
                fetch_census.occurrence(url)

    def test_acquire_writes_deterministic_hashed_manifest(self) -> None:
        urls = [
            "https://www.census.gov/construction/c30/pdf/pr202402.pdf",
            "https://www2.census.gov/retail/releases/historical/marts/adv2402.pdf",
        ]

        def fake_extract(arguments: list[str], **_: object) -> subprocess.CompletedProcess[str]:
            pathlib.Path(arguments[-1]).write_bytes(
                b"deterministic extracted first-party parser input\n"
            )
            return subprocess.CompletedProcess(arguments, 0, "", "")

        with tempfile.TemporaryDirectory(prefix="ea-census-acquisition-test-") as directory:
            output = pathlib.Path(directory) / "output"
            with mock.patch.object(
                fetch_census, "fetch", side_effect=lambda url: ("PDF:" + url).encode("utf-8")
            ), mock.patch.object(
                fetch_census, "extractor_identity", return_value="pdftotext-test-1"
            ), mock.patch.object(
                fetch_census.subprocess, "run", side_effect=fake_extract
            ):
                fetch_census.acquire(list(reversed(urls)), output, "/test/pdftotext")

            manifest = (output / "manifest.tsv").read_text(encoding="utf-8")
            self.assertTrue(manifest.startswith(
                "manifest_version\t1\nparser_version\tcensus_economic_release_v1\n"
            ))
            entries = manifest.splitlines()[3:]
            self.assertEqual(entries, sorted(entries))
            self.assertEqual(len(entries), 2)
            self.assertIn("construction-spending/2024/pr202402.txt", entries[0])
            self.assertIn("retail-sales-advance/2024/adv2402.txt", entries[1])
            self.assertIn("\tpdf_text\t", manifest)
            self.assertIn("\tpdftotext-test-1", manifest)


if __name__ == "__main__":
    unittest.main()
