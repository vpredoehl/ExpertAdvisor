#!/usr/bin/env python3

from __future__ import annotations

import importlib
import io
import pathlib
import sys
import tempfile
import unittest
import urllib.parse
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))

acquisition = importlib.import_module("authoritative_acquisition")
bea = importlib.import_module("fetch_bea_economic_releases")
census = importlib.import_module("fetch_census_economic_releases")
dol_eta = importlib.import_module("fetch_dol_eta_weekly_claims")
federal_reserve = importlib.import_module(
    "fetch_federal_reserve_economic_releases"
)


class FakeResponse(io.BytesIO):
    def __init__(self, final_url: str, data: bytes = b"authoritative") -> None:
        super().__init__(data)
        self.final_url = final_url

    def geturl(self) -> str:
        return self.final_url

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class RedirectProvenanceTests(unittest.TestCase):
    def test_bea_accepts_only_canonical_equivalent_occurrence_redirect(self) -> None:
        requested = "https://bea.gov/news/2024/gross-domestic-product-third-estimate"
        final = "https://www.bea.gov/news/2024/gross-domestic-product-third-estimate"
        with mock.patch.object(
            bea.urllib.request, "urlopen", return_value=FakeResponse(final)
        ):
            result = bea.fetch(requested, bea.canonical_occurrence_url)
        self.assertEqual(result.final_url, final)

        landing = "https://www.bea.gov/news/archive"
        with mock.patch.object(
            bea.urllib.request, "urlopen", return_value=FakeResponse(landing)
        ), self.assertRaisesRegex(RuntimeError, "final resource"):
            bea.fetch(requested, bea.canonical_occurrence_url)

    def test_census_rejects_same_host_different_occurrence(self) -> None:
        requested = "https://www.census.gov/construction/c30/pdf/pr202402.pdf"
        other = "https://www.census.gov/construction/c30/pdf/pr202403.pdf"
        with mock.patch.object(
            census.urllib.request, "urlopen", return_value=FakeResponse(other)
        ), self.assertRaisesRegex(RuntimeError, "changed requested resource"):
            census.fetch(requested, lambda value: census.occurrence(value)[0])

    def test_dol_eta_rejects_same_host_generic_page(self) -> None:
        requested = "https://oui.doleta.gov/press/2025/010225.pdf"
        final = "https://oui.doleta.gov/unemploy/archive.asp"
        with mock.patch.object(
            dol_eta.urllib.request, "urlopen", return_value=FakeResponse(final)
        ), self.assertRaisesRegex(RuntimeError, "final resource"):
            dol_eta.fetch(requested, dol_eta.canonical_press_url)

    def test_dol_eta_archive_posts_requested_year(self) -> None:
        archive = (
            b'<a href="/press/2010/010710.asp">January 7</a>'
            b'<a href="/press/2025/010225.pdf">wrong year</a>'
        )
        captured: list[object] = []

        def open_archive(request: object, timeout: int) -> FakeResponse:
            captured.append(request)
            self.assertEqual(timeout, 30)
            return FakeResponse(dol_eta.ARCHIVE_URL, archive)

        dol_eta.archive_occurrences.cache_clear()
        with mock.patch.object(
            dol_eta.urllib.request, "urlopen", side_effect=open_archive
        ):
            self.assertEqual(
                dol_eta.enumerate_archive(2010),
                ["https://oui.doleta.gov/press/2010/010710.asp"],
            )
        self.assertEqual(len(captured), 1)
        request = captured[0]
        self.assertEqual(request.full_url, dol_eta.ARCHIVE_URL)
        self.assertEqual(
            urllib.parse.parse_qs(request.data.decode("ascii")),
            {"report": ["press"], "year": ["2010"]},
        )

    def test_federal_reserve_rejects_distinct_same_date_occurrence(self) -> None:
        requested = (
            "https://www.federalreserve.gov/newsevents/pressreleases/"
            "monetary20140917a.htm"
        )
        other = requested.replace("17a.htm", "17c.htm")
        canonicalize = lambda value: federal_reserve.canonical_occurrence_url(
            value, "fomc_statement"
        )[0]
        with mock.patch.object(
            federal_reserve.urllib.request,
            "urlopen",
            return_value=FakeResponse(other),
        ), self.assertRaisesRegex(RuntimeError, "changed requested resource"):
            federal_reserve.fetch(requested, canonicalize)

    def test_shared_bounded_read_fails_deterministically(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "exceeds 4 byte safety limit"):
            acquisition.read_bounded(
                io.BytesIO(b"12345"),
                requested_url="https://example.gov/occurrence",
                max_download_bytes=4,
            )

    def test_bea_retained_catalog_hash_is_enforced(self) -> None:
        url = "https://www.bea.gov/news/2024/gross-domestic-product-third-estimate"
        with tempfile.TemporaryDirectory() as temporary:
            root = pathlib.Path(temporary)
            catalog = root / "enumerated.tsv"
            catalog.write_text(
                "agency\tsource_url\tartifact_sha256\n"
                f"bea\t{url}\t{'0' * 64}\n",
                encoding="utf-8",
            )
            expected = bea.read_enumerated_source_manifest(catalog)
            with mock.patch.object(
                bea,
                "fetch",
                return_value=acquisition.Download(url, url, b"changed"),
            ), self.assertRaisesRegex(RuntimeError, "retained artifact hash mismatch"):
                bea.acquire([url], root / "output", expected)
            acquisition_rows = (root / "output" / "acquisition.tsv").read_text(
                encoding="utf-8"
            )
            self.assertIn("\tfailed\t", acquisition_rows)
            self.assertIn("hash differs", acquisition_rows)


if __name__ == "__main__":
    unittest.main()
