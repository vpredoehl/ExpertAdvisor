#!/usr/bin/env python3

from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
from unittest import mock


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
import fetch_bls_economic_releases as bls


SCHEDULE = b"""
<table class='release-list'><tbody>
<tr><td class='date-cell'>Friday, January 08, 2010</td><td class='time-cell'>08:30 AM</td><td class='desc-cell'><strong>Employment Situation</strong> for December 2009</td></tr>
<tr><td class='date-cell'>Tuesday, January 12, 2010</td><td class='time-cell'>10:00 AM</td><td class='desc-cell'><strong>Job Openings and Labor Turnover Survey</strong> for November 2009</td></tr>
<tr><td class='date-cell'>Friday, January 15, 2010</td><td class='time-cell'>08:30 AM</td><td class='desc-cell'><strong>Consumer Price Index</strong> for December 2009</td></tr>
<tr><td class='date-cell'>Wednesday, January 20, 2010</td><td class='time-cell'>08:30 AM</td><td class='desc-cell'><strong>Producer Price Index</strong> for December 2009</td></tr>
<tr><td class='date-cell'>Friday, March 12, 2010</td><td class='time-cell'>10:00 AM</td><td class='desc-cell'><strong>Employment Situation of Veterans</strong> for Annual 2009</td></tr>
</tbody></table>
"""


class BlsAcquisitionTests(unittest.TestCase):
    def test_schedule_rows_are_deterministic_occurrences(self) -> None:
        url = "https://www.bls.gov/schedule/2010/home.htm"
        rows = bls.parse_occurrences(SCHEDULE, url, 2010)
        self.assertEqual(len(rows), 5)
        self.assertEqual(
            {row["family"] for row in rows},
            {"CPI", "EMPLOYMENT", "EMPLOYMENT_ANNUAL", "JOLTS", "PPI"},
        )
        self.assertEqual(rows[0]["release_time"], "08:30:00")

    def test_acquisition_retains_annual_source_and_row_provenance(self) -> None:
        url = "https://www.bls.gov/schedule/2010/home.htm"
        with tempfile.TemporaryDirectory(prefix="ea-bls-acquisition-test-") as directory:
            output = pathlib.Path(directory) / "output"
            with mock.patch.object(
                bls, "fetch", return_value=bls.Download(url, url, SCHEDULE)
            ):
                bls.acquire([2010], output)
            manifest = (output / "manifest.tsv").read_text()
            self.assertIn("parser_version\tbls_schedule_release_v1", manifest)
            self.assertEqual(len(manifest.splitlines()[3:]), 5)
            self.assertIn("\tbls_schedule_row\t", manifest)
            self.assertIn("\tbls_schedule_row_v1", manifest)
            acquisition = (output / "acquisition.tsv").read_text()
            self.assertIn(url + "\t" + url, acquisition)


if __name__ == "__main__":
    unittest.main()
