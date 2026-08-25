#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path

ROOT = Path("EconomicCalendar/raw")
OUTPUT_ROOT = ROOT / "coverage_audit"

SOURCE_FAMILY_REPORT = (
    OUTPUT_ROOT / "economic_event_source_family_coverage.csv"
)
YEAR_REPORT = (
    OUTPUT_ROOT / "economic_event_year_coverage.csv"
)
ISSUES_REPORT = (
    OUTPUT_ROOT / "economic_event_coverage_issues.csv"
)
REMAINING_REPORT = (
    OUTPUT_ROOT / "economic_event_remaining_families.csv"
)
SUMMARY_REPORT = (
    OUTPUT_ROOT / "economic_event_coverage_summary.txt"
)

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

#
# Families that are already authoritative independent of Census import state.
# Census is classified dynamically from the live read-only database snapshot:
# zero rows means still planned, the exact validated 399-row population means
# imported, and any nonzero incomplete population is a hard audit failure.
#
BASE_EXPECTED_IMPORTED_FAMILIES = {
    ("BLS", "CPI"),
    ("BLS", "EMPLOYMENT"),
    ("BLS", "EMPLOYMENT_ANNUAL"),
    ("BLS", "JOLTS"),
    ("BLS", "PPI"),
    ("BEA", "GDP"),
    ("BEA", "PCE"),
    ("FEDERAL_RESERVE", "FOMC"),
    ("DOL_ETA", "WEEKLY_CLAIMS"),
}

CENSUS_IMPORTED_FAMILIES = {
    ("CENSUS", "RETAIL_SALES"),
    ("CENSUS", "DURABLE_GOODS"),
}

# Static full authoritative family contract retained for tests/importers.
# Runtime audit classification still uses expected_imported_families below.
EXPECTED_IMPORTED_FAMILIES = (
    BASE_EXPECTED_IMPORTED_FAMILIES | CENSUS_IMPORTED_FAMILIES
)

CENSUS_EXPECTED_TOTAL_ROWS = 399
CENSUS_EXPECTED_FAMILY_ROWS = {
    "RETAIL_SALES": 200,
    "DURABLE_GOODS": 199,
}

CENSUS_PLANNED_REMAINING_FAMILIES = [
    {
        "source_agency": "CENSUS",
        "event_family": "RETAIL_SALES",
        "status": "not_yet_imported",
        "note": (
            "Planned U.S. Census Bureau authoritative ingestion."
        ),
    },
    {
        "source_agency": "CENSUS",
        "event_family": "DURABLE_GOODS",
        "status": "not_yet_imported",
        "note": (
            "Planned U.S. Census Bureau authoritative ingestion."
        ),
    },
]

# Compatibility/exported completed-plan contract used by existing tests.
# Runtime pre-import Census reporting is handled dynamically through
# CENSUS_PLANNED_REMAINING_FAMILIES.
PLANNED_REMAINING_FAMILIES = []


#
# Full-year BLS cadence expectations.
#
# The four regular BLS families are monthly.
# EMPLOYMENT_ANNUAL is one authoritative annual event per year.
#
BLS_MONTHLY_FAMILIES = {
    "CPI",
    "EMPLOYMENT",
    "JOLTS",
    "PPI",
}

#
# 2025 was not a normal BLS release year because of the federal
# lapse in appropriations from 2025-10-01 through 2025-11-12.
#
# These are counts of actual market release events occurring during
# calendar year 2025, not counts of reference months.
#
# BLS explicitly:
#
#   * canceled the October 2025 CPI release;
#   * canceled the October 2025 Employment Situation release;
#   * canceled the September 2025 JOLTS release and published those
#     data with the October release;
#   * canceled the October 2025 PPI release and moved the November
#     2025 PPI release into January 2026.
#
BLS_2025_EXPECTED_RELEASE_COUNTS = {
    "CPI": 11,
    "EMPLOYMENT": 11,
    "JOLTS": 11,
    "PPI": 10,
}

BLS_2025_IRREGULAR_NOTES = {
    "CPI": (
        "Documented 2025 federal funding-lapse exception. "
        "BLS canceled the October 2025 CPI release; the November "
        "release occurred 2025-12-18 and December 2025 was released "
        "in January 2026."
    ),
    "EMPLOYMENT": (
        "Documented 2025 federal funding-lapse exception. "
        "BLS canceled the October 2025 Employment Situation release; "
        "October establishment data were incorporated with the "
        "November release."
    ),
    "JOLTS": (
        "Documented 2025 federal funding-lapse exception. "
        "BLS canceled the September 2025 JOLTS release and published "
        "September data with the October 2025 release."
    ),
    "PPI": (
        "Documented 2025 federal funding-lapse exception. "
        "BLS canceled the October 2025 PPI release and moved the "
        "November 2025 PPI release to January 14, 2026."
    ),
}

#
# BEA counts already established and validated by the dedicated
# BEA canonical validator.
#
BEA_EXPECTED_YEAR_COUNTS = {
    "GDP": {
        2010: 12,
        2011: 12,
        2012: 12,
        2013: 12,
        2014: 12,
        2015: 12,
        2016: 12,
        2017: 12,
        2018: 12,
        2019: 11,
        2020: 12,
        2021: 12,
        2022: 12,
        2023: 12,
        2024: 12,
        2025: 10,
        2026: 8,
    },
    "PCE": {
        2010: 12,
        2011: 12,
        2012: 12,
        2013: 12,
        2014: 12,
        2015: 12,
        2016: 12,
        2017: 12,
        2018: 12,
        2019: 11,
        2020: 12,
        2021: 12,
        2022: 12,
        2023: 12,
        2024: 12,
        2025: 10,
        2026: 7,
    },
}

BEA_IRREGULAR_NOTES = {
    (2019, "GDP"): (
        "Federal shutdown disrupted normal GDP sequencing; "
        "Q4 2018 Initial Estimate replaced the normal "
        "advance/second sequence."
    ),
    (2019, "PCE"): (
        "Federal shutdown caused delayed/combined PCE releases."
    ),
    (2025, "GDP"): (
        "Late-2025 shutdown/rescheduling altered GDP sequencing; "
        "Q3 2025 Updated Estimate was the third-estimate equivalent."
    ),
    (2025, "PCE"): (
        "Late-2025 shutdown/rescheduling altered monthly PCE cadence."
    ),
    (2026, "GDP"): (
        "Partial year through 2026-08-24."
    ),
    (2026, "PCE"): (
        "Partial year through 2026-08-24."
    ),
}

#
# FOMC counts already established and validated by the dedicated
# FOMC canonical validator.
#
FOMC_EXPECTED_YEAR_COUNTS = {
    2010: 9,
    2011: 8,
    2012: 8,
    2013: 8,
    2014: 8,
    2015: 8,
    2016: 8,
    2017: 8,
    2018: 8,
    2019: 8,
    2020: 10,
    2021: 8,
    2022: 8,
    2023: 8,
    2024: 8,
    2025: 8,
    2026: 5,
}

FOMC_IRREGULAR_NOTES = {
    2010: (
        "Nine genuine FOMC statement events, including the "
        "unscheduled 2010-05-09 liquidity-swap statement."
    ),
    2020: (
        "Ten genuine FOMC statement events due to extraordinary "
        "pandemic-era unscheduled actions."
    ),
    2026: (
        "Partial year through 2026-08-24."
    ),
}

#
# DOL/ETA Weekly Claims counts established from the validated authoritative
# 2010-present manifest and canonical production import.
#
# These are release-year counts, not an assumed 52-per-year cadence. Calendar
# structure, archive irregularities, and partial-year coverage mean the
# authoritative count can legitimately differ from 52.
#
DOL_ETA_WEEKLY_CLAIMS_EXPECTED_YEAR_COUNTS = {
    2010: 52,
    2011: 52,
    2012: 52,
    2013: 52,
    2014: 53,
    2015: 52,
    2016: 52,
    2017: 52,
    2018: 52,
    2019: 51,
    2020: 53,
    2021: 52,
    2022: 52,
    2023: 52,
    2024: 52,
    2025: 46,
    2026: 33,
}

DOL_ETA_WEEKLY_CLAIMS_IRREGULAR_NOTES = {
    2019: (
        "Validated authoritative DOL/ETA archive contains 51 Weekly Claims "
        "release occurrences in release year 2019."
    ),
    2025: (
        "Validated authoritative DOL/ETA archive contains 46 Weekly Claims "
        "release occurrences in release year 2025."
    ),
    2026: (
        "Partial year through 2026-08-24; latest imported Weekly Claims "
        "release is 2026-08-20."
    ),
}

#
# Census counts established by validate_census_canonical.py. These are
# release-year counts, so delayed prior-reference-period publications remain
# assigned to the calendar year in which the market received the release.
#
CENSUS_EXPECTED_YEAR_COUNTS = {
    "RETAIL_SALES": {
        **{year: 12 for year in range(2010, 2025)},
        2025: 11,
        2026: 9,
    },
    "DURABLE_GOODS": {
        **{year: 12 for year in range(2010, 2025)},
        2025: 11,
        2026: 8,
    },
}

CENSUS_IRREGULAR_NOTES = {
    (2019, "RETAIL_SALES"): (
        "The 2019 federal shutdown delayed the December 2018 and early-2019 "
        "reference-period sequence; twelve actual release events still "
        "occurred during release year 2019."
    ),
    (2019, "DURABLE_GOODS"): (
        "The 2019 federal shutdown delayed the December 2018 and early-2019 "
        "reference-period sequence; twelve actual release events still "
        "occurred during release year 2019."
    ),
    (2025, "RETAIL_SALES"): (
        "Late-2025 scheduling shifted November 2025 and December 2025 "
        "reference-period releases into 2026; release year 2025 contains "
        "eleven events."
    ),
    (2025, "DURABLE_GOODS"): (
        "Late-2025 scheduling shifted November 2025 and December 2025 "
        "reference-period releases into 2026; release year 2025 contains "
        "eleven events."
    ),
    (2026, "RETAIL_SALES"): (
        "Partial release year through 2026-08-24, including November and "
        "December 2025 reference-period releases."
    ),
    (2026, "DURABLE_GOODS"): (
        "Partial release year through 2026-08-24, including November and "
        "December 2025 reference-period releases; the June 2026 timestamp "
        "is the validated schedule-derived occurrence."
    ),
}

ISSUE_FIELDS = [
    "severity",
    "code",
    "message",
    "source_agency",
    "event_family",
    "event_timestamp_utc",
    "source_event_id",
]


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def run_psql(args, sql: str) -> str:
    cmd = [
        args.psql,
        "-X",
        "-q",
        "-A",
        "-t",
        "-v",
        "ON_ERROR_STOP=1",
    ]

    if args.host:
        cmd += ["-h", args.host]

    if args.port:
        cmd += ["-p", str(args.port)]

    if args.user:
        cmd += ["-U", args.user]

    cmd += [
        "-d",
        args.db,
        "-c",
        sql,
    ]

    env = os.environ.copy()

    #
    # Defense in depth: PostgreSQL itself rejects writes.
    #
    existing = env.get("PGOPTIONS", "")
    read_only = "-c default_transaction_read_only=on"
    env["PGOPTIONS"] = (
        f"{existing} {read_only}".strip()
    )

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    if result.returncode != 0:
        print(
            "psql failed:",
            file=sys.stderr,
        )
        print(
            result.stderr,
            file=sys.stderr,
        )
        raise SystemExit(
            result.returncode
        )

    return result.stdout


def read_database_rows(args):
    sql = """
BEGIN READ ONLY;

SELECT row_to_json(x)::text
FROM (
    SELECT
        economic_event_id,
        currency,
        event_family,
        event_timestamp_utc,
        source_agency,
        source_event_id,
        source_url,
        reference_period,
        event_importance,
        historical_time_confidence,
        source_release_date,
        source_release_time,
        source_timezone
    FROM economic_event
    WHERE source_agency IN (
        'BLS',
        'BEA',
        'FEDERAL_RESERVE',
        'CENSUS',
        'DOL_ETA'
    )
    ORDER BY
        event_timestamp_utc,
        source_agency,
        event_family,
        economic_event_id
) AS x;

COMMIT;
"""

    output = run_psql(
        args,
        sql,
    )

    rows = []

    for line in output.splitlines():
        line = line.strip()

        if not line:
            continue

        rows.append(
            json.loads(line)
        )

    return rows


def parse_timestamp(value: str):
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        raise ValueError(
            "timezone-naive timestamp"
        )

    return dt


def add_issue(
    issues,
    severity,
    code,
    message,
    row=None,
):
    issue = {
        "severity": severity,
        "code": code,
        "message": message,
        "source_agency": "",
        "event_family": "",
        "event_timestamp_utc": "",
        "source_event_id": "",
    }

    if row:
        for field in (
            "source_agency",
            "event_family",
            "event_timestamp_utc",
            "source_event_id",
        ):
            issue[field] = (
                row.get(field) or ""
            )

    issues.append(issue)


def expected_year_count(
    source_agency: str,
    family: str,
    year: int,
):
    #
    # BLS.
    #
    if source_agency == "BLS":
        if family in BLS_MONTHLY_FAMILIES:
            if 2010 <= year <= 2024:
                return 12, "normal", ""

            if year == 2025:
                return (
                    BLS_2025_EXPECTED_RELEASE_COUNTS[family],
                    "documented_irregular",
                    BLS_2025_IRREGULAR_NOTES[family],
                )

            if year == 2026:
                return (
                    None,
                    "partial_year",
                    (
                        "Partial release year through 2026-08-24. "
                        "Includes delayed 2025-reference-period "
                        "releases; exact count is reported but is not "
                        "forced to a normal full-year cadence."
                    ),
                )

        if family == "EMPLOYMENT_ANNUAL":
            if 2010 <= year <= 2026:
                return 1, "normal", ""

    #
    # BEA.
    #
    if source_agency == "BEA":
        family_counts = BEA_EXPECTED_YEAR_COUNTS.get(
            family,
            {},
        )

        if year in family_counts:
            note = BEA_IRREGULAR_NOTES.get(
                (year, family),
                "",
            )

            return (
                family_counts[year],
                (
                    "documented_irregular"
                    if note
                    else "normal"
                ),
                note,
            )

    #
    # Federal Reserve.
    #
    if (
        source_agency == "FEDERAL_RESERVE"
        and family == "FOMC"
    ):
        if year in FOMC_EXPECTED_YEAR_COUNTS:
            note = FOMC_IRREGULAR_NOTES.get(
                year,
                "",
            )

            return (
                FOMC_EXPECTED_YEAR_COUNTS[year],
                (
                    "documented_irregular"
                    if note
                    else "normal"
                ),
                note,
            )

    #
    # DOL/ETA Weekly Claims.
    #
    if (
        source_agency == "DOL_ETA"
        and family == "WEEKLY_CLAIMS"
    ):
        if year in DOL_ETA_WEEKLY_CLAIMS_EXPECTED_YEAR_COUNTS:
            note = DOL_ETA_WEEKLY_CLAIMS_IRREGULAR_NOTES.get(
                year,
                "",
            )
            return (
                DOL_ETA_WEEKLY_CLAIMS_EXPECTED_YEAR_COUNTS[year],
                "documented_irregular" if note else "normal",
                note,
            )

    #
    # Census.
    #
    if source_agency == "CENSUS":
        family_counts = CENSUS_EXPECTED_YEAR_COUNTS.get(
            family,
            {},
        )

        if year in family_counts:
            note = CENSUS_IRREGULAR_NOTES.get(
                (year, family),
                "",
            )

            return (
                family_counts[year],
                (
                    "documented_irregular"
                    if note
                    else "normal"
                ),
                note,
            )

    return (
        None,
        "no_expectation",
        "",
    )


def write_csv(
    path: Path,
    fields,
    rows,
):
    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only combined economic_event "
            "coverage audit for BLS, BEA, Federal Reserve, "
            "Census, and DOL/ETA authoritative data."
        )
    )

    parser.add_argument(
        "--db",
        default="LSTM",
    )
    parser.add_argument(
        "--host",
    )
    parser.add_argument(
        "--port",
        type=int,
    )
    parser.add_argument(
        "--user",
    )
    parser.add_argument(
        "--psql",
        default="psql",
    )

    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "Combined economic_event coverage audit"
    )
    print()
    print(
        f"Requested window : "
        f"{START_DATE} .. {END_DATE}"
    )
    print(
        f"Database         : {args.db}"
    )
    print()
    print(
        "Reading BLS / BEA / Federal Reserve / Census / DOL/ETA "
        "economic_event rows (read-only)..."
    )

    rows = read_database_rows(args)

    issues = []

    census_rows = [
        row for row in rows
        if row.get("source_agency") == "CENSUS"
    ]
    census_family_counts = Counter(
        row.get("event_family") or ""
        for row in census_rows
    )

    census_import_complete = (
        len(census_rows) == CENSUS_EXPECTED_TOTAL_ROWS
        and all(
            census_family_counts[family] == expected
            for family, expected
            in CENSUS_EXPECTED_FAMILY_ROWS.items()
        )
        and set(census_family_counts)
        == set(CENSUS_EXPECTED_FAMILY_ROWS)
    )

    if not census_rows:
        expected_imported_families = set(
            BASE_EXPECTED_IMPORTED_FAMILIES
        )
        planned_remaining_families = list(
            CENSUS_PLANNED_REMAINING_FAMILIES
        )
        census_import_state = "not_yet_imported"
    elif census_import_complete:
        expected_imported_families = (
            set(BASE_EXPECTED_IMPORTED_FAMILIES)
            | CENSUS_IMPORTED_FAMILIES
        )
        planned_remaining_families = []
        census_import_state = "complete"
    else:
        expected_imported_families = (
            set(BASE_EXPECTED_IMPORTED_FAMILIES)
            | CENSUS_IMPORTED_FAMILIES
        )
        planned_remaining_families = []
        census_import_state = "partial_invalid"
        add_issue(
            issues,
            "error",
            "partial_census_import",
            (
                "Census economic_event population is nonzero but does not "
                "match the validated complete import: expected 399 total "
                "rows with RETAIL_SALES=200 and DURABLE_GOODS=199; found "
                f"total={len(census_rows)}, "
                f"RETAIL_SALES={census_family_counts['RETAIL_SALES']}, "
                f"DURABLE_GOODS={census_family_counts['DURABLE_GOODS']}."
            ),
        )

    #
    # Canonical-key and source-event-id checks.
    #
    canonical_keys = defaultdict(list)
    source_event_ids = defaultdict(list)

    family_rows = defaultdict(list)
    year_counts = Counter()

    for row in rows:
        source = (
            row.get("source_agency")
            or ""
        )
        family = (
            row.get("event_family")
            or ""
        )

        family_key = (
            source,
            family,
        )

        family_rows[family_key].append(
            row
        )

        #
        # Timestamp validity/window.
        #
        timestamp_raw = (
            row.get(
                "event_timestamp_utc"
            )
            or ""
        )

        try:
            dt = parse_timestamp(
                timestamp_raw
            )
        except Exception as exc:
            add_issue(
                issues,
                "error",
                "invalid_timestamp",
                f"Invalid timestamp: {exc}",
                row,
            )
            continue

        local_date_raw = row.get(
            "source_release_date"
        )

        if local_date_raw:
            try:
                local_date = (
                    datetime.strptime(
                        local_date_raw,
                        "%Y-%m-%d",
                    ).date()
                )
            except Exception as exc:
                add_issue(
                    issues,
                    "error",
                    "invalid_release_date",
                    (
                        "Invalid "
                        "source_release_date: "
                        f"{exc}"
                    ),
                    row,
                )
                continue
        else:
            #
            # Every imported authoritative
            # row should have this populated.
            #
            add_issue(
                issues,
                "error",
                "missing_release_date",
                (
                    "source_release_date "
                    "is NULL/empty"
                ),
                row,
            )

            local_date = (
                dt.date()
            )

        if (
            local_date < START_DATE
            or local_date > END_DATE
        ):
            add_issue(
                issues,
                "error",
                "outside_requested_window",
                (
                    f"{local_date} outside "
                    f"{START_DATE}..{END_DATE}"
                ),
                row,
            )

        year_counts[
            (
                source,
                family,
                local_date.year,
            )
        ] += 1

        canonical_key = (
            source,
            family,
            timestamp_raw,
        )

        canonical_keys[
            canonical_key
        ].append(row)

        source_event_id = row.get(
            "source_event_id"
        )

        if not source_event_id:
            add_issue(
                issues,
                "error",
                "missing_source_event_id",
                "source_event_id is NULL/empty",
                row,
            )
        else:
            source_event_ids[
                (
                    source,
                    source_event_id,
                )
            ].append(row)

        #
        # Common provenance invariants.
        #
        if row.get("currency") != "USD":
            add_issue(
                issues,
                "error",
                "unexpected_currency",
                (
                    "Expected USD, found "
                    f"{row.get('currency')!r}"
                ),
                row,
            )

        if (
            row.get("event_importance")
            != 3
        ):
            add_issue(
                issues,
                "error",
                "unexpected_importance",
                (
                    "Expected event_importance=3, "
                    f"found "
                    f"{row.get('event_importance')!r}"
                ),
                row,
            )

        confidence = row.get(
            "historical_time_confidence"
        )

        reconstructed_dol_eta_time = (
            row.get("source_agency") == "DOL_ETA"
            and row.get("event_family") == "WEEKLY_CLAIMS"
            and confidence == "reconstructed"
            and str(row.get("source_release_date") or "")[:4]
            in {"2011", "2012"}
            and row.get("source_release_time") == "08:30:00"
            and row.get("source_timezone") == "America/New_York"
        )

        if confidence != "exact" and not reconstructed_dol_eta_time:
            add_issue(
                issues,
                "error",
                "unexpected_time_confidence",
                (
                    "Expected historical_time_confidence='exact' "
                    "except documented DOL/ETA 2011-2012 "
                    "Weekly Claims reconstructed timestamps; found "
                    f"{confidence!r}"
                ),
                row,
            )

        if not row.get("source_url"):
            add_issue(
                issues,
                "error",
                "missing_source_url",
                "source_url is NULL/empty",
                row,
            )

        if not row.get(
            "reference_period"
        ):
            add_issue(
                issues,
                "error",
                "missing_reference_period",
                (
                    "reference_period "
                    "is NULL/empty"
                ),
                row,
            )

        if not row.get(
            "source_release_time"
        ):
            add_issue(
                issues,
                "error",
                "missing_release_time",
                (
                    "source_release_time "
                    "is NULL/empty"
                ),
                row,
            )

        if (
            row.get("source_timezone")
            != "America/New_York"
        ):
            add_issue(
                issues,
                "error",
                "unexpected_timezone",
                (
                    "Expected "
                    "America/New_York, found "
                    f"{row.get('source_timezone')!r}"
                ),
                row,
            )

    #
    # Duplicate canonical keys.
    #
    for key, matches in (
        canonical_keys.items()
    ):
        if len(matches) > 1:
            for row in matches:
                add_issue(
                    issues,
                    "error",
                    "duplicate_canonical_key",
                    (
                        "Duplicate "
                        "(source_agency,"
                        "event_family,"
                        "event_timestamp_utc)"
                    ),
                    row,
                )

    #
    # Duplicate source IDs.
    #
    for key, matches in (
        source_event_ids.items()
    ):
        if len(matches) > 1:
            for row in matches:
                add_issue(
                    issues,
                    "error",
                    "duplicate_source_event_id",
                    (
                        "Duplicate "
                        "(source_agency,"
                        "source_event_id)"
                    ),
                    row,
                )

    #
    # Required completed family presence.
    #
    actual_family_set = set(
        family_rows
    )

    missing_imported = sorted(
        expected_imported_families
        - actual_family_set
    )

    for source, family in (
        missing_imported
    ):
        add_issue(
            issues,
            "error",
            "missing_expected_family",
            (
                f"Expected imported family "
                f"{source}/{family} "
                "has no rows"
            ),
        )

    #
    # Unexpected source/family combinations
    # within the audited source agencies.
    #
    unexpected_families = sorted(
        actual_family_set
        - expected_imported_families
    )

    for source, family in (
        unexpected_families
    ):
        add_issue(
            issues,
            "warning",
            "unexpected_source_family",
            (
                f"Unexpected audited family "
                f"{source}/{family}"
            ),
        )

    #
    # Source/family summary.
    #
    source_family_report_rows = []

    for source, family in sorted(
        family_rows
    ):
        items = family_rows[
            (source, family)
        ]

        timestamps = []

        for row in items:
            try:
                timestamps.append(
                    parse_timestamp(
                        row[
                            "event_timestamp_utc"
                        ]
                    )
                )
            except Exception:
                pass

        source_family_report_rows.append({
            "source_agency":
                source,
            "event_family":
                family,
            "rows":
                len(items),
            "first_event_utc":
                (
                    min(timestamps).isoformat()
                    if timestamps
                    else ""
                ),
            "last_event_utc":
                (
                    max(timestamps).isoformat()
                    if timestamps
                    else ""
                ),
            "canonical_keys":
                len({
                    (
                        r["source_agency"],
                        r["event_family"],
                        r[
                            "event_timestamp_utc"
                        ],
                    )
                    for r in items
                }),
            "source_event_ids":
                len({
                    r["source_event_id"]
                    for r in items
                    if r.get(
                        "source_event_id"
                    )
                }),
            "status":
                (
                    "complete_imported_family"
                    if (
                        source,
                        family,
                    )
                    in expected_imported_families
                    else
                    "unexpected_family"
                ),
        })

    #
    # Year/family coverage.
    #
    year_report_rows = []

    coverage_mismatches = 0

    for (
        source,
        family,
    ) in sorted(
        expected_imported_families
    ):
        for year in range(
            2010,
            2027,
        ):
            actual = year_counts[
                (
                    source,
                    family,
                    year,
                )
            ]

            (
                expected,
                expected_status,
                note,
            ) = expected_year_count(
                source,
                family,
                year,
            )

            if expected is None:
                coverage_status = (
                    expected_status
                )
            elif actual == expected:
                coverage_status = (
                    expected_status
                    if expected_status
                    != "normal"
                    else "complete"
                )
            else:
                coverage_status = (
                    "count_mismatch"
                )

                coverage_mismatches += 1

                add_issue(
                    issues,
                    "error",
                    "year_family_count_mismatch",
                    (
                        f"{source}/{family} "
                        f"{year}: expected "
                        f"{expected}, found "
                        f"{actual}"
                    ),
                )

            year_report_rows.append({
                "year":
                    year,
                "source_agency":
                    source,
                "event_family":
                    family,
                "expected_count":
                    (
                        ""
                        if expected is None
                        else expected
                    ),
                "actual_count":
                    actual,
                "coverage_status":
                    coverage_status,
                "note":
                    note,
            })

    #
    # Remaining authoritative families.
    #
    remaining_rows = []

    for planned in (
        planned_remaining_families
    ):
        key = (
            planned["source_agency"],
            planned["event_family"],
        )

        actual_count = len(
            family_rows.get(
                key,
                [],
            )
        )

        remaining_rows.append({
            **planned,
            "current_rows":
                actual_count,
        })

    #
    # Persist reports.
    #
    write_csv(
        SOURCE_FAMILY_REPORT,
        [
            "source_agency",
            "event_family",
            "rows",
            "first_event_utc",
            "last_event_utc",
            "canonical_keys",
            "source_event_ids",
            "status",
        ],
        source_family_report_rows,
    )

    write_csv(
        YEAR_REPORT,
        [
            "year",
            "source_agency",
            "event_family",
            "expected_count",
            "actual_count",
            "coverage_status",
            "note",
        ],
        year_report_rows,
    )

    write_csv(
        ISSUES_REPORT,
        ISSUE_FIELDS,
        issues,
    )

    write_csv(
        REMAINING_REPORT,
        [
            "source_agency",
            "event_family",
            "status",
            "current_rows",
            "note",
        ],
        remaining_rows,
    )

    #
    # Human-readable authoritative snapshot.
    #
    total_by_source = Counter(
        row["source_agency"]
        for row in rows
    )

    total_by_family = Counter(
        (
            row["source_agency"],
            row["event_family"],
        )
        for row in rows
    )

    errors = [
        issue
        for issue in issues
        if issue["severity"] == "error"
    ]

    warnings = [
        issue
        for issue in issues
        if issue["severity"] == "warning"
    ]

    summary = []

    summary.append(
        "Combined economic_event coverage audit"
    )
    summary.append("")
    summary.append(
        f"Requested window : "
        f"{START_DATE} .. {END_DATE}"
    )
    summary.append(
        f"Audited rows     : {len(rows)}"
    )
    summary.append(
        f"Census state     : {census_import_state}"
    )
    summary.append("")

    summary.append(
        "Rows by authoritative source:"
    )

    for source in (
        "BLS",
        "BEA",
        "FEDERAL_RESERVE",
        "CENSUS",
        "DOL_ETA",
    ):
        summary.append(
            f"  {source:<16} "
            f"{total_by_source[source]:>5}"
        )

    summary.append("")
    summary.append(
        "Rows by imported family:"
    )

    for (
        source,
        family,
    ) in sorted(
        expected_imported_families
    ):
        summary.append(
            f"  {source:<16} "
            f"{family:<20} "
            f"{total_by_family[(source, family)]:>5}"
        )

    summary.append("")
    summary.append(
        "Canonical integrity:"
    )
    summary.append(
        f"  duplicate canonical keys : "
        f"{sum(1 for v in canonical_keys.values() if len(v) > 1)}"
    )
    summary.append(
        f"  duplicate source IDs     : "
        f"{sum(1 for v in source_event_ids.values() if len(v) > 1)}"
    )
    summary.append(
        f"  year-count mismatches    : "
        f"{coverage_mismatches}"
    )
    summary.append("")

    summary.append(
        "Remaining planned authoritative families:"
    )

    for row in remaining_rows:
        summary.append(
            f"  {row['source_agency']:<16} "
            f"{row['event_family']:<20} "
            f"{row['status']}"
        )

    if not remaining_rows:
        summary.append(
            "  none in the current authoritative family universe"
        )

    summary.append("")
    summary.append(
        f"Validation errors   : "
        f"{len(errors)}"
    )
    summary.append(
        f"Validation warnings : "
        f"{len(warnings)}"
    )
    summary.append("")

    summary.append(
        f"Source/family report: "
        f"{SOURCE_FAMILY_REPORT}"
    )
    summary.append(
        f"Year report         : "
        f"{YEAR_REPORT}"
    )
    summary.append(
        f"Issues              : "
        f"{ISSUES_REPORT}"
    )
    summary.append(
        f"Remaining families  : "
        f"{REMAINING_REPORT}"
    )
    summary.append("")

    if errors:
        summary.append(
            "RESULT: FAIL - combined authoritative "
            "coverage audit found errors."
        )
    else:
        if census_import_complete:
            summary.append(
                "RESULT: PASS - all currently imported "
                "BLS, BEA, Federal Reserve, Census, and DOL/ETA families "
                "satisfy the combined coverage audit."
            )
        else:
            summary.append(
                "RESULT: PASS - all currently imported BLS, BEA, "
                "Federal Reserve, and DOL/ETA families satisfy the combined "
                "coverage audit; Census remains not_yet_imported."
            )

    summary.append(
        "No database writes were performed."
    )

    SUMMARY_REPORT.write_text(
        "\n".join(summary) + "\n",
        encoding="utf-8",
    )

    print()
    print(
        "\n".join(summary)
    )

    if errors:
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
