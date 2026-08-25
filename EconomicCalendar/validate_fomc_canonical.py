#!/usr/bin/env python3

from __future__ import annotations

import csv
import re
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("EconomicCalendar/raw/federal_reserve")

INPUT = ROOT / "fomc_canonical_events.csv"
AUXILIARY = ROOT / "manifest_auxiliary.csv"

ERRORS = ROOT / "fomc_validation_errors.csv"
WARNINGS = ROOT / "fomc_validation_warnings.csv"
YEAR_COUNTS = ROOT / "fomc_validation_year_counts.csv"
TIME_COUNTS = ROOT / "fomc_validation_time_counts.csv"

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EXPECTED_ROWS = 136
EXPECTED_AGENCY = "FEDERAL_RESERVE"
EXPECTED_FAMILY = "FOMC"
EXPECTED_TZ = "America/New_York"

EASTERN = ZoneInfo(EXPECTED_TZ)

ALLOWED_PARSERS = {
    "for_release_at",
    "fomc_minutes_statement_release",
}

#
# Exact canonical release-time distribution currently established from
# authoritative Federal Reserve statement pages / official minutes.
#
EXPECTED_TIME_COUNTS = {
    "08:00:00": 1,
    "10:00:00": 1,
    "12:30:00": 8,
    "14:00:00": 107,
    "14:15:00": 17,
    "17:00:00": 1,
    "21:15:00": 1,
}

#
# Four directly observed unusual releases that must remain exactly present.
#
EXPECTED_DIRECT_UNUSUAL = {
    ("2010-05-09", "21:15:00"),
    ("2020-03-03", "10:00:00"),
    ("2020-03-15", "17:00:00"),
    ("2020-03-23", "08:00:00"),
}

#
# Expected canonical count by actual release year.
#
EXPECTED_YEAR_COUNTS = {
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

IRREGULAR_YEAR_NOTES = {
    2010: (
        "Nine genuine FOMC statement events. Includes the unscheduled "
        "2010-05-09 21:15 ET statement announcing reestablishment of "
        "temporary U.S. dollar liquidity swap facilities."
    ),
    2020: (
        "Ten genuine FOMC statement events due to extraordinary pandemic-era "
        "unscheduled actions, including releases on 2020-03-03, "
        "2020-03-15, and 2020-03-23."
    ),
    2026: (
        "Partial year through 2026-08-24; five FOMC statement events "
        "are present through 2026-07-29."
    ),
}

#
# Auxiliary content categories that must not leak into canonical FOMC events.
#
AUXILIARY_TITLE_MARKERS = (
    "policy normalization",
    "balance sheet normalization",
    "principles for reducing",
    "plans for reducing",
    "monetary policy implementation",
    "temporary u.s. dollar liquidity arrangements with other central banks",
    "fima repo facility",
    "foreign and international monetary authorities",
)

ISSUE_FIELDS = [
    "severity",
    "code",
    "message",
    "source_agency",
    "event_family",
    "event_timestamp_utc",
    "source_local_date",
    "source_local_time",
    "timestamp_parser",
    "title",
    "url",
    "timestamp_source_url",
]


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def add_issue(items, severity, code, message, row=None):
    issue = {
        "severity": severity,
        "code": code,
        "message": message,
        "source_agency": "",
        "event_family": "",
        "event_timestamp_utc": "",
        "source_local_date": "",
        "source_local_time": "",
        "timestamp_parser": "",
        "title": "",
        "url": "",
        "timestamp_source_url": "",
    }

    if row:
        for key in (
            "source_agency",
            "event_family",
            "event_timestamp_utc",
            "source_local_date",
            "source_local_time",
            "timestamp_parser",
            "title",
            "url",
            "timestamp_source_url",
        ):
            issue[key] = row.get(key, "")

    items.append(issue)


def write_issues(path: Path, rows):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=ISSUE_FIELDS,
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_aware_iso(value: str):
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        raise ValueError("timezone-naive timestamp")

    return dt


def parse_date(value: str):
    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def parse_time(value: str):
    return datetime.strptime(
        value,
        "%H:%M:%S",
    ).time()


def main():
    rows = read_csv(INPUT)

    errors = []
    warnings = []

    print("FOMC canonical invariant/coverage validation")
    print()

    if not rows:
        print("ERROR: canonical FOMC file is empty")
        return 1

    required_columns = {
        "source_agency",
        "event_family",
        "event_timestamp_utc",
        "source_local_date",
        "source_local_time",
        "source_timezone",
        "title",
        "url",
        "timestamp_parser",
        "timestamp_source_url",
    }

    missing_columns = sorted(
        required_columns - set(rows[0].keys())
    )

    if missing_columns:
        print(
            "ERROR: missing required columns:",
            ", ".join(missing_columns),
        )
        return 1

    #
    # 1. Exact canonical row count.
    #
    if len(rows) != EXPECTED_ROWS:
        add_issue(
            errors,
            "error",
            "unexpected_total_count",
            f"Expected exactly {EXPECTED_ROWS} canonical rows, "
            f"found {len(rows)}",
        )

    seen_keys = defaultdict(list)
    year_counts = Counter()
    time_counts = Counter()
    parser_counts = Counter()

    direct_unusual_found = set()

    #
    # Row-level invariants.
    #
    for row in rows:
        agency = row["source_agency"].strip()
        family = row["event_family"].strip()
        utc_raw = row["event_timestamp_utc"].strip()
        local_date_raw = row["source_local_date"].strip()
        local_time_raw = row["source_local_time"].strip()
        source_timezone = row["source_timezone"].strip()
        title = row["title"].strip()
        url = row["url"].strip()
        parser_name = row["timestamp_parser"].strip()
        timestamp_source_url = row.get(
            "timestamp_source_url",
            "",
        ).strip()

        #
        # 2. Agency/family.
        #
        if agency != EXPECTED_AGENCY:
            add_issue(
                errors,
                "error",
                "unexpected_agency",
                f"Expected {EXPECTED_AGENCY}, found {agency!r}",
                row,
            )

        if family != EXPECTED_FAMILY:
            add_issue(
                errors,
                "error",
                "unexpected_family",
                f"Expected {EXPECTED_FAMILY}, found {family!r}",
                row,
            )

        #
        # 6. Nonempty title / Federal Reserve URL.
        #
        if not title:
            add_issue(
                errors,
                "error",
                "empty_title",
                "FOMC title is empty",
                row,
            )

        if not url:
            add_issue(
                errors,
                "error",
                "empty_url",
                "FOMC URL is empty",
                row,
            )
        elif not url.startswith(
            "https://www.federalreserve.gov/"
        ):
            add_issue(
                errors,
                "error",
                "unexpected_url_domain",
                "URL is not an official federalreserve.gov URL",
                row,
            )

        #
        # 7. Timestamp parser provenance.
        #
        if parser_name not in ALLOWED_PARSERS:
            add_issue(
                errors,
                "error",
                "unexpected_timestamp_parser",
                f"Unexpected timestamp_parser={parser_name!r}",
                row,
            )

        #
        # 8. Minutes-recovered rows require official source URL.
        #
        if parser_name == "fomc_minutes_statement_release":
            if not timestamp_source_url:
                add_issue(
                    errors,
                    "error",
                    "missing_minutes_timestamp_source_url",
                    "Minutes-recovered event has no timestamp_source_url",
                    row,
                )
            elif not timestamp_source_url.startswith(
                "https://www.federalreserve.gov/"
            ):
                add_issue(
                    errors,
                    "error",
                    "invalid_minutes_timestamp_source_url",
                    "Minutes timestamp source is not a federalreserve.gov URL",
                    row,
                )

        #
        # 3. Parse timestamps/date/time and enforce UTC/local consistency.
        #
        try:
            utc_dt = parse_aware_iso(utc_raw)
        except Exception as exc:
            add_issue(
                errors,
                "error",
                "invalid_utc_timestamp",
                f"Invalid event_timestamp_utc: {exc}",
                row,
            )
            continue

        try:
            local_date = parse_date(local_date_raw)
        except Exception as exc:
            add_issue(
                errors,
                "error",
                "invalid_local_date",
                f"Invalid source_local_date: {exc}",
                row,
            )
            continue

        try:
            local_time = parse_time(local_time_raw)
        except Exception as exc:
            add_issue(
                errors,
                "error",
                "invalid_local_time",
                f"Invalid source_local_time: {exc}",
                row,
            )
            continue

        if source_timezone != EXPECTED_TZ:
            add_issue(
                errors,
                "error",
                "unexpected_timezone",
                f"Expected source_timezone={EXPECTED_TZ}, "
                f"found {source_timezone!r}",
                row,
            )

        converted = utc_dt.astimezone(EASTERN)

        if converted.date() != local_date:
            add_issue(
                errors,
                "error",
                "utc_local_date_mismatch",
                f"UTC timestamp converts to {converted.date()}, "
                f"not {local_date}",
                row,
            )

        if (
            converted.time().replace(tzinfo=None)
            != local_time
        ):
            add_issue(
                errors,
                "error",
                "utc_local_time_mismatch",
                f"UTC timestamp converts to "
                f"{converted.time().replace(tzinfo=None)}, "
                f"not {local_time}",
                row,
            )

        #
        # 4. Requested date window.
        #
        if not START_DATE <= local_date <= END_DATE:
            add_issue(
                errors,
                "error",
                "outside_requested_window",
                f"Release date {local_date} is outside "
                f"{START_DATE}..{END_DATE}",
                row,
            )

        #
        # 5. Canonical uniqueness.
        #
        key = (
            agency,
            family,
            utc_raw,
        )
        seen_keys[key].append(row)

        year_counts[local_date.year] += 1
        time_counts[local_time_raw] += 1
        parser_counts[parser_name] += 1

        #
        # 9. Track exact directly observed unusual releases.
        #
        unusual_key = (
            local_date_raw,
            local_time_raw,
        )

        if unusual_key in EXPECTED_DIRECT_UNUSUAL:
            if parser_name != "for_release_at":
                add_issue(
                    errors,
                    "error",
                    "unusual_event_not_directly_observed",
                    f"Expected unusual event {unusual_key} to use "
                    f"for_release_at provenance",
                    row,
                )

            direct_unusual_found.add(unusual_key)

        #
        # 12. Reject auxiliary-content leakage by title.
        #
        title_lower = title.lower()

        for marker in AUXILIARY_TITLE_MARKERS:
            if marker in title_lower:
                add_issue(
                    errors,
                    "error",
                    "auxiliary_release_leaked_into_canonical",
                    f"Canonical title contains auxiliary marker "
                    f"{marker!r}",
                    row,
                )
                break

    #
    # 5. Duplicate canonical keys.
    #
    for key, items in seen_keys.items():
        if len(items) > 1:
            for row in items:
                add_issue(
                    errors,
                    "error",
                    "duplicate_canonical_key",
                    "Duplicate "
                    "(source_agency,event_family,event_timestamp_utc)",
                    row,
                )

    #
    # 9. All four direct unusual releases must be present exactly once.
    #
    for expected in sorted(EXPECTED_DIRECT_UNUSUAL):
        matching = [
            row
            for row in rows
            if (
                row["source_local_date"],
                row["source_local_time"],
            )
            == expected
        ]

        if len(matching) != 1:
            add_issue(
                errors,
                "error",
                "unexpected_direct_unusual_release_count",
                f"Expected exactly one direct unusual release "
                f"{expected[0]} {expected[1]}, found {len(matching)}",
                matching[0] if matching else None,
            )

    if direct_unusual_found != EXPECTED_DIRECT_UNUSUAL:
        missing = sorted(
            EXPECTED_DIRECT_UNUSUAL
            - direct_unusual_found
        )

        if missing:
            add_issue(
                errors,
                "error",
                "missing_direct_unusual_release",
                "Missing direct unusual release(s): "
                + ", ".join(
                    f"{d} {t}"
                    for d, t in missing
                ),
            )

    #
    # 10. Exact historical release-time distribution.
    #
    unexpected_times = sorted(
        set(time_counts)
        - set(EXPECTED_TIME_COUNTS)
    )

    if unexpected_times:
        add_issue(
            errors,
            "error",
            "unexpected_release_time",
            "Unexpected FOMC release time(s): "
            + ", ".join(unexpected_times),
        )

    for release_time, expected_count in (
        EXPECTED_TIME_COUNTS.items()
    ):
        actual_count = time_counts[release_time]

        if actual_count != expected_count:
            add_issue(
                errors,
                "error",
                "release_time_count_mismatch",
                f"{release_time}: expected {expected_count}, "
                f"found {actual_count}",
            )

    #
    # 11. Exact year-count coverage.
    #
    count_rows = []

    for year in range(2010, 2027):
        expected = EXPECTED_YEAR_COUNTS[year]
        actual = year_counts[year]

        if actual != expected:
            add_issue(
                errors,
                "error",
                "year_count_mismatch",
                f"{year}: expected {expected} canonical FOMC events, "
                f"found {actual}",
            )

        if year in IRREGULAR_YEAR_NOTES:
            status = "documented_irregular"
            note = IRREGULAR_YEAR_NOTES[year]
        else:
            status = "normal"
            note = ""

            if expected != 8:
                add_issue(
                    errors,
                    "error",
                    "undocumented_irregular_year",
                    f"{year} expected count {expected} but is not "
                    f"documented as irregular",
                )

        count_rows.append({
            "year": year,
            "expected_count": expected,
            "actual_count": actual,
            "status": status,
            "note": note,
        })

    #
    # 12. Exact URL exclusion using manifest_auxiliary.csv.
    #
    if not AUXILIARY.exists():
        add_issue(
            errors,
            "error",
            "missing_auxiliary_manifest",
            f"Required auxiliary manifest is missing: {AUXILIARY}",
        )
    else:
        auxiliary_rows = read_csv(AUXILIARY)

        auxiliary_urls = {
            row.get("url", "").strip()
            for row in auxiliary_rows
            if row.get("url", "").strip()
        }

        canonical_urls = {
            row["url"].strip()
            for row in rows
            if row["url"].strip()
        }

        leaked_urls = sorted(
            auxiliary_urls & canonical_urls
        )

        for leaked_url in leaked_urls:
            matching = next(
                (
                    row
                    for row in rows
                    if row["url"].strip() == leaked_url
                ),
                None,
            )

            add_issue(
                errors,
                "error",
                "auxiliary_url_leaked_into_canonical",
                f"Rejected auxiliary URL appears in canonical set: "
                f"{leaked_url}",
                matching,
            )

    #
    # Write reports.
    #
    write_issues(ERRORS, errors)
    write_issues(WARNINGS, warnings)

    with YEAR_COUNTS.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "year",
                "expected_count",
                "actual_count",
                "status",
                "note",
            ],
        )
        writer.writeheader()
        writer.writerows(count_rows)

    with TIME_COUNTS.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "source_local_time",
                "expected_count",
                "actual_count",
                "status",
            ],
        )
        writer.writeheader()

        all_times = sorted(
            set(EXPECTED_TIME_COUNTS)
            | set(time_counts)
        )

        for release_time in all_times:
            expected = EXPECTED_TIME_COUNTS.get(
                release_time,
                0,
            )
            actual = time_counts.get(
                release_time,
                0,
            )

            writer.writerow({
                "source_local_time":
                    release_time,
                "expected_count":
                    expected,
                "actual_count":
                    actual,
                "status":
                    (
                        "match"
                        if actual == expected
                        else "mismatch"
                    ),
            })

    duplicate_key_count = sum(
        1
        for values in seen_keys.values()
        if len(values) > 1
    )

    print(f"Canonical rows       : {len(rows)}")
    print(f"Duplicate keys       : {duplicate_key_count}")
    print(
        f"Direct unusual events: "
        f"{len(direct_unusual_found)}/"
        f"{len(EXPECTED_DIRECT_UNUSUAL)}"
    )
    print(f"Validation errors    : {len(errors)}")
    print(f"Validation warnings  : {len(warnings)}")
    print()

    print("Release-time distribution:")

    for release_time in sorted(time_counts):
        print(
            f"  {release_time:<10} "
            f"{time_counts[release_time]:>4}"
        )

    print()
    print("Timestamp provenance:")

    for parser_name in sorted(parser_counts):
        print(
            f"  {parser_name:<34} "
            f"{parser_counts[parser_name]:>4}"
        )

    print()
    print(f"Errors               : {ERRORS}")
    print(f"Warnings             : {WARNINGS}")
    print(f"Year counts          : {YEAR_COUNTS}")
    print(f"Time counts          : {TIME_COUNTS}")
    print()

    if errors:
        print(
            "RESULT: FAIL - do not import FOMC events "
            "into economic_event"
        )
        return 1

    print(
        "RESULT: PASS - canonical FOMC dataset satisfies "
        "all invariant and coverage checks"
    )
    print("No database writes were performed.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
