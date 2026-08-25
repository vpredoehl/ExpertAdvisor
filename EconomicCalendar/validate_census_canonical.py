#!/usr/bin/env python3

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from datetime import date, datetime, timedelta
from pathlib import Path
from urllib.parse import urlparse
from zoneinfo import ZoneInfo


ROOT = Path("EconomicCalendar/raw/census")

INPUT = ROOT / "census_events_extracted.csv"

ERRORS = ROOT / "census_validation_errors.csv"
YEAR_COUNTS = ROOT / "census_validation_year_counts.csv"
REFERENCE_COVERAGE = ROOT / "census_validation_reference_coverage.csv"
PARSER_COUNTS = ROOT / "census_validation_parser_counts.csv"
TIMEZONE_COUNTS = ROOT / "census_validation_timezone_counts.csv"


EXPECTED_ROWS = 399

EXPECTED_FAMILY_COUNTS = {
    "RETAIL_SALES": 200,
    "DURABLE_GOODS": 199,
}

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EXPECTED_AGENCY = "CENSUS"
EXPECTED_TIMEZONE = "America/New_York"
EXPECTED_LOCAL_TIME = "08:30:00"

EASTERN = ZoneInfo(EXPECTED_TIMEZONE)


#
# Exact parser universe established by the authoritative extraction.
#
ALLOWED_PARSERS = {
    "census_for_immediate_release",
    "census_for_immediate_release_malformed_text_repair",
    "census_for_release_at",
    "census_durable_release_schedule",
}

EXPECTED_PARSER_COUNTS = {
    "census_for_immediate_release": 164,
    "census_for_immediate_release_malformed_text_repair": 5,
    "census_for_release_at": 229,
    "census_durable_release_schedule": 1,
}


#
# Exact EST / EDT distribution from the authoritative 399-row snapshot.
#
EXPECTED_TIMEZONE_TOKEN_COUNTS = {
    "EST": 133,
    "EDT": 266,
}


#
# The only five rows for which narrowly repaired pdftotext text was
# required. These identities are pinned so that the repair mechanism
# cannot silently expand to additional releases later.
#
EXPECTED_MALFORMED_REPAIRS = {
    ("DURABLE_GOODS", 2015, 3),
    ("DURABLE_GOODS", 2015, 5),
    ("DURABLE_GOODS", 2015, 6),
    ("DURABLE_GOODS", 2015, 8),
    ("DURABLE_GOODS", 2015, 9),
}


#
# The one and only allowed schedule-derived timestamp.
#
EXPECTED_SCHEDULE_ROW = {
    "event_family": "DURABLE_GOODS",
    "reference_year": 2026,
    "reference_month": 6,
    "source_local_date": "2026-07-27",
    "source_local_time": "08:30:00",
    "timestamp_source_url": (
        "https://www.census.gov/"
        "manufacturing/m3/release_schedule.html"
    ),
}


#
# Release-year counts, not reference-period counts.
#
# 2010..2024 each contain 12 actual release events per family.
#
# 2019 still contains 12 releases per family, but is explicitly marked
# irregular because the 2018-2019 federal funding lapse materially
# shifted the normal release sequence.
#
# 2025 and 2026 are also explicitly irregular because late-2025
# reference periods crossed into the 2026 release year.
#
EXPECTED_RELEASE_YEAR_COUNTS = {
    "RETAIL_SALES": {
        **{
            year: 12
            for year in range(2010, 2025)
        },
        2025: 11,
        2026: 9,
    },
    "DURABLE_GOODS": {
        **{
            year: 12
            for year in range(2010, 2025)
        },
        2025: 11,
        2026: 8,
    },
}

IRREGULAR_YEAR_NOTES = {
    ("RETAIL_SALES", 2019): (
        "Documented federal funding-lapse sequence. "
        "December 2018, January 2019, and February 2019 "
        "reference-period releases were materially delayed."
    ),
    ("DURABLE_GOODS", 2019): (
        "Documented federal funding-lapse sequence. "
        "December 2018, January 2019, and February 2019 "
        "reference-period releases were materially delayed."
    ),
    ("RETAIL_SALES", 2025): (
        "Late-2025 funding disruption shifted the normal release "
        "sequence; only 11 Retail Sales release events occurred "
        "during calendar year 2025."
    ),
    ("DURABLE_GOODS", 2025): (
        "Late-2025 funding disruption shifted the normal release "
        "sequence; only 11 Durable Goods release events occurred "
        "during calendar year 2025."
    ),
    ("RETAIL_SALES", 2026): (
        "Partial release year through 2026-08-24 and includes "
        "late-2025 reference-period releases."
    ),
    ("DURABLE_GOODS", 2026): (
        "Partial release year through 2026-08-24 and includes "
        "late-2025 reference-period releases."
    ),
}


#
# Pin the historically irregular 2019 shutdown sequence.
#
EXPECTED_2019_SHUTDOWN_SEQUENCE = {
    (
        "RETAIL_SALES",
        2018,
        12,
    ): (
        "2019-02-14",
        "08:30:00",
    ),
    (
        "RETAIL_SALES",
        2019,
        1,
    ): (
        "2019-03-11",
        "08:30:00",
    ),
    (
        "RETAIL_SALES",
        2019,
        2,
    ): (
        "2019-04-01",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2018,
        12,
    ): (
        "2019-02-21",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2019,
        1,
    ): (
        "2019-03-13",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2019,
        2,
    ): (
        "2019-04-02",
        "08:30:00",
    ),
}


#
# Pin the late-2025 / 2026 cross-year release sequence.
#
EXPECTED_2025_2026_CROSS_YEAR = {
    (
        "RETAIL_SALES",
        2025,
        9,
    ): (
        "2025-11-25",
        "08:30:00",
    ),
    (
        "RETAIL_SALES",
        2025,
        10,
    ): (
        "2025-12-16",
        "08:30:00",
    ),
    (
        "RETAIL_SALES",
        2025,
        11,
    ): (
        "2026-01-14",
        "08:30:00",
    ),
    (
        "RETAIL_SALES",
        2025,
        12,
    ): (
        "2026-02-10",
        "08:30:00",
    ),

    (
        "DURABLE_GOODS",
        2025,
        9,
    ): (
        "2025-11-26",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2025,
        10,
    ): (
        "2025-12-23",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2025,
        11,
    ): (
        "2026-01-26",
        "08:30:00",
    ),
    (
        "DURABLE_GOODS",
        2025,
        12,
    ): (
        "2026-02-18",
        "08:30:00",
    ),
}


CENSUS_HOSTS = {
    "census.gov",
    "www.census.gov",
    "www2.census.gov",
}


ISSUE_FIELDS = [
    "severity",
    "code",
    "message",
    "source_agency",
    "event_family",
    "event_timestamp_utc",
    "source_local_date",
    "source_local_time",
    "reference_year",
    "reference_month",
    "reference_period",
    "timestamp_parser",
    "filename",
    "url",
    "timestamp_source_url",
]


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
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


def add_issue(
    issues,
    code,
    message,
    row=None,
):
    issue = {
        "severity": "error",
        "code": code,
        "message": message,
        "source_agency": "",
        "event_family": "",
        "event_timestamp_utc": "",
        "source_local_date": "",
        "source_local_time": "",
        "reference_year": "",
        "reference_month": "",
        "reference_period": "",
        "timestamp_parser": "",
        "filename": "",
        "url": "",
        "timestamp_source_url": "",
    }

    if row:
        for field in ISSUE_FIELDS:
            if field in {
                "severity",
                "code",
                "message",
            }:
                continue

            issue[field] = (
                row.get(field)
                or ""
            )

    issues.append(issue)


def parse_aware_utc(
    value: str,
):
    dt = datetime.fromisoformat(
        value
    )

    if dt.tzinfo is None:
        raise ValueError(
            "timezone-naive timestamp"
        )

    if dt.utcoffset() != timedelta(0):
        raise ValueError(
            "event_timestamp_utc does not have UTC offset"
        )

    return dt


def parse_date(
    value: str,
):
    return datetime.strptime(
        value,
        "%Y-%m-%d",
    ).date()


def parse_time(
    value: str,
):
    return datetime.strptime(
        value,
        "%H:%M:%S",
    ).time()


def is_census_url(
    value: str,
):
    if not value:
        return False

    parsed = urlparse(
        value
    )

    return (
        parsed.scheme == "https"
        and (
            parsed.hostname
            or ""
        ).lower()
        in CENSUS_HOSTS
    )


def month_iter(
    first: tuple[int, int],
    last: tuple[int, int],
):
    year, month = first

    while (
        year,
        month,
    ) <= last:
        yield year, month

        month += 1

        if month == 13:
            month = 1
            year += 1


def expected_reference_keys():
    result = set()

    for year, month in month_iter(
        (2009, 12),
        (2026, 7),
    ):
        result.add(
            (
                "RETAIL_SALES",
                year,
                month,
            )
        )

    for year, month in month_iter(
        (2009, 12),
        (2026, 6),
    ):
        result.add(
            (
                "DURABLE_GOODS",
                year,
                month,
            )
        )

    return result


def exact_row_by_reference(
    rows,
    key,
):
    family, year, month = key

    return [
        row
        for row in rows
        if (
            row["event_family"]
            == family
            and int(
                row["reference_year"]
            )
            == year
            and int(
                row["reference_month"]
            )
            == month
        )
    ]


def validate_pinned_sequence(
    rows,
    expected,
    issues,
    code_prefix,
):
    for key, (
        expected_date,
        expected_time,
    ) in expected.items():
        matching = (
            exact_row_by_reference(
                rows,
                key,
            )
        )

        if len(matching) != 1:
            add_issue(
                issues,
                (
                    f"{code_prefix}_"
                    "row_count"
                ),
                (
                    f"{key}: expected exactly "
                    f"one row, found "
                    f"{len(matching)}"
                ),
                (
                    matching[0]
                    if matching
                    else None
                ),
            )
            continue

        row = matching[0]

        if (
            row[
                "source_local_date"
            ]
            != expected_date
            or
            row[
                "source_local_time"
            ]
            != expected_time
        ):
            add_issue(
                issues,
                (
                    f"{code_prefix}_"
                    "timestamp_mismatch"
                ),
                (
                    f"{key}: expected "
                    f"{expected_date} "
                    f"{expected_time}, found "
                    f"{row['source_local_date']} "
                    f"{row['source_local_time']}"
                ),
                row,
            )


def main():
    print(
        "Census canonical invariant/"
        "coverage validation"
    )
    print()

    if not INPUT.exists():
        print(
            f"ERROR: missing canonical input: "
            f"{INPUT}"
        )
        return 1

    rows = read_csv(
        INPUT
    )

    errors = []

    if not rows:
        print(
            "ERROR: canonical Census file "
            "is empty"
        )
        return 1

    required_columns = {
        "source_agency",
        "event_family",
        "event_timestamp_utc",
        "source_local_date",
        "source_local_time",
        "source_timezone",
        "release_timezone_token",
        "reference_year",
        "reference_month",
        "reference_period",
        "title",
        "url",
        "final_url",
        "filename",
        "discovery_source",
        "timestamp_parser",
        "release_text",
        "timestamp_source_url",
    }

    missing_columns = sorted(
        required_columns
        - set(
            rows[0].keys()
        )
    )

    if missing_columns:
        print(
            "ERROR: missing required columns:",
            ", ".join(
                missing_columns
            ),
        )
        return 1

    #
    # 1. Exact total.
    #
    if len(rows) != EXPECTED_ROWS:
        add_issue(
            errors,
            "unexpected_total_count",
            (
                f"Expected exactly "
                f"{EXPECTED_ROWS} rows, "
                f"found {len(rows)}"
            ),
        )

    family_counts = Counter()
    parser_counts = Counter()
    timezone_counts = Counter()
    release_year_counts = Counter()

    canonical_keys = defaultdict(
        list
    )
    reference_keys = defaultdict(
        list
    )

    malformed_rows = []
    schedule_rows = []

    #
    # Row-level invariants.
    #
    for row in rows:
        agency = (
            row["source_agency"]
            .strip()
        )
        family = (
            row["event_family"]
            .strip()
        )
        timestamp_raw = (
            row["event_timestamp_utc"]
            .strip()
        )
        local_date_raw = (
            row["source_local_date"]
            .strip()
        )
        local_time_raw = (
            row["source_local_time"]
            .strip()
        )
        source_timezone = (
            row["source_timezone"]
            .strip()
        )
        timezone_token = (
            row["release_timezone_token"]
            .strip()
            .upper()
        )
        parser_name = (
            row["timestamp_parser"]
            .strip()
        )
        title = (
            row["title"]
            .strip()
        )
        url = (
            row["url"]
            .strip()
        )
        final_url = (
            row["final_url"]
            .strip()
        )
        timestamp_source_url = (
            row["timestamp_source_url"]
            .strip()
        )
        reference_period = (
            row["reference_period"]
            .strip()
        )
        filename = (
            row["filename"]
            .strip()
        )

        #
        # 2. Family universe and counts.
        #
        if family not in (
            EXPECTED_FAMILY_COUNTS
        ):
            add_issue(
                errors,
                "unexpected_family",
                (
                    f"Unexpected Census "
                    f"event_family={family!r}"
                ),
                row,
            )

        family_counts[
            family
        ] += 1

        #
        # 3. Agency.
        #
        if agency != EXPECTED_AGENCY:
            add_issue(
                errors,
                "unexpected_agency",
                (
                    f"Expected "
                    f"{EXPECTED_AGENCY}, "
                    f"found {agency!r}"
                ),
                row,
            )

        #
        # Reference key.
        #
        try:
            reference_year = int(
                row["reference_year"]
            )
            reference_month = int(
                row["reference_month"]
            )

            if not (
                1
                <= reference_month
                <= 12
            ):
                raise ValueError(
                    "month outside 1..12"
                )

        except Exception as exc:
            add_issue(
                errors,
                "invalid_reference_period_key",
                (
                    "Invalid reference "
                    f"year/month: {exc}"
                ),
                row,
            )
            continue

        reference_key = (
            family,
            reference_year,
            reference_month,
        )

        reference_keys[
            reference_key
        ].append(
            row
        )

        #
        # 8. Required textual provenance.
        #
        if not title:
            add_issue(
                errors,
                "empty_title",
                "title is empty",
                row,
            )

        if not reference_period:
            add_issue(
                errors,
                "empty_reference_period",
                (
                    "reference_period "
                    "is empty"
                ),
                row,
            )

        if not url:
            add_issue(
                errors,
                "empty_url",
                "url is empty",
                row,
            )
        elif not is_census_url(
            url
        ):
            add_issue(
                errors,
                "invalid_census_url",
                (
                    "url is not an "
                    "official HTTPS Census URL"
                ),
                row,
            )

        if not final_url:
            add_issue(
                errors,
                "empty_final_url",
                "final_url is empty",
                row,
            )
        elif not is_census_url(
            final_url
        ):
            add_issue(
                errors,
                "invalid_final_census_url",
                (
                    "final_url is not an "
                    "official HTTPS Census URL"
                ),
                row,
            )

        if not timestamp_source_url:
            add_issue(
                errors,
                "empty_timestamp_source_url",
                (
                    "timestamp_source_url "
                    "is empty"
                ),
                row,
            )
        elif not is_census_url(
            timestamp_source_url
        ):
            add_issue(
                errors,
                "invalid_timestamp_source_url",
                (
                    "timestamp_source_url "
                    "is not an official "
                    "HTTPS Census URL"
                ),
                row,
            )

        #
        # 9. Parser universe.
        #
        if parser_name not in (
            ALLOWED_PARSERS
        ):
            add_issue(
                errors,
                "unexpected_timestamp_parser",
                (
                    "Unexpected "
                    f"timestamp_parser="
                    f"{parser_name!r}"
                ),
                row,
            )

        parser_counts[
            parser_name
        ] += 1

        if (
            parser_name
            ==
            "census_for_immediate_release_malformed_text_repair"
        ):
            malformed_rows.append(
                row
            )

        if (
            parser_name
            ==
            "census_durable_release_schedule"
        ):
            schedule_rows.append(
                row
            )

        #
        # 4. Timestamp awareness / UTC / local consistency.
        #
        try:
            utc_dt = (
                parse_aware_utc(
                    timestamp_raw
                )
            )
        except Exception as exc:
            add_issue(
                errors,
                "invalid_utc_timestamp",
                (
                    "Invalid "
                    "event_timestamp_utc: "
                    f"{exc}"
                ),
                row,
            )
            continue

        try:
            local_date = (
                parse_date(
                    local_date_raw
                )
            )
        except Exception as exc:
            add_issue(
                errors,
                "invalid_local_date",
                (
                    "Invalid "
                    "source_local_date: "
                    f"{exc}"
                ),
                row,
            )
            continue

        try:
            local_time = (
                parse_time(
                    local_time_raw
                )
            )
        except Exception as exc:
            add_issue(
                errors,
                "invalid_local_time",
                (
                    "Invalid "
                    "source_local_time: "
                    f"{exc}"
                ),
                row,
            )
            continue

        if (
            source_timezone
            != EXPECTED_TIMEZONE
        ):
            add_issue(
                errors,
                "unexpected_timezone",
                (
                    "Expected "
                    f"{EXPECTED_TIMEZONE}, "
                    f"found "
                    f"{source_timezone!r}"
                ),
                row,
            )

        converted = (
            utc_dt.astimezone(
                EASTERN
            )
        )

        if (
            converted.date()
            != local_date
        ):
            add_issue(
                errors,
                "utc_local_date_mismatch",
                (
                    "UTC timestamp converts "
                    f"to {converted.date()}, "
                    f"not {local_date}"
                ),
                row,
            )

        converted_time = (
            converted.time()
            .replace(
                tzinfo=None
            )
        )

        if (
            converted_time
            != local_time
        ):
            add_issue(
                errors,
                "utc_local_time_mismatch",
                (
                    "UTC timestamp converts "
                    f"to {converted_time}, "
                    f"not {local_time}"
                ),
                row,
            )

        #
        # 5. Release window.
        #
        if not (
            START_DATE
            <= local_date
            <= END_DATE
        ):
            add_issue(
                errors,
                "outside_release_window",
                (
                    f"Release date "
                    f"{local_date} outside "
                    f"{START_DATE}.."
                    f"{END_DATE}"
                ),
                row,
            )

        #
        # 6. Canonical uniqueness.
        #
        canonical_key = (
            agency,
            family,
            timestamp_raw,
        )

        canonical_keys[
            canonical_key
        ].append(
            row
        )

        #
        # 12. Exact 08:30 release time.
        #
        if (
            local_time_raw
            != EXPECTED_LOCAL_TIME
        ):
            add_issue(
                errors,
                "unexpected_release_time",
                (
                    "Expected Census release "
                    f"time {EXPECTED_LOCAL_TIME}, "
                    f"found {local_time_raw!r}"
                ),
                row,
            )

        #
        # 13. EST / EDT token consistency.
        #
        if timezone_token not in {
            "EST",
            "EDT",
        }:
            add_issue(
                errors,
                "unexpected_timezone_token",
                (
                    "Expected EST or EDT, "
                    f"found {timezone_token!r}"
                ),
                row,
            )
        else:
            timezone_counts[
                timezone_token
            ] += 1

            if (
                converted.tzname()
                != timezone_token
            ):
                add_issue(
                    errors,
                    "timezone_token_date_mismatch",
                    (
                        f"Stored token "
                        f"{timezone_token} "
                        "does not agree with "
                        "America/New_York="
                        f"{converted.tzname()} "
                        f"on {local_date}"
                    ),
                    row,
                )

        release_year_counts[
            (
                family,
                local_date.year,
            )
        ] += 1

        #
        # 15. Artifact/parser restrictions.
        #
        suffix = (
            Path(filename)
            .suffix
            .lower()
        )

        if (
            parser_name
            ==
            "census_durable_release_schedule"
        ):
            if suffix not in {
                ".html",
                ".htm",
            }:
                add_issue(
                    errors,
                    "schedule_row_not_html",
                    (
                        "Schedule-derived row "
                        "must use the archived "
                        "Durable current HTML "
                        "artifact"
                    ),
                    row,
                )
        else:
            if suffix != ".pdf":
                add_issue(
                    errors,
                    "unexpected_non_pdf_artifact",
                    (
                        "Only the sole June 2026 "
                        "Durable schedule-derived "
                        "row may use HTML; all "
                        "other canonical rows "
                        "must be PDF-derived"
                    ),
                    row,
                )

    #
    # 2. Exact family counts.
    #
    for (
        family,
        expected_count,
    ) in (
        EXPECTED_FAMILY_COUNTS.items()
    ):
        actual = (
            family_counts[
                family
            ]
        )

        if actual != expected_count:
            add_issue(
                errors,
                "family_count_mismatch",
                (
                    f"{family}: expected "
                    f"{expected_count}, "
                    f"found {actual}"
                ),
            )

    unexpected_family_counts = (
        set(
            family_counts
        )
        - set(
            EXPECTED_FAMILY_COUNTS
        )
    )

    if unexpected_family_counts:
        add_issue(
            errors,
            "unexpected_family_set",
            (
                "Unexpected family/families: "
                + ", ".join(
                    sorted(
                        unexpected_family_counts
                    )
                )
            ),
        )

    #
    # 6. Duplicate canonical keys.
    #
    duplicate_key_count = 0

    for key, matches in (
        canonical_keys.items()
    ):
        if len(matches) > 1:
            duplicate_key_count += 1

            for row in matches:
                add_issue(
                    errors,
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
    # 7. Complete family/reference-month grid.
    #
    expected_refs = (
        expected_reference_keys()
    )

    actual_refs = set(
        reference_keys
    )

    missing_refs = sorted(
        expected_refs
        - actual_refs
    )

    unexpected_refs = sorted(
        actual_refs
        - expected_refs
    )

    for key in missing_refs:
        add_issue(
            errors,
            "missing_reference_month",
            (
                "Missing expected "
                "family/reference month: "
                f"{key}"
            ),
        )

    for key in unexpected_refs:
        matching = (
            reference_keys[
                key
            ]
        )

        add_issue(
            errors,
            "unexpected_reference_month",
            (
                "Unexpected family/"
                "reference month: "
                f"{key}"
            ),
            (
                matching[0]
                if matching
                else None
            ),
        )

    for key in sorted(
        expected_refs
        & actual_refs
    ):
        matching = (
            reference_keys[
                key
            ]
        )

        if len(matching) != 1:
            for row in matching:
                add_issue(
                    errors,
                    "duplicate_reference_month",
                    (
                        "Expected exactly "
                        "one row for "
                        f"{key}, found "
                        f"{len(matching)}"
                    ),
                    row,
                )

    #
    # 9. Exact parser counts.
    #
    for (
        parser_name,
        expected_count,
    ) in (
        EXPECTED_PARSER_COUNTS.items()
    ):
        actual = (
            parser_counts[
                parser_name
            ]
        )

        if actual != expected_count:
            add_issue(
                errors,
                "parser_count_mismatch",
                (
                    f"{parser_name}: "
                    f"expected "
                    f"{expected_count}, "
                    f"found {actual}"
                ),
            )

    unexpected_parsers = (
        set(
            parser_counts
        )
        - set(
            EXPECTED_PARSER_COUNTS
        )
    )

    if unexpected_parsers:
        add_issue(
            errors,
            "unexpected_parser_set",
            (
                "Unexpected parser(s): "
                + ", ".join(
                    sorted(
                        unexpected_parsers
                    )
                )
            ),
        )

    #
    # 10. Exact malformed repair identities.
    #
    actual_malformed = {
        (
            row[
                "event_family"
            ],
            int(
                row[
                    "reference_year"
                ]
            ),
            int(
                row[
                    "reference_month"
                ]
            ),
        )
        for row in malformed_rows
    }

    if (
        len(malformed_rows)
        != 5
    ):
        add_issue(
            errors,
            "malformed_repair_count_mismatch",
            (
                "Expected exactly 5 "
                "malformed-text repairs, "
                f"found {len(malformed_rows)}"
            ),
        )

    if (
        actual_malformed
        != EXPECTED_MALFORMED_REPAIRS
    ):
        missing = sorted(
            EXPECTED_MALFORMED_REPAIRS
            - actual_malformed
        )

        extra = sorted(
            actual_malformed
            - EXPECTED_MALFORMED_REPAIRS
        )

        add_issue(
            errors,
            "malformed_repair_identity_mismatch",
            (
                "Malformed repair identities "
                f"changed. missing={missing}, "
                f"extra={extra}"
            ),
        )

    #
    # 11. Sole schedule-derived row.
    #
    if len(schedule_rows) != 1:
        add_issue(
            errors,
            "schedule_row_count_mismatch",
            (
                "Expected exactly one "
                "schedule-derived row, "
                f"found {len(schedule_rows)}"
            ),
        )
    else:
        row = schedule_rows[0]

        checks = {
            "event_family":
                EXPECTED_SCHEDULE_ROW[
                    "event_family"
                ],
            "source_local_date":
                EXPECTED_SCHEDULE_ROW[
                    "source_local_date"
                ],
            "source_local_time":
                EXPECTED_SCHEDULE_ROW[
                    "source_local_time"
                ],
            "timestamp_source_url":
                EXPECTED_SCHEDULE_ROW[
                    "timestamp_source_url"
                ],
        }

        for field, expected in (
            checks.items()
        ):
            if (
                row[field]
                != expected
            ):
                add_issue(
                    errors,
                    "schedule_row_identity_mismatch",
                    (
                        f"Schedule row "
                        f"{field}: expected "
                        f"{expected!r}, found "
                        f"{row[field]!r}"
                    ),
                    row,
                )

        try:
            ref_year = int(
                row[
                    "reference_year"
                ]
            )
            ref_month = int(
                row[
                    "reference_month"
                ]
            )
        except Exception:
            ref_year = None
            ref_month = None

        if (
            ref_year
            != EXPECTED_SCHEDULE_ROW[
                "reference_year"
            ]
            or ref_month
            != EXPECTED_SCHEDULE_ROW[
                "reference_month"
            ]
        ):
            add_issue(
                errors,
                "schedule_reference_mismatch",
                (
                    "Schedule-derived row "
                    "must be exactly "
                    "DURABLE_GOODS / "
                    "June 2026"
                ),
                row,
            )

    #
    # 13. Exact timezone token distribution.
    #
    for token, expected in (
        EXPECTED_TIMEZONE_TOKEN_COUNTS.items()
    ):
        actual = (
            timezone_counts[
                token
            ]
        )

        if actual != expected:
            add_issue(
                errors,
                "timezone_token_count_mismatch",
                (
                    f"{token}: expected "
                    f"{expected}, found "
                    f"{actual}"
                ),
            )

    #
    # 14. Exact release-year counts.
    #
    year_report_rows = []

    for family in (
        "RETAIL_SALES",
        "DURABLE_GOODS",
    ):
        for year in range(
            2010,
            2027,
        ):
            expected = (
                EXPECTED_RELEASE_YEAR_COUNTS[
                    family
                ][
                    year
                ]
            )

            actual = (
                release_year_counts[
                    (
                        family,
                        year,
                    )
                ]
            )

            note = (
                IRREGULAR_YEAR_NOTES.get(
                    (
                        family,
                        year,
                    ),
                    "",
                )
            )

            if actual != expected:
                status = (
                    "count_mismatch"
                )

                add_issue(
                    errors,
                    "release_year_count_mismatch",
                    (
                        f"{family} {year}: "
                        f"expected {expected}, "
                        f"found {actual}"
                    ),
                )

            elif note:
                status = (
                    "documented_irregular"
                )

            else:
                status = "complete"

            year_report_rows.append({
                "year":
                    year,
                "event_family":
                    family,
                "expected_release_count":
                    expected,
                "actual_release_count":
                    actual,
                "status":
                    status,
                "note":
                    note,
            })

    #
    # Pin irregular historical sequences rather than merely documenting
    # their aggregate year counts.
    #
    validate_pinned_sequence(
        rows,
        EXPECTED_2019_SHUTDOWN_SEQUENCE,
        errors,
        "shutdown_2019",
    )

    validate_pinned_sequence(
        rows,
        EXPECTED_2025_2026_CROSS_YEAR,
        errors,
        "cross_year_2025_2026",
    )

    #
    # Reference coverage report.
    #
    reference_report_rows = []

    all_reference_keys = sorted(
        expected_refs
        | actual_refs
    )

    for (
        family,
        year,
        month,
    ) in all_reference_keys:
        count = len(
            reference_keys.get(
                (
                    family,
                    year,
                    month,
                ),
                [],
            )
        )

        expected = (
            1
            if (
                family,
                year,
                month,
            )
            in expected_refs
            else 0
        )

        reference_report_rows.append({
            "event_family":
                family,
            "reference_year":
                year,
            "reference_month":
                f"{month:02d}",
            "expected_count":
                expected,
            "actual_count":
                count,
            "status":
                (
                    "match"
                    if count == expected
                    else "mismatch"
                ),
        })

    #
    # Reports.
    #
    write_csv(
        ERRORS,
        ISSUE_FIELDS,
        errors,
    )

    write_csv(
        YEAR_COUNTS,
        [
            "year",
            "event_family",
            "expected_release_count",
            "actual_release_count",
            "status",
            "note",
        ],
        year_report_rows,
    )

    write_csv(
        REFERENCE_COVERAGE,
        [
            "event_family",
            "reference_year",
            "reference_month",
            "expected_count",
            "actual_count",
            "status",
        ],
        reference_report_rows,
    )

    parser_report_rows = []

    for parser_name in sorted(
        set(
            EXPECTED_PARSER_COUNTS
        )
        | set(
            parser_counts
        )
    ):
        expected = (
            EXPECTED_PARSER_COUNTS.get(
                parser_name,
                0,
            )
        )

        actual = (
            parser_counts.get(
                parser_name,
                0,
            )
        )

        parser_report_rows.append({
            "timestamp_parser":
                parser_name,
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

    write_csv(
        PARSER_COUNTS,
        [
            "timestamp_parser",
            "expected_count",
            "actual_count",
            "status",
        ],
        parser_report_rows,
    )

    timezone_report_rows = []

    for token in sorted(
        set(
            EXPECTED_TIMEZONE_TOKEN_COUNTS
        )
        | set(
            timezone_counts
        )
    ):
        expected = (
            EXPECTED_TIMEZONE_TOKEN_COUNTS.get(
                token,
                0,
            )
        )

        actual = (
            timezone_counts.get(
                token,
                0,
            )
        )

        timezone_report_rows.append({
            "release_timezone_token":
                token,
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

    write_csv(
        TIMEZONE_COUNTS,
        [
            "release_timezone_token",
            "expected_count",
            "actual_count",
            "status",
        ],
        timezone_report_rows,
    )

    #
    # Human-readable summary.
    #
    print(
        f"Canonical rows       : "
        f"{len(rows)}"
    )
    print(
        f"Retail Sales         : "
        f"{family_counts['RETAIL_SALES']}"
    )
    print(
        f"Durable Goods        : "
        f"{family_counts['DURABLE_GOODS']}"
    )
    print(
        f"Duplicate keys       : "
        f"{duplicate_key_count}"
    )
    print(
        f"Malformed repairs    : "
        f"{len(malformed_rows)}"
    )
    print(
        f"Schedule-derived     : "
        f"{len(schedule_rows)}"
    )
    print(
        f"Validation errors    : "
        f"{len(errors)}"
    )
    print()

    print(
        "Timestamp parser distribution:"
    )

    for parser_name in sorted(
        parser_counts
    ):
        print(
            f"  {parser_name:<58} "
            f"{parser_counts[parser_name]:>4}"
        )

    print()
    print(
        "Release-time distribution:"
    )

    local_time_counts = Counter(
        row[
            "source_local_time"
        ]
        for row in rows
    )

    for release_time, count in sorted(
        local_time_counts.items()
    ):
        print(
            f"  {release_time:<10} "
            f"{count:>4}"
        )

    print()
    print(
        "Release timezone tokens:"
    )

    for token, count in sorted(
        timezone_counts.items()
    ):
        print(
            f"  {token:<6} "
            f"{count:>4}"
        )

    print()
    print(
        "Release-year distribution:"
    )
    print()
    print(
        f"{'YEAR':<6} "
        f"{'RETAIL':>8} "
        f"{'DURABLE':>8}"
    )
    print(
        "-" * 26
    )

    for year in range(
        2010,
        2027,
    ):
        print(
            f"{year:<6} "
            f"{release_year_counts[('RETAIL_SALES', year)]:>8} "
            f"{release_year_counts[('DURABLE_GOODS', year)]:>8}"
        )

    print()
    print(
        f"Errors             : "
        f"{ERRORS}"
    )
    print(
        f"Year counts        : "
        f"{YEAR_COUNTS}"
    )
    print(
        f"Reference coverage : "
        f"{REFERENCE_COVERAGE}"
    )
    print(
        f"Parser counts      : "
        f"{PARSER_COUNTS}"
    )
    print(
        f"Timezone counts    : "
        f"{TIMEZONE_COUNTS}"
    )
    print()

    if errors:
        print(
            "RESULT: FAIL - do not import "
            "Census events into economic_event"
        )
        print(
            "No database writes were performed."
        )
        return 1

    print(
        "RESULT: PASS - canonical Census dataset "
        "satisfies all invariant and coverage checks"
    )
    print(
        "All 399 release timestamps remain "
        "authoritatively constrained."
    )
    print(
        "No database writes were performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
