#!/usr/bin/env python3

from __future__ import annotations

import csv
import html
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import date, datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path("EconomicCalendar/raw/census")

MANIFEST = ROOT / "manifest.csv"

RETAIL_DIR = (
    ROOT
    / "releases"
    / "retail_sales"
)

DURABLE_DIR = (
    ROOT
    / "releases"
    / "durable_goods"
)

DURABLE_SCHEDULE = (
    ROOT
    / "index_pages"
    / "durable_schedule.html"
)

OUTPUT = (
    ROOT
    / "census_events_extracted.csv"
)

FAILURES = (
    ROOT
    / "census_timestamp_parse_failures.csv"
)

DUPLICATES = (
    ROOT
    / "census_duplicate_candidates.csv"
)

EXPECTED_ROWS = 399

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EASTERN = ZoneInfo(
    "America/New_York"
)

MONTHS = {
    "january": 1,
    "february": 2,
    "march": 3,
    "april": 4,
    "may": 5,
    "june": 6,
    "july": 7,
    "august": 8,
    "september": 9,
    "october": 10,
    "november": 11,
    "december": 12,
}

MONTH_PATTERN = (
    "January|February|March|April|May|June|"
    "July|August|September|October|November|December"
)

WEEKDAY_PATTERN = (
    "Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday"
)

#
# Historical format:
#
#   FOR IMMEDIATE RELEASE
#   THURSDAY, JANUARY 28, 2010, AT 8:30 A.M. EST
#
IMMEDIATE_RELEASE_RE = re.compile(
    rf"""
    \bFOR\s+IMMEDIATE\s+RELEASE\b
    .{{0,250}}?
    (?:
        (?P<weekday>{WEEKDAY_PATTERN})
        \s*,\s*
    )?
    (?P<month>{MONTH_PATTERN})
    \s+
    (?P<day>\d{{1,2}})
    \s*,\s*
    (?P<year>20\d{{2}})
    \s*,?\s*
    AT\s+
    (?P<hour>\d{{1,2}})
    :
    (?P<minute>\d{{2}})
    \s*
    (?P<ampm>
        A\.?\s*M\.?
        |
        P\.?\s*M\.?
    )
    \s*
    (?P<tz>EST|EDT)
    \b
    """,
    re.I | re.X | re.S,
)

#
# Modern format:
#
#   FOR RELEASE AT 8:30 AM EDT, WEDNESDAY, MARCH 13, 2019
#
RELEASE_AT_RE = re.compile(
    rf"""
    \bFOR\s+RELEASE\s+AT\s+
    (?P<hour>\d{{1,2}})
    :
    (?P<minute>\d{{2}})
    \s*
    (?P<ampm>
        A\.?\s*M\.?
        |
        P\.?\s*M\.?
    )
    \s*
    (?P<tz>EST|EDT)
    \s*,?\s*
    (?:
        (?P<weekday>{WEEKDAY_PATTERN})
        \s*,\s*
    )?
    (?P<month>{MONTH_PATTERN})
    \s+
    (?P<day>\d{{1,2}})
    \s*,\s*
    (?P<year>20\d{{2}})
    \b
    """,
    re.I | re.X,
)

#
# Narrow malformed-pdftotext recovery.
#
# We do NOT globally normalize arbitrary words. These repairs are used
# only on a temporary candidate string when the normal authoritative
# release-header parsers failed.
#
# Known examples from Census PDF text extraction can split:
#
#   RELEAS E
#   SEPTEMB ER
#
# The repair scope is intentionally limited to those exact tokens.
#
MALFORMED_TOKEN_REPAIRS = (
    (
        re.compile(
            r"\bRELEAS\s+E\b",
            re.I,
        ),
        "RELEASE",
    ),
    (
        re.compile(
            r"\bSEPTEMB\s+ER\b",
            re.I,
        ),
        "SEPTEMBER",
    ),
)

#
# June 2026 Durable Goods is the one permitted schedule-derived
# timestamp in this canonical snapshot.
#
DURABLE_SCHEDULE_REFERENCE = (
    2026,
    6,
)

DURABLE_SCHEDULE_RELEASE_DATE = date(
    2026,
    7,
    27,
)

DURABLE_SCHEDULE_RELEASE_TIME = (
    8,
    30,
)

EXPECTED_FAMILIES = {
    "RETAIL_SALES",
    "DURABLE_GOODS",
}


def read_csv(path: Path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        return list(
            csv.DictReader(f)
        )


def artifact_path(row):
    family = row[
        "event_family"
    ]

    if family == "RETAIL_SALES":
        return (
            RETAIL_DIR
            / row["filename"]
        )

    if family == "DURABLE_GOODS":
        return (
            DURABLE_DIR
            / row["filename"]
        )

    raise ValueError(
        f"unexpected family {family!r}"
    )


def pdf_to_text(path: Path):
    result = subprocess.run(
        [
            "pdftotext",
            "-layout",
            str(path),
            "-",
        ],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "pdftotext failed: "
            + result.stderr.strip()
        )

    if not result.stdout.strip():
        raise RuntimeError(
            "pdftotext returned empty text"
        )

    return result.stdout


def html_to_text(path: Path):
    text = path.read_text(
        encoding="utf-8",
        errors="replace",
    )

    text = html.unescape(text)

    text = re.sub(
        r"""
        </?
        (?:
            p|div|br|li|
            h1|h2|h3|h4|
            tr|td|th|
            span|section|
            article
        )
        [^>]*>
        """,
        "\n",
        text,
        flags=re.I | re.X,
    )

    text = re.sub(
        r"<[^>]+>",
        " ",
        text,
    )

    lines = []

    for line in text.splitlines():
        line = re.sub(
            r"\s+",
            " ",
            line,
        ).strip()

        if line:
            lines.append(line)

    return "\n".join(lines)


def normalize_ampm(value: str):
    return (
        value
        .replace(".", "")
        .replace(" ", "")
        .upper()
    )


def clock_to_24h(
    hour: int,
    minute: int,
    ampm: str,
):
    normalized = normalize_ampm(
        ampm
    )

    if not 1 <= hour <= 12:
        raise ValueError(
            f"invalid 12-hour clock hour: {hour}"
        )

    if not 0 <= minute <= 59:
        raise ValueError(
            f"invalid minute: {minute}"
        )

    if normalized == "PM":
        if hour != 12:
            hour += 12
    elif normalized == "AM":
        if hour == 12:
            hour = 0
    else:
        raise ValueError(
            f"invalid AM/PM token: {ampm!r}"
        )

    return hour, minute


def parsed_from_match(
    match,
    parser_name: str,
):
    month_name = (
        match.group("month")
        .lower()
    )

    month = MONTHS[
        month_name
    ]

    release_date = date(
        int(
            match.group("year")
        ),
        month,
        int(
            match.group("day")
        ),
    )

    hour, minute = clock_to_24h(
        int(
            match.group("hour")
        ),
        int(
            match.group("minute")
        ),
        match.group("ampm"),
    )

    matched_text = re.sub(
        r"\s+",
        " ",
        match.group(0),
    ).strip()

    return {
        "release_date":
            release_date,
        "hour":
            hour,
        "minute":
            minute,
        "timezone_token":
            match.group("tz").upper(),
        "timestamp_parser":
            parser_name,
        "release_text":
            matched_text,
    }


def parse_release_header(
    text: str,
):
    #
    # First pass: untouched authoritative extracted text.
    #
    for (
        parser_name,
        regex,
    ) in (
        (
            "census_for_immediate_release",
            IMMEDIATE_RELEASE_RE,
        ),
        (
            "census_for_release_at",
            RELEASE_AT_RE,
        ),
    ):
        match = regex.search(text)

        if match:
            return parsed_from_match(
                match,
                parser_name,
            )

    #
    # Second pass: extremely narrow pdftotext token repair.
    #
    repaired = text
    changed = False

    for regex, replacement in (
        MALFORMED_TOKEN_REPAIRS
    ):
        new_text, count = regex.subn(
            replacement,
            repaired,
        )

        if count:
            changed = True
            repaired = new_text

    if not changed:
        return None

    for (
        parser_name,
        regex,
    ) in (
        (
            "census_for_immediate_release_malformed_text_repair",
            IMMEDIATE_RELEASE_RE,
        ),
        (
            "census_for_release_at_malformed_text_repair",
            RELEASE_AT_RE,
        ),
    ):
        match = regex.search(
            repaired
        )

        if match:
            parsed = parsed_from_match(
                match,
                parser_name,
            )

            parsed[
                "release_text"
            ] = (
                "[narrow pdftotext repair] "
                + parsed["release_text"]
            )

            return parsed

    return None


def parse_schedule_rows(
    schedule_html: str,
):
    #
    # Parse <tr> rows instead of relying on a global date search.
    #
    rows = re.findall(
        r"<tr[^>]*>(.*?)</tr>",
        schedule_html,
        re.I | re.S,
    )

    result = []

    for raw_row in rows:
        cells = re.findall(
            r"<t[dh][^>]*>(.*?)</t[dh]>",
            raw_row,
            re.I | re.S,
        )

        cleaned = []

        for cell in cells:
            value = html.unescape(
                re.sub(
                    r"<[^>]+>",
                    " ",
                    cell,
                )
            )

            value = re.sub(
                r"\s+",
                " ",
                value,
            ).strip()

            if value:
                cleaned.append(
                    value
                )

        if cleaned:
            result.append(
                cleaned
            )

    return result


def durable_schedule_timestamp():
    if not DURABLE_SCHEDULE.exists():
        raise RuntimeError(
            "archived durable schedule is missing: "
            f"{DURABLE_SCHEDULE}"
        )

    schedule_html = (
        DURABLE_SCHEDULE
        .read_text(
            encoding="utf-8",
            errors="replace",
        )
    )

    #
    # First verify the archived page itself says the Advance column
    # is an 8:30 a.m. release time.
    #
    flattened = html.unescape(
        re.sub(
            r"<[^>]+>",
            " ",
            schedule_html,
        )
    )

    flattened = re.sub(
        r"\s+",
        " ",
        flattened,
    )

    if not re.search(
        r"""
        Advance\s+Report\s+on\s+Durable\s+Goods
        .*?
        8:30\s*a\.?m\.?\s+release\s+time
        """,
        flattened,
        re.I | re.X | re.S,
    ):
        raise RuntimeError(
            "durable schedule does not establish "
            "8:30 a.m. as the Advance Report release time"
        )

    rows = parse_schedule_rows(
        schedule_html
    )

    matching = []

    for cells in rows:
        joined = " | ".join(
            cells
        )

        if not re.search(
            r"\bJune\s+2026\b",
            joined,
            re.I,
        ):
            continue

        matching.append(
            cells
        )

    if len(matching) != 1:
        raise RuntimeError(
            "expected exactly one June 2026 "
            "durable schedule row; "
            f"found {len(matching)}"
        )

    cells = matching[0]

    if len(cells) < 2:
        raise RuntimeError(
            "June 2026 durable schedule row "
            "does not contain an advance-release date"
        )

    #
    # First cell is survey month; second is Advance Report release.
    #
    advance_date_raw = (
        cells[1]
    )

    parsed_date = None

    for fmt in (
        "%m/%d/%Y",
        "%m/%d/%y",
    ):
        try:
            parsed_date = (
                datetime.strptime(
                    advance_date_raw,
                    fmt,
                ).date()
            )
            break
        except ValueError:
            pass

    if parsed_date is None:
        raise RuntimeError(
            "could not parse June 2026 durable "
            "advance-release date from schedule: "
            f"{advance_date_raw!r}"
        )

    if (
        parsed_date
        != DURABLE_SCHEDULE_RELEASE_DATE
    ):
        raise RuntimeError(
            "unexpected June 2026 durable schedule date: "
            f"expected {DURABLE_SCHEDULE_RELEASE_DATE}, "
            f"found {parsed_date}"
        )

    hour, minute = (
        DURABLE_SCHEDULE_RELEASE_TIME
    )

    return {
        "release_date":
            parsed_date,
        "hour":
            hour,
        "minute":
            minute,
        "timezone_token":
            "EDT",
        "timestamp_parser":
            "census_durable_release_schedule",
        "release_text":
            (
                "June 2026 | "
                f"{advance_date_raw} | "
                "Advance Report on Durable Goods "
                "(8:30 a.m. release time)"
            ),
        "timestamp_source_url":
            (
                "https://www.census.gov/"
                "manufacturing/m3/"
                "release_schedule.html"
            ),
    }


def timestamp_for(parsed):
    local_dt = datetime(
        parsed["release_date"].year,
        parsed["release_date"].month,
        parsed["release_date"].day,
        parsed["hour"],
        parsed["minute"],
        tzinfo=EASTERN,
    )

    utc_dt = local_dt.astimezone(
        timezone.utc
    )

    return local_dt, utc_dt


def timezone_token_matches(
    local_dt: datetime,
    token: str,
):
    expected = local_dt.tzname()

    return token == expected


def expected_reference_key(row):
    return (
        int(
            row["reference_year"]
        ),
        int(
            row["reference_month"]
        ),
    )


def source_title(
    family: str,
    reference_period: str,
):
    if family == "RETAIL_SALES":
        return (
            "Advance Monthly Sales for "
            "Retail and Food Services - "
            f"{reference_period}"
        )

    if family == "DURABLE_GOODS":
        return (
            "Advance Report on Durable Goods "
            "Manufacturers' Shipments, "
            "Inventories and Orders - "
            f"{reference_period}"
        )

    raise ValueError(
        f"unexpected family {family!r}"
    )


def main():
    print(
        "Census RETAIL_SALES / DURABLE_GOODS "
        "timestamp extraction"
    )
    print()

    if not MANIFEST.exists():
        print(
            f"ERROR: missing manifest: {MANIFEST}"
        )
        return 1

    rows = read_csv(
        MANIFEST
    )

    print(
        f"Manifest rows : {len(rows)}"
    )

    if len(rows) != EXPECTED_ROWS:
        print(
            "ERROR: expected exactly "
            f"{EXPECTED_ROWS} manifest rows, "
            f"found {len(rows)}"
        )
        return 1

    #
    # Fail before parsing if the manifest itself is not the completed
    # acquisition snapshot.
    #
    manifest_keys = defaultdict(
        list
    )

    for row in rows:
        family = row.get(
            "event_family",
            "",
        )

        if family not in EXPECTED_FAMILIES:
            print(
                "ERROR: unexpected Census family "
                f"in manifest: {family!r}"
            )
            return 1

        if row.get(
            "source_agency"
        ) != "CENSUS":
            print(
                "ERROR: non-CENSUS row in manifest"
            )
            return 1

        if row.get(
            "http_status"
        ) != "200":
            print(
                "ERROR: manifest contains non-200 row: "
                f"{family} "
                f"{row.get('reference_period')}"
            )
            return 1

        key = (
            family,
            row["reference_year"],
            row["reference_month"],
        )

        manifest_keys[key].append(
            row
        )

    duplicate_manifest_keys = [
        key
        for key, values
        in manifest_keys.items()
        if len(values) != 1
    ]

    if duplicate_manifest_keys:
        print(
            "ERROR: duplicate family/reference-period "
            "manifest rows exist"
        )

        for key in duplicate_manifest_keys:
            print(
                " ",
                key,
                len(
                    manifest_keys[key]
                ),
            )

        return 1

    extracted = []
    failures = []

    parser_counts = Counter()
    family_counts = Counter()
    year_counts = Counter()
    time_counts = Counter()
    timezone_token_counts = Counter()

    schedule_cache = None

    for i, row in enumerate(
        rows,
        start=1,
    ):
        family = row[
            "event_family"
        ]

        reference_year = int(
            row["reference_year"]
        )
        reference_month = int(
            row["reference_month"]
        )

        reference_period = row[
            "reference_period"
        ]

        path = artifact_path(
            row
        )

        base_failure = {
            "source_agency":
                "CENSUS",
            "event_family":
                family,
            "reference_year":
                reference_year,
            "reference_month":
                f"{reference_month:02d}",
            "reference_period":
                reference_period,
            "filename":
                row["filename"],
            "url":
                row["url"],
            "error":
                "",
            "detail":
                "",
        }

        if not path.exists():
            failure = dict(
                base_failure
            )
            failure[
                "error"
            ] = "missing_artifact"

            failures.append(
                failure
            )

            print(
                f"[{i:03d}/{len(rows):03d}] "
                f"FAIL {family:<14} "
                f"{reference_year}-"
                f"{reference_month:02d} "
                "missing artifact"
            )
            continue

        suffix = (
            path.suffix.lower()
        )

        parsed = None

        #
        # Exactly one allowed schedule-derived timestamp:
        #
        #   DURABLE_GOODS / June 2026
        #
        if (
            family == "DURABLE_GOODS"
            and (
                reference_year,
                reference_month,
            )
            == DURABLE_SCHEDULE_REFERENCE
        ):
            if suffix not in {
                ".html",
                ".htm",
            }:
                failure = dict(
                    base_failure
                )
                failure[
                    "error"
                ] = (
                    "unexpected_durable_"
                    "june_2026_artifact_type"
                )
                failure[
                    "detail"
                ] = suffix

                failures.append(
                    failure
                )

                print(
                    f"[{i:03d}/{len(rows):03d}] "
                    "FAIL DURABLE_GOODS 2026-06 "
                    f"unexpected artifact {suffix}"
                )
                continue

            try:
                if schedule_cache is None:
                    schedule_cache = (
                        durable_schedule_timestamp()
                    )

                parsed = dict(
                    schedule_cache
                )
            except Exception as exc:
                failure = dict(
                    base_failure
                )
                failure[
                    "error"
                ] = (
                    "durable_schedule_parse_failure"
                )
                failure[
                    "detail"
                ] = str(exc)

                failures.append(
                    failure
                )

                print(
                    f"[{i:03d}/{len(rows):03d}] "
                    "FAIL DURABLE_GOODS 2026-06 "
                    f"{exc}"
                )
                continue

        else:
            #
            # Every other canonical artifact must establish its
            # timestamp from the release itself.
            #
            if suffix == ".pdf":
                try:
                    text = pdf_to_text(
                        path
                    )
                except Exception as exc:
                    failure = dict(
                        base_failure
                    )
                    failure[
                        "error"
                    ] = (
                        "pdftotext_failure"
                    )
                    failure[
                        "detail"
                    ] = str(exc)

                    failures.append(
                        failure
                    )

                    print(
                        f"[{i:03d}/{len(rows):03d}] "
                        f"FAIL {family:<14} "
                        f"{reference_year}-"
                        f"{reference_month:02d} "
                        f"{exc}"
                    )
                    continue

            elif suffix in {
                ".html",
                ".htm",
            }:
                try:
                    text = html_to_text(
                        path
                    )
                except Exception as exc:
                    failure = dict(
                        base_failure
                    )
                    failure[
                        "error"
                    ] = (
                        "html_text_failure"
                    )
                    failure[
                        "detail"
                    ] = str(exc)

                    failures.append(
                        failure
                    )

                    print(
                        f"[{i:03d}/{len(rows):03d}] "
                        f"FAIL {family:<14} "
                        f"{reference_year}-"
                        f"{reference_month:02d} "
                        f"{exc}"
                    )
                    continue

            else:
                failure = dict(
                    base_failure
                )
                failure[
                    "error"
                ] = (
                    "unsupported_artifact_type"
                )
                failure[
                    "detail"
                ] = suffix

                failures.append(
                    failure
                )

                print(
                    f"[{i:03d}/{len(rows):03d}] "
                    f"FAIL {family:<14} "
                    f"{reference_year}-"
                    f"{reference_month:02d} "
                    f"unsupported {suffix}"
                )
                continue

            parsed = parse_release_header(
                text
            )

            if parsed is None:
                failure = dict(
                    base_failure
                )
                failure[
                    "error"
                ] = (
                    "authoritative_release_"
                    "header_not_found"
                )
                failure[
                    "detail"
                ] = (
                    "No supported FOR IMMEDIATE "
                    "RELEASE / FOR RELEASE AT "
                    "header matched"
                )

                failures.append(
                    failure
                )

                print(
                    f"[{i:03d}/{len(rows):03d}] "
                    f"FAIL {family:<14} "
                    f"{reference_year}-"
                    f"{reference_month:02d} "
                    "release header not found"
                )
                continue

            parsed[
                "timestamp_source_url"
            ] = row["url"]

        try:
            local_dt, utc_dt = (
                timestamp_for(
                    parsed
                )
            )
        except Exception as exc:
            failure = dict(
                base_failure
            )
            failure[
                "error"
            ] = (
                "timestamp_conversion_failure"
            )
            failure[
                "detail"
            ] = str(exc)

            failures.append(
                failure
            )

            print(
                f"[{i:03d}/{len(rows):03d}] "
                f"FAIL {family:<14} "
                f"{reference_year}-"
                f"{reference_month:02d} "
                f"{exc}"
            )
            continue

        #
        # The explicit EST/EDT token must agree with the actual
        # America/New_York offset on that date. This catches a bad
        # header parse rather than silently accepting it.
        #
        if not timezone_token_matches(
            local_dt,
            parsed[
                "timezone_token"
            ],
        ):
            failure = dict(
                base_failure
            )
            failure[
                "error"
            ] = (
                "timezone_token_date_mismatch"
            )
            failure[
                "detail"
            ] = (
                "Header/schedule token "
                f"{parsed['timezone_token']} "
                "does not agree with "
                f"America/New_York={local_dt.tzname()} "
                f"on {local_dt.date()}"
            )

            failures.append(
                failure
            )

            print(
                f"[{i:03d}/{len(rows):03d}] "
                f"FAIL {family:<14} "
                f"{reference_year}-"
                f"{reference_month:02d} "
                "EST/EDT mismatch"
            )
            continue

        #
        # We intentionally included December 2009 reference artifacts
        # because their actual release events belong to January 2010.
        #
        if not (
            START_DATE
            <= local_dt.date()
            <= END_DATE
        ):
            failure = dict(
                base_failure
            )
            failure[
                "error"
            ] = (
                "release_date_outside_window"
            )
            failure[
                "detail"
            ] = (
                f"{local_dt.date()} outside "
                f"{START_DATE}..{END_DATE}"
            )

            failures.append(
                failure
            )

            print(
                f"[{i:03d}/{len(rows):03d}] "
                f"FAIL {family:<14} "
                f"{reference_year}-"
                f"{reference_month:02d} "
                "release date outside window"
            )
            continue

        parser_counts[
            parsed[
                "timestamp_parser"
            ]
        ] += 1

        family_counts[
            family
        ] += 1

        year_counts[
            (
                family,
                local_dt.year,
            )
        ] += 1

        local_time_string = (
            local_dt.time()
            .replace(
                tzinfo=None
            )
            .isoformat()
        )

        time_counts[
            (
                family,
                local_time_string,
            )
        ] += 1

        timezone_token_counts[
            parsed[
                "timezone_token"
            ]
        ] += 1

        extracted.append({
            "source_agency":
                "CENSUS",
            "event_family":
                family,
            "event_timestamp_utc":
                utc_dt.isoformat(),
            "source_local_date":
                local_dt.date().isoformat(),
            "source_local_time":
                local_time_string,
            "source_timezone":
                "America/New_York",
            "release_timezone_token":
                parsed[
                    "timezone_token"
                ],
            "reference_year":
                reference_year,
            "reference_month":
                f"{reference_month:02d}",
            "reference_period":
                reference_period,
            "title":
                source_title(
                    family,
                    reference_period,
                ),
            "url":
                row["url"],
            "final_url":
                row["final_url"],
            "filename":
                row["filename"],
            "discovery_source":
                row["discovery_source"],
            "timestamp_parser":
                parsed[
                    "timestamp_parser"
                ],
            "release_text":
                parsed[
                    "release_text"
                ],
            "timestamp_source_url":
                parsed[
                    "timestamp_source_url"
                ],
        })

        print(
            f"[{i:03d}/{len(rows):03d}] "
            f"OK   {family:<14} "
            f"{reference_year}-"
            f"{reference_month:02d} "
            f"=> "
            f"{local_dt.date()} "
            f"{local_time_string} "
            f"{parsed['timezone_token']} "
            f"[{parsed['timestamp_parser']}]"
        )

    #
    # Canonical uniqueness:
    #
    #   source_agency
    #   event_family
    #   event_timestamp_utc
    #
    groups = defaultdict(
        list
    )

    for row in extracted:
        key = (
            row["source_agency"],
            row["event_family"],
            row[
                "event_timestamp_utc"
            ],
        )

        groups[key].append(
            row
        )

    duplicates = []

    for items in groups.values():
        if len(items) > 1:
            duplicates.extend(
                items
            )

    extracted.sort(
        key=lambda row: (
            row[
                "event_timestamp_utc"
            ],
            row[
                "event_family"
            ],
            row[
                "reference_year"
            ],
            row[
                "reference_month"
            ],
        )
    )

    fields = [
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
    ]

    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            extracted
        )

    failure_fields = [
        "source_agency",
        "event_family",
        "reference_year",
        "reference_month",
        "reference_period",
        "filename",
        "url",
        "error",
        "detail",
    ]

    with FAILURES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=
                failure_fields,
        )

        writer.writeheader()
        writer.writerows(
            failures
        )

    with DUPLICATES.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )

        writer.writeheader()
        writer.writerows(
            duplicates
        )

    print()
    print(
        "Census timestamp extraction summary"
    )
    print()
    print(
        f"Manifest candidates : {len(rows)}"
    )
    print(
        f"Resolved timestamps : {len(extracted)}"
    )
    print(
        f"Parse failures      : {len(failures)}"
    )
    print(
        f"Duplicate rows      : {len(duplicates)}"
    )
    print()

    print(
        "Family distribution:"
    )

    for family in (
        "RETAIL_SALES",
        "DURABLE_GOODS",
    ):
        print(
            f"  {family:<16} "
            f"{family_counts[family]:>4}"
        )

    print()
    print(
        "Timestamp parser distribution:"
    )

    for parser_name, count in sorted(
        parser_counts.items()
    ):
        print(
            f"  {parser_name:<52} "
            f"{count:>4}"
        )

    print()
    print(
        "Release-time distribution:"
    )

    for family in (
        "RETAIL_SALES",
        "DURABLE_GOODS",
    ):
        print(
            f"  {family}:"
        )

        family_times = [
            (
                local_time,
                count,
            )
            for (
                time_family,
                local_time,
            ), count
            in time_counts.items()
            if time_family == family
        ]

        for (
            local_time,
            count,
        ) in sorted(
            family_times
        ):
            print(
                f"    {local_time:<10} "
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
            f"{year_counts[('RETAIL_SALES', year)]:>8} "
            f"{year_counts[('DURABLE_GOODS', year)]:>8}"
        )

    print()
    print(
        "Release timezone tokens:"
    )

    for token, count in sorted(
        timezone_token_counts.items()
    ):
        print(
            f"  {token:<6} {count:>4}"
        )

    print()
    print(
        f"Events     : {OUTPUT}"
    )
    print(
        f"Failures   : {FAILURES}"
    )
    print(
        f"Duplicates : {DUPLICATES}"
    )
    print()

    #
    # Fail closed.
    #
    if failures:
        print(
            "RESULT: FAIL - Census timestamp "
            "extraction has unresolved rows."
        )
        print(
            "No database writes were performed."
        )
        return 1

    if duplicates:
        print(
            "RESULT: FAIL - duplicate canonical "
            "Census timestamp keys exist."
        )
        print(
            "No database writes were performed."
        )
        return 1

    if len(extracted) != EXPECTED_ROWS:
        print(
            "RESULT: FAIL - expected exactly "
            f"{EXPECTED_ROWS} resolved Census events, "
            f"found {len(extracted)}."
        )
        print(
            "No database writes were performed."
        )
        return 1

    if (
        family_counts["RETAIL_SALES"]
        != 200
    ):
        print(
            "RESULT: FAIL - expected 200 "
            "RETAIL_SALES events, found "
            f"{family_counts['RETAIL_SALES']}."
        )
        print(
            "No database writes were performed."
        )
        return 1

    if (
        family_counts["DURABLE_GOODS"]
        != 199
    ):
        print(
            "RESULT: FAIL - expected 199 "
            "DURABLE_GOODS events, found "
            f"{family_counts['DURABLE_GOODS']}."
        )
        print(
            "No database writes were performed."
        )
        return 1

    schedule_rows = [
        row
        for row in extracted
        if (
            row[
                "timestamp_parser"
            ]
            == "census_durable_release_schedule"
        )
    ]

    if len(schedule_rows) != 1:
        print(
            "RESULT: FAIL - expected exactly one "
            "schedule-derived Census timestamp, "
            f"found {len(schedule_rows)}."
        )
        print(
            "No database writes were performed."
        )
        return 1

    schedule_row = (
        schedule_rows[0]
    )

    if not (
        schedule_row[
            "event_family"
        ]
        == "DURABLE_GOODS"
        and schedule_row[
            "reference_year"
        ]
        == 2026
        and schedule_row[
            "reference_month"
        ]
        == "06"
        and schedule_row[
            "source_local_date"
        ]
        == "2026-07-27"
        and schedule_row[
            "source_local_time"
        ]
        == "08:30:00"
    ):
        print(
            "RESULT: FAIL - the sole schedule-derived "
            "timestamp is not exactly June 2026 "
            "Durable Goods => 2026-07-27 08:30 ET."
        )
        print(
            "No database writes were performed."
        )
        return 1

    print(
        "RESULT: PASS - all 399 Census release "
        "timestamps resolved with authoritative provenance."
    )
    print(
        "No default release time was assumed."
    )
    print(
        "No database writes were performed."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
