#!/usr/bin/env python3

from __future__ import annotations

import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict
from datetime import date, datetime
from pathlib import Path
from typing import Any


DB_NAME = os.environ.get(
    "LSTM_DB_NAME",
    "LSTM",
)

START_DATE = date(
    2010,
    1,
    1,
)

END_DATE = date(
    2026,
    8,
    24,
)

ROOT = Path(
    "EconomicCalendar/raw/oanda"
)

DAILY_ROOT = (
    ROOT
    / "daily"
)

MATCHES_OUTPUT = (
    ROOT
    / "oanda_consensus_matches.csv"
)

UNMATCHED_OUTPUT = (
    ROOT
    / "oanda_consensus_unmatched.csv"
)

AMBIGUOUS_OUTPUT = (
    ROOT
    / "oanda_consensus_ambiguous.csv"
)

FAMILY_YEAR_OUTPUT = (
    ROOT
    / "oanda_consensus_coverage_by_family_year.csv"
)

SUMMARY_OUTPUT = (
    ROOT
    / "oanda_consensus_coverage_summary.csv"
)

REPORT_ID_OUTPUT = (
    ROOT
    / "oanda_report_id_candidates.csv"
)

HTTP_FAILURES_OUTPUT = (
    ROOT
    / "oanda_http_failures.csv"
)


TARGET_FAMILIES = {
    "CPI",
    "DURABLE_GOODS",
    "EMPLOYMENT",
    "FOMC",
    "GDP",
    "JOLTS",
    "PCE",
    "PPI",
    "RETAIL_SALES",
}


#
# Second-pass OANDA report mapping.
#
PINNED_REPORT_IDS = {
    "CPI": {
        699,
    },
    "DURABLE_GOODS": {
        59,
    },
    "EMPLOYMENT": {
        707,
    },
    "FOMC": {
        82,
    },
    "GDP": {
        690,
    },
    "JOLTS": {
        1371,
    },
    "PCE": {
        694,
    },
    "PPI": {
        703,
    },
    "RETAIL_SALES": {
        696,
    },
}


PINNED_EVENT_NAMES = {
    "CPI": {
        "Consumer Price Index",
    },
    "DURABLE_GOODS": {
        "Durable Goods Orders - pre..",
    },
    "EMPLOYMENT": {
        "Non-Farm Employment Change",
    },
    "FOMC": {
        "FOMC Interest Rate Decision",
    },
    "GDP": {
        "GDP (Annualized) - pre..",
        "GDP (Annualized) - rev..",
        "GDP (Annualized) - fin..",
    },
    "JOLTS": {
        "JOLTS Job Openings",
    },
    "PCE": {
        "Personal Consumption Expenditures",
    },
    "PPI": {
        "Producer Price Index",
    },
    "RETAIL_SALES": {
        "Retail Sales",
    },
}


#
# One observed CPI exception.
#
CPI_REPORT_698_EXCEPTIONS = {
    date(
        2025,
        12,
        18,
    ),
}


#
# Exact OANDA event-ID exceptions established by direct inspection of
# the cached daily payload and the authoritative release metadata.
#
# 2019-04-29 PCE:
#
# OANDA contains two report-694 March rows at the same timestamp.
# The canonical March PCE observation is event 89952:
#
#     Actual   0.9% m/m
#     Forecast 0.6% m/m
#     Previous 0.1% m/m
#
# The other report-694 row cannot be distinguished using period alone.
#
EXACT_OANDA_EVENT_ID_EXCEPTIONS = {
    (
        "PCE",
        date(
            2019,
            4,
            29,
        ),
    ): 89952,
}


MATCH_FIELDS = [
    "event_family",
    "official_event_timestamp_utc",
    "official_release_date",
    "official_source_agency",
    "official_reference_period",
    "oanda_date",
    "oanda_event",
    "oanda_report_id",
    "oanda_event_id",
    "oanda_period",
    "oanda_priority",
    "oanda_actual",
    "oanda_forecast",
    "oanda_previous",
    "oanda_timestamp",
    "forecast_present",
    "previous_present",
    "actual_present",
    "match_rule",
    "source_file",
]


UNMATCHED_FIELDS = [
    "event_family",
    "official_event_timestamp_utc",
    "official_release_date",
    "official_source_agency",
    "official_reference_period",
    "reason",
]


AMBIGUOUS_FIELDS = [
    "event_family",
    "official_event_timestamp_utc",
    "official_release_date",
    "official_source_agency",
    "official_reference_period",
    "oanda_date",
    "oanda_event",
    "oanda_report_id",
    "oanda_event_id",
    "oanda_period",
    "oanda_priority",
    "oanda_actual",
    "oanda_forecast",
    "oanda_previous",
    "oanda_timestamp",
    "source_file",
]


MONTH_NAMES = {
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


def connect_read_only():
    try:
        import psycopg

        conn = psycopg.connect(
            dbname=DB_NAME,
        )

        conn.autocommit = False

        with conn.cursor() as cur:
            cur.execute(
                "SET TRANSACTION READ ONLY"
            )

        return conn

    except ImportError:
        pass

    try:
        import psycopg2

        conn = psycopg2.connect(
            dbname=DB_NAME,
        )

        conn.autocommit = False

        cur = conn.cursor()

        try:
            cur.execute(
                "SET TRANSACTION READ ONLY"
            )
        finally:
            cur.close()

        return conn

    except ImportError as exc:
        raise RuntimeError(
            "Neither psycopg nor psycopg2 is installed"
        ) from exc


def write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, Any]],
):
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

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

        for row in rows:
            writer.writerow({
                field:
                    row.get(
                        field,
                        "",
                    )
                for field in fields
            })


def iso_timestamp(
    value,
) -> str:
    if isinstance(
        value,
        datetime,
    ):
        return value.isoformat()

    return str(
        value
        or ""
    )


def official_release_date(
    value,
) -> date:
    if isinstance(
        value,
        datetime,
    ):
        return value.date()

    text = str(
        value
    ).strip()

    return datetime.fromisoformat(
        text.replace(
            "Z",
            "+00:00",
        )
    ).date()


def load_authoritative_events():
    conn = connect_read_only()

    try:
        cur = conn.cursor()

        try:
            cur.execute(
                """
                SELECT
                    source_agency,
                    event_family,
                    event_timestamp_utc,
                    reference_period
                FROM economic_event
                WHERE event_family = ANY(%s)
                  AND event_timestamp_utc >= %s
                  AND event_timestamp_utc < %s
                ORDER BY
                    event_timestamp_utc,
                    event_family,
                    source_agency
                """,
                (
                    sorted(
                        TARGET_FAMILIES
                    ),
                    datetime(
                        START_DATE.year,
                        START_DATE.month,
                        START_DATE.day,
                    ),
                    datetime(
                        2026,
                        8,
                        25,
                    ),
                ),
            )

            rows = cur.fetchall()

        finally:
            cur.close()

    finally:
        try:
            conn.rollback()
        finally:
            conn.close()

    result = []

    for (
        source_agency,
        family,
        timestamp,
        reference_period,
    ) in rows:
        result.append({
            "source_agency":
                source_agency,
            "event_family":
                family,
            "event_timestamp_utc":
                timestamp,
            "release_date":
                official_release_date(
                    timestamp
                ),
            "reference_period":
                str(
                    reference_period
                    or ""
                ).strip(),
        })

    return result


def parse_json_file(
    path: Path,
):
    try:
        with path.open(
            "r",
            encoding="utf-8",
        ) as f:
            value = json.load(
                f
            )

    except Exception:
        return []

    if isinstance(
        value,
        list,
    ):
        return value

    if isinstance(
        value,
        dict,
    ):
        for key in (
            "data",
            "rows",
            "events",
        ):
            candidate = value.get(
                key
            )

            if isinstance(
                candidate,
                list,
            ):
                return candidate

    return []


def load_cached_oanda_rows():
    if not DAILY_ROOT.exists():
        raise RuntimeError(
            "Missing cached OANDA daily directory: "
            f"{DAILY_ROOT}"
        )

    result = []

    for path in sorted(
        DAILY_ROOT.rglob(
            "*"
        )
    ):
        if not path.is_file():
            continue

        if path.suffix.lower() not in {
            ".json",
            ".txt",
        }:
            continue

        rows = parse_json_file(
            path
        )

        for row in rows:
            if not isinstance(
                row,
                dict,
            ):
                continue

            country = str(
                row.get(
                    "Country",
                    "",
                )
            ).strip().upper()

            if country not in {
                "US",
                "USA",
            }:
                continue

            copied = dict(
                row
            )

            copied[
                "_source_file"
            ] = str(
                path
            )

            result.append(
                copied
            )

    return result


def parse_oanda_date(
    value,
):
    text = str(
        value
        or ""
    ).strip()

    if not text:
        return None

    for fmt in (
        "%Y-%m-%d %H:%M",
        "%Y-%m-%d",
    ):
        try:
            return datetime.strptime(
                text,
                fmt,
            )
        except ValueError:
            pass

    return None


def report_id_of(
    row,
):
    value = row.get(
        "IDReport"
    )

    if value in (
        None,
        "",
    ):
        value = row.get(
            "ReportID"
        )

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def event_id_of(
    row,
):
    value = row.get(
        "ID"
    )

    if value in (
        None,
        "",
    ):
        value = row.get(
            "id"
        )

    return value


def event_id_int(
    row,
):
    value = event_id_of(
        row
    )

    try:
        return int(
            value
        )
    except (
        TypeError,
        ValueError,
    ):
        return None


def event_name_of(
    row,
):
    return str(
        row.get(
            "Event",
            "",
        )
        or ""
    ).strip()


def has_value(
    value,
):
    return bool(
        str(
            value
            or ""
        ).strip()
    )


def normalize_text(
    value,
):
    return " ".join(
        str(
            value
            or ""
        )
        .strip()
        .lower()
        .split()
    )


def extract_months_in_order(
    value,
):
    text = normalize_text(
        value
    )

    found = []

    for match in re.finditer(
        r"\b("
        + "|".join(
            MONTH_NAMES.keys()
        )
        + r")\b",
        text,
    ):
        found.append(
            MONTH_NAMES[
                match.group(1)
            ]
        )

    return found


def extract_quarters_in_order(
    value,
):
    text = normalize_text(
        value
    )

    found = []

    for match in re.finditer(
        r"\b(?:q([1-4])|([1-4])q)\b",
        text,
    ):
        digit = (
            match.group(1)
            or match.group(2)
        )

        found.append(
            int(
                digit
            )
        )

    return found


def candidate_period_position(
    official_reference_period,
    candidate_period,
):
    """
    Return the candidate period's position inside the authoritative
    reference-period sequence.

    Examples:

        "November 2025"            + "November" -> 0
        "October and November 2025"+ "October"  -> 0
        "October and November 2025"+ "November" -> 1
        "Q3 2025 Updated"          + "3Q"       -> 0

    None means the period cannot be established as belonging to the
    authoritative reference period.
    """

    official_months = extract_months_in_order(
        official_reference_period
    )

    candidate_months = extract_months_in_order(
        candidate_period
    )

    if (
        official_months
        and candidate_months
    ):
        candidate_month = (
            candidate_months[-1]
        )

        positions = [
            index
            for index, month
            in enumerate(
                official_months
            )
            if month
            == candidate_month
        ]

        if positions:
            return positions[-1]

        return None

    official_quarters = extract_quarters_in_order(
        official_reference_period
    )

    candidate_quarters = extract_quarters_in_order(
        candidate_period
    )

    if (
        official_quarters
        and candidate_quarters
    ):
        candidate_quarter = (
            candidate_quarters[-1]
        )

        positions = [
            index
            for index, quarter
            in enumerate(
                official_quarters
            )
            if quarter
            == candidate_quarter
        ]

        if positions:
            return positions[-1]

        return None

    #
    # Conservative textual fallback.
    #
    official_text = normalize_text(
        official_reference_period
    )

    candidate_text = normalize_text(
        candidate_period
    )

    if (
        official_text
        and candidate_text
        and candidate_text
        in official_text
    ):
        return official_text.rfind(
            candidate_text
        )

    return None


def candidate_allowed_for_family(
    family: str,
    release_date: date,
    row: dict[str, Any],
):
    report_id = report_id_of(
        row
    )

    event_name = event_name_of(
        row
    )

    if (
        family == "CPI"
        and
        release_date
        in CPI_REPORT_698_EXCEPTIONS
        and
        report_id == 698
        and
        event_name
        == "Consumer Price Index"
    ):
        return True

    pinned_reports = (
        PINNED_REPORT_IDS.get(
            family
        )
    )

    if (
        pinned_reports is not None
        and report_id
        not in pinned_reports
    ):
        return False

    pinned_names = (
        PINNED_EVENT_NAMES.get(
            family
        )
    )

    if (
        pinned_names is not None
        and event_name
        not in pinned_names
    ):
        return False

    return True


def base_candidate_match_rule(
    family,
    release_date,
    row,
):
    report_id = report_id_of(
        row
    )

    if (
        family == "CPI"
        and release_date
        in CPI_REPORT_698_EXCEPTIONS
        and report_id == 698
    ):
        return (
            "pinned_report_698_"
            "single_release_exception"
        )

    return (
        "pinned_report_and_event_name"
    )


def build_date_index(
    oanda_rows,
):
    result = defaultdict(
        list
    )

    for row in oanda_rows:
        dt = parse_oanda_date(
            row.get(
                "Date"
            )
        )

        if dt is None:
            continue

        result[
            dt.date()
        ].append(
            row
        )

    return result


def matching_candidates(
    family,
    release_date,
    date_index,
):
    same_day = (
        date_index.get(
            release_date,
            []
        )
    )

    return [
        row
        for row in same_day
        if candidate_allowed_for_family(
            family,
            release_date,
            row,
        )
    ]


def equivalent_consensus_signature(
    row,
):
    """
    Fields that must agree before multiple cached OANDA rows may be
    treated as equivalent representations of the same consensus result.

    Event ID and Event text are deliberately excluded because the GDP
    2026-01-22 cache contains separate rev/final OANDA records carrying
    the same release values.
    """

    return (
        report_id_of(
            row
        ),
        str(
            row.get(
                "Timestamp",
                "",
            )
        ),
        normalize_text(
            row.get(
                "Period",
                "",
            )
        ),
        normalize_text(
            row.get(
                "Actual",
                "",
            )
        ),
        normalize_text(
            row.get(
                "Forecast",
                "",
            )
        ),
        normalize_text(
            row.get(
                "Previous",
                "",
            )
        ),
        normalize_text(
            row.get(
                "Correction",
                "",
            )
        ),
        str(
            row.get(
                "Priority",
                "",
            )
        ),
    )


def collapse_equivalent_candidates(
    candidates,
):
    if len(
        candidates
    ) <= 1:
        return candidates

    signatures = {
        equivalent_consensus_signature(
            candidate
        )
        for candidate in candidates
    }

    if len(
        signatures
    ) != 1:
        return candidates

    #
    # All consensus-bearing content is identical.  Pick a stable
    # representative only after proving equivalence.
    #
    return [
        sorted(
            candidates,
            key=lambda row: (
                event_id_int(
                    row
                )
                if event_id_int(
                    row
                )
                is not None
                else sys.maxsize,
                event_name_of(
                    row
                ),
            ),
        )[0]
    ]


def resolve_candidates(
    official,
    candidates,
):
    """
    Conservatively reduce pinned same-day candidates to at most one.

    Returns:

        (resolved_candidates, match_rule)

    If more than one candidate remains, the caller still treats the
    authoritative event as ambiguous.
    """

    if len(
        candidates
    ) <= 1:
        if candidates:
            return (
                candidates,
                base_candidate_match_rule(
                    official[
                        "event_family"
                    ],
                    official[
                        "release_date"
                    ],
                    candidates[0],
                ),
            )

        return (
            candidates,
            "",
        )

    family = official[
        "event_family"
    ]

    release_date = official[
        "release_date"
    ]

    reference_period = official.get(
        "reference_period",
        "",
    )

    #
    # Rule 1:
    # exact event-ID exceptions established by direct inspection.
    #
    exception_id = (
        EXACT_OANDA_EVENT_ID_EXCEPTIONS.get(
            (
                family,
                release_date,
            )
        )
    )

    if exception_id is not None:
        exact = [
            candidate
            for candidate in candidates
            if event_id_int(
                candidate
            )
            == exception_id
        ]

        if len(
            exact
        ) == 1:
            return (
                exact,
                "exact_oanda_event_id_exception",
            )

    #
    # Rule 2:
    # use the authoritative reference period.
    #
    period_matches = []

    for candidate in candidates:
        position = candidate_period_position(
            reference_period,
            candidate.get(
                "Period",
                "",
            ),
        )

        if position is not None:
            period_matches.append(
                (
                    position,
                    candidate,
                )
            )

    if period_matches:
        #
        # For a multi-period authoritative release, take the latest
        # period explicitly represented by that release.
        #
        latest_position = max(
            position
            for position, _
            in period_matches
        )

        candidates = [
            candidate
            for position, candidate
            in period_matches
            if position
            == latest_position
        ]

        if len(
            candidates
        ) == 1:
            return (
                candidates,
                "authoritative_reference_period",
            )

    #
    # Rule 3:
    # If candidates otherwise represent the same authoritative period,
    # prefer the row carrying the contemporaneous consensus forecast.
    #
    forecast_candidates = [
        candidate
        for candidate in candidates
        if has_value(
            candidate.get(
                "Forecast"
            )
        )
    ]

    if (
        forecast_candidates
        and
        len(
            forecast_candidates
        )
        < len(
            candidates
        )
    ):
        candidates = (
            forecast_candidates
        )

        if len(
            candidates
        ) == 1:
            return (
                candidates,
                "forecast_bearing_candidate",
            )

    #
    # Rule 4:
    # collapse only rows whose consensus-bearing content is genuinely
    # equivalent.  This resolves the 2026-01-22 GDP rev/final duplicate
    # without making an arbitrary economic-data choice.
    #
    collapsed = (
        collapse_equivalent_candidates(
            candidates
        )
    )

    if (
        len(
            collapsed
        ) == 1
        and len(
            candidates
        ) > 1
    ):
        return (
            collapsed,
            "equivalent_oanda_duplicate_collapse",
        )

    return (
        candidates,
        "",
    )


def to_match_row(
    official,
    candidate,
    match_rule="",
):
    family = official[
        "event_family"
    ]

    release_date = official[
        "release_date"
    ]

    if not match_rule:
        match_rule = (
            base_candidate_match_rule(
                family,
                release_date,
                candidate,
            )
        )

    return {
        "event_family":
            family,
        "official_event_timestamp_utc":
            iso_timestamp(
                official[
                    "event_timestamp_utc"
                ]
            ),
        "official_release_date":
            release_date.isoformat(),
        "official_source_agency":
            official[
                "source_agency"
            ],
        "official_reference_period":
            official.get(
                "reference_period",
                "",
            ),
        "oanda_date":
            candidate.get(
                "Date",
                "",
            ),
        "oanda_event":
            candidate.get(
                "Event",
                "",
            ),
        "oanda_report_id":
            report_id_of(
                candidate
            ),
        "oanda_event_id":
            event_id_of(
                candidate
            ),
        "oanda_period":
            candidate.get(
                "Period",
                "",
            ),
        "oanda_priority":
            candidate.get(
                "Priority",
                "",
            ),
        "oanda_actual":
            candidate.get(
                "Actual",
                "",
            ),
        "oanda_forecast":
            candidate.get(
                "Forecast",
                "",
            ),
        "oanda_previous":
            candidate.get(
                "Previous",
                "",
            ),
        "oanda_timestamp":
            candidate.get(
                "Timestamp",
                "",
            ),
        "forecast_present":
            int(
                has_value(
                    candidate.get(
                        "Forecast"
                    )
                )
            ),
        "previous_present":
            int(
                has_value(
                    candidate.get(
                        "Previous"
                    )
                )
            ),
        "actual_present":
            int(
                has_value(
                    candidate.get(
                        "Actual"
                    )
                )
            ),
        "match_rule":
            match_rule,
        "source_file":
            candidate.get(
                "_source_file",
                "",
            ),
    }


def to_ambiguous_row(
    official,
    candidate,
):
    return {
        "event_family":
            official[
                "event_family"
            ],
        "official_event_timestamp_utc":
            iso_timestamp(
                official[
                    "event_timestamp_utc"
                ]
            ),
        "official_release_date":
            official[
                "release_date"
            ].isoformat(),
        "official_source_agency":
            official[
                "source_agency"
            ],
        "official_reference_period":
            official.get(
                "reference_period",
                "",
            ),
        "oanda_date":
            candidate.get(
                "Date",
                "",
            ),
        "oanda_event":
            candidate.get(
                "Event",
                "",
            ),
        "oanda_report_id":
            report_id_of(
                candidate
            ),
        "oanda_event_id":
            event_id_of(
                candidate
            ),
        "oanda_period":
            candidate.get(
                "Period",
                "",
            ),
        "oanda_priority":
            candidate.get(
                "Priority",
                "",
            ),
        "oanda_actual":
            candidate.get(
                "Actual",
                "",
            ),
        "oanda_forecast":
            candidate.get(
                "Forecast",
                "",
            ),
        "oanda_previous":
            candidate.get(
                "Previous",
                "",
            ),
        "oanda_timestamp":
            candidate.get(
                "Timestamp",
                "",
            ),
        "source_file":
            candidate.get(
                "_source_file",
                "",
            ),
    }


def report_id_candidates(
    oanda_rows,
):
    counts = Counter()

    for row in oanda_rows:
        report_id = report_id_of(
            row
        )

        event_name = event_name_of(
            row
        )

        if report_id is None:
            continue

        for (
            family,
            report_ids,
        ) in (
            PINNED_REPORT_IDS.items()
        ):
            if report_id in report_ids:
                key = (
                    family,
                    report_id,
                    event_name,
                )

                counts[
                    key
                ] += 1

        if (
            report_id == 698
            and
            event_name
            == "Consumer Price Index"
        ):
            key = (
                "CPI",
                report_id,
                event_name,
            )

            counts[
                key
            ] += 1

    rows = []

    for (
        family,
        report_id,
        event_name,
    ), count in sorted(
        counts.items()
    ):
        pinned = (
            report_id
            in PINNED_REPORT_IDS.get(
                family,
                set(),
            )
        )

        rows.append({
            "event_family":
                family,
            "oanda_report_id":
                report_id,
            "oanda_event":
                event_name,
            "candidate_matches":
                count,
            "currently_pinned":
                (
                    "true"
                    if pinned
                    else "false"
                ),
        })

    return rows


def main():
    print(
        "OANDA consensus coverage mapper"
    )
    print()
    print(
        "Authoritative source : economic_event"
    )
    print(
        f"Database             : {DB_NAME}"
    )
    print(
        "Read-only DB mode    : enforced"
    )
    print(
        "Coverage window      : "
        f"{START_DATE} .. {END_DATE}"
    )
    print(
        "Matching mode        : "
        "pinned reports + reference-period disambiguation"
    )
    print()

    authoritative = (
        load_authoritative_events()
    )

    family_counts = Counter(
        row[
            "event_family"
        ]
        for row in authoritative
    )

    print(
        "Authoritative events :",
        len(
            authoritative
        ),
    )

    for family in sorted(
        family_counts
    ):
        print(
            f"  {family:<18} "
            f"{family_counts[family]}"
        )

    print()

    oanda_rows = (
        load_cached_oanda_rows()
    )

    date_index = build_date_index(
        oanda_rows
    )

    matches = []
    unmatched = []
    ambiguous = []

    stats = defaultdict(
        Counter
    )

    first_forecast = {}
    last_forecast = {}

    first_previous = {}
    last_previous = {}

    first_match = {}
    last_match = {}

    total = len(
        authoritative
    )

    for index, official in enumerate(
        authoritative,
        start=1,
    ):
        family = official[
            "event_family"
        ]

        release_date = official[
            "release_date"
        ]

        candidates = matching_candidates(
            family,
            release_date,
            date_index,
        )

        if not candidates:
            stats[
                family
            ][
                "unmatched"
            ] += 1

            unmatched.append({
                "event_family":
                    family,
                "official_event_timestamp_utc":
                    iso_timestamp(
                        official[
                            "event_timestamp_utc"
                        ]
                    ),
                "official_release_date":
                    release_date.isoformat(),
                "official_source_agency":
                    official[
                        "source_agency"
                    ],
                "official_reference_period":
                    official.get(
                        "reference_period",
                        "",
                    ),
                "reason":
                    "no_pinned_oanda_candidate",
            })

            print(
                f"[{index:04d}/{total}] "
                f"MISS "
                f"{family:<17} "
                f"{iso_timestamp(official['event_timestamp_utc'])}"
            )

            continue

        candidates, match_rule = (
            resolve_candidates(
                official,
                candidates,
            )
        )

        if len(
            candidates
        ) != 1:
            stats[
                family
            ][
                "ambiguous"
            ] += 1

            for candidate in candidates:
                ambiguous.append(
                    to_ambiguous_row(
                        official,
                        candidate,
                    )
                )

            print(
                f"[{index:04d}/{total}] "
                f"AMB  "
                f"{family:<17} "
                f"{release_date} "
                f"ref={official.get('reference_period', '')!r} "
                f"candidates={len(candidates)}"
            )

            continue

        candidate = (
            candidates[0]
        )

        match = to_match_row(
            official,
            candidate,
            match_rule,
        )

        matches.append(
            match
        )

        stats[
            family
        ][
            "matched"
        ] += 1

        if match[
            "forecast_present"
        ]:
            stats[
                family
            ][
                "forecast"
            ] += 1

            timestamp = official[
                "event_timestamp_utc"
            ]

            if (
                family
                not in first_forecast
                or timestamp
                < first_forecast[
                    family
                ]
            ):
                first_forecast[
                    family
                ] = timestamp

            if (
                family
                not in last_forecast
                or timestamp
                > last_forecast[
                    family
                ]
            ):
                last_forecast[
                    family
                ] = timestamp

        if match[
            "previous_present"
        ]:
            stats[
                family
            ][
                "previous"
            ] += 1

            timestamp = official[
                "event_timestamp_utc"
            ]

            if (
                family
                not in first_previous
                or timestamp
                < first_previous[
                    family
                ]
            ):
                first_previous[
                    family
                ] = timestamp

            if (
                family
                not in last_previous
                or timestamp
                > last_previous[
                    family
                ]
            ):
                last_previous[
                    family
                ] = timestamp

        if match[
            "actual_present"
        ]:
            stats[
                family
            ][
                "actual"
            ] += 1

        timestamp = official[
            "event_timestamp_utc"
        ]

        if (
            family
            not in first_match
            or timestamp
            < first_match[
                family
            ]
        ):
            first_match[
                family
            ] = timestamp

        if (
            family
            not in last_match
            or timestamp
            > last_match[
                family
            ]
        ):
            last_match[
                family
            ] = timestamp

        forecast_flag = (
            "F"
            if match[
                "forecast_present"
            ]
            else "-"
        )

        print(
            f"[{index:04d}/{total}] "
            f"OK   "
            f"{family:<17} "
            f"{release_date} "
            f"{forecast_flag} "
            f"report={match['oanda_report_id']} "
            f"id={match['oanda_event_id']} "
            f"{match['oanda_event']} "
            f"[{match['match_rule']}]"
        )

    write_csv(
        MATCHES_OUTPUT,
        MATCH_FIELDS,
        matches,
    )

    write_csv(
        UNMATCHED_OUTPUT,
        UNMATCHED_FIELDS,
        unmatched,
    )

    write_csv(
        AMBIGUOUS_OUTPUT,
        AMBIGUOUS_FIELDS,
        ambiguous,
    )

    #
    # Family/year coverage.
    #
    family_year = defaultdict(
        Counter
    )

    for official in authoritative:
        key = (
            official[
                "event_family"
            ],
            official[
                "release_date"
            ].year,
        )

        family_year[
            key
        ][
            "expected"
        ] += 1

    for match in matches:
        dt = datetime.fromisoformat(
            match[
                "official_event_timestamp_utc"
            ].replace(
                "Z",
                "+00:00",
            )
        )

        key = (
            match[
                "event_family"
            ],
            dt.year,
        )

        family_year[
            key
        ][
            "matched"
        ] += 1

        family_year[
            key
        ][
            "forecast"
        ] += int(
            match[
                "forecast_present"
            ]
        )

        family_year[
            key
        ][
            "previous"
        ] += int(
            match[
                "previous_present"
            ]
        )

        family_year[
            key
        ][
            "actual"
        ] += int(
            match[
                "actual_present"
            ]
        )

    ambiguous_event_keys = set()

    for row in ambiguous:
        key = (
            row[
                "event_family"
            ],
            row[
                "official_event_timestamp_utc"
            ],
        )

        ambiguous_event_keys.add(
            key
        )

    for (
        family,
        timestamp,
    ) in ambiguous_event_keys:
        dt = datetime.fromisoformat(
            timestamp.replace(
                "Z",
                "+00:00",
            )
        )

        family_year[
            (
                family,
                dt.year,
            )
        ][
            "ambiguous"
        ] += 1

    family_year_rows = []

    for (
        family,
        year,
    ) in sorted(
        family_year
    ):
        counts = family_year[
            (
                family,
                year,
            )
        ]

        expected = counts[
            "expected"
        ]

        matched = counts[
            "matched"
        ]

        forecast = counts[
            "forecast"
        ]

        previous = counts[
            "previous"
        ]

        actual = counts[
            "actual"
        ]

        ambiguous_count = counts[
            "ambiguous"
        ]

        family_year_rows.append({
            "event_family":
                family,
            "release_year":
                year,
            "expected_events":
                expected,
            "matched_events":
                matched,
            "forecast_events":
                forecast,
            "previous_events":
                previous,
            "actual_events":
                actual,
            "ambiguous_events":
                ambiguous_count,
            "match_coverage_pct":
                round(
                    matched
                    / expected,
                    4,
                )
                if expected
                else 0.0,
            "forecast_coverage_pct":
                round(
                    forecast
                    / expected,
                    4,
                )
                if expected
                else 0.0,
            "previous_coverage_pct":
                round(
                    previous
                    / expected,
                    4,
                )
                if expected
                else 0.0,
        })

    write_csv(
        FAMILY_YEAR_OUTPUT,
        [
            "event_family",
            "release_year",
            "expected_events",
            "matched_events",
            "forecast_events",
            "previous_events",
            "actual_events",
            "ambiguous_events",
            "match_coverage_pct",
            "forecast_coverage_pct",
            "previous_coverage_pct",
        ],
        family_year_rows,
    )

    #
    # Summary.
    #
    summary_rows = []

    for family in sorted(
        family_counts
    ):
        expected = family_counts[
            family
        ]

        summary_rows.append({
            "event_family":
                family,
            "expected_events":
                expected,
            "matched_events":
                stats[
                    family
                ][
                    "matched"
                ],
            "forecast_events":
                stats[
                    family
                ][
                    "forecast"
                ],
            "previous_events":
                stats[
                    family
                ][
                    "previous"
                ],
            "ambiguous_events":
                stats[
                    family
                ][
                    "ambiguous"
                ],
            "first_official_release":
                next(
                    (
                        iso_timestamp(
                            r[
                                "event_timestamp_utc"
                            ]
                        )
                        for r in authoritative
                        if r[
                            "event_family"
                        ]
                        == family
                    ),
                    "",
                ),
            "first_oanda_match":
                (
                    iso_timestamp(
                        first_match[
                            family
                        ]
                    )
                    if family
                    in first_match
                    else ""
                ),
            "last_oanda_match":
                (
                    iso_timestamp(
                        last_match[
                            family
                        ]
                    )
                    if family
                    in last_match
                    else ""
                ),
            "first_forecast":
                (
                    iso_timestamp(
                        first_forecast[
                            family
                        ]
                    )
                    if family
                    in first_forecast
                    else ""
                ),
            "last_forecast":
                (
                    iso_timestamp(
                        last_forecast[
                            family
                        ]
                    )
                    if family
                    in last_forecast
                    else ""
                ),
            "first_previous":
                (
                    iso_timestamp(
                        first_previous[
                            family
                        ]
                    )
                    if family
                    in first_previous
                    else ""
                ),
            "last_previous":
                (
                    iso_timestamp(
                        last_previous[
                            family
                        ]
                    )
                    if family
                    in last_previous
                    else ""
                ),
            "forecast_coverage_pct":
                round(
                    stats[
                        family
                    ][
                        "forecast"
                    ]
                    / expected,
                    4,
                )
                if expected
                else 0.0,
        })

    write_csv(
        SUMMARY_OUTPUT,
        [
            "event_family",
            "expected_events",
            "matched_events",
            "forecast_events",
            "previous_events",
            "ambiguous_events",
            "first_official_release",
            "first_oanda_match",
            "last_oanda_match",
            "first_forecast",
            "last_forecast",
            "first_previous",
            "last_previous",
            "forecast_coverage_pct",
        ],
        summary_rows,
    )

    report_rows = (
        report_id_candidates(
            oanda_rows
        )
    )

    write_csv(
        REPORT_ID_OUTPUT,
        [
            "event_family",
            "oanda_report_id",
            "oanda_event",
            "candidate_matches",
            "currently_pinned",
        ],
        report_rows,
    )

    write_csv(
        HTTP_FAILURES_OUTPUT,
        [
            "date",
            "error",
        ],
        [],
    )

    total_matches = len(
        matches
    )

    total_unmatched = len(
        unmatched
    )

    total_ambiguous = len(
        ambiguous_event_keys
    )

    print()
    print(
        "=" * 112
    )
    print(
        "OANDA CONSENSUS COVERAGE SUMMARY"
    )
    print(
        "=" * 112
    )
    print()

    print(
        f"{'FAMILY':<20}"
        f"{'EXP':>7}"
        f"{'MATCH':>8}"
        f"{'FCST':>7}"
        f"{'PREV':>7}"
        f"{'AMB':>7}  "
        f"{'FIRST FORECAST':<27}"
        f"{'LAST FORECAST':<27}"
    )

    print(
        "-" * 112
    )

    for row in summary_rows:
        print(
            f"{row['event_family']:<20}"
            f"{row['expected_events']:>7}"
            f"{row['matched_events']:>8}"
            f"{row['forecast_events']:>7}"
            f"{row['previous_events']:>7}"
            f"{row['ambiguous_events']:>7}  "
            f"{row['first_forecast']:<27}"
            f"{row['last_forecast']:<27}"
        )

    print()
    print(
        f"Matched rows     : {total_matches}"
    )
    print(
        f"Unmatched rows   : {total_unmatched}"
    )
    print(
        f"Ambiguous rows   : {total_ambiguous}"
    )
    print(
        "HTTP failures    : 0"
    )
    print()

    print(
        f"Matches          : {MATCHES_OUTPUT}"
    )
    print(
        f"Unmatched        : {UNMATCHED_OUTPUT}"
    )
    print(
        f"Ambiguous        : {AMBIGUOUS_OUTPUT}"
    )
    print(
        f"Family/year      : {FAMILY_YEAR_OUTPUT}"
    )
    print(
        f"Summary          : {SUMMARY_OUTPUT}"
    )
    print(
        f"Report IDs       : {REPORT_ID_OUTPUT}"
    )
    print(
        f"HTTP failures    : {HTTP_FAILURES_OUTPUT}"
    )
    print()
    print(
        "No network requests were performed."
    )
    print(
        "No database writes were performed."
    )
    print(
        "OANDA data was used only as a read-only "
        "consensus-coverage candidate source."
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(
        main()
    )
