#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import os
import re
import subprocess
import sys
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

ROOT = Path("EconomicCalendar/raw/bea")
DEFAULT_INPUT = ROOT / "bea_canonical_events.csv"
DEFAULT_REPORT = ROOT / "bea_import_prepared.csv"

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EXPECTED_AGENCY = "BEA"
EXPECTED_CURRENCY = "USD"
EXPECTED_IMPORTANCE = 3
EXPECTED_CONFIDENCE = "exact"
EXPECTED_TIMEZONE = "America/New_York"

QUARTER_RE = re.compile(
    r"""
    (?:
        (?P<qnum>[1-4])(?:st|nd|rd|th)\s+quarter
        |
        (?P<qword>first|second|third|fourth)\s+quarter
    )
    (?:\s+and\s+(?:annual|year))?
    \s+
    (?P<year>20\d{2})
    """,
    re.I | re.X,
)

GDP_ESTIMATE_RE = re.compile(
    r"\b(advance|initial|second|third|updated)\s+estimate\b",
    re.I,
)

MONTH_WORD = (
    r"January|February|March|April|May|June|"
    r"July|August|September|October|November|December"
)

PCE_NORMAL_RE = re.compile(
    rf"""
    personal\s+income\s+and\s+outlays
    [,:]?\s*
    (?P<m1>{MONTH_WORD})
    (?:\s+and\s+(?P<m2>{MONTH_WORD}))?
    \s+
    (?P<year>20\d{{2}})
    """,
    re.I | re.X,
)

PCE_OUTLAYS_RE = re.compile(
    rf"""
    personal\s+outlays
    [,:]?\s*
    (?P<month>{MONTH_WORD})
    \s+
    (?P<year>20\d{{2}})
    """,
    re.I | re.X,
)


def sql_literal(value) -> str:
    if value is None:
        return "NULL"

    return "'" + str(value).replace("'", "''") + "'"


def run_psql(args, sql: str) -> subprocess.CompletedProcess:
    cmd = [
        args.psql,
        "-X",
        "-v",
        "ON_ERROR_STOP=1",
        "-d",
        args.db,
    ]

    if args.host:
        cmd += ["-h", args.host]

    if args.port:
        cmd += ["-p", str(args.port)]

    if args.user:
        cmd += ["-U", args.user]

    env = os.environ.copy()

    return subprocess.run(
        cmd,
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )


def normalize_timestamp(value: str) -> datetime:
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        raise ValueError("timezone-naive event_timestamp_utc")

    return dt.astimezone(timezone.utc)


def compact_utc(dt: datetime) -> str:
    return dt.astimezone(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def slugify(value: str) -> str:
    value = value.lower()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-")


def derive_gdp_reference(title: str) -> str:
    qm = QUARTER_RE.search(title)
    em = GDP_ESTIMATE_RE.search(title)

    if not qm:
        raise ValueError(
            f"could not derive GDP quarter/year from title: {title!r}"
        )

    if not em:
        raise ValueError(
            f"could not derive GDP estimate type from title: {title!r}"
        )

    if qm.group("qnum"):
        quarter = int(qm.group("qnum"))
    else:
        quarter = {
            "first": 1,
            "second": 2,
            "third": 3,
            "fourth": 4,
        }[qm.group("qword").lower()]

    year = int(qm.group("year"))
    estimate = em.group(1).capitalize()

    return f"Q{quarter} {year} {estimate}"


def derive_pce_reference(title: str) -> str:
    #
    # Normal case:
    #   Personal Income and Outlays, April 2024
    #
    # Combined case:
    #   Personal Income and Outlays, October and November 2025
    #
    m = PCE_NORMAL_RE.search(title)

    if m:
        year = int(m.group("year"))
        m1 = m.group("m1").capitalize()
        m2 = m.group("m2")

        if m2:
            return f"{m1} and {m2.capitalize()} {year}"

        return f"{m1} {year}"

    #
    # 2019 shutdown special:
    #   Personal Income, February 2019;
    #   Personal Outlays, January 2019
    #
    # For PCE/outlays, January is the applicable reference period.
    #
    m = PCE_OUTLAYS_RE.search(title)

    if m:
        return (
            f"{m.group('month').capitalize()} "
            f"{int(m.group('year'))}"
        )

    raise ValueError(
        f"could not derive PCE reference period from title: {title!r}"
    )


def derive_reference_period(family: str, title: str) -> str:
    if family == "GDP":
        return derive_gdp_reference(title)

    if family == "PCE":
        return derive_pce_reference(title)

    raise ValueError(f"unsupported event family: {family}")


def prepare_row(row: dict) -> dict:
    agency = row.get("source_agency", "").strip()
    family = row.get("event_family", "").strip()
    timestamp_raw = row.get("event_timestamp_utc", "").strip()
    local_date_raw = row.get("source_local_date", "").strip()
    local_time_raw = row.get("source_local_time", "").strip()
    source_timezone = row.get("source_timezone", "").strip()
    title = row.get("title", "").strip()
    url = row.get("url", "").strip()

    if agency != EXPECTED_AGENCY:
        raise ValueError(
            f"unexpected source_agency {agency!r}"
        )

    if family not in {"GDP", "PCE"}:
        raise ValueError(
            f"unexpected event_family {family!r}"
        )

    if source_timezone != EXPECTED_TIMEZONE:
        raise ValueError(
            f"unexpected source_timezone {source_timezone!r}"
        )

    if not title:
        raise ValueError("empty title")

    if not url:
        raise ValueError("empty url")

    if not url.startswith("https://www.bea.gov/"):
        raise ValueError(
            f"unexpected BEA source URL {url!r}"
        )

    utc_dt = normalize_timestamp(timestamp_raw)

    local_date = datetime.strptime(
        local_date_raw,
        "%Y-%m-%d",
    ).date()

    local_time = datetime.strptime(
        local_time_raw,
        "%H:%M:%S",
    ).time()

    if not START_DATE <= local_date <= END_DATE:
        raise ValueError(
            f"release date {local_date} outside "
            f"{START_DATE}..{END_DATE}"
        )

    reference_period = derive_reference_period(
        family,
        title,
    )

    event_id_suffix = slugify(reference_period)

    source_event_id = (
        f"bea:{family.lower()}:"
        f"{compact_utc(utc_dt)}:"
        f"{event_id_suffix}"
    )

    return {
        "currency": EXPECTED_CURRENCY,
        "event_family": family,
        "event_timestamp_utc": utc_dt.isoformat(),
        "source_agency": EXPECTED_AGENCY,
        "source_event_id": source_event_id,
        "source_url": url,
        "reference_period": reference_period,
        "event_importance": EXPECTED_IMPORTANCE,
        "historical_time_confidence": EXPECTED_CONFIDENCE,
        "source_release_date": local_date.isoformat(),
        "source_release_time": local_time.isoformat(),
        "source_timezone": EXPECTED_TIMEZONE,
        "title": title,
    }


def load_and_prepare(path: Path):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        source_rows = list(csv.DictReader(f))

    prepared = []
    errors = []

    for line_number, row in enumerate(
        source_rows,
        start=2,
    ):
        try:
            prepared.append(
                prepare_row(row)
            )
        except Exception as exc:
            errors.append(
                (line_number, str(exc), row)
            )

    return source_rows, prepared, errors


def write_prepared_report(
    path: Path,
    prepared: list[dict],
):
    fields = [
        "currency",
        "event_family",
        "event_timestamp_utc",
        "source_agency",
        "source_event_id",
        "source_url",
        "reference_period",
        "event_importance",
        "historical_time_confidence",
        "source_release_date",
        "source_release_time",
        "source_timezone",
        "title",
    ]

    with path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        w = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        w.writeheader()
        w.writerows(prepared)


def validate_prepared(prepared: list[dict]):
    canonical = Counter()
    source_ids = Counter()

    for row in prepared:
        canonical[
            (
                row["source_agency"],
                row["event_family"],
                row["event_timestamp_utc"],
            )
        ] += 1

        source_ids[
            (
                row["source_agency"],
                row["source_event_id"],
            )
        ] += 1

    duplicate_keys = [
        k for k, count in canonical.items()
        if count != 1
    ]

    duplicate_source_ids = [
        k for k, count in source_ids.items()
        if count != 1
    ]

    if duplicate_keys:
        raise ValueError(
            f"{len(duplicate_keys)} duplicate canonical key(s) "
            "in prepared data"
        )

    if duplicate_source_ids:
        raise ValueError(
            f"{len(duplicate_source_ids)} duplicate source_event_id(s) "
            "in prepared data"
        )


def values_sql(prepared: list[dict]) -> str:
    values = []

    for row in prepared:
        values.append(
            "("
            + ", ".join([
                sql_literal(row["currency"]),
                sql_literal(row["event_family"]),
                sql_literal(row["event_timestamp_utc"]),
                sql_literal(row["source_agency"]),
                sql_literal(row["source_event_id"]),
                sql_literal(row["source_url"]),
                sql_literal(row["reference_period"]),
                str(row["event_importance"]),
                sql_literal(
                    row["historical_time_confidence"]
                ),
                sql_literal(row["source_release_date"]),
                sql_literal(row["source_release_time"]),
                sql_literal(row["source_timezone"]),
            ])
            + ")"
        )

    return ",\n".join(values)


def build_sql(
    prepared: list[dict],
    commit: bool,
) -> str:
    vals = values_sql(prepared)
    source_count = len(prepared)

    transaction_end = "COMMIT;" if commit else "ROLLBACK;"

    return f"""
\\set ON_ERROR_STOP on

BEGIN;

CREATE TEMP TABLE bea_import_source (
    currency text NOT NULL,
    event_family text NOT NULL,
    event_timestamp_utc timestamptz NOT NULL,
    source_agency text NOT NULL,
    source_event_id text NOT NULL,
    source_url text NOT NULL,
    reference_period text,
    event_importance smallint NOT NULL,
    historical_time_confidence text NOT NULL,
    source_release_date date,
    source_release_time time,
    source_timezone text
) ON COMMIT DROP;

INSERT INTO bea_import_source (
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
)
VALUES
{vals};

DO $$
DECLARE
    source_rows bigint;
    source_distinct_keys bigint;
    source_distinct_ids bigint;
BEGIN
    SELECT COUNT(*)
    INTO source_rows
    FROM bea_import_source;

    IF source_rows <> {source_count} THEN
        RAISE EXCEPTION
            'staging row count mismatch: expected %, found %',
            {source_count},
            source_rows;
    END IF;

    SELECT COUNT(*)
    INTO source_distinct_keys
    FROM (
        SELECT DISTINCT
            source_agency,
            event_family,
            event_timestamp_utc
        FROM bea_import_source
    ) q;

    IF source_distinct_keys <> source_rows THEN
        RAISE EXCEPTION
            'duplicate canonical key inside staging data';
    END IF;

    SELECT COUNT(*)
    INTO source_distinct_ids
    FROM (
        SELECT DISTINCT
            source_agency,
            source_event_id
        FROM bea_import_source
    ) q;

    IF source_distinct_ids <> source_rows THEN
        RAISE EXCEPTION
            'duplicate source_event_id inside staging data';
    END IF;
END
$$;

CREATE TEMP TABLE bea_import_metrics (
    metric text PRIMARY KEY,
    value bigint NOT NULL
) ON COMMIT DROP;

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'pre_bea_count',
    COUNT(*)
FROM economic_event
WHERE source_agency = 'BEA'
  AND event_family IN ('GDP', 'PCE');

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'pre_matching_keys',
    COUNT(*)
FROM bea_import_source s
JOIN economic_event e
  ON e.source_agency = s.source_agency
 AND e.event_family = s.event_family
 AND e.event_timestamp_utc = s.event_timestamp_utc;

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'expected_insert',
    {source_count} - value
FROM bea_import_metrics
WHERE metric = 'pre_matching_keys';

CREATE TEMP TABLE bea_inserted_ids (
    economic_event_id bigint PRIMARY KEY
) ON COMMIT DROP;

WITH inserted AS (
    INSERT INTO economic_event (
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
    )
    SELECT
        s.currency,
        s.event_family,
        s.event_timestamp_utc,
        s.source_agency,
        s.source_event_id,
        s.source_url,
        s.reference_period,
        s.event_importance,
        s.historical_time_confidence,
        s.source_release_date,
        s.source_release_time,
        s.source_timezone
    FROM bea_import_source s
    ORDER BY
        s.event_timestamp_utc,
        s.event_family
    ON CONFLICT (
        source_agency,
        event_family,
        event_timestamp_utc
    )
    DO NOTHING
    RETURNING economic_event_id
)
INSERT INTO bea_inserted_ids(economic_event_id)
SELECT economic_event_id
FROM inserted;

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'inserted',
    COUNT(*)
FROM bea_inserted_ids;

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'post_matching_keys',
    COUNT(*)
FROM bea_import_source s
JOIN economic_event e
  ON e.source_agency = s.source_agency
 AND e.event_family = s.event_family
 AND e.event_timestamp_utc = s.event_timestamp_utc;

INSERT INTO bea_import_metrics(metric, value)
SELECT
    'post_bea_count',
    COUNT(*)
FROM economic_event
WHERE source_agency = 'BEA'
  AND event_family IN ('GDP', 'PCE');

DO $$
DECLARE
    pre_bea bigint;
    pre_match bigint;
    expected bigint;
    inserted_count bigint;
    post_match bigint;
    post_bea bigint;
BEGIN
    SELECT value INTO pre_bea
    FROM bea_import_metrics
    WHERE metric = 'pre_bea_count';

    SELECT value INTO pre_match
    FROM bea_import_metrics
    WHERE metric = 'pre_matching_keys';

    SELECT value INTO expected
    FROM bea_import_metrics
    WHERE metric = 'expected_insert';

    SELECT value INTO inserted_count
    FROM bea_import_metrics
    WHERE metric = 'inserted';

    SELECT value INTO post_match
    FROM bea_import_metrics
    WHERE metric = 'post_matching_keys';

    SELECT value INTO post_bea
    FROM bea_import_metrics
    WHERE metric = 'post_bea_count';

    IF inserted_count <> expected THEN
        RAISE EXCEPTION
            'insert count mismatch: expected %, inserted %',
            expected,
            inserted_count;
    END IF;

    IF post_match <> {source_count} THEN
        RAISE EXCEPTION
            'post-import canonical coverage mismatch: expected %, found %',
            {source_count},
            post_match;
    END IF;

    IF post_bea <> pre_bea + inserted_count THEN
        RAISE EXCEPTION
            'BEA row-count delta mismatch: pre %, inserted %, post %',
            pre_bea,
            inserted_count,
            post_bea;
    END IF;

    RAISE NOTICE
        'BEA_IMPORT_VERIFY source={source_count} pre_match=% expected_insert=% inserted=% post_match=% pre_bea=% post_bea=%',
        pre_match,
        expected,
        inserted_count,
        post_match,
        pre_bea,
        post_bea;
END
$$;

SELECT
    metric,
    value
FROM bea_import_metrics
ORDER BY
    CASE metric
        WHEN 'pre_bea_count' THEN 1
        WHEN 'pre_matching_keys' THEN 2
        WHEN 'expected_insert' THEN 3
        WHEN 'inserted' THEN 4
        WHEN 'post_matching_keys' THEN 5
        WHEN 'post_bea_count' THEN 6
        ELSE 99
    END;

{transaction_end}
"""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import canonical BEA GDP/PCE events into economic_event. "
            "Default mode performs the real INSERT path and ROLLBACKs. "
            "Use --commit to persist only after all invariants pass."
        )
    )

    mode = parser.add_mutually_exclusive_group()

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Execute import transaction and ROLLBACK (default).",
    )

    mode.add_argument(
        "--commit",
        action="store_true",
        help="Persist import if all verification checks pass.",
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )

    parser.add_argument(
        "--prepared-report",
        default=str(DEFAULT_REPORT),
    )

    parser.add_argument("--db", default="LSTM")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--user")
    parser.add_argument("--psql", default="psql")

    args = parser.parse_args()

    input_path = Path(args.input)
    report_path = Path(args.prepared_report)

    source_rows, prepared, errors = load_and_prepare(
        input_path
    )

    print("BEA economic_event importer")
    print()
    print(f"Input          : {input_path}")
    print(f"Source rows    : {len(source_rows)}")
    print(f"Prepared rows  : {len(prepared)}")
    print(f"Prepare errors : {len(errors)}")
    print(
        f"Mode           : "
        f"{'COMMIT' if args.commit else 'DRY RUN / ROLLBACK'}"
    )
    print()

    if errors:
        for line, message, row in errors:
            print(
                f"ERROR source row {line}: {message}",
                file=sys.stderr,
            )

        print(
            "RESULT: ABORTED before database transaction",
            file=sys.stderr,
        )
        return 1

    if len(prepared) != len(source_rows):
        print(
            "ERROR: prepared/source row-count mismatch",
            file=sys.stderr,
        )
        return 1

    try:
        validate_prepared(prepared)
    except Exception as exc:
        print(
            f"ERROR: prepared-data validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    write_prepared_report(
        report_path,
        prepared,
    )

    counts = Counter(
        row["event_family"]
        for row in prepared
    )

    print(f"GDP prepared   : {counts['GDP']}")
    print(f"PCE prepared   : {counts['PCE']}")
    print(f"Prepared report: {report_path}")
    print()

    sql = build_sql(
        prepared,
        commit=args.commit,
    )

    result = run_psql(
        args,
        sql,
    )

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        print()
        print(
            "RESULT: FAILED - PostgreSQL transaction did not complete."
        )
        print(
            "No partial import can survive because the operation "
            "uses one transaction."
        )
        return result.returncode

    print()

    if args.commit:
        print(
            "RESULT: COMMITTED - BEA economic_event import "
            "passed all transaction invariants."
        )
    else:
        print(
            "RESULT: CLEAN DRY RUN - identical INSERT path passed; "
            "transaction was ROLLED BACK."
        )
        print("No database changes were retained.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
