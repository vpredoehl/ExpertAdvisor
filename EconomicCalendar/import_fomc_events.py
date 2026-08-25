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

ROOT = Path("EconomicCalendar/raw/federal_reserve")

DEFAULT_INPUT = ROOT / "fomc_canonical_events.csv"
DEFAULT_REPORT = ROOT / "fomc_import_prepared.csv"

VALIDATOR = Path("EconomicCalendar/validate_fomc_canonical.py")

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EXPECTED_ROWS = 136

EXPECTED_AGENCY = "FEDERAL_RESERVE"
EXPECTED_FAMILY = "FOMC"
EXPECTED_CURRENCY = "USD"
EXPECTED_IMPORTANCE = 3
EXPECTED_CONFIDENCE = "exact"
EXPECTED_TIMEZONE = "America/New_York"

ALLOWED_PARSERS = {
    "for_release_at",
    "fomc_minutes_statement_release",
}


def sql_literal(value) -> str:
    if value is None:
        return "NULL"

    return "'" + str(value).replace("'", "''") + "'"


def run_command(cmd: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def run_validator() -> None:
    if not VALIDATOR.exists():
        raise SystemExit(
            f"ERROR: required validator is missing: {VALIDATOR}"
        )

    result = run_command([
        sys.executable,
        str(VALIDATOR),
    ])

    if result.stdout:
        print(result.stdout, end="")

    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)

    if result.returncode != 0:
        raise SystemExit(
            "ERROR: FOMC canonical validation failed. "
            "Import aborted before database access."
        )


def run_psql(
    args,
    sql: str,
) -> subprocess.CompletedProcess:
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
        raise ValueError(
            "timezone-naive event_timestamp_utc"
        )

    return dt.astimezone(timezone.utc)


def compact_utc(dt: datetime) -> str:
    return dt.astimezone(
        timezone.utc
    ).strftime("%Y%m%dT%H%M%SZ")


def derive_reference_period(
    local_date: date,
) -> str:
    #
    # FOMC does not have a monthly/quarterly reference period
    # analogous to CPI/PCE/GDP.
    #
    # Store the actual statement date as a deterministic,
    # human-readable reference.
    #
    return f"FOMC {local_date.isoformat()}"


def derive_source_event_id(
    utc_dt: datetime,
) -> str:
    return (
        "federal_reserve:fomc:"
        f"{compact_utc(utc_dt)}"
    )


def prepare_row(row: dict) -> dict:
    agency = row.get(
        "source_agency",
        "",
    ).strip()

    family = row.get(
        "event_family",
        "",
    ).strip()

    timestamp_raw = row.get(
        "event_timestamp_utc",
        "",
    ).strip()

    local_date_raw = row.get(
        "source_local_date",
        "",
    ).strip()

    local_time_raw = row.get(
        "source_local_time",
        "",
    ).strip()

    source_timezone = row.get(
        "source_timezone",
        "",
    ).strip()

    title = row.get(
        "title",
        "",
    ).strip()

    url = row.get(
        "url",
        "",
    ).strip()

    parser_name = row.get(
        "timestamp_parser",
        "",
    ).strip()

    timestamp_source_url = row.get(
        "timestamp_source_url",
        "",
    ).strip()

    if agency != EXPECTED_AGENCY:
        raise ValueError(
            f"unexpected source_agency {agency!r}"
        )

    if family != EXPECTED_FAMILY:
        raise ValueError(
            f"unexpected event_family {family!r}"
        )

    if source_timezone != EXPECTED_TIMEZONE:
        raise ValueError(
            f"unexpected source_timezone "
            f"{source_timezone!r}"
        )

    if parser_name not in ALLOWED_PARSERS:
        raise ValueError(
            f"unexpected timestamp_parser "
            f"{parser_name!r}"
        )

    if not title:
        raise ValueError(
            "empty title"
        )

    if not url:
        raise ValueError(
            "empty source URL"
        )

    if not url.startswith(
        "https://www.federalreserve.gov/"
    ):
        raise ValueError(
            f"unexpected Federal Reserve URL "
            f"{url!r}"
        )

    if (
        parser_name
        == "fomc_minutes_statement_release"
    ):
        if not timestamp_source_url:
            raise ValueError(
                "minutes-recovered event has "
                "empty timestamp_source_url"
            )

        if not timestamp_source_url.startswith(
            "https://www.federalreserve.gov/"
        ):
            raise ValueError(
                "minutes timestamp_source_url "
                "is not federalreserve.gov"
            )

    utc_dt = normalize_timestamp(
        timestamp_raw
    )

    local_date = datetime.strptime(
        local_date_raw,
        "%Y-%m-%d",
    ).date()

    local_time = datetime.strptime(
        local_time_raw,
        "%H:%M:%S",
    ).time()

    if not (
        START_DATE
        <= local_date
        <= END_DATE
    ):
        raise ValueError(
            f"release date {local_date} "
            f"outside "
            f"{START_DATE}..{END_DATE}"
        )

    reference_period = (
        derive_reference_period(
            local_date
        )
    )

    source_event_id = (
        derive_source_event_id(
            utc_dt
        )
    )

    return {
        "currency":
            EXPECTED_CURRENCY,

        "event_family":
            EXPECTED_FAMILY,

        "event_timestamp_utc":
            utc_dt.isoformat(),

        "source_agency":
            EXPECTED_AGENCY,

        "source_event_id":
            source_event_id,

        #
        # economic_event has one source_url column.
        #
        # Preserve the actual FOMC statement URL here.
        # For minutes-recovered historical timestamps,
        # timestamp_source_url remains preserved in the
        # canonical/raw provenance files.
        #
        "source_url":
            url,

        "reference_period":
            reference_period,

        "event_importance":
            EXPECTED_IMPORTANCE,

        "historical_time_confidence":
            EXPECTED_CONFIDENCE,

        "source_release_date":
            local_date.isoformat(),

        "source_release_time":
            local_time.isoformat(),

        "source_timezone":
            EXPECTED_TIMEZONE,

        #
        # These are retained only in the prepared audit
        # report. They are not economic_event columns.
        #
        "title":
            title,

        "timestamp_parser":
            parser_name,

        "timestamp_source_url":
            timestamp_source_url,
    }


def load_and_prepare(
    path: Path,
):
    with path.open(
        newline="",
        encoding="utf-8",
    ) as f:
        source_rows = list(
            csv.DictReader(f)
        )

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
                (
                    line_number,
                    str(exc),
                    row,
                )
            )

    return (
        source_rows,
        prepared,
        errors,
    )


def validate_prepared(
    prepared: list[dict],
) -> None:
    if len(prepared) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly "
            f"{EXPECTED_ROWS} prepared rows, "
            f"found {len(prepared)}"
        )

    canonical_keys = Counter()
    source_ids = Counter()

    for row in prepared:
        canonical_keys[
            (
                row["source_agency"],
                row["event_family"],
                row[
                    "event_timestamp_utc"
                ],
            )
        ] += 1

        source_ids[
            (
                row["source_agency"],
                row["source_event_id"],
            )
        ] += 1

    duplicate_keys = [
        key
        for key, count
        in canonical_keys.items()
        if count != 1
    ]

    duplicate_ids = [
        key
        for key, count
        in source_ids.items()
        if count != 1
    ]

    if duplicate_keys:
        raise ValueError(
            f"{len(duplicate_keys)} "
            "duplicate canonical key(s) "
            "in prepared data"
        )

    if duplicate_ids:
        raise ValueError(
            f"{len(duplicate_ids)} "
            "duplicate source_event_id(s) "
            "in prepared data"
        )


def write_prepared_report(
    path: Path,
    prepared: list[dict],
) -> None:
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
        "timestamp_parser",
        "timestamp_source_url",
    ]

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
        writer.writerows(prepared)


def values_sql(
    prepared: list[dict],
) -> str:
    rows = []

    for row in prepared:
        rows.append(
            "("
            + ", ".join([
                sql_literal(
                    row["currency"]
                ),
                sql_literal(
                    row["event_family"]
                ),
                sql_literal(
                    row[
                        "event_timestamp_utc"
                    ]
                ),
                sql_literal(
                    row["source_agency"]
                ),
                sql_literal(
                    row["source_event_id"]
                ),
                sql_literal(
                    row["source_url"]
                ),
                sql_literal(
                    row["reference_period"]
                ),
                str(
                    row["event_importance"]
                ),
                sql_literal(
                    row[
                        "historical_time_confidence"
                    ]
                ),
                sql_literal(
                    row[
                        "source_release_date"
                    ]
                ),
                sql_literal(
                    row[
                        "source_release_time"
                    ]
                ),
                sql_literal(
                    row["source_timezone"]
                ),
            ])
            + ")"
        )

    return ",\n".join(rows)


def build_sql(
    prepared: list[dict],
    commit: bool,
) -> str:
    values = values_sql(
        prepared
    )

    source_count = len(
        prepared
    )

    transaction_end = (
        "COMMIT;"
        if commit
        else "ROLLBACK;"
    )

    return f"""
\\set ON_ERROR_STOP on

BEGIN;

CREATE TEMP TABLE fomc_import_source (
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

INSERT INTO fomc_import_source (
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
{values};

DO $$
DECLARE
    source_rows bigint;
    distinct_keys bigint;
    distinct_ids bigint;
BEGIN
    SELECT COUNT(*)
    INTO source_rows
    FROM fomc_import_source;

    IF source_rows <> {source_count} THEN
        RAISE EXCEPTION
            'FOMC staging row count mismatch: expected %, found %',
            {source_count},
            source_rows;
    END IF;

    SELECT COUNT(*)
    INTO distinct_keys
    FROM (
        SELECT DISTINCT
            source_agency,
            event_family,
            event_timestamp_utc
        FROM fomc_import_source
    ) q;

    IF distinct_keys <> source_rows THEN
        RAISE EXCEPTION
            'duplicate FOMC canonical key inside staging data';
    END IF;

    SELECT COUNT(*)
    INTO distinct_ids
    FROM (
        SELECT DISTINCT
            source_agency,
            source_event_id
        FROM fomc_import_source
    ) q;

    IF distinct_ids <> source_rows THEN
        RAISE EXCEPTION
            'duplicate FOMC source_event_id inside staging data';
    END IF;
END
$$;


CREATE TEMP TABLE fomc_import_metrics (
    metric text PRIMARY KEY,
    value bigint NOT NULL
) ON COMMIT DROP;


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'pre_fomc_count',
    COUNT(*)
FROM economic_event
WHERE source_agency = '{EXPECTED_AGENCY}'
  AND event_family = '{EXPECTED_FAMILY}';


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'pre_matching_keys',
    COUNT(*)
FROM fomc_import_source s
JOIN economic_event e
  ON e.source_agency =
     s.source_agency
 AND e.event_family =
     s.event_family
 AND e.event_timestamp_utc =
     s.event_timestamp_utc;


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'expected_insert',
    {source_count} - value
FROM fomc_import_metrics
WHERE metric =
      'pre_matching_keys';


CREATE TEMP TABLE fomc_inserted_ids (
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
    FROM fomc_import_source s
    ORDER BY
        s.event_timestamp_utc
    ON CONFLICT (
        source_agency,
        event_family,
        event_timestamp_utc
    )
    DO NOTHING
    RETURNING
        economic_event_id
)
INSERT INTO fomc_inserted_ids (
    economic_event_id
)
SELECT economic_event_id
FROM inserted;


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'inserted',
    COUNT(*)
FROM fomc_inserted_ids;


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'post_matching_keys',
    COUNT(*)
FROM fomc_import_source s
JOIN economic_event e
  ON e.source_agency =
     s.source_agency
 AND e.event_family =
     s.event_family
 AND e.event_timestamp_utc =
     s.event_timestamp_utc;


INSERT INTO fomc_import_metrics (
    metric,
    value
)
SELECT
    'post_fomc_count',
    COUNT(*)
FROM economic_event
WHERE source_agency = '{EXPECTED_AGENCY}'
  AND event_family = '{EXPECTED_FAMILY}';


DO $$
DECLARE
    pre_fomc bigint;
    pre_match bigint;
    expected bigint;
    inserted_count bigint;
    post_match bigint;
    post_fomc bigint;
BEGIN
    SELECT value
    INTO pre_fomc
    FROM fomc_import_metrics
    WHERE metric =
          'pre_fomc_count';

    SELECT value
    INTO pre_match
    FROM fomc_import_metrics
    WHERE metric =
          'pre_matching_keys';

    SELECT value
    INTO expected
    FROM fomc_import_metrics
    WHERE metric =
          'expected_insert';

    SELECT value
    INTO inserted_count
    FROM fomc_import_metrics
    WHERE metric =
          'inserted';

    SELECT value
    INTO post_match
    FROM fomc_import_metrics
    WHERE metric =
          'post_matching_keys';

    SELECT value
    INTO post_fomc
    FROM fomc_import_metrics
    WHERE metric =
          'post_fomc_count';


    IF inserted_count <> expected THEN
        RAISE EXCEPTION
            'FOMC insert count mismatch: expected %, inserted %',
            expected,
            inserted_count;
    END IF;


    IF post_match <> {source_count} THEN
        RAISE EXCEPTION
            'FOMC post-import canonical coverage mismatch: '
            'expected %, found %',
            {source_count},
            post_match;
    END IF;


    IF post_fomc <>
       pre_fomc + inserted_count
    THEN
        RAISE EXCEPTION
            'FOMC row-count delta mismatch: pre %, inserted %, post %',
            pre_fomc,
            inserted_count,
            post_fomc;
    END IF;


    IF post_match <> {EXPECTED_ROWS} THEN
        RAISE EXCEPTION
            'FOMC canonical coverage must equal {EXPECTED_ROWS}; found %',
            post_match;
    END IF;


    RAISE NOTICE
        'FOMC_IMPORT_VERIFY source={source_count} '
        'pre_match=% expected_insert=% inserted=% '
        'post_match=% pre_fomc=% post_fomc=%',
        pre_match,
        expected,
        inserted_count,
        post_match,
        pre_fomc,
        post_fomc;
END
$$;


SELECT
    metric,
    value
FROM fomc_import_metrics
ORDER BY
    CASE metric
        WHEN 'pre_fomc_count'
            THEN 1
        WHEN 'pre_matching_keys'
            THEN 2
        WHEN 'expected_insert'
            THEN 3
        WHEN 'inserted'
            THEN 4
        WHEN 'post_matching_keys'
            THEN 5
        WHEN 'post_fomc_count'
            THEN 6
        ELSE 99
    END;


{transaction_end}
"""


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Import canonical Federal Reserve FOMC "
            "statement events into economic_event. "
            "Default mode executes the real INSERT path "
            "inside one transaction and ROLLBACKs. "
            "Use --commit to persist only after all "
            "verification checks pass."
        )
    )

    mode = (
        parser
        .add_mutually_exclusive_group()
    )

    mode.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Execute actual import path and "
            "ROLLBACK (default)."
        ),
    )

    mode.add_argument(
        "--commit",
        action="store_true",
        help=(
            "Persist import only if all "
            "verification checks pass."
        ),
    )

    parser.add_argument(
        "--input",
        default=str(
            DEFAULT_INPUT
        ),
    )

    parser.add_argument(
        "--prepared-report",
        default=str(
            DEFAULT_REPORT
        ),
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

    parser.add_argument(
        "--skip-validator",
        action="store_true",
        help=(
            "Skip validate_fomc_canonical.py. "
            "Not recommended."
        ),
    )

    args = parser.parse_args()

    input_path = Path(
        args.input
    )

    report_path = Path(
        args.prepared_report
    )

    print(
        "Federal Reserve FOMC "
        "economic_event importer"
    )
    print()

    #
    # Fail closed through the canonical validator
    # before touching PostgreSQL.
    #
    if not args.skip_validator:
        print(
            "Running canonical "
            "FOMC validator..."
        )
        print()

        run_validator()

        print()
        print(
            "Canonical validator: PASS"
        )
        print()

    source_rows, prepared, errors = (
        load_and_prepare(
            input_path
        )
    )

    print(
        f"Input          : "
        f"{input_path}"
    )

    print(
        f"Source rows    : "
        f"{len(source_rows)}"
    )

    print(
        f"Prepared rows  : "
        f"{len(prepared)}"
    )

    print(
        f"Prepare errors : "
        f"{len(errors)}"
    )

    print(
        "Mode           : "
        + (
            "COMMIT"
            if args.commit
            else "DRY RUN / ROLLBACK"
        )
    )

    print()

    if errors:
        for (
            line_number,
            message,
            row,
        ) in errors:
            print(
                f"ERROR source row "
                f"{line_number}: "
                f"{message}",
                file=sys.stderr,
            )

        print(
            "RESULT: ABORTED before "
            "database transaction",
            file=sys.stderr,
        )

        return 1

    if len(source_rows) != EXPECTED_ROWS:
        print(
            f"ERROR: expected "
            f"{EXPECTED_ROWS} source rows; "
            f"found {len(source_rows)}",
            file=sys.stderr,
        )
        return 1

    if (
        len(prepared)
        != len(source_rows)
    ):
        print(
            "ERROR: prepared/source "
            "row-count mismatch",
            file=sys.stderr,
        )
        return 1

    try:
        validate_prepared(
            prepared
        )
    except Exception as exc:
        print(
            "ERROR: prepared-data "
            f"validation failed: {exc}",
            file=sys.stderr,
        )
        return 1

    write_prepared_report(
        report_path,
        prepared,
    )

    family_counts = Counter(
        row["event_family"]
        for row in prepared
    )

    parser_counts = Counter(
        row["timestamp_parser"]
        for row in prepared
    )

    print(
        f"FOMC prepared  : "
        f"{family_counts['FOMC']}"
    )

    print(
        f"Prepared report: "
        f"{report_path}"
    )

    print()
    print(
        "Timestamp provenance:"
    )

    for parser_name in sorted(
        parser_counts
    ):
        print(
            f"  {parser_name:<34} "
            f"{parser_counts[parser_name]:>4}"
        )

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
        print(
            result.stdout,
            end="",
        )

    if result.stderr:
        print(
            result.stderr,
            end="",
            file=sys.stderr,
        )

    if result.returncode != 0:
        print()
        print(
            "RESULT: FAILED - PostgreSQL "
            "transaction did not complete."
        )

        print(
            "No partial FOMC import can survive "
            "because the operation uses one "
            "transaction."
        )

        return result.returncode

    print()

    if args.commit:
        print(
            "RESULT: COMMITTED - "
            "FOMC economic_event import "
            "passed all transaction invariants."
        )
    else:
        print(
            "RESULT: CLEAN DRY RUN - "
            "identical INSERT path passed; "
            "transaction was ROLLED BACK."
        )

        print(
            "No database changes were retained."
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
