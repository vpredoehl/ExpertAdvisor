#!/usr/bin/env python3

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import date, datetime, time, timezone
from pathlib import Path
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo


DEFAULT_INPUT = Path(
    "EconomicCalendar/raw/census/census_events_extracted.csv"
)
DEFAULT_PREPARED_REPORT = Path(
    "EconomicCalendar/raw/census/census_import_prepared.csv"
)

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)

EXPECTED_ROWS = 399
EXPECTED_FAMILY_COUNTS = {
    "RETAIL_SALES": 200,
    "DURABLE_GOODS": 199,
}
EXPECTED_PARSER_COUNTS = {
    "census_for_release_at": 229,
    "census_for_immediate_release": 164,
    "census_for_immediate_release_malformed_text_repair": 5,
    "census_durable_release_schedule": 1,
}
EXPECTED_DISCOVERY_COUNTS = {
    "historical_archive": 397,
    "current_release_page_self": 1,
    "current_release_pdf": 1,
}

EXPECTED_AGENCY = "CENSUS"
EXPECTED_CURRENCY = "USD"
EXPECTED_IMPORTANCE = 3
EXPECTED_CONFIDENCE = "exact"
EXPECTED_TIMEZONE = "America/New_York"
EXPECTED_LOCAL_TIME = time(8, 30)

TARGET_FIELDS = (
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
)

# economic_event intentionally has no title or timestamp-provenance URL
# columns. Preserve those source facts in the deterministic prepared report,
# matching the established BEA/FOMC importer convention.
PREPARED_REPORT_FIELDS = TARGET_FIELDS + (
    "title",
    "timestamp_parser",
    "timestamp_source_url",
    "discovery_source",
    "release_timezone_token",
    "filename",
)

REQUIRED_INPUT_FIELDS = {
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


@dataclass(frozen=True)
class DatabaseConfig:
    database: str
    host: str | None = None
    port: int | None = None
    user: str | None = None
    psql: str = "psql"


@dataclass(frozen=True)
class ImportEffects:
    would_insert: int
    unchanged: int
    would_update: int
    rejected: int
    diagnostics: tuple[str, ...] = ()


def canonical_timestamp(value: str) -> tuple[datetime, str]:
    if not value:
        raise ValueError("missing event_timestamp_utc")

    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError(f"invalid event_timestamp_utc: {value!r}") from exc

    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise ValueError("timezone-naive event_timestamp_utc")

    utc = parsed.astimezone(timezone.utc)
    return utc, utc.isoformat()


def authoritative_census_url(value: str, field: str) -> str:
    if not value:
        raise ValueError(f"missing {field}")

    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()

    if (
        parsed.scheme != "https"
        or not parsed.path
        or not (
            hostname == "census.gov"
            or hostname.endswith(".census.gov")
        )
    ):
        raise ValueError(
            f"{field} is not an authoritative Census HTTPS URL: {value!r}"
        )

    return value


def source_event_id(
    family: str,
    reference_year: int,
    reference_month: int,
) -> str:
    if family not in EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"unexpected event_family {family!r}")
    if not 1 <= reference_month <= 12:
        raise ValueError(f"invalid reference_month {reference_month!r}")

    family_identity = family.lower().replace("_", "-")
    return (
        f"census:{family_identity}-"
        f"{reference_year:04d}-{reference_month:02d}"
    )


def canonical_key(row: dict) -> tuple[str, str, str]:
    return (
        row["source_agency"],
        row["event_family"],
        row["event_timestamp_utc"],
    )


def _required(row: dict, field: str) -> str:
    value = (row.get(field) or "").strip()
    if not value:
        raise ValueError(f"missing {field}")
    return value


def prepare_row(row: dict) -> dict:
    agency = _required(row, "source_agency")
    family = _required(row, "event_family")

    if agency != EXPECTED_AGENCY:
        raise ValueError(f"unexpected source_agency {agency!r}")
    if family not in EXPECTED_FAMILY_COUNTS:
        raise ValueError(f"unexpected event_family {family!r}")

    utc, timestamp = canonical_timestamp(
        _required(row, "event_timestamp_utc")
    )

    local_date_raw = _required(row, "source_local_date")
    local_time_raw = _required(row, "source_local_time")
    source_timezone = _required(row, "source_timezone")

    try:
        local_date = date.fromisoformat(local_date_raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid source_local_date {local_date_raw!r}"
        ) from exc

    try:
        local_time = time.fromisoformat(local_time_raw)
    except ValueError as exc:
        raise ValueError(
            f"invalid source_local_time {local_time_raw!r}"
        ) from exc

    if source_timezone != EXPECTED_TIMEZONE:
        raise ValueError(f"unexpected source_timezone {source_timezone!r}")
    if not START_DATE <= local_date <= END_DATE:
        raise ValueError(
            f"source release date {local_date} outside "
            f"{START_DATE}..{END_DATE}"
        )
    if local_time != EXPECTED_LOCAL_TIME:
        raise ValueError(
            f"unexpected source local release time {local_time_raw!r}"
        )

    local = datetime.combine(
        local_date,
        local_time,
        ZoneInfo(EXPECTED_TIMEZONE),
    )
    if local.astimezone(timezone.utc) != utc:
        raise ValueError(
            "event_timestamp_utc does not match the authoritative "
            "America/New_York release date/time"
        )

    timezone_token = _required(row, "release_timezone_token")
    if timezone_token not in {"EST", "EDT"}:
        raise ValueError(
            f"unexpected release_timezone_token {timezone_token!r}"
        )
    if local.tzname() != timezone_token:
        raise ValueError(
            f"release_timezone_token {timezone_token!r} contradicts "
            f"America/New_York ({local.tzname()!r})"
        )

    reference_period = _required(row, "reference_period")
    try:
        reference_year = int(_required(row, "reference_year"))
        reference_month = int(_required(row, "reference_month"))
        reference_from_text = datetime.strptime(
            reference_period,
            "%B %Y",
        )
    except ValueError as exc:
        raise ValueError(
            f"invalid Census reference period fields: {reference_period!r}"
        ) from exc

    if (
        reference_from_text.year != reference_year
        or reference_from_text.month != reference_month
    ):
        raise ValueError(
            "reference_period contradicts reference_year/reference_month"
        )

    title = _required(row, "title")
    primary_url = authoritative_census_url(_required(row, "url"), "url")
    final_url = authoritative_census_url(
        _required(row, "final_url"),
        "final_url",
    )
    timestamp_source_url = authoritative_census_url(
        _required(row, "timestamp_source_url"),
        "timestamp_source_url",
    )
    if final_url != primary_url:
        raise ValueError("final_url differs from canonical primary url")

    parser_name = _required(row, "timestamp_parser")
    discovery_source = _required(row, "discovery_source")
    if parser_name not in EXPECTED_PARSER_COUNTS:
        raise ValueError(f"unexpected timestamp_parser {parser_name!r}")
    if discovery_source not in EXPECTED_DISCOVERY_COUNTS:
        raise ValueError(
            f"unexpected discovery_source {discovery_source!r}"
        )

    filename = _required(row, "filename")
    _required(row, "release_text")

    if parser_name == "census_durable_release_schedule":
        expected_schedule_identity = (
            family == "DURABLE_GOODS"
            and reference_year == 2026
            and reference_month == 6
            and local_date == date(2026, 7, 27)
            and timestamp_source_url
            == "https://www.census.gov/manufacturing/m3/release_schedule.html"
        )
        if not expected_schedule_identity:
            raise ValueError(
                "unexpected schedule-derived timestamp provenance state"
            )
    elif timestamp_source_url != primary_url:
        raise ValueError(
            "non-schedule timestamp_source_url differs from primary url"
        )

    return {
        "currency": EXPECTED_CURRENCY,
        "event_family": family,
        "event_timestamp_utc": timestamp,
        "source_agency": EXPECTED_AGENCY,
        "source_event_id": source_event_id(
            family,
            reference_year,
            reference_month,
        ),
        "source_url": primary_url,
        "reference_period": reference_period,
        "event_importance": EXPECTED_IMPORTANCE,
        "historical_time_confidence": EXPECTED_CONFIDENCE,
        "source_release_date": local_date.isoformat(),
        "source_release_time": local_time.isoformat(),
        "source_timezone": EXPECTED_TIMEZONE,
        "title": title,
        "timestamp_parser": parser_name,
        "timestamp_source_url": timestamp_source_url,
        "discovery_source": discovery_source,
        "release_timezone_token": timezone_token,
        "filename": filename,
    }


def load_and_prepare(path: Path) -> tuple[list[dict], list[dict]]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields = set(reader.fieldnames or ())
        missing_fields = sorted(REQUIRED_INPUT_FIELDS - fields)
        if missing_fields:
            raise ValueError(
                "canonical CSV is missing required column(s): "
                + ", ".join(missing_fields)
            )
        source_rows = list(reader)

    if len(source_rows) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly {EXPECTED_ROWS} input rows, "
            f"found {len(source_rows)}"
        )

    prepared: list[dict] = []
    errors: list[str] = []
    for line_number, row in enumerate(source_rows, start=2):
        try:
            prepared.append(prepare_row(row))
        except Exception as exc:
            errors.append(f"line {line_number}: {exc}")

    if errors:
        raise ValueError(
            "Census canonical mapping failed:\n" + "\n".join(errors)
        )

    validate_prepared(prepared)
    return source_rows, prepared


def validate_prepared(prepared: list[dict]) -> None:
    if len(prepared) != EXPECTED_ROWS:
        raise ValueError(
            f"expected exactly {EXPECTED_ROWS} prepared rows, "
            f"found {len(prepared)}"
        )

    family_counts = Counter(row["event_family"] for row in prepared)
    parser_counts = Counter(row["timestamp_parser"] for row in prepared)
    discovery_counts = Counter(row["discovery_source"] for row in prepared)

    if family_counts != Counter(EXPECTED_FAMILY_COUNTS):
        raise ValueError(
            f"unexpected family counts: {dict(sorted(family_counts.items()))}"
        )
    if parser_counts != Counter(EXPECTED_PARSER_COUNTS):
        raise ValueError(
            f"unexpected timestamp parser counts: "
            f"{dict(sorted(parser_counts.items()))}"
        )
    if discovery_counts != Counter(EXPECTED_DISCOVERY_COUNTS):
        raise ValueError(
            f"unexpected discovery-source counts: "
            f"{dict(sorted(discovery_counts.items()))}"
        )

    keys = Counter(canonical_key(row) for row in prepared)
    source_ids = Counter(row["source_event_id"] for row in prepared)
    if any(count != 1 for count in keys.values()):
        raise ValueError("duplicate canonical economic_event key in input")
    if any(count != 1 for count in source_ids.values()):
        raise ValueError("duplicate Census source_event_id in input")
    if len(keys) != EXPECTED_ROWS or len(source_ids) != EXPECTED_ROWS:
        raise ValueError("canonical key/source_event_id count mismatch")

    for row in prepared:
        for field in TARGET_FIELDS:
            value = row.get(field)
            if value is None or (isinstance(value, str) and not value.strip()):
                raise ValueError(
                    f"required canonical target field {field!r} is empty"
                )


def write_prepared_report(path: Path, prepared: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=PREPARED_REPORT_FIELDS)
        writer.writeheader()
        writer.writerows(prepared)


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    return "'" + str(value).replace("'", "''") + "'"


def psql_command(config: DatabaseConfig) -> list[str]:
    command = [
        config.psql,
        "-X",
        "-q",
        "-A",
        "-t",
        "-v",
        "ON_ERROR_STOP=1",
        "-d",
        config.database,
    ]
    if config.host:
        command += ["-h", config.host]
    if config.port:
        command += ["-p", str(config.port)]
    if config.user:
        command += ["-U", config.user]
    return command


def run_psql(
    config: DatabaseConfig,
    sql: str,
    *,
    read_only: bool,
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    if read_only:
        existing = environment.get("PGOPTIONS", "")
        environment["PGOPTIONS"] = (
            f"{existing} -c default_transaction_read_only=on".strip()
        )

    return subprocess.run(
        psql_command(config),
        input=sql,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=environment,
    )


def build_dry_run_sql() -> str:
    return """
BEGIN READ ONLY;

SELECT row_to_json(existing)::text
FROM (
    SELECT
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
    WHERE source_agency = 'CENSUS'
    ORDER BY economic_event_id
) AS existing;

COMMIT;
"""


def load_database_rows(
    config: DatabaseConfig,
) -> list[dict]:
    result = run_psql(
        config,
        build_dry_run_sql(),
        read_only=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "psql dry run failed")

    rows = []
    for line in result.stdout.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _normalized_database_row(row: dict) -> dict:
    normalized = dict(row)
    _, normalized["event_timestamp_utc"] = canonical_timestamp(
        str(row["event_timestamp_utc"])
    )
    normalized["event_importance"] = int(row["event_importance"])
    for field in (
        "source_release_date",
        "source_release_time",
    ):
        if normalized.get(field) is not None:
            normalized[field] = str(normalized[field])
    return normalized


def _target_equal(existing: dict, prepared: dict) -> bool:
    normalized = _normalized_database_row(existing)
    return all(normalized.get(field) == prepared[field] for field in TARGET_FIELDS)


def classify_database_effects(
    prepared: list[dict],
    existing_rows: list[dict],
) -> ImportEffects:
    by_id: defaultdict[str, list[dict]] = defaultdict(list)
    by_key: defaultdict[tuple[str, str, str], list[dict]] = defaultdict(list)

    for existing in existing_rows:
        normalized = _normalized_database_row(existing)
        source_id = normalized.get("source_event_id")
        if source_id:
            by_id[source_id].append(existing)
        by_key[canonical_key(normalized)].append(existing)

    would_insert = 0
    unchanged = 0
    rejected = 0
    diagnostics: list[str] = []

    for row in prepared:
        identity_matches = by_id.get(row["source_event_id"], [])
        key_matches = by_key.get(canonical_key(row), [])

        if not identity_matches and not key_matches:
            would_insert += 1
            continue

        exact = [
            existing
            for existing in identity_matches
            if _target_equal(existing, row)
        ]
        if (
            len(exact) == 1
            and len(identity_matches) == 1
            and len(key_matches) == 1
            and key_matches[0] is exact[0]
        ):
            unchanged += 1
            continue

        rejected += 1
        diagnostics.append(
            f"{row['source_event_id']}: "
            "authoritative_identity_or_timestamp_collision"
        )

    return ImportEffects(
        would_insert=would_insert,
        unchanged=unchanged,
        would_update=0,
        rejected=rejected,
        diagnostics=tuple(diagnostics),
    )


def _values_sql(prepared: list[dict]) -> str:
    rows = []
    for row in prepared:
        rows.append(
            "("
            + ", ".join(
                sql_literal(row[field])
                if field != "event_importance"
                else str(row[field])
                for field in TARGET_FIELDS
            )
            + ")"
        )
    return ",\n".join(rows)


def build_commit_sql(prepared: list[dict]) -> str:
    values = _values_sql(prepared)
    target_column_list = ",\n        ".join(TARGET_FIELDS)
    selected_columns = ",\n        ".join(f"s.{field}" for field in TARGET_FIELDS)
    equality = "\n        AND ".join(
        f"e.{field} IS NOT DISTINCT FROM s.{field}"
        for field in TARGET_FIELDS
    )

    return f"""
BEGIN;
SET TRANSACTION ISOLATION LEVEL SERIALIZABLE;

CREATE TEMP TABLE census_import_source (
    currency text NOT NULL,
    event_family text NOT NULL,
    event_timestamp_utc timestamptz NOT NULL,
    source_agency text NOT NULL,
    source_event_id text NOT NULL,
    source_url text NOT NULL,
    reference_period text NOT NULL,
    event_importance smallint NOT NULL,
    historical_time_confidence text NOT NULL,
    source_release_date date NOT NULL,
    source_release_time time NOT NULL,
    source_timezone text NOT NULL
) ON COMMIT DROP;

INSERT INTO census_import_source (
        {target_column_list}
)
VALUES
{values};

DO $$
DECLARE
    staged_rows bigint;
    staged_keys bigint;
    staged_ids bigint;
BEGIN
    SELECT count(*) INTO staged_rows FROM census_import_source;
    SELECT count(*) INTO staged_keys FROM (
        SELECT DISTINCT source_agency, event_family, event_timestamp_utc
        FROM census_import_source
    ) AS keys;
    SELECT count(*) INTO staged_ids FROM (
        SELECT DISTINCT source_agency, source_event_id
        FROM census_import_source
    ) AS ids;

    IF staged_rows <> {EXPECTED_ROWS}
       OR staged_keys <> {EXPECTED_ROWS}
       OR staged_ids <> {EXPECTED_ROWS} THEN
        RAISE EXCEPTION
            'Census staging invariant failed: rows %, keys %, ids %',
            staged_rows, staged_keys, staged_ids;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM census_import_source s
        JOIN economic_event e
          ON e.source_agency = s.source_agency
         AND (
              e.source_event_id = s.source_event_id
              OR (
                   e.event_family = s.event_family
                   AND e.event_timestamp_utc = s.event_timestamp_utc
              )
         )
        WHERE NOT (
            {equality}
        )
    ) THEN
        RAISE EXCEPTION
            'Census authoritative identity or canonical timestamp conflict';
    END IF;
END
$$;

CREATE TEMP TABLE census_import_metrics (
    metric text PRIMARY KEY,
    value bigint NOT NULL
) ON COMMIT DROP;

INSERT INTO census_import_metrics
SELECT 'pre_census_family_count', count(*)
FROM economic_event
WHERE source_agency = 'CENSUS'
  AND event_family IN ('RETAIL_SALES', 'DURABLE_GOODS');

INSERT INTO census_import_metrics
SELECT 'pre_matching_exact', count(*)
FROM census_import_source s
JOIN economic_event e
  ON {equality};

CREATE TEMP TABLE census_inserted_ids (
    economic_event_id bigint PRIMARY KEY
) ON COMMIT DROP;

WITH inserted AS (
    INSERT INTO economic_event (
        {target_column_list}
    )
    SELECT
        {selected_columns}
    FROM census_import_source s
    ORDER BY s.event_timestamp_utc, s.event_family
    ON CONFLICT DO NOTHING
    RETURNING economic_event_id
)
INSERT INTO census_inserted_ids
SELECT economic_event_id FROM inserted;

INSERT INTO census_import_metrics
SELECT 'inserted', count(*) FROM census_inserted_ids;

INSERT INTO census_import_metrics
SELECT 'post_matching_exact', count(*)
FROM census_import_source s
JOIN economic_event e
  ON {equality};

INSERT INTO census_import_metrics
SELECT 'post_census_family_count', count(*)
FROM economic_event
WHERE source_agency = 'CENSUS'
  AND event_family IN ('RETAIL_SALES', 'DURABLE_GOODS');

DO $$
DECLARE
    pre_count bigint;
    pre_match bigint;
    inserted_count bigint;
    post_match bigint;
    post_count bigint;
BEGIN
    SELECT value INTO pre_count FROM census_import_metrics
    WHERE metric = 'pre_census_family_count';
    SELECT value INTO pre_match FROM census_import_metrics
    WHERE metric = 'pre_matching_exact';
    SELECT value INTO inserted_count FROM census_import_metrics
    WHERE metric = 'inserted';
    SELECT value INTO post_match FROM census_import_metrics
    WHERE metric = 'post_matching_exact';
    SELECT value INTO post_count FROM census_import_metrics
    WHERE metric = 'post_census_family_count';

    IF inserted_count <> {EXPECTED_ROWS} - pre_match THEN
        RAISE EXCEPTION
            'Census insert count mismatch: pre-match %, inserted %',
            pre_match, inserted_count;
    END IF;
    IF post_match <> {EXPECTED_ROWS} THEN
        RAISE EXCEPTION
            'Census post-import exact coverage mismatch: found %',
            post_match;
    END IF;
    IF post_count <> pre_count + inserted_count THEN
        RAISE EXCEPTION
            'Census row-count delta mismatch: pre %, inserted %, post %',
            pre_count, inserted_count, post_count;
    END IF;
END
$$;

SELECT metric || '=' || value
FROM census_import_metrics
ORDER BY CASE metric
    WHEN 'pre_census_family_count' THEN 1
    WHEN 'pre_matching_exact' THEN 2
    WHEN 'inserted' THEN 3
    WHEN 'post_matching_exact' THEN 4
    WHEN 'post_census_family_count' THEN 5
    ELSE 99
END;

COMMIT;
"""


def print_mapping_summary(prepared: list[dict]) -> None:
    families = Counter(row["event_family"] for row in prepared)
    print(f"Prepared rows       : {len(prepared)}")
    print(f"RETAIL_SALES        : {families['RETAIL_SALES']}")
    print(f"DURABLE_GOODS       : {families['DURABLE_GOODS']}")
    print(f"Canonical keys      : {len({canonical_key(row) for row in prepared})}")
    print(f"Source event IDs    : {len({row['source_event_id'] for row in prepared})}")


def parse_arguments(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Import the validated 399-row Census canonical CSV into "
            "economic_event. Dry run is PostgreSQL read-only; --commit is "
            "the only write mode."
        )
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate, map, and compare using a read-only transaction.",
    )
    mode.add_argument(
        "--commit",
        action="store_true",
        help="Import atomically in one explicit PostgreSQL transaction.",
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument(
        "--prepared-report",
        type=Path,
        default=DEFAULT_PREPARED_REPORT,
    )
    parser.add_argument("--db", default="LSTM")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--user")
    parser.add_argument("--psql", default="psql")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_arguments(argv)
    config = DatabaseConfig(
        database=args.db,
        host=args.host,
        port=args.port,
        user=args.user,
        psql=args.psql,
    )

    try:
        source_rows, prepared = load_and_prepare(args.input)
        write_prepared_report(args.prepared_report, prepared)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        print("RESULT: ABORTED before database access", file=sys.stderr)
        return 1

    print("Census economic_event importer")
    print(f"Input               : {args.input}")
    print(f"Input rows          : {len(source_rows)}")
    print_mapping_summary(prepared)
    print(f"Prepared report     : {args.prepared_report}")

    if args.dry_run:
        print("Mode                : DRY RUN / READ ONLY")
        try:
            existing = load_database_rows(config)
            effects = classify_database_effects(prepared, existing)
        except Exception as exc:
            print(f"ERROR: {exc}", file=sys.stderr)
            print("RESULT: FAILED during read-only database comparison")
            return 1

        print(f"Would insert        : {effects.would_insert}")
        print(f"Would update        : {effects.would_update}")
        print(f"Unchanged           : {effects.unchanged}")
        print(f"Rejected conflicts  : {effects.rejected}")
        for diagnostic in effects.diagnostics:
            print(f"CONFLICT: {diagnostic}", file=sys.stderr)

        if effects.rejected:
            print("RESULT: FAIL - immutable database conflicts detected")
            print("DATABASE WRITES: 0 (read-only transaction enforced)")
            return 2

        print("RESULT: PASS - Census dry run completed")
        print("DATABASE WRITES: 0 (read-only transaction enforced)")
        return 0

    print("Mode                : EXPLICIT TRANSACTIONAL COMMIT")
    result = run_psql(
        config,
        build_commit_sql(prepared),
        read_only=False,
    )
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        print("RESULT: FAILED - transaction rolled back")
        return result.returncode

    print("RESULT: COMMITTED - all 399 Census rows verified atomically")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
