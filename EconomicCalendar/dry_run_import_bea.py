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

ROOT = Path("EconomicCalendar/raw/bea")
DEFAULT_INPUT = ROOT / "bea_canonical_events.csv"
DEFAULT_REPORT = ROOT / "bea_dry_run_import_report.csv"
DEFAULT_SUMMARY = ROOT / "bea_dry_run_import_summary.txt"

KEY_FIELDS = (
    "source_agency",
    "event_family",
    "event_timestamp_utc",
)

COMPARE_CANDIDATES = (
    "source_local_date",
    "source_local_time",
    "source_timezone",
    "title",
    "url",
)

START_DATE = date(2010, 1, 1)
END_DATE = date(2026, 8, 24)


def ident(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


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

    cmd += ["-d", args.db, "-c", sql]

    env = os.environ.copy()

    # Defense in depth: force PostgreSQL transactions to be read-only.
    existing = env.get("PGOPTIONS", "")
    ro = "-c default_transaction_read_only=on"
    env["PGOPTIONS"] = f"{existing} {ro}".strip()

    result = subprocess.run(
        cmd,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
    )

    if result.returncode != 0:
        print("psql failed:", file=sys.stderr)
        print(result.stderr, file=sys.stderr)
        raise SystemExit(result.returncode)

    return result.stdout


def get_db_columns(args):
    sql = f"""
BEGIN READ ONLY;

SELECT column_name
FROM information_schema.columns
WHERE table_schema = {sql_literal(args.schema)}
  AND table_name = 'economic_event'
ORDER BY ordinal_position;

COMMIT;
"""

    output = run_psql(args, sql)

    # BEGIN / COMMIT don't appear under -qAt; only SELECT rows do.
    return {
        line.strip()
        for line in output.splitlines()
        if line.strip()
    }


def sql_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def normalize_timestamp(value: str) -> str:
    dt = datetime.fromisoformat(value)

    if dt.tzinfo is None:
        raise ValueError("timezone-naive timestamp")

    return dt.isoformat()


def normalize_date(value: str) -> str:
    return datetime.strptime(value, "%Y-%m-%d").date().isoformat()


def normalize_time(value: str) -> str:
    return datetime.strptime(value, "%H:%M:%S").time().isoformat()


def normalize_field(field: str, value):
    if value is None:
        return ""

    value = str(value).strip()

    if field == "event_timestamp_utc":
        return normalize_timestamp(value)

    if field == "source_local_date":
        return normalize_date(value)

    if field == "source_local_time":
        return normalize_time(value)

    return value


def validate_source_row(row, source_key_counts):
    problems = []

    for field in KEY_FIELDS:
        if not row.get(field, "").strip():
            problems.append(f"missing {field}")

    if row.get("source_agency") != "BEA":
        problems.append(
            f"source_agency={row.get('source_agency')!r}, expected 'BEA'"
        )

    if row.get("event_family") not in {"GDP", "PCE"}:
        problems.append(
            f"unexpected event_family={row.get('event_family')!r}"
        )

    try:
        ts = normalize_timestamp(row.get("event_timestamp_utc", ""))
    except Exception as exc:
        problems.append(f"invalid event_timestamp_utc: {exc}")
        ts = None

    try:
        local_date = datetime.strptime(
            row.get("source_local_date", ""),
            "%Y-%m-%d",
        ).date()

        if not START_DATE <= local_date <= END_DATE:
            problems.append(
                f"source_local_date {local_date} outside "
                f"{START_DATE}..{END_DATE}"
            )
    except Exception as exc:
        problems.append(f"invalid source_local_date: {exc}")

    try:
        normalize_time(row.get("source_local_time", ""))
    except Exception as exc:
        problems.append(f"invalid source_local_time: {exc}")

    if row.get("source_timezone") != "America/New_York":
        problems.append(
            "source_timezone must be America/New_York"
        )

    if not row.get("title", "").strip():
        problems.append("empty title")

    if not row.get("url", "").strip():
        problems.append("empty url")

    if ts is not None:
        key = (
            row.get("source_agency", ""),
            row.get("event_family", ""),
            ts,
        )

        if source_key_counts[key] > 1:
            problems.append(
                "duplicate canonical key inside source CSV"
            )

    return problems


def load_source(path: Path):
    with path.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    return rows


def fetch_existing_db_rows(args, db_columns):
    selected = [
        field
        for field in KEY_FIELDS + COMPARE_CANDIDATES
        if field in db_columns
    ]

    missing_keys = [
        field
        for field in KEY_FIELDS
        if field not in db_columns
    ]

    if missing_keys:
        raise SystemExit(
            "ERROR: economic_event is missing required key column(s): "
            + ", ".join(missing_keys)
        )

    cols = ", ".join(ident(c) for c in selected)
    table = f"{ident(args.schema)}.{ident('economic_event')}"

    sql = f"""
BEGIN READ ONLY;

SELECT row_to_json(x)::text
FROM (
    SELECT {cols}
    FROM {table}
    WHERE {ident('source_agency')} = 'BEA'
      AND {ident('event_family')} IN ('GDP', 'PCE')
) AS x;

COMMIT;
"""

    output = run_psql(args, sql)

    rows = []

    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        rows.append(json.loads(line))

    return selected, rows


def canonical_key(row):
    return (
        normalize_field(
            "source_agency",
            row.get("source_agency", ""),
        ),
        normalize_field(
            "event_family",
            row.get("event_family", ""),
        ),
        normalize_field(
            "event_timestamp_utc",
            row.get("event_timestamp_utc", ""),
        ),
    )


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Read-only dry-run comparison of canonical BEA events "
            "against economic_event."
        )
    )

    parser.add_argument(
        "--input",
        default=str(DEFAULT_INPUT),
    )
    parser.add_argument(
        "--report",
        default=str(DEFAULT_REPORT),
    )
    parser.add_argument(
        "--summary",
        default=str(DEFAULT_SUMMARY),
    )

    parser.add_argument("--db", default="LSTM")
    parser.add_argument("--schema", default="public")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--user")
    parser.add_argument("--psql", default="psql")

    args = parser.parse_args()

    input_path = Path(args.input)
    report_path = Path(args.report)
    summary_path = Path(args.summary)

    source_rows = load_source(input_path)

    #
    # Detect canonical duplicates in source itself before DB comparison.
    #
    source_key_counts = Counter()

    for row in source_rows:
        try:
            key = canonical_key(row)
            source_key_counts[key] += 1
        except Exception:
            pass

    print("BEA economic_event dry-run import")
    print()
    print(f"Input             : {input_path}")
    print(f"Canonical rows    : {len(source_rows)}")
    print(f"Database          : {args.db}")
    print(f"Schema            : {args.schema}")
    print()
    print("Inspecting economic_event schema (read-only)...")

    db_columns = get_db_columns(args)

    if not db_columns:
        raise SystemExit(
            f"ERROR: {args.schema}.economic_event was not found "
            "or has no visible columns"
        )

    selected_db_fields, db_rows = fetch_existing_db_rows(
        args,
        db_columns,
    )

    compare_fields = [
        field
        for field in COMPARE_CANDIDATES
        if field in db_columns
    ]

    print(
        "Comparable DB fields: "
        + (
            ", ".join(compare_fields)
            if compare_fields
            else "(key fields only)"
        )
    )
    print(f"Existing BEA DB rows: {len(db_rows)}")
    print()

    db_by_key = defaultdict(list)

    for row in db_rows:
        try:
            db_by_key[canonical_key(row)].append(row)
        except Exception as exc:
            raise SystemExit(
                f"ERROR: existing DB row has invalid canonical key: {exc}"
            )

    report_rows = []

    for source_row_number, row in enumerate(
        source_rows,
        start=2,  # header is CSV row 1
    ):
        problems = validate_source_row(
            row,
            source_key_counts,
        )

        result = {
            "source_row": source_row_number,
            "classification": "",
            "difference_fields": "",
            "reason": "",
            "existing_db_matches": 0,
            "source_agency": row.get("source_agency", ""),
            "event_family": row.get("event_family", ""),
            "event_timestamp_utc": row.get(
                "event_timestamp_utc",
                "",
            ),
            "source_local_date": row.get(
                "source_local_date",
                "",
            ),
            "source_local_time": row.get(
                "source_local_time",
                "",
            ),
            "source_timezone": row.get(
                "source_timezone",
                "",
            ),
            "title": row.get("title", ""),
            "url": row.get("url", ""),
        }

        if problems:
            result["classification"] = "invalid"
            result["reason"] = "; ".join(problems)
            report_rows.append(result)
            continue

        key = canonical_key(row)
        existing = db_by_key.get(key, [])

        result["existing_db_matches"] = len(existing)

        if not existing:
            result["classification"] = "would_insert"
            result["reason"] = "canonical key not present"
            report_rows.append(result)
            continue

        if len(existing) > 1:
            result["classification"] = "conflict"
            result["reason"] = (
                "multiple existing DB rows share canonical key"
            )
            report_rows.append(result)
            continue

        db_row = existing[0]

        differences = []

        for field in compare_fields:
            try:
                source_value = normalize_field(
                    field,
                    row.get(field, ""),
                )
                db_value = normalize_field(
                    field,
                    db_row.get(field, ""),
                )
            except Exception as exc:
                differences.append(
                    f"{field}(normalization_error:{exc})"
                )
                continue

            if source_value != db_value:
                differences.append(field)

        if differences:
            result["classification"] = "conflict"
            result["difference_fields"] = ",".join(
                differences
            )
            result["reason"] = (
                "canonical key exists but comparable field(s) differ"
            )
        else:
            result["classification"] = "already_present"
            result["reason"] = (
                "canonical key exists and comparable fields match"
            )

        report_rows.append(result)

    fields = [
        "source_row",
        "classification",
        "difference_fields",
        "reason",
        "existing_db_matches",
        "source_agency",
        "event_family",
        "event_timestamp_utc",
        "source_local_date",
        "source_local_time",
        "source_timezone",
        "title",
        "url",
    ]

    with report_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields,
        )
        writer.writeheader()
        writer.writerows(report_rows)

    counts = Counter(
        r["classification"]
        for r in report_rows
    )

    summary_lines = [
        "BEA economic_event dry-run import",
        "",
        f"Input rows       : {len(source_rows)}",
        f"Existing BEA rows: {len(db_rows)}",
        "",
        f"would_insert     : {counts['would_insert']}",
        f"already_present  : {counts['already_present']}",
        f"conflict         : {counts['conflict']}",
        f"invalid          : {counts['invalid']}",
        "",
        "DB fields compared:",
        "  " + (
            ", ".join(compare_fields)
            if compare_fields
            else "(canonical key only)"
        ),
        "",
        "NO DATABASE WRITES WERE PERFORMED.",
    ]

    summary_path.write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )

    print("\n".join(summary_lines))
    print()
    print(f"Report            : {report_path}")
    print(f"Summary           : {summary_path}")

    if counts["conflict"] or counts["invalid"]:
        print()
        print(
            "RESULT: REVIEW REQUIRED - "
            "do not perform actual import"
        )
        return 1

    print()
    print(
        "RESULT: CLEAN DRY RUN - "
        "no conflicts or invalid rows"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
