#!/usr/bin/env python3
"""Read-only deterministic audit for economic-event actual provenance."""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import io
import json
import os
import subprocess
from typing import Iterable


def psql_rows(database: str, query: str) -> list[dict[str, str]]:
    environment = os.environ.copy()
    current = environment.get("PGOPTIONS", "")
    environment["PGOPTIONS"] = (
        current + " -c default_transaction_read_only=on"
    ).strip()
    command = [
        "psql", "-X", "--no-psqlrc", "--set", "ON_ERROR_STOP=1",
        "--dbname", database, "--command",
        "COPY (" + query.strip().rstrip(";")
        + ") TO STDOUT WITH (FORMAT csv, HEADER true)",
    ]
    result = subprocess.run(
        command,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "economic_event_actual_provenance_audit_failed:"
            + " ".join(result.stderr.split())
        )
    return list(csv.DictReader(io.StringIO(result.stdout)))


def nullable(value: str) -> object:
    if value == "":
        return None
    if value in {"t", "f"}:
        return value == "t"
    return value


def normalized(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for row in rows:
        converted = {key: nullable(value) for key, value in row.items()}
        for key in ("source_provenance",):
            if isinstance(converted.get(key), str):
                converted[key] = json.loads(str(converted[key]))
        result.append(converted)
    return result


def sql_timestamp(value: str) -> str:
    parsed = dt.datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("cutoff_requires_timezone")
    canonical = parsed.astimezone(dt.timezone.utc).isoformat(
        timespec="microseconds"
    )
    return "'" + canonical.replace("'", "''") + "'::timestamptz"


def aggregate_report(database: str) -> dict[str, object]:
    summary = normalized(psql_rows(database, """
        SELECT
            count(*) AS total_events,
            count(*) FILTER (WHERE observation_count > 0)
                AS total_events_with_actuals,
            count(*) FILTER (WHERE observation_count = 0)
                AS events_with_no_actual_observation,
            count(*) FILTER (
                WHERE provenance_state = 'proven_first_release')
                AS proven_first_release,
            count(*) FILTER (
                WHERE provenance_state <> 'proven_first_release')
                AS events_without_usable_first_release_provenance,
            count(*) FILTER (WHERE provenance_state = 'ambiguous')
                AS ambiguous,
            count(*) FILTER (
                WHERE observation_count > 0 AND
                      provenance_state = 'provenance_unavailable')
                AS provenance_unavailable,
            count(*) FILTER (WHERE observation_count > 1)
                AS events_with_multiple_actual_observations,
            count(*) FILTER (
                WHERE canonical_differs_from_first_release IS TRUE)
                AS canonical_differs_from_first_release
        FROM economic_event_first_release_actual
    """))[0]
    by_source = normalized(psql_rows(database, """
        SELECT
            source_name,
            source_role,
            observation_kind,
            count(*) AS observations,
            count(DISTINCT economic_event_id) AS events,
            count(*) FILTER (
                WHERE source_publication_time_status = 'exact')
                AS exact_source_publication,
            count(*) FILTER (
                WHERE source_publication_time_status = 'unavailable')
                AS source_publication_unavailable
        FROM economic_event_actual_observation
        GROUP BY source_name, source_role, observation_kind
        ORDER BY source_name, source_role, observation_kind
    """))
    by_family = normalized(psql_rows(database, """
        SELECT
            event_family,
            count(*) FILTER (WHERE observation_count > 0)
                AS events_with_actuals,
            count(*) FILTER (WHERE observation_count = 0)
                AS events_with_no_actual_observation,
            count(*) FILTER (
                WHERE provenance_state = 'proven_first_release')
                AS proven_first_release,
            count(*) FILTER (WHERE provenance_state = 'ambiguous')
                AS ambiguous,
            count(*) FILTER (
                WHERE observation_count > 0 AND
                      provenance_state = 'provenance_unavailable')
                AS provenance_unavailable,
            count(*) FILTER (WHERE observation_count > 1)
                AS multiple_observations,
            count(*) FILTER (
                WHERE canonical_differs_from_first_release IS TRUE)
                AS canonical_differs_from_first_release
        FROM economic_event_first_release_actual
        GROUP BY event_family
        ORDER BY event_family
    """))
    return {
        "contract": "economic_event_actual_provenance_audit_v1",
        "database_interaction": "read_only",
        "summary": summary,
        "by_source": by_source,
        "by_event_family": by_family,
    }


def event_report(
    database: str, event_id: int, cutoff: str | None
) -> dict[str, object]:
    assessment = normalized(psql_rows(database, f"""
        SELECT *
        FROM economic_event_first_release_actual
        WHERE economic_event_id = {event_id}
    """))
    observations = normalized(psql_rows(database, f"""
        SELECT
            economic_event_actual_observation_id,
            economic_event_id,
            source_name,
            source_role,
            source_native_event_id,
            source_observation_id,
            evidence_key,
            observation_kind,
            revision_sequence,
            source_publication_at,
            source_publication_time_status,
            observed_at,
            ingested_at,
            availability_proof,
            proven_available_at,
            source_url,
            source_artifact_path,
            source_artifact_sha256,
            semantic_contract,
            source_provenance,
            actual_raw,
            actual_value_kind,
            actual_value_low,
            actual_value_high,
            actual_canonical_value_low,
            actual_canonical_value_high,
            actual_unit,
            actual_scale,
            actual_qualifier
        FROM economic_event_actual_observation
        WHERE economic_event_id = {event_id}
        ORDER BY proven_available_at,
                 economic_event_actual_observation_id
    """))
    result: dict[str, object] = {
        "contract": "economic_event_actual_provenance_event_audit_v1",
        "database_interaction": "read_only",
        "assessment": assessment[0] if assessment else None,
        "observations": observations,
    }
    if cutoff is not None:
        instant = sql_timestamp(cutoff)
        result["pit_cutoff"] = cutoff
        result["pit_visible_first_release"] = normalized(psql_rows(
            database,
            "SELECT * FROM economic_event_first_release_actual_at("
            + instant + f") WHERE economic_event_id = {event_id}",
        ))
    return result


def main(argv: Iterable[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Read-only actual provenance coverage/event audit"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--event-id", type=int)
    parser.add_argument("--cutoff")
    args = parser.parse_args(list(argv) if argv is not None else None)
    if args.cutoff and args.event_id is None:
        parser.error("--cutoff requires --event-id")
    report = (
        event_report(args.db, args.event_id, args.cutoff)
        if args.event_id is not None
        else aggregate_report(args.db)
    )
    print(json.dumps(report, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(str(error), file=os.sys.stderr)
        raise SystemExit(1)
