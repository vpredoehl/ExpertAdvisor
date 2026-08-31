#!/usr/bin/env python3
"""Dry-run, audit, and disposable-database import for Phase 10 actuals."""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import pathlib
import subprocess
import sys
from typing import Iterable, Mapping, Sequence

from release_actual_ingestion import (
    BeaArtifact,
    BlsArtifact,
    CensusArtifact,
    Consensus,
    EconomicEvent,
    ImportDecision,
    Rejection,
    ReleaseActualCandidate,
    canonical_instant,
    coverage_audit,
    deterministic_json_lines,
    extract_bea_candidates,
    extract_bls_candidates,
    extract_census_candidates,
    load_bea_artifacts,
    load_bls_artifacts,
    load_census_artifacts,
    match_candidates,
)


EVENT_SQL = """
SELECT economic_event_id, source_agency, event_family,
       event_timestamp_utc, source_event_id, source_url, reference_period
FROM economic_event
ORDER BY economic_event_id
"""

CONSENSUS_SQL = """
SELECT economic_event_id, consensus_value_kind AS value_kind,
       consensus_unit AS unit, consensus_qualifier AS qualifier
FROM economic_event_selected_consensus
ORDER BY economic_event_id
"""

ACTUAL_SQL = """
SELECT economic_event_release_actual_id, economic_event_id, source_agency,
       source_observation_id, publication_state, revision_sequence,
       available_at, retrieved_at, source_url, source_artifact_path,
       source_artifact_sha256, semantic_contract, source_provenance::text,
       actual_raw, actual_value_kind, actual_value_low, actual_value_high,
       actual_canonical_value_low, actual_canonical_value_high,
       actual_unit, actual_scale, actual_qualifier
FROM economic_event_release_actual
ORDER BY economic_event_release_actual_id
"""


def psql_csv(database: str, query: str, read_only: bool = True) -> list[dict[str, str]]:
    environment = os.environ.copy()
    if read_only:
        current = environment.get("PGOPTIONS", "")
        environment["PGOPTIONS"] = (current + " -c default_transaction_read_only=on").strip()
    command = [
        "psql", "-X", "--no-psqlrc", "--set", "ON_ERROR_STOP=1",
        "--dbname", database, "--command",
        "COPY (" + query.strip().rstrip(";") + ") TO STDOUT WITH (FORMAT csv, HEADER true)",
    ]
    result = subprocess.run(
        command, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True, env=environment,
    )
    if result.returncode != 0:
        raise RuntimeError("psql_read_failed:" + " ".join(result.stderr.split()))
    return list(csv.DictReader(io.StringIO(result.stdout)))


def relation_exists(database: str, relation: str) -> bool:
    rows = psql_csv(
        database,
        "SELECT to_regclass('public." + relation + "') IS NOT NULL AS present",
    )
    return bool(rows and rows[0]["present"] == "t")


def load_database(database: str) -> tuple[list[EconomicEvent], list[Consensus], list[Mapping[str, object]]]:
    events = [EconomicEvent(
        economic_event_id=int(row["economic_event_id"]),
        source_agency=row["source_agency"],
        event_family=row["event_family"],
        event_timestamp_utc=canonical_instant(row["event_timestamp_utc"]),
        source_event_id=row["source_event_id"],
        source_url=row["source_url"],
        reference_period=row["reference_period"] or None,
    ) for row in psql_csv(database, EVENT_SQL)]
    consensus = [Consensus(
        economic_event_id=int(row["economic_event_id"]),
        value_kind=row["value_kind"],
        unit=row["unit"],
        qualifier=row["qualifier"] or None,
    ) for row in psql_csv(database, CONSENSUS_SQL)]
    existing: list[Mapping[str, object]] = []
    if relation_exists(database, "economic_event_release_actual"):
        existing = [
            {key: (None if value == "" else value) for key, value in row.items()}
            for row in psql_csv(database, ACTUAL_SQL)
        ]
    return events, consensus, existing


def git_archive_admissions(repo_root: pathlib.Path, source_root: pathlib.Path) -> dict[str, tuple[str, str]]:
    relative_root = source_root.resolve().relative_to(repo_root.resolve()).as_posix()
    dirty = subprocess.run(
        ["git", "diff", "--quiet", "HEAD", "--", relative_root],
        cwd=repo_root,
        check=False,
    )
    if dirty.returncode != 0:
        raise RuntimeError("source_archive_has_uncommitted_changes")
    changed_history = subprocess.run(
        [
            "git", "log", "--diff-filter=MDRC", "--format=", "--name-only",
            "--", relative_root,
        ],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if changed_history.returncode != 0:
        raise RuntimeError(
            "git_archive_history_failed:"
            + " ".join(changed_history.stderr.split())
        )
    if changed_history.stdout.strip():
        raise RuntimeError("source_archive_artifact_changed_after_admission")
    result = subprocess.run(
        [
            "git", "log", "--reverse", "--diff-filter=A",
            "--format=@@%H%x09%cI", "--name-only", "--", relative_root,
        ],
        cwd=repo_root,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("git_archive_history_failed:" + " ".join(result.stderr.split()))
    current: tuple[str, str] | None = None
    admissions: dict[str, tuple[str, str]] = {}
    for line in result.stdout.splitlines():
        if line.startswith("@@"):
            commit, timestamp = line[2:].split("\t", 1)
            current = (commit, timestamp)
        elif line and current is not None:
            admissions.setdefault(line, current)
    return admissions


def sql_literal(value: object) -> str:
    if value is None:
        return "NULL"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, Mapping):
        text = json.dumps(value, sort_keys=True, separators=(",", ":"))
        return "'" + text.replace("'", "''") + "'::jsonb"
    return "'" + str(value).replace("'", "''") + "'"


INSERT_COLUMNS = (
    "economic_event_id", "source_agency", "source_observation_id",
    "publication_state", "revision_sequence", "available_at", "retrieved_at",
    "source_url", "source_artifact_path", "source_artifact_sha256",
    "semantic_contract", "source_provenance", "actual_raw",
    "actual_value_kind", "actual_value_low", "actual_value_high",
    "actual_canonical_value_low", "actual_canonical_value_high", "actual_unit",
    "actual_scale", "actual_qualifier",
)


def build_insert_sql(decisions: Sequence[ImportDecision]) -> str:
    rows = sorted(
        (decision for decision in decisions if decision.decision == "matched"),
        key=lambda decision: (
            canonical_instant(decision.candidate.available_at),
            int(decision.matched_economic_event_id or -1),
            decision.candidate.source_observation_id,
        ),
    )
    if not rows:
        return "BEGIN;\nCOMMIT;\n"
    values = []
    for decision in rows:
        assert decision.matched_economic_event_id is not None
        persisted = decision.candidate.persisted_values(decision.matched_economic_event_id)
        values.append("(" + ",".join(sql_literal(persisted[column]) for column in INSERT_COLUMNS) + ")")
    return (
        "BEGIN;\n"
        "INSERT INTO economic_event_release_actual (\n    "
        + ",\n    ".join(INSERT_COLUMNS)
        + "\n) VALUES\n"
        + ",\n".join(values)
        + ";\nCOMMIT;\n"
    )


def execute_disposable_import(database: str, sql: str) -> None:
    if not database.lower().startswith("ea_"):
        raise ValueError("refusing_non_disposable_database")
    result = subprocess.run(
        ["psql", "-X", "--no-psqlrc", "--set", "ON_ERROR_STOP=1", "--dbname", database],
        input=sql,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError("transactional_import_failed:" + " ".join(result.stderr.split()))


def write_new(path: pathlib.Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def audit_with_artifact_inventory(
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
    artifacts: Sequence[CensusArtifact | BeaArtifact | BlsArtifact],
) -> dict[str, object]:
    report = coverage_audit(events, consensus, decisions, rejections)
    event_by_source = {event.source_event_id: event for event in events}
    artifact_by_path = {artifact.repository_path: artifact for artifact in artifacts}

    def update(values: dict[str, object], family: str, source_ids: set[str]) -> None:
        values["source_artifacts"] = len(source_ids)
        values["unsupported_semantic_cases"] = sum(
            rejection.candidate_event_family == family
            and rejection.artifact in artifact_by_path
            and artifact_by_path[rejection.artifact].source_event_id in source_ids
            and rejection.decision in {"invalid_semantics", "unsupported"}
            for rejection in rejections
        )

    target_start = "2010-01-01T00:00:00.000000Z"
    target_end = "2025-01-01T00:00:00.000000Z"
    for period_name in ("families", "post_target_period"):
        period = report[period_name]
        assert isinstance(period, dict)
        for family, values in period.items():
            assert isinstance(values, dict)
            source_ids = {
                artifact.source_event_id for artifact in artifacts
                if artifact.event_family == family and artifact.source_event_id in event_by_source
            }
            if period_name == "families":
                source_ids = {
                    source_id for source_id in source_ids
                    if target_start <= event_by_source[source_id].event_timestamp_utc < target_end
                }
            else:
                source_ids = {
                    source_id for source_id in source_ids
                    if event_by_source[source_id].event_timestamp_utc >= target_end
                }
            update(values, family, source_ids)

    by_year = report["by_year"]
    assert isinstance(by_year, dict)
    for key, values in by_year.items():
        assert isinstance(values, dict)
        year, family = key.split(":", 1)
        source_ids = {
            artifact.source_event_id for artifact in artifacts
            if artifact.event_family == family
            and artifact.source_event_id in event_by_source
            and event_by_source[artifact.source_event_id].event_timestamp_utc.startswith(year + "-")
        }
        update(values, family, source_ids)

    report["adapter_scope"] = {
        "supported": {
            "BEA": ["GDP", "PCE"],
            "BLS": ["CPI", "EMPLOYMENT", "JOLTS", "PPI"],
            "CENSUS": ["DURABLE_GOODS", "RETAIL_SALES"],
        },
        "unsupported": {
            "DOL_ETA": {
                "families": ["WEEKLY_CLAIMS"],
                "reason": "no local authoritative historical source archive exists",
            },
            "FEDERAL_RESERVE": {
                "families": ["FOMC"],
                "reason": "persisted consensus is predominantly range-valued and width-75 surprise is scalar-only",
            },
        },
    }
    return report


def main(argv: Iterable[str] | None = None) -> int:
    default_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Authoritative initial/revision actual dry-run and import workflow"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=default_root)
    parser.add_argument("--census-manifest", type=pathlib.Path)
    parser.add_argument("--census-prepared", type=pathlib.Path)
    parser.add_argument("--bea-canonical", type=pathlib.Path)
    parser.add_argument("--bea-prepared", type=pathlib.Path)
    parser.add_argument("--bls-manifest", type=pathlib.Path)
    parser.add_argument("--dry-run-output", type=pathlib.Path, required=True)
    parser.add_argument("--coverage-output", type=pathlib.Path, required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--allow-disposable-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    census_root = repo_root / "EconomicCalendar" / "raw" / "census"
    manifest = (args.census_manifest or census_root / "manifest.csv").resolve()
    prepared = (args.census_prepared or census_root / "census_import_prepared.csv").resolve()
    admissions = git_archive_admissions(repo_root, census_root / "releases")
    artifacts, source_ids = load_census_artifacts(
        repo_root, manifest, prepared, admissions
    )

    bea_root = repo_root / "EconomicCalendar" / "raw" / "bea"
    bea_canonical = (args.bea_canonical or bea_root / "bea_canonical_events.csv").resolve()
    bea_prepared = (args.bea_prepared or bea_root / "bea_import_prepared.csv").resolve()
    bea_admissions = git_archive_admissions(repo_root, bea_root / "releases")
    bea_admissions.update(
        git_archive_admissions(repo_root, bea_root / "recovery_v2" / "releases")
    )
    bea_artifacts, gdp_initial_by_quarter = load_bea_artifacts(
        repo_root, bea_canonical, bea_prepared, bea_admissions
    )

    events, consensus, existing = load_database(args.db)
    bls_artifacts: list[BlsArtifact] = []
    if args.bls_manifest:
        bls_artifacts = load_bls_artifacts(
            repo_root, args.bls_manifest.resolve(), events
        )

    candidates: list[ReleaseActualCandidate] = []
    rejections = []
    for artifact in artifacts:
        found, rejected = extract_census_candidates(artifact, source_ids)
        candidates.extend(found)
        rejections.extend(rejected)
    for artifact in bea_artifacts:
        found, rejected = extract_bea_candidates(artifact, gdp_initial_by_quarter)
        candidates.extend(found)
        rejections.extend(rejected)
    for artifact in bls_artifacts:
        found, rejected = extract_bls_candidates(artifact)
        candidates.extend(found)
        rejections.extend(rejected)

    decisions = match_candidates(candidates, events, existing)
    write_new(args.dry_run_output, deterministic_json_lines(decisions, rejections))
    coverage = audit_with_artifact_inventory(
        events, consensus, decisions, rejections,
        [*artifacts, *bea_artifacts, *bls_artifacts]
    )
    write_new(
        args.coverage_output,
        json.dumps(coverage, sort_keys=True, indent=2) + "\n",
    )

    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.decision] = counts.get(decision.decision, 0) + 1
    for rejection in rejections:
        counts[rejection.decision] = counts.get(rejection.decision, 0) + 1
    print("Phase 10 authoritative release-actual workflow")
    print(f"Artifacts: {len(artifacts) + len(bea_artifacts) + len(bls_artifacts)}")
    print(f"Candidates: {len(candidates)}")
    print("Decisions: " + json.dumps(counts, sort_keys=True))
    print(f"Dry run: {args.dry_run_output.resolve()}")
    print(f"Coverage: {args.coverage_output.resolve()}")

    if args.commit:
        if not args.allow_disposable_write:
            raise ValueError("--commit requires --allow-disposable-write")
        if not relation_exists(args.db, "economic_event_release_actual"):
            raise ValueError("economic_event_release_actual_missing")
        execute_disposable_import(args.db, build_insert_sql(decisions))
        print("RESULT: COMMITTED TO EXPLICIT DISPOSABLE DATABASE")
    else:
        print("RESULT: READ-ONLY DRY RUN")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PHASE10_RELEASE_ACTUAL_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
