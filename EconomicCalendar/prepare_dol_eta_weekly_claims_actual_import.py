#!/usr/bin/env python3
"""Prepare and optionally test a DOL/ETA Weekly Claims actual import package.

Catalog access is read-only.  Writes are permitted only to an explicitly
named disposable ``ea_*`` database through the established append-only import
SQL path.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import pathlib
import sys
from typing import Iterable, Mapping, Sequence

from dol_eta_weekly_claims_actual import (
    PARSER_VERSION,
    DolEtaArtifact,
    extract_dol_eta_corpus,
    load_dol_eta_artifacts,
)
from import_economic_event_release_actual import (
    build_insert_sql,
    execute_disposable_import,
    load_database,
)
from prepare_pce_production_import import deterministic_manifest, manifest_rows
from release_actual_ingestion import (
    Consensus,
    EconomicEvent,
    ImportDecision,
    Rejection,
    match_candidates,
)


CONTRACT = "dol_eta_weekly_claims_actual_remediation_phase7_v1"
IMPORTABLE = frozenset({"matched", "duplicate_identical"})


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _json_lines(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def classification_rows(
    acquisition_manifest: pathlib.Path,
    artifacts: Sequence[DolEtaArtifact],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
    events: Sequence[EconomicEvent],
) -> list[dict[str, object]]:
    acquisition: dict[str, dict[str, object]] = {}
    for line in acquisition_manifest.read_text(encoding="utf-8").splitlines():
        if line:
            row = json.loads(line)
            acquisition[str(row["source_url"])] = row
    artifact_by_path = {
        row.source_repository_path: row for row in artifacts
    }
    decision_by_artifact: dict[str, list[ImportDecision]] = collections.defaultdict(list)
    for decision in decisions:
        decision_by_artifact[decision.candidate.source_artifact_path].append(decision)
    rejection_by_artifact: dict[str, list[Rejection]] = collections.defaultdict(list)
    for rejection in rejections:
        rejection_by_artifact[rejection.artifact].append(rejection)

    rows: list[dict[str, object]] = []
    represented_urls: set[str] = set()
    for source_url in sorted(acquisition):
        source = acquisition[source_url]
        represented_urls.add(source_url)
        artifact_path = str(source.get("local_artifact_path") or "")
        artifact = artifact_by_path.get(artifact_path)
        observed = decision_by_artifact.get(artifact_path, [])
        rejected = rejection_by_artifact.get(artifact_path, [])
        if source.get("http_acquisition_result") != "succeeded":
            disposition = "missing_source_artifact"
            diagnostic = str(source.get("diagnostic") or "acquisition_failed")
        elif rejected and not observed:
            disposition = rejected[0].decision
            diagnostic = rejected[0].rejection_reason
        elif observed:
            initial = next(
                (row for row in observed if row.candidate.publication_state == "initial"),
                observed[0],
            )
            disposition = (
                "eligible" if initial.decision in IMPORTABLE else initial.decision
            )
            diagnostic = initial.rejection_reason
        else:
            disposition = "extraction_failed"
            diagnostic = "artifact_not_classified"
        initial_decision = next(
            (
                row for row in observed
                if row.candidate.publication_state == "initial"
            ),
            None,
        )
        rows.append({
            "manifest_version": 1,
            "parser_version": PARSER_VERSION,
            "release_date": source.get("release_date"),
            "source_url": source_url,
            "source_artifact_identity": source.get("source_artifact_identity"),
            "http_acquisition_result": source.get("http_acquisition_result"),
            "local_artifact_path": source.get("local_artifact_path"),
            "source_artifact_sha256": source.get("source_artifact_sha256"),
            "parser_artifact_path": source.get("parser_artifact_path"),
            "parser_artifact_sha256": source.get("parser_artifact_sha256"),
            "extractor": source.get("extractor"),
            "retrieved_at": source.get("retrieved_at"),
            "source_publication_timestamp_evidence": (
                initial_decision.candidate.source_provenance.get(
                    "publication_timestamp_evidence"
                ) if initial_decision else None
            ),
            "mapped_economic_event_id": (
                initial_decision.matched_economic_event_id
                if initial_decision else None
            ),
            "mapped_source_event_id": (
                initial_decision.candidate.candidate_source_event_id
                if initial_decision else None
            ),
            "extraction_disposition": (
                "parsed" if observed else disposition
            ),
            "import_eligibility_disposition": disposition,
            "diagnostic": diagnostic,
            "observation_count": len(observed),
            "initial_observation_count": sum(
                row.candidate.publication_state == "initial" for row in observed
            ),
            "revision_observation_count": sum(
                row.candidate.publication_state == "revision" for row in observed
            ),
        })

    catalog_urls = {
        event.source_url for event in events
        if event.source_agency == "DOL_ETA"
        and event.event_family == "WEEKLY_CLAIMS"
    }
    for source_url in sorted(catalog_urls - represented_urls):
        event = next(row for row in events if row.source_url == source_url)
        rows.append({
            "manifest_version": 1,
            "parser_version": PARSER_VERSION,
            "release_date": event.event_timestamp_utc[:10],
            "source_url": source_url,
            "source_artifact_identity": None,
            "http_acquisition_result": "not_attempted",
            "local_artifact_path": None,
            "source_artifact_sha256": None,
            "parser_artifact_path": None,
            "parser_artifact_sha256": None,
            "extractor": None,
            "retrieved_at": None,
            "source_publication_timestamp_evidence": None,
            "mapped_economic_event_id": event.economic_event_id,
            "mapped_source_event_id": event.source_event_id,
            "extraction_disposition": "missing_source_artifact",
            "import_eligibility_disposition": "missing_source_artifact",
            "diagnostic": "authoritative_release_artifact_not_acquired",
            "observation_count": 0,
            "initial_observation_count": 0,
            "revision_observation_count": 0,
        })
    rows.sort(key=lambda row: (
        str(row["release_date"] or ""), str(row["source_url"])
    ))
    return rows


def coverage_audit(
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    existing: Sequence[Mapping[str, object]],
    decisions: Sequence[ImportDecision],
    classifications: Sequence[Mapping[str, object]],
    sql: str,
) -> dict[str, object]:
    target = [
        row for row in events
        if row.source_agency == "DOL_ETA"
        and row.event_family == "WEEKLY_CLAIMS"
    ]
    target_ids = {row.economic_event_id for row in target}
    accepted = [
        row for row in decisions
        if row.decision in IMPORTABLE
        and row.matched_economic_event_id in target_ids
    ]
    initials = [
        row for row in accepted if row.candidate.publication_state == "initial"
    ]
    revisions = [
        row for row in accepted if row.candidate.publication_state == "revision"
    ]
    initial_ids = {
        int(row.matched_economic_event_id) for row in initials
        if row.matched_economic_event_id is not None
    }
    consensus_ids = {
        row.economic_event_id for row in consensus if row.economic_event_id in target_ids
    }
    compatible_ids = {
        int(row.matched_economic_event_id) for row in initials
        if row.matched_economic_event_id is not None
        and row.matched_economic_event_id in consensus_ids
        and next(
            item for item in consensus
            if item.economic_event_id == row.matched_economic_event_id
        ).value_kind == row.candidate.actual_value_kind
        and next(
            item for item in consensus
            if item.economic_event_id == row.matched_economic_event_id
        ).unit == row.candidate.actual_unit
        and next(
            item for item in consensus
            if item.economic_event_id == row.matched_economic_event_id
        ).qualifier == row.candidate.actual_qualifier
    }
    dispositions = collections.Counter(
        str(row["import_eligibility_disposition"]) for row in classifications
    )
    existing_ids = {
        int(row["economic_event_id"]) for row in existing
        if int(row["economic_event_id"]) in target_ids
    }
    matched = sum(row.decision == "matched" for row in decisions)
    return {
        "contract": CONTRACT,
        "database_interaction": "read_only_catalog_or_explicit_disposable_write",
        "production_writes": False,
        "event_count": len(target),
        "event_date_range": {
            "first": min((row.event_timestamp_utc for row in target), default=None),
            "last": max((row.event_timestamp_utc for row in target), default=None),
        },
        "candidate_release_artifacts": len(classifications),
        "acquired_artifacts": sum(
            row["http_acquisition_result"] == "succeeded"
            for row in classifications
        ),
        "failed_or_unavailable_artifacts": sum(
            row["http_acquisition_result"] != "succeeded"
            for row in classifications
        ),
        "eligibility_dispositions": dict(sorted(dispositions.items())),
        "mapped_initial_events": len(initial_ids),
        "qualifying_initial_observations": len(initials),
        "qualifying_revision_observations": len(revisions),
        "qualifying_immutable_observations": len(initials) + len(revisions),
        "unmapped_or_ambiguous": sum(
            key in {"missing_event_mapping", "event_mapping_ambiguous"}
            for key in (
                str(row["import_eligibility_disposition"])
                for row in classifications
            )
        ),
        "publication_time_unproven": dispositions["publication_time_unproven"],
        "initial_or_revision_semantics_insufficient": (
            dispositions["actual_value_unavailable"]
            + dispositions["actual_value_ambiguous"]
            + dispositions["unsupported_source_format"]
        ),
        "selected_consensus_events": len(consensus_ids),
        "usable_consensus_actual_intersection": len(compatible_ids),
        "actual_only_events": len(initial_ids - consensus_ids),
        "remaining_consensus_blocker": len(initial_ids - consensus_ids),
        "before_actual_events": len(existing_ids),
        "after_actual_events_if_applied": len(existing_ids | initial_ids),
        "matched_insert_observations": matched,
        "duplicate_identical_observations": sum(
            row.decision == "duplicate_identical" for row in decisions
        ),
        "conflicting_observations": sum(
            row.decision == "conflict" for row in decisions
        ),
        "sql_operation": "INSERT_ONLY_TRANSACTION",
        "sql_sha256": _sha256(sql),
    }


def prepare(
    database: str,
    repo_root: pathlib.Path,
    acquisition_manifest: pathlib.Path,
) -> tuple[str, str, str, Mapping[str, object]]:
    events, consensus, existing = load_database(database)
    artifacts = load_dol_eta_artifacts(repo_root, acquisition_manifest)
    candidates, rejections = extract_dol_eta_corpus(artifacts, events)
    decisions = match_candidates(candidates, events, existing)
    payload = deterministic_manifest(manifest_rows([
        row for row in decisions if row.decision == "matched"
    ]))
    sql = build_insert_sql([
        row for row in decisions if row.decision == "matched"
    ])
    normalized_sql = " " + " ".join(sql.upper().split()) + " "
    if any(token in normalized_sql for token in (
        " UPDATE ", " DELETE ", " ON CONFLICT ", " UPSERT ",
    )):
        raise ValueError("non_append_only_dol_eta_import_payload")
    classifications = classification_rows(
        acquisition_manifest, artifacts, decisions, rejections, events
    )
    audit = coverage_audit(
        events, consensus, existing, decisions, classifications, sql
    )
    return payload, sql, _json_lines(classifications), audit


def _write(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=root)
    parser.add_argument(
        "--acquisition-manifest", type=pathlib.Path,
        default=root / "EconomicCalendar/raw/dol_eta/actual_acquisition_manifest.jsonl",
    )
    parser.add_argument("--import-manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--sql-output", type=pathlib.Path, required=True)
    parser.add_argument("--classification-output", type=pathlib.Path, required=True)
    parser.add_argument("--audit-output", type=pathlib.Path, required=True)
    parser.add_argument("--commit", action="store_true")
    parser.add_argument("--allow-disposable-write", action="store_true")
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload, sql, classifications, audit = prepare(
        args.db, args.repo_root.resolve(), args.acquisition_manifest.resolve()
    )
    _write(args.import_manifest_output, payload)
    _write(args.sql_output, sql)
    _write(args.classification_output, classifications)
    _write(args.audit_output, json.dumps(audit, sort_keys=True, indent=2) + "\n")
    if args.commit:
        if not args.allow_disposable_write:
            raise ValueError("--commit requires --allow-disposable-write")
        execute_disposable_import(args.db, sql)
        result = "COMMITTED_TO_EXPLICIT_DISPOSABLE_DATABASE"
    else:
        result = "READ_ONLY_DRY_RUN"
    print(
        "DOL_ETA_WEEKLY_CLAIMS_ACTUAL_PREPARATION"
        f",events={audit['event_count']}"
        f",artifacts={audit['candidate_release_artifacts']}"
        f",mapped_initials={audit['mapped_initial_events']}"
        f",observations={audit['qualifying_immutable_observations']}"
        f",result={result}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"DOL_ETA_WEEKLY_CLAIMS_ACTUAL_PREPARATION_FAILED:{error}", file=sys.stderr)
        raise SystemExit(1)
