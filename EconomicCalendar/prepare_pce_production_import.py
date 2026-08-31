#!/usr/bin/env python3
"""Prepare a deterministic, no-write PCE production import manifest.

This Phase 11 workflow reads the authoritative local BEA archive and the
current PostgreSQL event/consensus catalog.  It can write review artifacts to
the filesystem, but it deliberately has no database mutation option.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Iterable, Mapping, Sequence

from import_economic_event_release_actual import (
    audit_with_artifact_inventory,
    build_insert_sql,
    git_archive_admissions,
    load_database,
)
from release_actual_ingestion import (
    BeaArtifact,
    Consensus,
    EconomicEvent,
    ImportDecision,
    Rejection,
    ReleaseActualCandidate,
    canonical_instant,
    extract_bea_candidates,
    load_bea_artifacts,
    match_candidates,
)


MANIFEST_CONTRACT = "bea_pce_production_import_manifest_v1"
IMPORTABLE_DECISIONS = frozenset({"matched", "duplicate_identical"})
IDENTITY_REASONS = frozenset({
    "source_agency_mismatch",
    "canonical_event_identity_mismatch",
    "canonical_source_url_mismatch",
    "available_at_predates_event",
    "initial_available_at_event_mismatch",
})
MANIFEST_FIELDS = (
    "economic_event_id",
    "source_agency",
    "source_observation_id",
    "publication_state",
    "revision_sequence",
    "available_at",
    "retrieved_at",
    "source_url",
    "source_artifact_path",
    "source_artifact_sha256",
    "semantic_contract",
    "source_provenance",
    "actual_raw",
    "actual_value_kind",
    "actual_value_low",
    "actual_value_high",
    "actual_canonical_value_low",
    "actual_canonical_value_high",
    "actual_unit",
    "actual_scale",
    "actual_qualifier",
)


def load_pce_candidates(
    repo_root: pathlib.Path,
    canonical_path: pathlib.Path | None = None,
    prepared_path: pathlib.Path | None = None,
) -> tuple[list[ReleaseActualCandidate], list[Rejection], list[BeaArtifact]]:
    """Load only source-backed BEA PCE initials from the immutable archive."""
    bea_root = repo_root / "EconomicCalendar" / "raw" / "bea"
    canonical = (canonical_path or bea_root / "bea_canonical_events.csv").resolve()
    prepared = (prepared_path or bea_root / "bea_import_prepared.csv").resolve()
    admissions = git_archive_admissions(repo_root, bea_root / "releases")
    admissions.update(
        git_archive_admissions(repo_root, bea_root / "recovery_v2" / "releases")
    )
    artifacts, gdp_initial_by_quarter = load_bea_artifacts(
        repo_root, canonical, prepared, admissions
    )
    pce_artifacts = [row for row in artifacts if row.event_family == "PCE"]
    candidates: list[ReleaseActualCandidate] = []
    rejections: list[Rejection] = []
    for artifact in pce_artifacts:
        found, rejected = extract_bea_candidates(
            artifact, gdp_initial_by_quarter
        )
        candidates.extend(found)
        rejections.extend(rejected)
    for candidate in candidates:
        if (candidate.candidate_event_family != "PCE" or
                candidate.publication_state != "initial" or
                candidate.revision_sequence != 0):
            raise ValueError("pce_manifest_non_initial_candidate")
    return candidates, rejections, pce_artifacts


def manifest_rows(decisions: Sequence[ImportDecision]) -> list[dict[str, object]]:
    """Return exact migration-088 payloads in one canonical ordering."""
    rows: list[dict[str, object]] = []
    for decision in decisions:
        if decision.decision not in IMPORTABLE_DECISIONS:
            continue
        if decision.matched_economic_event_id is None:
            raise ValueError("importable_decision_missing_event_identity")
        persisted = decision.candidate.persisted_values(
            decision.matched_economic_event_id
        )
        if tuple(persisted) != MANIFEST_FIELDS:
            raise ValueError("migration_088_manifest_field_drift")
        rows.append(persisted)
    rows.sort(key=lambda row: (
        canonical_instant(str(row["available_at"])),
        int(row["economic_event_id"]),
        str(row["source_observation_id"]),
    ))
    return rows


def deterministic_manifest(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def manifest_sha256(manifest: str) -> str:
    return hashlib.sha256(manifest.encode("utf-8")).hexdigest()


def _consensus_compatible(
    candidate: ReleaseActualCandidate, consensus: Consensus
) -> bool:
    return (
        consensus.value_kind == candidate.actual_value_kind
        and consensus.unit == candidate.actual_unit
        and consensus.qualifier == candidate.actual_qualifier
    )


def reconciliation_rows(
    decisions: Sequence[ImportDecision],
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
) -> list[dict[str, object]]:
    event_by_id = {row.economic_event_id: row for row in events}
    consensus_by_id = {row.economic_event_id: row for row in consensus}
    result: list[dict[str, object]] = []
    for decision in decisions:
        event = event_by_id.get(decision.matched_economic_event_id or -1)
        selected = consensus_by_id.get(decision.matched_economic_event_id or -1)
        compatible = (
            _consensus_compatible(decision.candidate, selected)
            if selected is not None else None
        )
        if decision.decision == "unmatched":
            classification = "MISSING_EVENT"
        elif decision.decision == "ambiguous":
            classification = "IDENTITY_MISMATCH"
        elif decision.decision == "invalid_semantics":
            classification = (
                "IDENTITY_MISMATCH"
                if decision.rejection_reason in IDENTITY_REASONS
                else "INVALID_SEMANTICS"
            )
        elif decision.decision == "duplicate_identical":
            classification = "DUPLICATE_EXISTING"
        elif decision.decision == "conflict":
            classification = "CONFLICT_EXISTING"
        elif selected is None:
            classification = "MISSING_CONSENSUS"
        elif not compatible:
            classification = "INCOMPATIBLE_CONSENSUS"
        else:
            classification = "READY"

        result.append({
            "classification": classification,
            "import_eligible": decision.decision in IMPORTABLE_DECISIONS,
            "economic_event_id": decision.matched_economic_event_id,
            "source_agency": event.source_agency if event else None,
            "event_family": event.event_family if event else None,
            "candidate_source_event_id": (
                decision.candidate.candidate_source_event_id
            ),
            "catalog_source_event_id": event.source_event_id if event else None,
            "candidate_reference_period": (
                decision.candidate.candidate_reference_period
            ),
            "catalog_reference_period": event.reference_period if event else None,
            "candidate_source_url": decision.candidate.source_url,
            "catalog_source_url": event.source_url if event else None,
            "candidate_available_at": decision.candidate.available_at,
            "catalog_event_timestamp_utc": (
                event.event_timestamp_utc if event else None
            ),
            "selected_consensus_present": selected is not None,
            "selected_consensus_value_kind": (
                selected.value_kind if selected else None
            ),
            "selected_consensus_unit": selected.unit if selected else None,
            "selected_consensus_qualifier": (
                selected.qualifier if selected else None
            ),
            "consensus_actual_semantics_compatible": compatible,
            "source_observation_id": decision.candidate.source_observation_id,
            "decision": decision.decision,
            "rejection_reason": decision.rejection_reason,
        })
    result.sort(key=lambda row: (
        str(row["candidate_available_at"]),
        int(row["economic_event_id"] or -1),
        str(row["source_observation_id"]),
    ))
    return result


def readiness_audit(
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
    artifacts: Sequence[BeaArtifact],
    manifest: str,
) -> dict[str, object]:
    reconciliation = reconciliation_rows(decisions, events, consensus)
    classifications = Counter(row["classification"] for row in reconciliation)
    coverage = audit_with_artifact_inventory(
        events, consensus, decisions, rejections, artifacts
    )
    importable = [row for row in reconciliation if row["import_eligible"]]
    availability = sorted(str(row["candidate_available_at"]) for row in importable)
    return {
        "contract": "bea_pce_production_readiness_v1",
        "database_interaction": "read_only_catalog_queries_only",
        "manifest_contract": MANIFEST_CONTRACT,
        "manifest_sha256": manifest_sha256(manifest),
        "manifest_row_count": len(manifest.splitlines()),
        "candidate_initial_count": len(decisions),
        "import_eligible_count": len(importable),
        "classification_counts": {
            key: classifications.get(key, 0)
            for key in (
                "READY", "MISSING_EVENT", "IDENTITY_MISMATCH",
                "MISSING_CONSENSUS", "INCOMPATIBLE_CONSENSUS",
                "INVALID_SEMANTICS", "DUPLICATE_EXISTING",
                "CONFLICT_EXISTING",
            )
        },
        "rejected_source_observation_count": len(rejections),
        "rejected_source_observations": [dataclasses.asdict(row) for row in sorted(
            rejections,
            key=lambda row: (
                row.artifact,
                row.candidate_reference_period or "",
                row.rejection_reason,
            ),
        )],
        "earliest_import_eligible_available_at": availability[0] if availability else None,
        "latest_import_eligible_available_at": availability[-1] if availability else None,
        "historical_2010_through_2024": coverage["families"]["PCE"],
        "post_2025": coverage["post_target_period"]["PCE"],
        "reconciliation": reconciliation,
    }


def write_text(path: pathlib.Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    default_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Prepare deterministic PCE production import review artifacts"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=default_root)
    parser.add_argument("--manifest-output", type=pathlib.Path, required=True)
    parser.add_argument("--audit-output", type=pathlib.Path, required=True)
    parser.add_argument("--sql-output", type=pathlib.Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    repo_root = args.repo_root.resolve()
    candidates, rejections, pce_artifacts = load_pce_candidates(repo_root)
    events, consensus, existing = load_database(args.db)
    decisions = match_candidates(candidates, events, existing)
    rows = manifest_rows(decisions)
    manifest = deterministic_manifest(rows)
    audit = readiness_audit(
        events, consensus, decisions, rejections, pce_artifacts, manifest
    )

    insertable = [
        row for row in decisions if row.decision == "matched"
    ]
    sql = build_insert_sql(insertable)
    if any(token in sql.upper() for token in (
        " UPDATE ", " DELETE ", "ON CONFLICT", "UPSERT"
    )):
        raise ValueError("non_append_only_import_payload")

    write_text(args.manifest_output, manifest)
    write_text(
        args.audit_output,
        json.dumps(audit, sort_keys=True, indent=2) + "\n",
    )
    write_text(args.sql_output, sql)

    print("Phase 11 PCE production readiness")
    print(f"Candidates: {len(decisions)}")
    print(f"Import eligible: {len(rows)}")
    print(
        "Classifications: "
        + json.dumps(audit["classification_counts"], sort_keys=True)
    )
    print(f"Rejected source observations: {len(rejections)}")
    print(f"Manifest SHA-256: {audit['manifest_sha256']}")
    print(f"Manifest: {args.manifest_output.resolve()}")
    print(f"Audit: {args.audit_output.resolve()}")
    print(f"SQL: {args.sql_output.resolve()}")
    print("RESULT: READ-ONLY PRODUCTION CATALOG / FILESYSTEM ARTIFACTS ONLY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PHASE11_PCE_READINESS_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
