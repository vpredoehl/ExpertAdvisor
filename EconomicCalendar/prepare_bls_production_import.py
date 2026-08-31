#!/usr/bin/env python3
"""Prepare a deterministic, read-only Phase 17 BLS production import plan."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter, defaultdict
from decimal import Decimal
from typing import Iterable, Mapping, Sequence

from import_economic_event_release_actual import build_insert_sql, load_database
from prepare_pce_production_import import deterministic_manifest, manifest_rows
from release_actual_ingestion import (
    BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT,
    BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT,
    BLS_SUPPORTED_FAMILIES,
    BlsArtifact,
    Consensus,
    EconomicEvent,
    ImportDecision,
    Rejection,
    ReleaseActualCandidate,
    canonical_instant,
    extract_bls_candidates,
    load_bls_artifacts,
    match_candidates,
)


CONTRACT = "bls_production_initial_actual_import_phase17_v1"
FAMILY_ORDER = ("CPI", "EMPLOYMENT", "PPI", "JOLTS")
IMPORTABLE = frozenset({"matched", "duplicate_identical"})


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_candidates(
    repo_root: pathlib.Path,
    events: Sequence[EconomicEvent],
    manifest_path: pathlib.Path | None = None,
) -> tuple[list[ReleaseActualCandidate], list[Rejection], list[BlsArtifact]]:
    manifest = (
        manifest_path
        or repo_root / "EconomicCalendar/raw/bls/manifest.jsonl"
    ).resolve()
    artifacts = load_bls_artifacts(repo_root, manifest, events)
    candidates: list[ReleaseActualCandidate] = []
    rejections: list[Rejection] = []
    for item in artifacts:
        found, rejected = extract_bls_candidates(item)
        candidates.extend(found)
        rejections.extend(rejected)
    for candidate in candidates:
        if (
            candidate.source_agency != "BLS"
            or candidate.candidate_event_family not in BLS_SUPPORTED_FAMILIES
            or candidate.publication_state != "initial"
            or candidate.revision_sequence != 0
        ):
            raise ValueError("bls_non_initial_candidate")
    return candidates, rejections, artifacts


def compatible(candidate: ReleaseActualCandidate, consensus: Consensus) -> bool:
    return (
        candidate.actual_value_kind == consensus.value_kind
        and candidate.actual_unit == consensus.unit
        and candidate.actual_qualifier == consensus.qualifier
    )


def ppi_provider_actual_validation(
    repo_root: pathlib.Path,
    candidates: Sequence[ReleaseActualCandidate],
    artifacts: Sequence[BlsArtifact],
) -> dict[str, object]:
    release_date_by_url = {row.source_url: row.release_date for row in artifacts}
    candidate_by_identity = {
        (
            row.candidate_reference_period,
            release_date_by_url[row.source_url],
        ): row
        for row in candidates
        if row.candidate_event_family == "PPI"
    }
    source = repo_root / "EconomicCalendar/raw/oanda/oanda_consensus_matches.csv"
    compared = 0
    finished_goods = 0
    final_demand = 0
    mismatches: list[dict[str, str]] = []
    with source.open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["event_family"] != "PPI" or not row["oanda_actual"]:
                continue
            key = (row["official_reference_period"], row["official_release_date"])
            candidate = candidate_by_identity.get(key)
            if candidate is None:
                mismatches.append({
                    "reference_period": key[0],
                    "release_date": key[1],
                    "reason": "bls_candidate_missing",
                })
                continue
            provider = Decimal(
                row["oanda_actual"].replace("% m/m", "").replace(",", ".")
            )
            actual = Decimal(candidate.actual_canonical_value_low)
            if provider != actual:
                mismatches.append({
                    "reference_period": key[0],
                    "release_date": key[1],
                    "provider_actual": str(provider),
                    "bls_actual": str(actual),
                    "reason": "value_mismatch",
                })
                continue
            compared += 1
            if candidate.semantic_contract == BLS_PPI_FINISHED_GOODS_SEMANTIC_CONTRACT:
                finished_goods += 1
            elif candidate.semantic_contract == BLS_PPI_FINAL_DEMAND_SEMANTIC_CONTRACT:
                final_demand += 1
            else:
                raise ValueError("ppi_candidate_contract_invalid")
    if mismatches:
        raise ValueError("ppi_provider_actual_mismatch:" + json.dumps(mismatches, sort_keys=True))
    return {
        "source": source.relative_to(repo_root).as_posix(),
        "comparison": "provider_same-event_actual_equals_bls_initial_headline",
        "matched_observations": compared,
        "mismatches": 0,
        "finished_goods_matches": finished_goods,
        "final_demand_matches": final_demand,
    }


def classification_rows(
    events: Sequence[EconomicEvent],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    represented: set[str] = set()
    for decision in decisions:
        candidate = decision.candidate
        represented.add(candidate.candidate_source_event_id)
        rows.append({
            "artifact": candidate.source_artifact_path,
            "available_at": candidate.available_at,
            "decision": decision.decision,
            "economic_event_id": decision.matched_economic_event_id,
            "event_family": candidate.candidate_event_family,
            "publication_state": candidate.publication_state,
            "reference_period": candidate.candidate_reference_period,
            "rejection_reason": decision.rejection_reason,
            "revision_sequence": candidate.revision_sequence,
            "semantic_contract": candidate.semantic_contract,
            "source_event_id": candidate.candidate_source_event_id,
        })
    for rejection in rejections:
        rows.append({
            "artifact": rejection.artifact,
            "available_at": None,
            "decision": rejection.decision,
            "economic_event_id": None,
            "event_family": rejection.candidate_event_family,
            "publication_state": None,
            "reference_period": rejection.candidate_reference_period,
            "rejection_reason": rejection.rejection_reason,
            "revision_sequence": None,
            "semantic_contract": None,
            "source_event_id": None,
        })
    for event in events:
        if (
            event.source_agency == "BLS"
            and event.event_family in FAMILY_ORDER
            and event.source_event_id not in represented
            and not any(
                row["event_family"] == event.event_family
                and row["reference_period"] == event.reference_period
                for row in rows
            )
        ):
            rows.append({
                "artifact": None,
                "available_at": None,
                "decision": "needs_source_acquisition",
                "economic_event_id": event.economic_event_id,
                "event_family": event.event_family,
                "publication_state": None,
                "reference_period": event.reference_period,
                "rejection_reason": "authoritative_release_artifact_missing",
                "revision_sequence": None,
                "semantic_contract": None,
                "source_event_id": event.source_event_id,
            })
    rows.sort(key=lambda row: (
        FAMILY_ORDER.index(str(row["event_family"])),
        str(row["available_at"] or ""),
        int(row["economic_event_id"] or -1),
        str(row["artifact"] or ""),
    ))
    return rows


def deterministic_json_lines(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def coverage_audit(
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    existing: Sequence[Mapping[str, object]],
    decisions: Sequence[ImportDecision],
    classifications: Sequence[Mapping[str, object]],
    artifacts: Sequence[BlsArtifact],
    manifest: str,
    sql: str,
    ppi_validation: Mapping[str, object],
) -> dict[str, object]:
    consensus_by_id = {row.economic_event_id: row for row in consensus}
    existing_ids = {int(row["economic_event_id"]) for row in existing}
    artifact_counts = Counter(row.event_family for row in artifacts)
    accepted = [row for row in decisions if row.decision in IMPORTABLE]
    accepted_by_family: dict[str, list[ImportDecision]] = defaultdict(list)
    for row in accepted:
        accepted_by_family[row.candidate.candidate_event_family].append(row)
    family_reports: dict[str, object] = {}
    for family in FAMILY_ORDER:
        family_events = [
            row for row in events
            if row.source_agency == "BLS" and row.event_family == family
        ]
        ids = {row.economic_event_id for row in family_events}
        selected_ids = ids.intersection(consensus_by_id)
        accepted_rows = accepted_by_family[family]
        certified_ids = {
            int(row.matched_economic_event_id)
            for row in accepted_rows if row.matched_economic_event_id is not None
        }
        compatible_ids = {
            int(row.matched_economic_event_id)
            for row in accepted_rows
            if row.matched_economic_event_id in consensus_by_id
            and compatible(
                row.candidate,
                consensus_by_id[int(row.matched_economic_event_id)],
            )
        }
        family_classes = [
            row for row in classifications if row["event_family"] == family
        ]
        decisions_count = Counter(str(row["decision"]) for row in family_classes)
        times = sorted(row.candidate.available_at for row in accepted_rows)
        if decisions_count["needs_source_acquisition"]:
            decision = "PARTIAL"
        elif decisions_count["missing_initial_provenance"]:
            decision = "PARTIAL"
        else:
            decision = "READY"
        before_ids = ids.intersection(existing_ids)
        family_reports[family] = {
            "retained_authoritative_artifacts": artifact_counts[family],
            "certified_initial_actuals": len(certified_ids),
            "selected_consensus": len(selected_ids),
            "usable_consensus_actual_intersection": len(compatible_ids),
            "actual_only": len(certified_ids - selected_ids),
            "consensus_only": len(selected_ids - certified_ids),
            "earliest_certified_release": times[0] if times else None,
            "latest_certified_release": times[-1] if times else None,
            "excluded_revision": 0,
            "excluded_ambiguous": decisions_count["ambiguous"],
            "excluded_invalid_semantics": decisions_count["invalid_semantics"],
            "excluded_missing_initial_provenance": decisions_count[
                "missing_initial_provenance"
            ],
            "acquisition_failures": decisions_count["needs_source_acquisition"],
            "conflicts": decisions_count["conflict"],
            "duplicate_identical": decisions_count["duplicate_identical"],
            "before_family_actual_count": len(before_ids),
            "after_family_actual_count": len(before_ids.union(certified_ids)),
            "before_usable_intersection": len(before_ids.intersection(selected_ids)),
            "after_usable_intersection": len(compatible_ids),
            "decision": decision,
        }
    matched = sum(row.decision == "matched" for row in decisions)
    duplicates = sum(row.decision == "duplicate_identical" for row in decisions)
    conflicts = sum(row.decision == "conflict" for row in decisions)
    return {
        "contract": CONTRACT,
        "database_interaction": "read_only_production_catalog",
        "production_writes": False,
        "payload": {
            "manifest_rows": len(manifest.splitlines()),
            "manifest_sha256": sha256_text(manifest),
            "sql_sha256": sha256_text(sql),
            "expected_insert_count": matched,
            "duplicate_identical_count": duplicates,
            "conflict_count": conflicts,
            "before_total_release_actuals": len(existing),
            "after_total_release_actuals": len(existing) + matched,
            "sql_operation": "INSERT_ONLY_TRANSACTION",
        },
        "ppi_regime_compatibility": dict(ppi_validation),
        "families": family_reports,
        "decision": "BLS_PRODUCTION_IMPORT_READY" if conflicts == 0 and matched else "NOT_READY",
    }


def prepare(
    database: str,
    repo_root: pathlib.Path,
    manifest_path: pathlib.Path | None = None,
) -> tuple[str, str, str, Mapping[str, object]]:
    events, consensus, existing = load_database(database)
    candidates, rejections, artifacts = load_candidates(
        repo_root, events, manifest_path
    )
    decisions = match_candidates(candidates, events, existing)
    payload = deterministic_manifest(manifest_rows(decisions))
    sql = build_insert_sql([
        row for row in decisions if row.decision == "matched"
    ])
    upper = " " + " ".join(sql.upper().split()) + " "
    if any(token in upper for token in (
        " UPDATE ", " DELETE ", " ON CONFLICT ", " UPSERT ",
    )):
        raise ValueError("non_append_only_bls_import_payload")
    classifications = classification_rows(events, decisions, rejections)
    ppi_validation = ppi_provider_actual_validation(
        repo_root, candidates, artifacts
    )
    audit = coverage_audit(
        events,
        consensus,
        existing,
        decisions,
        classifications,
        artifacts,
        payload,
        sql,
        ppi_validation,
    )
    return payload, sql, deterministic_json_lines(classifications), audit


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def main(argv: Iterable[str] | None = None) -> int:
    default_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Phase 17 BLS initial-actual payload"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=default_root)
    parser.add_argument("--bls-manifest", type=pathlib.Path)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    payload, sql, classifications, audit = prepare(
        args.db,
        args.repo_root.resolve(),
        args.bls_manifest.resolve() if args.bls_manifest else None,
    )
    output = args.output_directory.resolve()
    write_text(output / "bls-initial-actual-manifest.jsonl", payload)
    write_text(output / "bls-initial-actual-import.sql", sql)
    write_text(output / "bls-candidate-classifications.jsonl", classifications)
    write_text(
        output / "bls-coverage-audit.json",
        json.dumps(audit, sort_keys=True, indent=2) + "\n",
    )
    print("Phase 17 BLS production import readiness")
    print(json.dumps(audit["payload"], sort_keys=True))
    print(f"Output: {output}")
    print(f"RESULT: {audit['decision']} / PRODUCTION NOT MODIFIED")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PHASE17_BLS_READINESS_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
