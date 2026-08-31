#!/usr/bin/env python3
"""Prepare deterministic Phase 15 multi-family initial-actual review payloads.

The workflow is deliberately read-only with respect to PostgreSQL.  It reuses
the provenance-certified BEA and Census archive adapters, emits append-only SQL
only for source-backed revision-0 initials, and records every other target
family/event as an explicit fail-closed classification.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import pathlib
import sys
from collections import Counter
from typing import Iterable, Mapping, Sequence

from import_economic_event_release_actual import (
    build_insert_sql,
    git_archive_admissions,
    load_database,
)
from prepare_pce_production_import import (
    deterministic_manifest,
    manifest_rows,
)
from release_actual_ingestion import (
    BeaArtifact,
    CensusArtifact,
    Consensus,
    EconomicEvent,
    ImportDecision,
    Rejection,
    ReleaseActualCandidate,
    canonical_instant,
    extract_bea_candidates,
    extract_census_candidates,
    load_bea_artifacts,
    load_census_artifacts,
    match_candidates,
)


TARGET_FAMILIES = (
    "CPI",
    "EMPLOYMENT",
    "GDP",
    "RETAIL_SALES",
    "PPI",
    "JOLTS",
    "FOMC",
)
IMPORTABLE_FAMILIES = ("GDP", "RETAIL_SALES")
CLASSIFICATIONS = (
    "importable_initial",
    "duplicate_identical",
    "revision_only",
    "ambiguous_match",
    "unmatched",
    "unsupported_semantics",
    "unsupported_unit",
    "source_agency_mismatch",
    "causal_violation",
    "insufficient_initial_release_provenance",
)
READY = "READY_FOR_PRODUCTION_INITIAL_ACTUAL_IMPORT"
NOT_READY = "NOT_READY_FOR_PRODUCTION_INITIAL_ACTUAL_IMPORT"

FAMILY_CONTRACTS: Mapping[str, Mapping[str, object]] = {
    "CPI": {
        "source_agency": "BLS",
        "source_release_type": "Consumer Price Index news release",
        "supported_measure": None,
        "unit": None,
        "qualifier": None,
        "value_kind": None,
        "scale": None,
        "reference_period_contract": "monthly, not certified without retained release artifact",
        "release_timestamp_contract": "not certified without retained release artifact",
        "initial_status_proof": None,
        "revised_values_coexist": None,
        "semantic_contract": None,
        "readiness": NOT_READY,
        "readiness_reason": "no_retained_bls_release_specific_artifacts",
        "authoritative_archive_index": "https://www.bls.gov/bls/news-release/cpi.htm",
    },
    "EMPLOYMENT": {
        "source_agency": "BLS",
        "source_release_type": "Employment Situation news release",
        "supported_measure": None,
        "unit": None,
        "qualifier": None,
        "value_kind": None,
        "scale": None,
        "reference_period_contract": "monthly, not certified without retained release artifact",
        "release_timestamp_contract": "not certified without retained release artifact",
        "initial_status_proof": None,
        "revised_values_coexist": None,
        "semantic_contract": None,
        "readiness": NOT_READY,
        "readiness_reason": "no_retained_bls_release_specific_artifacts",
        "authoritative_archive_index": "https://www.bls.gov/bls/news-release/empsit.htm",
    },
    "GDP": {
        "source_agency": "BEA",
        "source_release_type": "GDP advance or initial estimate news release",
        "supported_measure": "real_gdp_annualized_quarter_over_quarter",
        "unit": "percent",
        "qualifier": None,
        "value_kind": "scalar",
        "scale": "1",
        "reference_period_contract": "calendar quarter plus Advance or Initial estimate identity",
        "release_timestamp_contract": "exact BEA release instant equals catalog event timestamp",
        "initial_status_proof": "release title and headline explicitly identify advance or initial estimate",
        "revised_values_coexist": True,
        "semantic_contract": "bea_real_gdp_annualized_quarterly_percent_v1",
        "readiness": READY,
        "readiness_reason": "retained_release_specific_advance_estimates",
        "authoritative_archive_index": "https://www.bea.gov/news/archive",
    },
    "RETAIL_SALES": {
        "source_agency": "CENSUS",
        "source_release_type": "Advance Monthly Sales for Retail and Food Services",
        "supported_measure": "retail_and_food_services_sales_month_over_month",
        "unit": "percent",
        "qualifier": "m/m",
        "value_kind": "scalar",
        "scale": "1",
        "reference_period_contract": "named calendar month in advance monthly release",
        "release_timestamp_contract": "FOR IMMEDIATE RELEASE instant equals catalog event timestamp",
        "initial_status_proof": "ADVANCE marker and current-period headline statistic coexist in exact release PDF",
        "revised_values_coexist": True,
        "semantic_contract": "census_advance_release_headline_mom_percent_v1",
        "readiness": READY,
        "readiness_reason": "retained_release_specific_advance_releases",
        "authoritative_archive_index": "https://www.census.gov/retail/index.html",
    },
    "PPI": {
        "source_agency": "BLS",
        "source_release_type": "Producer Price Index news release",
        "supported_measure": None,
        "unit": None,
        "qualifier": None,
        "value_kind": None,
        "scale": None,
        "reference_period_contract": "monthly, not certified without retained release artifact",
        "release_timestamp_contract": "not certified without retained release artifact",
        "initial_status_proof": None,
        "revised_values_coexist": None,
        "semantic_contract": None,
        "readiness": NOT_READY,
        "readiness_reason": "no_retained_bls_release_specific_artifacts_and_measure_regime_ambiguity",
        "authoritative_archive_index": "https://www.bls.gov/bls/news-release/ppi.htm",
    },
    "JOLTS": {
        "source_agency": "BLS",
        "source_release_type": "Job Openings and Labor Turnover Survey news release",
        "supported_measure": None,
        "unit": None,
        "qualifier": None,
        "value_kind": None,
        "scale": None,
        "reference_period_contract": "monthly, not certified without retained release artifact",
        "release_timestamp_contract": "not certified without retained release artifact",
        "initial_status_proof": None,
        "revised_values_coexist": None,
        "semantic_contract": None,
        "readiness": NOT_READY,
        "readiness_reason": "no_retained_bls_release_specific_artifacts",
        "authoritative_archive_index": "https://www.bls.gov/bls/news-release/jolts.htm",
    },
    "FOMC": {
        "source_agency": "FEDERAL_RESERVE",
        "source_release_type": "FOMC statement",
        "supported_measure": None,
        "unit": None,
        "qualifier": None,
        "value_kind": None,
        "scale": None,
        "reference_period_contract": "meeting statement identity",
        "release_timestamp_contract": "exact statement release instant retained",
        "initial_status_proof": "statement itself is retained, but no compatible actual-value contract is defined",
        "revised_values_coexist": False,
        "semantic_contract": None,
        "readiness": NOT_READY,
        "readiness_reason": "narrative_decisions_are_not_actuals_and_range_outcomes_are_not_scalar_surprise_compatible",
        "authoritative_archive_index": "https://www.federalreserve.gov/monetarypolicy/fomccalendars.htm",
    },
}


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def load_supported_source_corpus(
    repo_root: pathlib.Path,
) -> tuple[
    list[ReleaseActualCandidate],
    list[Rejection],
    Mapping[str, Sequence[BeaArtifact | CensusArtifact]],
]:
    census_root = repo_root / "EconomicCalendar" / "raw" / "census"
    census_admissions = git_archive_admissions(
        repo_root, census_root / "releases"
    )
    census_artifacts, source_ids = load_census_artifacts(
        repo_root,
        census_root / "manifest.csv",
        census_root / "census_import_prepared.csv",
        census_admissions,
    )

    bea_root = repo_root / "EconomicCalendar" / "raw" / "bea"
    bea_admissions = git_archive_admissions(repo_root, bea_root / "releases")
    bea_admissions.update(
        git_archive_admissions(repo_root, bea_root / "recovery_v2" / "releases")
    )
    bea_artifacts, gdp_initial_by_quarter = load_bea_artifacts(
        repo_root,
        bea_root / "bea_canonical_events.csv",
        bea_root / "bea_import_prepared.csv",
        bea_admissions,
    )

    retained: dict[str, Sequence[BeaArtifact | CensusArtifact]] = {
        "GDP": tuple(row for row in bea_artifacts if row.event_family == "GDP"),
        "RETAIL_SALES": tuple(
            row for row in census_artifacts
            if row.event_family == "RETAIL_SALES"
        ),
    }
    candidates: list[ReleaseActualCandidate] = []
    rejections: list[Rejection] = []
    for artifact in retained["GDP"]:
        found, rejected = extract_bea_candidates(
            artifact, gdp_initial_by_quarter
        )
        candidates.extend(found)
        rejections.extend(rejected)
    for artifact in retained["RETAIL_SALES"]:
        found, rejected = extract_census_candidates(artifact, source_ids)
        candidates.extend(found)
        rejections.extend(rejected)
    return candidates, rejections, retained


def _decision_classification(decision: ImportDecision) -> str:
    candidate = decision.candidate
    if decision.decision == "matched":
        return (
            "importable_initial"
            if candidate.publication_state == "initial"
            else "revision_only"
        )
    if decision.decision == "duplicate_identical":
        return (
            "duplicate_identical"
            if candidate.publication_state == "initial"
            else "revision_only"
        )
    if decision.decision == "ambiguous":
        return "ambiguous_match"
    if decision.decision == "unmatched":
        return "unmatched"
    if decision.rejection_reason == "source_agency_mismatch":
        return "source_agency_mismatch"
    if decision.rejection_reason in {
        "available_at_predates_event", "initial_available_at_event_mismatch"
    }:
        return "causal_violation"
    if decision.decision == "conflict":
        raise ValueError("conflicting_duplicate_requires_manual_resolution")
    return "unsupported_semantics"


def _rejection_classification(rejection: Rejection) -> str:
    if rejection.decision == "missing_initial_provenance":
        return "insufficient_initial_release_provenance"
    if "unit" in rejection.rejection_reason:
        return "unsupported_unit"
    return "unsupported_semantics"


def classification_rows(
    events: Sequence[EconomicEvent],
    decisions: Sequence[ImportDecision],
    rejections: Sequence[Rejection],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    represented_initial_ids: set[int] = set()
    for decision in decisions:
        classification = _decision_classification(decision)
        if (
            decision.candidate.publication_state == "initial"
            and decision.matched_economic_event_id is not None
        ):
            represented_initial_ids.add(decision.matched_economic_event_id)
        rows.append({
            "family": decision.candidate.candidate_event_family,
            "classification": classification,
            "economic_event_id": decision.matched_economic_event_id,
            "source_agency": decision.candidate.source_agency,
            "source_event_id": decision.candidate.candidate_source_event_id,
            "reference_period": decision.candidate.candidate_reference_period,
            "publication_state": decision.candidate.publication_state,
            "revision_sequence": decision.candidate.revision_sequence,
            "available_at": decision.candidate.available_at,
            "source_artifact_path": decision.candidate.source_artifact_path,
            "source_artifact_sha256": decision.candidate.source_artifact_sha256,
            "semantic_contract": decision.candidate.semantic_contract,
            "reason": decision.rejection_reason,
        })
    for rejection in rejections:
        rows.append({
            "family": rejection.candidate_event_family,
            "classification": _rejection_classification(rejection),
            "economic_event_id": None,
            "source_agency": rejection.source_family,
            "source_event_id": None,
            "reference_period": rejection.candidate_reference_period,
            "publication_state": None,
            "revision_sequence": None,
            "available_at": None,
            "source_artifact_path": rejection.artifact,
            "source_artifact_sha256": None,
            "semantic_contract": None,
            "reason": rejection.rejection_reason,
        })

    # A missing archive is itself a deterministic fail-closed outcome for each
    # authoritative catalog event; no candidate is silently discarded.
    for event in events:
        if event.event_family not in TARGET_FAMILIES:
            continue
        if event.economic_event_id in represented_initial_ids:
            continue
        if event.event_family in IMPORTABLE_FAMILIES:
            # GDP second/third estimates are already represented by revision
            # candidates. Retail parser failures are represented as rejections.
            continue
        classification = (
            "unsupported_semantics"
            if event.event_family == "FOMC"
            else "insufficient_initial_release_provenance"
        )
        rows.append({
            "family": event.event_family,
            "classification": classification,
            "economic_event_id": event.economic_event_id,
            "source_agency": event.source_agency,
            "source_event_id": event.source_event_id,
            "reference_period": event.reference_period,
            "publication_state": None,
            "revision_sequence": None,
            "available_at": None,
            "source_artifact_path": None,
            "source_artifact_sha256": None,
            "semantic_contract": None,
            "reason": FAMILY_CONTRACTS[event.event_family]["readiness_reason"],
        })
    rows.sort(key=lambda row: (
        TARGET_FAMILIES.index(str(row["family"])),
        str(row["available_at"] or ""),
        int(row["economic_event_id"] or -1),
        str(row["source_artifact_path"] or ""),
        str(row["publication_state"] or ""),
        int(row["revision_sequence"] or -1),
    ))
    return rows


def _compatible(candidate: ReleaseActualCandidate, consensus: Consensus) -> bool:
    return (
        candidate.actual_value_kind == consensus.value_kind
        and candidate.actual_unit == consensus.unit
        and candidate.actual_qualifier == consensus.qualifier
    )


def _fomc_artifact_inventory(repo_root: pathlib.Path) -> dict[str, object]:
    raw_root = repo_root / "EconomicCalendar" / "raw" / "federal_reserve"
    with (raw_root / "manifest_filtered.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        artifacts = list(csv.DictReader(source))
    with (raw_root / "fomc_import_prepared.csv").open(
        newline="", encoding="utf-8-sig"
    ) as source:
        prepared = list(csv.DictReader(source))
    times = sorted(canonical_instant(row["event_timestamp_utc"]) for row in prepared)
    return {
        "source_artifacts_examined": len(artifacts),
        "earliest_retained_release": times[0] if times else None,
        "latest_retained_release": times[-1] if times else None,
    }


def coverage_audit(
    repo_root: pathlib.Path,
    events: Sequence[EconomicEvent],
    consensus: Sequence[Consensus],
    decisions: Sequence[ImportDecision],
    classifications: Sequence[Mapping[str, object]],
    retained: Mapping[str, Sequence[BeaArtifact | CensusArtifact]],
    manifests: Mapping[str, str],
) -> dict[str, object]:
    consensus_by_id = {row.economic_event_id: row for row in consensus}
    family_reports: dict[str, object] = {}
    fomc_inventory = _fomc_artifact_inventory(repo_root)
    for family in TARGET_FAMILIES:
        family_events = [row for row in events if row.event_family == family]
        event_ids = {row.economic_event_id for row in family_events}
        selected_ids = event_ids.intersection(consensus_by_id)
        initial_decisions = [
            row for row in decisions
            if row.candidate.candidate_event_family == family
            and row.candidate.publication_state == "initial"
            and row.decision in {"matched", "duplicate_identical"}
            and row.matched_economic_event_id is not None
        ]
        certified_by_id = {
            int(row.matched_economic_event_id): row for row in initial_decisions
        }
        certified_ids = set(certified_by_id)
        intersection_ids = {
            event_id for event_id in certified_ids.intersection(selected_ids)
            if _compatible(
                certified_by_id[event_id].candidate,
                consensus_by_id[event_id],
            )
        }
        family_classes = [
            str(row["classification"])
            for row in classifications if row["family"] == family
        ]
        class_counts = Counter(family_classes)
        artifacts = retained.get(family, ())
        retained_times = sorted(row.available_at for row in artifacts)
        artifact_inventory = {
            "source_artifacts_examined": len(artifacts),
            "earliest_retained_release": retained_times[0] if retained_times else None,
            "latest_retained_release": retained_times[-1] if retained_times else None,
        }
        if family == "FOMC":
            artifact_inventory = fomc_inventory
        certified_times = sorted(
            row.candidate.available_at for row in initial_decisions
        )
        manifest = manifests.get(family, "")
        family_reports[family] = {
            **FAMILY_CONTRACTS[family],
            **artifact_inventory,
            "catalog_event_count": len(family_events),
            "candidate_observations": len(family_classes),
            "classification_counts": {
                key: class_counts.get(key, 0) for key in CLASSIFICATIONS
            },
            "certified_initial_observations": len(certified_ids),
            "revisions_excluded": class_counts.get("revision_only", 0),
            "unmatched": class_counts.get("unmatched", 0),
            "ambiguous": class_counts.get("ambiguous_match", 0),
            "unsupported_semantics": class_counts.get("unsupported_semantics", 0),
            "provenance_failures": class_counts.get(
                "insufficient_initial_release_provenance", 0
            ),
            "causal_failures": class_counts.get("causal_violation", 0),
            "selected_consensus_rows": len(selected_ids),
            "certified_initial_actual_rows": len(certified_ids),
            "usable_consensus_actual_intersection": len(intersection_ids),
            "actual_only_rows": len(certified_ids - selected_ids),
            "consensus_only_rows": len(selected_ids - certified_ids),
            "earliest_certified_event": certified_times[0] if certified_times else None,
            "latest_certified_event": certified_times[-1] if certified_times else None,
            "manifest_row_count": len(manifest.splitlines()),
            "manifest_sha256": sha256_text(manifest) if manifest else None,
        }
    return {
        "contract": "authoritative_multi_family_initial_actual_phase15_v1",
        "database_interaction": "read_only_catalog_queries_only",
        "production_writes": False,
        "manifest_serialization": "utf8_json_lines_sorted_keys_compact_lf",
        "target_families": list(TARGET_FAMILIES),
        "families": family_reports,
    }


def deterministic_json_lines(rows: Sequence[Mapping[str, object]]) -> str:
    return "".join(
        json.dumps(dict(row), sort_keys=True, separators=(",", ":")) + "\n"
        for row in rows
    )


def write_text(path: pathlib.Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(value, encoding="utf-8")


def prepare(
    database: str,
    repo_root: pathlib.Path,
) -> tuple[Mapping[str, str], str, Mapping[str, object], Mapping[str, str]]:
    candidates, rejections, retained = load_supported_source_corpus(repo_root)
    events, consensus, existing = load_database(database)
    decisions = match_candidates(candidates, events, existing)

    manifests: dict[str, str] = {}
    sql: dict[str, str] = {}
    for family in IMPORTABLE_FAMILIES:
        family_initials = [
            row for row in decisions
            if row.candidate.candidate_event_family == family
            and row.candidate.publication_state == "initial"
        ]
        manifests[family] = deterministic_manifest(manifest_rows(family_initials))
        sql[family] = build_insert_sql([
            row for row in family_initials if row.decision == "matched"
        ])
        upper = sql[family].upper()
        if any(token in upper for token in (
            " UPDATE ", " DELETE ", "ON CONFLICT", "UPSERT"
        )):
            raise ValueError("non_append_only_import_payload")

    classes = classification_rows(events, decisions, rejections)
    classifications = deterministic_json_lines(classes)
    audit = coverage_audit(
        repo_root, events, consensus, decisions, classes, retained, manifests
    )
    return manifests, classifications, audit, sql


def main(argv: Iterable[str] | None = None) -> int:
    default_root = pathlib.Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(
        description="Prepare deterministic Phase 15 initial-actual payloads"
    )
    parser.add_argument("--db", required=True)
    parser.add_argument("--repo-root", type=pathlib.Path, default=default_root)
    parser.add_argument("--output-directory", type=pathlib.Path, required=True)
    args = parser.parse_args(list(argv) if argv is not None else None)

    manifests, classifications, audit, sql = prepare(
        args.db, args.repo_root.resolve()
    )
    output = args.output_directory.resolve()
    names = {"GDP": "gdp", "RETAIL_SALES": "retail-sales"}
    for family, stem in names.items():
        write_text(output / f"{stem}-initial-actual-manifest.jsonl", manifests[family])
        write_text(output / f"{stem}-initial-actual-import.sql", sql[family])
    write_text(output / "multi-family-candidate-classifications.jsonl", classifications)
    write_text(
        output / "multi-family-coverage-audit.json",
        json.dumps(audit, sort_keys=True, indent=2) + "\n",
    )

    print("Phase 15 authoritative multi-family initial-actual readiness")
    for family in TARGET_FAMILIES:
        report = audit["families"][family]
        print(
            f"{family}: {report['readiness']} "
            f"certified={report['certified_initial_observations']} "
            f"intersection={report['usable_consensus_actual_intersection']}"
        )
    print(f"Output: {output}")
    print("RESULT: READ-ONLY PRODUCTION CATALOG / FILESYSTEM ARTIFACTS ONLY")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as error:
        print(f"PHASE15_MULTI_FAMILY_READINESS_FAILED: {error}", file=sys.stderr)
        raise SystemExit(1)
