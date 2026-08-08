#!/usr/bin/env python3
"""Committed Phase H H1 trust-boundary readiness contract.

This module does not create evidence.  It is the single fail-closed predicate
used after the governing authority, executions, snapshots, provenance, and
reports have each been independently validated.
"""
from __future__ import annotations

import csv
from pathlib import Path


READINESS_COMPONENTS = (
    "authority_complete",
    "trusted_generator_execution_complete",
    "raw_envelopes_complete",
    "snapshots_complete",
    "runtime_records_complete",
    "trusted_validator_execution_complete",
    "validator_results_complete",
    "acl_catalog_independent",
    "provenance_graph_complete",
    "report_complete",
    "no_legacy_path_reachable",
)

LEGACY_INVENTORY_FIELDS = (
    "entry_id", "legacy_class", "entry_point", "disposition", "replacement", "diagnostic",
)
ALLOWED_DISPOSITIONS = {"removed", "replaced", "fail_closed"}


class TrustModelError(RuntimeError):
    pass


def evaluate_readiness(components: dict[str, bool]) -> tuple[bool, str]:
    if set(components) != set(READINESS_COMPONENTS):
        missing = sorted(set(READINESS_COMPONENTS) - set(components))
        extra = sorted(set(components) - set(READINESS_COMPONENTS))
        offender = (missing or extra or ["readiness"])[0]
        raise TrustModelError(f"H1T001 readiness-component-schema:{offender}")
    for component in READINESS_COMPONENTS:
        if components[component] is not True:
            return False, f"H1T1{READINESS_COMPONENTS.index(component):02d} {component}=false"
    return True, "H1T199 all-trust-boundaries-complete"


def validate_legacy_inventory(path: Path) -> list[dict[str, str]]:
    try:
        with path.open(newline="") as source:
            reader = csv.DictReader(source, delimiter="\t")
            if tuple(reader.fieldnames or ()) != LEGACY_INVENTORY_FIELDS:
                raise TrustModelError("H1T201 invalid-legacy-path-inventory-schema")
            rows = list(reader)
    except OSError as error:
        raise TrustModelError("H1T201 missing-legacy-path-inventory") from error
    if not rows:
        raise TrustModelError("H1T201 empty-legacy-path-inventory")
    identifiers: set[str] = set()
    for row in rows:
        identifier = row["entry_id"]
        if not identifier or identifier in identifiers:
            raise TrustModelError("H1T202 duplicate-or-empty-legacy-entry")
        identifiers.add(identifier)
        if row["disposition"] not in ALLOWED_DISPOSITIONS:
            raise TrustModelError(f"H1T203 legacy-path-still-reachable:{identifier}")
        if not row["entry_point"] or not row["replacement"] or not row["diagnostic"]:
            raise TrustModelError(f"H1T204 incomplete-legacy-path-disposition:{identifier}")
    return rows


def validator_execution_complete(receipts: list[dict[str, str]], validators: set[str],
                                 results: list[dict[str, str]], run_id: str) -> bool:
    if {row.get("validator_id", "") for row in receipts} != validators:
        return False
    execution_ids = [row.get("validator_execution_id", "") for row in receipts]
    if not execution_ids or len(execution_ids) != len(set(execution_ids)):
        return False
    if any(row.get("version") != "h1-validator-execution-receipt-v2" or
           row.get("run_id") != run_id or row.get("actual_exit_status") != "0" or
           row.get("execution_status") != "completed" for row in receipts):
        return False
    receipt_by_execution = {row["validator_execution_id"]: row for row in receipts}
    if any(row.get("validator_execution_id") not in receipt_by_execution or
           receipt_by_execution[row["validator_execution_id"]].get("validator_id") != row.get("validator_id")
           for row in results):
        return False
    declared = {item for row in receipts for item in row.get("output_validator_result_ids", "").split(",") if item}
    return declared == {row.get("validator_result_id", "") for row in results}


def snapshot_coverage_complete(snapshot_rows: list[dict[str, str]], artifact_ids: set[str],
                               run_id: str) -> bool:
    by_artifact = {row.get("artifact_id", ""): row for row in snapshot_rows}
    if set(by_artifact) != artifact_ids:
        return False
    return all(row.get("run_id") == run_id and row.get("snapshot_id") and
               row.get("lexical_path") and row.get("device", "").isdigit() and
               row.get("inode", "").isdigit() and row.get("size", "").isdigit() and
               len(row.get("digest", "")) == 64 for row in snapshot_rows)


def generator_execution_complete(receipts: list[dict[str, str]], envelopes: list[dict[str, str]],
                                 snapshots: list[dict[str, str]], requirement_ids: set[str],
                                 run_id: str) -> bool:
    receipt_by_id = {row.get("generator_execution_id", ""): row for row in receipts}
    envelope_by_requirement = {row.get("requirement_id", ""): row for row in envelopes}
    snapshot_ids = {row.get("snapshot_id", "") for row in snapshots}
    if (len(receipt_by_id) != len(receipts) or len(envelope_by_requirement) != len(envelopes) or
            set(envelope_by_requirement) != requirement_ids or not snapshot_ids):
        return False
    referenced: set[str] = set()
    for envelope in envelopes:
        execution_id = envelope.get("generator_execution_id", "")
        receipt = receipt_by_id.get(execution_id)
        referenced.add(execution_id)
        if (receipt is None or receipt.get("version") != "h1-generator-execution-receipt-v2" or
                receipt.get("run_id") != run_id or receipt.get("actual_exit_status") != "0" or
                receipt.get("execution_status") != "completed"):
            return False
        bound_snapshots = {value for value in receipt.get("output_snapshot_ids", "").split(",") if value}
        if not bound_snapshots or not bound_snapshots.issubset(snapshot_ids):
            return False
    return referenced == set(receipt_by_id)
