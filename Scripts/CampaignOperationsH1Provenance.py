#!/usr/bin/env python3
"""Exact bidirectional materialized provenance-chain validator for H1."""
from __future__ import annotations

CHAIN_FIELDS = (
    "normative_clause_id", "requirement_id", "obligation_id",
    "generator_execution_id", "generator_receipt_id", "raw_envelope_id",
    "snapshot_id", "runtime_record_id", "validator_execution_id",
    "validator_receipt_id", "validator_result_id", "report_entry_id",
    "generated_report_id", "run_id",
)
NODE_SEQUENCE = CHAIN_FIELDS[:-1]
EDGE_FIELDS = ("source_type", "source_id", "edge_type", "target_type", "target_id", "run_id")


class ProvenanceError(RuntimeError):
    pass


def validate_provenance(chains: list[dict[str, str]], edges: list[dict[str, str]],
                        obligation_ids: set[str], run_id: str,
                        required_clause_obligation_pairs: set[tuple[str, str]] | None = None) -> None:
    if not chains:
        raise ProvenanceError("H1P001 missing-provenance-chains")
    if any(tuple(row) != CHAIN_FIELDS for row in chains):
        raise ProvenanceError("H1P002 invalid-provenance-chain-schema")
    by_obligation: dict[str, list[dict[str, str]]] = {}
    actual_pairs: set[tuple[str, str]] = set()
    for row in chains:
        obligation = row["obligation_id"]
        if row["run_id"] != run_id or any(not row[field] for field in NODE_SEQUENCE):
            raise ProvenanceError(f"H1P004 incomplete-or-stale-chain:{obligation}")
        pair = (row["normative_clause_id"], obligation)
        if pair in actual_pairs:
            raise ProvenanceError(f"H1P003 duplicate-clause-obligation-chain:{obligation}")
        actual_pairs.add(pair)
        by_obligation.setdefault(obligation, []).append(row)
    if set(by_obligation) != obligation_ids:
        offender = sorted(set(by_obligation) ^ obligation_ids)[0]
        raise ProvenanceError(f"H1P005 obligation-chain-set-mismatch:{offender}")
    if required_clause_obligation_pairs is not None and actual_pairs != required_clause_obligation_pairs:
        offender = sorted(actual_pairs ^ required_clause_obligation_pairs)[0]
        raise ProvenanceError(f"H1P005 clause-obligation-chain-set-mismatch:{offender[1]}")
    actual: set[tuple[str, str, str, str, str, str]] = set()
    for edge in edges:
        if tuple(edge) != EDGE_FIELDS:
            raise ProvenanceError("H1P006 invalid-provenance-edge-schema")
        item = tuple(edge[field] for field in EDGE_FIELDS)
        if item in actual:
            raise ProvenanceError("H1P007 duplicate-provenance-edge")
        actual.add(item)
    expected: set[tuple[str, str, str, str, str, str]] = set()
    for row in chains:
        for source_type, target_type in zip(NODE_SEQUENCE, NODE_SEQUENCE[1:]):
            source_id, target_id = row[source_type], row[target_type]
            expected.add((source_type, source_id, f"{source_type}_to_{target_type}",
                          target_type, target_id, run_id))
            expected.add((target_type, target_id, f"{target_type}_to_{source_type}",
                          source_type, source_id, run_id))
    if actual != expected:
        offender = next(iter(expected - actual or actual - expected))
        detail = "missing" if offender in expected - actual else "extra"
        raise ProvenanceError(f"H1P008 {detail}-provenance-edge:{offender[1]}")
