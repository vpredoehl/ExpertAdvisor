#!/usr/bin/env python3
"""Generate and authenticate ADR-0019B H1 workflow-lock evidence."""

from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

RUNTIME_HEADER = [
    "format_version", "run_id", "h1lock_id", "generator_id", "generator_version",
    "implementation_path", "entry_point", "emitted_runtime_record_id",
    "output_artifact_id", "raw_artifact_id", "first_operation", "second_operation",
    "workflow_classification",
    "first_pid", "second_pid", "first_application", "second_application",
    "first_held_lock_set", "second_held_lock_set", "requested_lock", "lock_type",
    "lock_identity", "lock_mode", "first_granted", "second_waiting",
    "blocking_pids_first", "blocking_pids_second", "complete_wait_for_graph",
    "permitted_wait_direction", "observed_wait_direction",
    "prohibited_reverse_direction", "reverse_wait_observed", "cycle_detected",
    "first_activity_state", "second_activity_state", "first_transaction_outcome",
    "second_transaction_outcome", "lock_release_verified", "partial_evidence_count",
    "cleanup_result", "raw_artifact_path", "raw_artifact_digest", "record_digest",
]

FORMAT_VERSION = "h1-lock-runtime-v3"
RAW_FORMAT_VERSION = "h1-lock-raw-v3"
GENERATOR_ID = "GEN-LOCK"
GENERATOR_VERSION = "h1-generator-registry-v2"
IMPLEMENTATION_PATH = "Scripts/CampaignOperationsH1LockEvidence.py"
ENTRY_POINT = "generate"


def fail(key: str) -> None:
    print(f"H1L001 key={key} stage=lock-runtime-reconciliation", file=sys.stderr)
    raise SystemExit(1)


def read_tsv(path: Path) -> list[list[str]]:
    try:
        with path.open(newline="") as source:
            return list(csv.reader(source, delimiter="\t"))
    except OSError:
        fail(f"missing:{path}")


def read_matrix(path: Path) -> dict[str, dict[str, str]]:
    rows = read_tsv(path)
    header = ["test_id", "operation_pair", "first_path", "second_path",
              "shared_boundaries", "first_conflict", "permitted_wait",
              "prohibited_wait", "expected_result", "requirement_id",
              "first_workflow", "second_workflow", "expected_classification",
              "seam_disposition"]
    if not rows or rows[0] != header:
        fail("matrix-header")
    result: dict[str, dict[str, str]] = {}
    for values in rows[1:]:
        if len(values) != len(header):
            fail("matrix-fields")
        row = dict(zip(header, values))
        identifier = row["test_id"]
        if identifier in result:
            fail(f"matrix-duplicate:{identifier}")
        if row["expected_classification"] not in {
            "full_workflow_vs_full_workflow", "accepted_final_seam"
        }:
            fail(f"classification:{identifier}:{row['expected_classification']}")
        result[identifier] = row
    return result


def identity(row: list[str]) -> str:
    if row[4] == "advisory":
        return f"{row[8]}/{row[9]}"
    if row[4] == "transactionid":
        return row[15] or "unknown"
    return row[5] or "unknown"


def lock_tuple(row: list[str]) -> str:
    return f"{row[4]}:{identity(row)}:{row[6]}"


def split_pids(value: str) -> list[str]:
    try:
        return sorted({part for part in value.split(",") if part}, key=int)
    except ValueError:
        fail("blocking-pid")


def has_cycle(edges: dict[str, list[str]]) -> bool:
    def walk(origin: str, node: str, path: set[str]) -> bool:
        for following in edges.get(node, []):
            if following == origin or following in path:
                return True
            if walk(origin, following, path | {following}):
                return True
        return False
    return any(walk(node, node, {node}) for node in edges)


def parse_raw(path: Path, expected_id: str, expected_run_id: str) -> tuple[dict[str, str], list[list[str]]]:
    rows = read_tsv(path)
    if not rows or rows[0] != ["format", RAW_FORMAT_VERSION]:
        fail(f"raw-header:{expected_id}")
    meta_rows = [row for row in rows[1:] if row and row[0] == "meta"]
    catalog = [row for row in rows[1:] if row and row[0] == "catalog"]
    if len(meta_rows) != 1 or len(meta_rows[0]) != 17:
        fail(f"raw-meta:{expected_id}")
    if not catalog or any(len(row) != 16 for row in catalog):
        fail(f"raw-catalog-fields:{expected_id}")
    keys = ["record_type", "run_id", "h1lock_id", "raw_artifact_id",
            "generator_id", "generator_version", "first_pid", "second_pid",
            "first_application", "second_application", "first_outcome",
            "second_outcome", "release_verified", "partial_evidence_count",
            "cleanup_result", "evidence_query", "release_query"]
    meta = dict(zip(keys, meta_rows[0]))
    if meta["run_id"] != expected_run_id:
        fail(f"raw-run:{expected_id}")
    if (meta["h1lock_id"] != expected_id or
            meta["raw_artifact_id"] != f"ART-LOCK-RAW-{expected_id}" or
            meta["generator_id"] != GENERATOR_ID or
            meta["generator_version"] != GENERATOR_VERSION or
            any(row[1] != expected_id for row in catalog)):
        fail(f"raw-id:{expected_id}")
    return meta, catalog


def normalize(identifier: str, contract: dict[str, str], meta: dict[str, str],
              catalog: list[list[str]], raw_path: str, digest: str,
              run_id: str) -> list[str]:
    first_pid, second_pid = meta["first_pid"], meta["second_pid"]
    first_app, second_app = meta["first_application"], meta["second_application"]
    if not first_pid.isdigit() or not second_pid.isdigit() or first_pid == second_pid:
        fail(f"pid:{identifier}")
    if first_app == second_app:
        fail(f"application:{identifier}")
    if any(row[2] not in {first_app, second_app} or row[3] not in {first_pid, second_pid}
           for row in catalog):
        fail(f"raw-identity:{identifier}")
    first_rows = [row for row in catalog if row[2] == first_app and row[3] == first_pid]
    second_rows = [row for row in catalog if row[2] == second_app and row[3] == second_pid]
    if not first_rows or not second_rows:
        fail(f"raw-pid:{identifier}")
    held_first = sorted({lock_tuple(row) for row in first_rows if row[7] == "t"})
    held_second = sorted({lock_tuple(row) for row in second_rows if row[7] == "t"})
    requested_rows = sorted([row for row in catalog if row[7] == "f"], key=lock_tuple)
    requested = ";".join(lock_tuple(row) for row in requested_rows) or "none:none:none"
    requested_first = requested_rows[0] if requested_rows else None
    first_blockers = sorted({pid for row in first_rows for pid in split_pids(row[10])}, key=int)
    second_blockers = sorted({pid for row in second_rows for pid in split_pids(row[10])}, key=int)
    edges = {first_pid: first_blockers, second_pid: second_blockers}
    graph = ";".join(f"{pid}>{blocker}" for pid in sorted(edges, key=int)
                     for blocker in edges[pid]) or "none"
    first_blocked_by_second = second_pid in first_blockers
    second_blocked_by_first = first_pid in second_blockers
    # Direction labels name the two raw backend positions.  The two raw
    # directions are always distinct; the permitted direction is consulted
    # only after observation, as the acceptance expectation.
    if contract["permitted_wait"] == "second->first":
        first_to_second = contract["prohibited_wait"]
        second_to_first = contract["permitted_wait"]
    elif contract["permitted_wait"] == "none":
        first_to_second, second_to_first = "first->second", "second->first"
    else:
        first_to_second = contract["permitted_wait"]
        second_to_first = contract["prohibited_wait"]
    if first_blocked_by_second and second_blocked_by_first:
        fail(f"mutual-blocking:{identifier}")
    if first_blocked_by_second:
        observed = first_to_second
    elif second_blocked_by_first:
        observed = second_to_first
    else:
        observed = "none"
    if observed == contract["prohibited_wait"] and observed != "none":
        fail(f"prohibited-direction:{identifier}")
    state = lambda rows: sorted({f"{row[11]}:{row[12]}:{row[13]}" for row in rows})[0]
    if meta["release_verified"] not in {"true", "false"}:
        fail(f"release:{identifier}")
    if not meta["partial_evidence_count"].isdigit():
        fail(f"partial-evidence:{identifier}")
    boolean = lambda value: "true" if value else "false"
    values = [FORMAT_VERSION, run_id, identifier, GENERATOR_ID, GENERATOR_VERSION,
            IMPLEMENTATION_PATH, ENTRY_POINT, f"RT-{identifier}",
            f"ART-RECORD-{identifier}", f"ART-LOCK-RAW-{identifier}",
            contract["first_workflow"], contract["second_workflow"],
            contract["expected_classification"], first_pid, second_pid, first_app,
            second_app, ",".join(held_first) or "none", ",".join(held_second) or "none",
            requested, requested_first[4] if requested_first else "none",
            identity(requested_first) if requested_first else "none",
            requested_first[6] if requested_first else "none",
            boolean(any(row[7] == "t" for row in first_rows)),
            boolean(bool(second_blockers) or any(row[7] == "f" for row in second_rows)),
            ",".join(first_blockers) or "none", ",".join(second_blockers) or "none",
            graph, contract["permitted_wait"], observed, contract["prohibited_wait"],
            boolean(observed == contract["prohibited_wait"] and observed != "none"),
            boolean(has_cycle(edges)), state(first_rows), state(second_rows),
            meta["first_outcome"], meta["second_outcome"], meta["release_verified"],
            meta["partial_evidence_count"], meta["cleanup_result"], raw_path, digest]
    record_digest = hashlib.sha256("\t".join(values).encode()).hexdigest()
    return values + [record_digest]


def generate(args: list[str]) -> None:
    if len(args) != 7:
        raise SystemExit("usage: generate MATRIX CATALOG META OUTPUT RAW_ROOT RUN_ID")
    matrix_path, catalog_path, meta_path, output_path, raw_root = map(Path, args[1:6])
    run_id = args[6]
    if not run_id or any(character.isspace() for character in run_id):
        fail("run-id")
    contracts = read_matrix(matrix_path)
    catalog_rows, meta_rows = read_tsv(catalog_path), read_tsv(meta_path)
    raw_root.mkdir(parents=True, exist_ok=True)
    output: list[list[str]] = []
    for identifier in sorted(contracts):
        selected_catalog = [row for row in catalog_rows if row and row[0] == identifier]
        selected_meta = [row for row in meta_rows if row and row[0] == identifier]
        if len(selected_meta) != 1 or len(selected_meta[0]) != 12:
            fail(f"source-meta:{identifier}")
        raw_file = raw_root / f"{identifier}.tsv"
        with raw_file.open("w", newline="") as target:
            writer = csv.writer(target, delimiter="\t", lineterminator="\n")
            writer.writerow(["format", RAW_FORMAT_VERSION])
            writer.writerow(["meta", run_id, identifier,
                             f"ART-LOCK-RAW-{identifier}", GENERATOR_ID,
                             GENERATOR_VERSION] + selected_meta[0][1:])
            writer.writerows([["catalog"] + row for row in selected_catalog])
        digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        meta, raw_catalog = parse_raw(raw_file, identifier, run_id)
        output.append(normalize(identifier, contracts[identifier], meta, raw_catalog,
                                f"raw-lock/{identifier}.tsv", digest, run_id))
    with output_path.open("w", newline="") as target:
        writer = csv.writer(target, delimiter="\t", lineterminator="\n")
        writer.writerow(RUNTIME_HEADER)
        writer.writerows(output)


def validate(args: list[str]) -> None:
    if len(args) not in {4, 5}:
        raise SystemExit("usage: validate MATRIX RUNTIME ARTIFACT_ROOT [RUN_ID]")
    contracts = read_matrix(Path(args[1]))
    runtime_path, artifact_root = Path(args[2]), Path(args[3])
    rows = read_tsv(runtime_path)
    if not rows or rows[0] != RUNTIME_HEADER:
        fail("runtime-header")
    requested_run_id = args[4] if len(args) == 5 else ""
    found: set[str] = set()
    for values in rows[1:]:
        if len(values) != len(RUNTIME_HEADER):
            fail(f"fields:{len(values)}")
        row = dict(zip(RUNTIME_HEADER, values))
        identifier = row["h1lock_id"]
        if identifier not in contracts:
            fail(f"unknown:{identifier}")
        if identifier in found:
            fail(f"duplicate:{identifier}")
        found.add(identifier)
        if row["format_version"] != FORMAT_VERSION:
            fail(f"version:{identifier}")
        if requested_run_id and row["run_id"] != requested_run_id:
            fail(f"stale-run:{identifier}")
        if (row["generator_id"] != GENERATOR_ID or
                row["generator_version"] != GENERATOR_VERSION or
                row["implementation_path"] != IMPLEMENTATION_PATH or
                row["entry_point"] != ENTRY_POINT or
                row["emitted_runtime_record_id"] != f"RT-{identifier}" or
                row["output_artifact_id"] != f"ART-RECORD-{identifier}" or
                row["raw_artifact_id"] != f"ART-LOCK-RAW-{identifier}"):
            fail(f"generator-identity:{identifier}")
        raw_relative = row["raw_artifact_path"]
        if raw_relative.startswith("/") or ".." in Path(raw_relative).parts:
            fail(f"raw-path:{identifier}")
        raw_file = artifact_root / raw_relative
        if not raw_file.is_file():
            fail(f"missing-raw:{identifier}")
        digest = hashlib.sha256(raw_file.read_bytes()).hexdigest()
        if digest != row["raw_artifact_digest"]:
            fail(f"raw-digest:{identifier}")
        meta, catalog = parse_raw(raw_file, identifier, row["run_id"])
        expected = normalize(identifier, contracts[identifier], meta, catalog,
                             raw_relative, digest, row["run_id"])
        for index, (actual, wanted) in enumerate(zip(values, expected)):
            if actual != wanted:
                fail(f"normalized:{identifier}:{RUNTIME_HEADER[index]}")
    missing = sorted(set(contracts) - found)
    if missing:
        fail(f"missing:{missing[0]}")
    print(f"H1_LOCK_RUNTIME_RECONCILIATION_OK rows={len(found)}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        raise SystemExit("expected generate or validate")
    if len(sys.argv) == 8 and sys.argv[1:3] == ["generate", "attest"]:
        from CampaignOperationsH1EvidencePayloadGenerator import main as attest
        sys.argv = [sys.argv[0], "generate", *sys.argv[3:]]
        attest()
        raise SystemExit(0)
    {"generate": generate, "validate": validate}.get(sys.argv[1],
        lambda _: (_ for _ in ()).throw(SystemExit("unknown mode")))(sys.argv[1:])
