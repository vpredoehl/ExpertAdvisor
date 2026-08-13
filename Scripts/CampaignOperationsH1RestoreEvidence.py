#!/usr/bin/env python3
"""Validate retained restore A-J semantics and supporting artifact bytes."""
from __future__ import annotations
import csv
import hashlib
import io
import sys
from pathlib import Path

from CampaignOperationsH1ArtifactSnapshot import SnapshotError, capture_regular_file

HEADER = ["format_version", "run_id", "scenario_id", "source_cluster_state",
          "target_cluster_pre_state", "role_creation_source", "dump_type",
          "pre_restore_audit", "restore_attempted", "restore_result",
          "post_role_audit", "post_database_audit", "expected_sqlstate",
          "expected_diagnostic", "actual_sqlstate", "actual_diagnostic",
          "historical_bytes", "final_disposition", "artifact_id", "artifact_path",
          "artifact_digest", "record_digest"]

EXPECTED = {
    "A": ("success", "true", "success", "success", "success", "00000", "restore-supported", "exact", "supported"),
    "B": ("success", "true", "success", "success", "success", "00000", "restore-supported", "exact", "supported"),
    "C": ("success", "true", "success", "success", "success", "00000", "restore-supported", "exact", "supported"),
    "D": ("failed", "false", "not-attempted", "not-applicable", "not-applicable", "42501", "H1A002", "not-applicable", "rejected-before-restore"),
    "E": ("failed", "false", "not-attempted", "not-applicable", "not-applicable", "42501", "H1A003", "not-applicable", "rejected-before-restore"),
    "F": ("failed", "false", "not-attempted", "not-applicable", "not-applicable", "42501", "H1A002", "not-applicable", "rejected-before-restore"),
    "G": ("success", "true", "success", "success", "failed", "42501", "H1A004", "exact", "blocked"),
    "H": ("success", "true", "success", "success", "failed", "42501", "H1A007", "exact", "blocked"),
    "I": ("success", "true", "success", "success", "failed", "42501", "H1A006", "exact", "blocked"),
    "J": ("success", "true", "success", "success", "failed", "42501", "H1A005", "exact", "blocked"),
}


def fail(key: str, detail: str) -> None:
    print(f"H1R001 key={key} stage=restore-runtime-reconciliation detail={detail}", file=sys.stderr)
    raise SystemExit(1)


def validate_snapshot_bytes(runtime_bytes: bytes, artifact_bytes: dict[str, bytes],
                            requested_run: str = "") -> None:
    """Validate restore evidence exclusively from an immutable snapshot bundle."""
    try:
        data = list(csv.reader(io.StringIO(runtime_bytes.decode("utf-8")), delimiter="\t"))
    except (UnicodeDecodeError, csv.Error):
        fail("runtime", "invalid-runtime-bytes")
    if not data or data[0] != HEADER:
        fail("header", "invalid-schema")
    found = set()
    for number, values in enumerate(data[1:], 2):
        if len(values) != len(HEADER) or any(value == "" for value in values):
            fail(str(number), "field-count-or-empty")
        row = dict(zip(HEADER, values)); scenario = row["scenario_id"]
        if scenario not in EXPECTED or scenario in found:
            fail(scenario, "unknown-or-duplicate-scenario")
        found.add(scenario)
        if row["format_version"] != "h1-restore-runtime-v2" or (requested_run and row["run_id"] != requested_run):
            fail(scenario, "stale-version-or-run")
        fields = (row["pre_restore_audit"], row["restore_attempted"], row["restore_result"],
                  row["post_role_audit"], row["post_database_audit"], row["expected_sqlstate"],
                  row["expected_diagnostic"], row["historical_bytes"], row["final_disposition"])
        if fields != EXPECTED[scenario] or row["actual_sqlstate"] != row["expected_sqlstate"] or row["actual_diagnostic"] != row["expected_diagnostic"]:
            fail(scenario, "semantic-outcome-mismatch")
        if row["artifact_id"] != f"ART-RECORD-H1RESTORE{scenario}":
            fail(scenario, "artifact-namespace-mismatch")
        relative = Path(row["artifact_path"])
        if relative.is_absolute() or ".." in relative.parts:
            fail(scenario, "unsafe-artifact-path")
        retained = artifact_bytes.get(relative.as_posix())
        if retained is None or hashlib.sha256(retained).hexdigest() != row["artifact_digest"]:
            fail(scenario, "stale-supporting-artifact")
        computed = hashlib.sha256("\t".join(values[:-1]).encode()).hexdigest()
        if computed != row["record_digest"]:
            fail(scenario, "stale-record-digest")
    if found != set(EXPECTED):
        fail(next(iter(set(EXPECTED) - found), "row-count"), "missing-scenario")


def main() -> None:
    if len(sys.argv) in {4, 5} and sys.argv[1] == "--snapshot-bundle":
        runtime = Path(sys.argv[2]); root = Path(sys.argv[3])
        requested_run = sys.argv[4] if len(sys.argv) == 5 else ""
        snapshot_run = requested_run or "standalone-restore"
        try:
            runtime_snapshot = capture_regular_file(runtime, "ART-SUPPORT-RESTORE", snapshot_run,
                                                    runtime.name)
            _, rows = runtime_snapshot.tsv()
            artifacts = {}
            for row in rows:
                relative = Path(row.get("artifact_path", ""))
                if relative.is_absolute() or ".." in relative.parts:
                    fail(row.get("scenario_id", "row"), "unsafe-artifact-path")
                artifact = capture_regular_file(root / relative,
                                                row.get("artifact_id", "RESTORE-ARTIFACT"),
                                                snapshot_run, relative.as_posix())
                artifacts[relative.as_posix()] = artifact.data
            validate_snapshot_bytes(runtime_snapshot.data, artifacts, requested_run)
            print("H1_RESTORE_RUNTIME_RECONCILIATION_OK scenarios=10")
            return
        except SnapshotError as error:
            fail(error.key, error.detail)
    # The former pathname-reading CLI is retained only as a fail-closed tombstone.
    if len(sys.argv) != 2 or sys.argv[1] != "--reject-pathname-mode":
        print("H1R002 key=restore-pathname-mode stage=restore-runtime-reconciliation "
              "detail=registered-artifact-path-reopen-prohibited-use-snapshot-bundle", file=sys.stderr)
        raise SystemExit(1)
    raise SystemExit(1)
    print("H1_RESTORE_RUNTIME_RECONCILIATION_OK scenarios=10")


if __name__ == "__main__":
    main()
