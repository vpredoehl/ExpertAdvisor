#!/usr/bin/env python3
"""Expected-value-independent raw observation payload generator.

The registered production generator entry points delegate their attestation
mode here.  This process receives one immutable observed artifact, parses only
observed bytes, and emits the class-specific JSON consumed by the trusted
runner.  Expected manifests are intentionally neither accepted nor opened.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import sys
from pathlib import Path

from CampaignOperationsH1EvidenceAuthority import RAW_CLASS_PAYLOADS


def rows(text: str) -> list[dict[str, str]]:
    try:
        reader = csv.DictReader(io.StringIO(text), delimiter="\t")
        if not reader.fieldnames:
            return []
        return [{str(key): ("" if value is None else value if isinstance(value, str)
                            else "\t".join(value))
                 for key, value in row.items() if key is not None}
                for row in reader]
    except csv.Error:
        return []


def first(values: list[dict[str, str]], *names: str) -> str:
    for row in values:
        for name in names:
            if row.get(name):
                return row[name]
    return ""


def payload(evidence_class: str, raw: bytes, requirement_id: str,
            run_id: str) -> dict[str, object]:
    text = raw.decode("utf-8", errors="replace")
    parsed = rows(text)
    pairs = dict(re.findall(r"\b([a-z0-9_]+)=([^\s]+)", text))
    digest = hashlib.sha256(raw).hexdigest()
    receipt = {
        "query_execution_id": os.environ["H1_TRUSTED_GENERATOR_EXECUTION_ID"],
        "requirement_id": requirement_id, "run_id": run_id,
        "observed_digest": digest, "observed_size": len(raw),
    }
    if evidence_class == "runtime":
        value = {"query_contract": first(parsed, "test_source", "query_contract"),
                 "sqlstate": first(parsed, "sqlstate", "actual_sqlstate"),
                 "diagnostic": first(parsed, "diagnostic", "actual_diagnostic"),
                 "object_identity": first(parsed, "object_identity"),
                 "transaction_outcome": first(parsed, "actual_status", "transaction_outcome"),
                 "observation_rows": parsed, "query_receipt": receipt}
    elif evidence_class == "role_security":
        value = {"query_contract": first(parsed, "test_source", "query_contract"),
                 "role_identity": first(parsed, "object_identity", "role_identity"),
                 "sqlstate": first(parsed, "sqlstate", "actual_sqlstate"),
                 "diagnostic": first(parsed, "diagnostic", "actual_diagnostic"),
                 "catalog_rows": parsed, "query_receipt": receipt}
    elif evidence_class == "lock":
        value = {"pg_locks": parsed, "pg_blocking_pids": first(parsed, "blocker_pids", "raw_blocker_pids"),
                 "pg_stat_activity": parsed, "backend_identities": [first(parsed, "first_pid"), first(parsed, "second_pid")],
                 "transaction_outcomes": [first(parsed, "first_outcome"), first(parsed, "second_outcome")],
                 "query_receipt": receipt}
    elif evidence_class in {"acl_origin", "acl_catalog"}:
        value = {"query_id": receipt["query_execution_id"], "catalog_rows": parsed,
                 "owner": first(parsed, "owner"), "raw_acl": first(parsed, "raw_acl", "raw_acl_text"),
                 "origin": first(parsed, "origin", "actual_origin"),
                 ("acldefault_source" if evidence_class == "acl_origin" else "default_acl_source"):
                    first(parsed, "acldefault_source", "default_acl_source", "acldefault_type"),
                 "expanded_acl_rows": parsed}
        if evidence_class == "acl_origin":
            value["deployment_audit"] = first(parsed, "deployment_audit", "actual_diagnostic")
    elif evidence_class == "restore":
        value = {"invocation": first(parsed, "command", "test_source"),
                 "source_cluster_id": first(parsed, "source_cluster_id", "source_database"),
                 "target_cluster_id": first(parsed, "target_cluster_id", "target_database"),
                 "dump_digest": first(parsed, "dump_digest", "artifact_digest") or digest,
                 "pre_audit": first(parsed, "pre_audit", "pre_restore_audit"),
                 "post_audit": first(parsed, "post_audit", "post_restore_audit"),
                 "historical_bytes": text}
    elif evidence_class == "release_build":
        value = {"xcodebuild_identity": first(parsed, "xcodebuild_identity") or pairs.get("xcodebuild_version", ""),
                 "command": first(parsed, "command") or pairs.get("command", ""),
                 "configuration": first(parsed, "configuration") or ("Release" if "Release" in text else ""),
                 "product_identity": first(parsed, "product_identity", "product_path"),
                 "derived_data_identity": first(parsed, "derived_data_identity", "derived_data_path"),
                 "complete_log": text}
    elif evidence_class == "strict_compile":
        value = {"compiler_identity": first(parsed, "compiler_identity") or text.splitlines()[0] if text else "",
                 "command": first(parsed, "command"), "translation_units": first(parsed, "translation_units"),
                 "raw_diagnostics": text}
    elif evidence_class == "checksum":
        value = {"migration_bytes_digest": first(parsed, "migration_bytes_digest", "migration_digest"),
                 "embedded_checksum_source": first(parsed, "embedded_checksum_source", "embedded_checksum"),
                 "hash_implementation": first(parsed, "hash_implementation") or pairs.get("hash_implementation", ""),
                 "ledger_output": text, "replay_output": text}
    elif evidence_class == "worktree":
        value = {"git_identity": first(parsed, "git_identity") or pairs.get("git_identity", ""),
                 "command": first(parsed, "command") or pairs.get("command", ""),
                 "repository_identity": str(Path.cwd()), "status_output": text, "diff_output": text}
    elif evidence_class == "mutation":
        value = {"command": first(parsed, "command") or pairs.get("command", ""),
                 "fixture_identity": requirement_id, "expected_failure": first(parsed, "expected_failure_code"),
                 "actual_failure": first(parsed, "actual_failure_code"), "output_log": text}
    elif evidence_class == "exclusion":
        value = {"scanner_identity": pairs.get("scanner_identity", pairs.get("command", "")),
                 "command": pairs.get("command", ""),
                 "scanned_source_paths": pairs.get("scanned_paths", "").split(",") if pairs.get("scanned_paths") else [],
                 "patterns": pairs.get("patterns", "").split(",") if pairs.get("patterns") else [],
                 "result_rows": parsed or text.splitlines()}
    elif evidence_class == "report_freshness":
        value = {"report_id": requirement_id, "input_artifact_ids": [],
                 "input_digests": [], "output_digest": digest}
    elif evidence_class == "full_pipeline":
        value = {"command": first(parsed, "command") or pairs.get("command", ""),
                 "input_snapshot_ids": [], "validator_execution_ids": [],
                 "graph_digest": digest, "report_digest": digest}
    else:
        raise RuntimeError(f"unsupported-evidence-class:{evidence_class}")
    missing = RAW_CLASS_PAYLOADS[evidence_class] - set(value)
    if missing:
        raise RuntimeError(f"internal-incomplete-payload:{sorted(missing)[0]}")
    return value


def main() -> None:
    if len(sys.argv) != 7 or sys.argv[1] != "generate":
        raise SystemExit("usage: generate EVIDENCE_CLASS REQUIREMENT_ID RUN_ID INPUT OUTPUT")
    evidence_class, requirement_id, run_id = sys.argv[2:5]
    input_text, output_text = sys.argv[5:7]
    output_path = Path(output_text)
    if not os.environ.get("H1_TRUSTED_GENERATOR_EXECUTION_ID"):
        raise SystemExit("trusted-generator-execution-required")
    if input_text != "-":
        raise SystemExit("snapshot-bytes-stdin-required")
    raw = sys.stdin.buffer.read()
    if not raw:
        raise SystemExit("empty-observed-input")
    output_path.write_text(json.dumps(payload(evidence_class, raw, requirement_id, run_id),
                                      sort_keys=True, separators=(",", ":")) + "\n")


if __name__ == "__main__":
    main()
