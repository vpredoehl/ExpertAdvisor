#!/usr/bin/env python3
"""Materialize trusted v2 generator receipts/envelopes for observed H1 artifacts."""
from __future__ import annotations

import csv
import hashlib
import json
import sys
import tempfile
from pathlib import Path

from CampaignOperationsH1ArtifactSnapshot import capture_regular_file
from CampaignOperationsH1EvidenceAuthority import RAW_ENVELOPE_FIELDS
from CampaignOperationsH1TrustedGenerator import TrustedGeneratorRunner

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "Tests/fixtures"


def read(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as source:
        return list(csv.DictReader(source, delimiter="\t"))


def write(path: Path, rows: list[dict[str, str]], fields: list[str] | None = None) -> None:
    if not rows:
        raise RuntimeError(f"empty-materialization:{path.name}")
    temporary = path.with_name(path.name + ".pending")
    with temporary.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields or list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def evidence_classes() -> dict[str, str]:
    result = {row["obligation_id"]: row["evidence_class"]
              for row in read(REGISTRY / "CampaignOperationsH1EvidenceObligations.tsv")}
    # The broad normative runtime obligation is observed by the dedicated
    # exclusion scanner generator class; expected CLI/source policy remains a
    # validator-only input.
    result["H1-H2-H4-EXCLUSION"] = "exclusion"
    result.update({
        "H1-ASSURANCE-FULL-PIPELINE": "full_pipeline",
        "H1-ASSURANCE-LOCK-MUTATIONS": "mutation",
        "H1-ASSURANCE-ACL-MUTATIONS": "mutation",
        "H1-ASSURANCE-GRAPH-MUTATIONS": "mutation",
        "H1-ASSURANCE-MANIFEST-MUTATIONS": "mutation",
        "H1-ASSURANCE-PARSER-CLASSIFICATION": "mutation",
        "H1-ASSURANCE-RESTORE": "restore",
        "H1-ASSURANCE-STRICT-COMPILE": "strict_compile",
        "H1-ASSURANCE-RELEASE-BUILD": "release_build",
        "H1-ASSURANCE-CHECKSUM": "checksum",
        "H1-ASSURANCE-DETERMINISM": "report_freshness",
        "H1-ASSURANCE-WORKTREE": "worktree",
    })
    for row in read(REGISTRY / "CampaignOperationsH1AssuranceControls.tsv"):
        if row["control_kind"] == "mutation_case":
            result[row["control_id"]] = "mutation"
    return result


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("usage: ROOT RUN_ID [FIXTURE_ID ...]")
    evidence_root, run_id = Path(sys.argv[1]), sys.argv[2]
    requested = set(sys.argv[3:])
    fixtures = {row["fixture_id"]: row for row in read(REGISTRY / "CampaignOperationsH1Fixtures.tsv")}
    artifacts = {row["artifact_id"]: row for row in read(REGISTRY / "CampaignOperationsH1Artifacts.tsv")}
    generators = {row["generator_id"]: row for row in read(REGISTRY / "CampaignOperationsH1Generators.tsv")}
    classes = evidence_classes()
    receipt_path = evidence_root / "h1-generator-execution-receipts.tsv"
    envelope_path = evidence_root / "h1-raw-evidence-envelopes.tsv"
    snapshot_path = evidence_root / "h1-generator-output-snapshots.tsv"
    receipts = read(receipt_path); envelopes = read(envelope_path); snapshots = read(snapshot_path)
    envelope_by_requirement = {row["requirement_id"]: row for row in envelopes}
    old_execution_ids = {row["generator_execution_id"] for row in envelopes}
    snapshot_by_artifact = {row["artifact_id"]: row for row in snapshots}
    runner = TrustedGeneratorRunner(ROOT, generators)
    generated = 0
    with tempfile.TemporaryDirectory(prefix="ea-h1-trusted-generators-") as temporary_text:
        temporary = Path(temporary_text)
        for fixture_id, fixture in sorted(fixtures.items()):
            if requested and fixture_id not in requested:
                continue
            requirement_id = fixture["requirement_id"]
            evidence_class = classes.get(requirement_id)
            if evidence_class is None:
                raise RuntimeError(f"missing-evidence-class:{requirement_id}")
            # The catalog generator has already directly queried PostgreSQL and
            # emitted one independently attested envelope per requirement.
            if fixture["generator_id"] == "GEN-ACL-MANIFEST" and requirement_id in envelope_by_requirement:
                continue
            artifact = artifacts[fixture["artifact_id"]]
            source = evidence_root / artifact["path"]
            if not source.is_file():
                if requested:
                    raise RuntimeError(f"missing-observed-artifact:{fixture_id}:{artifact['path']}")
                continue
            input_snapshot = capture_regular_file(source, artifact["artifact_id"], run_id, artifact["path"])
            payload_id = f"ART-GENERATOR-PAYLOAD-{fixture_id}"
            payload_path = temporary / f"{fixture_id}.json"
            observation = runner.execute(
                fixture["generator_id"], run_id,
                ["attest", evidence_class, requirement_id, run_id, "-", str(payload_path)],
                {artifact["artifact_id"]: input_snapshot}, {payload_id: payload_path},
                stdin_bytes=input_snapshot.data)
            runner.validate(observation, run_id, {artifact["artifact_id"]: input_snapshot})
            envelope = runner.raw_envelope(observation, evidence_class, requirement_id, run_id, payload_id)
            envelope_id = f"RAW-{run_id}-{fixture_id}"
            envelope_digest = hashlib.sha256(json.dumps(
                envelope, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
            envelope_by_requirement[requirement_id] = {
                "raw_envelope_id": envelope_id, **envelope, "envelope_digest": envelope_digest}
            receipts.append(observation.receipt)
            snapshot = observation.output_snapshots[payload_id]
            snapshot_by_artifact[payload_id] = {
                "artifact_id": payload_id, "snapshot_id": snapshot.snapshot_id,
                "lexical_path": snapshot.lexical_path, "device": str(snapshot.device),
                "inode": str(snapshot.inode), "size": str(snapshot.size),
                "digest": snapshot.digest, "run_id": run_id}
            generated += 1
    referenced = {row["generator_execution_id"] for row in envelope_by_requirement.values()}
    receipts = [row for row in receipts
                if row.get("generator_execution_id") in referenced or row.get("generator_execution_id") not in old_execution_ids]
    unique_receipts = {row["generator_execution_id"]: row for row in receipts}
    write(receipt_path, sorted(unique_receipts.values(), key=lambda row: row["generator_execution_id"]))
    write(envelope_path, sorted(envelope_by_requirement.values(), key=lambda row: row["requirement_id"]),
          ["raw_envelope_id", *RAW_ENVELOPE_FIELDS, "envelope_digest"])
    write(snapshot_path, sorted(snapshot_by_artifact.values(), key=lambda row: row["artifact_id"]))
    print(f"H1_TRUSTED_EVIDENCE_GENERATION_OK generated={generated} envelopes={len(envelope_by_requirement)}")


if __name__ == "__main__":
    main()
