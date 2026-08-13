#!/usr/bin/env python3
"""Trusted generator observer and v2 raw-envelope materializer."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from CampaignOperationsH1ArtifactSnapshot import ArtifactSnapshot
from CampaignOperationsH1EvidenceAuthority import RAW_CLASS_PAYLOADS, RAW_ENVELOPE_FIELDS
from CampaignOperationsH1TrustedRunner import (
    ExecutionObservation, RunnerError, TrustedValidatorRunner,
)

GENERATOR_RECEIPT_VERSION = "h1-generator-execution-receipt-v2"


@dataclass(frozen=True)
class GeneratorObservation:
    receipt: dict[str, str]
    stdout: bytes
    stderr: bytes
    output_snapshots: dict[str, ArtifactSnapshot]
    _validator_observation: ExecutionObservation


class TrustedGeneratorRunner:
    """Reuse the audited subprocess observer while exposing generator identities."""

    def __init__(self, repository_root: Path, generators: dict[str, dict[str, str]]):
        translated = {
            generator_id: {
                "version": row["version"], "validator_id": generator_id,
                "implementation": row["implementation"], "entry_point": row["entry_point"],
                "executable_required": row["executable_required"],
            }
            for generator_id, row in generators.items()
        }
        self._runner = TrustedValidatorRunner(
            repository_root, translated, execution_prefix="GEXEC",
            execution_environment="H1_TRUSTED_GENERATOR_EXECUTION_ID")
        self._generators = generators
        self._original: dict[str, ExecutionObservation] = {}

    @staticmethod
    def _translate(original: dict[str, str]) -> dict[str, str]:
        return {
            "version": GENERATOR_RECEIPT_VERSION,
            "generator_execution_id": original["validator_execution_id"],
            "generator_id": original["validator_id"],
            "generator_version": original["validator_version"],
            "implementation_path": original["implementation_path"],
            "implementation_digest": original["implementation_digest"],
            "entry_point": original["entry_point"],
            "invocation_contract": original["invocation_contract"],
            "process_id": original["process_id"],
            "process_start_identity": original["process_start_identity"],
            "observer_process_id": original["observer_process_id"],
            "run_id": original["run_id"],
            "wall_start_time": original["wall_start_time"],
            "wall_completion_time": original["wall_completion_time"],
            "monotonic_start_ns": original["monotonic_start_ns"],
            "monotonic_completion_ns": original["monotonic_completion_ns"],
            "input_artifact_ids": original["input_artifact_ids"],
            "input_snapshot_ids": original["input_snapshot_ids"],
            "input_artifact_digests": original["input_artifact_digests"],
            "output_artifact_ids": original["output_artifact_ids"],
            "output_snapshot_ids": original["output_snapshot_ids"],
            "output_artifact_digests": original["output_artifact_digests"],
            "stdout_digest": original["stdout_digest"], "stderr_digest": original["stderr_digest"],
            "actual_exit_status": original["actual_exit_status"],
            "execution_status": original["execution_status"],
            "attestation_digest": original["receipt_digest"],
        }

    def execute(self, generator_id: str, run_id: str, command_arguments: list[str],
                input_artifacts: dict[str, str | ArtifactSnapshot],
                declared_outputs: dict[str, Path], stdin_bytes: bytes | None = None) -> GeneratorObservation:
        observed = self._runner.execute(generator_id, run_id, command_arguments, input_artifacts,
                                        declared_outputs, set(), stdin_bytes=stdin_bytes)
        receipt = self._translate(observed.receipt)
        execution_id = receipt["generator_execution_id"]
        self._original[execution_id] = observed
        return GeneratorObservation(receipt, observed.stdout, observed.stderr,
                                    observed.output_snapshots, observed)

    def validate(self, observation: GeneratorObservation, run_id: str,
                 input_artifacts: dict[str, str | ArtifactSnapshot]) -> None:
        execution_id = observation.receipt.get("generator_execution_id", "")
        original = self._original.get(execution_id)
        if original is None or observation.receipt != self._translate(original.receipt):
            raise RunnerError("invalid-generator-receipt-attestation")
        self._runner.validate(original.receipt, run_id, input_artifacts,
                              observation.output_snapshots, set())

    def raw_envelope(self, observation: GeneratorObservation, evidence_class: str,
                     requirement_id: str, run_id: str, payload_artifact_id: str) -> dict[str, str]:
        if evidence_class not in RAW_CLASS_PAYLOADS:
            raise RunnerError("unknown-evidence-class")
        snapshot = observation.output_snapshots.get(payload_artifact_id)
        if snapshot is None:
            raise RunnerError("missing-generator-payload-snapshot")
        try:
            payload = json.loads(snapshot.text())
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise RunnerError("invalid-generator-json-payload") from error
        if not isinstance(payload, dict) or RAW_CLASS_PAYLOADS[evidence_class] - set(payload):
            raise RunnerError("incomplete-class-specific-generator-payload")
        if {"expected", "actual", "comparison", "comparison_result", "success", "status"}.intersection(payload):
            raise RunnerError("self-asserted-generator-result")
        receipt = observation.receipt
        row = {
            "evidence_version": "h1-raw-execution-evidence-v2", "evidence_class": evidence_class,
            "requirement_id": requirement_id, "run_id": run_id,
            "implementation_version": receipt["generator_version"],
            "invocation_contract": receipt["invocation_contract"],
            "tool_identity": receipt["implementation_path"], "tool_digest": receipt["implementation_digest"],
            "process_id": receipt["process_id"], "monotonic_start_ns": receipt["monotonic_start_ns"],
            "monotonic_completion_ns": receipt["monotonic_completion_ns"],
            "actual_exit_status": receipt["actual_exit_status"],
            "input_snapshot_ids": receipt["input_snapshot_ids"],
            "input_digests": receipt["input_artifact_digests"],
            "output_artifact_ids": receipt["output_artifact_ids"],
            "output_digests": receipt["output_artifact_digests"],
            "stdout": observation.stdout.decode("utf-8", errors="replace"),
            "stderr": observation.stderr.decode("utf-8", errors="replace"),
            "payload_json": json.dumps(payload, sort_keys=True, separators=(",", ":")),
            "generator_execution_id": receipt["generator_execution_id"],
        }
        if list(row) != RAW_ENVELOPE_FIELDS:
            raise RunnerError("raw-envelope-schema-internal-error")
        return row
