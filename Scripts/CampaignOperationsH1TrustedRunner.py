#!/usr/bin/env python3
"""Trusted subprocess observer and receipt attestor for H1 validators.

Validator implementations produce result artifacts.  This runner alone owns
the transient attestation key, observes the child process, captures its actual
status/output/timing, checks declared artifacts, and signs the receipt.  The
key is never passed to the validator child or stored in the repository.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import subprocess
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from CampaignOperationsH1ArtifactSnapshot import ArtifactSnapshot, SnapshotError, capture_regular_file


RECEIPT_VERSION = "h1-validator-execution-receipt-v2"
RECEIPT_FIELDS = [
    "version", "validator_execution_id", "validator_id", "validator_version",
    "implementation_path", "implementation_digest", "entry_point", "invocation_contract",
    "process_id", "process_start_identity", "observer_process_id", "run_id",
    "wall_start_time", "wall_completion_time", "monotonic_start_ns", "monotonic_completion_ns",
    "input_artifact_ids", "input_snapshot_ids", "input_artifact_digests",
    "output_artifact_ids", "output_snapshot_ids", "output_artifact_digests",
    "output_validator_result_ids", "stdout_digest", "stderr_digest", "actual_exit_status",
    "execution_status", "receipt_digest",
]


class RunnerError(RuntimeError):
    pass


def _snapshot(path: Path, artifact_id: str, run_id: str) -> ArtifactSnapshot:
    try:
        return capture_regular_file(path, artifact_id, run_id, path.name)
    except SnapshotError as error:
        raise RunnerError(f"snapshot-binding-failed:{error.detail}") from error


def _bound_digest(value: str | ArtifactSnapshot | Path, artifact_id: str, run_id: str) -> str:
    if isinstance(value, ArtifactSnapshot):
        if value.run_id != run_id:
            raise RunnerError(f"stale-artifact-snapshot:{artifact_id}")
        return value.digest
    if isinstance(value, Path):
        return _snapshot(value, artifact_id, run_id).digest
    if not re.fullmatch(r"[0-9a-f]{64}", value):
        raise RunnerError(f"invalid-input-artifact-digest:{artifact_id}")
    return value


def _bound_snapshot_id(value: str | ArtifactSnapshot | Path, artifact_id: str, run_id: str) -> str:
    if isinstance(value, ArtifactSnapshot):
        if value.run_id != run_id:
            raise RunnerError(f"stale-artifact-snapshot:{artifact_id}")
        return value.snapshot_id
    if isinstance(value, Path):
        return _snapshot(value, artifact_id, run_id).snapshot_id
    # Digest-only inputs are supported for isolated runner unit tests but are
    # explicitly not descriptor-bound and therefore cannot satisfy readiness.
    return "UNBOUND-DIGEST-ONLY"


def _payload(row: dict[str, str]) -> bytes:
    return json.dumps({field: row[field] for field in RECEIPT_FIELDS[:-1]},
                      sort_keys=True, separators=(",", ":")).encode()


@dataclass(frozen=True)
class ExecutionObservation:
    receipt: dict[str, str]
    stdout: bytes
    stderr: bytes
    output_snapshots: dict[str, ArtifactSnapshot]


class TrustedValidatorRunner:
    def __init__(self, repository_root: Path, validators: dict[str, dict[str, str]],
                 max_age_seconds: int = 3600, execution_prefix: str = "VEXEC",
                 execution_environment: str = "H1_TRUSTED_VALIDATOR_EXECUTION_ID"):
        self.repository_root = repository_root
        self.validators = validators
        self.max_age_seconds = max_age_seconds
        self.execution_prefix = execution_prefix
        self.execution_environment = execution_environment
        self.__key = os.urandom(32)
        self._execution_ids: set[str] = set()
        self._validated_execution_ids: set[str] = set()

    def _registered(self, validator_id: str) -> tuple[dict[str, str], Path]:
        registry = self.validators.get(validator_id)
        if registry is None:
            raise RunnerError("unregistered-validator")
        implementation_text = registry.get("implementation", "")
        implementation = self.repository_root / implementation_text
        if (not implementation_text or Path(implementation_text).is_absolute() or
                ".." in Path(implementation_text).parts or not implementation.is_file()):
            raise RunnerError("invalid-registered-implementation")
        if not registry.get("entry_point") or registry.get("executable_required") != "true":
            raise RunnerError("invalid-registered-entry-point")
        return registry, implementation

    def execute(self, validator_id: str, run_id: str, command_arguments: list[str],
                input_artifacts: dict[str, str | ArtifactSnapshot], declared_outputs: dict[str, Path],
                expected_validator_result_ids: set[str], stdin_bytes: bytes | None = None) -> ExecutionObservation:
        registry, implementation = self._registered(validator_id)
        execution_id = f"{self.execution_prefix}-{run_id}-{validator_id}-{uuid.uuid4().hex}"
        if execution_id in self._execution_ids:
            raise RunnerError("duplicate-execution-id")
        self._execution_ids.add(execution_id)
        implementation_snapshot = _snapshot(implementation, f"IMPL-{validator_id}", run_id)
        parent_directories = {path.parent for path in declared_outputs.values()}
        before_paths: set[tuple[str, str]] = set()
        for directory in parent_directories:
            if directory.is_dir():
                with os.scandir(directory) as entries:
                    before_paths.update((str(directory), entry.name) for entry in entries)
        command = [str(implementation), registry["entry_point"], *command_arguments]
        environment = {key: value for key, value in os.environ.items()
                       if key not in {"H1_VALIDATOR_RECEIPT_KEY", "H1_RECEIPT_SIGNING_KEY"}}
        environment[self.execution_environment] = execution_id
        wall_start = datetime.now(timezone.utc)
        monotonic_start = time.monotonic_ns()
        process = subprocess.Popen(command, cwd=self.repository_root, env=environment,
                                   stdin=subprocess.PIPE if stdin_bytes is not None else None,
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        process_identity = ""
        try:
            identity = subprocess.run(["ps", "-o", "lstart=", "-p", str(process.pid)],
                                      capture_output=True, text=True, check=False)
            process_identity = identity.stdout.strip()
            stdout, stderr = process.communicate(input=stdin_bytes)
        finally:
            monotonic_completion = time.monotonic_ns()
            wall_completion = datetime.now(timezone.utc)
        if not process_identity:
            raise RunnerError("unobservable-process-identity")
        if process.returncode is None:
            raise RunnerError("missing-actual-exit-status")
        if process.returncode != 0:
            diagnostic = stderr.decode("utf-8", errors="replace").strip().replace("\n", " | ")
            raise RunnerError(f"validator-exited-nonzero:{process.returncode}:{diagnostic[:1000]}")
        output_ids, output_digests = [], []
        output_snapshots: dict[str, ArtifactSnapshot] = {}
        actual_result_ids: set[str] = set()
        for artifact_id, path in sorted(declared_outputs.items()):
            try:
                output_snapshot = _snapshot(path, artifact_id, run_id)
            except RunnerError as error:
                if "artifact-open-no-follow-failed" in str(error):
                    raise RunnerError(f"missing-declared-output:{artifact_id}") from error
                raise
            if output_snapshot.file_type != "regular":
                raise RunnerError(f"missing-declared-output:{artifact_id}")
            output_snapshots[artifact_id] = output_snapshot
            output_ids.append(artifact_id); output_digests.append(output_snapshot.digest)
            if path.suffix == ".tsv":
                header, *rows = output_snapshot.text().splitlines()
                fields = header.split("\t")
                if "validator_result_id" in fields:
                    index = fields.index("validator_result_id")
                    for row in rows:
                        values = row.split("\t")
                        if len(values) == len(fields): actual_result_ids.add(values[index])
        declared_paths = {(str(path.parent), path.name) for path in declared_outputs.values()}
        for directory in parent_directories:
            with os.scandir(directory) as entries:
                for entry in entries:
                    identity = (str(directory), entry.name)
                    if (entry.is_file(follow_symlinks=False) and identity not in declared_paths and
                            identity not in before_paths):
                        raise RunnerError(f"undeclared-output:{entry.name}")
        if actual_result_ids != expected_validator_result_ids:
            raise RunnerError("output-validator-result-set-mismatch")
        receipt = {
            "version": RECEIPT_VERSION, "validator_execution_id": execution_id,
            "validator_id": validator_id, "validator_version": registry["version"],
            "implementation_path": registry["implementation"], "implementation_digest": implementation_snapshot.digest,
            "entry_point": registry["entry_point"],
            "invocation_contract": json.dumps(command, separators=(",", ":")),
            "process_id": str(process.pid), "process_start_identity": process_identity,
            "observer_process_id": str(os.getpid()), "run_id": run_id,
            "wall_start_time": wall_start.isoformat(), "wall_completion_time": wall_completion.isoformat(),
            "monotonic_start_ns": str(monotonic_start), "monotonic_completion_ns": str(monotonic_completion),
            "input_artifact_ids": ",".join(sorted(input_artifacts)),
            "input_snapshot_ids": ",".join(_bound_snapshot_id(input_artifacts[key], key, run_id)
                                                for key in sorted(input_artifacts)),
            "input_artifact_digests": ",".join(_bound_digest(input_artifacts[key], key, run_id)
                                                   for key in sorted(input_artifacts)),
            "output_artifact_ids": ",".join(output_ids),
            "output_snapshot_ids": ",".join(output_snapshots[key].snapshot_id for key in output_ids),
            "output_artifact_digests": ",".join(output_digests),
            "output_validator_result_ids": ",".join(sorted(actual_result_ids)),
            "stdout_digest": hashlib.sha256(stdout).hexdigest(), "stderr_digest": hashlib.sha256(stderr).hexdigest(),
            "actual_exit_status": str(process.returncode), "execution_status": "completed",
        }
        receipt["receipt_digest"] = hmac.new(self.__key, _payload(receipt), hashlib.sha256).hexdigest()
        return ExecutionObservation(receipt, stdout, stderr, output_snapshots)

    def validate(self, receipt: dict[str, str], run_id: str,
                 input_artifacts: dict[str, str | ArtifactSnapshot],
                 output_artifacts: dict[str, Path | ArtifactSnapshot],
                 expected_validator_result_ids: set[str]) -> None:
        if list(receipt) != RECEIPT_FIELDS:
            raise RunnerError("invalid-receipt-schema")
        if receipt["version"] != RECEIPT_VERSION or receipt["run_id"] != run_id:
            raise RunnerError("stale-receipt")
        execution_id = receipt["validator_execution_id"]
        if not execution_id or not re.fullmatch(rf"{re.escape(self.execution_prefix)}-.+", execution_id):
            raise RunnerError("invalid-execution-id")
        if execution_id in self._validated_execution_ids:
            raise RunnerError("duplicate-or-reused-receipt")
        registry, implementation = self._registered(receipt["validator_id"])
        implementation_snapshot = _snapshot(implementation, f"IMPL-{receipt['validator_id']}", run_id)
        bindings = {
            "validator_version": registry["version"], "implementation_path": registry["implementation"],
            "implementation_digest": implementation_snapshot.digest, "entry_point": registry["entry_point"],
            "input_artifact_ids": ",".join(sorted(input_artifacts)),
            "input_snapshot_ids": ",".join(_bound_snapshot_id(input_artifacts[key], key, run_id)
                                                for key in sorted(input_artifacts)),
            "input_artifact_digests": ",".join(_bound_digest(input_artifacts[key], key, run_id)
                                                   for key in sorted(input_artifacts)),
            "output_artifact_ids": ",".join(sorted(output_artifacts)),
            "output_snapshot_ids": ",".join(_bound_snapshot_id(output_artifacts[key], key, run_id)
                                                 for key in sorted(output_artifacts)),
            "output_artifact_digests": ",".join(_bound_digest(output_artifacts[key], key, run_id)
                                                    for key in sorted(output_artifacts)),
            "output_validator_result_ids": ",".join(sorted(expected_validator_result_ids)),
        }
        for field, expected in bindings.items():
            if receipt.get(field) != expected:
                raise RunnerError(f"registry-or-artifact-binding-mismatch:{field}")
        try:
            invocation = json.loads(receipt["invocation_contract"])
        except (json.JSONDecodeError, TypeError):
            raise RunnerError("invalid-invocation-contract")
        if (not isinstance(invocation, list) or len(invocation) < 2 or
                invocation[0] != str(implementation) or invocation[1] != registry["entry_point"]):
            raise RunnerError("registry-or-artifact-binding-mismatch:invocation_contract")
        if (not receipt["process_id"].isdigit() or not receipt["process_start_identity"] or
                not receipt["observer_process_id"].isdigit()):
            raise RunnerError("invalid-process-identity")
        if (not receipt["monotonic_start_ns"].isdigit() or not receipt["monotonic_completion_ns"].isdigit() or
                int(receipt["monotonic_completion_ns"]) < int(receipt["monotonic_start_ns"])):
            raise RunnerError("invalid-monotonic-times")
        if receipt["actual_exit_status"] != "0" or receipt["execution_status"] != "completed":
            raise RunnerError("unsuccessful-execution")
        completion = datetime.fromisoformat(receipt["wall_completion_time"])
        if abs((datetime.now(timezone.utc) - completion).total_seconds()) > self.max_age_seconds:
            raise RunnerError("stale-receipt")
        expected_mac = hmac.new(self.__key, _payload(receipt), hashlib.sha256).hexdigest()
        if not hmac.compare_digest(expected_mac, receipt["receipt_digest"]):
            raise RunnerError("invalid-receipt-attestation")
        self._validated_execution_ids.add(execution_id)
