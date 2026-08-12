#!/usr/bin/env python3
"""Deployment-owned H4 supervisor for the bounded Campaign Manager command.

This module deliberately has no database client.  It only invokes the existing
read-only H1 readiness command and the existing H3 run-once command.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import pwd
import re
import signal
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

MAX_LIMIT = 100
STOP_ACTIONS = {"STOP_DISABLED", "STOP_DEGRADED_OPERATOR_REQUIRED",
                "STOP_INVALID_CONFIGURATION", "STOP_MALFORMED_RESULT"}
RETRY_ACTION = "BACKOFF_AND_RETRY_WITH_READINESS"
NORMAL_ACTION = "CONTINUE_AFTER_NORMAL_INTERVAL"
RESTORE_ACTION = "RESTORE_PERSISTED_NEXT_ACTION"
ALL_ACTIONS = STOP_ACTIONS | {RETRY_ACTION, NORMAL_ACTION, RESTORE_ACTION}
H3_LOGIN_ENVIRONMENT_KEY = "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER"
DISPATCH_SERVICE_LOGIN_ENVIRONMENT_KEY = "CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER"

NORMAL_CLASSIFICATIONS = {"work_completion", "no_work_completion", "request_local_semantic_failure"}
RETRY_CLASSIFICATIONS = {"database_failure", "connectivity_transport_failure", "process_interruption"}
STOP_CLASSIFICATIONS = {
    "STOP_DISABLED": {"production_disabled"},
    "STOP_DEGRADED_OPERATOR_REQUIRED": {
        "scheduler_protocol_ineffective", "manager_build_not_ready", "privilege_failure",
        "indeterminate", "retry_budget_exhausted", "graceful_stop",
        "duplicate_drift_observed"},
    "STOP_INVALID_CONFIGURATION": {"invalid_configuration"},
    "STOP_MALFORMED_RESULT": {"malformed_result", "unclassifiable_process_state", "restart_incomplete_state"},
}

PERSISTED_REQUIRED_FIELDS = {
    "complete", "deployment_identity", "deployment_execution_identity", "target_database_identity", "target_environment", "postgresql_login_identity",
    "classification", "next_action", "retry_count", "status", "service_alive",
    "terminal_result_validity", "graceful_drain_state", "duplicate_drift_detected",
    "invocation_start", "invocation_end", "child_exit_status", "next_scheduled_invocation",
    "work_occurred", "last_valid_work_result", "last_valid_no_work_result",
    "readiness_check_at", "readiness_result", "readiness_blocker", "updated_at",
}
PERSISTED_OPTIONAL_FIELDS = {"restored_next_action", "readiness_diagnostic", "resolution_of", "schedule_origin_at"}


class ConfigurationError(ValueError):
    pass


def now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def future(seconds: float) -> str:
    return datetime.fromtimestamp(time.time() + seconds, timezone.utc).isoformat().replace("+00:00", "Z")


def bounded_schedule(seconds: float) -> tuple[str, str]:
    """Create one bounded deadline and the immutable timestamp that bounds it."""
    scheduled_from = time.time()
    return (
        datetime.fromtimestamp(scheduled_from, timezone.utc).isoformat().replace("+00:00", "Z"),
        datetime.fromtimestamp(scheduled_from + seconds, timezone.utc).isoformat().replace("+00:00", "Z"),
    )


def parse_timestamp(value: object) -> Optional[datetime]:
    """Accept only a finite, timezone-aware persisted UTC schedule timestamp."""
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None or not math.isfinite(parsed.timestamp()):
            return None
        return parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None


def redact(value: str) -> str:
    """Keep diagnostics useful without ever copying an environment secret."""
    return re.sub(r"(?i)(password|secret|token|key)=([^\s,;]+)", r"\1=<redacted>", value)


def parse_record(line: str, name: str, required: set[str]) -> Optional[dict[str, str]]:
    parts = line.rstrip("\n").split(",")
    if not parts or parts[0] != name:
        return None
    result: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        if not key or key in result:
            return None
        result[key] = value
    return result if set(result) == required else None


def parse_record_with_required_fields(line: str, name: str, required: set[str]) -> Optional[dict[str, str]]:
    """Parse an extensible existing machine record while requiring stable fields."""
    parts = line.rstrip("\n").split(",")
    if not parts or parts[0] != name:
        return None
    result: dict[str, str] = {}
    for part in parts[1:]:
        if "=" not in part:
            return None
        key, value = part.split("=", 1)
        if not key or key in result:
            return None
        result[key] = value
    return result if required <= set(result) else None


def int_value(value: str, minimum: int = 0) -> Optional[int]:
    if not re.fullmatch(r"[0-9]+", value):
        return None
    parsed = int(value)
    return parsed if parsed >= minimum else None


def output_lines(value: str) -> list[str]:
    return [line for line in value.splitlines() if line]


def contains_transport(value: str) -> bool:
    return bool(re.search(r"(?i)(connection.*(refused|lost|reset|unavailable|fail)|"
                          r"broken_connection|network|transport|could not connect|sqlstate.?08)", value))


def contains_indeterminate(value: str) -> bool:
    return bool(re.search(r"(?i)(commit[ _-]?(outcome[ _-]?)?unknown|"
                          r"indeterminate|reconciliation[ _-]?required)", value))


@dataclass(frozen=True)
class Config:
    path: Path
    deployment_identity: str
    deployment_execution_identity: str
    target_database_identity: str
    target_environment: str
    postgresql_login_identity: str
    executable_path: Path
    executable_sha256: str
    connection_environment_file: Path
    validated_child_environment: dict[str, str]
    limit: int
    normal_interval_seconds: float
    backoff_seconds: tuple[float, ...]
    retry_budget: int
    graceful_drain_timeout_seconds: float
    state_directory: Path
    log_directory: Path
    log_retention_policy: str

    @staticmethod
    def load(path: Path) -> "Config":
        if (not path.is_absolute() or not path.is_file() or
                (path.stat().st_mode & 0o077)):
            raise ConfigurationError(
                "configuration file must be absolute, regular, and owner-only"
            )
        try:
            raw = json.loads(path.read_text())
        except Exception as error:
            raise ConfigurationError("configuration is not readable JSON") from error
        if not isinstance(raw, dict):
            raise ConfigurationError("configuration root must be an object")
        required = {"deployment_identity", "deployment_execution_identity", "target_database_identity", "target_environment", "postgresql_login_identity",
                    "executable_path", "executable_sha256", "connection_environment_file",
                    "limit", "normal_interval_seconds", "backoff_seconds", "retry_budget",
                    "graceful_drain_timeout_seconds", "state_directory", "log_directory", "log_retention_policy"}
        if set(raw) != required:
            raise ConfigurationError("configuration keys are incomplete or unrecognized")
        def text(key: str) -> str:
            value = raw[key]
            if not isinstance(value, str) or not value.strip():
                raise ConfigurationError("required text configuration is invalid")
            return value
        limit = raw["limit"]
        retry = raw["retry_budget"]
        interval = raw["normal_interval_seconds"]
        drain = raw["graceful_drain_timeout_seconds"]
        backoff = raw["backoff_seconds"]
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= MAX_LIMIT:
            raise ConfigurationError("limit must be a positive integer no greater than 100")
        if not isinstance(retry, int) or isinstance(retry, bool) or retry <= 0:
            raise ConfigurationError("retry_budget must be finite and positive")
        def finite_positive(value, name: str) -> float:
            if (not isinstance(value, (int, float)) or isinstance(value, bool) or
                    not math.isfinite(value) or value <= 0):
                raise ConfigurationError(f"{name} must be finite and positive")
            return float(value)
        interval = finite_positive(interval, "normal_interval_seconds")
        drain = finite_positive(drain, "graceful_drain_timeout_seconds")
        if not isinstance(backoff, list) or not backoff:
            raise ConfigurationError("backoff_seconds must be a bounded finite positive list")
        backoff_values = tuple(finite_positive(x, "backoff_seconds") for x in backoff)
        executable = Path(text("executable_path"))
        if not executable.is_absolute() or not executable.is_file() or not os.access(executable, os.X_OK):
            raise ConfigurationError("executable_path must name an executable regular file")
        digest = text("executable_sha256").lower()
        if not re.fullmatch(r"[0-9a-f]{64}", digest):
            raise ConfigurationError("executable_sha256 must be a SHA-256 digest")
        observed = hashlib.sha256(executable.read_bytes()).hexdigest()
        if observed != digest:
            raise ConfigurationError("executable build identity does not match executable_sha256")
        env_file = Path(text("connection_environment_file"))
        if not env_file.is_absolute() or not env_file.is_file() or (env_file.stat().st_mode & 0o077):
            raise ConfigurationError("connection environment file must be absolute, regular, and owner-only")
        for directory_key in ("state_directory", "log_directory"):
            directory = Path(text(directory_key))
            if not directory.is_absolute():
                raise ConfigurationError("state and log directories must be absolute")

            # Reject symlinks before exists(): a dangling symlink must not be
            # accepted merely because Path.exists() follows its missing target.
            if directory.is_symlink():
                raise ConfigurationError(
                    "state and log paths must not be symlinks"
                )

            if directory.exists():
                if not directory.is_dir():
                    raise ConfigurationError(
                        "existing state and log paths must be directories"
                    )
                directory_stat = directory.stat()

                # main() separately proves that the process execution identity
                # equals deployment_execution_identity. Therefore ownership by
                # this effective UID is the filesystem form of that contract.
                if directory_stat.st_uid != os.geteuid():
                    raise ConfigurationError(
                        "existing state and log directories must be owned by "
                        "the deployment execution identity"
                    )

                if directory_stat.st_mode & 0o077:
                    raise ConfigurationError(
                        "existing state and log directories must be owner-only"
                    )
        login_identity = text("postgresql_login_identity")
        child_environment = Config._validated_child_environment(env_file)
        if child_environment[H3_LOGIN_ENVIRONMENT_KEY] != login_identity:
            raise ConfigurationError("postgresql_login_identity must match the configured Manager production LOGIN")
        if child_environment[DISPATCH_SERVICE_LOGIN_ENVIRONMENT_KEY] == login_identity:
            raise ConfigurationError("dispatch service LOGIN must be distinct from the Manager production LOGIN")
        return Config(path, text("deployment_identity"), text("deployment_execution_identity"),
                      text("target_database_identity"), text("target_environment"), login_identity,
                      executable, digest, env_file, child_environment, limit,
                      interval, backoff_values, retry, drain,
                      Path(text("state_directory")), Path(text("log_directory")), text("log_retention_policy"))

    @staticmethod
    def _validated_child_environment(connection_environment_file: Path) -> dict[str, str]:
        values: dict[str, str] = {}
        try:
            lines = connection_environment_file.read_text().splitlines()
        except Exception as error:
            raise ConfigurationError("connection environment file is unreadable") from error
        allowed = {"LSTM_DB_HOST", "LSTM_DB_NAME",
                   "CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER",
                   "CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER",
                   "PGPASSFILE", "PGSSLMODE", "PGSSLROOTCERT"}
        for line in lines:
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                raise ConfigurationError("connection environment file has an invalid line")
            key, value = line.split("=", 1)
            if key not in allowed or not value or key in values or "\x00" in value:
                raise ConfigurationError("connection environment file has an unapproved setting")
            values[key] = value
        if (not values.get("LSTM_DB_HOST") or not values.get("LSTM_DB_NAME") or
                not values.get("CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER") or
                not values.get("CAMPAIGN_OPERATIONS_PRODUCTION_DISPATCH_SERVICE_DB_USER")):
            raise ConfigurationError("connection environment must explicitly identify LSTM_DB_HOST, LSTM_DB_NAME, the Manager production LOGIN, and the distinct dispatch-service LOGIN")
        # H4 invokes only Manager/readiness commands.  PGSERVICE/PGUSER remain
        # rejected so an ambient connection identity cannot replace the
        # reviewed explicit Manager LOGIN.
        # Do not inherit arbitrary parent LSTM/PG substitutions.
        environment = {key: os.environ[key] for key in ("PATH", "HOME", "TMPDIR", "LANG", "LC_ALL") if key in os.environ}
        environment.update(values)
        return environment

    def child_environment(self) -> dict[str, str]:
        """Return the startup-validated environment without rereading deployment input."""
        return dict(self.validated_child_environment)

    def validate_execution_identity(self) -> None:
        try:
            observed = pwd.getpwuid(os.geteuid()).pw_name
        except KeyError as error:
            raise ConfigurationError("deployment execution identity cannot be resolved") from error
        if observed != self.deployment_execution_identity:
            raise ConfigurationError("deployment execution identity does not match the reviewed launchd account")


class DurableState:
    def __init__(self, config: Config):
        self.config = config
        self.path = config.state_directory / "campaign_operations_h4_state.json"
        self.health_path = config.state_directory / "campaign_operations_h4_health.json"
        self.transitions_path = config.state_directory / "campaign_operations_h4_transitions.jsonl"

    def _sync_directory(self, directory: Path) -> None:
        descriptor = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def write(self, record: dict) -> None:
        self.config.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        record = dict(record)
        record["deployment_identity"] = self.config.deployment_identity
        record["deployment_execution_identity"] = self.config.deployment_execution_identity
        record["target_database_identity"] = self.config.target_database_identity
        record["target_environment"] = self.config.target_environment
        record["postgresql_login_identity"] = self.config.postgresql_login_identity
        record["updated_at"] = now()
        self._append_transition(record)
        self._atomic(self.path, record)
        self._atomic(self.health_path, record)

    def _append_transition(self, record: dict) -> None:
        """Append deployment-only classifier/state evidence before publication."""
        transition = {"event": "state_transition", "record": record}
        with open(self.transitions_path, "a", encoding="utf-8") as output:
            json.dump(transition, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        self._sync_directory(self.config.state_directory)

    def _atomic(self, path: Path, record: dict) -> None:
        descriptor, temporary_name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
        temporary = Path(temporary_name)
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            json.dump(record, output, sort_keys=True)
            output.write("\n")
            output.flush()
            os.fsync(output.fileno())
        try:
            os.replace(temporary, path)
            self._sync_directory(path.parent)
        finally:
            if temporary.exists():
                temporary.unlink()

    def read(self) -> Optional[dict]:
        if not self.path.exists():
            return None
        try:
            value = json.loads(self.path.read_text())
        except Exception:
            return {"complete": False}
        if not isinstance(value, dict):
            return {"complete": False}
        return value

    @staticmethod
    def write_invalid_configuration_stop(config_path: Path, diagnostic: str) -> None:
        """Best-effort deployment-owned evidence for a rejected startup config."""
        try:
            raw = json.loads(config_path.read_text())
            state_directory = raw.get("state_directory") if isinstance(raw, dict) else None
            if not isinstance(state_directory, str) or not Path(state_directory).is_absolute():
                return
            directory = Path(state_directory)
            directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            identity_fields = ("deployment_identity", "deployment_execution_identity",
                               "target_database_identity", "target_environment",
                               "postgresql_login_identity")
            identities = {key: raw.get(key) if isinstance(raw.get(key), str) else "<invalid-configuration>"
                          for key in identity_fields}
            record = {"complete": True, **identities, "classification": "invalid_configuration",
                      "next_action": "STOP_INVALID_CONFIGURATION", "retry_count": 0,
                      "status": "stopped", "service_alive": False,
                      "terminal_result_validity": "not_applicable",
                      "graceful_drain_state": "not_requested", "duplicate_drift_detected": None,
                      "invocation_start": None, "invocation_end": None, "child_exit_status": None,
                      "next_scheduled_invocation": None, "work_occurred": None,
                      "last_valid_work_result": False, "last_valid_no_work_result": False,
                      "readiness_check_at": None, "readiness_result": "invalid_configuration",
                      "readiness_blocker": "invalid_configuration", "readiness_diagnostic": redact(diagnostic),
                      "updated_at": now()}
            transition_path = directory / "campaign_operations_h4_transitions.jsonl"
            with open(transition_path, "a", encoding="utf-8") as output:
                json.dump({"event": "state_transition", "record": record}, output, sort_keys=True)
                output.write("\n")
                output.flush()
                os.fsync(output.fileno())
            for name in ("campaign_operations_h4_state.json", "campaign_operations_h4_health.json"):
                destination = directory / name
                descriptor, temporary_name = tempfile.mkstemp(prefix=name + ".", suffix=".tmp", dir=directory)
                temporary = Path(temporary_name)
                try:
                    with os.fdopen(descriptor, "w", encoding="utf-8") as output:
                        json.dump(record, output, sort_keys=True)
                        output.write("\n")
                        output.flush()
                        os.fsync(output.fileno())
                    os.replace(temporary, destination)
                finally:
                    if temporary.exists():
                        temporary.unlink()
            descriptor = os.open(directory, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        except Exception:
            return


class Supervisor:
    def __init__(self, config: Config, runner: Callable = subprocess.run, sleeper: Callable = time.sleep,
                 before_h3_launch: Optional[Callable[[], None]] = None):
        self.config, self.runner, self.sleeper = config, runner, sleeper
        self.state = DurableState(config)
        self.before_h3_launch = before_h3_launch
        self.shutdown_requested = False
        self.active_child: Optional[subprocess.Popen] = None
        self.forced_termination = False
        self.last_readiness: dict = {}
        self.last_invocation_start: Optional[str] = None
        self.duplicate_drift_marker = (
            config.state_directory /
            "campaign_operations_h4_duplicate_drift.marker"
        )

    def duplicate_drift_observed(self) -> bool:
        """Deployment-owned duplicate observation input.

        The marker is diagnostic/operational evidence only. It provides no
        ownership, liveness, fencing, or database-coordination authority.
        Its presence means the deployment owner has independently observed
        duplicate-instance drift and H4 must suppress further launches.
        """
        return self.duplicate_drift_marker.exists()

    def _capture(self, label: str, completed) -> None:
        self.config.log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        for stream, value in (("stdout", completed.stdout), ("stderr", completed.stderr)):
            path = self.config.log_directory / f"{stamp}-{label}.{stream}.log"
            with open(path, "w", encoding="utf-8") as output:
                output.write(value or "")
                output.flush()
                os.fsync(output.fileno())
            os.chmod(path, 0o600)
        self.state._sync_directory(self.config.log_directory)

    def _run(self, args: list[str], label: str):
        # The injected runner keeps classifier tests deterministic. Production H3
        # runs use Popen so a requested maintenance drain can be bounded.
        if label == "h3":
            # This is the final pre-launch boundary: no injected test runner or
            # production Popen may begin a child after a stop request.
            if self.before_h3_launch:
                self.before_h3_launch()
            if self.shutdown_requested:
                return None
        if label == "h3" and self.runner is subprocess.run:
            child = self._start_h3_child(args)
            if child is None:
                return None
            self.active_child = child
            drain_started: Optional[float] = None
            try:
                while child.poll() is None:
                    if self.shutdown_requested:
                        if drain_started is None:
                            drain_started = time.monotonic()
                        elif time.monotonic() - drain_started >= self.config.graceful_drain_timeout_seconds:
                            self.forced_termination = True
                            child.terminate()
                            try:
                                child.wait(timeout=5)
                            except subprocess.TimeoutExpired:
                                child.kill()
                            break
                    time.sleep(0.1)
                stdout, stderr = child.communicate()
                completed = subprocess.CompletedProcess(args, child.returncode, stdout, stderr)
            finally:
                self.active_child = None
            self._capture(label, completed)
            return completed
        completed = self.runner(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                text=True, env=self.config.child_environment(), check=False)
        self._capture(label, completed)
        return completed

    def _start_h3_child(self, args: list[str]) -> Optional[subprocess.Popen]:
        """Linearize the final stop decision with H3 child creation.

        On macOS/POSIX, temporarily block the two supervisor stop signals so a
        handler cannot run after this decision but before fork/exec. A pending
        pre-existing stop is observed before creation; a later signal is
        handled immediately after the child exists by the established drain
        path. The child restores the caller's mask before exec so H3 retains
        normal SIGTERM/SIGINT semantics.
        """
        stop_signals = {signal.SIGTERM, signal.SIGINT}
        previous_mask = signal.pthread_sigmask(signal.SIG_BLOCK, stop_signals)
        try:
            if self.before_h3_launch:
                self.before_h3_launch()
            if self.shutdown_requested or signal.sigpending() & stop_signals:
                return None
            return subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                    text=True, env=self.config.child_environment(),
                                    preexec_fn=lambda: signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask))
        finally:
            signal.pthread_sigmask(signal.SIG_SETMASK, previous_mask)

    def readiness(self) -> tuple[bool, str, str]:
        completed = self._run([str(self.config.executable_path), "--campaign-operations-production-readiness"], "readiness")
        stdout, stderr = output_lines(completed.stdout or ""), output_lines(completed.stderr or "")
        records = [parse_record_with_required_fields(line, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS", {"ready", "blockers"}) for line in stdout]
        # H1's machine record is extensible.  These two fields are stable and
        # blocker names are the authoritative blocked-result discriminators.
        valid = len(records) == 1 and records[0] is not None and len(stdout) == 1
        fields = records[0] if valid else {}
        diagnostic = redact((completed.stderr or "") + " " + (completed.stdout or ""))
        if valid and completed.returncode == 0 and fields.get("ready") == "true" and fields.get("blockers") == "none":
            return True, "ready", diagnostic
        if valid and completed.returncode == 2 and fields.get("ready") == "false" and fields.get("blockers") not in {"", "none"}:
            blockers = set(fields["blockers"].split(";"))
            # ADR-0020 §2.5 ordering is intentional.  Identity fields are
            # report data, never a privilege signal by themselves.
            if {"enablement_absent", "enablement_ineffective"} & blockers or fields.get("enablement_effective") == "false":
                return False, "production_disabled", diagnostic
            if "scheduler_protocol_evidence" in blockers:
                return False, "scheduler_protocol_ineffective", diagnostic
            if {"actual_manager_build_contract", "manager_build_contract_mismatch", "manager_service_contract"} & blockers:
                return False, "manager_build_not_ready", diagnostic
            if {"session_principal_identity", "principal_role_membership"} & blockers:
                return False, "privilege_failure", diagnostic
            return False, "readiness_unclassifiable", diagnostic
        error_records = [parse_record_with_required_fields(line, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS", {"status", "diagnostic_code"}) for line in stderr]
        text = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if (len(error_records) == 1 and error_records[0] is not None and len(stderr) == 1 and
                error_records[0].get("status") == "failed"):
            diagnostic_code = error_records[0]["diagnostic_code"]
            if diagnostic_code == "postgresql_42501":
                return False, "privilege_failure", diagnostic
            if diagnostic_code.startswith("postgresql_"):
                return False, "database_failure", diagnostic
        if contains_transport(text):
            return False, "database_failure", diagnostic
        return False, "readiness_unclassifiable", diagnostic

    def classify_h3(self, completed, forced: bool = False) -> tuple[str, str, bool, Optional[bool]]:
        """Return normalized outcome, action, terminal validity, work indication."""
        if forced or completed.returncode is not None and completed.returncode < 0:
            return "process_interruption", RETRY_ACTION, False, None
        stdout, stderr = output_lines(completed.stdout or ""), output_lines(completed.stderr or "")
        summary_name = "CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE"
        request_name = "CAMPAIGN_OPERATIONS_MANAGER_REQUEST"
        stopped_name = "CAMPAIGN_OPERATIONS_MANAGER_STOPPED"
        summary_required = {"dispatch_limit", "candidates_selected", "processed", "stopped_early", "stop_reason", "candidate_request_ids"}
        request_required = {"request_id", "expected_request_version", "operation_key", "outcome", "dispatch_classification", "replay_disposition", "newly_committed", "exact_replay", "diagnostic_code"}
        stop_required = {"reason", "diagnostic"}
        summaries = [parse_record(x, summary_name, summary_required) for x in stdout if x.startswith(summary_name)]
        requests = [parse_record(x, request_name, request_required) for x in stdout if x.startswith(request_name)]
        stops = [parse_record(x, stopped_name, stop_required) for x in stderr if x.startswith(stopped_name)]
        recognized = len(summaries) + len(requests) == len(stdout) and len(stops) == len(stderr)
        if len(summaries) != 1 or any(x is None for x in summaries + requests + stops) or not recognized:
            return self._nonterminal(completed)
        summary = summaries[0]
        assert summary is not None
        selected, processed = int_value(summary["candidates_selected"]), int_value(summary["processed"])
        if (summary["dispatch_limit"] != str(self.config.limit) or selected is None or processed is None or
                summary["stopped_early"] not in {"true", "false"} or
                summary["stop_reason"] not in {"none", "production_disabled", "scheduler_protocol_ineffective", "manager_build_not_ready", "privilege_failure", "database_failure"}):
            return "malformed_result", "STOP_MALFORMED_RESULT", False, None
        candidates = [] if summary["candidate_request_ids"] == "none" else summary["candidate_request_ids"].split(":")
        if (selected > self.config.limit or len(candidates) != selected or any(int_value(x, 1) is None for x in candidates) or
                len(set(candidates)) != len(candidates) or len(requests) != processed or processed > selected):
            return "malformed_result", "STOP_MALFORMED_RESULT", False, None
        request_ids = [r["request_id"] for r in requests if r is not None]
        if (any(r is None or int_value(r["request_id"], 1) is None or
                r["outcome"] not in {"dispatch_result", "request_local_semantic_failure"} for r in requests) or
                len(set(request_ids)) != len(request_ids) or request_ids != candidates[:processed]):
            return "malformed_result", "STOP_MALFORMED_RESULT", False, None
        stopped = summary["stopped_early"] == "true"
        if (not stopped and (completed.returncode != 0 or summary["stop_reason"] != "none" or stops or processed != selected)):
            return "malformed_result", "STOP_MALFORMED_RESULT", False, None
        if stopped and (completed.returncode != 2 or len(stops) != 1 or stops[0] is None or stops[0]["reason"] != summary["stop_reason"] or summary["stop_reason"] == "none"):
            return "malformed_result", "STOP_MALFORMED_RESULT", False, None
        if stopped:
            reason, diagnostic = summary["stop_reason"], stops[0]["diagnostic"]
            if contains_indeterminate(diagnostic):
                return "indeterminate", "STOP_DEGRADED_OPERATOR_REQUIRED", True, None
            actions = {"production_disabled": "STOP_DISABLED", "scheduler_protocol_ineffective": "STOP_DEGRADED_OPERATOR_REQUIRED", "manager_build_not_ready": "STOP_DEGRADED_OPERATOR_REQUIRED", "privilege_failure": "STOP_DEGRADED_OPERATOR_REQUIRED", "database_failure": RETRY_ACTION}
            return reason, actions[reason], True, None
        local_failure = any(r["outcome"] == "request_local_semantic_failure" for r in requests if r)
        return ("request_local_semantic_failure" if local_failure else ("work_completion" if processed else "no_work_completion"), NORMAL_ACTION, True, processed > 0)

    def _nonterminal(self, completed) -> tuple[str, str, bool, Optional[bool]]:
        evidence = (completed.stdout or "") + "\n" + (completed.stderr or "")
        if completed.returncode not in (0, 2) and contains_transport(evidence):
            return "connectivity_transport_failure", RETRY_ACTION, False, None
        return "unclassifiable_process_state", "STOP_MALFORMED_RESULT", False, None

    def _action_for_readiness(self, result: str) -> str:
        return {"production_disabled": "STOP_DISABLED", "scheduler_protocol_ineffective": "STOP_DEGRADED_OPERATOR_REQUIRED", "manager_build_not_ready": "STOP_DEGRADED_OPERATOR_REQUIRED", "privilege_failure": "STOP_DEGRADED_OPERATOR_REQUIRED", "database_failure": RETRY_ACTION}.get(result, "STOP_DEGRADED_OPERATOR_REQUIRED")

    def persist(self, classification: str, action: str, retry_count: int, **extra) -> None:
        next_scheduled = extra.pop("next_scheduled_invocation", None)
        schedule_origin = extra.pop("schedule_origin_at", None)
        if action == NORMAL_ACTION and next_scheduled is None:
            schedule_origin, next_scheduled = bounded_schedule(self.config.normal_interval_seconds)
        if action == RETRY_ACTION and next_scheduled is None:
            retry_index = min(max(retry_count - 1, 0), len(self.config.backoff_seconds) - 1)
            schedule_origin, next_scheduled = bounded_schedule(self.config.backoff_seconds[retry_index])
        implied_work = True if classification in {"work_completion", "request_local_semantic_failure"} else False if classification == "no_work_completion" else None
        record = {"complete": True, "service_alive": action not in STOP_ACTIONS, "classification": classification,
                  "next_action": action, "retry_count": retry_count, "status": "stopped" if action in STOP_ACTIONS else "scheduled",
                  "terminal_result_validity": extra.pop("terminal_result_validity", "not_applicable"),
                  "graceful_drain_state": "requested" if self.shutdown_requested else "not_requested",
                  # Null means no deployment-owned duplicate observation has
                  # been supplied; it is never a manufactured negative.
                  "duplicate_drift_detected": None, "invocation_start": self.last_invocation_start,
                  "invocation_end": None, "child_exit_status": None,
                  "next_scheduled_invocation": next_scheduled, "work_occurred": implied_work,
                  "last_valid_work_result": classification == "work_completion",
                  "last_valid_no_work_result": classification == "no_work_completion",
                  "readiness_check_at": None, "readiness_result": None,
                  "readiness_blocker": None, **self.last_readiness, **extra}
        if next_scheduled is not None:
            # This timestamp proves the original bounded relationship even
            # after restart-state records receive later updated_at values.
            record["schedule_origin_at"] = schedule_origin or now()
        self.state.write(record)

    def wait(self, seconds: float) -> None:
        """Normal production sleeps remain interruptible before another launch."""
        if self.sleeper is not time.sleep:
            self.sleeper(seconds)
            return
        deadline = time.monotonic() + seconds
        while not self.shutdown_requested and time.monotonic() < deadline:
            time.sleep(min(0.2, deadline - time.monotonic()))

    def _complete_state_action(self, existing: dict) -> Optional[tuple[str, int, Optional[str]]]:
        """Validate every H4-owned value needed to restore an action.

        This deliberately never interprets child logs or database state.  A
        persisted record is usable only when its full current shape and its
        prior classifier/action relationship are independently coherent.
        """
        keys = set(existing)
        if not PERSISTED_REQUIRED_FIELDS <= keys or not keys <= PERSISTED_REQUIRED_FIELDS | PERSISTED_OPTIONAL_FIELDS:
            return None
        if existing.get("complete") is not True:
            return None
        if (existing.get("deployment_identity") != self.config.deployment_identity or
                existing.get("deployment_execution_identity") != self.config.deployment_execution_identity or
                existing.get("target_database_identity") != self.config.target_database_identity or
                existing.get("target_environment") != self.config.target_environment or
                existing.get("postgresql_login_identity") != self.config.postgresql_login_identity):
            return None
        action, classification, retry_count = existing.get("next_action"), existing.get("classification"), existing.get("retry_count")
        if (action not in ALL_ACTIONS or not isinstance(classification, str) or
                not isinstance(retry_count, int) or isinstance(retry_count, bool) or
                retry_count < 0 or retry_count > self.config.retry_budget):
            return None
        if existing.get("terminal_result_validity") not in {"valid", "missing", "malformed", "not_applicable", "missing_or_incomplete"}:
            return None
        if existing.get("graceful_drain_state") not in {"not_requested", "requested", "draining", "forced"}:
            return None
        duplicate_drift = existing.get("duplicate_drift_detected")
        if duplicate_drift not in {None, True}:
            return None
        if duplicate_drift is True and not (
                action == "STOP_DEGRADED_OPERATOR_REQUIRED" and
                classification == "duplicate_drift_observed"):
            return None
        for key in ("service_alive", "last_valid_work_result", "last_valid_no_work_result"):
            if not isinstance(existing.get(key), bool):
                return None
        if existing.get("work_occurred") not in {True, False, None}:
            return None
        if existing.get("child_exit_status") is not None and (not isinstance(existing.get("child_exit_status"), int) or isinstance(existing.get("child_exit_status"), bool)):
            return None
        if parse_timestamp(existing.get("updated_at")) is None:
            return None
        for key in ("invocation_start", "invocation_end", "next_scheduled_invocation", "readiness_check_at"):
            if existing.get(key) is not None and (not isinstance(existing.get(key), str) or not existing[key]):
                return None
        if "schedule_origin_at" in existing and (not isinstance(existing["schedule_origin_at"], str) or
                                                   not existing["schedule_origin_at"]):
            return None
        for key in ("readiness_result", "readiness_blocker", "readiness_diagnostic", "resolution_of"):
            if key in existing and existing[key] is not None and (not isinstance(existing[key], str) or not existing[key]):
                return None
        if existing.get("status") not in {"scheduled", "stopped", "resolved_pending_start"}:
            return None
        if action in STOP_ACTIONS:
            if existing["status"] != "stopped" or existing["service_alive"]:
                return None
        elif action == RESTORE_ACTION:
            restored_action = existing.get("restored_next_action")
            if restored_action not in STOP_ACTIONS | {RETRY_ACTION, NORMAL_ACTION}:
                return None
            if classification == "operator_stop_resolution":
                if restored_action != NORMAL_ACTION or existing["status"] != "resolved_pending_start" or existing["service_alive"]:
                    return None
            elif classification != "restart_complete_state" or existing["status"] != "scheduled" or not existing["service_alive"]:
                return None
            if classification == "restart_complete_state":
                if restored_action in {NORMAL_ACTION, RETRY_ACTION} and not self._valid_persisted_schedule(existing, restored_action, retry_count):
                    return None
                if restored_action in STOP_ACTIONS and (existing.get("next_scheduled_invocation") is not None or
                                                       "schedule_origin_at" in existing):
                    return None
            elif (existing.get("next_scheduled_invocation") is not None or "schedule_origin_at" in existing or
                  retry_count != 0):
                return None
            return restored_action, retry_count, existing.get("next_scheduled_invocation")
        elif "restored_next_action" in existing or existing["status"] != "scheduled" or not existing["service_alive"]:
            return None
        if action == NORMAL_ACTION and classification not in NORMAL_CLASSIFICATIONS:
            return None
        if action == RETRY_ACTION and classification not in RETRY_CLASSIFICATIONS:
            return None
        if action in STOP_ACTIONS and classification not in STOP_CLASSIFICATIONS[action]:
            return None
        if action in {NORMAL_ACTION, RETRY_ACTION} and not self._valid_persisted_schedule(existing, action, retry_count):
            return None
        if action in STOP_ACTIONS and (existing.get("next_scheduled_invocation") is not None or
                                       "schedule_origin_at" in existing):
            return None
        if classification == "work_completion" and (existing["work_occurred"] is not True or not existing["last_valid_work_result"]):
            return None
        if classification == "no_work_completion" and (existing["work_occurred"] is not False or not existing["last_valid_no_work_result"]):
            return None
        if classification == "request_local_semantic_failure" and existing["work_occurred"] is not True:
            return None
        return action, retry_count, existing.get("next_scheduled_invocation")

    def _valid_persisted_schedule(self, existing: dict, action: str, retry_count: int) -> bool:
        scheduled = parse_timestamp(existing.get("next_scheduled_invocation"))
        updated = parse_timestamp(existing.get("updated_at"))
        if scheduled is None or updated is None:
            return False
        schedule_origin_value = existing.get("schedule_origin_at")
        if schedule_origin_value is None:
            # A restart-state record must carry its original schedule proof;
            # its own updated_at is necessarily later than the prior action.
            if existing.get("next_action") == RESTORE_ACTION:
                return False
            schedule_origin = updated
        else:
            schedule_origin = parse_timestamp(schedule_origin_value)
            if schedule_origin is None or schedule_origin > updated:
                return False
        if action == NORMAL_ACTION:
            if retry_count != 0:
                return False
            maximum_delay = self.config.normal_interval_seconds
        elif action == RETRY_ACTION:
            if retry_count < 1 or retry_count >= self.config.retry_budget:
                return False
            maximum_delay = self.config.backoff_seconds[min(retry_count - 1, len(self.config.backoff_seconds) - 1)]
        else:
            return False
        delay = scheduled.timestamp() - schedule_origin.timestamp()
        return math.isfinite(delay) and 0 < delay <= maximum_delay

    def restore(self, existing: Optional[dict]) -> Optional[tuple[str, int, Optional[str]]]:
        if existing is None:
            return None
        restored = self._complete_state_action(existing)
        if restored is None:
            self.persist("restart_incomplete_state", "STOP_MALFORMED_RESULT", 0, terminal_result_validity="missing_or_incomplete")
            return "STOP_MALFORMED_RESULT", 0, None
        action, retry_count, next_scheduled = restored

        # A valid persisted STOP remains the authoritative current H4 state.
        # Do not rewrite it as a scheduled/alive RESTORE wrapper merely because
        # the supervisor restarted.
        if action in STOP_ACTIONS:
            return action, retry_count, next_scheduled

        schedule_origin = existing.get("schedule_origin_at", existing["updated_at"])
        self.persist("restart_complete_state", RESTORE_ACTION, retry_count,
                     restored_next_action=action, terminal_result_validity=existing["terminal_result_validity"],
                     next_scheduled_invocation=next_scheduled, schedule_origin_at=schedule_origin)
        return action, retry_count, next_scheduled

    def resolve_stop_state(self) -> None:
        existing = self.state.read()
        restored = self._complete_state_action(existing) if existing else None
        if restored is None or restored[0] not in STOP_ACTIONS:
            raise ConfigurationError("resolve-stop-state requires one complete H4 STOP state")
        self.persist("operator_stop_resolution", RESTORE_ACTION, 0,
                     restored_next_action=NORMAL_ACTION, terminal_result_validity="not_applicable",
                     status="resolved_pending_start", service_alive=False,
                     next_scheduled_invocation=None, resolution_of=existing["classification"])

    def run(self, max_cycles: Optional[int] = None) -> int:
        restored = self.restore(self.state.read())
        retry_count = 0
        restored_schedule: Optional[str] = None
        if restored:
            action, retry_count, restored_schedule = restored
            if action in STOP_ACTIONS:
                return 0
            if restored_schedule is not None:
                scheduled = parse_timestamp(restored_schedule)
                assert scheduled is not None  # Validated by _complete_state_action.
                remaining = scheduled.timestamp() - time.time()
                if remaining > 0:
                    self.wait(remaining)
                if self.shutdown_requested:
                    self.persist("graceful_stop", "STOP_DEGRADED_OPERATOR_REQUIRED", retry_count,
                                 terminal_result_validity="not_applicable")
                    return 0
            # RESTORE_PERSISTED_NEXT_ACTION is durable; every later launch still preflights.
        cycles = 0
        while not self.shutdown_requested and (max_cycles is None or cycles < max_cycles):
            if self.duplicate_drift_observed():
                self.persist(
                    "duplicate_drift_observed",
                    "STOP_DEGRADED_OPERATOR_REQUIRED",
                    retry_count,
                    terminal_result_validity="not_applicable",
                    duplicate_drift_detected=True,
                )
                return 0

            ready, readiness_class, diagnostic = self.readiness()
            self.last_readiness = {"readiness_check_at": now(), "readiness_result": "ready" if ready else "blocked",
                                   "readiness_blocker": None if ready else readiness_class}
            if not ready:
                action = self._action_for_readiness(readiness_class)
                if action == RETRY_ACTION:
                    retry_count += 1
                    if retry_count >= self.config.retry_budget:
                        self.persist("retry_budget_exhausted", "STOP_DEGRADED_OPERATOR_REQUIRED", retry_count, readiness_result=readiness_class, readiness_diagnostic=diagnostic)
                        return 0
                    self.persist(readiness_class, action, retry_count, readiness_result=readiness_class, readiness_diagnostic=diagnostic)
                    self.wait(self.config.backoff_seconds[min(retry_count - 1, len(self.config.backoff_seconds) - 1)])
                    continue
                self.persist(readiness_class, action, retry_count, readiness_result=readiness_class, readiness_diagnostic=diagnostic)
                return 0
            if self.shutdown_requested:
                break
            self.last_invocation_start = now()
            self.state.write({"complete": False, "status": "invoking", "service_alive": True, "retry_count": retry_count, "invocation_start": self.last_invocation_start, "readiness_result": "ready", "readiness_check_at": self.last_readiness["readiness_check_at"], "readiness_blocker": None, "graceful_drain_state": "not_requested", "duplicate_drift_detected": None})
            completed = self._run([str(self.config.executable_path), "--campaign-operations-manager-run-once", str(self.config.limit), "--yes"], "h3")
            if completed is None:
                self.persist("graceful_stop", "STOP_DEGRADED_OPERATOR_REQUIRED", retry_count,
                             terminal_result_validity="not_applicable")
                return 0
            classification, action, valid, work = self.classify_h3(completed, self.forced_termination)
            cycles += 1
            if action == RETRY_ACTION:
                retry_count += 1
                if retry_count >= self.config.retry_budget:
                    self.persist("retry_budget_exhausted", "STOP_DEGRADED_OPERATOR_REQUIRED", retry_count, child_exit_status=completed.returncode, terminal_result_validity="valid" if valid else "missing", work_occurred=work)
                    return 0
                self.persist(classification, action, retry_count, child_exit_status=completed.returncode, invocation_end=now(), terminal_result_validity="valid" if valid else "missing", work_occurred=work)
                if self.shutdown_requested:
                    return 0
                if max_cycles is None or cycles < max_cycles:
                    self.wait(self.config.backoff_seconds[min(retry_count - 1, len(self.config.backoff_seconds) - 1)])
                continue
            self.persist(classification, action, 0 if action == NORMAL_ACTION else retry_count,
                         child_exit_status=completed.returncode, invocation_end=now(),
                         terminal_result_validity="valid" if valid else "malformed", work_occurred=work,
                         last_valid_work_result=work is True,
                         last_valid_no_work_result=valid and work is False,
                         next_scheduled_invocation=None)
            if action in STOP_ACTIONS:
                return 0
            retry_count = 0
            if max_cycles is None or cycles < max_cycles:
                self.wait(self.config.normal_interval_seconds)
        self.persist("graceful_stop", "STOP_DEGRADED_OPERATOR_REQUIRED", retry_count, terminal_result_validity="not_applicable")
        return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="External Campaign Operations H4 supervisor")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--resolve-stop-state", action="store_true",
                        help="deliberately resolve one complete deployment-owned H4 STOP state; does not launch H3")
    arguments = parser.parse_args()
    try:
        config = Config.load(arguments.config)
        config.validate_execution_identity()
        supervisor = Supervisor(config)
        if arguments.resolve_stop_state:
            supervisor.resolve_stop_state()
            return 0
    except ConfigurationError as error:
        DurableState.write_invalid_configuration_stop(arguments.config, str(error))
        print(f"H4 supervisor configuration rejected: {redact(str(error))}", file=sys.stderr)
        return 2
    def request_shutdown(_signum, _frame):
        supervisor.shutdown_requested = True
    signal.signal(signal.SIGTERM, request_shutdown)
    signal.signal(signal.SIGINT, request_shutdown)
    return supervisor.run()


if __name__ == "__main__":
    raise SystemExit(main())
