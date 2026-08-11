#!/usr/bin/env python3
"""Fixture tests for the deployment-owned H4 supervisor; no database is used."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("h4", ROOT / "Scripts/CampaignOperationsH4Supervisor.py")
h4 = importlib.util.module_from_spec(spec)
assert spec.loader
sys.modules[spec.name] = h4
spec.loader.exec_module(h4)


def result(status, stdout="", stderr=""):
    return subprocess.CompletedProcess(["fake"], status, stdout, stderr)


def summary(limit=1, selected=0, processed=0, stopped="false", reason="none", ids="none"):
    return f"CAMPAIGN_OPERATIONS_MANAGER_RUN_ONCE,dispatch_limit={limit},candidates_selected={selected},processed={processed},stopped_early={stopped},stop_reason={reason},candidate_request_ids={ids}\n"


def request(request_id=1, outcome="dispatch_result"):
    return (f"CAMPAIGN_OPERATIONS_MANAGER_REQUEST,request_id={request_id},expected_request_version=1,operation_key=k-{request_id},outcome={outcome},dispatch_classification=none,replay_disposition=none,newly_committed=false,exact_replay=false,diagnostic_code=none\n")


class H4SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        root = Path(self.temp.name)
        self.executable = root / "LSTM_Release"
        self.executable.write_text("#!/bin/sh\nexit 0\n")
        self.executable.chmod(0o700)
        self.env = root / "connection.env"
        self.env.write_text("LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nCAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=test_manager_login\n")
        self.env.chmod(0o600)
        self.config_path = root / "config.json"
        self.write_config()
        self.config = h4.Config.load(self.config_path)
        self.supervisor = h4.Supervisor(self.config, sleeper=lambda _: None)

    def tearDown(self):
        self.temp.cleanup()

    def write_config(self, **changes):
        root = Path(self.temp.name)
        value = {"deployment_identity":"test.h4", "deployment_execution_identity":"test-h4", "target_database_identity":"test-db", "target_environment":"test", "postgresql_login_identity":"test_manager_login", "executable_path":str(self.executable), "executable_sha256":hashlib.sha256(self.executable.read_bytes()).hexdigest(), "connection_environment_file":str(self.env), "limit":1, "normal_interval_seconds":1, "backoff_seconds":[1,2], "retry_budget":2, "graceful_drain_timeout_seconds":1, "state_directory":str(root / "state"), "log_directory":str(root / "logs"), "log_retention_policy":"test-policy"}
        value.update(changes)
        self.config_path.write_text(json.dumps(value))
        self.config_path.chmod(0o600)

    def classify(self, completed, forced=False):
        return self.supervisor.classify_h3(completed, forced)

    def test_exact_command_and_limit_validation(self):
        commands = []
        def runner(args, **kwargs):
            commands.append(args)
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n") if "readiness" in args[1] else result(0, summary())
        supervisor = h4.Supervisor(self.config, runner=runner, sleeper=lambda _: None)
        supervisor.run(max_cycles=1)
        self.assertEqual(commands[1], [str(self.executable), "--campaign-operations-manager-run-once", "1", "--yes"])
        self.write_config(limit=101)
        with self.assertRaises(h4.ConfigurationError): h4.Config.load(self.config_path)

    def test_existing_state_directory_must_be_owner_only(self):
        self.config.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.state_directory.chmod(0o755)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

    def test_existing_log_directory_must_be_owner_only(self):
        self.config.log_directory.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.config.log_directory.chmod(0o755)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

    def test_existing_state_directory_must_not_be_symlink(self):
        target = self.config.state_directory.parent / "state-target"
        target.mkdir(mode=0o700)
        self.config.state_directory.symlink_to(target, target_is_directory=True)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

    def test_tampered_stop_in_untrusted_state_directory_is_rejected(self):
        self.config.state_directory.mkdir(mode=0o700, parents=True, exist_ok=True)

        # Simulate a syntactically plausible replacement of a prior durable
        # STOP with a launch-permitting scheduled state.
        state_path = (
            self.config.state_directory /
            "campaign_operations_h4_state.json"
        )
        state_path.write_text(json.dumps({
            "complete": True,
            "deployment_identity": self.config.deployment_identity,
            "deployment_execution_identity":
                self.config.deployment_execution_identity,
            "target_database_identity": self.config.target_database_identity,
            "target_environment": self.config.target_environment,
            "postgresql_login_identity": self.config.postgresql_login_identity,
            "classification": "no_work_completion",
            "next_action": h4.NORMAL_ACTION,
            "retry_count": 0,
            "status": "scheduled",
            "service_alive": True,
            "terminal_result_validity": "valid",
            "graceful_drain_state": "not_requested",
            "duplicate_drift_detected": None,
            "invocation_start": None,
            "invocation_end": None,
            "child_exit_status": 0,
            "next_scheduled_invocation": h4.future(1),
            "work_occurred": False,
            "last_valid_work_result": False,
            "last_valid_no_work_result": True,
            "readiness_check_at": None,
            "readiness_result": None,
            "readiness_blocker": None,
            "updated_at": h4.now(),
        }))

        # An untrusted principal's ability to replace deployment state is
        # represented by an insecure containing directory. Configuration must
        # fail before the persisted record can become restart authority.
        self.config.state_directory.chmod(0o777)

        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

    def test_main_configuration_file_must_be_owner_only(self):
        self.config_path.chmod(0o644)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

        self.config_path.chmod(0o600)
        self.assertIsInstance(h4.Config.load(self.config_path), h4.Config)

    def test_duplicate_drift_marker_stops_before_readiness_or_h3(self):
        calls = []

        def runner(args, **kwargs):
            calls.append(args)
            return result(0)

        supervisor = h4.Supervisor(
            self.config,
            runner=runner,
            sleeper=lambda _: None,
        )

        supervisor.duplicate_drift_marker.parent.mkdir(
            mode=0o700, parents=True, exist_ok=True
        )
        supervisor.duplicate_drift_marker.write_text(
            "observed by deployment owner\n"
        )

        self.assertEqual(supervisor.run(max_cycles=1), 0)
        self.assertEqual(calls, [])

        state = supervisor.state.read()
        health = json.loads(
            (self.config.state_directory /
             "campaign_operations_h4_health.json").read_text()
        )

        self.assertEqual(
            state["classification"],
            "duplicate_drift_observed",
        )
        self.assertEqual(
            state["next_action"],
            "STOP_DEGRADED_OPERATOR_REQUIRED",
        )
        self.assertEqual(state["status"], "stopped")
        self.assertFalse(state["service_alive"])
        self.assertTrue(state["duplicate_drift_detected"])

        self.assertEqual(
            health["classification"],
            "duplicate_drift_observed",
        )
        self.assertTrue(health["duplicate_drift_detected"])

        # The resulting MUST-stop state remains durable across restart.
        restarted = h4.Supervisor(
            self.config,
            runner=runner,
            sleeper=lambda _: None,
        )
        restarted.run(max_cycles=1)

        restored = restarted.state.read()
        self.assertEqual(
            restored["next_action"],
            "STOP_DEGRADED_OPERATOR_REQUIRED",
        )
        self.assertTrue(restored["duplicate_drift_detected"])
        self.assertEqual(calls, [])

    def test_invalid_configuration_and_credential_secrecy(self):
        self.write_config(normal_interval_seconds=0)
        with self.assertRaises(h4.ConfigurationError): h4.Config.load(self.config_path)
        for key, value in (("normal_interval_seconds", float("nan")),
                           ("normal_interval_seconds", float("inf")),
                           ("graceful_drain_timeout_seconds", float("-inf")),
                           ("backoff_seconds", [1, float("nan")]),
                           ("backoff_seconds", [float("inf")])):
            with self.subTest(key=key, value=value):
                self.write_config(**{key: value})
                with self.assertRaises(h4.ConfigurationError): h4.Config.load(self.config_path)
        self.write_config(postgresql_login_identity="ambient-manager")
        with self.assertRaises(h4.ConfigurationError): h4.Config.load(self.config_path)
        self.write_config()
        self.config = h4.Config.load(self.config_path)
        with mock.patch.object(h4.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="wrong-user")):
            with self.assertRaises(h4.ConfigurationError): self.config.validate_execution_identity()
        with mock.patch.object(h4.pwd, "getpwuid", return_value=SimpleNamespace(pw_name="test-h4")):
            self.config.validate_execution_identity()
        for contents in (
                "LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nCAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=test_manager_login\nPGSERVICE=ambient\n",
                "LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nCAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=test_manager_login\nPGUSER=ambient\n",
                "LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nnot-an-assignment\n",
                "LSTM_DB_HOST=127.0.0.1\n",
                "LSTM_DB_NAME=test\n"):
            self.env.write_text(contents)
            with self.subTest(contents=contents), self.assertRaises(h4.ConfigurationError):
                h4.Config.load(self.config_path)
        self.env.write_text("LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nCAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=test_manager_login\n")
        self.env.chmod(0o644)
        with self.assertRaises(h4.ConfigurationError): h4.Config.load(self.config_path)
        self.env.chmod(0o600)
        self.assertEqual(h4.redact("password=secret-value"), "password=<redacted>")

    def test_manager_production_login_is_required_and_matches_reviewed_identity(self):
        self.env.write_text("LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\n")
        self.env.chmod(0o600)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

        self.env.write_text("LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nCAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=other_login\n")
        self.env.chmod(0o600)
        with self.assertRaises(h4.ConfigurationError):
            h4.Config.load(self.config_path)

    def test_invalid_connection_environment_stops_at_startup_with_health_evidence(self):
        self.env.write_text("LSTM_DB_HOST=127.0.0.1\nLSTM_DB_NAME=test\nPGSERVICE=ambient\n")
        with mock.patch.object(sys, "argv", ["h4", "--config", str(self.config_path)]), \
                mock.patch("sys.stderr"):
            self.assertEqual(h4.main(), 2)
        health = json.loads((self.config.state_directory / "campaign_operations_h4_health.json").read_text())
        self.assertEqual(health["next_action"], "STOP_INVALID_CONFIGURATION")
        self.assertEqual(health["classification"], "invalid_configuration")
        self.assertFalse(health["service_alive"])

    def test_readiness_structured_precedence_never_uses_identity_field_names(self):
        def classify(record, status=2, stderr=""):
            supervisor = h4.Supervisor(self.config,
                runner=lambda *args, **kwargs: result(status, record, stderr), sleeper=lambda _: None)
            return supervisor.readiness()[1]
        identity = ",session_principal=manager,current_principal=manager,reader_member=true,dispatcher_member=true,phase5_transactional_member=true,scheduler_evidence_reader_member=true"
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=manager_build_contract_mismatch,build_comparison=mismatch" + identity), "manager_build_not_ready")
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=scheduler_protocol_evidence,enablement_effective=true" + identity), "scheduler_protocol_ineffective")
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=principal_role_membership,build_comparison=match" + identity), "privilege_failure")
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=enablement_ineffective;scheduler_protocol_evidence,enablement_effective=false" + identity), "production_disabled")
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=unknown_blocker" + identity), "readiness_unclassifiable")
        self.assertEqual(classify("CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=false,blockers=manager_build_contract_mismatch" + identity, status=0), "readiness_unclassifiable")
        self.assertEqual(classify("", status=1, stderr="CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,status=failed,diagnostic_code=postgresql_08006\n"), "database_failure")

    def test_work_no_work_and_request_local_failure(self):
        self.assertEqual(self.classify(result(0, summary(1, 1, 1, ids="1") + request()))[1], h4.NORMAL_ACTION)
        self.assertEqual(self.classify(result(0, summary()))[0], "no_work_completion")
        self.assertEqual(self.classify(result(0, summary(1, 1, 1, ids="1") + request(outcome="request_local_semantic_failure")))[0], "request_local_semantic_failure")

    def test_all_global_stops_and_indeterminate_precedence(self):
        expected = {"production_disabled":"STOP_DISABLED", "scheduler_protocol_ineffective":"STOP_DEGRADED_OPERATOR_REQUIRED", "manager_build_not_ready":"STOP_DEGRADED_OPERATOR_REQUIRED", "privilege_failure":"STOP_DEGRADED_OPERATOR_REQUIRED", "database_failure":h4.RETRY_ACTION}
        for reason, action in expected.items():
            with self.subTest(reason=reason):
                self.assertEqual(self.classify(result(2, summary(1, 1, 0, "true", reason, "1"), f"CAMPAIGN_OPERATIONS_MANAGER_STOPPED,reason={reason},diagnostic=ordinary\n"))[1], action)
        self.assertEqual(self.classify(result(2, summary(1, 1, 0, "true", "database_failure", "1"), "CAMPAIGN_OPERATIONS_MANAGER_STOPPED,reason=database_failure,diagnostic=commit_outcome_unknown\n"))[1], "STOP_DEGRADED_OPERATOR_REQUIRED")

    def test_malformed_interruption_and_unclassifiable_fail_closed(self):
        self.assertEqual(self.classify(result(0, ""))[1], "STOP_MALFORMED_RESULT")
        self.assertEqual(self.classify(result(0, summary() + summary()))[1], "STOP_MALFORMED_RESULT")
        self.assertEqual(self.classify(result(-9))[0], "process_interruption")
        self.assertEqual(self.classify(result(1, "", "segmentation mystery"))[1], "STOP_MALFORMED_RESULT")
        self.assertEqual(self.classify(result(-15), forced=True)[1], h4.RETRY_ACTION)

    def test_h3_request_records_exactly_match_processed_candidate_prefix(self):
        self.write_config(limit=2)
        supervisor = h4.Supervisor(h4.Config.load(self.config_path), sleeper=lambda _: None)
        duplicate = summary(2, 2, 2, ids="1:2") + request(1) + request(1)
        self.assertEqual(supervisor.classify_h3(result(0, duplicate))[1], "STOP_MALFORMED_RESULT")
        out_of_order = summary(2, 2, 2, ids="1:2") + request(2) + request(1)
        self.assertEqual(supervisor.classify_h3(result(0, out_of_order))[1], "STOP_MALFORMED_RESULT")

    def test_transport_retry_budget_and_must_stop(self):
        self.assertEqual(self.classify(result(1, "", "could not connect to database"))[1], h4.RETRY_ACTION)
        calls=[]
        def runner(args, **kwargs):
            calls.append(args)
            return result(1, "", "could not connect to database") if "readiness" in args[1] else result(0, summary())
        supervisor = h4.Supervisor(self.config, runner=runner, sleeper=lambda _: None)
        supervisor.run(max_cycles=4)
        self.assertEqual(len(calls), 2)  # finite readiness retries, never launches H3
        self.assertEqual(json.loads((self.config.state_directory / "campaign_operations_h4_state.json").read_text())["next_action"], "STOP_DEGRADED_OPERATOR_REQUIRED")

    def test_readiness_before_every_launch_and_on_restored_action(self):
        calls=[]
        def runner(args, **kwargs):
            calls.append(args[1])
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n") if "readiness" in args[1] else result(0, summary())
        supervisor = h4.Supervisor(self.config, runner=runner, sleeper=lambda _: None)
        supervisor.run(max_cycles=2)
        self.assertEqual(calls, ["--campaign-operations-production-readiness", "--campaign-operations-manager-run-once", "--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])
        supervisor.persist("work_completion", h4.NORMAL_ACTION, 0)
        calls.clear()
        supervisor = h4.Supervisor(self.config, runner=runner, sleeper=lambda _: None)
        supervisor.run(max_cycles=1)
        self.assertEqual(calls[:2], ["--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])

    def test_conflict_shutdown_and_forced_drain(self):
        self.assertEqual(self.classify(result(0, summary(), "CAMPAIGN_OPERATIONS_MANAGER_STOPPED,reason=production_disabled,diagnostic=x\n"))[1], "STOP_MALFORMED_RESULT")
        calls=[]
        supervisor = h4.Supervisor(self.config, runner=lambda *a, **k: calls.append(a) or result(0), sleeper=lambda _: None)
        supervisor.shutdown_requested = True
        supervisor.run(max_cycles=1)
        self.assertFalse(calls)  # stop requested before launch suppresses readiness/child starts
        self.executable.write_text("#!/bin/sh\nsleep 2\n")
        self.executable.chmod(0o700)
        self.write_config(graceful_drain_timeout_seconds=0.01)
        forced = h4.Supervisor(h4.Config.load(self.config_path))
        timer = threading.Timer(0.05, lambda: setattr(forced, "shutdown_requested", True))
        timer.start()
        completed = forced._run([str(self.executable), "--campaign-operations-manager-run-once", "1", "--yes"], "h3")
        timer.cancel()
        self.assertTrue(forced.forced_termination)
        self.assertLess(completed.returncode, 0)

    def test_shutdown_at_final_prelaunch_boundary_starts_no_h3_child(self):
        calls=[]
        def runner(args, **kwargs):
            calls.append(args[1])
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n")
        supervisor = h4.Supervisor(self.config, runner=runner, sleeper=lambda _: None,
                                   before_h3_launch=lambda: setattr(supervisor, "shutdown_requested", True))
        supervisor.run(max_cycles=1)
        self.assertEqual(calls, ["--campaign-operations-production-readiness"])
        self.assertEqual(supervisor.state.read()["classification"], "graceful_stop")

    def test_pending_shutdown_at_protected_popen_boundary_starts_no_h3_child(self):
        supervisor = h4.Supervisor(self.config)
        with mock.patch.object(h4.signal, "sigpending", return_value={h4.signal.SIGTERM}), \
                mock.patch.object(h4.subprocess, "Popen") as popen:
            self.assertIsNone(supervisor._run([str(self.executable), "--campaign-operations-manager-run-once", "1", "--yes"], "h3"))
        popen.assert_not_called()

    def test_restored_schedule_waits_only_remaining_time_then_preflights_h3(self):
        self.write_config(normal_interval_seconds=10)
        config = h4.Config.load(self.config_path)
        original = h4.Supervisor(config, sleeper=lambda _: None)
        original.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
        state_path = config.state_directory / "campaign_operations_h4_state.json"
        pending = original.state.read()
        pending["updated_at"] = h4.now()
        pending["next_scheduled_invocation"] = h4.future(4)
        state_path.write_text(json.dumps(pending))
        first_restore = h4.Supervisor(config, sleeper=lambda _: None)
        self.assertEqual(first_restore.restore(first_restore.state.read())[0], h4.NORMAL_ACTION)
        events, waits = [], []
        def runner(args, **kwargs):
            events.append(args[1])
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n") if "readiness" in args[1] else result(0, summary())
        supervisor = h4.Supervisor(config, runner=runner,
                                   sleeper=lambda seconds: (events.append("wait"), waits.append(seconds)))
        supervisor.run(max_cycles=1)
        self.assertEqual(events, ["wait", "--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 0)
        self.assertLess(waits[0], config.normal_interval_seconds)
        self.assertLess(waits[0], 4)

    def test_elapsed_restored_normal_schedule_does_not_add_another_interval(self):
        self.write_config(normal_interval_seconds=10)
        config = h4.Config.load(self.config_path)
        original = h4.Supervisor(config, sleeper=lambda _: None)
        original.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
        pending = original.state.read()
        pending["updated_at"] = h4.datetime.fromtimestamp(time.time() - 10, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        pending["schedule_origin_at"] = pending["updated_at"]
        pending["next_scheduled_invocation"] = h4.datetime.fromtimestamp(time.time() - 5, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        (config.state_directory / "campaign_operations_h4_state.json").write_text(json.dumps(pending))
        calls, waits = [], []
        def runner(args, **kwargs):
            calls.append(args[1])
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n") if "readiness" in args[1] else result(0, summary())
        h4.Supervisor(config, runner=runner, sleeper=waits.append).run(max_cycles=1)
        self.assertEqual(waits, [])
        self.assertEqual(calls, ["--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])

    def test_repeated_restart_after_elapsed_normal_deadline_restores_without_waiting(self):
        self.write_config(normal_interval_seconds=10)
        config = h4.Config.load(self.config_path)
        original = h4.Supervisor(config, sleeper=lambda _: None)
        original.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
        state_path = config.state_directory / "campaign_operations_h4_state.json"
        pending = original.state.read()
        pending["schedule_origin_at"] = h4.datetime.fromtimestamp(time.time() - 10, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        pending["updated_at"] = pending["schedule_origin_at"]
        pending["next_scheduled_invocation"] = h4.datetime.fromtimestamp(time.time() - 5, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        state_path.write_text(json.dumps(pending))
        first_restore = h4.Supervisor(config, sleeper=lambda _: None)
        self.assertEqual(first_restore.restore(first_restore.state.read())[0], h4.NORMAL_ACTION)
        events, waits, restored = [], [], []
        def runner(args, **kwargs):
            events.append(args[1])
            return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n") if "readiness" in args[1] else result(0, summary())
        second_restore = h4.Supervisor(config, runner=runner,
                                       sleeper=lambda seconds: (events.append("wait"), waits.append(seconds)))
        restore = second_restore.restore
        second_restore.restore = lambda existing: restored.append(restore(existing)) or restored[-1]
        second_restore.run(max_cycles=1)
        self.assertEqual(restored[0][0], h4.NORMAL_ACTION)
        self.assertEqual(waits, [])
        self.assertEqual(events, ["--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])

    def test_repeated_restart_after_elapsed_retry_deadline_preserves_count_without_backoff(self):
        self.write_config(backoff_seconds=[10, 20])
        config = h4.Config.load(self.config_path)
        original = h4.Supervisor(config, sleeper=lambda _: None)
        original.persist("database_failure", h4.RETRY_ACTION, 1)
        state_path = config.state_directory / "campaign_operations_h4_state.json"
        pending = original.state.read()
        pending["schedule_origin_at"] = h4.datetime.fromtimestamp(time.time() - 10, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        pending["updated_at"] = pending["schedule_origin_at"]
        pending["next_scheduled_invocation"] = h4.datetime.fromtimestamp(time.time() - 5, h4.timezone.utc).isoformat().replace("+00:00", "Z")
        state_path.write_text(json.dumps(pending))
        first_restore = h4.Supervisor(config, sleeper=lambda _: None)
        self.assertEqual(first_restore.restore(first_restore.state.read()), (h4.RETRY_ACTION, 1, pending["next_scheduled_invocation"]))
        events, waits, restored, invocation_retry_counts = [], [], [], []
        def runner(args, **kwargs):
            events.append(args[1])
            if "readiness" in args[1]:
                return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n")
            invocation_retry_counts.append(json.loads(state_path.read_text())["retry_count"])
            return result(0, summary())
        second_restore = h4.Supervisor(config, runner=runner,
                                       sleeper=lambda seconds: (events.append("wait"), waits.append(seconds)))
        restore = second_restore.restore
        second_restore.restore = lambda existing: restored.append(restore(existing)) or restored[-1]
        second_restore.run(max_cycles=1)
        self.assertEqual(restored[0][0:2], (h4.RETRY_ACTION, 1))
        self.assertEqual(invocation_retry_counts, [1])
        self.assertEqual(waits, [])
        self.assertEqual(events, ["--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])

    def test_restored_retry_backoff_preserves_count_then_preflights_h3(self):
        self.write_config(backoff_seconds=[10, 20])
        config = h4.Config.load(self.config_path)
        original = h4.Supervisor(config, sleeper=lambda _: None)
        original.persist("database_failure", h4.RETRY_ACTION, 1)
        pending = original.state.read()
        pending["updated_at"] = h4.now()
        pending["next_scheduled_invocation"] = h4.future(4)
        (config.state_directory / "campaign_operations_h4_state.json").write_text(json.dumps(pending))
        events, waits, invocation_retry_counts = [], [], []
        def runner(args, **kwargs):
            events.append(args[1])
            if "readiness" in args[1]:
                return result(0, "CAMPAIGN_OPERATIONS_PRODUCTION_READINESS,ready=true,blockers=none\n")
            invocation_retry_counts.append(json.loads((config.state_directory / "campaign_operations_h4_state.json").read_text())["retry_count"])
            return result(0, summary())
        h4.Supervisor(config, runner=runner,
                      sleeper=lambda seconds: (events.append("wait"), waits.append(seconds))).run(max_cycles=1)
        self.assertEqual(events, ["wait", "--campaign-operations-production-readiness", "--campaign-operations-manager-run-once"])
        self.assertEqual(invocation_retry_counts, [1])
        self.assertEqual(len(waits), 1)
        self.assertGreater(waits[0], 0)
        self.assertLess(waits[0], config.backoff_seconds[0])

    def test_restored_retry_must_remain_below_retry_budget(self):
        original = h4.Supervisor(self.config, sleeper=lambda _: None)
        original.persist("database_failure", h4.RETRY_ACTION, 1)
        state_path = self.config.state_directory / "campaign_operations_h4_state.json"
        self.assertEqual(original.restore(original.state.read())[0], h4.RETRY_ACTION)
        for retry_count in (self.config.retry_budget, self.config.retry_budget + 1):
            original.persist("database_failure", h4.RETRY_ACTION, 1)
            malformed = original.state.read()
            malformed["retry_count"] = retry_count
            state_path.write_text(json.dumps(malformed))
            with self.subTest(retry_count=retry_count):
                self.assertEqual(original.restore(original.state.read())[0], "STOP_MALFORMED_RESULT")

    def test_restored_schedule_must_be_parseable_bounded_and_action_consistent(self):
        self.write_config(normal_interval_seconds=10)
        config = h4.Config.load(self.config_path)
        supervisor = h4.Supervisor(config, sleeper=lambda _: None)
        state_path = config.state_directory / "campaign_operations_h4_state.json"
        for schedule in (
                "not-a-timestamp",
                h4.datetime.fromtimestamp(time.time() + 11, h4.timezone.utc).isoformat().replace("+00:00", "Z"),
                h4.datetime.fromtimestamp(time.time() + 1, h4.timezone.utc).replace(tzinfo=None).isoformat()):
            supervisor.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
            malformed = supervisor.state.read()
            malformed["next_scheduled_invocation"] = schedule
            state_path.write_text(json.dumps(malformed))
            with self.subTest(schedule=schedule):
                self.assertEqual(supervisor.restore(supervisor.state.read())[0], "STOP_MALFORMED_RESULT")
        supervisor.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
        malformed = supervisor.state.read()
        malformed.update({"classification": "restart_complete_state", "next_action": h4.RESTORE_ACTION,
                          "restored_next_action": h4.NORMAL_ACTION})
        malformed.pop("schedule_origin_at")
        state_path.write_text(json.dumps(malformed))
        self.assertEqual(supervisor.restore(supervisor.state.read())[0], "STOP_MALFORMED_RESULT")

    def test_restart_of_persisted_stop_preserves_truthful_stopped_health(self):
        self.supervisor.persist("production_disabled", "STOP_DISABLED", 0)

        calls = []
        waits = []

        def runner(args, **kwargs):
            calls.append(args)
            return result(0)

        restarted = h4.Supervisor(
            self.config,
            runner=runner,
            sleeper=waits.append,
        )

        restarted.run(max_cycles=1)

        self.assertEqual(calls, [])
        self.assertEqual(waits, [])

        state = restarted.state.read()
        health = json.loads(
            (self.config.state_directory / "campaign_operations_h4_health.json").read_text()
        )

        self.assertEqual(state["classification"], "production_disabled")
        self.assertEqual(state["next_action"], "STOP_DISABLED")
        self.assertEqual(state["status"], "stopped")
        self.assertFalse(state["service_alive"])

        self.assertEqual(health["classification"], "production_disabled")
        self.assertEqual(health["next_action"], "STOP_DISABLED")
        self.assertEqual(health["status"], "stopped")
        self.assertFalse(health["service_alive"])

        # A repeated unresolved restart must remain stopped too.
        restarted_again = h4.Supervisor(
            self.config,
            runner=runner,
            sleeper=waits.append,
        )
        restarted_again.run(max_cycles=1)

        state_again = restarted_again.state.read()
        self.assertEqual(state_again["next_action"], "STOP_DISABLED")
        self.assertEqual(state_again["status"], "stopped")
        self.assertFalse(state_again["service_alive"])

    def test_restored_stop_remains_stopped_without_waiting_or_launching(self):
        self.supervisor.persist("production_disabled", "STOP_DISABLED", 0)
        self.assertEqual(self.supervisor.restore(self.supervisor.state.read())[0], "STOP_DISABLED")
        calls, waits = [], []
        supervisor = h4.Supervisor(self.config,
                                   runner=lambda args, **kwargs: calls.append(args) or result(0),
                                   sleeper=waits.append)
        supervisor.run(max_cycles=1)
        self.assertEqual(calls, [])
        self.assertEqual(waits, [])

    def test_restart_and_health_record(self):
        self.supervisor.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
        first_state = self.supervisor.state.read()
        self.assertGreater(first_state["next_scheduled_invocation"], first_state["updated_at"])
        restored = self.supervisor.restore(self.supervisor.state.read())
        self.assertEqual(restored[0], h4.NORMAL_ACTION)
        self.assertEqual(self.supervisor.state.read()["next_action"], h4.RESTORE_ACTION)
        self.config.state_directory.mkdir(parents=True, exist_ok=True)
        state_path = self.config.state_directory / "campaign_operations_h4_state.json"
        def missing_classification(record):
            record.pop("classification")
        for label, mutation in (
                ("non_integer_retry", {"retry_count": "oops"}), ("negative_retry", {"retry_count": -1}),
                ("out_of_range_retry", {"retry_count": 3}), ("missing_classification", missing_classification),
                ("unknown_classification", {"classification": "unknown"}),
                ("contradictory_classification_action", {"classification": "privilege_failure"}),
                ("unknown_action", {"next_action": "unknown"}), ("dead_scheduled_service", {"service_alive": False}),
                ("contradictory_status", {"status": "stopped"}), ("unknown_field", {"unreviewed": True}),
                ("mismatched_environment", {"target_environment": "other"}),
                ("incomplete_invocation", {"complete": False})):
            self.supervisor.persist("work_completion", h4.NORMAL_ACTION, 0, work_occurred=True)
            malformed = self.supervisor.state.read()
            if callable(mutation):
                mutation(malformed)
            else:
                malformed.update(mutation)
            state_path.write_text(json.dumps(malformed))
            with self.subTest(mutation=label):
                self.assertEqual(self.supervisor.restore(self.supervisor.state.read())[0], "STOP_MALFORMED_RESULT")
        state_path.write_text("{broken")
        self.assertEqual(self.supervisor.restore(self.supervisor.state.read())[0], "STOP_MALFORMED_RESULT")
        self.supervisor.persist("production_disabled", "STOP_DISABLED", 0)
        self.assertFalse(self.supervisor.state.read()["service_alive"])
        self.supervisor.resolve_stop_state()
        resolved = self.supervisor.state.read()
        self.assertEqual(resolved["classification"], "operator_stop_resolution")
        self.assertFalse(resolved["service_alive"])
        self.assertEqual(self.supervisor.restore(resolved)[0], h4.NORMAL_ACTION)
        health = json.loads((self.config.state_directory / "campaign_operations_h4_health.json").read_text())
        for field in ("deployment_identity", "deployment_execution_identity", "target_database_identity", "target_environment", "postgresql_login_identity", "service_alive", "classification", "next_action", "retry_count", "terminal_result_validity", "graceful_drain_state", "duplicate_drift_detected", "invocation_start", "invocation_end", "child_exit_status", "readiness_check_at", "readiness_result", "readiness_blocker", "last_valid_work_result", "last_valid_no_work_result"):
            self.assertIn(field, health)
        self.assertIsNone(health["duplicate_drift_detected"])
        self.assertTrue((self.config.state_directory / "campaign_operations_h4_transitions.jsonl").read_text())

    def test_static_deployment_and_no_database_authority(self):
        plist = (ROOT / "Deployment/CampaignOperationsH4/com.expertadvisor.campaign-operations-h4.plist").read_text()
        configuration = json.loads((ROOT / "Deployment/CampaignOperationsH4/campaign-operations-h4.example.json").read_text())
        env = (ROOT / "Deployment/CampaignOperationsH4/campaign-operations-h4.connection.env.example").read_text()
        source = (ROOT / "Scripts/CampaignOperationsH4Supervisor.py").read_text()
        self.assertIn("com.expertadvisor.campaign-operations-h4", plist)
        self.assertIn("<key>UserName</key><string>expertadvisor-h4</string>", plist)
        self.assertIn("<key>Crashed</key><true/>", plist)
        self.assertIn("<key>ThrottleInterval</key><integer>60</integer>", plist)
        self.assertEqual(configuration["deployment_execution_identity"], "expertadvisor-h4")
        self.assertEqual(configuration["postgresql_login_identity"], "REPLACE_WITH_MANAGER_LOGIN")
        self.assertIn("CAMPAIGN_OPERATIONS_PRODUCTION_MANAGER_DB_USER=REPLACE_WITH_MANAGER_LOGIN", env)
        self.assertNotIn("PGSERVICE=", env)
        self.assertIn("--campaign-operations-manager-run-once", source)
        for forbidden in ("psycopg", "CREATE TABLE", "advisory", "heartbeat", "--schedule-experiments"):
            self.assertNotIn(forbidden, source)
        self.assertIsNone(re.search(r"\blease\b", source, re.IGNORECASE))


if __name__ == "__main__":
    unittest.main()
