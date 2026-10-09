#!/usr/bin/env python3
"""Bounded native capacity/priority qualification using inert private workers.

This deliberately reuses the Phase24T SCRAM harness and never connects to a
production database.  The workers are process-control fixtures; they do not
load Metal or model data.
"""
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "displacement", ROOT / "Tests/SchedulerDisplacementRecoveryTests.py")
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
m.PHASE_T = True
m.CLI = ROOT / "DerivedData/ExpertAdvisor/Phase24U/build/LSTM_Release"


class CapacityQualification(m.Qualification):
    def __init__(self, output):
        super().__init__(output)
        self.scheduler_workers = {}
        self.scheduler_discoveries = []
        self.scheduler_cleanup_errors = []

    def registry(self):
        super().registry()
        # ANALYZE is a separate worker role, outside the semantic registry.
        self.analyze_helper = self.out / "synthetic-analyze-worker"
        self.analyze_helper.write_text(
            "#!/usr/bin/env python3\n"
            "import argparse, signal, time\n"
            "p = argparse.ArgumentParser()\n"
            "p.add_argument('--analyze-experiment', type=int, required=True)\n"
            "p.add_argument('--scheduler-worker-attempt-id', type=int)\n"
            "p.add_argument('--auto-generate-reports', action='store_true')\n"
            "p.add_argument('--experiment-report-dir')\n"
            "a = p.parse_args()\n"
            "signal.signal(signal.SIGTERM, lambda *_: exit(0))\n"
            "print(f'SYNTHETIC_ANALYZE_READY,{a.analyze_experiment}', flush=True)\n"
            "while True: time.sleep(0.2)\n"
        )
        self.analyze_helper.chmod(0o700)
        self.artifacts[13, "analyze"] = self.helper
        # The priority scenario launches a scheduler-owned ANALYZE child before
        # the capacity scenario. Install its private observation filter as soon
        # as the helper exists, before any scheduler cycle can launch it.
        self.enable_scheduler_worker_observation()

    def enable_scheduler_worker_observation(self):
        """Keep the base safety filter, admitting only our exact helper path."""
        wrapper = self.out / "bin" / "ps"
        helper = str(self.analyze_helper)
        wrapper.write_text(
            "#!/usr/bin/python3 -B\n"
            "import os, subprocess, sys\n"
            "allowed=set(open(os.environ['EA_TEST_OWNED_PIDS']).read().split())\n"
            f"marker={helper!r}\n"
            "def private_pids():\n"
            " r=subprocess.run(['/bin/ps','-axo','pid=,command='],capture_output=True,text=True,timeout=3)\n"
            " out=set()\n"
            " for line in r.stdout.splitlines():\n"
            "  if marker in line and '--analyze-experiment=' in line:\n"
            "   fields=line.strip().split(None,1)\n"
            "   if fields and fields[0].isdigit(): out.add(fields[0])\n"
            " return out\n"
            "visible=allowed|private_pids()\n"
            "if '-p' in sys.argv:\n"
            " pid=sys.argv[sys.argv.index('-p')+1]\n"
            " if pid not in visible: sys.exit(1)\n"
            " sys.exit(subprocess.run(['/bin/ps']+sys.argv[1:],timeout=3).returncode)\n"
            "if sys.argv[1:2]==['-axo']:\n"
            " for pid in sorted(visible):\n"
            "  r=subprocess.run(['/bin/ps','-p',pid,'-o']+sys.argv[2:],capture_output=True,text=True,timeout=3)\n"
            "  sys.stdout.write(r.stdout)\n"
            " sys.exit(0)\n"
            "sys.exit(1)\n")
        wrapper.chmod(0o700)

    def scheduler_args(self, train, infer, once=True):
        args = super().scheduler_args(train, infer, once)
        if hasattr(self, "analyze_helper"):
            args = [
                str(a) for a in args
                if not str(a).startswith("--analyze-worker=")
            ]
            args.append("--analyze-worker=" + str(self.analyze_helper))
        return args

    def ensure_model(self, eid):
        self.sql(
            f"WITH m AS (INSERT INTO model(experiment_id,name,comment) "
            f"VALUES({eid},'synthetic-{eid}','capacity fixture') RETURNING model_id) "
            f"UPDATE experiment SET last_model_id=m.model_id FROM m WHERE experiment_id={eid};")

    def worker_capacity(self, phase):
        return int(self.sql(
            "SELECT count(*) FROM experiment_scheduler_worker_attempt "
            f"WHERE capacity_class={m.quote(phase)} AND lifecycle_state IN {m.ACTIVE};"))

    def cycle_w(self, name, train=2, infer=1, analyze=18):
        args = self.scheduler_args(train, infer)
        args = [a for a in args if a != "--max-analyze-procs=0"]
        args += [f"--max-analyze-procs={analyze}",
                 "--phase-priority=train:infer:analyze"]
        try:
            result = self.run(name, args)
            self.snapshot(name)
        finally:
            # A scheduler can launch a child before reporting a later cycle
            # error; discover it even when run() raises.
            self.track_scheduler_analyze_workers()
        return result.stdout + result.stderr

    def scheduler_attempt_rows(self, first, last):
        raw = self.sql(
            "SELECT e.experiment_id,a.worker_attempt_id,a.worker_pid,"
            "a.worker_process_start_identity,e.status,a.lifecycle_state,a.command_line "
            "FROM experiment e JOIN experiment_scheduler_worker_attempt a "
            "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
            f"WHERE e.experiment_id BETWEEN {first} AND {last} ORDER BY e.experiment_id;")
        rows = []
        for line in raw.splitlines():
            fields = line.split("|", 6)
            if len(fields) == 7:
                rows.append({"experiment": int(fields[0]), "attempt": int(fields[1]),
                             "pid": int(fields[2]), "start": fields[3],
                             "status": fields[4], "state": fields[5],
                             "command": fields[6]})
        return rows

    def private_analyze_pids(self):
        result = subprocess.run(
            ["/bin/ps", "-axo", "pid=,command="],
            text=True, capture_output=True, timeout=5)
        marker = str(self.analyze_helper)
        pids = set()
        for line in result.stdout.splitlines():
            if marker in line and "--analyze-experiment=" in line:
                fields = line.strip().split(None, 1)
                if fields and fields[0].isdigit():
                    pids.add(int(fields[0]))
        return pids

    def inspect_with_diagnostics(self, pid):
        started = time.monotonic()
        try:
            result = subprocess.run(
                [str(self.helper), f"--inspect-managed-test-process={pid}"],
                env=self.env, text=True, capture_output=True, timeout=5)
            elapsed = time.monotonic() - started
            stdout = result.stdout.strip()
            identity = stdout.split("|") if result.returncode == 0 and stdout else None
            details = {
                "exit_status": result.returncode,
                "stdout": result.stdout,
                "stderr": result.stderr,
                "elapsed_seconds": elapsed,
            }
            if result.returncode != 0:
                diagnostic_started = time.monotonic()
                try:
                    diagnostic = subprocess.run(
                        [str(self.helper),
                         f"--diagnose-managed-test-process={pid}"],
                        env=self.env, text=True, capture_output=True, timeout=5)
                    details.update({
                        "diagnostic_exit_status": diagnostic.returncode,
                        "diagnostic_stdout": diagnostic.stdout,
                        "diagnostic_stderr": diagnostic.stderr,
                        "diagnostic_elapsed_seconds": time.monotonic() - diagnostic_started,
                    })
                except subprocess.TimeoutExpired as diagnostic_error:
                    details.update({
                        "diagnostic_exit_status": None,
                        "diagnostic_stdout": diagnostic_error.stdout or "",
                        "diagnostic_stderr": diagnostic_error.stderr or "",
                        "diagnostic_elapsed_seconds": time.monotonic() - diagnostic_started,
                        "diagnostic_exception": "TimeoutExpired",
                    })
            return identity, details
        except subprocess.TimeoutExpired as error:
            return None, {
                "exit_status": None,
                "stdout": error.stdout or "",
                "stderr": error.stderr or "",
                "elapsed_seconds": time.monotonic() - started,
                "exception": "TimeoutExpired",
            }

    def record_inspection_failure(self, discovery, details, classification, observed=None):
        payload = {
            "experiment_id": discovery["experiment"],
            "worker_attempt_id": discovery["attempt"],
            "worker_pid": discovery["pid"],
            "expected_process_start_identity": discovery["start"],
            "expected_process_group_id": discovery["group"],
            "scheduler_lifecycle_state": discovery["lifecycle"],
            "classification": classification,
            "observed_identity": observed,
            **details,
        }
        with (self.out / "scheduler-analyze-inspection.jsonl").open("a") as f:
            f.write(json.dumps(payload) + "\n")

    def track_scheduler_analyze_workers(self):
        """Discover every active scheduler-owned synthetic ANALYZE attempt."""
        raw = self.sql(
            "SELECT e.experiment_id,a.worker_attempt_id,a.worker_pid,"
            "a.worker_process_start_identity,a.worker_process_group_id,"
            "a.lifecycle_state,a.command_line "
            "FROM experiment e JOIN experiment_scheduler_worker_attempt a "
            "ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id "
            "WHERE a.capacity_class='analyze' AND a.worker_pid IS NOT NULL "
            "AND a.lifecycle_state IN ('reserved','spawned','running','observed');")
        for line in raw.splitlines():
            fields = line.split("|")
            if len(fields) != 7:
                continue
            eid, attempt, pid = map(int, fields[:3])
            discovery = {"experiment": eid, "attempt": attempt, "pid": pid,
                         "start": fields[3], "group": int(fields[4]),
                         "lifecycle": fields[5], "persisted_command": fields[6]}
            if not any(d["attempt"] == attempt for d in self.scheduler_discoveries):
                self.scheduler_discoveries.append(discovery)
                with (self.out / "scheduler-analyze-discoveries.jsonl").open("a") as f:
                    f.write(json.dumps(discovery) + "\n")
            identity = None
            failure = None
            for _ in range(25):
                identity, details = self.inspect_with_diagnostics(pid)
                if not identity or len(identity) != 5:
                    classification = ("inspection_subprocess_failure"
                                      if details["exit_status"] not in (0, None)
                                      else "missing_process_observation")
                    self.record_inspection_failure(
                        discovery, details, classification, identity)
                    time.sleep(0.02)
                    continue
                if identity[0] != str(pid):
                    failure = f"PID mismatch for {pid}: {identity}"
                    self.record_inspection_failure(
                        discovery, details, "process_start_identity_mismatch", identity)
                    break
                if identity[2] != fields[3]:
                    failure = f"process-start identity mismatch for PID {pid}: {identity}"
                    self.record_inspection_failure(
                        discovery, details, "process_start_identity_mismatch", identity)
                    break
                if identity[1] != str(discovery["group"]):
                    self.record_inspection_failure(
                        discovery, details, "process_group_mismatch", identity)
                    time.sleep(0.02)
                    continue
                command = identity[4]
                if (str(self.analyze_helper) not in command or
                        f"--analyze-experiment={eid}" not in command or
                        f"--scheduler-worker-attempt-id={attempt}" not in command):
                    self.record_inspection_failure(
                        discovery, details, "command_line_mismatch", identity)
                    # The scheduler may still have the child at its exec gate;
                    # bounded polling permits that transient representation.
                    time.sleep(0.02)
                    continue
                break
            if failure:
                discovery["failure"] = failure
                with (self.out / "scheduler-analyze-discoveries.jsonl").open("a") as f:
                    f.write(json.dumps({"event": "failure", **discovery}) + "\n")
                raise AssertionError(failure)
            if not identity or identity[0] != str(pid) or identity[2] != fields[3]:
                discovery["failure"] = "identity unavailable after bounded stabilization"
                with (self.out / "scheduler-analyze-discoveries.jsonl").open("a") as f:
                    f.write(json.dumps({"event": "failure", **discovery}) + "\n")
                raise AssertionError(
                    f"scheduler ANALYZE identity unavailable after bounded polling for PID {pid}")
            if (str(self.analyze_helper) not in identity[4] or
                    identity[1] != str(discovery["group"]) or
                    f"--analyze-experiment={eid}" not in identity[4] or
                    f"--scheduler-worker-attempt-id={attempt}" not in identity[4]):
                discovery["failure"] = "exact command identity unavailable after bounded stabilization"
                with (self.out / "scheduler-analyze-discoveries.jsonl").open("a") as f:
                    f.write(json.dumps({"event": "failure", **discovery}) + "\n")
                raise AssertionError(
                    f"scheduler ANALYZE command identity unavailable for PID {pid}")
            previous = self.scheduler_workers.get(pid)
            if previous and previous["identity"] != identity:
                raise AssertionError(f"scheduler ANALYZE PID identity changed: {pid}")
            self.scheduler_workers[pid] = {
                "experiment": eid, "attempt": attempt,
                "process_group_id": discovery["group"], "identity": identity}

    def recover_scheduler_discoveries_for_cleanup(self):
        """Retry only recorded DB discoveries; never infer ownership from ps."""
        for discovery in self.scheduler_discoveries:
            pid = discovery["pid"]
            if pid in self.scheduler_workers:
                continue
            for _ in range(25):
                identity, details = self.inspect_with_diagnostics(pid)
                if (identity and len(identity) == 5 and identity[0] == str(pid) and
                        identity[2] == discovery["start"] and
                        identity[1] == str(discovery["group"]) and
                        str(self.analyze_helper) in identity[4] and
                        f"--analyze-experiment={discovery['experiment']}" in identity[4] and
                        f"--scheduler-worker-attempt-id={discovery['attempt']}" in identity[4]):
                    self.scheduler_workers[pid] = {
                        "experiment": discovery["experiment"],
                        "attempt": discovery["attempt"],
                        "process_group_id": discovery["group"],
                        "identity": identity}
                    break
                if not identity or len(identity) != 5:
                    classification = ("inspection_subprocess_failure"
                                      if details["exit_status"] not in (0, None)
                                      else "missing_process_observation")
                elif identity[2] != discovery["start"]:
                    classification = "process_start_identity_mismatch"
                elif identity[1] != str(discovery["group"]):
                    classification = "process_group_mismatch"
                else:
                    classification = "command_line_mismatch"
                self.record_inspection_failure(
                    discovery, details, classification, identity)
                time.sleep(0.02)

    def terminate_scheduler_analyze_workers(self):
        """SIGTERM only exact identities, then prove every one exited."""
        for pid, record in list(self.scheduler_workers.items()):
            observed = self.inspect(pid)
            if not observed:
                continue
            assert observed == record["identity"], (
                f"refusing signal after identity change for private PID {pid}")
            assert observed[1] == str(record["process_group_id"])
            assert f"--analyze-experiment={record['experiment']}" in observed[4]
            assert str(self.analyze_helper) in observed[4]
            assert f"--scheduler-worker-attempt-id={record['attempt']}" in observed[4]
            if observed[1] == str(pid) and int(observed[1]) > 1:
                os.killpg(pid, signal.SIGTERM)
            else:
                os.kill(pid, signal.SIGTERM)
            with (self.out / "signals.jsonl").open("a") as f:
                f.write(json.dumps({"scheduler_analyze_experiment": record["experiment"],
                                    "identity": observed, "signal": int(signal.SIGTERM)}) + "\n")
        for pid in list(self.scheduler_workers):
            self.wait(lambda pid=pid: not self.inspect(pid),
                      f"scheduler ANALYZE worker {pid} exit", seconds=5)
        survivors = self.private_analyze_pids()
        assert not survivors, f"untracked synthetic ANALYZE workers survive: {sorted(survivors)}"

    def test_scheduler_managed_analyze_capacity(self):
        """Qualify launch, observation, cap enforcement, and orphan cleanup."""
        first, last = 981000, 981018
        for eid in range(first, last + 1):
            self.queue_analyze(eid)

        active = []
        output = ""
        for cycle in range(1, 5):
            output += self.cycle_w(f"native12-analyze-launch-{cycle}")
            active = self.scheduler_attempt_rows(first, last)
            if len(active) == 18:
                break
        assert len(active) == 18, f"expected 18 attempts, got {len(active)}"
        active_ids = {row["experiment"] for row in active}
        assert active_ids == set(range(first, first + 18))
        assert self.sql(f"SELECT status FROM experiment WHERE experiment_id={last};") == "pending"
        assert self.sql(
            f"SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id={last};"
        ) == ""

        for row in active:
            observed = self.inspect(row["pid"])
            assert observed and observed[0] == str(row["pid"])
            assert observed[2] == row["start"]
            assert str(self.analyze_helper) in observed[4]
            assert f"--analyze-experiment={row['experiment']}" in observed[4]
            assert row["pid"] in self.scheduler_workers
            assert self.scheduler_workers[row["pid"]]["identity"] == observed
            assert not self.state(row["pid"]).startswith("T")
        assert self.worker_capacity("analyze") == 18

        second = self.cycle_w("native12-analyze-cap-enforcement")
        assert self.worker_capacity("analyze") == 18
        assert len(self.scheduler_attempt_rows(first, last)) == 18
        assert self.sql(f"SELECT status FROM experiment WHERE experiment_id={last};") == "pending"
        assert self.sql(
            f"SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id={last};"
        ) == ""
        self.results["scheduler_managed_analyze_capacity"] = (
            "PASS; 18 scheduler-launched workers observed by exact PID/start/command identity; "
            "nineteenth pending and second cycle remained capped")
        (self.out / "native12-evidence.json").write_text(json.dumps({
            "capacity": 18, "pending_experiment": last,
            "active": active, "launch_output_tail": output[-4000:],
            "second_cycle_output_tail": second[-4000:],
        }, indent=2))

        # Terminate only workers whose current identity still matches the
        # scheduler attempt captured above, then let normal reconciliation
        # clear the resulting terminal attempts.
        self.terminate_scheduler_analyze_workers()

        self.run("native12-analyze-reconcile", self.scheduler_args(2, 1) + [
            "--max-analyze-procs=18", "--phase-priority=train:infer:analyze",
            "--recover-orphans-only"])
        assert self.worker_capacity("analyze") == 0
        remaining = self.scheduler_attempt_rows(first, last)
        assert not remaining, f"stale active attempts remain: {remaining}"
        assert all(not self.inspect(pid) for pid in self.scheduler_workers)
        self.results["scheduler_managed_analyze_cleanup"] = (
            "PASS; identity-verified termination and bounded reconciliation cleared all active attempts")
        self.scheduler_workers.clear()

    def queue_analyze(self, eid, priority="normal"):
        # ANALYZE candidates are queued without a stopped worker attempt.
        self.sql(
            "SET expertadvisor.scheduler_protocol_generation='52';"
            "INSERT INTO experiment("
            "experiment_id,symbol,prediction_horizon,c_next_threshold,"
            "core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,"
            "train_start,train_end,infer_start,infer_end,status,phase,"
            "duplicate_nonce,scheduler_priority,resume_requested,"
            "scheduler_resume_origin,model_input_width,"
            "model_input_semantic_layout_version"
            ") VALUES("
            f"{eid},'synthetic{eid}',1,0.0008,1,1,20,20,"
            "'2020-01-01','2020-02-01','2020-02-01','2020-03-01',"
            f"'pending','analyze',{eid},{m.quote(priority)},"
            "false,'none',171,13);"
        )
        self.ensure_model(eid)

    def test_capacity(self):
        # Two active native TRAIN fixtures are independently observed and
        # retained under the configured TRAIN=2 capacity.
        self.launch(980100, phase="train", layout=13, priority="normal")
        self.launch(980101, phase="train", layout=13, priority="normal")
        out = self.cycle_w("train-capacity-2")
        assert self.worker_capacity("train") == 2
        assert all(not self.state(self.workers[e]["pid"]).startswith("T")
                   for e in (980100, 980101))
        self.results["train_capacity_2"] = "PASS; two independent active attempts and PIDs"
        self.retire()

        # With TRAIN owning execution, an eligible INFER worker remains
        # stopped/pending.  Once TRAIN drains, the same INFER attempt resumes.
        self.launch(980110, phase="train", layout=13, priority="normal")
        self.launch(980111, phase="train", layout=13, priority="normal")
        self.launch(980112, phase="infer", layout=9, priority="normal",
                    status="pending", origin="operator")
        self.cycle_w("train-infer-exclusive")
        self.expect(980112, "pending", True)
        assert self.worker_capacity("train") == 2
        infer_identity = self.inspect(self.workers[980112]["pid"])
        infer_attempt = self.sql(
            "SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id=980112;")
        # Drain only TRAIN. Keep the excluded INFER process and persisted
        # attempt so normal reconciliation must readmit that exact worker.
        for eid in (980110, 980111):
            self.signal_worker(eid, signal.SIGTERM)
            self.workers[eid]["process"].wait(timeout=4)
        self.cycle_w("infer-after-train-drain")
        # Scheduler automatically resumes eligible INFER after TRAIN drains.
        self.expect(980112, "running", False)
        assert self.inspect(self.workers[980112]["pid"]) == infer_identity
        assert self.sql(
            "SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id=980112;") == infer_attempt
        assert self.sql(
            "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=980112;") == "1"
        assert self.worker_capacity("train") == 0
        assert self.worker_capacity("infer") == 1
        self.results["train_infer_exclusive"] = (
            "PASS; INFER remained stopped during TRAIN execution; identical PID/start identity "
            "and single original attempt automatically resumed after TRAIN drain")
        self.retire()

        # Eighteen active ANALYZE fixtures are observed in one capacity class;
        # a nineteenth pending, model-backed candidate remains pending.
        for eid in range(980200, 980218):
            self.launch(eid, phase="analyze", layout=13, priority="normal")
            self.ensure_model(eid)
        self.cycle_w("analyze-capacity-18")
        assert self.worker_capacity("analyze") == 18
        self.queue_analyze(980218)
        self.cycle_w("analyze-capacity-19-rejected")
        assert self.sql(
            "SELECT status FROM experiment WHERE experiment_id=980218;"
        ) == "pending"
        assert self.sql(
            "SELECT active_scheduler_worker_attempt_id "
            "FROM experiment WHERE experiment_id=980218;"
        ) == ""
        assert self.worker_capacity("analyze") == 18
        self.results["analyze_capacity_18"] = "PASS; 18 active attempts, nineteenth not admitted"
        self.retire()
        # retire() handles registered worker fixtures, but the nineteenth
        # candidate has no worker and must be retired separately.
        self.sql(
            "SET expertadvisor.scheduler_protocol_generation='52';"
            "UPDATE experiment SET status='failed' "
            "WHERE experiment_id=980218 AND status='pending';"
        )

    def test_priority(self):
        # A high-priority ANALYZE candidate drains lower-priority TRAIN and
        # INFER workers under the existing global phase-priority policy.
        self.launch(980300, phase="train", layout=13, priority="low")
        self.launch(980301, phase="infer", layout=9, priority="low",
                    status="pending", origin="operator")
        self.queue_analyze(980302, priority="high")
        first = self.cycle_w("high-analyze-drain-1")
        self.expect(980300, "pending", True)
        self.expect(980301, "pending", True)
        # The scheduler admitted high-priority ANALYZE in the first cycle.
        # The cycle-level tracker has registered its exact PID and command;
        # terminate it before the next Native12 fixture is created.
        assert self.sql(
            "SELECT status FROM experiment WHERE experiment_id=980302;"
        ) == "running"
        assert self.sql(
            "SELECT active_scheduler_worker_attempt_id "
            "FROM experiment WHERE experiment_id=980302;"
        ) != ""
        assert self.worker_capacity("train") == 0
        assert self.worker_capacity("infer") == 0
        assert self.worker_capacity("analyze") == 1
        assert "selected_phase=analyze" in first
        assert "SCHEDULER_EXEC:" in first
        assert "--analyze-experiment=980302" in first
        self.results["high_priority_analyze"] = (
            "PASS admission/displacement; high ANALYZE selected and launched "
            "while lower TRAIN/INFER remained stopped."
        )
        self.terminate_scheduler_analyze_workers()
        self.scheduler_workers.clear()
        self.retire()

    def tests(self):
        self.test_capacity()
        self.test_priority()
        self.test_scheduler_managed_analyze_capacity()

    def cleanup(self):
        # Scheduler-launched workers are not in the base harness's fixture map.
        # Clean them first, using the same identity gate as the test body.
        self.recover_scheduler_discoveries_for_cleanup()
        for pid, record in list(self.scheduler_workers.items()):
            try:
                observed = self.inspect(pid)
                if observed == record["identity"]:
                    if (f"--analyze-experiment={record['experiment']}" not in observed[4] or
                            f"--scheduler-worker-attempt-id={record['attempt']}" not in observed[4] or
                            str(self.analyze_helper) not in observed[4] or
                            observed[1] != str(record["process_group_id"])):
                        self.scheduler_cleanup_errors.append(
                            f"command identity changed; refused signal for private PID {pid}")
                        continue
                    if observed[1] == str(pid) and int(observed[1]) > 1:
                        os.killpg(pid, signal.SIGTERM)
                    else:
                        os.kill(pid, signal.SIGTERM)
                    self.wait(lambda pid=pid: not self.inspect(pid),
                              f"cleanup scheduler worker {pid}", seconds=5)
                elif observed:
                    self.scheduler_cleanup_errors.append(
                        f"identity changed; refused signal for private PID {pid}")
            except Exception as error:
                self.scheduler_cleanup_errors.append(str(error))
        self.scheduler_workers.clear()
        if hasattr(self, "analyze_helper"):
            survivors = self.private_analyze_pids()
            if survivors:
                self.scheduler_cleanup_errors.append(
                    f"synthetic ANALYZE workers remain after cleanup: {sorted(survivors)}")
        if self.scheduler_cleanup_errors:
            # Do not stop PostgreSQL while an unidentified synthetic worker
            # remains; the operator must inspect the retained private cluster.
            self.results["cleanup"] = self.scheduler_cleanup_errors
            (self.out / "results.json").write_text(
                json.dumps(self.results, indent=2))
            raise AssertionError("scheduler worker cleanup failed: " +
                                 repr(self.scheduler_cleanup_errors))
        super().cleanup()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("usage: SchedulerCapacityPriorityQualification.py EVIDENCE_DIR")
    q = CapacityQualification(Path(sys.argv[1]))
    try:
        q.setup()
        q.tests()
    except Exception as error:
        q.results["qualification"] = {
            "status": "FAILED_OR_BLOCKED",
            "diagnostic": str(error),
        }
        raise
    finally:
        q.cleanup()
    print(json.dumps(q.results, indent=2))
