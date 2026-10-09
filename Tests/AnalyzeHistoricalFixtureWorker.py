"""Private SCRAM restore and single managed ANALYZE execution for Phase 24X."""
import json
import os
from pathlib import Path
import secrets
import shutil
import signal
import subprocess
import time

from AnalyzeHistoricalFixtureQualification import PG, ROUTING, PRODUCTS, OBSERVER, sha, verify_analysis, select_fixture_registry
from AnalyzeResourceQualificationPreflight import vm_counters

if not __debug__:
    raise RuntimeError("qualification safety assertions require Python without -O")


class PrivateTrial:
    def __init__(self, collector, fixture):
        self.c, self.fixture = collector, fixture
        self.out = collector.out
        self.data = self.out / "pgdata"
        self.port = "55489"
        self.db = "ea_phase24x_historical_lstm"
        self.started = False
        self.scheduler = None
        self.scheduler_identity = None
        self.server_identity = None
        self.owned_attempts = {}
        self.resources = None
        self.result = {"single_worker": "FAIL", "cleanup": "NOT_STARTED"}
        self.c.env.update({"PGHOST": "127.0.0.1", "PGPORT": self.port,
                           "PGUSER": "phase24x_admin", "PGDATABASE": "postgres",
                           "PGPASSFILE": str(self.out / "pgpass"), "PGCONNECT_TIMEOUT": "5",
                           "LSTM_DB_HOST": "127.0.0.1", "LSTM_DB_NAME": self.db,
                           "FOREX_DB_HOST": "127.0.0.1", "FOREX_DB_NAME": "ea_phase24x_unused_forex"})
        (self.out / "environment.json").write_text(json.dumps(self.c.env, indent=2) + "\n")

    def run(self, name, args, timeout=30):
        result = self.c.run(name, args, timeout)
        assert result is not None, name
        return result

    def sql(self, name, statement, database=None):
        # Explicit connection flags never fall back to a production destination.
        file = self.out / (name + ".sql")
        file.write_text(statement)
        return self.run(name, [PG / "psql", "-X", "-Atq", "-v", "ON_ERROR_STOP=1",
                              "-h", "127.0.0.1", "-p", self.port, "-U", "phase24x_admin",
                              "-d", database or self.db, "-f", file])

    def inspect(self, pid):
        args = [str(OBSERVER), f"--inspect-managed-test-process={pid}"]
        result = self.c.run("inspect-" + str(pid), args, 5)
        if result is None:
            # An absent process is valid only if the OS also reports no process.
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                return None
            raise RuntimeError("live process identity could not be established")
        fields = result.strip().split("|", 4)
        assert len(fields) == 5 and int(fields[0]) == pid
        return fields

    def setup(self):
        admin, runtime = secrets.token_hex(32), secrets.token_hex(32)
        password = self.out / "init-password"
        password.write_text(admin)
        password.chmod(0o600)
        passfile = self.out / "pgpass"
        passfile.write_text(f"127.0.0.1:{self.port}:*:phase24x_admin:{admin}\n"
                            f"127.0.0.1:{self.port}:*:pqxx:{runtime}\n")
        passfile.chmod(0o600)
        self.run("initdb", [PG / "initdb", "-D", self.data, "-U", "phase24x_admin",
                            "--pwfile=" + str(password), "--auth=scram-sha-256", "--encoding=UTF8", "--locale=C"])
        with (self.data / "postgresql.conf").open("a") as file:
            file.write(f"\nlisten_addresses='127.0.0.1'\nport={self.port}\nunix_socket_directories=''\n"
                       "max_connections=32\nshared_buffers='32MB'\n")
        self.run("start-private-postgres", [PG / "pg_ctl", "-D", self.data, "-l", self.out / "postgres.log", "-w", "start"])
        self.started = True
        pid = int((self.data / "postmaster.pid").read_text().splitlines()[0])
        self.server_identity = self.inspect(pid)
        assert self.server_identity and str(self.data) in self.server_identity[4]
        destination = self.sql("destination-before-writes", "SELECT current_database(),current_user,inet_server_addr(),inet_server_port(),current_setting('data_directory'); SELECT system_identifier FROM pg_control_system();", "postgres")
        assert destination.splitlines()[0] == f"postgres|phase24x_admin|127.0.0.1|{self.port}|{self.data}"
        self.result["database_identity"] = destination.strip()
        # Backup owners are private NOLOGIN roles, never copied credentials.
        role_sql = self.out / "private-roles.sql"
        role_sql.write_text("CREATE ROLE pqxx LOGIN SUPERUSER PASSWORD '" + runtime + "';\n" +
                            "\n".join(f"CREATE ROLE {role} NOLOGIN;" for role in
                                      ("vjp", "campaign_operations_owner", "campaign_operations_h1_boundary_authority",
                                       "campaign_operations_h2_runtime", "campaign_operations_runtime", "campaign_operations_h1_owner_read")) +
                            f"\nCREATE DATABASE {self.db};\n")
        role_sql.chmod(0o600)
        self.run("private-role-bootstrap", [PG / "psql", "-X", "-v", "ON_ERROR_STOP=1", "-h", "127.0.0.1", "-p", self.port, "-U", "phase24x_admin", "-d", "postgres", "-f", role_sql])
        role_sql.unlink()  # Contains fresh disposable credentials, not evidence.
        for name in ("pre-data", "data", "matrix.copy", "sequences", "post-data"):
            self.run("restore-" + name, [PG / "psql", "-X", "-q", "-v", "ON_ERROR_STOP=1",
                     "-h", "127.0.0.1", "-p", self.port, "-U", "phase24x_admin", "-d", self.db,
                     "-f", self.out / (name + ".sql")], 60)
        checks = self.sql("restored-state", "SELECT current_database(),inet_server_port(),current_setting('data_directory'); SELECT count(*) FROM experiment; SELECT experiment_id,status,phase,last_model_id,worker_pid,active_scheduler_worker_attempt_id FROM experiment; SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE lifecycle_state IN ('reserved','spawned','running','observed','stopped','identity_ambiguous'); SELECT required_generation,cutover_state FROM experiment_scheduler_protocol; SELECT authority_state FROM experiment_scheduler_lease;")
        lines = checks.strip().splitlines()
        assert lines == [f"{self.db}|{self.port}|{self.data}", "1",
                         f"{self.fixture['experiment_id']}|completed|done|{self.fixture['model_id']}||",
                         "0", "52|complete", "released"]
        self.result["snapshot_restore"] = "PASS (original protocol and released lease preserved)"
        # All routing bytes come from the retained development archive; never synthetic.
        registry = select_fixture_registry(json.loads((ROUTING / "registry.json").read_text()), self.fixture)
        runtime_ids = {w["runtime_identity"] for w in registry["workers"]}
        registry["runtimes"] = [x for x in registry["runtimes"] if x["identity"] in runtime_ids]
        routing = self.out / "routing"
        routing.mkdir()
        for relative in {str(Path(w["executable"]).parent) for w in registry["workers"]} | {"runtime/" + x for x in runtime_ids}:
            shutil.copytree(ROUTING / relative, routing / relative, symlinks=True)
        for worker in registry["workers"]:
            assert sha(routing / worker["executable"]) == worker["sha256"]
        (routing / "registry.json").write_text(json.dumps(registry, indent=2) + "\n")
        binary_dir = self.out / "bin"
        binary_dir.mkdir()
        for name in ("LSTM_Release", "lstm-analyze-worker"):
            shutil.copy2(PRODUCTS / name, binary_dir / name)
            assert sha(binary_dir / name) == sha(PRODUCTS / name)
        self.result["executable_sha256"] = {name: sha(binary_dir / name) for name in ("LSTM_Release", "lstm-analyze-worker")}

    def discover(self):
        eid = self.fixture["experiment_id"]
        raw = self.sql("discover-attempts", f"SELECT worker_attempt_id,worker_pid,worker_process_start_identity,worker_process_group_id,canonical_executable_path,command_line,lifecycle_state FROM experiment_scheduler_worker_attempt WHERE experiment_id={eid} AND canonical_executable_path='{self.out}/bin/lstm-analyze-worker' ORDER BY worker_attempt_id;")
        for line in raw.strip().splitlines():
            fields = line.split("|", 6)
            if len(fields) == 7 and fields[1]:
                self.owned_attempts[fields[0]] = fields
        (self.out / "owned-attempts.json").write_text(json.dumps(self.owned_attempts, indent=2) + "\n")

    def execute(self):
        baseline = vm_counters(self.run("pre-worker-vm", ["/usr/bin/vm_stat"]))
        pressure = self.run("pre-worker-pressure", ["/usr/sbin/sysctl", "kern.memorystatus_vm_pressure_level"])
        assert pressure.strip().endswith(": 1"), "normal pressure required"
        eid = self.fixture["experiment_id"]
        command = self.out / "bin/LSTM_Release"
        self.run("private-scheduler-status", [command, "--scheduler-status"])
        # Sole lifecycle change before dispatch uses the accepted authoritative workflow.
        self.run("requeue-analysis", [command, "--requeue-analysis=" + eid, "--yes"])
        status = self.sql("pending-analysis", "SELECT status,phase,worker_pid,active_scheduler_worker_attempt_id,worker_process_group_id,worker_process_start_identity FROM experiment;")
        assert status.strip() == "pending|analyze||||"
        args = [str(command), "--schedule-experiments", "--max-train-procs=0", "--max-infer-procs=0",
                "--max-analyze-procs=1", "--phase-priority=train:infer:analyze", "--scheduler-poll-seconds=1",
                "--semantic-worker-registry=" + str(self.out / "routing/registry.json"),
                "--analyze-worker=" + str(self.out / "bin/lstm-analyze-worker"),
                "--scheduler-log-dir=" + str(self.out / "worker-logs"), "--scheduler-verbose"]
        started = time.monotonic()
        environment = self.resources.scheduler_environment() if self.resources else self.c.env
        with (self.out / "scheduler.log").open("w") as log:
            self.scheduler = subprocess.Popen(args, cwd=self.out, env=environment, stdout=log, stderr=log, start_new_session=True)
        self.scheduler_identity = self.inspect(self.scheduler.pid)
        assert self.scheduler_identity and self.scheduler_identity[3] == str(command)
        (self.out / "scheduler-process.json").write_text(json.dumps({"argv": args, "identity": self.scheduler_identity, "deadline_seconds": 120, "environment":environment}, indent=2) + "\n")
        while time.monotonic() - started < 120:
            if self.resources:
                self.resources.check()
            assert self.scheduler.poll() is None, "private scheduler exited before completion"
            self.discover()
            state = self.sql("completion-state", "SELECT status,phase,active_scheduler_worker_attempt_id FROM experiment;")
            assert not state.strip().startswith("failed|"), "real ANALYZE worker failed"
            if state.strip() == "completed|done|":
                break
            current = vm_counters(self.run("worker-vm", ["/usr/bin/vm_stat"]))
            assert current.get("Swapouts", 0) == baseline.get("Swapouts", 0), "new host swap-out activity"
            assert self.run("worker-pressure", ["/usr/sbin/sysctl", "kern.memorystatus_vm_pressure_level"]).strip().endswith(": 1")
            time.sleep(0.25)
        else:
            raise TimeoutError("single-worker deadline exceeded")
        self.discover()
        assert len(self.owned_attempts) == 1, "exactly one genuine analysis attempt required"
        attempt = next(iter(self.owned_attempts.values()))
        assert attempt[6] == "completed"
        assert self.sql("observed-worker-exit", f"SELECT exit_code FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id={attempt[0]};").strip() == "0"
        actual = json.loads(self.sql("analysis-after", "SELECT row_to_json(a) FROM experiment_analysis_result a;").strip())
        expected = self.fixture["expected_analysis"]
        verify_analysis(actual, expected)
        self.result.update({"single_worker": "PASS", "seconds_until_reconciled": time.monotonic() - started,
                            "analysis_result": actual, "new_attempt": attempt,
                            "concurrent_workers_launched": False})

    def safe_signal(self, identity, sig):
        pid = int(identity[0])
        current = self.inspect(pid)
        if current is None:
            return
        assert current == identity, "identity changed; refuse signal"
        assert current[3].startswith(str(self.out / "bin/")), "foreign executable; refuse signal"
        os.kill(pid, sig)
        with (self.out / "signals.jsonl").open("a") as file:
            file.write(json.dumps({"identity": current, "signal": int(sig)}) + "\n")

    def cleanup(self):
        # pg_ctl can time out after a postmaster starts. Recover only the exact
        # private data-directory PID so startup failures do not leak a server.
        if not self.started and (self.data / "postmaster.pid").exists():
            pid = int((self.data / "postmaster.pid").read_text().splitlines()[0])
            observed = self.inspect(pid)
            assert observed and str(self.data) in observed[4]
            self.server_identity, self.started = observed, True
        if self.scheduler and self.scheduler.poll() is None:
            # Discover DB-bound children before stopping their private owner.
            self.discover()
            assert self.scheduler_identity, "scheduler identity unavailable; preserve private state"
            self.safe_signal(self.scheduler_identity, signal.SIGTERM)
            self.scheduler.wait(timeout=15)
        for fields in self.owned_attempts.values():
            pid = int(fields[1])
            current = self.inspect(pid)
            if current is None:
                continue
            assert current[2] == fields[2] and current[1] == fields[3]
            assert current[3] == fields[4]
            assert f"--analyze-experiment={self.fixture['experiment_id']}" in current[4]
            assert f"--scheduler-worker-attempt-id={fields[0]}" in current[4]
            self.safe_signal(current, signal.SIGTERM)
            deadline = time.monotonic() + 5
            while self.inspect(pid) is not None and time.monotonic() < deadline:
                time.sleep(0.1)
            assert self.inspect(pid) is None, "owned worker survived; preserve database"
        if self.started:
            if self.resources:
                self.resources.before_database_shutdown()
            pid = int((self.data / "postmaster.pid").read_text().splitlines()[0])
            assert self.server_identity == self.inspect(pid), "private postmaster identity changed"
            self.run("stop-private-postgres", [PG / "pg_ctl", "-D", self.data, "-m", "fast", "-w", "stop"])
            assert self.inspect(pid) is None
            self.started = False
        self.result["cleanup"] = "PASS"


def qualify(collector, fixture, instrument=False):
    (collector.out / "worker-harness-source.py").write_bytes(Path(__file__).read_bytes())
    trial = PrivateTrial(collector, fixture)
    try:
        if instrument:
            from AnalyzeResourceInstrumentation import ResourceRecorder
            trial.resources = ResourceRecorder(trial)
        trial.setup()
        if trial.resources:
            trial.resources.start()
        trial.execute()
    except BaseException as error:
        trial.result["error"] = str(error)
        raise
    finally:
        try:
            trial.cleanup()
        except BaseException as error:
            trial.result["cleanup"] = "FAIL: " + str(error)
            raise
        finally:
            try:
                if trial.resources:
                    trial.resources.finish()
            finally:
                (collector.out / "single-worker-results.json").write_text(json.dumps(trial.result, indent=2) + "\n")
