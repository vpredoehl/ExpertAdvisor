"""Private Darwin sampling with owning-parent high-water/CPU accounting.

No scientific workflow is implemented here. Coverage failure is fatal even
when the native worker succeeds. Raw receipts survive all failures.
"""
import ctypes
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import statistics
import threading
import time

from AnalyzeHistoricalFixtureQualification import PG, sha

INTERVAL = 0.005
MAX_GAP = 0.050


def cpu_percent(previous, current, seconds):
    delta = (current["user_ns"] + current["system_ns"] -
             previous["user_ns"] - previous["system_ns"])
    if seconds <= 0 or delta < 0:
        raise ValueError("invalid CPU interval/counter")
    return delta / 1e9 / seconds * 100  # 100% means one occupied CPU core.


def verify_destination(actual, database, port, directory):
    if actual != {"database":database,"host":"127.0.0.1","port":int(port),"data_directory":str(directory)}:
        raise RuntimeError("resource observer destination does not match verified private cluster")


def child_pids(buffer, count):
    # proc_listchildpids returns a PID count, unlike proc_listpids' byte count.
    if count < 0 or count >= len(buffer):
        raise RuntimeError("private PostgreSQL process census incomplete")
    return list(buffer)[:count]


def summarize(samples, lifecycle, attempt):
    """Pure offline qualification gate. Never infers a peak from sampled RSS."""
    pid, identity = int(attempt[1]), attempt[2]
    errors = []
    forks = [r for r in lifecycle if r["event"] == "fork" and r["pid"] == pid]
    exits = [r for r in lifecycle if r["event"] == "reaped" and r["pid"] == pid]
    worker = [(s, s.get("worker")) for s in samples if s.get("worker")]
    alive = [(s, p) for s, p in worker if p["state"] != 5 and p["usage_available"]]
    for _, p in worker:
        if p["pid"] != pid or p["start_identity"] != identity:
            errors.append("worker identity mismatch")
    if len(forks) != 1 or len(exits) != 1:
        errors.append("missing owning-parent lifecycle accounting")
    if len(alive) < 2:
        errors.append("fewer than two live worker samples (short-lived process missed)")
    if not any(p["executable"] == attempt[4] for _,p in alive):
        errors.append("real ANALYZE executable was never sampled")
    times = [s["monotonic"] for s in samples]
    gaps = [b-a for a,b in zip(times, times[1:])]
    if not gaps or max(gaps) > MAX_GAP:
        errors.append("sampling gap exceeds 50 ms")
    if not samples or any(s.get("error") for s in samples):
        errors.append("native sampler error")
    hosts = [s["host"] for s in samples if s.get("host")]
    if len(hosts) != len(samples) or any(h["pressure"] != 1 for h in hosts):
        errors.append("memory pressure missing or unsafe")
    swap = {key: hosts[-1][key] - hosts[0][key] if hosts else None
            for key in ("swapins_pages", "swapouts_pages", "pageouts_pages", "swap_used_bytes")}
    if swap["swapouts_pages"] is not None and swap["swapouts_pages"] != 0:
        errors.append("new host swap-out activity")
    if any(swap[k] is not None and swap[k] < 0 for k in ("swapins_pages", "swapouts_pages", "pageouts_pages")):
        errors.append("system counters reset")
    result = {"qualification": "FAIL", "errors": errors, "requested_interval_seconds": INTERVAL,
              "sample_count": len(samples), "max_observed_gap_seconds": max(gaps, default=None),
              "missed_interval_count": max(0,int((times[-1]-times[0])/INTERVAL)+1-len(times)) if times else 0,
              "gaps_with_skipped_full_interval": sum(max(0, int(g / INTERVAL) - 1) for g in gaps),
              "median_observed_interval_seconds": statistics.median(gaps) if gaps else None,
              "sampling_lateness_seconds_max": max((s.get("lateness_seconds",0) for s in samples),default=0),
              "live_worker_sample_count": len(alive), "missing_short_lived_worker": len(alive) < 2,
              "swap_delta": swap, "memory_pressure_levels": sorted({h["pressure"] for h in hosts}),
              "window_start_utc": samples[0]["utc"] if samples else None,
              "window_end_utc": samples[-1]["utc"] if samples else None,
              "cpu_percent_convention": "100 percent = one CPU core",
              "gpu_measurement": "not collected; final ANALYZE does not execute Metal model inference",
              "measurement_overhead": "external sampler plus one persistent PostgreSQL observer connection; child wait4 accounting"}
    if forks and exits and alive:
        fork, exit = forks[0], exits[0]
        if not fork.get("identity") or fork["identity"]["start_identity"] != identity:
            errors.append("fork provenance does not match native worker attempt")
        # The owning parent can reap a zombie whose libproc identity is no
        # longer available. PID reuse is impossible before its child is reaped;
        # the fork receipt plus native attempt fence remains authoritative.
        if exit.get("identity_before_wait") and exit["identity_before_wait"]["start_identity"] != identity:
            errors.append("exit provenance does not match native worker attempt")
        if exit["status"] != 0 or exit["peak_rss_bytes"] <= 0:
            errors.append("missing successful exit or kernel peak RSS")
        start = alive[0][1]["start_epoch"]
        if samples[0]["epoch"] > fork["fork_before_epoch"] or samples[-1]["epoch"] < exit["utc_epoch"]:
            errors.append("measurement window does not include startup and reap")
        last_alive_before = alive[-1][1].get("observed_epoch_before",alive[-1][0]["epoch"])
        dead = [s for s in samples if s.get("worker_observed_epoch_after",s["epoch"]) >= last_alive_before and
                (s.get("worker_gone") or (s.get("worker") and s["worker"]["state"] == 5))]
        if not dead:
            errors.append("worker shutdown not sampled")
            end = exit["utc_epoch"]
        else:
            end = dead[0].get("worker_observed_epoch_after",dead[0]["epoch"])
        first_alive_after = alive[0][1].get("observed_epoch_after",alive[0][0]["epoch"])
        if first_alive_after - start > MAX_GAP or end - last_alive_before > MAX_GAP:
            errors.append("startup/shutdown coverage exceeds 50 ms")
        durations = [last_alive_before - start, end - start]
        cpu = exit["user_seconds"] + exit["system_seconds"]
        if cpu <= 0 or durations[0] < 0 or durations[1] <= 0:
            errors.append("invalid worker duration or CPU accounting")
        # Process samples include the fork-to-exec startup; kernel high-water
        # includes all child work through exit, independently of sample timing.
        rss = [p["rss_bytes"] for _, p in alive]
        if exit["peak_rss_bytes"] < max(rss):
            errors.append("kernel high-water is below sampled RSS")
        result["worker"] = {"pid": pid, "start_identity": identity,
            "exit_identity_source": "owning-parent fork/wait4 pair, matched to persisted native attempt",
            "peak_rss_bytes": exit["peak_rss_bytes"], "sampled_max_rss_bytes": max(rss),
            "sampled_min_rss_bytes": min(rss), "sampled_mean_rss_bytes": sum(rss)/len(rss),
            "peak_physical_footprint_sampled_bytes": max(p["physical_footprint_bytes"] for _,p in alive),
            "user_cpu_seconds": exit["user_seconds"], "system_cpu_seconds": exit["system_seconds"],
            "total_cpu_seconds": cpu, "execution_duration_seconds_bounds": durations,
            "execution_duration_uncertainty_seconds": durations[1]-durations[0],
            "average_cpu_percent_bounds": [cpu/durations[1]*100, cpu/durations[0]*100] if durations[0]>0 else None,
            "sampled_peak_cpu_percent": max((p.get("cpu_percent",0) for _,p in alive), default=0),
            "kernel_input_blocks": exit["input_blocks"], "kernel_output_blocks": exit["output_blocks"],
            "sampled_disk_read_bytes": max(p["disk_read_bytes"] for _,p in alive),
            "sampled_disk_write_bytes": max(p["disk_write_bytes"] for _,p in alive)}
        connections = [s["postgres_connections"] for s in samples if s.get("postgres_connections") and
                       start <= s["epoch"] <= end]
        if len(connections) < 2 or max((x["worker"] for x in connections), default=0) < 1:
            errors.append("insufficient PostgreSQL connection coverage during worker")
        result["postgres_connections"] = {"worker_window_samples": len(connections),
            "max_total_clients": max((x["total"] for x in connections), default=0),
            "max_worker_clients": max((x["worker"] for x in connections), default=0),
            "max_scheduler_clients": max((x["scheduler"] for x in connections), default=0),
            "max_observer_clients": max((x["observer"] for x in connections), default=0),
            "max_total_excluding_observer": max((x["total"]-x["observer"] for x in connections), default=0)}
    pg = [s.get("postgres", []) for s in samples]
    result["private_postgres"] = {"sampled_peak_sum_rss_bytes": max((sum(p["rss_bytes"] for p in group) for group in pg),default=0),
        "sampled_peak_sum_physical_footprint_bytes": max((sum(p["physical_footprint_bytes"] for p in group) for group in pg),default=0),
        "rss_caveat": "process sum may double-count shared PostgreSQL memory",
        "per_process": {}}
    for group in pg:
        for p in group:
            key = f"{p['pid']}@{p['start_identity']}"
            existing = result["private_postgres"]["per_process"].setdefault(key,{
                "pid": p["pid"], "start_identity": p["start_identity"], "samples": 0})
            existing["samples"] += 1
            for name in ("rss_bytes","physical_footprint_bytes","user_ns","system_ns","disk_read_bytes","disk_write_bytes","cpu_percent"):
                existing["sampled_max_"+name] = max(existing.get("sampled_max_"+name,0),p.get(name,0))
    seen = {p["pid"] for group in pg for p in group}
    backend_pids = {pid for s in samples for pid in s.get("postgres_connections",{}).get("backend_pids",[])}
    result["private_postgres"]["unsampled_observed_backend_pids"] = sorted(backend_pids-seen)
    result["private_postgres"]["missed_process_read_count"] = sum(len(s.get("postgres_missing_sample_pids",[])) for s in samples)
    result["private_postgres"]["missed_process_read_pids"] = sorted({pid for s in samples for pid in s.get("postgres_missing_sample_pids",[])})
    result["private_postgres"]["sampled_total_cpu_seconds_lower_bound"] = sum(
        (p["sampled_max_user_ns"]+p["sampled_max_system_ns"])/1e9 for p in result["private_postgres"]["per_process"].values())
    if not result["private_postgres"]["per_process"]:
        errors.append("private PostgreSQL processes not measured")
    result["qualification"] = "PASS" if not errors else "FAIL"
    return result


class ResourceRecorder:
    def __init__(self, trial):
        self.t = trial
        self.out = trial.out
        self.samples = []
        self.stop_event = threading.Event()
        self.ready = threading.Event()
        self.shutdown_database = threading.Event()
        self.database_closed = threading.Event()
        self.error = None
        source = Path(__file__).with_name("AnalyzeResourceProbe.cpp")
        (self.out / source.name).write_bytes(source.read_bytes())
        (self.out / Path(__file__).name).write_bytes(Path(__file__).read_bytes())
        self.library_path = self.out / "resource-probe.dylib"
        trial.run("build-resource-probe", ["/usr/bin/clang++", "-std=c++20", "-O2", "-Wall", "-Wextra", "-Werror",
                  "-dynamiclib", source, "-o", self.library_path], 60)
        self.native = ctypes.CDLL(str(self.library_path))
        self.native.ea_process.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_size_t]
        self.native.ea_host.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
        self.native.ea_children.argtypes = [ctypes.c_int, ctypes.c_void_p, ctypes.c_int]
        self.libpq_path = Path(trial.run("resource-pg-libdir", [PG / "pg_config", "--libdir"], 5).strip()) / "libpq.5.dylib"
        self.libpq = ctypes.CDLL(str(self.libpq_path))
        trial.c.env.update({"PGAPPNAME": "phase24x_harness"})
        self.thread = threading.Thread(target=self.collect, name="phase24x-resource-sampler", daemon=True)
        (self.out / "measurement-config.json").write_text(json.dumps({"interval_seconds": INTERVAL,
            "max_gap_seconds": MAX_GAP, "library_sha256": sha(self.library_path),
            "methods": ["proc_pid_rusage v2", "host_statistics64", "sysctl pressure/swapusage", "owning-parent wait4", "private pg_stat_activity"],
            "postgres_query_timeout_ms": 100, "libpq":str(self.libpq_path),
            "libpq_sha256":sha(self.libpq_path)},indent=2)+"\n")

    def process(self, pid):
        buffer = ctypes.create_string_buffer(8192)
        before_epoch = time.time()
        before_monotonic = time.monotonic()
        rc = self.native.ea_process(pid, buffer, len(buffer))
        if rc < 0:
            raise RuntimeError("process identity changed during measurement")
        if rc != 1:
            return None
        result = json.loads(buffer.value)
        result.update({"observed_epoch_before":before_epoch,"observed_epoch_after":time.time(),
                       "observed_monotonic":before_monotonic})
        return result

    def host(self):
        buffer = ctypes.create_string_buffer(4096)
        if self.native.ea_host(buffer, len(buffer)) != 1:
            raise RuntimeError("host resource counters unavailable")
        return json.loads(buffer.value)

    def database(self):
        lib = self.libpq
        lib.PQconnectdb.argtypes = [ctypes.c_char_p]; lib.PQconnectdb.restype = ctypes.c_void_p
        lib.PQstatus.argtypes = [ctypes.c_void_p]; lib.PQstatus.restype = ctypes.c_int
        lib.PQerrorMessage.argtypes = [ctypes.c_void_p]; lib.PQerrorMessage.restype = ctypes.c_char_p
        lib.PQexec.argtypes = [ctypes.c_void_p,ctypes.c_char_p]; lib.PQexec.restype = ctypes.c_void_p
        lib.PQresultStatus.argtypes = [ctypes.c_void_p]; lib.PQresultStatus.restype = ctypes.c_int
        lib.PQgetvalue.argtypes = [ctypes.c_void_p,ctypes.c_int,ctypes.c_int]; lib.PQgetvalue.restype = ctypes.c_char_p
        lib.PQclear.argtypes = [ctypes.c_void_p]; lib.PQfinish.argtypes = [ctypes.c_void_p]
        # All addresses and credentials refer to the already verified private cluster.
        if any(c in str(self.out) for c in ("'", "\\", "\n", "\r")):
            raise RuntimeError("unsupported connection path characters")
        # Clear inherited libpq defaults in this disposable harness process.
        # Child commands already use Collector's independently sanitized env.
        for name in list(os.environ):
            if name.startswith("PG"):
                os.environ.pop(name)
        conninfo = (f"host=127.0.0.1 hostaddr=127.0.0.1 sslmode=disable port={self.t.port} dbname={self.t.db} user=phase24x_admin "
                    f"passfile='{self.out}/pgpass' connect_timeout=5 application_name=phase24x_monitor "
                    "options='-c statement_timeout=100'").encode()
        conn = lib.PQconnectdb(conninfo)
        if not conn or lib.PQstatus(conn) != 0:
            message = lib.PQerrorMessage(conn).decode(errors="replace") if conn else "allocation failed"
            if conn: lib.PQfinish(conn)
            raise RuntimeError("private resource observer database connection failed: " + message)
        check = lib.PQexec(conn,b"SELECT json_build_object('database',current_database(),'host',host(inet_server_addr()),'port',inet_server_port(),'data_directory',current_setting('data_directory'));")
        try:
            if not check or lib.PQresultStatus(check) != 2:
                raise RuntimeError("resource destination query failed: " + lib.PQerrorMessage(conn).decode(errors="replace"))
            actual = json.loads(lib.PQgetvalue(check,0,0))
            (self.out / "resource-observer-destination.json").write_text(json.dumps(actual,indent=2)+"\n")
            verify_destination(actual,self.t.db,self.t.port,self.t.data)
        except BaseException:
            lib.PQfinish(conn)
            raise
        finally:
            if check: lib.PQclear(check)
        return lib, conn

    def query(self, lib, conn):
        sql = b"SELECT pg_stat_clear_snapshot(); SELECT json_build_object('total',count(*),'worker',count(*) FILTER (WHERE application_name LIKE 'phase24x_analyze_%'),'scheduler',count(*) FILTER (WHERE application_name='phase24x_scheduler'),'observer',count(*) FILTER (WHERE application_name='phase24x_monitor'),'backend_pids',coalesce(json_agg(pid),'[]'::json)) FROM pg_stat_activity WHERE datname=current_database() AND backend_type='client backend';"
        value = lib.PQexec(conn, sql)
        try:
            if not value or lib.PQresultStatus(value) != 2:
                raise RuntimeError("private resource connection query failed")
            return json.loads(lib.PQgetvalue(value,0,0))
        finally:
            if value: lib.PQclear(value)

    def collect(self):
        conn = lib = None
        previous = {}
        try:
            lib, conn = self.database()
            root_pid = int(self.t.server_identity[0])
            expected_root = self.t.server_identity[2]
            baseline = None
            scheduled = time.monotonic()
            with (self.out / "resource-samples.jsonl").open("w") as raw:
                while not self.stop_event.is_set():
                    tick = time.monotonic()
                    sample = {"monotonic": tick, "scheduled_monotonic": scheduled,
                              "lateness_seconds":max(0,tick-scheduled),"epoch": time.time(), "utc": datetime.now(timezone.utc).isoformat(),
                              "host": self.host(), "postgres": []}
                    if baseline is None: baseline = sample["host"]
                    root = self.process(root_pid)
                    if root:
                        if root["start_identity"] != expected_root:
                            raise RuntimeError("private postmaster identity changed")
                        children = (ctypes.c_int * 64)()
                        count = self.native.ea_children(root_pid, children, ctypes.sizeof(children))
                        ids = child_pids(children,count)
                        sample["postgres_child_pids"] = ids
                        sample["postgres_missing_sample_pids"] = []
                        for pid in [root_pid] + ids:
                            p = self.process(pid)
                            if p and p["usage_available"] and (pid == root_pid or p["ppid"] == root_pid):
                                sample["postgres"].append(p)
                            else:
                                sample["postgres_missing_sample_pids"].append(pid)
                    elif not self.shutdown_database.is_set():
                        raise RuntimeError("private postmaster disappeared unexpectedly")
                    if self.shutdown_database.is_set() and conn:
                        lib.PQfinish(conn); conn = None
                        self.database_closed.set()
                    if conn:
                        sample["postgres_connections"] = self.query(lib,conn)
                    lifecycle_path = self.out / "resource-lifecycle.jsonl"
                    records = []
                    if lifecycle_path.exists():
                        # A concurrent append can expose a partial last line.
                        data = lifecycle_path.read_text()
                        records = [json.loads(line) for line in data.splitlines(keepends=True) if line.endswith("\n")]
                    forks = [r for r in records if r["event"] == "fork"]
                    if len(forks) > 1:
                        raise RuntimeError("more than one private child spawned")
                    if forks:
                        birth = forks[0]
                        p = self.process(birth["pid"])
                        sample["worker_observed_epoch_after"] = time.time()
                        if p:
                            if not birth["identity"] or p["start_identity"] != birth["identity"]["start_identity"]:
                                raise RuntimeError("worker PID reused")
                            allowed = {str(self.out / "bin/LSTM_Release"),str(self.out / "bin/lstm-analyze-worker")}
                            if p["state"] != 5 and p["executable"] not in allowed:
                                raise RuntimeError("foreign worker executable")
                            sample["worker"] = p
                        else:
                            sample["worker_gone"] = True
                    for p in sample["postgres"] + ([sample["worker"]] if sample.get("worker") else []):
                        key = (p["pid"],p["start_identity"])
                        if p["usage_available"] and p["state"] != 5:
                            if key in previous:
                                prev_tick, prev = previous[key]
                                p["cpu_percent"] = cpu_percent(prev,p,p["observed_monotonic"]-prev_tick)
                            previous[key] = (p["observed_monotonic"],p.copy())
                    sample["collection_seconds"] = time.monotonic()-tick
                    self.samples.append(sample)
                    raw.write(json.dumps(sample)+"\n"); raw.flush()
                    self.ready.set()
                    if sample["host"]["pressure"] != 1 or sample["host"]["swapouts_pages"] > baseline["swapouts_pages"]:
                        raise RuntimeError("unsafe host pressure or new swap-out activity")
                    scheduled += INTERVAL
                    # Preserve requested cadence without a burst of catch-up queries.
                    if scheduled < time.monotonic():
                        scheduled += (int((time.monotonic()-scheduled)/INTERVAL)+1)*INTERVAL
                    self.stop_event.wait(max(0,scheduled-time.monotonic()))
        except BaseException as error:
            self.error = str(error)
            with (self.out / "resource-sampler-error.json").open("w") as file:
                json.dump({"error": self.error, "utc": datetime.now(timezone.utc).isoformat()},file)
            self.ready.set()
        finally:
            if conn: lib.PQfinish(conn)
            self.database_closed.set()

    def start(self):
        self.thread.start()
        if not self.ready.wait(6): raise RuntimeError("resource sampler startup timed out")
        self.check()
        if not self.samples: raise RuntimeError("resource sampler produced no baseline")

    def check(self):
        if self.error: raise RuntimeError(self.error)

    def scheduler_environment(self):
        env = dict(self.t.c.env)
        env.update({"PGAPPNAME": "phase24x_scheduler", "DYLD_INSERT_LIBRARIES": str(self.library_path),
                    "EA_PHASE24X_SCHEDULER": str(self.out / "bin/LSTM_Release"),
                    "EA_PHASE24X_WORKER": str(self.out / "bin/lstm-analyze-worker"),
                    "EA_PHASE24X_LIFECYCLE": str(self.out / "resource-lifecycle.jsonl")})
        return env

    def before_database_shutdown(self):
        self.shutdown_database.set()
        if self.thread.ident is None:
            self.database_closed.set()
        if not self.database_closed.wait(2):
            raise RuntimeError("private resource observer connection failed to close before server shutdown")

    def finish(self):
        # Include post-shutdown samples; never hold a production process handle.
        time.sleep(INTERVAL*2)
        self.stop_event.set()
        if self.thread.ident is not None:
            self.thread.join(timeout=6)
        if self.thread.is_alive(): raise RuntimeError("resource sampler failed to stop")
        records = []
        path = self.out / "resource-lifecycle.jsonl"
        if path.exists(): records = [json.loads(line) for line in path.read_text().splitlines()]
        attempt = self.t.result.get("new_attempt", ["", "0", "", "", "", "", ""])
        summary = summarize(self.samples,records,attempt)
        if self.error:
            summary["errors"].append(self.error); summary["qualification"] = "FAIL"
        (self.out / "resource-summary.json").write_text(json.dumps(summary,indent=2)+"\n")
        self.t.result["resource_qualification"] = summary["qualification"]
        self.t.result["resource_summary"] = str(self.out / "resource-summary.json")
        if summary["qualification"] != "PASS":
            raise RuntimeError("inadequate resource qualification: " + "; ".join(summary["errors"]))
