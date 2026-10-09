#!/usr/bin/env python3
"""Phase 24X read-only evidence collector; deliberately cannot launch workloads.

No database connection, scheduler invocation, worker launch, or signal is used.
Offline backup data is inventory evidence, never proof of a qualified fixture.
Exit 2 means real workload qualification has not been established.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
AREA = ROOT / "DerivedData/ExpertAdvisor/Phase24X"
PG_RESTORE = Path("/opt/homebrew/opt/postgresql@17/bin/pg_restore")
TABLES = ("experiment", "inference_eval_result", "schema_migrations",
          "experiment_scheduler_protocol")


def copy_rows(text, table):
    """Read one public COPY section without interpreting SQL or executing it."""
    columns = None
    rows = []
    for line in text.splitlines():
        prefix = f"COPY public.{table} ("
        if line.startswith(prefix) and line.endswith(") FROM stdin;"):
            columns = line[len(prefix):].split(") FROM stdin;", 1)[0].split(", ")
        elif columns is not None and line == "\\.":
            columns = None
        elif columns is not None:
            fields = line.split("\t")
            if len(fields) != len(columns):
                raise ValueError("malformed COPY row")
            rows.append(dict(zip(columns, fields)))
    return rows


def private_log_inventory(experiments):
    result = Counter()
    sizes = []
    for row in experiments:
        for column in ("train_log_path", "infer_log_path"):
            raw = row.get(column, "\\N")
            if raw == "\\N":
                result["null"] += 1
                continue
            path = (ROOT / raw).resolve()
            # Never inspect a production path, even if supplied by a dump.
            if not path.is_relative_to(ROOT.resolve()):
                result["outside_checkout_rejected"] += 1
            elif path.is_file():
                result["present_in_checkout"] += 1
                sizes.append(path.stat().st_size)
            else:
                result["missing_in_checkout"] += 1
    return {"counts": dict(result), "max_bytes": max(sizes, default=None),
            "total_bytes": sum(sizes)}


def vm_counters(text):
    return {key: int(value) for key, value in re.findall(
        r"^(Swapins|Swapouts|Pageouts):\s+(\d+)\.", text, re.MULTILINE)}


class Collector:
    def __init__(self, output):
        self.out = output.resolve()
        area = AREA.resolve()
        if (not area.is_relative_to(ROOT.resolve()) or
                not self.out.is_relative_to(area) or self.out == area):
            raise ValueError("use a fresh child of Rollover DerivedData/ExpertAdvisor/Phase24X")
        self.out.mkdir(parents=True, exist_ok=False, mode=0o700)
        self.sequence = 0
        source = Path(__file__).read_bytes()
        (self.out / "collector-source.py").write_bytes(source)
        (self.out / "collector-identity.json").write_text(json.dumps({
            "source": str(Path(__file__).resolve()),
            "sha256": hashlib.sha256(source).hexdigest()}, indent=2) + "\n")
        # Do not inherit PG credentials, services, CLI overrides, or PYTHONPATH.
        self.env = {"PATH": "/usr/bin:/bin:/usr/sbin:/sbin", "LC_ALL": "C",
                    "LANG": "C"}
        self.failures = []

    def run(self, name, argv, timeout=10):
        self.sequence += 1
        started = time.monotonic()
        stamp = datetime.now(timezone.utc).isoformat()
        try:
            r = subprocess.run(list(map(str, argv)), cwd=ROOT, env=self.env,
                               capture_output=True, text=True, timeout=timeout)
            code, stdout, stderr = r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired as error:
            code = None
            stdout = error.stdout or b""
            stderr = error.stderr or b""
            stdout = stdout.decode(errors="replace") if isinstance(stdout, bytes) else stdout
            stderr = stderr.decode(errors="replace") if isinstance(stderr, bytes) else stderr
        log = f"{self.sequence:03d}-{name}.log"
        (self.out / log).write_text(stdout + stderr)
        record = {"utc": stamp, "name": name, "argv": list(map(str, argv)),
                  "cwd": str(ROOT), "exit": code, "timeout_seconds": timeout,
                  "elapsed_seconds": time.monotonic() - started, "log": log}
        with (self.out / "commands.jsonl").open("a") as file:
            file.write(json.dumps(record) + "\n")
        if code != 0:
            self.failures.append(record)
            return None
        return stdout

    def collect(self):
        branch = self.run("branch", ["/usr/bin/git", "branch", "--show-current"])
        head = self.run("head", ["/usr/bin/git", "rev-parse", "HEAD"])
        self.run("status", ["/usr/bin/git", "status", "--short"])
        dump = ROOT / "Database/backups/LSTM_latest.dump"
        data = {}
        self.run("backup-toc", [PG_RESTORE, "--list", dump])
        for table in TABLES:
            raw = self.run("backup-" + table, [PG_RESTORE, "--data-only",
                           "--schema=public", "--table=" + table, "--file=-", dump], 30)
            data[table] = copy_rows(raw, table) if raw is not None else []
        experiments = data["experiment"]
        audit = {
            "backup_bytes": dump.stat().st_size,
            "experiment_count": len(experiments),
            "experiment_states": dict(Counter(row["status"] + "/" + row["phase"]
                                              for row in experiments)),
            "inference_count": len(data["inference_eval_result"]),
            "inference_scopes": dict(Counter(row["inference_scope"]
                                             for row in data["inference_eval_result"])),
            "latest_migration": max((row["version"] for row in data["schema_migrations"]),
                                    default=None),
            "protocol": data["experiment_scheduler_protocol"],
            "logs": private_log_inventory(experiments),
            "scientific_representativeness": "UNVERIFIED",
            "private_database_validated": False,
        }
        (self.out / "input-inventory.json").write_text(json.dumps(audit, indent=2) + "\n")
        self.run("hardware", ["/usr/sbin/sysctl", "hw.memsize", "hw.ncpu",
                              "hw.physicalcpu", "hw.logicalcpu"])
        self.run("display-hardware", ["/usr/sbin/system_profiler", "SPDisplaysDataType"], 20)
        samples = []
        for index in range(3):
            sample = {"utc": datetime.now(timezone.utc).isoformat()}
            self.run(f"processes-{index}", ["/bin/ps", "-axo",
                     "pid=,ppid=,pgid=,lstart=,stat=,%cpu=,rss=,command="])
            raw = self.run(f"vm-{index}", ["/usr/bin/vm_stat"])
            sample["vm"] = vm_counters(raw or "")
            sample["swap"] = self.run(f"swap-{index}", ["/usr/sbin/sysctl", "vm.swapusage"])
            sample["pressure_level"] = self.run(f"pressure-level-{index}",
                ["/usr/sbin/sysctl", "kern.memorystatus_vm_pressure_level"])
            self.run(f"pressure-{index}", ["/usr/bin/memory_pressure", "-Q"])
            samples.append(sample)
            if index < 2:
                time.sleep(5)
        self.run("disk-baseline", ["/usr/sbin/iostat", "-d", "-c", "2", "-w", "1"])
        (self.out / "baseline-samples.json").write_text(json.dumps(samples, indent=2) + "\n")
        deltas = {key: samples[-1]["vm"][key] - samples[0]["vm"][key]
                  for key in samples[0]["vm"] if key in samples[-1]["vm"]}
        results = {
            "qualification": "BLOCKED", "branch": (branch or "").strip(),
            "head": (head or "").strip(), "workloads_launched": 0,
            "database_connections": 0, "signals_sent": 0,
            "cleanup": "NOT_NEEDED (no workers or servers launched)",
            "baseline_vm_counter_deltas": deltas,
            "collection_failures": self.failures,
            "gpu_utilization": "NOT_MEASURED (hardware inventory only)",
            "per_concurrency": [{"workers": count, "result": "NOT_RUN",
                                 "reason": "representative private inputs and isolation unproven"}
                                for count in (1, 2, 4, 8, 12, 18)],
            "missing_requirements": [
                "Validated private database with representative ANALYZE inputs and private logs",
                "Authoritative workflow to obtain fresh pending ANALYZE jobs with no forged attempts",
                "Proven scheduler bootstrap/process isolation and identity-safe bounded cleanup",
                "Acceptable host pressure/swap baseline and measured TRAIN/INFER resource reserves",
                "Validated real-workload measurement and completion assertions"],
        }
        (self.out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = Collector(args.output).collect()
    print(json.dumps({"qualification": result["qualification"], "output": str(args.output)}))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
