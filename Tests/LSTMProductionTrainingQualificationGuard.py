#!/usr/bin/env python3
"""Benchmark-local environment policy; never changes training or other processes.

CPU intervals are sampled about every five seconds. Preparation tolerates brief
spikes; training (including warmup) uses the original immediate CPU stop gates.
Every collected snapshot and its decision is appended before an abort is raised.
"""
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import re
import subprocess
import time


@dataclass(frozen=True)
class Policy:
    process_cpu_percent: float = 50.0
    aggregate_cpu_percent: float = 200.0
    service_cpu_percent: float = 10.0
    preparation_contention_seconds: float = 15.0
    minimum_cpu_interval_seconds: float = 4.0
    maximum_cpu_interval_seconds: float = 15.0
    idle_gpu_percent: int = 25
    swapin_interval_bytes: int = 16 * 1024 * 1024


class EnvironmentPolicy:
    def __init__(self, policy=Policy()):
        self.policy = policy
        self.contention_seconds = 0.0
        self.previous_swap = None
        self.initial_swapouts = None

    def evaluate(self, snapshot, phase):
        """Return an auditable decision; callers must persist it before stopping."""
        result = {"action": "ALLOW", "phase": phase, "reason": "environment gates clear"}
        def stop(reason):
            result.update(action="INVALIDATE" if phase == "training" else "ABORT", reason=reason)
            return result
        if snapshot.get("collector_error"):
            return stop("hard safety checker: " + snapshot["collector_error"])
        if snapshot["memory_pressure_level"] != "1":
            return stop("memory pressure is not normal")
        if snapshot["thermal_state"] != 0:
            return stop("thermal state is not normal")
        swap = snapshot["swap_counters"]
        if self.initial_swapouts is None:
            self.initial_swapouts = swap[1]
        if swap[1] != self.initial_swapouts:
            return stop("swap-out counter grew")
        if self.previous_swap is not None and (
                swap[0] - self.previous_swap[0]) * snapshot["page_size"] >= self.policy.swapin_interval_bytes:
            return stop("substantial interval swap-in")
        self.previous_swap = swap
        if snapshot["idle_gpu_check"] and (
                snapshot["gpu_utilization_percent"] is None or
                snapshot["gpu_utilization_percent"] > self.policy.idle_gpu_percent):
            return stop("idle GPU contention or missing GPU evidence")
        interval = snapshot.get("interval_seconds")
        if interval is None:
            return result
        if interval > self.policy.maximum_cpu_interval_seconds:
            self.contention_seconds = 0.0
            return stop("environment sampling gap exceeds 15 seconds")
        if interval < self.policy.minimum_cpu_interval_seconds:
            result["reason"] = "CPU interval too short; hard gates checked"
            return result
        background = snapshot["background_cpu"]
        services = [p for p in background if re.search(
            r"/(?:backupd|fileproviderd|mds|mds_stores|mdworker[^/]*|corespotlightd)$|/[^/]*Provider[^/]*$",
            p["command"]) and p["cpu_percent"] >= self.policy.service_cpu_percent]
        high = [p for p in background if p["cpu_percent"] >= self.policy.process_cpu_percent]
        total = sum(p["cpu_percent"] for p in background)
        result.update(background_cpu_total_percent=total, cpu_threshold_exceeded=bool(
            services or high or total >= self.policy.aggregate_cpu_percent))
        if not result["cpu_threshold_exceeded"]:
            self.contention_seconds = 0.0
            return result
        offender = (services or high or [{"aggregate_cpu_percent": total}])[0]
        reason = "background CPU threshold exceeded: " + json.dumps(offender)
        if phase == "training":
            return stop(reason)
        self.contention_seconds += interval
        result["consecutive_contention_seconds"] = self.contention_seconds
        if self.contention_seconds >= self.policy.preparation_contention_seconds:
            return stop("sustained preparation/preflight " + reason)
        result.update(action="TOLERATE_PREPARATION_SPIKE", reason=reason)
        return result


def cpu_seconds(value):
    days, _, clock = value.rpartition("-")
    parts = list(map(float, clock.split(":")))
    return sum(v * 60 ** i for i, v in enumerate(reversed(parts))) + (int(days) * 86400 if days else 0)


class Monitor:
    def __init__(self, collector, directory, thermal_tool):
        self.collector = collector
        self.directory = Path(directory)
        self.thermal_tool = thermal_tool
        self.environment = EnvironmentPolicy()
        self.previous_cpu = None
        self.prefix = None
        self.training_started = False
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "guard-policy.json").write_text(json.dumps(asdict(self.environment.policy), indent=2))

    def begin_run(self, prefix):
        self.prefix = Path(prefix)
        self.training_started = False
        self.environment.contention_seconds = 0.0

    def end_run(self):
        self.prefix = None
        self.training_started = False
        self.environment.contention_seconds = 0.0

    def phase(self):
        if self.prefix is None:
            return "preflight"
        log = self.prefix.with_suffix(".log")
        if not self.training_started and log.exists():
            with log.open("rb") as stream:
                self.training_started = any(line.startswith(b"DATASET,") for line in stream.read(64 * 1024).splitlines())
        return "training" if self.training_started else "preparation"

    def poll(self, allowed_pid=None):
        snapshot = {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
        phase = self.phase()
        try:
            snapshot = self.collector(allowed_pid)
            conflicts = [line for line in snapshot["processes"].splitlines() if re.search(
                r"/Phase25B3/(?:[^ ]*/)*(?:timing|evidence)(?:\s|$)", line)
                and int(line.split()[0]) != allowed_pid]
            if conflicts:
                raise RuntimeError("competing qualification fixture: " + "\n".join(conflicts))
            snapshot["thermal_state"] = int(subprocess.check_output([str(self.thermal_tool)], text=True))
            snapshot["swap_counters"] = [int(re.search(rf"{label}:\s+(\d+)", snapshot["vm_stat"]).group(1))
                                         for label in ("Swapins", "Swapouts")]
            snapshot["page_size"] = int(re.search(r"page size of (\d+) bytes", snapshot["vm_stat"]).group(1))
            gpu = re.search(r'"Device Utilization %"=(\d+)', snapshot["gpu_statistics"] or "")
            snapshot["gpu_utilization_percent"] = int(gpu.group(1)) if gpu else None
            snapshot["idle_gpu_check"] = allowed_pid is None
            raw = subprocess.check_output(["ps", "-axo", "pid,ppid,time,comm"], text=True)
            current = {}
            for line in raw.splitlines()[1:]:
                pid, ppid, used, command = line.split(None, 3)
                current[int(pid)] = (int(ppid), cpu_seconds(used), command)
            now = time.monotonic()
            background = []
            if self.previous_cpu is not None:
                previous_time, previous = self.previous_cpu
                interval = now - previous_time
                snapshot["interval_seconds"] = interval
                for pid, (_, used, command) in current.items():
                    if pid in (allowed_pid, os.getpid()) or pid not in previous:
                        continue
                    background.append({"pid": pid, "command": command,
                        "cpu_percent": max(0.0, 100 * (used - previous[pid][1]) / interval)})
            self.previous_cpu = (now, current)
            snapshot["background_cpu"] = sorted(background, key=lambda p: p["cpu_percent"], reverse=True)
        except Exception as error:
            snapshot["collector_error"] = str(error)
        # Re-read the flushed DATASET marker after sampling. An interval crossing
        # preparation -> training is conservatively subject to training gates.
        phase = self.phase()
        decision = self.environment.evaluate(snapshot, phase)
        record = {"snapshot": snapshot, "decision": decision,
                  "run_prefix": str(self.prefix) if self.prefix else None}
        with (self.directory / "environment-decisions.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
            stream.flush()
        if decision["action"] in ("ABORT", "INVALIDATE"):
            raise RuntimeError(decision["action"] + ": " + decision["reason"])
        return {**snapshot, "guard_decision": decision}
