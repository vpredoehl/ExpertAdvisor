#!/usr/bin/env python3
"""Guarded qualification over the canonical completed-experiment input pipeline.

No production worker is launched. Production sessions in the fixture enforce
read-only access and close before training. The sustained-pairs mode uses six
alternating fresh-process pairs with 8 warmup and 504 measured chronological
CalculateBatch updates.
"""
import argparse
import fcntl
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import statistics
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "DerivedData/ExpertAdvisor/Phase25B3"
spec = importlib.util.spec_from_file_location("phase25b2", ROOT / "Tests/LSTMTrainingThroughputBenchmark.py")
prior = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prior)


def safety(allowed_pid=None):
    snapshot = prior.check_safety(allowed_pid)
    conflicts = [line for line in snapshot["processes"].splitlines() if re.search(
        r"/Phase25B3/(?:timing|evidence)(?:\s|$)|(?:/|\s)lstm-(?:train|infer|analyze)-worker(?:\s|$)", line)
        and int(line.split()[0]) != allowed_pid]
    if conflicts:
        raise RuntimeError("competing workload: " + "\n".join(conflicts))
    snapshot["swap"] = subprocess.check_output(["sysctl", "vm.swapusage"], text=True).strip()
    output = subprocess.check_output(["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1", "-l"], text=True)
    match = re.search(r'"PerformanceStatistics" = (\{[^\n]+\})', output)
    snapshot["gpu_statistics"] = match.group(1) if match else None
    snapshot["vm_stat"] = subprocess.check_output(["vm_stat"], text=True)
    return snapshot


def digest(path):
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def run_one(variant, path, warmup, measured, name, output_directory=DIRECTORY,
            diagnostics=False, command_timing=False):
    output_directory.mkdir(parents=True, exist_ok=True)
    prefix = output_directory / name
    snapshots = [safety()]
    started = time.monotonic()
    with prefix.with_suffix(".log").open("w") as log:
        environment = {**os.environ, "EA_LSTM_FORWARD_AFFINE": path}
        if diagnostics:
            environment["EA_LSTM_PROFILE_HOTSPOTS"] = "1"
        if command_timing:
            environment["EA_LSTM_COMMAND_BUFFER_TIMING"] = "1"
        process = subprocess.Popen([str(DIRECTORY / variant), str(warmup), str(measured), str(prefix)],
            cwd=DIRECTORY, env=environment, stdout=log, stderr=subprocess.STDOUT)
        try:
            while True:
                try:
                    code = process.wait(timeout=5)
                    break
                except subprocess.TimeoutExpired:
                    snapshots.append(safety(process.pid))
        except BaseException:
            process.terminate(); process.wait()
            raise
    wall_seconds = time.monotonic() - started
    snapshots.append(safety())
    prefix.with_suffix(".safety.json").write_text(json.dumps(snapshots, indent=2))
    if code:
        raise RuntimeError(f"fixture exited {code}: {prefix}.log")
    records = {}
    updates = []
    for line in prefix.with_suffix(".log").read_text().splitlines():
        if not line.startswith(("UPDATE,", "SUMMARY,", "DATASET,")):
            continue
        kind, *fields = line.split(",")
        record = dict(field.split("=", 1) for field in fields)
        record = {key: value if key == "path" else float(value) for key, value in record.items()}
        if kind == "UPDATE": updates.append(record)
        else: records[kind.lower()] = record
    assert records["summary"]["path"] == path
    records.update(name=name, process_wall_seconds=wall_seconds, updates=updates,
        state_sha256=digest(prefix.with_suffix(".state")), input_sha256=digest(prefix.with_suffix(".inputs")))
    measured_updates = [record for record in updates if record["measured"]]
    records["mean_ms"] = statistics.mean(record["ms"] for record in measured_updates)
    latencies = [record["ms"] for record in measured_updates]
    records["median_ms"] = statistics.median(latencies)
    records["p90_ms"] = statistics.quantiles(latencies, n=10, method="inclusive")[8]
    records["p95_ms"] = statistics.quantiles(latencies, n=20, method="inclusive")[18]
    records["p99_ms"] = statistics.quantiles(latencies, n=100, method="inclusive")[98]
    midpoint = len(measured_updates) // 2
    records["early_mean_ms"] = statistics.mean(latencies[:midpoint])
    records["late_mean_ms"] = statistics.mean(latencies[midpoint:])
    records["early_updates_per_second"] = 1000 / records["early_mean_ms"]
    records["late_updates_per_second"] = 1000 / records["late_mean_ms"]
    records["rss_first_bytes"] = measured_updates[0]["rss_bytes"]
    records["rss_last_bytes"] = measured_updates[-1]["rss_bytes"]
    records["rss_min_bytes"] = min(record["rss_bytes"] for record in measured_updates)
    records["rss_max_bytes"] = max(record["rss_bytes"] for record in measured_updates)
    records["metal_first_bytes"] = measured_updates[0]["metal_bytes"]
    records["metal_last_bytes"] = measured_updates[-1]["metal_bytes"]
    records["metal_min_bytes"] = min(record["metal_bytes"] for record in measured_updates)
    records["metal_max_bytes"] = max(record["metal_bytes"] for record in measured_updates)
    records["safety_snapshot_count"] = len(snapshots)
    records["updates_per_second"] = 1000 / records["mean_ms"]
    records["cpu_percent"] = 100 * sum(record["cpu_seconds"] for record in measured_updates) / (sum(record["ms"] for record in measured_updates) / 1000)
    prefix.with_suffix(".json").write_text(json.dumps(records, indent=2))
    print(json.dumps({key: records[key] for key in ("name", "mean_ms", "updates_per_second", "cpu_percent", "state_sha256", "input_sha256", "process_wall_seconds")}), flush=True)
    return records


def timestamp_pairs():
    output_directory = DIRECTORY / "DiagnosticT"
    pairs = []
    for pair in range(1, 4):
        order = ("metann", "combined") if pair % 2 else ("combined", "metann")
        result = {"pair": pair, "order": order}
        for path in order:
            result[path] = run_one("timing", path, 8, 64,
                                   f"pair{pair}_{path}", output_directory,
                                   diagnostics=True, command_timing=True)
        assert result["metann"]["input_sha256"] == result["combined"]["input_sha256"], "input parity failed"
        assert result["metann"]["state_sha256"] == result["combined"]["state_sha256"], "state parity failed"
        pairs.append(result)
        (output_directory / "pairs.json").write_text(json.dumps(pairs, indent=2))
    (output_directory / "summary.json").write_text(json.dumps({
        "pairs": len(pairs),
        "combined_wins": sum(pair["combined"]["mean_ms"] < pair["metann"]["mean_ms"] for pair in pairs),
        "input_bitwise_equal": True,
        "loss_and_state_bitwise_equal": True,
    }, indent=2))
    print(json.dumps({"pairs": len(pairs), "combined_wins": sum(
        pair["combined"]["mean_ms"] < pair["metann"]["mean_ms"] for pair in pairs)}, indent=2), flush=True)


def diagnostic_pairs():
    output_directory = DIRECTORY / "DiagnosticS"
    pairs = []
    for pair in range(1, 4):
        order = ("metann", "combined") if pair % 2 else ("combined", "metann")
        result = {"pair": pair, "order": order}
        for path in order:
            result[path] = run_one("timing", path, 8, 64,
                                   f"pair{pair}_{path}", output_directory, diagnostics=True)
        assert result["metann"]["input_sha256"] == result["combined"]["input_sha256"], "input parity failed"
        assert result["metann"]["state_sha256"] == result["combined"]["state_sha256"], "state parity failed"
        pairs.append(result)
        (output_directory / "pairs.json").write_text(json.dumps(pairs, indent=2))
    summary = {}
    for path in ("metann", "combined"):
        trials = [pair[path] for pair in pairs]
        values = [trial["mean_ms"] for trial in trials]
        summary[path] = {
            "mean_ms": statistics.mean(values),
            "trial_mean_sd_ms": statistics.stdev(values),
            "median_ms": statistics.median(trial["median_ms"] for trial in trials),
            "p90_ms": statistics.mean(trial["p90_ms"] for trial in trials),
            "p95_ms": statistics.mean(trial["p95_ms"] for trial in trials),
            "p99_ms": statistics.mean(trial["p99_ms"] for trial in trials),
            "early_mean_ms": statistics.mean(trial["early_mean_ms"] for trial in trials),
            "late_mean_ms": statistics.mean(trial["late_mean_ms"] for trial in trials),
            "updates_per_second": 1000 / statistics.mean(values),
            "cpu_percent": statistics.mean(trial["cpu_percent"] for trial in trials),
            "metal_first_bytes": statistics.mean(trial["metal_first_bytes"] for trial in trials),
            "metal_last_bytes": statistics.mean(trial["metal_last_bytes"] for trial in trials),
        }
    summary.update(
        combined_wins=sum(pair["combined"]["mean_ms"] < pair["metann"]["mean_ms"] for pair in pairs),
        input_bitwise_equal=True,
        loss_and_state_bitwise_equal=True,
    )
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def benchmark():
    pairs = []
    for pair in range(1, 6):
        order = ("metann", "combined") if pair % 2 else ("combined", "metann")
        result = {"pair": pair, "order": order}
        for path in order:
            result[path] = run_one("timing", path, 8, 120, f"trial{pair}_{path}")
        assert result["metann"]["input_sha256"] == result["combined"]["input_sha256"], "input parity failed"
        assert result["metann"]["state_sha256"] == result["combined"]["state_sha256"], "state parity failed"
        pairs.append(result)
        (DIRECTORY / "trials.json").write_text(json.dumps(pairs, indent=2))
    assert len({trial[path]["input_sha256"] for trial in pairs for path in ("metann", "combined")}) == 1
    assert len({trial[path]["state_sha256"] for trial in pairs for path in ("metann", "combined")}) == 1
    summary = {}
    for path in ("metann", "combined"):
        values = [pair[path]["mean_ms"] for pair in pairs]
        summary[path] = dict(mean_ms=statistics.mean(values), trial_mean_sd_ms=statistics.stdev(values),
            updates_per_second=1000/statistics.mean(values), cpu_percent=statistics.mean(pair[path]["cpu_percent"] for pair in pairs))
    summary.update(speedup=summary["metann"]["mean_ms"]/summary["combined"]["mean_ms"],
        combined_wins=sum(pair["combined"]["mean_ms"] < pair["metann"]["mean_ms"] for pair in pairs),
        input_bitwise_equal=True, loss_and_state_bitwise_equal=True)
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


def sustained_pairs():
    output_directory = DIRECTORY / "SustainedR"
    pairs = []
    for pair in range(1, 7):
        order = ("metann", "combined") if pair % 2 else ("combined", "metann")
        result = {"pair": pair, "order": order}
        for path in order:
            result[path] = run_one("timing", path, 8, 504,
                                   f"pair{pair}_{path}", output_directory)
        assert result["metann"]["input_sha256"] == result["combined"]["input_sha256"], "input parity failed"
        assert result["metann"]["state_sha256"] == result["combined"]["state_sha256"], "state parity failed"
        pairs.append(result)
        (output_directory / "pairs.json").write_text(json.dumps(pairs, indent=2))
    summary = {}
    for path in ("metann", "combined"):
        trials = [pair[path] for pair in pairs]
        values = [trial["mean_ms"] for trial in trials]
        summary[path] = {
            "mean_ms": statistics.mean(values),
            "trial_mean_sd_ms": statistics.stdev(values),
            "median_ms": statistics.median(trial["median_ms"] for trial in trials),
            "p90_ms": statistics.mean(trial["p90_ms"] for trial in trials),
            "p95_ms": statistics.mean(trial["p95_ms"] for trial in trials),
            "p99_ms": statistics.mean(trial["p99_ms"] for trial in trials),
            "early_mean_ms": statistics.mean(trial["early_mean_ms"] for trial in trials),
            "late_mean_ms": statistics.mean(trial["late_mean_ms"] for trial in trials),
            "early_updates_per_second": statistics.mean(trial["early_updates_per_second"] for trial in trials),
            "late_updates_per_second": statistics.mean(trial["late_updates_per_second"] for trial in trials),
            "updates_per_second": 1000 / statistics.mean(values),
            "cpu_percent": statistics.mean(trial["cpu_percent"] for trial in trials),
            "rss_first_bytes": statistics.mean(trial["rss_first_bytes"] for trial in trials),
            "rss_last_bytes": statistics.mean(trial["rss_last_bytes"] for trial in trials),
            "rss_min_bytes": min(trial["rss_min_bytes"] for trial in trials),
            "rss_max_bytes": max(trial["rss_max_bytes"] for trial in trials),
            "metal_first_bytes": statistics.mean(trial["metal_first_bytes"] for trial in trials),
            "metal_last_bytes": statistics.mean(trial["metal_last_bytes"] for trial in trials),
            "metal_min_bytes": min(trial["metal_min_bytes"] for trial in trials),
            "metal_max_bytes": max(trial["metal_max_bytes"] for trial in trials),
        }
    summary.update(
        speedup=summary["metann"]["mean_ms"] / summary["combined"]["mean_ms"],
        combined_wins=sum(pair["combined"]["mean_ms"] < pair["metann"]["mean_ms"] for pair in pairs),
        input_bitwise_equal=True,
        loss_and_state_bitwise_equal=True,
    )
    (output_directory / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--evidence", action="store_true")
    parser.add_argument("--sustained", action="store_true")
    parser.add_argument("--sustained-pairs", action="store_true")
    parser.add_argument("--diagnostic-pairs", action="store_true")
    parser.add_argument("--timestamp-pairs", action="store_true")
    args = parser.parse_args()
    with (ROOT / "DerivedData/ExpertAdvisor/Phase25B2/runner.lock").open("a") as shared_lock:
        fcntl.flock(shared_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.pilot: run_one("timing", "metann", 0, 1, "pilot_metann")
        elif args.evidence:
            for path in ("metann", "combined"):
                run_one("evidence", path, 8, 120, f"evidence_{path}")
        elif args.sustained:
            for path in ("metann", "combined"):
                run_one("timing", path, 8, 504, f"sustained_{path}")
        elif args.sustained_pairs:
            sustained_pairs()
        elif args.diagnostic_pairs:
            diagnostic_pairs()
        elif args.timestamp_pairs:
            timestamp_pairs()
        else: benchmark()
