#!/usr/bin/env python3
"""Sequential, safety-checked Phase 25B-2 runner; never launches LSTM_Release.

Build with the sibling .sh first. --pilot runs one untallied update; the normal
run uses ten fresh-process pairs with alternating order, 8 warmup + 16 measured
updates each. --validate exercises existing tensor observers; after the timed
trials, --validate-trajectory checks every step of the same 24-update trajectory.
"""
import argparse
import fcntl
import hashlib
import itertools
import json
import math
import os
from pathlib import Path
import random
import re
import statistics
import struct
import subprocess
import time
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DIRECTORY = ROOT / "DerivedData/ExpertAdvisor/Phase25B2"


def check_safety(allowed_pid=None):
    processes = subprocess.check_output(["ps", "-axo", "pid,ppid,%cpu,rss,command"], text=True)
    conflicts = [line for line in processes.splitlines() if re.search(
        r"(?:/|\s)LSTM_(?:Release|Debug|Scheduler|Train|Infer|Analyze)[^\s]*|/Phase25B2/(?:timing|evidence)(?:\s|$)|ollama runner|llama-server|(?:python[^\s]* .*|mlx[^\s]* )(?:train|infer)", line)
        and int(line.split()[0]) != allowed_pid]
    if conflicts:
        raise RuntimeError("GPU/production process conflict; deferred: " + "\n".join(conflicts))
    with urllib.request.urlopen("http://127.0.0.1:11434/api/ps", timeout=3) as response:
        models = json.load(response)["models"]
    if models:
        raise RuntimeError("Ollama has loaded models; deferred without stopping it")
    scheduler = subprocess.run(["launchctl", "print", f"gui/{os.getuid()}/com.vjp.lstm.scheduler"],
                               text=True, capture_output=True, check=False)
    if re.search(r"^\s*pid = \d+", scheduler.stdout, re.M):
        raise RuntimeError("Scheduler running; deferred")
    pressure = subprocess.check_output(["sysctl", "-n", "kern.memorystatus_vm_pressure_level"], text=True).strip()
    if pressure != "1":
        raise RuntimeError("Memory pressure is not normal: " + pressure)
    return {"utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "memory_pressure_level": pressure,
            "ollama_models": models, "scheduler": scheduler.stdout, "processes": processes}


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run_one(variant, mode, warmup, measured, name, windows=128):
    checks = [check_safety()]
    prefix = DIRECTORY / name
    env = {**os.environ, "EA_LSTM_FORWARD_AFFINE": name.split("_")[-1]}
    with (prefix.with_suffix(".log")).open("w") as log:
        process = subprocess.Popen([str(DIRECTORY / variant), mode, str(warmup), str(measured), str(prefix), str(windows)],
                                   cwd=DIRECTORY, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                result = process.wait(timeout=5)
                break
            except subprocess.TimeoutExpired:
                # If external work starts, only our fixture is terminated.
                try:
                    checks.append(check_safety(allowed_pid=process.pid))
                except Exception:
                    process.terminate()
                    process.wait()
                    raise
    checks.append(check_safety())
    (prefix.with_suffix(".safety.json")).write_text(json.dumps(checks, indent=2))
    if result != 0:
        raise RuntimeError(f"fixture exit {result}: {prefix}.log")
    lines = prefix.with_suffix(".log").read_text().splitlines()
    assert not any(line.startswith("DIAG_") for line in lines)
    summary = next(line for line in lines if line.startswith("SUMMARY,"))
    values = dict(field.split("=", 1) for field in summary.split(",")[1:])
    for key in values.keys() - {"path", "mode"}:
        values[key] = float(values[key])
    values["state_sha256"] = digest(prefix.with_suffix(".state"))
    values["update_ms"] = [float(dict(field.split("=", 1) for field in line.split(",")[1:])["ms"])
                           for line in lines if line.startswith("UPDATE,")]
    values["name"] = name
    print(json.dumps(values), flush=True)
    return values


def inspect_tensors(path, windows=128, updates=3):
    counts = {}; clipped = 0; preclip = None; batch_shapes = set()
    with path.open("rb") as stream:
        def u64():
            value = stream.read(8)
            if len(value) != 8: raise ValueError("truncated tensor evidence")
            return struct.unpack("=Q", value)[0]
        while stream.peek(1):
            stage = stream.read(u64()).decode()
            matrices = []
            for _ in range(u64()):
                rows, cols = u64(), u64()
                values = struct.unpack(f"={rows * cols}f", stream.read(rows * cols * 4))
                assert all(math.isfinite(value) for value in values)
                matrices.append(values)
                if stage == "forward_cache": batch_shapes.add((rows, cols))
            counts[stage] = counts.get(stage, 0) + 1
            if stage == "preclip": preclip = matrices
            if stage == "postclip":
                assert preclip is not None and len(preclip) == len(matrices) == 6
                for before, after in zip(preclip, matrices):
                    assert len(before) == len(after)
                    for x, y in zip(before, after):
                        assert y == max(-10.0, min(10.0, x)), "clipping mathematics mismatch"
                        clipped += abs(x) > 10.0
                preclip = None
    assert counts == {"forward_cache": 64 * updates, "preclip": updates, "postclip": updates}
    assert batch_shapes == {(windows, 171), (windows, 64)}
    return {"stage_counts": counts, "clipped_elements": clipped, "sha256": digest(path)}


def validate():
    evidence = {}
    for mode in ("legacy", "auxiliary", "log", "percent"):
        pair = [run_one("evidence", mode, 0, 3, f"evidence_{mode}_{path}")
                for path in ("metann", "combined")]
        assert pair[0]["state_sha256"] == pair[1]["state_sha256"], "state/loss parity failed"
        tensor_pair = [inspect_tensors(DIRECTORY / f"evidence_{mode}_{path}.tensors")
                       for path in ("metann", "combined")]
        assert tensor_pair[0] == tensor_pair[1], "timestep/gradient parity failed"
        # Observer presence must not change the unobserved parameter trajectory.
        plain = run_one("timing", mode, 0, 3, f"parity_{mode}_metann")
        assert plain["state_sha256"] == pair[0]["state_sha256"], "observer changed training"
        evidence[mode] = {"state_sha256": pair[0]["state_sha256"], **tensor_pair[0], "bitwise_equal": True,
                          "observer_on_off_equal": True}
    # A small supplementary regression batch avoids averaging away the large
    # gradients. Dimensions and clipping threshold remain identical to training.
    for mode in ("log", "percent"):
        pair = [run_one("evidence", mode, 0, 3, f"clipping_{mode}_{path}", windows=3)
                for path in ("metann", "combined")]
        assert pair[0]["state_sha256"] == pair[1]["state_sha256"]
        tensor_pair = [inspect_tensors(DIRECTORY / f"clipping_{mode}_{path}.tensors", windows=3)
                       for path in ("metann", "combined")]
        assert tensor_pair[0] == tensor_pair[1] and tensor_pair[0]["clipped_elements"] > 0
        evidence[f"clipping_{mode}"] = {"windows": 3, **tensor_pair[0], "bitwise_equal": True}
    (DIRECTORY / "equivalence.json").write_text(json.dumps(evidence, indent=2))


def validate_trajectory():
    pairs = json.loads((DIRECTORY / "trials.json").read_text())
    assert len(pairs) == 10
    references = {pair[path]["state_sha256"] for pair in pairs for path in ("metann", "combined")}
    assert len(references) == 1, "independent trial trajectories differ"
    results = [run_one("evidence", "legacy", 8, 16, f"trajectory_{path}")
               for path in ("metann", "combined")]
    assert {result["state_sha256"] for result in results} == references
    tensor_pair = [inspect_tensors(DIRECTORY / f"trajectory_{path}.tensors", updates=24)
                   for path in ("metann", "combined")]
    assert tensor_pair[0] == tensor_pair[1]
    (DIRECTORY / "trajectory-equivalence.json").write_text(json.dumps(
        {"updates": 24, "state_sha256": results[0]["state_sha256"], **tensor_pair[0],
         "bitwise_equal": True, "matches_all_timing_trials": True}, indent=2))


def benchmark():
    pairs = []
    for trial in range(1, 11):
        order = ("metann", "combined") if trial % 2 else ("combined", "metann")
        pair = {path: run_one("timing", "legacy", 8, 16, f"trial{trial:02}_{path}") for path in order}
        assert pair["metann"]["state_sha256"] == pair["combined"]["state_sha256"], "training trajectory differs"
        pairs.append({"trial": trial, "order": order, **pair})
        (DIRECTORY / "trials.json").write_text(json.dumps(pairs, indent=2))
    summary = {}
    for path in ("metann", "combined"):
        values = [pair[path]["mean_ms"] for pair in pairs]
        updates = [value for pair in pairs for value in pair[path]["update_ms"]]
        summary[path] = {"mean_ms": statistics.mean(values), "median_ms": statistics.median(values),
                         "sample_sd_ms": statistics.stdev(values), "cv_percent": statistics.stdev(values)/statistics.mean(values)*100,
                         "min_ms": min(values), "max_ms": max(values),
                         "updates_per_second_from_mean": 1000/statistics.mean(values),
                         "peak_rss_min_bytes": min(pair[path]["peak_rss_bytes"] for pair in pairs),
                         "peak_rss_max_bytes": max(pair[path]["peak_rss_bytes"] for pair in pairs),
                         "pooled_update_median_ms": statistics.median(updates),
                         "pooled_update_sample_sd_ms": statistics.stdev(updates)}
    differences = [pair["metann"]["mean_ms"] - pair["combined"]["mean_ms"] for pair in pairs]
    mean_difference = statistics.mean(differences)
    margin = 2.2621571628540993 * statistics.stdev(differences) / math.sqrt(10)
    # Exact two-sided paired sign-flip randomization test: trials, not updates,
    # are the units. No distribution fit or discarded outliers.
    null_means = [sum(sign*x for sign, x in zip(signs, differences))/10
                  for signs in itertools.product((-1, 1), repeat=10)]
    p_value = sum(abs(x) >= abs(mean_difference) - 1e-12 for x in null_means)/len(null_means)
    rng = random.Random(42)
    bootstrap = []
    for _ in range(20000):
        sample = [rng.choice(pairs) for _ in pairs]
        bootstrap.append(statistics.mean(pair["metann"]["mean_ms"] for pair in sample) /
                         statistics.mean(pair["combined"]["mean_ms"] for pair in sample))
    bootstrap.sort()
    summary.update(mean_speedup=summary["metann"]["mean_ms"]/summary["combined"]["mean_ms"],
                   latency_reduction_percent=(1-summary["combined"]["mean_ms"]/summary["metann"]["mean_ms"])*100,
                   median_speedup=summary["metann"]["median_ms"]/summary["combined"]["median_ms"],
                   paired_mean_saving_ms=mean_difference, paired_saving_95pct_t_interval_ms=[mean_difference-margin, mean_difference+margin],
                   paired_randomization_two_sided_p=p_value, speedup_95pct_paired_bootstrap=[bootstrap[499], bootstrap[19499]],
                   combined_wins=sum(x > 0 for x in differences), all_trajectories_bitwise_equal=True)
    (DIRECTORY / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--validate", action="store_true")
    parser.add_argument("--validate-trajectory", action="store_true")
    args = parser.parse_args()
    with (DIRECTORY / "runner.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if args.pilot: run_one("timing", "legacy", 0, 1, "pilot_metann")
        elif args.validate: validate()
        elif args.validate_trajectory: validate_trajectory()
        else: benchmark()
