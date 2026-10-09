#!/usr/bin/env python3
"""Focused regression for Native16 inspection diagnostic records."""
import importlib.util
import json
from pathlib import Path
import tempfile
from unittest import mock
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location(
    "capacity", ROOT / "Tests/SchedulerCapacityPriorityQualification.py")
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)

with tempfile.TemporaryDirectory(prefix="ea_native16_diag.") as directory:
    qualification = module.CapacityQualification.__new__(module.CapacityQualification)
    qualification.out = Path(directory)
    discovery = {
        "experiment": 980302, "attempt": 1, "pid": 52265,
        "start": "1791496552:300479", "group": 52265,
        "lifecycle": "spawned", "persisted_command": "synthetic",
    }
    qualification.record_inspection_failure(
        discovery,
        {"exit_status": 1, "stdout": "", "stderr": "",
         "elapsed_seconds": 0.001, "diagnostic_stdout": "stage=process_status_read"},
        "inspection_subprocess_failure", None)
    record = json.loads(
        (Path(directory) / "scheduler-analyze-inspection.jsonl").read_text())
    assert record["experiment_id"] == 980302
    assert record["worker_attempt_id"] == 1
    assert record["classification"] == "inspection_subprocess_failure"
    assert record["diagnostic_stdout"] == "stage=process_status_read"
    assert record["scheduler_lifecycle_state"] == "spawned"

    # A scheduler-owned priority worker is not in the base fixture PID list.
    # It must be inspectable immediately after registry setup, before the
    # later scheduler-managed capacity scenario has run.
    qualification.helper = Path(directory) / "GlobalExperimentControlProcessTests"
    qualification.artifacts = {}
    (qualification.out / "bin").mkdir()
    owned = qualification.out / "owned-pids"
    owned.write_text("101\n")
    with mock.patch.object(module.m.Qualification, "registry"):
        qualification.registry()
    wrapper = qualification.out / "bin" / "ps"
    private_command = (f"python3 {qualification.analyze_helper} "
                       "--analyze-experiment=980302 --scheduler-worker-attempt-id=1")
    census = f"101 registered-fixture\n202 {private_command}\n303 unrelated-worker\n"
    observation = f"202 202 S {private_command}\n"

    def fake_ps(argv, **kwargs):
        assert argv[0] == "/bin/ps"
        if argv[1:] == ["-axo", "pid=,command="]:
            return subprocess.CompletedProcess(argv, 0, census, "")
        assert argv[1:3] == ["-p", "202"]
        return subprocess.CompletedProcess(argv, 0, observation, "")

    for pid, expected in (("202", 0), ("303", 1)):
        with mock.patch.dict(module.os.environ, {"EA_TEST_OWNED_PIDS": str(owned)}), \
                mock.patch.object(sys, "argv", [str(wrapper), "-p", pid,
                                                "-o", "pid=", "-o", "command="]), \
                mock.patch.object(subprocess, "run", side_effect=fake_ps):
            try:
                exec(compile(wrapper.read_text(), str(wrapper), "exec"), {})
            except SystemExit as result:
                assert result.code == expected, (pid, result.code)
            else:
                raise AssertionError("private observation filter did not exit")

print("SchedulerCapacityPriorityInspectionDiagnosticsTests passed")
