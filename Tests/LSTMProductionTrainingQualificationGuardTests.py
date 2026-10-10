#!/usr/bin/env python3
"""Database/GPU-free regression tests for the benchmark-local safety policy."""
import json
from pathlib import Path
import tempfile
import unittest
from LSTMProductionTrainingQualificationGuard import EnvironmentPolicy, Monitor, cpu_seconds


def snapshot(cpu=0, command="/usr/libexec/triald", interval=5):
    return dict(memory_pressure_level="1", thermal_state=0, swap_counters=[420, 776],
                page_size=16384, idle_gpu_check=False, gpu_utilization_percent=0,
                interval_seconds=interval,
                background_cpu=[dict(pid=1206, command=command, cpu_percent=cpu)])


class GuardTests(unittest.TestCase):
    def test_single_triald_preparation_spike_is_tolerated_then_resets(self):
        policy = EnvironmentPolicy()
        self.assertEqual(policy.evaluate(snapshot(61.62), "preparation")["action"], "TOLERATE_PREPARATION_SPIKE")
        self.assertEqual(policy.evaluate(snapshot(), "preparation")["action"], "ALLOW")
        self.assertEqual(policy.contention_seconds, 0)

    def test_sustained_preparation_contention_aborts_at_fifteen_seconds(self):
        policy = EnvironmentPolicy()
        for _ in range(2):
            self.assertEqual(policy.evaluate(snapshot(61.62), "preparation")["action"], "TOLERATE_PREPARATION_SPIKE")
        self.assertEqual(policy.evaluate(snapshot(61.62), "preparation")["action"], "ABORT")

    def test_training_spike_invalidates_immediately(self):
        self.assertEqual(EnvironmentPolicy().evaluate(snapshot(61.62), "training")["action"], "INVALIDATE")

    def test_service_and_aggregate_thresholds(self):
        for sample in (snapshot(10, "/usr/libexec/fileproviderd"),
                       {**snapshot(), "background_cpu": [dict(pid=i, command="/other", cpu_percent=40) for i in range(5)]}):
            self.assertEqual(EnvironmentPolicy().evaluate(sample, "training")["action"], "INVALIDATE")
        self.assertEqual(EnvironmentPolicy().evaluate(snapshot(49.99), "training")["action"], "ALLOW")

    def test_thermal_pressure_workers_and_swap_are_hard_blockers(self):
        for change in (dict(thermal_state=1), dict(memory_pressure_level="2"),
                       dict(collector_error="GPU/production process conflict; deferred: LSTM_Release")):
            self.assertEqual(EnvironmentPolicy().evaluate({**snapshot(), **change}, "preparation")["action"], "ABORT")
        for change in (dict(swap_counters=[420, 777]), dict(swap_counters=[1444, 776])):
            policy = EnvironmentPolicy(); policy.evaluate(snapshot(), "preparation")
            self.assertEqual(policy.evaluate({**snapshot(), **change}, "preparation")["action"], "ABORT")

    def test_idle_gpu_and_sampling_gap(self):
        self.assertEqual(EnvironmentPolicy().evaluate({**snapshot(), "idle_gpu_check": True,
            "gpu_utilization_percent": 26}, "preflight")["action"], "ABORT")
        self.assertEqual(EnvironmentPolicy().evaluate(snapshot(interval=15.01), "training")["action"], "INVALIDATE")

    def test_phase_transition_and_abort_decision_are_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            def conflict(_):
                raise RuntimeError("competing LSTM worker")
            monitor = Monitor(conflict, directory, "/unused")
            prefix = Path(directory) / "run"
            monitor.begin_run(prefix)
            self.assertEqual(monitor.phase(), "preparation")
            prefix.with_suffix(".log").write_text("DATASET not a marker\nDATASET,rows=369905\n")
            self.assertEqual(monitor.phase(), "training")
            with self.assertRaisesRegex(RuntimeError, "INVALIDATE"):
                monitor.poll(123)
            record = json.loads((Path(directory) / "environment-decisions.jsonl").read_text())
            self.assertEqual(record["decision"]["action"], "INVALIDATE")
            self.assertIn("competing LSTM", record["snapshot"]["collector_error"])
            monitor.end_run()
            self.assertEqual(monitor.phase(), "preflight")

    def test_cpu_time_parser(self):
        self.assertAlmostEqual(cpu_seconds("03:05.50"), 185.5)
        self.assertEqual(cpu_seconds("1-02:03:04"), 93784)


if __name__ == "__main__":
    unittest.main()
