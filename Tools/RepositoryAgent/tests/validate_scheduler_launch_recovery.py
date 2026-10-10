#!/usr/bin/env python3
"""Regression guard for never-spawned scheduler reservation recovery."""

from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[3]
SOURCE = ROOT / "Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"

text = SOURCE.read_text()

start = text.index("'launch_reservation_within_recovery_grace'")
end = text.index(
    'if (terminal.empty())',
    start,
)
recovery = text[start:end]

assert "lifecycle_state='identity_ambiguous'" in text[
    start - 350:start
], "Expected early-grace ambiguity transition"

assert "recoveryReady" in text[
    start - 500:start
], "Expected recovery grace guard"

assert "reconciliation_result='never_spawned'" in recovery, (
    "Never-spawned recovery transition missing"
)

assert "a.lifecycle_state='reserved'" in recovery, (
    "Never-spawned recovery must still accept reserved attempts"
)

assert "a.lifecycle_state='identity_ambiguous'" in recovery, (
    "DEFECT: Early-grace ambiguous attempts cannot terminalize"
)

assert "launch_reservation_within_recovery_grace" in recovery, (
    "Ambiguous attempts must be restricted to the grace diagnostic"
)

for guard in (
    "a.ownership_origin='scheduler_launch'",
    "a.worker_pid IS NULL",
    "a.worker_process_group_id IS NULL",
    "a.worker_process_start_identity IS NULL",
    "a.reserved_at <= clock_timestamp()",
):
    assert guard in recovery, f"Missing recovery safety guard: {guard}"

print("PASS: Scheduler launch recovery accepts early-grace ambiguity")
