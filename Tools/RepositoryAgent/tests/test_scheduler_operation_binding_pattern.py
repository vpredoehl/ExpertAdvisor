#!/usr/bin/env python3
"""Index the production SchedulerCore callback pattern without source evidence reads."""
from __future__ import annotations

import os
import sys
import types

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
stub = types.ModuleType("expertadvisor_agent")

def list_files(prefix=""):
    base = os.path.join(ROOT, prefix)
    return [os.path.relpath(os.path.join(directory, name), ROOT)
            for directory, _, names in os.walk(base) for name in names]

def read_file(name, start=1, end=200):
    with open(os.path.join(ROOT, name), encoding="utf-8") as handle:
        return "\n".join(f"{number}: {line.rstrip()}" for number, line in
                         enumerate(handle.readlines()[start - 1:end], start))

stub.list_files = list_files
stub.read_file = read_file
stub.search = lambda pattern, max_results=100: ""
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.repository_index import RepositoryIndex


def main():
    index = RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))
    expected = (
        ("RunScheduler", "RunSchedulerOnce", "operation_binding", "operations.runCycle"),
        ("RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs", "operation_binding", "operations.runCheckpointAnalysis"),
        ("SchedulerEngine::run", "operations.runCycle", "operation_invocation", "operations.runCycle"),
        ("SchedulerCycleService::runOnce", "operations_.runCheckpointAnalysis", "operation_invocation", "operations_.runCheckpointAnalysis"),
    )
    for caller, callee, kind, operation in expected:
        matches = [edge for edge in index.callees_of(caller)
                   if (edge.callee, edge.relationship_kind, edge.operation) == (callee, kind, operation)]
        assert matches, (caller, callee, kind, operation)

    # The actual operation bindings and field invocations remain excluded from
    # direct semantics. A binding alone must never imply runtime execution.
    assert not index.relationship("RunScheduler", "RunSchedulerOnce")
    assert not index.relationship("RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs")
    assert not index.relationship("SchedulerEngine::run", "operations.runCycle")
    assert not index.relationship(
        "SchedulerCycleService::runOnce", "operations_.runCheckpointAnalysis")
    print("test_scheduler_operation_binding_pattern: PASS")


if __name__ == "__main__": main()
