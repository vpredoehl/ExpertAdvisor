#!/usr/bin/env python3
"""Regression coverage for the SchedulerCore operation callback boundary."""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: "forbidden"
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.repository_index import CallSite, FunctionBoundary


class Index:
    def __init__(self, edges): self.edges = edges; self.relationship_calls = 0
    def relationship(self, caller, callee): self.relationship_calls += 1; return []
    def callees_of(self, caller): return [x for x in self.edges if x.caller == caller]
    def function_for_line(self, file, line): return None


class Runtime:
    def __init__(self): self.calls = []
    def verify_claim(self, topic, claim, item):
        self.calls.append(item)
        return {"supports": True, "establishes": "exact operation source establishes the narrow claim", "reason": "test", "model_turns": 1}


class Interface(CodexRepositoryInterface):
    def __init__(self, index, runtime, ledger):
        super().__init__(claim_runtime=runtime, claim_ledger_path=str(ledger)); self.index = index; self.reads = []
    def _idx(self): return self.index
    def _claim_item(self, spec):
        self.reads.append((spec["file"], spec["start"], spec["end"]))
        return {**spec, "excerpt": "exact server-selected SchedulerCore source"}


def main():
    binding = CallSite("Sources/SchedulerCore/ProductionSchedulerDaemon.cpp", 10774,
                       "RunScheduler", "RunSchedulerOnce", "operation_binding", "operations.runCycle")
    invocation = CallSite("Sources/SchedulerCore/SchedulerCycleService.cpp", 48,
                         "SchedulerCycleService::runOnce", "operations_.runCheckpointAnalysis",
                         "operation_invocation", "operations_.runCheckpointAnalysis")
    with tempfile.TemporaryDirectory() as directory:
        runtime = Runtime(); index = Index([binding, invocation])
        iface = Interface(index, runtime, Path(directory) / "claims.json")
        request = {"op": "investigate_operation_relationship_claim", "topic_id": "scheduler",
                   "topic": "scheduler operations", "claim": "the operation is bound",
                   "caller": "RunScheduler", "callee": "RunSchedulerOnce",
                   "relationship_kind": "operation_binding"}
        result = iface.dispatch(request)
        assert result["selection_manifest"]["relationship_kind"] == "operation_binding"
        assert result["selection_manifest"]["selected_ranges"] == [{"file": binding.file, "start": 10762, "end": 10786}]
        assert index.relationship_calls == 0 and len(runtime.calls) == 1 and len(iface.reads) == 1
        request.update({"caller": invocation.caller, "callee": invocation.callee,
                        "relationship_kind": "operation_invocation"})
        result = iface.dispatch(request)
        assert result["selection_manifest"]["relationship_kind"] == "operation_invocation"
        assert index.relationship_calls == 0 and len(runtime.calls) == 2
    print("test_codex_operation_relationship_claim: PASS")


if __name__ == "__main__": main()
