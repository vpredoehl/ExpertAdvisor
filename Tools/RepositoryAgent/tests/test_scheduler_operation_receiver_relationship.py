#!/usr/bin/env python3
"""Regression for the singular SchedulerCore operation receiver bridge."""
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

from Tools.RepositoryAgent import repository_index as ri
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface


SOURCE = """\
int RunCheckpointEvalAnalyzeJobs(const CycleOptions& cycleOptions)
{
    return 0;
}

int RunSchedulerOnce(const CycleOptions& cycleOptions)
{
    SchedulerCycleOperations operation;
    operation.runCheckpointAnalysis = [&] {
        return RunCheckpointEvalAnalyzeJobs(cycleOptions);
    };
    return 0;
}

int RunScheduler(const CycleOptions& cycleOptions)
{
    SchedulerDaemonOperations operations;
    operations.runCycle = [&] {
        return RunSchedulerOnce(cycleOptions);
    };
    return 0;
}

int SchedulerEngine::run()
{
    return operations.runCycle();
}

int SchedulerCycleService::runOnce()
{
    return operations_.runCheckpointAnalysis();
}
"""


class Runtime:
    def __init__(self):
        self.items = []

    def verify_claim(self, topic, claim, item):
        self.items.append(item)
        return {
            "supports": True,
            "establishes": "the selected operation binding is source-visible",
            "reason": "test",
            "model_turns": 1,
        }


class Interface(CodexRepositoryInterface):
    def __init__(self, index, runtime, ledger):
        super().__init__(claim_runtime=runtime, claim_ledger_path=str(ledger))
        self.index = index
        self.reads = []

    def _idx(self):
        return self.index

    def _claim_item(self, spec):
        self.reads.append((spec["file"], spec["start"], spec["end"]))
        return {**spec, "excerpt": "exact server-selected SchedulerCore source"}


def main():
    old_list_files, old_read_file = ri.list_files, ri.read_file
    try:
        lines = SOURCE.splitlines()
        ri.list_files = lambda prefix="": ["Sources/SchedulerCore/OperationReceiver.cpp"]
        ri.read_file = lambda name, start=1, end=200: "\n".join(
            f"{line_no}: {text}" for line_no, text in
            enumerate(lines[start - 1:end], start)
        )
        index = ri.RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))

        def edges(caller):
            return [(edge.callee, edge.relationship_kind, edge.operation)
                    for edge in index.callees_of(caller)]

        # (a) The singular production-shaped receiver identifies the lexical
        # owner, precise field identity, target, and binding location.
        binding = [edge for edge in index.callees_of("RunSchedulerOnce")
                   if edge.relationship_kind == "operation_binding"]
        assert [(edge.callee, edge.operation) for edge in binding] == [
            ("RunCheckpointEvalAnalyzeJobs", "operation.runCheckpointAnalysis")
        ], binding
        assert binding[0].file == "Sources/SchedulerCore/OperationReceiver.cpp"
        assert binding[0].line == 10

        # (d) The invocation remains a field identity: it does not guess a
        # callback target. The analogous daemon bridge remains indexed too.
        assert ("operations.runCycle", "operation_invocation", "operations.runCycle") in edges("SchedulerEngine::run")
        assert ("operations_.runCheckpointAnalysis", "operation_invocation", "operations_.runCheckpointAnalysis") in edges("SchedulerCycleService::runOnce")
        assert ("RunSchedulerOnce", "operation_binding", "operations.runCycle") in edges("RunScheduler")

        # (e) All two bindings and two invocations remain outside direct-call
        # relationships; a binding alone does not establish execution.
        assert index.relationship("RunScheduler", "RunSchedulerOnce") == []
        assert index.relationship("RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs") == []
        for caller in ("RunScheduler", "RunSchedulerOnce", "SchedulerEngine::run", "SchedulerCycleService::runOnce"):
            assert all(edge.relationship_kind != "direct_invocation"
                       for edge in index.callees_of(caller)), edges(caller)

        with tempfile.TemporaryDirectory() as directory:
            runtime = Runtime()
            interface = Interface(index, runtime, Path(directory) / "claims.json")
            discovery = interface.dispatch({
                "op": "discover_operation_relationship_paths",
                "scope": "SchedulerCore",
                "from": "RunSchedulerOnce",
                "to": "RunCheckpointEvalAnalyzeJobs",
                "max_hops": 1,
                "relationship_kind": "operation_binding",
            })
            # (b) Metadata-only discovery sees the exact same canonical edge.
            assert discovery["paths"] == [["RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs"]], discovery
            assert discovery["evidentiary_status"] == "non_evidentiary"

            # Ordinary direct-call traversal cannot use either callback binding.
            for caller, callee in (("RunScheduler", "RunSchedulerOnce"),
                                   ("RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs")):
                direct = interface.dispatch({
                    "op": "discover_relationship_paths",
                    "scope": "SchedulerCore",
                    "from": caller,
                    "to": callee,
                    "max_hops": 1,
                })
                assert direct["paths"] == [], direct

            # (c) Evidentiary selection consumes that same canonical metadata.
            result = interface.dispatch({
                "op": "investigate_operation_relationship_claim",
                "topic_id": "scheduler",
                "topic": "scheduler operations",
                "claim": "checkpoint analysis is bound through the operation field",
                "caller": "RunSchedulerOnce",
                "callee": "RunCheckpointEvalAnalyzeJobs",
                "relationship_kind": "operation_binding",
            })
            assert result["selection_manifest"]["selection_status"] == "selected", result
            assert result["selection_manifest"]["relationship_kind"] == "operation_binding"
            assert len(runtime.items) == len(interface.reads) == 1
    finally:
        ri.list_files, ri.read_file = old_list_files, old_read_file
    print("test_scheduler_operation_receiver_relationship: PASS")


if __name__ == "__main__":
    main()
