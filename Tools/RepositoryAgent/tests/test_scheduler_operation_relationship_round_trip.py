#!/usr/bin/env python3
"""Production-shaped operation relationship discovery-to-claim regressions."""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
stub = types.ModuleType("expertadvisor_agent")


def list_files(prefix=""):
    base = os.path.join(ROOT, prefix)
    return [os.path.relpath(os.path.join(directory, name), ROOT)
            for directory, _, names in os.walk(base) for name in names]


def read_file(name, start=1, end=200):
    from Tools.RepositoryAgent.source_reader import RepositorySourceReader
    return RepositorySourceReader(Path(ROOT)).read_file(name, start, end)


stub.list_files = list_files
stub.read_file = read_file
stub.search = lambda pattern, max_results=100: ""
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.repository_index import RepositoryIndex


class Runtime:
    def __init__(self):
        self.items = []; self.positional_items = []

    def verify_claim(self, topic, claim, item):
        self.items.append(item)
        return {"supports": True, "establishes": "selected source supports claim",
                "reason": "test", "model_turns": 1}

    def verify_positional_operation_binding_claim(self, topic, claim, relationship):
        self.positional_items.append(relationship)
        return {"supports": True, "establishes": "selected positional source supports claim",
                "reason": "test", "model_turns": 1}


class Interface(CodexRepositoryInterface):
    def __init__(self, index, runtime, ledger):
        super().__init__(claim_runtime=runtime, claim_ledger_path=str(ledger))
        self.index = index

    def _idx(self):
        return self.index

    def _claim_item(self, spec):
        return {**spec, "excerpt": read_file(spec["file"], spec["start"], spec["end"])}


def operation_request(caller, callee, kind):
    return {
        "op": "investigate_operation_relationship_claim",
        "topic_id": "scheduler", "topic": "scheduler operations",
        "claim": "the exact production operation relationship is present",
        "caller": caller, "callee": callee, "relationship_kind": kind,
    }


def discover_request(caller, callee, kind):
    return {
        "op": "discover_operation_relationship_paths", "scope": "SchedulerCore",
        "from": caller, "to": callee, "max_hops": 1,
        "relationship_kind": kind,
    }


def round_trip(interface, caller, callee, kind):
    discovery = interface.dispatch(discover_request(caller, callee, kind))
    assert discovery["paths"] == [[caller, callee]], discovery
    result = interface.dispatch(operation_request(caller, callee, kind))
    assert result["selection_manifest"]["selection_status"] == "selected", result
    assert result["selection_manifest"]["relationship_kind"] == kind


def direct_round_trip(interface, caller, callee):
    discovery = interface.dispatch({
        "op": "discover_relationship_paths", "scope": "SchedulerCore",
        "from": caller, "to": callee, "max_hops": 1,
    })
    assert discovery["paths"] == [[caller, callee]], discovery
    result = interface.dispatch({
        "op": "investigate_relationship_claim", "topic_id": "scheduler",
        "topic": "scheduler operations",
        "claim": "the exact production direct relationship is present",
        "caller": caller, "callee": callee,
    })
    assert result["selection_manifest"]["selection_status"] == "selected", result


def main():
    index = RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))
    with tempfile.TemporaryDirectory() as directory:
        runtime = Runtime()
        interface = Interface(index, runtime, Path(directory) / "claims.json")

        # Discovery identities are passed unchanged to operation claim
        # selection for both directions of the runCycle runtime bridge.
        round_trip(interface, "RunScheduler", "RunSchedulerOnce", "operation_binding")
        round_trip(interface, "SchedulerEngine::run", "operations.runCycle",
                   "operation_invocation")

        # The production temporary and explicitly typed local receiver forms
        # close the direct bridges around the runCycle callback handoff.
        direct_round_trip(interface, "RunScheduler", "SchedulerEngine::run")
        direct_round_trip(interface, "RunSchedulerOnce",
                          "SchedulerCycleService::runOnce")

        # Existing checkpoint-cycle bridge remains symmetric and distinct.
        round_trip(interface, "RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs",
                   "operation_binding")
        round_trip(interface, "SchedulerCycleService::runOnce",
                   "operations_.runCheckpointAnalysis", "operation_invocation")

        # Concrete CheckpointAnalysisOperations uses exact positional aggregate
        # slots, mapped only through its declared production schema.
        for callee in ("ClaimCheckpointAnalysis", "ExecuteCheckpointAnalysisWork",
                       "FinalizeCheckpointAnalysis"):
            round_trip(interface, "RunCheckpointEvalAnalyzeJobs", callee,
                       "operation_binding")
        aggregate_edges = {(edge.callee, edge.operation) for edge in
                           index.callees_of("RunCheckpointEvalAnalyzeJobs")
                           if edge.relationship_kind == "operation_binding"}
        assert {
            ("ClaimCheckpointAnalysis", "operations.claim"),
            ("ExecuteCheckpointAnalysisWork", "operations.execute"),
            ("FinalizeCheckpointAnalysis", "operations.finalize"),
        } <= aggregate_edges, aggregate_edges

        # Unique short-call canonicalization agrees with direct discovery.
        discovery = interface.dispatch({
            "op": "discover_relationship_paths", "scope": "SchedulerCore",
            "from": "RunCheckpointEvalAnalyzeJobs",
            "to": "CheckpointAnalysisOrchestrationService::runOne", "max_hops": 1,
        })
        assert discovery["paths"] == [["RunCheckpointEvalAnalyzeJobs",
                                       "CheckpointAnalysisOrchestrationService::runOne"]], discovery
        direct = interface.dispatch({
            "op": "investigate_relationship_claim", "topic_id": "scheduler",
            "topic": "scheduler operations", "claim": "service runOne is directly invoked",
            "caller": "RunCheckpointEvalAnalyzeJobs",
            "callee": "CheckpointAnalysisOrchestrationService::runOne",
        })
        assert direct["selection_manifest"]["selection_status"] == "selected", direct

        # Operation bindings and field invocations remain outside ordinary
        # direct-call metadata and therefore never establish execution alone.
        for caller, callee in (
            ("RunScheduler", "RunSchedulerOnce"),
            ("SchedulerEngine::run", "operations.runCycle"),
            ("RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs"),
            ("SchedulerCycleService::runOnce", "operations_.runCheckpointAnalysis"),
            ("RunCheckpointEvalAnalyzeJobs", "ClaimCheckpointAnalysis"),
            ("RunCheckpointEvalAnalyzeJobs", "ExecuteCheckpointAnalysisWork"),
            ("RunCheckpointEvalAnalyzeJobs", "FinalizeCheckpointAnalysis"),
        ):
            assert index.relationship(caller, callee) == [], (caller, callee)
        assert len(runtime.items) == 7
        assert len(runtime.positional_items) == 3
    print("test_scheduler_operation_relationship_round_trip: PASS")


if __name__ == "__main__":
    main()
