#!/usr/bin/env python3
"""Adversarial coverage for bounded non-evidentiary relationship discovery."""
from __future__ import annotations

import sys
import types

READ_CALLS = []
stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: READ_CALLS.append((name, start, end)) or "forbidden"
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.repository_index import CallSite, FunctionBoundary


class FakeIndex:
    def __init__(self, files, functions, calls):
        self.files = files
        self.functions = functions
        self.calls_by_caller = calls

    def callees_of(self, caller):
        return list(self.calls_by_caller.get(caller, ()))


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def _idx(self):
        return self.index


def boundary(file, name, line=1):
    return FunctionBoundary(file, name, line, line + 1)


def edge(file, caller, callee, line=1):
    return CallSite(file, line, caller, callee)


class LegacyCallSite:
    def __init__(self, file, line, caller, callee):
        self.file = file
        self.line = line
        self.caller = caller
        self.callee = callee


def request(**extra):
    return {"op": "discover_relationship_paths", "scope": "SchedulerCore",
            "from": "A", "to": "D", "max_hops": 3, **extra}


def rejected(action, text):
    try:
        action()
    except ValueError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("request must fail closed")


def make(names, calls, extra_files=()):
    files = [f"Sources/SchedulerCore/{name}.cpp" for name in names] + list(extra_files)
    functions = [boundary(f"Sources/SchedulerCore/{name}.cpp", name) for name in names]
    return FakeInterface(FakeIndex(files, functions, calls))


def main():
    checkpoint = ["RunProductionSchedulerDaemon", "RunScheduler", "RunSchedulerOnce", "RunCheckpointEvalAnalyzeJobs"]
    # The production scheduler binds both of these calls as operation
    # callbacks.  They are real calls when the callback runs, but not direct
    # RunScheduler/RunSchedulerOnce relationships and therefore must not form
    # a direct-call discovery path through the lexical owner.
    checkpoint_calls = {
        checkpoint[0]: [edge("Sources/SchedulerCore/RunProductionSchedulerDaemon.cpp", checkpoint[0], checkpoint[1])],
        checkpoint[1]: [CallSite("Sources/SchedulerCore/RunScheduler.cpp", 1, checkpoint[1], checkpoint[2], "callback_invocation")],
        checkpoint[2]: [CallSite("Sources/SchedulerCore/RunSchedulerOnce.cpp", 1, checkpoint[2], checkpoint[3], "callback_invocation")],
    }
    result = make(checkpoint, checkpoint_calls).dispatch(request(
        **{"from": checkpoint[0], "to": checkpoint[3]}))
    assert result["mode"] == "bounded_relationship_path_discovery"
    assert result["evidentiary_status"] == "non_evidentiary"
    assert "investigate_relationship_chain_claim" in result["required_follow_up"]
    assert result["resolved_scope"] == "Sources/SchedulerCore"
    assert result["paths"] == [] and result["path_count"] == 0
    assert result["caps"] == {"max_hops": 4, "visited_nodes": 64, "examined_edges": 512, "returned_paths": 4}
    direct = make(checkpoint, checkpoint_calls).dispatch(request(
        **{"from": checkpoint[0], "to": checkpoint[1], "max_hops": 1}))
    assert direct["paths"] == [checkpoint[:2]]

    # The same concrete SchedulerCore operation bindings are separately
    # discoverable, metadata-only, and never change direct-call results.
    operation_request = {
        "op": "discover_operation_relationship_paths", "scope": "SchedulerCore",
        "from": checkpoint[1], "to": checkpoint[3], "max_hops": 2,
        "relationship_kind": "operation_binding",
    }
    operation_calls = {
        checkpoint[1]: [CallSite("Sources/SchedulerCore/RunScheduler.cpp", 1, checkpoint[1], checkpoint[2], "operation_binding", "operations.runCycle")],
        checkpoint[2]: [CallSite("Sources/SchedulerCore/RunSchedulerOnce.cpp", 1, checkpoint[2], checkpoint[3], "operation_binding", "operations.runCheckpointAnalysis")],
    }
    operation = make(checkpoint, operation_calls).dispatch(operation_request)
    assert operation["mode"] == "bounded_operation_relationship_path_discovery"
    assert operation["evidentiary_status"] == "non_evidentiary"
    assert operation["paths"] == [checkpoint[1:]]
    assert make(checkpoint, operation_calls).dispatch(request(
        **{"from": checkpoint[1], "to": checkpoint[3], "max_hops": 2})) ["paths"] == []

    # Breadth-first traversal returns shortest paths only, in canonical order.
    names = ["A", "B", "C", "D", "Long"]
    calls = {
        "A": [edge("Sources/SchedulerCore/A.cpp", "A", "Long"), edge("Sources/SchedulerCore/A.cpp", "A", "C"), edge("Sources/SchedulerCore/A.cpp", "A", "B")],
        "B": [edge("Sources/SchedulerCore/B.cpp", "B", "D")],
        "C": [edge("Sources/SchedulerCore/C.cpp", "C", "D")],
        "Long": [edge("Sources/SchedulerCore/Long.cpp", "Long", "C")],
    }
    equal = make(names, calls).dispatch(request())
    assert equal["paths"] == [["A", "B", "D"], ["A", "C", "D"]]
    assert make(["A", "D"], {}).dispatch(request())["paths"] == []

    # RepositoryIndex always records relationship_kind.  A legacy or malformed
    # materialized edge without it cannot be promoted to a direct invocation.
    missing_kind = make(["A", "D"], {
        "A": [LegacyCallSite("Sources/SchedulerCore/A.cpp", 1, "A", "D")],
    }).dispatch(request())
    assert missing_kind["paths"] == []

    # Exact max-hop boundary, invalid values, and strict unknown fields.
    assert make(checkpoint, checkpoint_calls).dispatch(request(
        **{"from": checkpoint[0], "to": checkpoint[3], "max_hops": 3}))["path_count"] == 0
    assert make(checkpoint, checkpoint_calls).dispatch(request(
        **{"from": checkpoint[0], "to": checkpoint[3], "max_hops": 2}))["paths"] == []
    for value in (0, 5, True, "3"):
        rejected(lambda value=value: make(["A", "D"], {}).dispatch(request(max_hops=value)), "max_hops")
    rejected(lambda: make(["A", "D"], {}).dispatch(request(extra="no")), "unexpected request fields")
    rejected(lambda: make(["A", "D"], {}).dispatch(request(scope="../SchedulerCore")), "scope")

    # Scope is direct-only. Descendant and external endpoints/edges cannot enter.
    nested = "Sources/SchedulerCore/Nested/C.cpp"
    iface = make(["A", "D"], {"A": [edge("Sources/SchedulerCore/A.cpp", "A", "C")], "C": [edge(nested, "C", "D")]}, [nested])
    iface.index.functions.append(boundary(nested, "C"))
    assert iface.dispatch(request())["paths"] == []
    outside = make(["A", "D"], {}, ["Sources/Other/X.cpp"])
    outside.index.functions.append(boundary("Sources/Other/X.cpp", "X"))
    rejected(lambda: outside.dispatch(request(to="X")), "outside resolved scope")

    # Neither endpoint nor unqualified call sites may choose among definitions.
    ambiguous_file = "Sources/SchedulerCore/X.cpp"
    ambiguous = make(["A", "D"], {"A": [edge("Sources/SchedulerCore/A.cpp", "A", "Shared")]}, [ambiguous_file, "Sources/SchedulerCore/Y.cpp"])
    ambiguous.index.functions += [boundary(ambiguous_file, "X::Shared"), boundary("Sources/SchedulerCore/Y.cpp", "Y::Shared")]
    assert ambiguous.dispatch(request())["paths"] == []
    rejected(lambda: ambiguous.dispatch(request(to="Shared")), "endpoint is ambiguous")

    # Each traversal admission cap rejects the whole result before return.
    node_names = ["A", "D"] + [f"N{i}" for i in range(64)]
    node_calls = {"A": [edge("Sources/SchedulerCore/A.cpp", "A", f"N{i}") for i in range(64)]}
    rejected(lambda: make(node_names, node_calls).dispatch(request(max_hops=1)), "visited-node cap")
    edge_names = ["A", "D", "E"]
    edge_calls = {"A": [edge("Sources/SchedulerCore/A.cpp", "A", "E", line=i + 1) for i in range(513)]}
    rejected(lambda: make(edge_names, edge_calls).dispatch(request(max_hops=1)), "examined-edge cap")
    result_names = ["A", "D"] + [f"B{i}" for i in range(5)]
    result_calls = {"A": [edge("Sources/SchedulerCore/A.cpp", "A", f"B{i}") for i in range(5)]}
    result_calls.update({f"B{i}": [edge(f"Sources/SchedulerCore/B{i}.cpp", f"B{i}", "D")] for i in range(5)})
    rejected(lambda: make(result_names, result_calls).dispatch(request(max_hops=2)), "result cap")

    # A supplied materialized index performs no source reads or model setup.
    assert not READ_CALLS
    assert equal["evidentiary_status"] == "non_evidentiary"
    assert make(["A", "D"], {})._claim_runtime is None
    print("test_codex_relationship_path_discovery: PASS")


if __name__ == "__main__":
    main()
