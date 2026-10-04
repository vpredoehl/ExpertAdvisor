#!/usr/bin/env python3
"""Adversarial deterministic coverage for non-evidentiary catalog discovery."""
from __future__ import annotations

import json
import sys
import types

READ_CALLS = []

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
def forbidden_read(name, start=1, end=200):
    READ_CALLS.append((name, start, end))
    return "SOURCE_SENTINEL_MUST_NOT_LEAK"
stub.read_file = forbidden_read
sys.modules.setdefault("expertadvisor_agent", stub)

from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.repository_index import FunctionBoundary


class FakeIndex:
    def __init__(self, files, functions):
        self.files = list(files)
        self.functions = list(functions)


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index):
        super().__init__()
        self.index = index

    def _idx(self):
        return self.index


def fn(file, name, line=1):
    return FunctionBoundary(file, name, line, line + 5)


def request(**extra):
    return {
        "op": "discover_catalog_targets",
        "scope": "SchedulerCore",
        "query_groups": [["checkpoint", "analysis"], ["checkpoint", "evaluation"], ["production", "scheduler", "daemon"]],
        **extra,
    }


def rejected(action, text):
    try:
        action()
    except ValueError as exc:
        assert text in str(exc), str(exc)
        return
    raise AssertionError("request must fail closed")


def main():
    files = [
        "Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.cpp",
        "Sources/SchedulerCore/CheckpointEvaluationService.cpp",
        "Sources/SchedulerCore/ProductionSchedulerDaemon.cpp",
        "Sources/SchedulerCore/SchedulerDaemonCli.cpp",
        "Sources/SchedulerCore/Unrelated.cpp",
        "Sources/SchedulerCore/Nested/CheckpointNested.cpp",
    ]
    functions = [
        fn(files[2], "RunProductionSchedulerDaemon", 80),
        fn(files[0], "CheckpointAnalysisOrchestrationService::runOne", 20),
        fn(files[3], "RunSchedulerDaemonCli", 10),
        fn(files[1], "CheckpointEvaluationService::evaluate", 40),
        fn(files[4], "Unrelated"),
    ]
    iface = FakeInterface(FakeIndex(list(reversed(files)), list(reversed(functions))))

    # A realistic bounded request surfaces canonical targets for checkpoint
    # orchestration/evaluation and scheduler-daemon wiring, with no source data.
    result = iface.dispatch(request())
    expected = {
        "schema_version": 1,
        "mode": "bounded_catalog_discovery",
        "evidentiary_status": "non_evidentiary",
        "required_follow_up": (
            "Use an existing bounded investigation operation before making "
            "repository-derived behavioral claims."
        ),
        "resolved_scope": "Sources/SchedulerCore",
        "scope_direct_file_count": 5,
        "query_groups": [["analysis", "checkpoint"], ["checkpoint", "evaluation"], ["daemon", "production", "scheduler"]],
        "matched_catalog_counts": {"files": 3, "symbols": 3, "total": 6},
        "candidates": {
            "files": sorted([files[0], files[1], files[2]]),
            "symbols": [
                {"identity": "CheckpointAnalysisOrchestrationService::runOne", "kind": "function_definition", "container_file": files[0], "container_ordinal": 1, "container_definition_count": 1},
                {"identity": "CheckpointEvaluationService::evaluate", "kind": "function_definition", "container_file": files[1], "container_ordinal": 1, "container_definition_count": 1},
                {"identity": "RunProductionSchedulerDaemon", "kind": "function_definition", "container_file": files[2], "container_ordinal": 1, "container_definition_count": 1},
            ],
        },
        "candidate_counts": {"files": 3, "symbols": 3, "total": 6},
        "caps": {
            "catalog_files": 100000, "catalog_directories": 200000,
            "matched_catalog_items": 48, "candidate_files": 16,
            "candidate_symbols": 16, "total_candidates": 24,
        },
    }
    assert result == expected
    encoded = json.dumps(result, sort_keys=True)
    assert "SOURCE_SENTINEL_MUST_NOT_LEAK" not in encoded
    assert "excerpt" not in encoded and "source_sha256" not in encoded
    assert iface._claim_runtime is None
    assert not READ_CALLS

    # Metadata ordering is stable despite filesystem/index enumeration order.
    reordered = FakeInterface(FakeIndex(files, functions)).dispatch(request())
    assert reordered == result
    reordered_groups = iface.dispatch(request(query_groups=[
        ["scheduler", "daemon", "production"], ["evaluation", "checkpoint"],
        ["checkpoint", "analysis"],
    ]))
    assert reordered_groups == result

    # Terms are conjunctive inside one group and never combine across catalog
    # objects; groups are alternatives.
    and_files = [
        "Sources/SchedulerCore/CheckpointOnly.cpp",
        "Sources/SchedulerCore/AnalysisOnly.cpp",
        "Sources/SchedulerCore/ProductionSchedulerDaemon.cpp",
    ]
    and_symbols = [
        fn(and_files[0], "CheckpointOnly"), fn(and_files[1], "AnalysisOnly"),
        fn(and_files[2], "RunProductionSchedulerDaemon"),
    ]
    and_iface = FakeInterface(FakeIndex(and_files, and_symbols))
    no_cross_object = and_iface.dispatch(request(query_groups=[["checkpoint", "analysis"]]))
    assert no_cross_object["candidate_counts"] == {"files": 0, "symbols": 0, "total": 0}
    or_groups = and_iface.dispatch(request(query_groups=[
        ["checkpoint", "analysis"], ["production", "scheduler", "daemon"],
    ]))
    assert or_groups["candidates"]["files"] == [and_files[2]]
    assert [row["identity"] for row in or_groups["candidates"]["symbols"]] == ["RunProductionSchedulerDaemon"]

    # Scope traversal and malformed/unrestricted-search-like inputs are closed.
    rejected(lambda: iface.dispatch(request(scope="../SchedulerCore")), "normalized")
    rejected(lambda: iface.dispatch(request(scope="/Sources/SchedulerCore")), "normalized")
    rejected(lambda: iface.dispatch(request(scope=".")), "normalized")
    rejected(lambda: iface.dispatch(request(query_groups=[])), "query_groups")
    rejected(lambda: iface.dispatch(request(query_groups=[[]])), "query_groups")
    rejected(lambda: iface.dispatch(request(query_groups=[["checkpoint", ".*"]])), "identifier")
    rejected(lambda: iface.dispatch(request(query_groups=[["source contents"]])), "identifier")
    rejected(lambda: iface.dispatch(request(query_groups=[["checkpoint", "checkpoint"]])), "duplicates")
    rejected(lambda: iface.dispatch(request(query_groups=[["checkpoint"], ["checkpoint"]])), "duplicate groups")
    rejected(lambda: iface.dispatch({"op": "discover_catalog_targets", "scope": "SchedulerCore", "query_terms": ["checkpoint"]}), "unexpected request fields")
    rejected(lambda: iface.dispatch(request(excerpt="please return source")), "unexpected request fields")
    rejected(lambda: iface.dispatch(request(source="please return source")), "unexpected request fields")
    rejected(lambda: iface.dispatch(request(content=True)), "unexpected request fields")
    rejected(lambda: iface.dispatch(request(path="../../...")), "unexpected request fields")
    rejected(lambda: iface.dispatch(request(limit=500)), "unexpected request fields")

    # A leaf logical scope with multiple catalog roots never chooses one.
    ambiguous = FakeInterface(FakeIndex([
        "Sources/One/Core/A.cpp", "LSTM/Two/Core/B.cpp",
    ], [fn("Sources/One/Core/A.cpp", "CheckpointA"), fn("LSTM/Two/Core/B.cpp", "DaemonB")]))
    rejected(lambda: ambiguous.dispatch(request(scope="Core")), "ambiguous")

    # A direct directory above the former 64-file scope limit remains usable
    # for a narrow metadata query.  The nested file never participates.
    many_scope = [f"Sources/SchedulerCore/File{n}.cpp" for n in range(64)] + [
        "Sources/SchedulerCore/CheckpointNeedle.cpp",
        "Sources/SchedulerCore/Nested/CheckpointIgnored.cpp",
    ]
    narrow = FakeInterface(FakeIndex(many_scope, [
        fn("Sources/SchedulerCore/CheckpointNeedle.cpp", "CheckpointNeedle"),
    ])).dispatch(request(query_groups=[["checkpoint"]]))
    assert narrow["scope_direct_file_count"] == 65
    assert narrow["matched_catalog_counts"] == {"files": 1, "symbols": 1, "total": 2}
    assert narrow["candidates"]["files"] == ["Sources/SchedulerCore/CheckpointNeedle.cpp"]

    # Broad metadata matching fails before final candidate selection and
    # returns no partial result.
    broad_files = [f"Sources/SchedulerCore/CheckpointAnalysisBroad{n}.cpp" for n in range(49)]
    rejected(lambda: FakeInterface(FakeIndex(broad_files, [])).dispatch(
        request(query_groups=[["checkpoint"], ["checkpoint", "analysis"]])), "matched catalog population cap")

    # Final candidate limits still fail before partial output is returned.
    many_files = [f"Sources/SchedulerCore/CheckpointFile{n}.cpp" for n in range(17)]
    rejected(lambda: FakeInterface(FakeIndex(many_files, [])).dispatch(request(query_groups=[["checkpoint"]])), "file cap")
    symbol_file = "Sources/SchedulerCore/Single.cpp"
    many_symbols = [fn(symbol_file, f"CheckpointSymbol{n}", n + 1) for n in range(17)]
    rejected(lambda: FakeInterface(FakeIndex([symbol_file], many_symbols)).dispatch(request(query_groups=[["checkpoint"]])), "symbol cap")

    # A total cap rejects a result which is individually below file/symbol caps.
    total_files = [f"Sources/SchedulerCore/CheckpointFile{n}.cpp" for n in range(12)]
    total_symbols = [fn("Sources/SchedulerCore/Other.cpp", f"DaemonSymbol{n}", n + 1) for n in range(13)]
    rejected(lambda: FakeInterface(FakeIndex(total_files + ["Sources/SchedulerCore/Other.cpp"], total_symbols)).dispatch(
        request(query_groups=[["checkpoint"], ["daemon"]])), "total candidate cap")

    # Canonical containers preserve legitimate same-named indexed definitions
    # in different files.  Only duplicate identity/container pairs are
    # ambiguous; source lines are neither returned nor used as identity.
    duplicate_name_files = [
        "Sources/SchedulerCore/DaemonOne.cpp", "Sources/SchedulerCore/DaemonTwo.cpp",
    ]
    duplicate_name_result = FakeInterface(FakeIndex(duplicate_name_files, [
        fn(duplicate_name_files[1], "DaemonEntry", 2),
        fn(duplicate_name_files[0], "DaemonEntry", 1),
    ])).dispatch(request(query_groups=[["daemon"]]))
    assert duplicate_name_result["candidates"]["symbols"] == [
        {"identity": "DaemonEntry", "kind": "function_definition", "container_file": duplicate_name_files[0], "container_ordinal": 1, "container_definition_count": 1},
        {"identity": "DaemonEntry", "kind": "function_definition", "container_file": duplicate_name_files[1], "container_ordinal": 1, "container_definition_count": 1},
    ]
    overloads = FakeInterface(FakeIndex([duplicate_name_files[0]], [
        fn(duplicate_name_files[0], "DaemonEntry", 1),
        fn(duplicate_name_files[0], "DaemonEntry", 20),
    ])).dispatch(request(query_groups=[["daemon"]]))
    assert overloads["candidates"]["symbols"] == [
        {"identity": "DaemonEntry", "kind": "function_definition", "container_file": duplicate_name_files[0], "container_ordinal": 1, "container_definition_count": 2},
        {"identity": "DaemonEntry", "kind": "function_definition", "container_file": duplicate_name_files[0], "container_ordinal": 2, "container_definition_count": 2},
    ]
    rejected(lambda: FakeInterface(FakeIndex([duplicate_name_files[0]], [
        fn(duplicate_name_files[0], "DaemonEntry", 1),
        fn(duplicate_name_files[0], "DaemonEntry", 1),
    ])).dispatch(request(query_groups=[["daemon"]])), "duplicate function definition metadata")

    print("test_codex_catalog_discovery: PASS")


if __name__ == "__main__":
    main()
