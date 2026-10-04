#!/usr/bin/env python3
"""Deterministic coverage for explicit relationship-set investigations."""
from __future__ import annotations

import sys
import tempfile
import types
from pathlib import Path

stub = types.ModuleType("expertadvisor_agent")
stub.list_files = lambda prefix="": []
stub.search = lambda pattern, max_results=100: ""
stub.read_file = lambda name, start=1, end=200: "stub source"
sys.modules.setdefault("expertadvisor_agent", stub)

from ..codex_interface import CodexRepositoryInterface
from ..repository_index import CallSite, FunctionBoundary


class FakeRuntime:
    def __init__(self, supports=True):
        self.calls = []
        self.supports = supports

    def _verdict(self):
        return {
            "supports": self.supports,
            "establishes": "evidence establishes claim" if self.supports else "",
            "reason": "test" if self.supports else "evidence does not establish claim",
            "model_turns": 1,
        }

    def verify_claim(self, topic, claim, item):
        self.calls.append(("single", item))
        return self._verdict()

    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle", items))
        return self._verdict()


class FakeIndex:
    def __init__(self, edges, boundaries=None):
        self.edges = edges
        self.boundaries = boundaries or {}
        self.relationship_calls = []

    def relationship(self, caller, callee):
        self.relationship_calls.append((caller, callee))
        return list(self.edges.get((caller, callee), []))

    def function_for_line(self, file, line):
        return self.boundaries.get((file, line))


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.claim_reads = []

    def _idx(self):
        return self.index

    def _claim_item(self, spec):
        file = self._required_text(spec, "file")
        start, end = int(spec["start"]), int(spec["end"])
        self.claim_reads.append((file, start, end))
        return {"file": file, "start": start, "end": end,
                "excerpt": f"{file}:{start}-{end}: independently reread source"}


def edge(file, line, caller, callee):
    return CallSite(file, line, caller, callee)


def request(relationships=None, **extra):
    return {
        "op": "investigate_relationship_set_claim",
        "topic_id": "relationship-set-topic",
        "topic": "relationship set verification",
        "claim": "the explicit direct relationships establish this claim",
        "relationships": relationships if relationships is not None else [
            {"caller": "A", "callee": "B"},
            {"caller": "A", "callee": "C"},
        ],
        **extra,
    }


def make(index, runtime, root, name):
    return FakeInterface(index, claim_runtime=runtime, claim_ledger_path=str(root / f"{name}.json"))


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid relationship-set request was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Basic fan-out retains caller-supplied relationship provenance while
        # routing two selected ranges through the existing bundle verifier.
        runtime = FakeRuntime()
        index = FakeIndex({
            ("A", "B"): [edge("Sources/A.cpp", 30, "A", "B")],
            ("A", "C"): [edge("Sources/C.cpp", 60, "A", "C")],
        })
        iface = make(index, runtime, root, "fan-out")
        result = iface.dispatch(request())
        manifest = result["selection_manifest"]
        assert index.relationship_calls == [("A", "B"), ("A", "C")]
        assert manifest["relationship_count"] == 2
        assert [(item["relationship_index"], item["caller"], item["callee"])
                for item in manifest["relationships"]] == [(0, "A", "B"), (1, "A", "C")]
        assert manifest["route"] == "multi_range"
        assert result["verification"]["manifest"]["mode"] == "targeted_source_bundle_claim"
        assert runtime.calls[0][0] == "bundle"
        assert iface.claim_reads == [("Sources/A.cpp", 18, 42), ("Sources/C.cpp", 48, 72)]

        # Fan-in is equally valid; the set is not required to be a chain.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("B", "D"): [edge("Sources/B.cpp", 40, "B", "D")],
            ("C", "D"): [edge("Sources/C.cpp", 40, "C", "D")],
        }), runtime, root, "fan-in")
        result = iface.dispatch(request([
            {"caller": "B", "callee": "D"}, {"caller": "C", "callee": "D"},
        ], topic_id="fan-in"))
        assert result["selection_manifest"]["selection_status"] == "selected"
        assert runtime.calls[0][0] == "bundle"

        # Three-way fan-out is supported without deriving any relationships.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge("Sources/B.cpp", 40, "A", "B")],
            ("A", "C"): [edge("Sources/C.cpp", 40, "A", "C")],
            ("A", "D"): [edge("Sources/D.cpp", 40, "A", "D")],
        }), runtime, root, "three-way")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "A", "callee": "C"},
            {"caller": "A", "callee": "D"},
        ], topic_id="three-way"))
        assert result["selection_manifest"]["relationship_count"] == 3
        assert result["selection_manifest"]["effective_range_count"] == 3

        # Request-level duplicate pairs are rejected before structural lookup,
        # source rereads, or model invocation.
        runtime = FakeRuntime()
        index = FakeIndex({})
        iface = make(index, runtime, root, "duplicate-request")
        expect_value_error(lambda: iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "A", "callee": "B"},
        ])))
        assert not index.relationship_calls and not iface.claim_reads and not runtime.calls

        # A missing edge invalidates the entire set only after all supplied
        # relationships have been structurally examined, before source reads.
        runtime = FakeRuntime()
        index = FakeIndex({("A", "B"): [edge("Sources/A.cpp", 40, "A", "B")]})
        iface = make(index, runtime, root, "missing")
        result = iface.dispatch(request(topic_id="missing"))
        assert index.relationship_calls == [("A", "B"), ("A", "C")]
        assert result["selection_manifest"]["relationships"][1]["selection_status"] == "no_direct_edge"
        assert not iface.claim_reads and not runtime.calls

        # A malformed indexed edge also fails closed before verification.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [{"file": "Sources/Bad.cpp", "line": "bad", "caller": "A", "callee": "B"}],
            ("A", "C"): [edge("Sources/C.cpp", 40, "A", "C")],
        }), runtime, root, "malformed")
        result = iface.dispatch(request(topic_id="malformed"))
        assert result["selection_manifest"]["relationships"][0]["selection_status"] == "invalid_direct_edge"
        assert not iface.claim_reads and not runtime.calls

        # Qualified mismatches retain the direct-relationship accounting and
        # do not substitute a same-short-name callee.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "Demo::B"): [edge("Sources/Wrong.cpp", 40, "A", "Other::B")],
            ("A", "C"): [edge("Sources/C.cpp", 40, "A", "C")],
        }), runtime, root, "qualified")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "Demo::B"}, {"caller": "A", "callee": "C"},
        ], topic_id="qualified"))
        entry = result["selection_manifest"]["relationships"][0]
        assert entry["normalization"]["qualified_mismatch_edge_count"] == 1
        assert entry["selection_status"] == "no_direct_edge"
        assert not iface.claim_reads and not runtime.calls

        # A shared physical candidate remains in each relationship's provenance
        # but appears once in evidence and selects the single-range verifier.
        shared = "Sources/Shared.cpp"
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(shared, 40, "A", "B")],
            ("C", "D"): [edge(shared, 40, "C", "D")],
        }), runtime, root, "dedupe")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "C", "callee": "D"},
        ], topic_id="dedupe"))
        manifest = result["selection_manifest"]
        assert all(item["candidate_range_count"] == 1 for item in manifest["relationships"])
        assert manifest["selected_ranges"] == [{"file": shared, "start": 28, "end": 52}]
        assert manifest["normalization"]["duplicate_candidate_range_count"] == 1
        assert manifest["route"] == "single_range" and runtime.calls[0][0] == "single"

        # Same-function overlap/touching merges; distinct-function touching
        # windows remain distinct evidence items.
        same_file = "Sources/Merged.cpp"
        boundary = FunctionBoundary(same_file, "Demo::f", 1, 100)
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(same_file, 30, "A", "B")],
            ("C", "D"): [edge(same_file, 42, "C", "D")],
        }, {(same_file, 30): boundary, (same_file, 42): boundary}), runtime, root, "same-function")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "C", "callee": "D"},
        ], topic_id="same-function"))
        assert result["selection_manifest"]["selected_ranges"] == [{"file": same_file, "start": 18, "end": 54}]
        assert result["selection_manifest"]["normalization"]["merged_range_count"] == 1

        split_file = "Sources/Split.cpp"
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(split_file, 18, "A", "B")],
            ("C", "D"): [edge(split_file, 43, "C", "D")],
        }, {
            (split_file, 18): FunctionBoundary(split_file, "Demo::first", 1, 30),
            (split_file, 43): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "distinct-function")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "C", "callee": "D"},
        ], topic_id="distinct-function"))
        assert result["selection_manifest"]["effective_range_count"] == 2
        assert result["selection_manifest"]["route"] == "multi_range"

        # Actual overlap at two distinct function boundaries fails before reads.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(split_file, 40, "A", "B")],
            ("C", "D"): [edge(split_file, 42, "C", "D")],
        }, {
            (split_file, 40): FunctionBoundary(split_file, "Demo::first", 1, 60),
            (split_file, 42): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "cross-function")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "C", "callee": "D"},
        ], topic_id="cross-function"))
        assert result["selection_manifest"]["selection_status"] == "cross_function_overlap"
        assert not iface.claim_reads and not runtime.calls

        # The global eight-range cap applies only after all relationship
        # candidates are normalized.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(f"Sources/A{n}.cpp", 40, "A", "B") for n in range(5)],
            ("C", "D"): [edge(f"Sources/C{n}.cpp", 40, "C", "D") for n in range(5)],
        }), runtime, root, "too-many")
        result = iface.dispatch(request([
            {"caller": "A", "callee": "B"}, {"caller": "C", "callee": "D"},
        ], topic_id="too-many"))
        assert result["selection_manifest"]["effective_range_count"] == 10
        assert result["selection_manifest"]["selection_status"] == "too_many_ranges"
        assert not iface.claim_reads and not runtime.calls

        # Structural selection does not decide semantic support.
        runtime = FakeRuntime(supports=False)
        iface = make(FakeIndex({
            ("A", "B"): [edge("Sources/A.cpp", 40, "A", "B")],
            ("A", "C"): [edge("Sources/C.cpp", 40, "A", "C")],
        }), runtime, root, "unsupported")
        result = iface.dispatch(request(topic_id="unsupported"))
        assert result["selection_manifest"]["selection_status"] == "selected"
        assert result["verification"]["verdict"]["supports"] is False

        # Closed interface schema and bounded object request contract reject
        # malformed relationship input before the structural index is used.
        valid = make(FakeIndex({}), FakeRuntime(), root, "invalid-requests")
        expect_value_error(lambda: valid.dispatch(request(relationships=[])))
        expect_value_error(lambda: valid.dispatch(request(relationships=[{"caller": "A", "callee": "B"}])))
        expect_value_error(lambda: valid.dispatch(request(relationships=[
            {"caller": "A", "callee": "B"} for _ in range(6)])))
        expect_value_error(lambda: valid.dispatch(request(relationships=[
            {"caller": "A", "callee": "B", "extra": "x"}, {"caller": "C", "callee": "D"}])))
        expect_value_error(lambda: valid.dispatch(request(relationships=[
            {"caller": "", "callee": "B"}, {"caller": "C", "callee": "D"}])))
        expect_value_error(lambda: valid.dispatch(request(relationships=[
            {"caller": "A", "callee": 1}, {"caller": "C", "callee": "D"}])))
        expect_value_error(lambda: valid.dispatch(request(extra="forbidden")))
    print("test_codex_relationship_set_claim: PASS")


if __name__ == "__main__":
    main()
