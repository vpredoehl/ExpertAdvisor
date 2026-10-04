#!/usr/bin/env python3
"""Deterministic coverage for server-owned relationship range selection."""
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
    def __init__(self):
        self.calls = []

    def verify_claim(self, topic, claim, item):
        self.calls.append(("single", item))
        return {"supports": True, "establishes": "single evidence establishes claim",
                "reason": "test", "model_turns": 1}

    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle", items))
        return {"supports": True, "establishes": "bundle evidence establishes claim",
                "reason": "test", "model_turns": 1}


class FakeIndex:
    def __init__(self, edges, boundaries=None):
        self.edges = edges
        self.boundaries = boundaries or {}
        self.function_for_line_calls = []

    def relationship(self, caller, callee):
        return list(self.edges)

    def function_for_line(self, file, line):
        self.function_for_line_calls.append((file, line))
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
        assert 1 <= start <= end and end - start + 1 <= 500
        self.claim_reads.append((file, start, end))
        return {"file": file, "start": start, "end": end,
                "excerpt": f"{file}:{start}-{end}: independently reread source"}


def request(**extra):
    return {
        "op": "investigate_relationship_claim",
        "topic_id": "relationship-topic",
        "topic": "relationship verification",
        "claim": "caller directly invokes callee",
        "caller": "Demo::caller",
        "callee": "Demo::callee",
        **extra,
    }


def interface(index, runtime, ledger):
    return FakeInterface(index, claim_runtime=runtime, claim_ledger_path=str(ledger))


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid relationship request was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # A single direct edge uses the existing one-range path and clips its
        # call-centered radius to the enclosing function rather than sending it.
        runtime = FakeRuntime()
        one_index = FakeIndex(
            [CallSite("Sources/One.cpp", 20, "Demo::caller", "Demo::callee")],
            {("Sources/One.cpp", 20): FunctionBoundary("Sources/One.cpp", "Demo::caller", 8, 100)},
        )
        one = interface(one_index, runtime, root / "one.json")
        result = one.dispatch(request())
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": "Sources/One.cpp", "start": 8, "end": 32}
        ]
        assert result["selection_manifest"]["route"] == "single_range"
        assert result["verification"]["manifest"]["mode"] == "targeted_source_claim"
        assert runtime.calls[0][0] == "single"
        assert one.claim_reads == [("Sources/One.cpp", 8, 32)]
        assert one_index.function_for_line_calls == [("Sources/One.cpp", 20)]

        # Multiple unsorted edges use canonical range ordering and the existing
        # MultiRange path; its evidence is independently reread.
        runtime = FakeRuntime()
        many_index = FakeIndex([
            CallSite("Sources/B.cpp", 60, "Demo::caller", "Demo::callee"),
            CallSite("Sources/A.cpp", 30, "Demo::caller", "Demo::callee"),
        ])
        many = interface(many_index, runtime, root / "many.json")
        result = many.dispatch(request(topic_id="many"))
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": "Sources/A.cpp", "start": 18, "end": 42},
            {"file": "Sources/B.cpp", "start": 48, "end": 72},
        ]
        assert result["selection_manifest"]["route"] == "multi_range"
        assert result["verification"]["manifest"]["mode"] == "targeted_source_bundle_claim"
        assert [item["file"] for item in runtime.calls[0][1]] == ["Sources/A.cpp", "Sources/B.cpp"]
        assert many.claim_reads == [("Sources/A.cpp", 18, 42), ("Sources/B.cpp", 48, 72)]

        # No relationship and excessive effective ranges both fail closed before
        # either evidence rereads or model calls.
        runtime = FakeRuntime()
        none = interface(FakeIndex([]), runtime, root / "none.json")
        result = none.dispatch(request(topic_id="none"))
        assert result["selection_manifest"]["selection_status"] == "no_direct_edge"
        assert result["verification"]["model_turn_count"] == 0
        assert not runtime.calls and not none.claim_reads

        runtime = FakeRuntime()
        excessive = interface(FakeIndex([
            CallSite(f"Sources/{n}.cpp", 50, "Demo::caller", "Demo::callee") for n in range(9)
        ]), runtime, root / "excessive.json")
        result = excessive.dispatch(request(topic_id="excessive"))
        assert result["selection_manifest"]["selection_status"] == "too_many_ranges"
        assert result["selection_manifest"]["effective_range_count"] == 9
        assert result["verification"]["model_turn_count"] == 0
        assert not runtime.calls and not excessive.claim_reads

        # Duplicate edges are removed before range generation.
        duplicate = CallSite("Sources/Duplicate.cpp", 40, "Demo::caller", "Demo::callee")
        runtime = FakeRuntime()
        deduplicated = interface(FakeIndex([duplicate, duplicate]), runtime, root / "dedupe.json")
        result = deduplicated.dispatch(request(topic_id="dedupe"))
        manifest = result["selection_manifest"]
        assert manifest["raw_direct_edge_count"] == 2
        assert manifest["direct_edge_count"] == 1
        assert manifest["normalization"]["duplicate_direct_edge_count"] == 1

        # Invalid structural edges fail closed and are accounted separately
        # from valid duplicate edges in the selection provenance.
        valid_edge = CallSite(
            "Sources/Accounting.cpp", 40, "Demo::caller", "Demo::callee"
        )
        runtime = FakeRuntime()
        invalid_accounting = interface(
            FakeIndex([
                valid_edge,
                valid_edge,
                {"file": "Sources/Bad.cpp", "line": "not-a-line",
                 "caller": "Demo::caller", "callee": "Demo::callee"},
            ]),
            runtime,
            root / "invalid-accounting.json",
        )
        result = invalid_accounting.dispatch(
            request(topic_id="invalid-accounting")
        )
        manifest = result["selection_manifest"]
        assert manifest["raw_direct_edge_count"] == 3
        assert manifest["direct_edge_count"] == 1
        assert manifest["normalization"]["duplicate_direct_edge_count"] == 1
        assert manifest["normalization"]["invalid_direct_edge_count"] == 1
        assert manifest["selection_status"] == "invalid_direct_edge"
        assert result["verification"]["model_turn_count"] == 0
        assert not runtime.calls and not invalid_accounting.claim_reads

        # A qualified callee request must not accept a different qualified
        # symbol merely because RepositoryIndex.relationship() returned a
        # same-short-name edge.
        runtime = FakeRuntime()
        qualified_mismatch = interface(FakeIndex([
            CallSite("Sources/Wrong.cpp", 40, "Demo::caller", "Other::callee")
        ]), runtime, root / "qualified-mismatch.json")
        result = qualified_mismatch.dispatch(
            request(topic_id="qualified-mismatch")
        )
        manifest = result["selection_manifest"]
        assert manifest["raw_direct_edge_count"] == 1
        assert manifest["direct_edge_count"] == 0
        assert manifest["normalization"]["qualified_mismatch_edge_count"] == 1
        assert manifest["selection_status"] == "no_direct_edge"
        assert result["verification"]["model_turn_count"] == 0
        assert not runtime.calls and not qualified_mismatch.claim_reads

        # If exact and same-short-name qualified results are both returned,
        # only the exact qualified relationship is retained.
        runtime = FakeRuntime()
        qualified_exact = interface(FakeIndex([
            CallSite("Sources/Wrong.cpp", 40, "Demo::caller", "Other::callee"),
            CallSite("Sources/Right.cpp", 50, "Demo::caller", "Demo::callee"),
        ]), runtime, root / "qualified-exact.json")
        result = qualified_exact.dispatch(request(topic_id="qualified-exact"))
        manifest = result["selection_manifest"]
        assert manifest["raw_direct_edge_count"] == 2
        assert manifest["direct_edge_count"] == 1
        assert manifest["normalization"]["qualified_mismatch_edge_count"] == 1
        assert manifest["selected_ranges"] == [
            {"file": "Sources/Right.cpp", "start": 38, "end": 62}
        ]

        # Unqualified requests retain RepositoryIndex's intentional short-name
        # relationship behavior.
        runtime = FakeRuntime()
        unqualified = interface(FakeIndex([
            CallSite("Sources/Short.cpp", 40, "Demo::caller", "Other::callee")
        ]), runtime, root / "unqualified.json")
        result = unqualified.dispatch(
            request(topic_id="unqualified", callee="callee")
        )
        assert result["selection_manifest"]["direct_edge_count"] == 1
        assert result["selection_manifest"]["selection_status"] == "selected"

        # Touching ranges from distinct enclosing functions must remain distinct
        # evidence ranges instead of being merged across a semantic boundary.
        runtime = FakeRuntime()
        adjacent_functions = interface(
            FakeIndex(
                [
                    CallSite("Sources/Adjacent.cpp", 29, "Demo::caller", "Demo::callee"),
                    CallSite("Sources/Adjacent.cpp", 32, "Demo::caller", "Demo::callee"),
                ],
                {
                    ("Sources/Adjacent.cpp", 29): FunctionBoundary(
                        "Sources/Adjacent.cpp", "Demo::first", 1, 30
                    ),
                    ("Sources/Adjacent.cpp", 32): FunctionBoundary(
                        "Sources/Adjacent.cpp", "Demo::second", 31, 60
                    ),
                },
            ),
            runtime,
            root / "adjacent-functions.json",
        )
        result = adjacent_functions.dispatch(
            request(topic_id="adjacent-functions")
        )
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": "Sources/Adjacent.cpp", "start": 17, "end": 30},
            {"file": "Sources/Adjacent.cpp", "start": 31, "end": 44},
        ]
        assert result["selection_manifest"]["route"] == "multi_range"
        assert result["selection_manifest"]["normalization"]["merged_range_count"] == 0
        assert runtime.calls[0][0] == "bundle"

        # Overlapping/touching same-file windows are merged before MultiRange;
        # no overlapping source ranges reach the verifier.
        runtime = FakeRuntime()
        merged = interface(FakeIndex([
            CallSite("Sources/Merged.cpp", 42, "Demo::caller", "Demo::callee"),
            CallSite("Sources/Other.cpp", 100, "Demo::caller", "Demo::callee"),
            CallSite("Sources/Merged.cpp", 30, "Demo::caller", "Demo::callee"),
        ]), runtime, root / "merged.json")
        result = merged.dispatch(request(topic_id="merged"))
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": "Sources/Merged.cpp", "start": 18, "end": 54},
            {"file": "Sources/Other.cpp", "start": 88, "end": 112},
        ]
        assert result["selection_manifest"]["normalization"]["merged_range_count"] == 1
        assert runtime.calls[0][0] == "bundle"

        # If an overlapping chain would require more than 500 lines, it also
        # fails closed instead of passing an overlap to MultiRange.
        runtime = FakeRuntime()
        oversized_overlap = interface(FakeIndex([
            CallSite("Sources/Long.cpp", 13 + n * 24, "Demo::caller", "Demo::callee")
            for n in range(22)
        ]), runtime, root / "oversized-overlap.json")
        result = oversized_overlap.dispatch(request(topic_id="oversized-overlap"))
        assert result["selection_manifest"]["selection_status"] == "overlapping_ranges_exceed_limit"
        assert result["selection_manifest"]["normalization"]["unmergeable_overlapping_range_count"] == 1
        assert not runtime.calls and not oversized_overlap.claim_reads

        # In the absence of a function boundary, the valid positive range is
        # deterministically centered on the call site.
        runtime = FakeRuntime()
        unbounded = interface(FakeIndex([
            CallSite("Sources/NoBoundary.cpp", 5, "Demo::caller", "Demo::callee")
        ]), runtime, root / "unbounded.json")
        result = unbounded.dispatch(request(topic_id="unbounded"))
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": "Sources/NoBoundary.cpp", "start": 1, "end": 17}
        ]

        # Underlying edge order cannot affect the selected deterministic manifest.
        edges = [
            CallSite("Sources/Z.cpp", 90, "Demo::caller", "Demo::callee"),
            CallSite("Sources/Y.cpp", 10, "Demo::caller", "Demo::callee"),
        ]
        left = interface(FakeIndex(edges), FakeRuntime(), root / "left.json")
        right = interface(FakeIndex(list(reversed(edges))), FakeRuntime(), root / "right.json")
        assert left.dispatch(request(topic_id="ordered"))["selection_manifest"] == right.dispatch(
            request(topic_id="ordered")
        )["selection_manifest"]

        # Only the structural anchors are accepted; callers cannot supply a
        # query, source range, or model-directed evidence selection.
        expect_value_error(lambda: one.dispatch(request(query="callee")))
        expect_value_error(lambda: one.dispatch(request(ranges=[])))
        expect_value_error(lambda: one.dispatch(request(file="Sources/One.cpp")))
        expect_value_error(lambda: one.dispatch(request(topic="  ")))
        expect_value_error(lambda: one.dispatch(request(caller=42)))
    print("test_codex_relationship_claim: PASS")


if __name__ == "__main__":
    main()
