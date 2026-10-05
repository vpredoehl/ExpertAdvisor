#!/usr/bin/env python3
"""Deterministic coverage for explicit relationship-chain investigations."""
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
from ..claim_evidence import VerifiedClaimLedger
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

    def verify_relationship_bundle_claim(self, topic, claim, relationships):
        self.calls.append(("relationship", relationships))
        return {"supports": True, "establishes": "relationship evidence establishes claim",
                "reason": "test", "model_turns": 1}


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


def request(**extra):
    return {
        "op": "investigate_relationship_chain_claim",
        "topic_id": "chain-topic", "topic": "chain verification",
        "claim": "the explicit chain supports this semantic claim",
        "path": ["A", "B", "C"], **extra,
    }


def make(index, runtime, root, name):
    return FakeInterface(index, claim_runtime=runtime, claim_ledger_path=str(root / f"{name}.json"))


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid chain request was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Every hop is structural before the source is reread. The verifier gets
        # each caller/callee paired only with server-selected reread evidence.
        runtime = FakeRuntime()
        index = FakeIndex({
            ("A", "B"): [edge("Sources/A.cpp", 30, "A", "B")],
            ("B", "C"): [edge("Sources/B.cpp", 60, "B", "C")],
        })
        iface = make(index, runtime, root, "success")
        result = iface.dispatch(request())
        manifest = result["selection_manifest"]
        assert index.relationship_calls == [("A", "B"), ("B", "C")]
        assert manifest["hop_count"] == 2
        assert [hop["selection_status"] for hop in manifest["hops"]] == ["selected", "selected"]
        assert manifest["selected_ranges"] == [
            {"file": "Sources/A.cpp", "start": 18, "end": 42},
            {"file": "Sources/B.cpp", "start": 48, "end": 72},
        ]
        assert manifest["route"] == "multi_range"
        assert result["verification"]["manifest"]["mode"] == "relationship_aware_claim"
        assert runtime.calls[0][0] == "relationship"
        checked = runtime.calls[0][1]
        assert [(item["caller"], item["callee"]) for item in checked] == [("A", "B"), ("B", "C")]
        assert [(item["file"], item["start"], item["end"]) for item in checked[0]["evidence"]] == [("Sources/A.cpp", 18, 42)]
        assert [(item["file"], item["start"], item["end"]) for item in checked[1]["evidence"]] == [("Sources/B.cpp", 48, 72)]
        assert iface.claim_reads == [("Sources/A.cpp", 18, 42), ("Sources/B.cpp", 48, 72)]

        # A relationship-aware ledger hit is keyed separately from a generic
        # bundle decision, avoids a second verifier call, and still rereads all
        # exact source ranges before lookup.
        repeat = iface.dispatch(request())
        assert repeat["verification"]["verdict"]["bundle_ledger_hit"] is True
        assert len(runtime.calls) == 1
        assert iface.claim_reads == [
            ("Sources/A.cpp", 18, 42), ("Sources/B.cpp", 48, 72),
            ("Sources/A.cpp", 18, 42), ("Sources/B.cpp", 48, 72),
        ]

        # A generic rejected bundle with identical source cannot alias the
        # relationship-aware ledger identity.
        generic_runtime = FakeRuntime()
        generic_iface = make(index, generic_runtime, root, "generic-rejected")
        generic_ledger = VerifiedClaimLedger(root / "generic-rejected.json")
        generic_items = [
            {"file": "Sources/A.cpp", "start": 18, "end": 42,
             "excerpt": "Sources/A.cpp:18-42: independently reread source"},
            {"file": "Sources/B.cpp", "start": 48, "end": 72,
             "excerpt": "Sources/B.cpp:48-72: independently reread source"},
        ]
        generic_ledger.record_bundle_decision("chain-topic", request()["claim"], generic_items,
                                              {"supports": False, "establishes": "", "reason": "old generic rejection"})
        generic_result = generic_iface.dispatch(request())
        assert generic_result["verification"]["verdict"]["supports"] is True
        assert generic_runtime.calls[0][0] == "relationship"

        # A missing later hop fails the complete chain before evidence rereads.
        runtime = FakeRuntime()
        index = FakeIndex({("A", "B"): [edge("Sources/A.cpp", 30, "A", "B")]})
        iface = make(index, runtime, root, "missing-second")
        result = iface.dispatch(request(topic_id="missing-second"))
        assert result["selection_manifest"]["selection_status"] == "no_direct_edge"
        assert result["selection_manifest"]["hops"][-1]["hop_index"] == 1
        assert result["selection_manifest"]["hops"][-1]["caller"] == "B"
        assert result["verification"]["repository_read_count"] == 0
        assert not iface.claim_reads and not runtime.calls

        runtime = FakeRuntime()
        iface = make(FakeIndex({("B", "C"): [edge("Sources/B.cpp", 60, "B", "C")]}), runtime, root, "missing-first")
        result = iface.dispatch(request(topic_id="missing-first"))
        assert result["selection_manifest"]["selection_status"] == "no_direct_edge"
        assert len(result["selection_manifest"]["hops"]) == 2
        assert not iface.claim_reads and not runtime.calls

        # Qualified returned callees must exactly match the requested hop.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "Demo::B"): [edge("Sources/Wrong.cpp", 40, "A", "Other::B")],
        }), runtime, root, "qualified")
        result = iface.dispatch(request(topic_id="qualified", path=["A", "Demo::B", "C"]))
        hop = result["selection_manifest"]["hops"][0]
        assert hop["normalization"]["qualified_mismatch_edge_count"] == 1
        assert hop["selection_status"] == "no_direct_edge"
        assert not iface.claim_reads and not runtime.calls

        # One malformed structural result invalidates its hop and the chain.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [{"file": "Sources/Bad.cpp", "line": "bad", "caller": "A", "callee": "B"}],
        }), runtime, root, "malformed")
        result = iface.dispatch(request(topic_id="malformed"))
        hop = result["selection_manifest"]["hops"][0]
        assert hop["selection_status"] == "invalid_direct_edge"
        assert hop["normalization"]["invalid_direct_edge_count"] == 1
        assert not iface.claim_reads and not runtime.calls

        # Invalid function metadata is structural, too, and prevents any reread.
        runtime = FakeRuntime()
        bad_boundary_file = "Sources/BadBoundary.cpp"
        iface = make(FakeIndex({
            ("A", "B"): [edge(bad_boundary_file, 40, "A", "B")],
            ("B", "C"): [edge("Sources/Next.cpp", 40, "B", "C")],
        }, {
            (bad_boundary_file, 40): FunctionBoundary(bad_boundary_file, "Demo::bad", 50, 40),
        }), runtime, root, "bad-boundary")
        result = iface.dispatch(request(topic_id="bad-boundary"))
        assert result["selection_manifest"]["hops"][0]["selection_status"] == "invalid_function_boundary"
        assert not iface.claim_reads and not runtime.calls

        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(bad_boundary_file, 40, "A", "B")],
            ("B", "C"): [edge("Sources/Next.cpp", 40, "B", "C")],
        }, {
            (bad_boundary_file, 40): FunctionBoundary(bad_boundary_file, "Demo::empty", 100, 120),
        }), runtime, root, "bad-candidate")
        result = iface.dispatch(request(topic_id="bad-candidate"))
        assert result["selection_manifest"]["hops"][0]["selection_status"] == "invalid_candidate_range"
        assert not iface.claim_reads and not runtime.calls

        # Duplicate structural records are accounted before candidate selection.
        duplicate = edge("Sources/Duplicate.cpp", 40, "A", "B")
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [duplicate, duplicate],
            ("B", "C"): [edge("Sources/Next.cpp", 40, "B", "C")],
        }), runtime, root, "duplicates")
        result = iface.dispatch(request(topic_id="duplicates"))
        hop = result["selection_manifest"]["hops"][0]
        assert hop["raw_direct_edge_count"] == 2 and hop["direct_edge_count"] == 1
        assert hop["normalization"]["duplicate_direct_edge_count"] == 1

        # Final normalization deduplicates physical evidence used by both hops.
        shared = "Sources/Shared.cpp"
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(shared, 40, "A", "B")],
            ("B", "C"): [edge(shared, 40, "B", "C")],
        }), runtime, root, "physical-dedupe")
        result = iface.dispatch(request(topic_id="physical-dedupe"))
        assert result["selection_manifest"]["candidate_range_count"] == 2
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": shared, "start": 28, "end": 52}
        ]
        assert result["selection_manifest"]["normalization"]["duplicate_candidate_range_count"] == 1
        assert result["selection_manifest"]["route"] == "single_range"
        assert runtime.calls[0][0] == "relationship"

        # Same-function candidates merge, while touching different functions do not.
        same_file = "Sources/Merged.cpp"
        boundary = FunctionBoundary(same_file, "Demo::f", 1, 100)
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(same_file, 30, "A", "B")],
            ("B", "C"): [edge(same_file, 42, "B", "C")],
        }, {(same_file, 30): boundary, (same_file, 42): boundary}), runtime, root, "same-function")
        result = iface.dispatch(request(topic_id="same-function"))
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": same_file, "start": 18, "end": 54}
        ]
        assert result["selection_manifest"]["normalization"]["merged_range_count"] == 1

        split_file = "Sources/Split.cpp"
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(split_file, 18, "A", "B")],
            ("B", "C"): [edge(split_file, 43, "B", "C")],
        }, {
            (split_file, 18): FunctionBoundary(split_file, "Demo::first", 1, 30),
            (split_file, 43): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "distinct-function")
        result = iface.dispatch(request(topic_id="distinct-function"))
        assert result["selection_manifest"]["selected_ranges"] == [
            {"file": split_file, "start": 6, "end": 30},
            {"file": split_file, "start": 31, "end": 55},
        ]
        assert result["selection_manifest"]["route"] == "multi_range"

        # Overlap across distinct enclosing functions cannot be passed to MultiRange.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(split_file, 40, "A", "B")],
            ("B", "C"): [edge(split_file, 42, "B", "C")],
        }, {
            (split_file, 40): FunctionBoundary(split_file, "Demo::first", 1, 60),
            (split_file, 42): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "cross-function")
        result = iface.dispatch(request(topic_id="cross-function"))
        assert result["selection_manifest"]["selection_status"] == "cross_function_overlap"
        assert not iface.claim_reads and not runtime.calls

        # The global maximum applies after all hop candidates are normalized.
        runtime = FakeRuntime()
        iface = make(FakeIndex({
            ("A", "B"): [edge(f"Sources/A{n}.cpp", 40, "A", "B") for n in range(5)],
            ("B", "C"): [edge(f"Sources/B{n}.cpp", 40, "B", "C") for n in range(5)],
        }), runtime, root, "too-many")
        result = iface.dispatch(request(topic_id="too-many"))
        assert result["selection_manifest"]["effective_range_count"] == 10
        assert result["selection_manifest"]["selection_status"] == "too_many_ranges"
        assert not iface.claim_reads and not runtime.calls

        # Closed request/path schema and the deliberately acyclic path policy.
        valid = make(FakeIndex({}), FakeRuntime(), root, "invalid-paths")
        expect_value_error(lambda: valid.dispatch(request(path=["A", "B"])))
        expect_value_error(lambda: valid.dispatch(request(path=["A", "B", "C", "D", "E", "F"])))
        expect_value_error(lambda: valid.dispatch(request(path="A -> B -> C")))
        expect_value_error(lambda: valid.dispatch(request(path=["A", "", "C"])))
        expect_value_error(lambda: valid.dispatch(request(path=["A", 3, "C"])))
        expect_value_error(lambda: valid.dispatch(request(path=["A", "B", "A"])))
        for field, value in (("query", "x"), ("ranges", []), ("file", "x"),
                             ("caller", "A"), ("callee", "B")):
            expect_value_error(lambda field=field, value=value: valid.dispatch(request(**{field: value})))

        # Sorting structural edges makes the complete selection manifest stable.
        unordered = {
            ("A", "B"): [edge("Sources/Z.cpp", 90, "A", "B"), edge("Sources/Y.cpp", 10, "A", "B")],
            ("B", "C"): [edge("Sources/X.cpp", 50, "B", "C")],
        }
        left = make(FakeIndex(unordered), FakeRuntime(), root, "left")
        right = make(FakeIndex({key: list(reversed(value)) for key, value in unordered.items()}), FakeRuntime(), root, "right")
        assert left.dispatch(request(topic_id="ordered"))["selection_manifest"] == right.dispatch(
            request(topic_id="ordered")
        )["selection_manifest"]
    print("test_codex_relationship_chain_claim: PASS")


if __name__ == "__main__":
    main()
