#!/usr/bin/env python3
"""Deterministic coverage for bounded server-owned symbol investigation."""
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
from ..repository_index import CallSite, FunctionBoundary, SymbolOccurrence


class FakeRuntime:
    def __init__(self, supports=True):
        self.calls = []
        self.supports = supports

    def verify_claim(self, topic, claim, item):
        self.calls.append(("single", topic, claim, item))
        return {
            "supports": self.supports,
            "establishes": "narrow source-visible behavior" if self.supports else "",
            "reason": "test semantic result",
            "model_turns": 1,
        }

    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle", topic, claim, items))
        return {
            "supports": self.supports,
            "establishes": "narrow bundled source-visible behavior" if self.supports else "",
            "reason": "test semantic result",
            "model_turns": 1,
        }


class FakeIndex:
    def __init__(self, definitions, boundaries, *, incoming=(), outgoing=()):
        self.definitions = definitions
        self.boundaries = boundaries
        self.incoming = list(incoming)
        self.outgoing = list(outgoing)
        self.find_calls = []
        self.caller_calls = []
        self.callee_calls = []

    def find_definitions(self, symbol):
        self.find_calls.append(symbol)
        return list(self.definitions.get(symbol, ()))

    def function_for_line(self, file, line):
        return self.boundaries.get((file, line))

    def callers_of(self, symbol):
        self.caller_calls.append(symbol)
        return list(self.incoming)

    def callees_of(self, symbol):
        self.callee_calls.append(symbol)
        return list(self.outgoing)


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index, **kwargs):
        super().__init__(**kwargs)
        self.index = index
        self.claim_reads = []
        self.revision = 1

    def _idx(self):
        return self.index

    def _claim_item(self, spec):
        file = self._required_text(spec, "file")
        start, end = int(spec["start"]), int(spec["end"])
        assert 1 <= start <= end and end - start + 1 <= 500
        self.claim_reads.append((file, start, end))
        return {
            "file": file, "start": start, "end": end,
            "excerpt": f"{file}:{start}-{end}: independently reread source revision={self.revision}",
        }


TARGET = "Demo::target"


def target_definition(file="Sources/Target.cpp", start=10, end=30):
    return (
        SymbolOccurrence(file, start, TARGET, "definition", TARGET),
        FunctionBoundary(file, TARGET, start, end),
    )


def index_for_target(*, file="Sources/Target.cpp", start=10, end=30, incoming=(), outgoing=(), extra_boundaries=None):
    definition, boundary = target_definition(file, start, end)
    boundaries = {(file, start): boundary}
    if extra_boundaries:
        boundaries.update(extra_boundaries)
    return FakeIndex({TARGET: [definition]}, boundaries, incoming=incoming, outgoing=outgoing)


def request(**extra):
    return {
        "op": "investigate_symbol",
        "topic_id": "symbol-topic",
        "topic": "symbol investigation",
        "symbol": TARGET,
        **extra,
    }


def make(index, runtime, root, name):
    return FakeInterface(index, claim_runtime=runtime, claim_ledger_path=str(root / f"{name}.json"))


def edge(file, line, caller, callee):
    return CallSite(file, line, caller, callee)


def expect_value_error(action):
    try:
        action()
    except ValueError:
        return
    raise AssertionError("invalid symbol request was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)

        # Exact resolution makes the definition extent primary evidence and
        # routes an otherwise self-contained investigation to the one-range verifier.
        runtime = FakeRuntime()
        iface = make(index_for_target(), runtime, root, "exact")
        result = iface.dispatch(request())
        manifest = result["selection_manifest"]
        assert manifest["requested_symbol"] == TARGET
        assert manifest["resolved_canonical_symbol"] == TARGET
        assert manifest["definition_source_extent"] == {"file": "Sources/Target.cpp", "start": 10, "end": 30}
        assert manifest["selected_ranges"] == [{"file": "Sources/Target.cpp", "start": 10, "end": 30}]
        assert manifest["route"] == "single_range"
        assert runtime.calls[0][0] == "single"

        # Missing, ambiguous, and malformed index metadata fail before any
        # source reread or semantic verifier invocation.
        runtime = FakeRuntime()
        missing = make(FakeIndex({}, {}), runtime, root, "missing")
        result = missing.dispatch(request())
        assert result["selection_manifest"]["selection_status"] == "missing_symbol"
        assert not missing.claim_reads and not runtime.calls

        duplicate, boundary = target_definition()
        runtime = FakeRuntime()
        ambiguous = make(FakeIndex({TARGET: [duplicate, duplicate]}, {(duplicate.file, duplicate.line): boundary}), runtime, root, "ambiguous")
        result = ambiguous.dispatch(request())
        assert result["selection_manifest"]["selection_status"] == "ambiguous_symbol"
        assert not ambiguous.claim_reads and not runtime.calls

        runtime = FakeRuntime()
        malformed = make(FakeIndex({TARGET: [duplicate]}, {(duplicate.file, duplicate.line): FunctionBoundary(duplicate.file, TARGET, 11, 30)}), runtime, root, "malformed")
        result = malformed.dispatch(request())
        assert result["selection_manifest"]["selection_status"] == "malformed_symbol_extent"
        assert not malformed.claim_reads and not runtime.calls

        # The operation has an exact closed schema and strict string fields.
        valid = make(index_for_target(), FakeRuntime(), root, "validation")
        expect_value_error(lambda: valid.dispatch(request(extra="forbidden")))
        expect_value_error(lambda: valid.dispatch(request(topic_id="")))
        expect_value_error(lambda: valid.dispatch(request(topic_id=1)))
        expect_value_error(lambda: valid.dispatch(request(topic="  ")))
        expect_value_error(lambda: valid.dispatch(request(topic=1)))
        expect_value_error(lambda: valid.dispatch(request(symbol="")))
        expect_value_error(lambda: valid.dispatch(request(symbol=1)))

        # Incoming and outgoing records are selected in stable physical order;
        # only the exact resolved endpoint is asked for, never a recursive walk.
        incoming_file = "Sources/Incoming.cpp"
        incoming_boundary = FunctionBoundary(incoming_file, "Demo::caller", 1, 100)
        target_boundary = FunctionBoundary("Sources/Target.cpp", TARGET, 10, 30)
        runtime = FakeRuntime()
        indexed = index_for_target(
            incoming=[edge(incoming_file, 40, "Demo::caller", TARGET)],
            outgoing=[edge("Sources/Target.cpp", 20, TARGET, "Demo::child")],
            extra_boundaries={(incoming_file, 40): incoming_boundary, ("Sources/Target.cpp", 20): target_boundary},
        )
        iface = make(indexed, runtime, root, "direct")
        result = iface.dispatch(request(topic_id="direct"))
        manifest = result["selection_manifest"]
        assert manifest["direct_structural_relationship_count"] == 2
        assert [(row["file"], row["line"]) for row in manifest["admitted_direct_relationships"]] == [
            (incoming_file, 40), ("Sources/Target.cpp", 20)
        ]
        assert indexed.caller_calls == [TARGET] and indexed.callee_calls == [TARGET]
        assert manifest["route"] == "multi_range" and runtime.calls[0][0] == "bundle"

        # Identical physical candidate evidence dedupes. A self-recursive
        # direct record is represented once, with both structural directions.
        self_boundary = FunctionBoundary("Sources/Self.cpp", TARGET, 1, 25)
        runtime = FakeRuntime()
        self_iface = make(index_for_target(file="Sources/Self.cpp", start=1, end=25,
            incoming=[edge("Sources/Self.cpp", 13, TARGET, TARGET)],
            outgoing=[edge("Sources/Self.cpp", 13, TARGET, TARGET)],
            extra_boundaries={("Sources/Self.cpp", 13): self_boundary}), runtime, root, "dedupe")
        result = self_iface.dispatch(request(topic_id="dedupe"))
        assert result["selection_manifest"]["normalization"]["duplicate_candidate_range_count"] == 1
        assert result["selection_manifest"]["effective_range_count"] == 1

        # Same-function touching/overlap merges, but function boundaries keep
        # touching ranges separate and cross-function overlap fails closed.
        merged_file = "Sources/Merged.cpp"
        caller_boundary = FunctionBoundary(merged_file, "Demo::caller", 1, 100)
        runtime = FakeRuntime()
        merged = make(index_for_target(incoming=[
            edge(merged_file, 30, "Demo::caller", TARGET),
            edge(merged_file, 42, "Demo::caller", TARGET),
        ], extra_boundaries={(merged_file, 30): caller_boundary, (merged_file, 42): caller_boundary}), runtime, root, "merged")
        result = merged.dispatch(request(topic_id="merged"))
        assert result["selection_manifest"]["normalization"]["merged_range_count"] == 1

        split_file = "Sources/Split.cpp"
        runtime = FakeRuntime()
        split = make(index_for_target(incoming=[
            edge(split_file, 18, "Demo::first", TARGET),
            edge(split_file, 43, "Demo::second", TARGET),
        ], extra_boundaries={
            (split_file, 18): FunctionBoundary(split_file, "Demo::first", 1, 30),
            (split_file, 43): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "split")
        result = split.dispatch(request(topic_id="split"))
        assert result["selection_manifest"]["effective_range_count"] == 3
        assert result["selection_manifest"]["route"] == "multi_range"

        runtime = FakeRuntime()
        cross = make(index_for_target(incoming=[
            edge(split_file, 40, "Demo::first", TARGET),
            edge(split_file, 42, "Demo::second", TARGET),
        ], extra_boundaries={
            (split_file, 40): FunctionBoundary(split_file, "Demo::first", 1, 60),
            (split_file, 42): FunctionBoundary(split_file, "Demo::second", 31, 80),
        }), runtime, root, "cross")
        result = cross.dispatch(request(topic_id="cross"))
        assert result["selection_manifest"]["selection_status"] == "cross_function_overlap"
        assert not cross.claim_reads and not runtime.calls

        # The global eight-range and individual 500-line limits are failures,
        # never silent evidence loss.
        runtime = FakeRuntime()
        excessive = make(index_for_target(incoming=[
            edge(f"Sources/Incoming{n}.cpp", 40, f"Demo::caller{n}", TARGET) for n in range(8)
        ]), runtime, root, "too-many")
        result = excessive.dispatch(request(topic_id="too-many"))
        assert result["selection_manifest"]["effective_range_count"] == 9
        assert result["selection_manifest"]["selection_status"] == "too_many_ranges"
        assert not excessive.claim_reads and not runtime.calls

        runtime = FakeRuntime()
        relationship_bound = make(index_for_target(incoming=[
            edge(f"Sources/Relation{n}.cpp", 40, f"Demo::caller{n}", TARGET) for n in range(17)
        ]), runtime, root, "relationship-bound")
        result = relationship_bound.dispatch(request(topic_id="relationship-bound"))
        assert result["selection_manifest"]["selection_status"] == "too_many_direct_relationships"
        assert result["selection_manifest"]["direct_structural_relationship_count"] == 17
        assert not relationship_bound.claim_reads and not runtime.calls

        runtime = FakeRuntime()
        long_extent = make(index_for_target(start=1, end=501), runtime, root, "long")
        result = long_extent.dispatch(request(topic_id="long"))
        assert result["selection_manifest"]["selection_status"] == "symbol_extent_over_limit"
        assert not long_extent.claim_reads and not runtime.calls

        # Structural selection and semantic judgment are independent.
        runtime = FakeRuntime(supports=False)
        unsupported = make(index_for_target(), runtime, root, "unsupported")
        result = unsupported.dispatch(request(topic_id="unsupported"))
        assert result["selection_manifest"]["selection_status"] == "selected"
        assert result["verification"]["verdict"]["supports"] is False

        # Ledger reuse does no model work but repeats the exact source reread.
        runtime = FakeRuntime()
        ledger_iface = make(index_for_target(), runtime, root, "ledger")
        first = ledger_iface.dispatch(request(topic_id="ledger"))
        second = ledger_iface.dispatch(request(topic_id="ledger"))
        assert first["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert second["verification"]["manifest"]["metrics"]["ledger_hit"] is True
        assert len(runtime.calls) == 1 and len(ledger_iface.claim_reads) == 2
        ledger_iface.revision = 2
        changed_source = ledger_iface.dispatch(request(topic_id="ledger"))
        assert changed_source["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert len(runtime.calls) == 2
        changed_topic = ledger_iface.dispatch(request(topic_id="ledger", topic="different semantic topic"))
        assert changed_topic["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert len(runtime.calls) == 3
    print("test_codex_symbol_investigation: PASS")


if __name__ == "__main__":
    main()
