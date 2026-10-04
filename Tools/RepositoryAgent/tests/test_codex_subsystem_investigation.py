#!/usr/bin/env python3
"""Adversarial coverage for bounded directory-root subsystem investigation."""
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

from ..codex_interface import (
    CodexRepositoryInterface, MAX_SUBSYSTEM_FILES, MAX_SUBSYSTEM_INVENTORY_SYMBOLS,
    MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS, MAX_SUBSYSTEM_SELECTED_SYMBOLS,
)
from ..repository_index import CallSite, FunctionBoundary


class FakeRuntime:
    def __init__(self): self.calls = []
    def verify_claim(self, topic, claim, item):
        self.calls.append(("single", topic, claim, item))
        return {"supports": True, "establishes": "bounded source behavior", "reason": "test", "model_turns": 1}
    def verify_bundle_claim(self, topic, claim, items):
        self.calls.append(("bundle", topic, claim, items))
        return {"supports": True, "establishes": "bounded source behavior", "reason": "test", "model_turns": 1}


class FakeIndex:
    def __init__(self, functions, *, incoming=None, outgoing=None):
        self.functions = list(functions)
        self._incoming = incoming or {}
        self._outgoing = outgoing or {}
        self.calls = []
    def function_for_line(self, file, line):
        for boundary in self.functions:
            if boundary.file == file and boundary.start_line <= line <= boundary.end_line:
                return boundary
        return None
    def callers_of(self, symbol):
        self.calls.append(("callers", symbol))
        return list(self._incoming.get(symbol, ()))
    def callees_of(self, symbol):
        self.calls.append(("callees", symbol))
        return list(self._outgoing.get(symbol, ()))


class FakeInterface(CodexRepositoryInterface):
    def __init__(self, index, *, files, directories=(), **kwargs):
        super().__init__(**kwargs); self.index = index; self.files = set(files); self.directories = set(directories); self.reads = []; self.revision = 1; self.listings = []
    def _idx(self): return self.index
    def _subsystem_exact_file_listing(self, file):
        self.listings.append(file)
        if file in self.files: return [file]
        if file in self.directories: return [f"{file}/Child.cpp"]
        return []
    def _claim_item(self, spec):
        self.reads.append((spec["file"], spec["start"], spec["end"]))
        return {**spec, "excerpt": f"{spec['file']}:{spec['start']}-{spec['end']}:revision={self.revision}"}


ROOT = "Sources/SchedulerCore"
RUN = "CheckpointAnalysisOrchestrationService::runOne"
PLAN = "PlanCheckpointAnalysisFinalization"


def req(**extra):
    return {"op": "investigate_subsystem", "topic_id": "subsystem-topic", "topic": "checkpoint analysis orchestration", "subsystem": ROOT, "files": ["Checkpoint.cpp"], **extra}


def make(root, functions, **kwargs):
    runtime = FakeRuntime()
    files = kwargs.pop("files", {f"{ROOT}/Checkpoint.cpp", f"{ROOT}/One.cpp", f"{ROOT}/Two.cpp", f"{ROOT}/Extra.cpp"})
    directories = kwargs.pop("directories", ())
    return FakeInterface(FakeIndex(functions, **kwargs), files=files, directories=directories, claim_runtime=runtime, claim_ledger_path=str(root / "ledger.json")), runtime


def expect_error(action):
    try: action()
    except ValueError: return
    raise AssertionError("invalid subsystem request was accepted")


def main():
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        run = FunctionBoundary(f"{ROOT}/Checkpoint.cpp", RUN, 10, 39)
        plan = FunctionBoundary(f"{ROOT}/Checkpoint.cpp", PLAN, 1, 9)
        external = FunctionBoundary("Sources/Elsewhere/Other.cpp", "External::work", 1, 10)
        internal = CallSite(run.file, 20, RUN, PLAN)
        outside = CallSite(run.file, 21, RUN, "External::work")
        iface, runtime = make(root, [plan, run, external], outgoing={RUN: [internal, outside]})
        result = iface.dispatch(req())
        manifest = result["selection_manifest"]
        assert manifest["selection_status"] == "selected"
        assert manifest["files"] == [f"{ROOT}/Checkpoint.cpp"]
        assert manifest["admitted_file_count"] == 1
        assert manifest["inventory_count"] == 2
        # Only the strongest lexical-match tier is selected; lower-score
        # internal endpoints may still be represented by direct call evidence.
        assert {row["symbol"] for row in manifest["selected_symbols"]} == {RUN}
        assert [(row["caller"], row["callee"]) for row in manifest["relationships"]] == [(RUN, PLAN)]
        assert manifest["relationship_accounting"]["external_relationship_count"] == 1
        assert all(row["file"].startswith(ROOT + "/") for row in manifest["selected_ranges"])
        # The call-site context is normalized with its enclosing selected
        # function range rather than being passed through as overlapping input.
        assert manifest["normalization"]["merged_range_count"] == 1
        assert result["verification"]["manifest"]["metrics"]["unrelated_topic_count"] == 0
        assert runtime.calls and not any(name == "search" for name, _ in iface.index.calls)
        assert set(name for _, name in iface.index.calls) <= {RUN, PLAN}
        assert iface.listings == [f"{ROOT}/Checkpoint.cpp"]

        # Exact schema and root validation reject traversal, outside-root syntax,
        # extra fields, empty/non-string fields before structural work.
        for bad in ("../Sources/SchedulerCore", "/Sources/SchedulerCore", "Sources//SchedulerCore", "Sources\\SchedulerCore"):
            expect_error(lambda bad=bad: iface.dispatch(req(subsystem=bad)))
        expect_error(lambda: iface.dispatch(req(extra="forbidden")))
        expect_error(lambda: iface.dispatch(req(subsystem="")))
        expect_error(lambda: iface.dispatch(req(subsystem=1)))
        expect_error(lambda: iface.dispatch({"op": "investigate_subsystem", "topic_id": "t", "topic": "topic", "subsystem": ROOT}))
        expect_error(lambda: iface.dispatch(req(files=[])))
        expect_error(lambda: iface.dispatch(req(files="Checkpoint.cpp")))
        for bad in ("/Checkpoint.cpp", "../Checkpoint.cpp", "one/../Checkpoint.cpp", "one\\Checkpoint.cpp", ".", "one//Checkpoint.cpp"):
            expect_error(lambda bad=bad: iface.dispatch(req(files=[bad])))
        expect_error(lambda: iface.dispatch(req(files=["Missing.cpp"])))
        expect_error(lambda: iface.dispatch(req(files=["Checkpoint.cpp", "Checkpoint.cpp"])))
        expect_error(lambda: iface.dispatch(req(files=["Checkpoint.cpp"] * (MAX_SUBSYSTEM_FILES + 1))))
        # A directory listing is not an attestation that the exact requested
        # path is a file, so directory input fails closed before indexing.
        directory_iface, _ = make(root, [run], directories={f"{ROOT}/Directory"})
        expect_error(lambda: directory_iface.dispatch(req(files=["Directory"])))
        # A qualified sibling cannot escape the containing subsystem.
        expect_error(lambda: iface.dispatch(req(files=["../../Elsewhere/Other.cpp"])))

        empty, empty_runtime = make(root, [external])
        result = empty.dispatch(req(topic_id="empty"))
        assert result["selection_manifest"]["selection_status"] == "no_subsystem_symbols"
        assert not empty.reads and not empty_runtime.calls

        # Duplicate heuristic names are accounted for globally, but only block
        # the request if its own strongest-match tier is ambiguous.
        duplicate_symbols = [
            FunctionBoundary(f"{ROOT}/One.cpp", "CheckpointDuplicate", 1, 2),
            FunctionBoundary(f"{ROOT}/Two.cpp", "CheckpointDuplicate", 1, 2),
        ]
        iface, runtime = make(root, duplicate_symbols)
        result = iface.dispatch(req(topic_id="ambiguous", topic="checkpoint", files=["One.cpp", "Two.cpp"]))
        assert result["selection_manifest"]["ambiguous_inventory_symbol_count"] == 1
        assert result["selection_manifest"]["selection_status"] == "ambiguous_topic_matched_symbols"
        assert not iface.reads and not runtime.calls

        # The cap is evaluated only after restriction to the requested file set.
        over_inventory = [FunctionBoundary(f"{ROOT}/Checkpoint.cpp", f"Function{n}", n * 3 + 1, n * 3 + 2) for n in range(MAX_SUBSYSTEM_INVENTORY_SYMBOLS + 1)]
        over_inventory.append(FunctionBoundary(f"{ROOT}/Other.cpp", "FunctionOutsideSet", 1, 2))
        iface, runtime = make(root, over_inventory)
        result = iface.dispatch(req(topic_id="over-inventory", topic="function"))
        assert result["selection_manifest"]["selection_status"] == "subsystem_inventory_over_limit"
        assert not iface.reads and not runtime.calls

        over_selected = [FunctionBoundary(f"{ROOT}/Checkpoint.cpp", f"CheckpointFunction{n}", n * 3 + 1, n * 3 + 2) for n in range(MAX_SUBSYSTEM_SELECTED_SYMBOLS + 1)]
        iface, runtime = make(root, over_selected)
        result = iface.dispatch(req(topic_id="over-selected", topic="checkpoint"))
        assert result["selection_manifest"]["selection_status"] == "too_many_topic_matched_symbols"
        assert not iface.reads and not runtime.calls

        child = FunctionBoundary(f"{ROOT}/Checkpoint.cpp", "Child", 1, 2)
        many_edges = [CallSite(run.file, 10 + n, RUN, "Child") for n in range(MAX_SUBSYSTEM_SELECTED_RELATIONSHIPS + 1)]
        iface, runtime = make(root, [run, child], outgoing={RUN: many_edges})
        result = iface.dispatch(req(topic_id="over-relationships", topic="checkpoint"))
        assert result["selection_manifest"]["selection_status"] == "too_many_subsystem_relationships"
        assert not iface.reads and not runtime.calls

        # Evidence-range and total-line caps are independent admission limits;
        # neither silently discards evidence nor invokes a verifier.
        over_ranges = [FunctionBoundary(f"{ROOT}/Checkpoint.cpp", f"CheckpointRange{n}", n * 3 + 1, n * 3 + 2) for n in range(9)]
        iface, runtime = make(root, over_ranges)
        result = iface.dispatch(req(topic_id="over-ranges", topic="checkpoint"))
        assert result["selection_manifest"]["selection_status"] == "too_many_ranges"
        assert not iface.reads and not runtime.calls

        over_lines = [FunctionBoundary(f"{ROOT}/Checkpoint.cpp", f"CheckpointLines{n}", n * 401 + 1, n * 401 + 401) for n in range(3)]
        iface, runtime = make(root, over_lines)
        result = iface.dispatch(req(topic_id="over-lines", topic="checkpoint"))
        assert result["selection_manifest"]["selection_status"] == "too_many_subsystem_evidence_lines"
        assert not iface.reads and not runtime.calls

        # Ledger reuse must freshly reread source, then invalidate on changed
        # content and on a different (but selector-equivalent) topic.
        iface, runtime = make(root, [plan, run], outgoing={RUN: [internal]})
        first = iface.dispatch(req(topic_id="ledger"))
        second = iface.dispatch(req(topic_id="ledger"))
        assert first["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert second["verification"]["manifest"]["metrics"]["ledger_hit"] is True
        assert second["verification"]["manifest"]["metrics"]["model_turn_count"] == 0
        assert len(runtime.calls) == 1 and len(iface.reads) == 2
        iface.revision = 2
        changed = iface.dispatch(req(topic_id="ledger"))
        assert changed["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        isolated = iface.dispatch(req(topic_id="ledger", topic="checkpoint unrelated"))
        assert isolated["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert len(runtime.calls) == 3

        # Ordering is canonicalized before selection, semantic identity, and
        # ledger lookup; adding an otherwise unused admitted file re-keys it.
        iface, runtime = make(root, [plan, run], outgoing={RUN: [internal]})
        ordered = iface.dispatch(req(topic_id="files", files=["Extra.cpp", "Checkpoint.cpp"]))
        reordered = iface.dispatch(req(topic_id="files", files=["Checkpoint.cpp", "Extra.cpp"]))
        changed_files = iface.dispatch(req(topic_id="files", files=["Checkpoint.cpp"]))
        assert ordered["selection_manifest"]["files"] == [f"{ROOT}/Checkpoint.cpp", f"{ROOT}/Extra.cpp"]
        assert reordered["verification"]["manifest"]["metrics"]["ledger_hit"] is True
        assert changed_files["verification"]["manifest"]["metrics"]["ledger_hit"] is False
        assert len(runtime.calls) == 2

        # Every direct edge is admitted only when its callsite, source extent,
        # and destination extent are all in the explicit set.
        other = FunctionBoundary(f"{ROOT}/Other.cpp", "Other::work", 1, 30)
        source_outside = CallSite(run.file, 20, "Other::work", RUN)
        destination_outside = CallSite(run.file, 21, RUN, "Other::work")
        callsite_outside = CallSite(other.file, 10, RUN, PLAN)
        iface, runtime = make(root, [plan, run, other], incoming={RUN: [source_outside]}, outgoing={RUN: [destination_outside, callsite_outside]})
        result = iface.dispatch(req(topic_id="external-edges"))
        accounting = result["selection_manifest"]["relationship_accounting"]
        assert result["selection_manifest"]["relationships"] == []
        assert accounting["external_relationship_count"] == 3
        assert accounting["unattributed_relationship_count"] == 0
        assert not any(name == "search" for name, _ in iface.index.calls)
    print("test_codex_subsystem_investigation: PASS")


if __name__ == "__main__": main()
