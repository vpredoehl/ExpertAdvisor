#!/usr/bin/env python3
"""Source-proof and ledger-isolation regressions for positional bindings."""
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


HEADER = """struct CheckpointAnalysisOperations
{
    std::function<void()> claim;
    std::function<void()> afterClaim;
    std::function<void()> execute;
    std::function<void()> afterWork;
    std::function<void()> finalize;
    std::function<void()> generateReports;
};
"""
SOURCE = """int ClaimCheckpointAnalysis();
int ExecuteCheckpointAnalysisWork();
int FinalizeCheckpointAnalysis();
int RunCheckpointEvalAnalyzeJobs()
{
    CheckpointAnalysisOperations operations{
        [&] { return ClaimCheckpointAnalysis(); },
        [&] { return AfterClaim(); },
        [&] { return ExecuteCheckpointAnalysisWork(); },
        [&] { return AfterWork(); },
        [&] { return FinalizeCheckpointAnalysis(); },
        [&] { return GenerateReports(); }};
    return 0;
}
"""
H = "Sources/SchedulerCore/CheckpointAnalysisOrchestrationService.hpp"
C = "Sources/SchedulerCore/ProductionSchedulerDaemon.cpp"


class Runtime:
    def __init__(self): self.calls = []
    def verify_claim(self, *args): return {"supports": True, "establishes": "named", "reason": "test", "model_turns": 1}
    def verify_positional_operation_binding_claim(self, topic, claim, relationship):
        self.calls.append(relationship)
        assert relationship["relationship_kind"] == "operation_binding"
        assert relationship["aggregate_type"] == "CheckpointAnalysisOperations"
        assert relationship["aggregate_fields"] == ["claim", "afterClaim", "execute", "afterWork", "finalize", "generateReports"]
        assert len(relationship["evidence"]) == 2
        assert "struct CheckpointAnalysisOperations" in relationship["evidence"][0]["excerpt"]
        assert "CheckpointAnalysisOperations operations{" in relationship["evidence"][1]["excerpt"]
        return {"supports": True, "establishes": relationship["operation"], "reason": "test", "model_turns": 1}


class Interface(CodexRepositoryInterface):
    def __init__(self, index, files, runtime, ledger):
        super().__init__(claim_runtime=runtime, claim_ledger_path=str(ledger)); self.index=index; self.files=files; self.reads=[]
    def _idx(self): return self.index
    def _claim_item(self, spec):
        self.reads.append((spec["file"], spec["start"], spec["end"]))
        lines=self.files[spec["file"]].splitlines()
        excerpt="\n".join(f"{n:6d} | {line}" for n,line in enumerate(lines[int(spec["start"])-1:int(spec["end"])], int(spec["start"])))
        return {**spec,"excerpt":excerpt}


def build(files):
    old_list, old_read = ri.list_files, ri.read_file
    ri.list_files=lambda prefix="": list(files)
    ri.read_file=lambda name,start=1,end=200: "\n".join(f"{n}: {line}" for n,line in enumerate(files[name].splitlines()[start-1:end],start))
    try: return ri.RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))
    finally: ri.list_files,ri.read_file=old_list,old_read


def request(callee):
    return {"op":"investigate_operation_relationship_claim","topic_id":"positional","topic":"operations","claim":"bound","caller":"RunCheckpointEvalAnalyzeJobs","callee":callee,"relationship_kind":"operation_binding"}


def main():
  with tempfile.TemporaryDirectory() as td:
    files={H:HEADER,C:SOURCE}; idx=build(files); runtime=Runtime(); iface=Interface(idx,files,runtime,Path(td)/"claims.json")
    # Every production-shaped slot uses declaration + complete initializer, not
    # the old call-centred radius; field identity differs per binding.
    for callee, field in (("ClaimCheckpointAnalysis","operations.claim"),("ExecuteCheckpointAnalysisWork","operations.execute"),("FinalizeCheckpointAnalysis","operations.finalize")):
      result=iface.dispatch(request(callee)); m=result["selection_manifest"]
      assert m["route"] == "positional_operation_binding_bundle"
      assert m["selected_ranges"] == [{"file":H,"start":1,"end":9},{"file":C,"start":6,"end":12}]
      assert result["verification"]["manifest"]["relationship"]["operation"] == field
    assert len(runtime.calls)==3 and len(iface.reads)==6
    # An accepted claim slot cannot alias execute: a fresh ledger must verify
    # both identities independently, while an identical claim is then cached.
    isolated_runtime=Runtime(); isolated=Interface(idx,files,isolated_runtime,Path(td)/"isolated.json")
    isolated.dispatch(request("ClaimCheckpointAnalysis")); assert len(isolated_runtime.calls)==1
    isolated.dispatch(request("ExecuteCheckpointAnalysisWork")); assert len(isolated_runtime.calls)==2
    isolated.dispatch(request("ClaimCheckpointAnalysis")); assert len(isolated_runtime.calls)==2
    # Source changing after structural selection fails closed before runtime.
    changed=Interface(idx,{H:HEADER,C:SOURCE.replace("return ExecuteCheckpointAnalysisWork", "return DifferentWork")},runtime,Path(td)/"changed.json")
    out=changed.dispatch(request("ExecuteCheckpointAnalysisWork")); assert out["verification"]["verdict"]["supports"] is False and len(runtime.calls)==3
    # Schema/order, initializer completeness/count, and mismatched operation
    # all suppress positional proof rather than accepting a local lambda.
    for bad_header,bad_source in ((HEADER.replace("claim;\n    std::function<void()> afterClaim", "afterClaim;\n    std::function<void()> claim"),SOURCE), (HEADER,SOURCE.replace("        [&] { return GenerateReports(); }};", "    };")), (HEADER,SOURCE.replace("return ExecuteCheckpointAnalysisWork", "return DifferentWork"))):
      bad=build({H:bad_header,C:bad_source})
      assert bad.positional_operation_binding_proof("RunCheckpointEvalAnalyzeJobs", "ExecuteCheckpointAnalysisWork") is None
  print("test_positional_operation_binding_evidence: PASS")

if __name__ == "__main__": main()
