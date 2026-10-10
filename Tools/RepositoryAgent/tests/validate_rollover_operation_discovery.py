"""Read-only scheduler validation through the local codex_assisted MCP adapter.

Requires an explicit Rollover source-root override. --qwen uses the existing
offline model configuration and a disposable ledger, never a production ledger.
"""
import argparse
import json
import os
import tempfile
from pathlib import Path

from Tools.RepositoryAgent.repository_agent_mcp import StdioMCPServer
from Tools.RepositoryAgent.repository_index import RepositoryIndex
from Tools.RepositoryAgent.source_reader import resolve_repository_root


EXPECTED = (
    ("reserveWorkerAttempt", "ReserveExperimentWorkerAttempt"),
    ("prepareReservedLaunch", "BuildTrainCommand"),
    ("prepareReservedLaunch", "BuildInferCommand"),
    ("prepareReservedLaunch", "BuildAnalyzeCommand"),
    ("launchPreparedWorker", "LaunchReservedChildProcess"),
)
OWNER = "RunFinalExperimentPhase"
DISPATCH = "FinalExperimentDispatchService::runPhase"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qwen", action="store_true")
    options = parser.parse_args()
    root = resolve_repository_root()
    if root != Path("/Volumes/Developer SSD/ExpertAdvisor-Rollover") or "EXPERTADVISOR_REPOSITORY_ROOT" not in os.environ:
        raise ValueError("validation requires the explicit read-only Rollover root")
    if options.qwen and os.environ.get("HF_HUB_OFFLINE") != "1":
        raise ValueError("Qwen validation requires HF_HUB_OFFLINE=1")
    index = RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))
    stats = {"mcp_calls": 0, "verifications": 0, "ledger_hits": 0, "model_turns": 0, "repository_reads": 0}
    with tempfile.TemporaryDirectory(prefix="repository-agent-24d3-") as directory:
        server = StdioMCPServer(profile="codex_assisted")
        server.iface._index = index
        server.iface._claim_ledger_path = str(Path(directory) / "claims.json")

        def call(name, arguments):
            stats["mcp_calls"] += 1
            response = server._handle_request({"jsonrpc": "2.0", "id": stats["mcp_calls"],
                "method": "tools/call", "params": {"name": name, "arguments": arguments}})["result"]
            assert not response.get("isError"), response
            return response["structuredContent"]

        listing = server._handle_request({"jsonrpc": "2.0", "id": 0, "method": "tools/list"})["result"]["tools"]
        assert len(listing) == 12
        assert not ({"read", "search", "source_excerpt", "resolve_symbol"} & {tool["name"] for tool in listing})
        for field, target in EXPECTED:
            operation = "operations." + field
            discovery = {"scope": "Sources/SchedulerCore", "from": OWNER, "to": target,
                         "max_hops": 1, "relationship_kind": "operation_implementation_call", "operation": operation}
            result = call("discover_operation_relationship_paths", discovery)
            assert result["paths"] == [[OWNER, target]], result
            bridge = call("discover_operation_relationship_paths", {**discovery, "from": DISPATCH,
                "max_hops": 3, "relationship_kind": "operation_invocation", "binding_owner": OWNER})
            assert bridge["paths"] == [[DISPATCH, "operations_." + field, OWNER + "::" + operation, target]], bridge
            assert not index.relationship(OWNER, target)
            assert not call("discover_relationship_paths", {"scope": "Sources/SchedulerCore",
                "from": OWNER, "to": target, "max_hops": 1})["paths"]
            assert not call("discover_operation_relationship_paths", {**discovery, "operation": "operations.unknown"})["paths"]
            proof = index.operation_implementation_proof(OWNER, target, operation)
            print(json.dumps({"operation": operation, "target": target, "evidence": proof}), flush=True)
        for field in sorted({field for field, _ in EXPECTED}):
            result = call("discover_operation_relationship_paths", {"scope": "Sources/SchedulerCore", "from": DISPATCH,
                "to": "operations_." + field, "max_hops": 1, "relationship_kind": "operation_invocation"})
            assert result["paths"] == [[DISPATCH, "operations_." + field]]
        binding = call("discover_operation_relationship_paths", {"scope": "Sources/SchedulerCore", "from": "RunScheduler",
            "to": "RunSchedulerOnce", "max_hops": 1, "relationship_kind": "operation_binding"})
        assert binding["paths"] == [["RunScheduler", "RunSchedulerOnce"]]
        if options.qwen:
            requests = [{"caller": OWNER, "callee": target, "operation": "operations." + field,
                         "relationship_kind": "operation_implementation_call",
                         "claim": f"The callback assigned to operations.{field} contains a source-visible call to {target}, conditional on the callback and its applicable branch executing; this is not an unconditional direct call by RunFinalExperimentPhase."}
                        for field, target in EXPECTED]
            requests += [{"caller": DISPATCH, "callee": "operations_." + field,
                          "relationship_kind": "operation_invocation",
                          "claim": f"runPhase invokes the operation field operations_.{field} on its applicable dispatch path, rather than directly invoking a guessed callback implementation."}
                         for field in sorted({field for field, _ in EXPECTED})]
            for request in requests + requests[:5]:
                result = call("investigate_operation_relationship_claim", {
                    "topic_id": "phase24d3-validation", "topic": "Bounded scheduler callback relationship", **request})
                verification = result["verification"]
                assert verification["verdict"]["supports"] is True, result
                metrics = verification["manifest"]["metrics"]
                stats["verifications"] += 1
                stats["ledger_hits"] += int(metrics["ledger_hit"])
                stats["model_turns"] += metrics["model_turn_count"]
                stats["repository_reads"] += metrics["repository_read_count"]
                print(json.dumps({"callee": request["callee"], "metrics": metrics}), flush=True)
            assert stats["ledger_hits"] == 5, stats
        print(json.dumps({"result": "PASS", **stats}), flush=True)


if __name__ == "__main__":
    main()
