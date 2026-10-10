"""Bounded callback discovery, targeted evidence and restricted MCP regressions."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from Tools.RepositoryAgent import repository_index as ri
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.repository_agent_mcp import StdioMCPServer


FILE = "Sources/SchedulerCore/Callback.cpp"
KIND = "operation_implementation_call"
SOURCE = """\
struct DispatchOperations
{
    Callback prepareReservedLaunch;
    Callback reserveWorkerAttempt;
};
class Dispatch
{
    DispatchOperations operations_;
};
void Train()
{
}
void Infer()
{
}
void Leaf()
{
}
void Owner()
{
    DispatchOperations operations;
    operations.prepareReservedLaunch = [&](int phase) {
        if (phase == 0) { Train(); }
        else { Infer(); }
    };
    operations.reserveWorkerAttempt = [&] { return Leaf(); };
}
void Dispatch::runPhase()
{
    operations_.prepareReservedLaunch(0);
    operations_.reserveWorkerAttempt();
}
"""


def build(source=SOURCE, extras=None):
    sources = {FILE: source, **(extras or {})}
    with patch.object(ri, "list_files", return_value=list(sources)), patch.object(
        ri, "read_file", side_effect=lambda file, start=1, end=200: "\n".join(
            f"{n}: {line}" for n, line in enumerate(sources[file].splitlines()[start - 1:end], start))
    ):
        return ri.RepositoryIndex().build(prefixes=("Sources/SchedulerCore",))


class Runtime:
    def __init__(self):
        self.calls = []
        self.supports = True

    def verify_claim(self, topic, claim, item):
        self.calls.append((claim, item))
        return {"supports": self.supports, "establishes": "conditional callback call" if self.supports else "",
                "reason": "fixture verifier", "model_turns": 1}


class Interface(CodexRepositoryInterface):
    def __init__(self, index, ledger=None):
        self.runtime = Runtime()
        super().__init__(claim_runtime=self.runtime, claim_ledger_path=ledger)
        self.index = index
        self.reads = []
        self.changed = False

    def _idx(self):
        return self.index

    def _claim_item(self, spec):
        self.reads.append(spec)
        lines = self.index._lines[spec["file"]][spec["start"] - 1:spec["end"]]
        excerpt = "\n".join(lines) + ("\nchanged" if self.changed else "")
        return {**spec, "excerpt": excerpt}


def discovery(**extra):
    return {"op": "discover_operation_relationship_paths", "scope": "SchedulerCore",
            "from": "Owner", "to": "Train", "operation": "operations.prepareReservedLaunch",
            "relationship_kind": KIND, "max_hops": 1, **extra}


def investigation(**extra):
    return {"op": "investigate_operation_relationship_claim", "caller": "Owner", "callee": "Train",
            "operation": "operations.prepareReservedLaunch", "relationship_kind": KIND,
            "topic_id": "callback", "topic": "callback evidence", "claim": "conditional Train call", **extra}


class BoundedOperationDiscoveryTests(unittest.TestCase):
    def test_explicit_implementation_discovery_is_metadata_only_and_deterministic(self):
        iface = Interface(build())
        first = iface.dispatch(discovery())
        self.assertEqual(first, iface.dispatch(discovery()))
        self.assertEqual(first["paths"], [["Owner", "Train"]])
        self.assertEqual(first["relationships"][0]["relationship_kind"], KIND)
        self.assertEqual(first["relationships"][0]["operation"], "operations.prepareReservedLaunch")
        self.assertEqual(first["evidentiary_status"], "non_evidentiary")
        self.assertEqual(iface.reads, [])
        self.assertEqual(iface.runtime.calls, [])
        self.assertNotIn("source", first)
        self.assertNotIn("line", first["relationships"][0])

    def test_no_direct_promotion_or_transitive_implementation_navigation(self):
        iface = Interface(build())
        self.assertEqual(iface.index.relationship("Owner", "Train"), [])
        self.assertEqual(iface.dispatch({"op": "discover_relationship_paths", "scope": "SchedulerCore",
                                        "from": "Owner", "to": "Train", "max_hops": 4})["paths"], [])
        self.assertEqual(iface.dispatch(discovery(to="Leaf", max_hops=4))["paths"], [])
        self.assertEqual(iface.index.trace_calls("Owner", max_depth=1)[0]["callee"], "Leaf")

    def test_typed_context_bridge_and_hop_boundary(self):
        iface = Interface(build())
        request = discovery(**{"from": "Dispatch::runPhase", "binding_owner": "Owner",
                               "relationship_kind": "operation_invocation", "max_hops": 3})
        result = iface.dispatch(request)
        self.assertEqual(result["paths"], [["Dispatch::runPhase", "operations_.prepareReservedLaunch",
                                            "Owner::operations.prepareReservedLaunch", "Train"]])
        self.assertEqual([row["relationship_kind"] for row in result["relationships"]],
                         ["operation_invocation", "operation_binding", KIND])
        self.assertIn("runtime callback wiring", result["required_follow_up"])
        self.assertEqual(result["investigation_targets"], [
            {"caller": "Dispatch::runPhase", "callee": "operations_.prepareReservedLaunch", "relationship_kind": "operation_invocation"},
            {"caller": "Owner", "callee": "Train", "operation": "operations.prepareReservedLaunch", "relationship_kind": KIND},
        ])
        self.assertEqual(iface.dispatch({**request, "max_hops": 2})["paths"], [])
        self.assertEqual(iface.dispatch({**request, "max_hops": 4})["paths"], result["paths"])

    def test_same_field_name_is_insufficient_for_a_bridge(self):
        variants = [
            SOURCE.replace("DispatchOperations operations_;", "OtherOperations operations_;") +
            "\nstruct OtherOperations\n{\nCallback prepareReservedLaunch;\n};\n",
            SOURCE.replace("DispatchOperations operations;", "auto operations = make();"),
            SOURCE.replace("DispatchOperations operations;", "DispatchOperations operations;\nDispatchOperations operations;"),
            SOURCE + "\nstruct DispatchOperations\n{\nCallback prepareReservedLaunch;\n};\n",
            SOURCE.replace("    Callback prepareReservedLaunch;", "    Callback unknown;"),
            SOURCE.replace("operations_.prepareReservedLaunch(0);", "unknown_.prepareReservedLaunch(0);"),
            SOURCE.replace("DispatchOperations operations;", "Wrong::DispatchOperations operations;"),
            SOURCE.replace("DispatchOperations operations;", "if (false) { DispatchOperations operations; }"),
            SOURCE.replace("    DispatchOperations operations_;", "    struct Nested { DispatchOperations operations_; };"),
        ]
        for source in variants:
            with self.subTest(source=source):
                result = Interface(build(source)).dispatch(discovery(**{
                    "from": "Dispatch::runPhase", "binding_owner": "Owner",
                    "relationship_kind": "operation_invocation", "max_hops": 3}))
                self.assertEqual(result["paths"], [])

    def test_wrong_field_unassigned_nested_or_repeated_assignment_is_not_invented(self):
        iface = Interface(build())
        self.assertEqual(iface.dispatch(discovery(operation="operations.unknown"))["paths"], [])
        for source in (
            SOURCE.replace("operations.prepareReservedLaunch =", "auto callback ="),
            SOURCE.replace("if (phase == 0) { Train(); }", "auto nested = [&] { Train(); };"),
            SOURCE.replace("operations.reserveWorkerAttempt =", "operations.prepareReservedLaunch ="),
        ):
            with self.subTest(source=source):
                self.assertEqual(Interface(build(source)).dispatch(discovery())["paths"], [])

    def test_scope_ambiguity_and_examined_edge_limits(self):
        iface = Interface(build(extras={"Sources/Other/Train.cpp": "void Train()\n{\n}\n"}))
        with self.assertRaisesRegex(ValueError, "exactly one"):
            iface.dispatch(discovery())
        source = SOURCE.replace("void Train()\n{\n}\n", "")
        iface = Interface(build(source, {"Sources/SchedulerCore/Nested/Train.cpp": "void Train()\n{\n}\n"}))
        with self.assertRaisesRegex(ValueError, "outside resolved scope"):
            iface.dispatch(discovery())
        iface = Interface(build())
        iface.index.calls_by_caller["Owner"] *= 200
        with self.assertRaisesRegex(ValueError, "examined-edge cap"):
            iface.dispatch(discovery())
        iface = Interface(build())
        iface.index.calls_by_caller["Owner"] *= 60
        iface.index.calls_by_caller["Dispatch::runPhase"] *= 200
        with self.assertRaisesRegex(ValueError, "examined-edge cap"):
            iface.dispatch(discovery(**{"from": "Dispatch::runPhase", "binding_owner": "Owner",
                                       "relationship_kind": "operation_invocation", "max_hops": 3}))

    def test_ambiguous_invocation_receiver_is_rejected(self):
        source = SOURCE.replace("    DispatchOperations operations_;", "    DispatchOperations operations_;\n    DispatchOperations otherOperations;")
        source = source.replace("    operations_.prepareReservedLaunch(0);", "    operations_.prepareReservedLaunch(0);\n    otherOperations.prepareReservedLaunch(0);")
        with self.assertRaisesRegex(ValueError, "receiver is ambiguous"):
            Interface(build(source)).dispatch(discovery(**{"from": "Dispatch::runPhase", "binding_owner": "Owner",
                                                          "relationship_kind": "operation_invocation", "max_hops": 3}))

    def test_qualified_slots_and_qualified_callee_round_trip(self):
        source = "namespace Named\n{\n" + SOURCE + "\n}\n"
        source = source.replace("DispatchOperations operations;", "Named::DispatchOperations operations;")
        result = Interface(build(source)).dispatch(discovery(**{
            "from": "Dispatch::runPhase", "binding_owner": "Owner",
            "relationship_kind": "operation_invocation", "max_hops": 3}))
        self.assertEqual(result["path_count"], 1)
        source = SOURCE.replace("void Train()", "void Engine::Train()")
        with tempfile.TemporaryDirectory() as directory:
            iface = Interface(build(source), str(Path(directory) / "claims.json"))
            discovered = iface.dispatch(discovery(to="Engine::Train"))
            self.assertEqual(discovered["paths"], [["Owner", "Engine::Train"]])
            verified = iface.dispatch(investigation(callee="Engine::Train"))
            self.assertTrue(verified["verification"]["verdict"]["supports"])

    def test_strict_arguments_and_legacy_one_hop_invocations(self):
        iface = Interface(build())
        for value in (0, 5, True, "3"):
            with self.assertRaisesRegex(ValueError, "max_hops"):
                iface.dispatch(discovery(max_hops=value))
        for extra in ({"scope": "../SchedulerCore"}, {"extra": "no"}, {"relationship_kind": "direct_invocation"}):
            with self.assertRaises(ValueError):
                iface.dispatch(discovery(**extra))
        missing = discovery()
        del missing["operation"]
        with self.assertRaisesRegex(ValueError, "operation"):
            iface.dispatch(missing)
        request = {"op": "discover_operation_relationship_paths", "scope": "SchedulerCore",
                   "from": "Dispatch::runPhase", "to": "operations_.prepareReservedLaunch",
                   "relationship_kind": "operation_invocation", "max_hops": 1}
        self.assertEqual(iface.dispatch(request)["path_count"], 1)
        with self.assertRaisesRegex(ValueError, "max_hops=1"):
            iface.dispatch({**request, "max_hops": 2})

    def test_single_call_binding_behavior_is_preserved(self):
        iface = Interface(build())
        result = iface.dispatch({"op": "discover_operation_relationship_paths", "scope": "SchedulerCore",
                                 "from": "Owner", "to": "Leaf", "relationship_kind": "operation_binding", "max_hops": 1})
        self.assertEqual(result["paths"], [["Owner", "Leaf"]])

    def test_targeted_verification_above_symbol_cap_and_ledger_replay(self):
        noisy = SOURCE.replace("    DispatchOperations operations;", "    DispatchOperations operations;\n" +
                               "\n".join(f"    Noise{n}();" for n in range(20)))
        with tempfile.TemporaryDirectory() as directory:
            iface = Interface(build(noisy), str(Path(directory) / "claims.json"))
            symbol = iface.dispatch({"op": "investigate_symbol", "symbol": "Owner", "topic_id": "callback", "topic": "callback evidence"})
            self.assertEqual(symbol["selection_manifest"]["selection_status"], "too_many_direct_relationships")
            first = iface.dispatch(investigation())
            self.assertTrue(first["verification"]["verdict"]["supports"])
            self.assertIn("operations.prepareReservedLaunch =", iface.runtime.calls[0][1]["excerpt"])
            self.assertIn("else { Infer(); }", iface.runtime.calls[0][1]["excerpt"])
            replay = iface.dispatch(investigation())
            self.assertTrue(replay["verification"]["manifest"]["metrics"]["ledger_hit"])
            self.assertEqual(replay["verification"]["manifest"]["metrics"]["model_turn_count"], 0)
            self.assertEqual(len(iface.runtime.calls), 1)
            self.assertEqual(len(iface.reads), 2)
            iface.dispatch(investigation(callee="Infer"))
            self.assertEqual(len(iface.runtime.calls), 2)
            iface.changed = True
            stale = iface.dispatch(investigation())
            self.assertEqual(stale["selection_manifest"]["selection_status"], "callback_source_changed")
            self.assertEqual(stale["verification"]["repository_read_count"], 1)
            self.assertEqual(len(iface.runtime.calls), 2)

    def test_verification_rejects_wrong_slot_and_oversized_assignment(self):
        with tempfile.TemporaryDirectory() as directory:
            iface = Interface(build(), str(Path(directory) / "claims.json"))
            wrong = iface.dispatch(investigation(operation="operations.reserveWorkerAttempt"))
            self.assertEqual(wrong["selection_manifest"]["selection_status"], "callback_assignment_unavailable")
            self.assertEqual(iface.reads, [])
            oversized = SOURCE.replace("        else { Infer(); }", "\n".join("        // padding" for _ in range(501)) + "\n        else { Infer(); }")
            result = Interface(build(oversized), str(Path(directory) / "large.json")).dispatch(investigation())
            self.assertEqual(result["selection_manifest"]["selection_status"], "callback_range_limit_exceeded")
            iface.runtime.supports = False
            result = iface.dispatch(investigation(claim="unconditional direct call"))
            self.assertFalse(result["verification"]["verdict"]["supports"])
            self.assertIn("do not promote", iface.runtime.calls[0][0])

    def test_field_identity_cannot_reuse_same_range_or_generic_decision(self):
        source = SOURCE.replace("        if (phase == 0) { Train(); }\n        else { Infer(); }\n    };\n    operations.reserveWorkerAttempt = [&] { return Leaf(); };",
                                "Train(); Infer(); }; operations.reserveWorkerAttempt = [&] { Train(); Infer(); };")
        source = source.replace("operations.prepareReservedLaunch = [&](int phase) {\n", "operations.prepareReservedLaunch = [&](int phase) { ")
        with tempfile.TemporaryDirectory() as directory:
            iface = Interface(build(source), str(Path(directory) / "claims.json"))
            first = iface.dispatch(investigation())
            second = iface.dispatch(investigation(operation="operations.reserveWorkerAttempt"))
            # Both callback assignments occupy exactly the same physical range.
            self.assertEqual(first["selection_manifest"]["selected_ranges"], second["selection_manifest"]["selected_ranges"])
            self.assertEqual(len(iface.runtime.calls), 2)
            spec = first["selection_manifest"]["selected_ranges"][0]
            iface.dispatch({"op": "investigate_source_claim", "topic_id": "callback", "topic": "callback evidence",
                            "claim": "conditional Train call", **spec})
            self.assertEqual(len(iface.runtime.calls), 3)
            self.assertTrue(iface.dispatch(investigation())["verification"]["manifest"]["metrics"]["ledger_hit"])

    def test_mcp_profile_contract_and_raw_tool_rejection(self):
        server = StdioMCPServer(profile="codex_assisted")
        self.assertEqual(len(server._tools), 12)
        for name in ("read", "source_excerpt", "search", "resolve_symbol"):
            result = server._handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/call",
                                    "params": {"name": name, "arguments": {}}})
            self.assertTrue(result["result"]["isError"])
        schema = server._tools_by_name["discover_operation_relationship_paths"]["inputSchema"]
        server._validate_schema({key: value for key, value in discovery().items() if key != "op"}, schema, location="arguments")
        with self.assertRaises(ValueError):
            server._validate_schema({"excerpt": "raw"}, schema, location="arguments")
        with tempfile.TemporaryDirectory() as directory:
            server.iface = Interface(build(), str(Path(directory) / "claims.json"))
            for request in (discovery(), investigation()):
                name = request["op"]
                response = server._handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/call",
                    "params": {"name": name, "arguments": {key: value for key, value in request.items() if key != "op"}}})
                self.assertFalse(response["result"]["isError"])


if __name__ == "__main__":
    unittest.main()
