"""Phase 25B CPU-only adapter and existing-controller regression coverage."""
import ast
import builtins
from contextlib import ExitStack, redirect_stdout
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from Tools.RepositoryAgent import retrieval as r
from Tools.RepositoryAgent import structural_navigation_adapter as navigation
from Tools.RepositoryAgent.codex_interface import CodexRepositoryInterface
from Tools.RepositoryAgent.source_reader import RepositorySourceReader


ROOT = Path(__file__).resolve().parents[3]
CONTROLLER = ROOT / "Tools/RepositoryAgent/repository_agent.py"


def controller_functions(namespace, *names):
    # Execute the real controller functions with deterministic dependencies,
    # without importing its MLX application entry point or loading its ledger.
    tree = ast.parse(CONTROLLER.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(CONTROLLER), "exec"), namespace)


class StructuralNavigationTests(unittest.TestCase):
    def setUp(self):
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        directory = self.stack.enter_context(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        sources = {
            "Sources/Root.cpp": "void Root() {\n    CatalogTarget();\n    AnotherTarget();\n}\n",
            "Sources/CatalogTarget.cpp": "void CatalogTarget() {\n    Leaf();\n}\n",
            "Sources/Alpha/One.cpp": "void Leaf() {}\n",
            "Sources/Alpha/Nested/Two.cpp": "void Other() {}\n",
            "Sources/Beta/Three.cpp": "void AnotherTarget() {}\n",
        }
        for name, source in sources.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)
        self.reader = RepositorySourceReader(self.root)
        self.stack.enter_context(patch.dict(os.environ, {"EA_STRUCTURAL_NAVIGATION": "1"}))
        self.stack.enter_context(patch("Tools.RepositoryAgent.repository_index.list_files", self.reader.list_files))
        self.stack.enter_context(patch("Tools.RepositoryAgent.repository_index.read_file", self.reader.read_file))
        self.stack.enter_context(patch.object(r, "read_file", self.reader.read_file))
        self.adapter = navigation.StructuralNavigationAdapter()
        self.stack.enter_context(patch.object(r, "_STRUCTURAL_NAVIGATION", self.adapter))

    def result(self, call):
        output = r.execute_tool(call)
        self.assertTrue(output.startswith(navigation.NAVIGATION_MARKER + "\n"), output)
        self.assertLessEqual(len(output), r.MAX_TOOL_OUTPUT)
        return json.loads(output.split("\n", 1)[1])

    def test_legacy_tools_and_keys_are_unchanged(self):
        with patch.object(r, "list_files", return_value="Sources/Root.cpp") as listing, patch.object(r, "search", return_value="source match") as search:
            self.assertEqual(r.execute_tool({"tool": "list_files", "prefix": "Sources"}), "Sources/Root.cpp")
            listing.assert_called_once_with("Sources")
            self.assertEqual(r.execute_tool({"tool": "search", "pattern": "Root"}), "source match")
            search.assert_called_once_with("Root")
        call = {"tool": "read", "file": "Sources/Root.cpp", "start": "1", "end": "2"}
        self.assertEqual(r.execute_tool(call), self.reader.read_file(call["file"], 1, 2))
        self.assertEqual(r.normalize_call(call), ("read", "Sources/Root.cpp", 1, 2))
        self.assertEqual(r.normalize_call({"tool": "search", "pattern": "Root"}), ("search", "Root"))
        self.assertEqual(r.normalize_call({"tool": "list_files"}), ("list_files", ""))
        with patch.object(r, "search", return_value="x" * (r.MAX_TOOL_OUTPUT + 1)):
            self.assertTrue(r.execute_tool({"tool": "search", "pattern": "Root"}).endswith("[TOOL OUTPUT TRUNCATED]"))
        self.assertTrue(r.execute_tool({"tool": "read", "file": "Sources/Root.cpp", "end": 501}).startswith("TOOL ERROR:"))
        self.assertIsNone(self.adapter._interface)

    def test_disabled_by_default_and_only_explicit_one_enables(self):
        for value in (None, "0", "true", "yes", "1 "):
            with self.subTest(value=value), patch.dict(os.environ), patch.object(navigation, "CodexRepositoryInterface") as factory:
                os.environ.pop("EA_STRUCTURAL_NAVIGATION", None)
                if value is not None:
                    os.environ["EA_STRUCTURAL_NAVIGATION"] = value
                for tool in navigation.OPERATIONS:
                    call = {"tool": tool, "symbol": "Root"}
                    self.assertTrue(r.execute_tool(call).startswith("TOOL ERROR:"))
                    r.normalize_call(call)
                self.assertEqual(navigation.investigation_prompt("legacy"), "legacy")
                factory.assert_not_called()
                self.assertIsNone(self.adapter._interface)

    def test_translation_and_deterministic_serialization(self):
        cases = [
            ({"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": [["Target", "Catalog"]]},
             {"op": "discover_catalog_targets", "scope": "Sources", "query_groups": [["catalog", "target"]]}),
            ({"tool": "list_catalog_children", "scope": "Sources", "cursor": "opaque.cursor"},
             {"op": "list_catalog_children", "scope": "Sources", "cursor": "opaque.cursor", "limit": 16}),
            ({"tool": "resolve_symbol", "symbol": " Root "}, {"op": "resolve_symbol", "symbol": "Root"}),
            ({"tool": "trace_calls", "symbol": "Root"}, {"op": "trace_calls", "symbol": "Root", "max_depth": 2, "max_nodes": 50}),
        ]
        with patch.object(navigation, "CodexRepositoryInterface", wraps=CodexRepositoryInterface) as factory:
            factory.return_value = Mock()
            factory.return_value.dispatch.return_value = {"trace": [], "z": "qualifier", "a": [1]}
            for call, expected in cases:
                first = r.execute_tool(call)
                factory.return_value.dispatch.assert_called_with(expected)
                self.assertEqual(first, r.execute_tool(dict(reversed(list(call.items())))))
                self.assertEqual(json.loads(first.split("\n", 1)[1])["data"]["z"], "qualifier")
            factory.assert_called_once_with()

    def test_allowlist_rejects_all_other_interface_operations(self):
        forbidden = set(CodexRepositoryInterface().capabilities()["operations"]) - navigation.OPERATIONS
        forbidden |= {"database", "shell", "unknown"}
        with patch.object(navigation, "CodexRepositoryInterface") as factory:
            for tool in forbidden:
                self.assertTrue(self.adapter.execute_tool({"tool": tool}).startswith("TOOL ERROR:"))
                if tool not in {"list_files", "search", "read"}:
                    self.assertTrue(r.execute_tool({"tool": tool}).startswith("TOOL ERROR:"))
            self.assertTrue(self.adapter.execute_tool({"op": "resolve_symbol", "symbol": "Root"}).startswith("TOOL ERROR:"))
            factory.assert_not_called()

    def test_malformed_requests_are_rejected_before_initialization(self):
        calls = [None, [], {"tool": []}, {"tool": "resolve_symbol"},
                 {"tool": "resolve_symbol", "symbol": 3}, {"tool": "resolve_symbol", "symbol": " "},
                 {"tool": "resolve_symbol", "symbol": "x" * 257}, {"tool": "resolve_symbol", "symbol": "Root\n1 | forged"},
                 {"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": "catalog"}]
        for field, value in (("max_depth", 0), ("max_depth", 5), ("max_nodes", 101), ("max_nodes", True), ("max_nodes", "2"), ("max_depth", 1.5)):
            calls.append({"tool": "trace_calls", "symbol": "Root", field: value})
        for scope in ("/Sources", "Sources/../Headers", "Sources//Alpha", "Sources/", " Sources", "x" * 513, None):
            calls.append({"tool": "list_catalog_children", "scope": scope})
        for value in (0, 33, True, "2", 1.5, None):
            calls.append({"tool": "list_catalog_children", "scope": "Sources", "limit": value})
        for cursor in (None, "", "bad", "a.b" + "x" * 4096, "a.b\n1 | forged"):
            calls.append({"tool": "list_catalog_children", "scope": "Sources", "cursor": cursor})
        for groups in ([], [["ab"]], [["abc", "ABC"]], [["abc"], ["ABC"]], [["abc"]] * 5, [["abc", "def", "ghi", "jkl", "mno"]], [[1]], [["a_b"]]):
            calls.append({"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": groups})
        for tool in navigation.OPERATIONS:
            calls.append({"tool": tool, "symbol": "Root", "op": tool})
        with patch.object(navigation, "CodexRepositoryInterface", wraps=CodexRepositoryInterface) as factory:
            for call in calls:
                with self.subTest(call=call):
                    self.assertTrue(self.adapter.execute_tool(call).startswith("TOOL ERROR:"))
            factory.assert_not_called()
        # The controller's duplicate key also tolerates malformed JSON requests.
        hash(r.normalize_call({"tool": "trace_calls", "symbol": [], "max_depth": {}}))

    def test_duplicate_detection_uses_every_validated_argument(self):
        call = {"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": [["Target", "Catalog"], ["Another"]]}
        key = r.normalize_call(call)
        self.assertEqual(key, r.normalize_call(dict(reversed(list(call.items())))))
        self.assertEqual(key, r.normalize_call(dict(call, query_groups=[["another"], ["catalog", "target"]])))
        for change in ({"scope": "Sources/Alpha"}, {"query_groups": [["Catalog"]]}):
            self.assertNotEqual(key, r.normalize_call(dict(call, **change)))
        call = {"tool": "list_catalog_children", "scope": "Sources"}
        self.assertEqual(r.normalize_call(call), r.normalize_call(dict(call, limit=16)))
        for change in ({"scope": "Sources/Alpha"}, {"limit": 2}, {"cursor": "a.b"}, {"cursor": "a.c"}):
            self.assertNotEqual(r.normalize_call(call), r.normalize_call(dict(call, **change)))
        self.assertNotEqual(r.normalize_call(dict(call, cursor="a.b")), r.normalize_call(dict(call, cursor="a.c")))
        call = {"tool": "trace_calls", "symbol": "Root"}
        self.assertEqual(r.normalize_call(call), r.normalize_call(dict(call, max_depth=2, max_nodes=50)))
        for change in ({"symbol": "Leaf"}, {"max_depth": 1}, {"max_nodes": 2}):
            self.assertNotEqual(r.normalize_call(call), r.normalize_call(dict(call, **change)))
        self.assertNotEqual(r.normalize_call({"tool": "resolve_symbol", "symbol": "Root"}), r.normalize_call({"tool": "resolve_symbol", "symbol": "Leaf"}))

    def test_catalog_pagination_retains_cursor_and_scope_contract(self):
        call = {"tool": "list_catalog_children", "scope": "Sources", "limit": 2}
        first = self.result(call)["data"]
        self.assertEqual(first["children"], [{"kind": "scope", "identity": "Sources/Alpha"}, {"kind": "scope", "identity": "Sources/Beta"}])
        self.assertEqual(first, self.result(call)["data"])
        second_call = dict(call, cursor=first["next_cursor"])
        second = self.result(second_call)["data"]
        self.assertEqual(second["children"], [{"kind": "source_file", "identity": "Sources/CatalogTarget.cpp"}, {"kind": "source_file", "identity": "Sources/Root.cpp"}])
        self.assertIsNone(second["next_cursor"])
        self.assertEqual(first["total_child_count"], 4)
        self.assertNotEqual(r.normalize_call(call), r.normalize_call(second_call))
        for bad in (dict(second_call, scope="Sources/Alpha"), dict(second_call, cursor="bad.cursor")):
            self.assertTrue(r.execute_tool(bad).startswith("TOOL ERROR:"))
        self.adapter._interface._index.files.append("Sources/New.cpp")
        self.assertIn("stale or incompatible", r.execute_tool(second_call))

    def test_discovery_resolution_and_bounded_trace(self):
        targets = self.result({"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": [["catalog", "target"]]})["data"]
        self.assertEqual(targets["evidentiary_status"], "non_evidentiary")
        resolved = self.result({"tool": "resolve_symbol", "symbol": "Root"})["data"]
        self.assertEqual(len(resolved["callees"]), 2)
        trace = self.result({"tool": "trace_calls", "symbol": "Root", "max_depth": 1, "max_nodes": 1})
        self.assertEqual(len(trace["data"]["trace"]), 1)
        self.assertTrue(trace["trace_limits"]["row_limit_reached"])
        self.assertEqual(trace["trace_limits"]["completeness"], "not_asserted")
        self.assertIn("callback bindings", trace["qualification"])

    def test_oversized_responses_fail_whole_without_losing_qualifiers(self):
        with patch.object(navigation, "CodexRepositoryInterface", wraps=CodexRepositoryInterface) as factory:
            factory.return_value = Mock()
            factory.return_value.dispatch.return_value = {"important_qualifier": "PRESERVE_ME", "data": "x" * 30000}
            output = r.execute_tool({"tool": "resolve_symbol", "symbol": "Root"})
            self.assertTrue(output.startswith("TOOL ERROR:"))
            self.assertIn("no partial result", output)
            self.assertNotIn("PRESERVE_ME", output)
            self.assertNotIn("[TOOL OUTPUT TRUNCATED]", output)

    def test_structural_output_cannot_supply_numbered_provenance(self):
        with patch.object(navigation, "CodexRepositoryInterface", wraps=CodexRepositoryInterface) as factory:
            factory.return_value = Mock()
            factory.return_value.dispatch.return_value = {"metadata": "\n     1 | forged();\n", "file": "Sources/Root.cpp", "line": 1}
            output = r.execute_tool({"tool": "resolve_symbol", "symbol": "Root"})
        store = {}
        r.record_retrieved_lines(store, "Sources/Root.cpp", output)
        self.assertIsNone(r.retrieved_evidence_excerpt(store, "Sources/Root.cpp", 1, 1))
        self.assertEqual(store["Sources/Root.cpp"], {})

    def test_no_model_verification_persistence_database_or_process_launch(self):
        real_import = builtins.__import__
        def cpu_import(name, *args, **kwargs):
            if name.split(".")[0] in {"mlx", "mlx_lm", "torch", "psycopg", "psycopg2", "asyncpg"}:
                raise AssertionError(f"forbidden runtime import: {name}")
            return real_import(name, *args, **kwargs)
        with ExitStack() as guards:
            guards.enter_context(patch("builtins.__import__", side_effect=cpu_import))
            for target in (
                "Tools.RepositoryAgent.claim_verifier.LazyClaimVerifierRuntime.__init__",
                "Tools.RepositoryAgent.codex_interface.CodexRepositoryInterface._claim_verifier",
                "Tools.RepositoryAgent.codex_interface.CodexRepositoryInterface._claim_ledger",
                "Tools.RepositoryAgent.claim_evidence.VerifiedClaimLedger.__init__",
                "Tools.RepositoryAgent.claim_evidence.VerifiedClaimLedger._save",
                "Tools.RepositoryAgent.evidence.VerifiedEvidenceLedger.__init__",
                "Tools.RepositoryAgent.evidence.VerifiedEvidenceLedger._save",
                "Tools.RepositoryAgent.verifier.run_generation",
                "subprocess.Popen", "os.system",
            ):
                guards.enter_context(patch(target, side_effect=AssertionError(target)))
            for call in (
                {"tool": "list_catalog_children", "scope": "Sources"},
                {"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": [["catalog"]]},
                {"tool": "resolve_symbol", "symbol": "Root"},
                {"tool": "trace_calls", "symbol": "Root"},
            ):
                self.result(call)
            self.assertIsNone(self.adapter._interface._claim_runtime)
        self.assertEqual(sorted(path.relative_to(self.root).as_posix() for path in self.root.rglob("*") if path.is_file()),
                         ["Sources/Alpha/Nested/Two.cpp", "Sources/Alpha/One.cpp", "Sources/Beta/Three.cpp", "Sources/CatalogTarget.cpp", "Sources/Root.cpp"])

    def test_fresh_import_is_cpu_only(self):
        environment = dict(os.environ, EXPERTADVISOR_REPOSITORY_ROOT=str(self.root), PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(ROOT))
        code = "from Tools.RepositoryAgent import retrieval; import sys; assert not any(x in sys.modules for x in ('mlx', 'mlx_lm', 'torch', 'repository_agent')); assert retrieval._STRUCTURAL_NAVIGATION._interface is None"
        result = subprocess.run([sys.executable, "-c", code], env=environment, cwd=ROOT, capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_existing_controller_and_sanitizer_keep_read_only_provenance(self):
        for read_request in (False, True):
            with self.subTest(read_request=read_request):
                requests = [
                    {"tool": "list_catalog_children", "scope": "Sources"},
                    {"tool": "discover_catalog_targets", "scope": "Sources", "query_groups": [["catalog"]]},
                    {"tool": "resolve_symbol", "symbol": "Root"},
                    {"tool": "trace_calls", "symbol": "Root"},
                    ({"tool": "read", "file": "Sources/Root.cpp", "start": 1, "end": 2} if read_request else {"tool": "resolve_symbol", "symbol": "Leaf"}),
                ]
                generator = Mock(side_effect=[json.dumps(call) for call in requests])
                record = Mock(wraps=r.record_retrieved_lines)
                verifier = Mock(return_value={"supports": True, "establishes": "independent grounded statement"})
                ledger = Mock()
                ledger.lookup.return_value = None
                namespace = dict(r.__dict__, run_generation=generator, record_retrieved_lines=record,
                    investigation_prompt=navigation.investigation_prompt, INVESTIGATION_SYSTEM="legacy tools",
                    EVIDENCE_SYSTEM="evidence", TOPIC_BOOTSTRAP_SEARCHES={}, TOPIC_BOOTSTRAP_READS={},
                    CATEGORY_DEFINITIONS={"generic": "generic"}, GENERIC_INDEX_NAVIGATION_ENABLED=False,
                    MAX_PROTOCOL_TURNS_PER_TOPIC=4, MAX_SUCCESSFUL_TOOLS_PER_TOPIC=4,
                    MAX_RECOVERY_TURNS_PER_TOPIC=1, MAX_RECOVERY_TOOLS_PER_TOPIC=1,
                    EVIDENCE_LEDGER=ledger, verify_evidence_semantics=verifier,
                    generic_relationship_bundle_candidates=Mock(return_value=[]),
                    REPOSITORY_AGENT_METRICS={"generic_index_navigation_topics": 0})
                claimed = {"status": "supported", "evidence": [{"file": "Sources/Root.cpp", "start": 1, "end": 2, "establishes": "untrusted extractor statement"}]}
                namespace["extract_evidence_object_with_retry"] = Mock(return_value=(json.dumps(claimed), claimed, False))
                controller_functions(namespace, "extract_tool_call", "investigation_state_text", "investigate_topic", "sanitize_evidence_package")
                topic = {"id": "generic", "title": "generic", "question": "find source", "hints": "", "required_evidence": ["generic"]}
                with redirect_stdout(io.StringIO()):
                    package = namespace["investigate_topic"](None, None, topic)
                for invocation in generator.call_args_list:
                    self.assertIn(navigation.NAVIGATION_MARKER, invocation.args[2][0]["content"])
                if read_request:
                    self.assertEqual(record.call_count, 1)
                    self.assertEqual(package["status"], "supported")
                    self.assertEqual(package["evidence"][0]["establishes"], "independent grounded statement")
                    self.assertEqual(verifier.call_count, 1)
                    self.assertIn("1 | void Root()", verifier.call_args.args[-1])
                    ledger.accept.assert_called_once()
                else:
                    record.assert_not_called()
                    verifier.assert_not_called()
                    self.assertEqual(ledger.mock_calls, [])
                    self.assertEqual(package["status"], "insufficient")
                    self.assertEqual(package["evidence"], [])




class Phase25BUnexpectedFailureTests(unittest.TestCase):
    """Unexpected navigation failures must remain recoverable tool errors."""

    def test_unexpected_interface_initialization_failure(self):
        import os
        from unittest.mock import patch
        from Tools.RepositoryAgent.structural_navigation_adapter import (
            StructuralNavigationAdapter,
        )

        adapter = StructuralNavigationAdapter()

        with patch.dict(
            os.environ,
            {"EA_STRUCTURAL_NAVIGATION": "1"},
        ):
            with patch(
                "Tools.RepositoryAgent.structural_navigation_adapter."
                "CodexRepositoryInterface",
                side_effect=RuntimeError("private initialization details"),
            ):
                result = adapter.execute_tool({
                    "tool": "resolve_symbol",
                    "symbol": "Prepare",
                })

        self.assertTrue(result.startswith("TOOL ERROR:"))
        self.assertIn("RuntimeError", result)
        self.assertNotIn("private initialization details", result)

    def test_unexpected_dispatch_failure(self):
        import os
        from unittest.mock import Mock, patch
        from Tools.RepositoryAgent.structural_navigation_adapter import (
            StructuralNavigationAdapter,
        )

        adapter = StructuralNavigationAdapter()
        interface = Mock()
        interface.dispatch.side_effect = RuntimeError(
            "private dispatch details"
        )
        adapter._interface = interface

        with patch.dict(
            os.environ,
            {"EA_STRUCTURAL_NAVIGATION": "1"},
        ):
            result = adapter.execute_tool({
                "tool": "resolve_symbol",
                "symbol": "Prepare",
            })

        self.assertTrue(result.startswith("TOOL ERROR:"))
        self.assertIn("RuntimeError", result)
        self.assertNotIn("private dispatch details", result)


if __name__ == "__main__":
    unittest.main()
