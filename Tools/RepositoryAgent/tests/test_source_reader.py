"""Development-root and read-only boundary checks; no production reads or MLX."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

# The default-mode compatibility test uses a reader sentinel, never production.
legacy = types.ModuleType("expertadvisor_agent")
legacy.list_files = lambda prefix="": "legacy listing"
legacy.read_file = lambda name, start=1, end=200: "legacy read"
legacy.search = lambda pattern, max_results=100: "legacy search"
sys.modules.setdefault("expertadvisor_agent", legacy)

from Tools.RepositoryAgent import source_reader as reader


class SourceReaderTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.base = Path(self.temporary.name).resolve()
        self.root = self.base / "repository"
        self.root.mkdir()
        for name, content in {
            "Sources/Marker.cpp": "int development_marker() { return 24; }\n",
            "Headers/Marker.hpp": "#pragma once\n",
            "LSTM/Marker.cpp": "int application_marker() { return 1; }\n",
            **{name: "development fixture\n" for name in reader.DEVELOPMENT_FILES},
            "Tests/Unrelated.sh": "private test\n",
            "Scripts/Unrelated.py": "private script\n",
            "ExpertAdvisor.xcodeproj/private.txt": "private project file\n",
            "Private/Secret.cpp": "private source\n",
        }.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(content)
        self.reader = reader.RepositorySourceReader(self.root)

    def child(self, code, **overrides):
        env = dict(os.environ)
        for name in (reader.ROOT_ENVIRONMENT_VARIABLE,
                     "EXPERTADVISOR_LEDGER_CACHE_NAMESPACE",
                     "EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER"):
            env.pop(name, None)
        env.update({"PYTHONDONTWRITEBYTECODE": "1", **overrides})
        result = subprocess.run([sys.executable, "-c", code], env=env,
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        return result

    def test_default_root_is_production_and_reader_delegates_unchanged(self):
        self.assertEqual(reader.resolve_repository_root({}),
                         Path("/Volumes/Developer SSD/ExpertAdvisor"))
        self.child("""
import sys, types
stub = types.ModuleType('expertadvisor_agent')
stub.list_files = lambda prefix='': 'default listing'
stub.read_file = lambda name, start=1, end=200: 'default read'
stub.search = lambda pattern, max_results=100: 'default search'
sys.modules['expertadvisor_agent'] = stub
from Tools.RepositoryAgent import source_reader
assert source_reader.list_files is stub.list_files
assert source_reader.read_file is stub.read_file
assert source_reader.search is stub.search
assert str(source_reader.REPOSITORY_ROOT) == '/Volumes/Developer SSD/ExpertAdvisor'
assert 'mlx_lm' not in sys.modules
""")

    def test_development_root_resolves_to_fixture(self):
        self.assertEqual(
            reader.resolve_repository_root({
                reader.ROOT_ENVIRONMENT_VARIABLE: str(self.root)
            }),
            self.root.resolve(strict=True),
        )

    def test_invalid_override_fails_instead_of_falling_back(self):
        for value in ("", ".", "relative/repository", str(self.base / "missing")):
            with self.subTest(value=value), self.assertRaises((ValueError, OSError)):
                reader.resolve_repository_root({reader.ROOT_ENVIRONMENT_VARIABLE: value})
        file = self.root / "Sources/Marker.cpp"
        with self.assertRaises(ValueError):
            reader.resolve_repository_root({reader.ROOT_ENVIRONMENT_VARIABLE: str(file)})

    def test_permitted_source_and_exact_phase24b_files(self):
        for name in ("Sources/Marker.cpp", "Headers/Marker.hpp", "LSTM/Marker.cpp",
                     *reader.DEVELOPMENT_FILES):
            with self.subTest(name=name):
                self.assertTrue(self.reader.read_file(name))
        for name in ("Tests/Unrelated.sh", "Scripts/Unrelated.py",
                     "ExpertAdvisor.xcodeproj/private.txt", "Private/Secret.cpp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.reader.read_file(name)

    def test_search_is_literal_and_case_insensitive(self):
        path = self.root / "Sources/Search.cpp"
        path.write_text("Alpha.Beta\nALPHA.BETA\nAlphaXBeta\n")

        results = self.reader.search("alpha.beta")
        self.assertEqual(len(results.splitlines()), 2)
        self.assertEqual(
            len(self.reader.search("alpha.beta", max_results=1).splitlines()),
            1,
        )

    def test_invalid_search_arguments_fail_closed(self):
        for pattern in ("", "x" * 257, None):
            with self.subTest(pattern=pattern):
                with self.assertRaises(ValueError):
                    self.reader.search(pattern)

        for limit in (0, -1, 101, "10", True):
            with self.subTest(limit=limit):
                with self.assertRaises(ValueError):
                    self.reader.search("marker", max_results=limit)

    def test_read_range_and_500_line_cap(self):
        path = self.root / "Sources/Long.cpp"
        path.write_text("\n".join(f"line {n}" for n in range(1, 602)))
        self.assertEqual(self.reader.read_file("Sources/Long.cpp", 3, 4),
                         "     3 | line 3\n     4 | line 4")
        self.assertEqual(len(self.reader.read_file("Sources/Long.cpp", 1, 999).splitlines()), 500)

    def test_path_traversal_and_absolute_paths_fail_closed(self):
        for name in ("../outside.cpp", "Sources/../LSTM/Marker.cpp",
                     "Sources/./Marker.cpp", "Sources//Marker.cpp", "Sources/Marker.cpp/",
                     "Sources\\Marker.cpp", "Sources/Marker.cpp\x00",
                     str(self.root / "Sources/Marker.cpp")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.reader.read_file(name)
        for prefix in ("../Sources", "/Sources", "Sources/../"):
            with self.subTest(prefix=prefix), self.assertRaises(ValueError):
                self.reader.list_files(prefix)

    def test_symlink_file_escape_and_disallowed_target_fail_closed(self):
        outside = self.base / "Outside.cpp"
        outside.write_text("outside secret\n")
        (self.root / "Sources/Escape.cpp").symlink_to(outside)
        (self.root / "Sources/Private.cpp").symlink_to(self.root / "Private/Secret.cpp")
        for name in ("Sources/Escape.cpp", "Sources/Private.cpp"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.reader.read_file(name)
        self.assertNotIn("Escape.cpp", self.reader.list_files())
        self.assertNotIn("Private.cpp", self.reader.list_files())
        self.assertEqual(self.reader.search("outside secret|private source"), "")

    def test_symlink_source_directory_escape_is_not_enumerated(self):
        root = self.base / "symlink_repository"
        root.mkdir()
        (root / "Sources").symlink_to(self.root / "Sources", target_is_directory=True)
        isolated = reader.RepositorySourceReader(root)
        self.assertEqual(isolated.list_files(), "")
        with self.assertRaises(ValueError):
            isolated.read_file("Sources/Marker.cpp")

    def test_symlink_to_root_is_not_enumerated(self):
        root = self.base / "recursive_repository"
        root.mkdir()
        (root / "Sources").symlink_to(root, target_is_directory=True)
        self.assertEqual(reader.RepositorySourceReader(root).list_files(), "")

    def test_reader_and_index_share_one_startup_selected_root(self):
        self.child("""
import os, sys
from pathlib import Path
from Tools.RepositoryAgent import source_reader as reader
from Tools.RepositoryAgent import repository_index as index
from Tools.RepositoryAgent import codex_interface as interface
from Tools.RepositoryAgent import retrieval
assert index.read_file is interface.read_file is retrieval.read_file is reader.read_file
assert index.list_files is interface.list_files is retrieval.list_files is reader.list_files
assert reader.read_file.__self__.root == reader.REPOSITORY_ROOT
os.environ['EXPERTADVISOR_REPOSITORY_ROOT'] = '/must/not/switch/at/runtime'
assert 'development_marker' in interface.read_file('Sources/Marker.cpp')
built = index.RepositoryIndex().build()
assert 'Sources/Marker.cpp' in built.files
assert any(f.name == 'development_marker' for f in built.functions)
assert all(Path(f).suffix in index._ALLOWED_SUFFIXES for f in built.files)
assert not any(f.endswith(('.sh', '.py', '.pbxproj')) for f in built.files)
assert 'expertadvisor_agent' not in sys.modules
assert 'mlx_lm' not in sys.modules
""", EXPERTADVISOR_REPOSITORY_ROOT=str(self.root))

    def test_production_and_development_claim_ledgers_are_isolated(self):
        code = """
import os
from pathlib import Path
from Tools.RepositoryAgent.claim_evidence import (
    VerifiedClaimLedger, CLAIM_LEDGER_CACHE_NAMESPACE, resolve_claim_ledger_path)
assert CLAIM_LEDGER_CACHE_NAMESPACE == os.environ['EXPECTED_NAMESPACE']
path = resolve_claim_ledger_path()
assert path == Path(os.environ['EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER'])
ledger = VerifiedClaimLedger(path)
assert ledger.lookup('topic', 'claim', 'Sources/A.cpp', 1, 1, 'source') is None
ledger.record_decision('topic', 'claim', 'Sources/A.cpp', 1, 1, 'source',
    {'supports': True, 'establishes': 'fixture', 'reason': 'fixture only'})
assert ledger.lookup('topic', 'claim', 'Sources/A.cpp', 1, 1, 'source')['supports']
"""
        production = self.base / "production-cache/verified_claims.json"
        development = self.base / "rollover-cache/verified_claims.json"
        self.child(code, EXPECTED_NAMESPACE="ExpertAdvisor",
                   EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER=str(production))
        before = production.read_bytes()
        self.child(code, EXPECTED_NAMESPACE="ExpertAdvisor-Rollover",
                   EXPERTADVISOR_LEDGER_CACHE_NAMESPACE="ExpertAdvisor-Rollover",
                   EXPERTADVISOR_CLAIM_EVIDENCE_LEDGER=str(development))
        self.assertEqual(production.read_bytes(), before)
        self.assertNotEqual(production.read_bytes(), development.read_bytes())

    def test_deterministic_mcp_is_read_only_and_does_not_load_qwen(self):
        before = {p.relative_to(self.root): p.read_bytes()
                  for p in self.root.rglob("*") if p.is_file()}
        self.child("""
import sys
from Tools.RepositoryAgent.repository_agent_mcp import StdioMCPServer
server = StdioMCPServer()
caps = server.iface.dispatch({'op': 'capabilities'})
assert caps['repository_read_only'] and caps['read_only']
for forbidden in ('shell', 'repository_write', 'git_mutation', 'database', 'build', 'test_execution'):
    response = server._handle_request({'jsonrpc': '2.0', 'id': 1, 'method': 'tools/call',
        'params': {'name': forbidden, 'arguments': {}}})
    assert response['result']['isError'] is True
server.iface.dispatch({'op': 'read', 'file': 'Sources/Marker.cpp', 'start': 1, 'end': 1})
assert server.iface.dispatch({'op': 'list_files', 'prefix': 'Tests'})['files'] == [
    'Tests/DedicatedTrainingWorkerArchitectureTests.sh',
    'Tests/ReleaseWorkerBuildConfigurationTests.sh']
server.iface.dispatch({'op': 'index_stats'})
assert server.iface._claim_runtime is None
assert 'mlx_lm' not in sys.modules
""", EXPERTADVISOR_REPOSITORY_ROOT=str(self.root))
        after = {p.relative_to(self.root): p.read_bytes()
                 for p in self.root.rglob("*") if p.is_file()}
        self.assertEqual(before, after)


    def test_directory_entry_budget_fails_closed(self):
        from unittest.mock import patch
        from Tools.RepositoryAgent import source_reader

        with patch.object(source_reader, "MAX_DIRECTORY_ENTRIES", 1):
            with self.assertRaisesRegex(
                ValueError, "directory-entry budget exceeded"
            ):
                self.reader.list_files()

    def test_oversized_source_line_fails_closed(self):
        from Tools.RepositoryAgent import source_reader

        path = self.root / "Sources" / "Oversized.cpp"
        path.write_text("X" * (source_reader.MAX_LINE_BYTES + 1))

        with self.assertRaisesRegex(
            ValueError, "source line exceeds"
        ):
            self.reader.read_file("Sources/Oversized.cpp")

        with self.assertRaisesRegex(
            ValueError, "source line exceeds"
        ):
            self.reader.search("not-present")

    def test_search_input_byte_budget_fails_closed(self):
        from unittest.mock import patch
        from Tools.RepositoryAgent import source_reader

        with patch.object(source_reader, "MAX_SEARCH_BYTES", 1):
            with self.assertRaisesRegex(
                ValueError, "search input byte budget exceeded"
            ):
                self.reader.search("not-present")

    def test_read_output_byte_budget_fails_closed(self):
        from unittest.mock import patch
        from Tools.RepositoryAgent import source_reader

        fixture = self.root / "Sources" / "OutputBudget.cpp"
        fixture.write_text("output budget fixture\\n")

        with patch.object(source_reader, "MAX_READ_BYTES", 1):
            with self.assertRaisesRegex(
                ValueError, "read-file output byte budget exceeded"
            ):
                self.reader.read_file("Sources/OutputBudget.cpp")

    def test_search_file_budget_fails_closed(self):
        from unittest.mock import patch
        from Tools.RepositoryAgent import source_reader

        with patch.object(source_reader, "MAX_SEARCH_FILES", 1):
            with self.assertRaisesRegex(
                ValueError, "search file budget exceeded"
            ):
                self.reader.search("not-present")

    def test_search_line_budget_fails_closed(self):
        from unittest.mock import patch
        from Tools.RepositoryAgent import source_reader

        with patch.object(source_reader, "MAX_SEARCH_LINES", 1):
            with self.assertRaisesRegex(
                ValueError, "search line budget exceeded"
            ):
                self.reader.search("not-present")



    def test_missing_optional_source_directory_is_allowed(self):
        from unittest.mock import patch

        original_resolve = Path.resolve

        def controlled_resolve(path, *args, **kwargs):
            if path == self.root / "Headers":
                raise FileNotFoundError("optional directory absent")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", controlled_resolve):
            self.reader.list_files()

    def test_unexpected_source_directory_error_fails_closed(self):
        from unittest.mock import patch

        original_resolve = Path.resolve

        def controlled_resolve(path, *args, **kwargs):
            if path == self.root / "Headers":
                raise PermissionError("simulated permission failure")
            return original_resolve(path, *args, **kwargs)

        with patch.object(Path, "resolve", controlled_resolve):
            with self.assertRaisesRegex(
                ValueError, "cannot resolve source directory"
            ):
                self.reader.list_files()

    def test_replacement_symlink_is_rejected_at_open(self):
        import os
        from unittest.mock import patch

        source = self.root / "Sources" / "DescriptorBoundary.cpp"
        outside = self.root.parent / "descriptor_boundary_secret.cpp"

        source.write_text("permitted source\\n")
        outside.write_text("outside secret\\n")

        original_open = os.open

        def substitute_before_open(path, flags, *args, **kwargs):
            if str(path) == str(source):
                source.unlink()
                source.symlink_to(outside)
            return original_open(path, flags, *args, **kwargs)

        try:
            with patch.object(os, "open", substitute_before_open):
                with self.assertRaisesRegex(
                    ValueError, "cannot open validated source file"
                ):
                    self.reader.read_file(
                        "Sources/DescriptorBoundary.cpp"
                    )
        finally:
            outside.unlink(missing_ok=True)

if __name__ == "__main__":
    unittest.main()
