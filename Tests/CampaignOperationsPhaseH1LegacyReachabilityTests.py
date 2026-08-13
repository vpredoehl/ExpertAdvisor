#!/usr/bin/env python3
from __future__ import annotations

import ast
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
import CampaignOperationsH1EvidenceAuthority as authority  # noqa: E402


class LegacyReachabilityTests(unittest.TestCase):
    def test_legacy_receipt_emitter_is_a_stable_fail_closed_tombstone(self):
        with self.assertRaises(SystemExit):
            authority.emit_execution_receipts()

    def test_graph_uses_trusted_runner_and_has_no_direct_subprocess_launcher(self):
        source = (ROOT / "Scripts/CampaignOperationsH1EvidenceGraph.py").read_text()
        self.assertIn("TrustedValidatorRunner", source)
        self.assertNotIn("subprocess.run", source)
        self.assertNotIn("subprocess.Popen", source)

    def test_validator_has_no_legacy_receipt_call_or_key_file_path(self):
        source = (ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py").read_text()
        tree = ast.parse(source)
        calls = [node.func.id for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)]
        self.assertNotIn("emit_execution_receipts", calls)
        self.assertNotIn("H1_VALIDATOR_RECEIPT_KEY", source)
        self.assertNotIn("h1-validator-execution-receipt-v1", source)

    def test_restore_consumer_contains_no_pathname_read(self):
        source = (ROOT / "Scripts/CampaignOperationsH1RestoreEvidence.py").read_text()
        for token in (".open(", ".read_bytes(", ".read_text(", ".is_file("):
            self.assertNotIn(token, source)

    def test_direct_validator_entry_is_rejected(self):
        with tempfile.TemporaryDirectory(prefix="ea-h1-legacy-validator-") as directory:
            result = subprocess.run(
                [sys.executable, str(ROOT / "Scripts/CampaignOperationsH1EvidenceValidator.py"),
                 "generate", directory, "run-1", str(Path(directory) / "result.tsv")],
                text=True, capture_output=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("H1V010 key=legacy-validator-entry", result.stderr)

    def test_migration_harness_contains_no_coauthored_acl_fields(self):
        source = (ROOT / "Tests/CampaignOperationsPhaseH1MigrationTests.sh").read_text()
        self.assertNotIn("expected_tuple_state", source)
        self.assertNotIn("actual_tuple_state", source)
        self.assertNotIn("h1-acl-catalog-runtime-v2", source)
        self.assertNotIn("H1ACL501", source)
        self.assertIn("CampaignOperationsH1CatalogPipeline.py", source)
        self.assertNotIn("explicit.tsv", source)
        self.assertNotIn("default.tsv", source)

    def test_traceability_parser_rejects_v1_runtime_records(self):
        source = (ROOT / "Tests/CampaignOperationsPhaseH1TraceabilityTests.sh").read_text()
        self.assertNotIn("h1-runtime-result-v1", source)
        self.assertNotIn("if($0!=legacy && $0!=current)", source)


if __name__ == "__main__":
    unittest.main(verbosity=2)
