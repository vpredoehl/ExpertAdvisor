#!/usr/bin/env python3
from __future__ import annotations

import copy
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1EvidenceAuthority import RAW_CLASS_PAYLOADS  # noqa: E402
from CampaignOperationsH1TrustedGenerator import TrustedGeneratorRunner  # noqa: E402
from CampaignOperationsH1TrustedRunner import RunnerError  # noqa: E402


class TrustedGeneratorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-generator-")
        self.root = Path(self.temp.name)
        self.script = self.root / "generator.py"
        self.script.write_text("""#!/usr/bin/env python3
import pathlib, sys
entry, output, payload = sys.argv[1:4]
if entry != 'generate': sys.exit(31)
pathlib.Path(output).write_text(payload)
""")
        self.script.chmod(0o700)
        registry = {"GEN": {"version": "h1-generator-registry-v2", "generator_id": "GEN",
                              "implementation": "generator.py", "entry_point": "generate",
                              "executable_required": "true"}}
        self.runner = TrustedGeneratorRunner(self.root, registry)

    def tearDown(self):
        self.temp.cleanup()

    def test_all_active_classes_produce_v2_envelopes_from_observed_payload_bytes(self):
        for evidence_class, fields in RAW_CLASS_PAYLOADS.items():
            with self.subTest(evidence_class=evidence_class):
                output = self.root / f"{evidence_class}.json"
                payload = json.dumps({field: [] for field in fields}, sort_keys=True)
                observed = self.runner.execute("GEN", "run-1", [str(output), payload], {},
                                               {f"ART-{evidence_class}": output})
                self.runner.validate(observed, "run-1", {})
                envelope = self.runner.raw_envelope(observed, evidence_class, "REQ", "run-1",
                                                    f"ART-{evidence_class}")
                self.assertEqual(envelope["evidence_version"], "h1-raw-execution-evidence-v2")
                self.assertEqual(envelope["generator_execution_id"],
                                 observed.receipt["generator_execution_id"])

    def test_forged_generator_identity_is_rejected(self):
        output = self.root / "payload.json"
        payload = json.dumps({field: [] for field in RAW_CLASS_PAYLOADS["lock"]})
        observed = self.runner.execute("GEN", "run-1", [str(output), payload], {}, {"ART": output})
        forged = copy.deepcopy(observed)
        forged.receipt["process_id"] = "999999"
        with self.assertRaisesRegex(RunnerError, "invalid-generator-receipt-attestation"):
            self.runner.validate(forged, "run-1", {})


if __name__ == "__main__":
    unittest.main(verbosity=2)
