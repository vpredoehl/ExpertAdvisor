#!/usr/bin/env python3
"""Permanent trusted-validator execution and receipt boundary regressions."""
from __future__ import annotations

import copy
import os
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Scripts"))
from CampaignOperationsH1TrustedRunner import RunnerError, TrustedValidatorRunner  # noqa: E402


class TrustedRunnerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="ea-h1-trusted-runner-")
        self.root = Path(self.temp.name)
        self.script = self.root / "validator.py"
        self.script.write_text("""#!/usr/bin/env python3
import os, pathlib, sys
entry, output, behavior = sys.argv[1:4]
if entry != 'validate_one': sys.exit(23)
if behavior == 'nonzero': sys.exit(17)
path = pathlib.Path(output)
path.write_text('validator_result_id\\tstatus\\nVR-ONE\\tPASS\\n')
if behavior == 'undeclared': (path.parent / 'rogue.out').write_text('rogue')
print('key_visible=' + str('H1_VALIDATOR_RECEIPT_KEY' in os.environ).lower())
""")
        self.script.chmod(0o700)
        self.output = self.root / "result.tsv"
        self.validators = {"VAL-ONE": {
            "version": "h1-validator-registry-v2", "validator_id": "VAL-ONE",
            "implementation": "validator.py", "entry_point": "validate_one", "executable_required": "true",
        }}
        self.runner = TrustedValidatorRunner(self.root, self.validators)
        self.inputs = {"ART-IN": "a" * 64}

    def tearDown(self): self.temp.cleanup()

    def execute(self, behavior="ok"):
        return self.runner.execute("VAL-ONE", "run-1", [str(self.output), behavior], self.inputs,
                                   {"ART-OUT": self.output}, {"VR-ONE"})

    def assert_rejected(self, expected, function, *args):
        with self.assertRaisesRegex(RunnerError, expected): function(*args)

    def test_distinct_process_is_observed_and_key_is_not_in_child_environment(self):
        observation = self.execute()
        self.assertNotEqual(observation.receipt["process_id"], str(os.getpid()))
        self.assertIn(b"key_visible=false", observation.stdout)
        self.runner.validate(observation.receipt, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_receipt_without_execution_is_rejected(self):
        observation = self.execute(); forged = copy.deepcopy(observation.receipt)
        forged["validator_execution_id"] = "VEXEC-run-1-never-executed"
        self.assert_rejected("invalid-receipt-attestation", self.runner.validate, forged, "run-1",
                             self.inputs, {"ART-OUT": self.output}, {"VR-ONE"})

    def test_nonexistent_process_identity_is_rejected(self):
        observation = self.execute(); forged = copy.deepcopy(observation.receipt)
        forged["process_id"] = "999999999"
        self.assert_rejected("invalid-receipt-attestation", self.runner.validate, forged, "run-1",
                             self.inputs, {"ART-OUT": self.output}, {"VR-ONE"})

    def test_asserted_zero_cannot_hide_nonzero_validator_exit(self):
        self.assert_rejected("validator-exited-nonzero:17", self.execute, "nonzero")

    def test_wrong_implementation_path_is_rejected_against_registry(self):
        observation = self.execute(); forged = copy.deepcopy(observation.receipt)
        forged["implementation_path"] = "other.py"
        self.assert_rejected("registry-or-artifact-binding-mismatch:implementation_path",
                             self.runner.validate, forged, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_wrong_entry_point_is_rejected_against_registry(self):
        observation = self.execute(); forged = copy.deepcopy(observation.receipt)
        forged["entry_point"] = "validate_other"
        self.assert_rejected("registry-or-artifact-binding-mismatch:entry_point",
                             self.runner.validate, forged, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_reused_receipt_is_rejected(self):
        observation = self.execute()
        self.runner.validate(observation.receipt, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})
        self.assert_rejected("duplicate-or-reused-receipt", self.runner.validate,
                             observation.receipt, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_stale_receipt_is_rejected(self):
        observation = self.execute(); forged = copy.deepcopy(observation.receipt)
        forged["wall_completion_time"] = "2000-01-01T00:00:00+00:00"
        self.assert_rejected("stale-receipt", self.runner.validate, forged, "run-1",
                             self.inputs, {"ART-OUT": self.output}, {"VR-ONE"})

    def test_output_mismatch_is_rejected(self):
        observation = self.execute(); self.output.write_text("changed\n")
        self.assert_rejected("registry-or-artifact-binding-mismatch:output_snapshot_ids",
                             self.runner.validate, observation.receipt, "run-1", self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_input_mismatch_is_rejected(self):
        observation = self.execute()
        self.assert_rejected("registry-or-artifact-binding-mismatch:input_artifact_digests",
                             self.runner.validate, observation.receipt, "run-1", {"ART-IN": "b" * 64},
                             {"ART-OUT": self.output}, {"VR-ONE"})

    def test_missing_and_undeclared_outputs_are_rejected(self):
        missing = self.root / "missing.tsv"
        self.assert_rejected("missing-declared-output:ART-OUT", self.runner.execute,
                             "VAL-ONE", "run-1", [str(self.output), "ok"], self.inputs,
                             {"ART-OUT": missing}, {"VR-ONE"})
        if self.output.exists(): self.output.unlink()
        self.assert_rejected("undeclared-output:rogue.out", self.execute, "undeclared")

    def test_one_execution_cannot_represent_multiple_independent_validators(self):
        self.assert_rejected("output-validator-result-set-mismatch", self.runner.execute,
                             "VAL-ONE", "run-1", [str(self.output), "ok"], self.inputs,
                             {"ART-OUT": self.output}, {"VR-ONE", "VR-TWO"})


if __name__ == "__main__": unittest.main(verbosity=2)
