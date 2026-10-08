#!/usr/bin/env python3
"""Offline regression checks for dedicated semantic-layout rollover.

Run: python3 -m unittest discover -s Scripts/tests -p 'test_dedicated_train_rollover.py'
No production registry or worker binaries are accessed.
"""
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[2] / "Scripts" / "RollSemanticWorkerLayout.py"
SPEC = importlib.util.spec_from_file_location("rollover_under_test", SCRIPT)
rollover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rollover)
publisher = rollover.publisher

TRAIN_COMMIT = "a" * 40
INFER_COMMIT = "b" * 40
TRAIN_DIGEST = "c" * 64
INFER_DIGEST = "d" * 64
RUNTIME_ID = "e" * 64


class DedicatedTrainRolloverTests(unittest.TestCase):
    def test_role_aware_train_manifest_and_capabilities(self):
        relative, manifest, worker = rollover._worker_value(
            14, 171, TRAIN_COMMIT, TRAIN_DIGEST, "train",
            publisher.WORKER_MANIFEST_SCHEMA_VERSION,
            ["train", "train_feature_ablation_v1"], RUNTIME_ID)
        self.assertEqual(
            relative, Path("layout14/train") / TRAIN_COMMIT / TRAIN_DIGEST)
        self.assertEqual(manifest["executable_identity"], "lstm-train-worker")
        self.assertEqual(manifest["worker_role"], "train")
        self.assertEqual(worker["executable"],
                         str(relative / "lstm-train-worker"))
        self.assertEqual(worker["capabilities"],
                         ["train", "train_feature_ablation_v1"])

    def test_legacy_train_artifact_contract_preserved(self):
        relative, manifest, worker = rollover._worker_value(
            14, 171, TRAIN_COMMIT, TRAIN_DIGEST, "train",
            publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
            rollover.training_capabilities(True), RUNTIME_ID)
        self.assertEqual(relative, Path("layout14") / TRAIN_COMMIT / TRAIN_DIGEST)
        self.assertEqual(manifest["executable_identity"], "LSTM_Release")
        self.assertNotIn("worker_role", manifest)
        self.assertIn("infer", worker["capabilities"])

    def test_dedicated_capabilities_are_narrow(self):
        with self.assertRaises(publisher.PublishError):
            rollover._worker_value(
                14, 171, TRAIN_COMMIT, TRAIN_DIGEST, "train",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.training_capabilities(True), RUNTIME_ID)

    def test_rejects_wrong_executable_basename(self):
        with tempfile.TemporaryDirectory() as directory:
            executable = Path(directory) / "LSTM_Release"
            executable.write_bytes(b"test")
            executable.chmod(0o755)
            with self.assertRaises(publisher.PublishError):
                rollover._resolve_executable(executable, "lstm-train-worker")

    def test_rejects_bad_inference_commit_before_registry_write(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            for executable in (training, inference):
                executable.write_bytes(b"fixture")
                executable.chmod(0o755)
            with self.assertRaises(publisher.PublishError):
                rollover.rollover(
                    root / "artifacts", training, inference, 14, 171,
                    TRAIN_COMMIT, dedicated_training=True,
                    inference_commit="invalid", check_embedded_commit=False,
                    runtime_resources={})
            self.assertFalse((root / "artifacts" / "registry.json").exists())

    def test_distinct_commit_identity_checks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            for executable in (training, inference):
                executable.write_bytes(b"fixture")
                executable.chmod(0o755)
            calls = []
            def verify_commit(path, commit):
                calls.append((path.name, commit))
                if len(calls) == 2:
                    raise publisher.PublishError("stopped after identity check")
            with patch.object(publisher, "verify_embedded_commit", side_effect=verify_commit), \
                 patch.object(publisher, "verify_worker_build_identity"):
                with self.assertRaisesRegex(publisher.PublishError, "stopped"):
                    rollover.rollover(
                        root / "artifacts", training, inference, 14, 171,
                        TRAIN_COMMIT, dedicated_training=True,
                        inference_commit=INFER_COMMIT)
            self.assertEqual(calls, [
                ("lstm-train-worker", TRAIN_COMMIT),
                ("lstm-infer-worker", INFER_COMMIT)])
            self.assertFalse((root / "artifacts" / "registry.json").exists())


if __name__ == "__main__":
    unittest.main()
