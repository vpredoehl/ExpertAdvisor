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

    def test_legacy_rollover_rejects_divergent_infer_commit_before_staging(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            train = root / "LSTM_Release"
            infer = root / "lstm-infer-worker"
            for executable in (train, infer):
                executable.write_bytes(b"fixture")
                executable.chmod(0o755)
            artifacts = root / "artifacts"
            with self.assertRaises(publisher.PublishError):
                rollover.rollover(
                    artifacts, train, infer, 14, 171, TRAIN_COMMIT,
                    inference_commit=INFER_COMMIT,
                    check_embedded_commit=False, runtime_resources={})
            self.assertFalse(artifacts.exists())

    def test_disposable_rollover_retains_previous_generation(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            # macOS /var is commonly a symlink to /private/var. The
            # publisher correctly requires canonical artifact paths.
            root = Path(directory).resolve(strict=True)
            artifacts = root / "artifacts"
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            legacy = root / "LSTM_Release"
            for binary in (training, inference, legacy):
                binary.write_bytes(binary.name.encode())
                binary.chmod(0o755)
            resources = {}
            for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
                resource = root / name
                resource.write_bytes(name.encode())
                resources[name] = resource
            # Rollover requires an existing current layout with both roles.
            # Seed the disposable registry with the publisher's own
            # validated artifact/registry format, without using rollover.
            artifacts.mkdir()
            runtime_manifest, runtime_id = publisher.runtime_manifest(resources)
            runtime = publisher.stage_runtime_package(
                artifacts, resources, runtime_manifest, runtime_id)
            legacy_digest = publisher.sha256(legacy)
            infer_digest = publisher.sha256(inference)
            legacy_rel, legacy_manifest, legacy_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, legacy_digest, "train",
                publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.training_capabilities(), runtime_id)
            infer_rel, infer_manifest, infer_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, infer_digest, "infer",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.INFERENCE_CAPABILITIES, runtime_id)
            rollover._stage_worker(
                artifacts, legacy, legacy_rel, legacy_manifest, legacy_digest, runtime)
            rollover._stage_worker(
                artifacts, inference, infer_rel, infer_manifest, infer_digest, runtime)
            seed = {
                "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                "current_layout": 13,
                "runtimes": [runtime],
                "workers": [legacy_worker, infer_worker],
            }
            publisher.validate_existing_registry(artifacts, seed)
            publisher.atomic_write_json(artifacts / "registry.json", seed)
            before = json.loads((artifacts / "registry.json").read_text())
            self.assertEqual(len(before["workers"]), 2)
            train_path, infer_path = rollover.rollover(
                artifacts, training, inference, 14, 171, TRAIN_COMMIT,
                inference_commit=INFER_COMMIT, dedicated_training=True,
                check_embedded_commit=False, runtime_resources=resources)
            after = json.loads((artifacts / "registry.json").read_text())
            self.assertEqual(after["current_layout"], 14)
            self.assertEqual(len(after["workers"]), 4)
            self.assertTrue(train_path.exists())
            self.assertTrue(infer_path.exists())
            for old in before["workers"]:
                matches = [w for w in after["workers"]
                           if w["executable"] == old["executable"]]
                self.assertEqual(len(matches), 1)
                self.assertEqual(matches[0]["worker_rule"], "historical")
                self.assertTrue((artifacts / old["executable"]).exists())
            new_train = next(w for w in after["workers"]
                             if w["semantic_layout"] == 14 and w["worker_role"] == "train")
            new_infer = next(w for w in after["workers"]
                             if w["semantic_layout"] == 14 and w["worker_role"] == "infer")
            self.assertEqual(new_train["artifact_manifest_schema_version"], 2)
            self.assertEqual(new_train["source_commit"], TRAIN_COMMIT)
            self.assertEqual(new_infer["source_commit"], INFER_COMMIT)

    def test_registry_replace_failure_preserves_prior_authority(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            artifacts = root / "artifacts"
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            legacy = root / "LSTM_Release"
            for binary in (training, inference, legacy):
                binary.write_bytes(binary.name.encode())
                binary.chmod(0o755)
            resources = {}
            for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
                resource = root / name
                resource.write_bytes(name.encode())
                resources[name] = resource
            # Rollover requires an existing current layout with both roles.
            # Seed the disposable registry with the publisher's own
            # validated artifact/registry format, without using rollover.
            artifacts.mkdir()
            runtime_manifest, runtime_id = publisher.runtime_manifest(resources)
            runtime = publisher.stage_runtime_package(
                artifacts, resources, runtime_manifest, runtime_id)
            legacy_digest = publisher.sha256(legacy)
            infer_digest = publisher.sha256(inference)
            legacy_rel, legacy_manifest, legacy_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, legacy_digest, "train",
                publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.training_capabilities(), runtime_id)
            infer_rel, infer_manifest, infer_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, infer_digest, "infer",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.INFERENCE_CAPABILITIES, runtime_id)
            rollover._stage_worker(
                artifacts, legacy, legacy_rel, legacy_manifest, legacy_digest, runtime)
            rollover._stage_worker(
                artifacts, inference, infer_rel, infer_manifest, infer_digest, runtime)
            seed = {
                "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                "current_layout": 13,
                "runtimes": [runtime],
                "workers": [legacy_worker, infer_worker],
            }
            publisher.validate_existing_registry(artifacts, seed)
            publisher.atomic_write_json(artifacts / "registry.json", seed)
            registry_path = artifacts / "registry.json"
            original = registry_path.read_bytes()
            with patch.object(publisher, "atomic_write_json",
                              side_effect=OSError("injected registry replacement failure")):
                with self.assertRaisesRegex(OSError, "injected registry"):
                    rollover.rollover(
                        artifacts, training, inference, 14, 171, TRAIN_COMMIT,
                        inference_commit=INFER_COMMIT, dedicated_training=True,
                        check_embedded_commit=False, runtime_resources=resources)
            self.assertEqual(registry_path.read_bytes(), original)
            restored = json.loads(registry_path.read_text())
            self.assertEqual(restored["current_layout"], 13)
            publisher.validate_existing_registry(artifacts, restored)
            for worker in restored["workers"]:
                self.assertTrue((artifacts / worker["executable"]).exists())

    def test_post_commit_link_failure_preserves_committed_registry(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            artifacts = root / "artifacts"
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            legacy = root / "LSTM_Release"
            for binary in (training, inference, legacy):
                binary.write_bytes(binary.name.encode())
                binary.chmod(0o755)
            resources = {}
            for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
                resource = root / name
                resource.write_bytes(name.encode())
                resources[name] = resource
            # Rollover requires an existing current layout with both roles.
            # Seed the disposable registry with the publisher's own
            # validated artifact/registry format, without using rollover.
            artifacts.mkdir()
            runtime_manifest, runtime_id = publisher.runtime_manifest(resources)
            runtime = publisher.stage_runtime_package(
                artifacts, resources, runtime_manifest, runtime_id)
            legacy_digest = publisher.sha256(legacy)
            infer_digest = publisher.sha256(inference)
            legacy_rel, legacy_manifest, legacy_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, legacy_digest, "train",
                publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.training_capabilities(), runtime_id)
            infer_rel, infer_manifest, infer_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, infer_digest, "infer",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.INFERENCE_CAPABILITIES, runtime_id)
            rollover._stage_worker(
                artifacts, legacy, legacy_rel, legacy_manifest, legacy_digest, runtime)
            rollover._stage_worker(
                artifacts, inference, infer_rel, infer_manifest, infer_digest, runtime)
            seed = {
                "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                "current_layout": 13,
                "runtimes": [runtime],
                "workers": [legacy_worker, infer_worker],
            }
            publisher.validate_existing_registry(artifacts, seed)
            publisher.atomic_write_json(artifacts / "registry.json", seed)
            registry_path = artifacts / "registry.json"
            original = registry_path.read_bytes()
            with patch.object(publisher, "update_current_link_after_registry_commit",
                              side_effect=publisher.RegistryCommittedLinkUpdateError(
                                  "injected post-commit link failure")):
                with self.assertRaisesRegex(
                        publisher.RegistryCommittedLinkUpdateError, "post-commit"):
                    rollover.rollover(
                        artifacts, training, inference, 14, 171, TRAIN_COMMIT,
                        inference_commit=INFER_COMMIT, dedicated_training=True,
                        check_embedded_commit=False, runtime_resources=resources)
            self.assertNotEqual(registry_path.read_bytes(), original)
            restored = json.loads(registry_path.read_text())
            self.assertEqual(restored["current_layout"], 14)
            publisher.validate_existing_registry(artifacts, restored)
            for worker in restored["workers"]:
                self.assertTrue((artifacts / worker["executable"]).exists())

    def test_immutable_artifact_conflict_preserves_prior_authority(self):
        import json
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve(strict=True)
            artifacts = root / "artifacts"
            training = root / "lstm-train-worker"
            inference = root / "lstm-infer-worker"
            legacy = root / "LSTM_Release"
            for binary in (training, inference, legacy):
                binary.write_bytes(binary.name.encode())
                binary.chmod(0o755)
            resources = {}
            for name, _ in publisher.RUNTIME_RESOURCE_SPECS:
                resource = root / name
                resource.write_bytes(name.encode())
                resources[name] = resource
            # Rollover requires an existing current layout with both roles.
            # Seed the disposable registry with the publisher's own
            # validated artifact/registry format, without using rollover.
            artifacts.mkdir()
            runtime_manifest, runtime_id = publisher.runtime_manifest(resources)
            runtime = publisher.stage_runtime_package(
                artifacts, resources, runtime_manifest, runtime_id)
            legacy_digest = publisher.sha256(legacy)
            infer_digest = publisher.sha256(inference)
            legacy_rel, legacy_manifest, legacy_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, legacy_digest, "train",
                publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.training_capabilities(), runtime_id)
            infer_rel, infer_manifest, infer_worker = rollover._worker_value(
                13, 171, TRAIN_COMMIT, infer_digest, "infer",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                rollover.INFERENCE_CAPABILITIES, runtime_id)
            rollover._stage_worker(
                artifacts, legacy, legacy_rel, legacy_manifest, legacy_digest, runtime)
            rollover._stage_worker(
                artifacts, inference, infer_rel, infer_manifest, infer_digest, runtime)
            seed = {
                "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                "current_layout": 13,
                "runtimes": [runtime],
                "workers": [legacy_worker, infer_worker],
            }
            publisher.validate_existing_registry(artifacts, seed)
            publisher.atomic_write_json(artifacts / "registry.json", seed)
            registry_path = artifacts / "registry.json"
            original = registry_path.read_bytes()
            # Poison the exact immutable TRAIN destination before publication.
            # A conflicting executable must never advance registry authority.
            digest = publisher.sha256(training)
            relative, _, _ = rollover._worker_value(
                14, 171, TRAIN_COMMIT, digest, "train",
                publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                ["train"], runtime_id)
            collision = artifacts / relative
            collision.mkdir(parents=True)
            (collision / "lstm-train-worker").write_bytes(b"conflicting-artifact")
            with self.assertRaises(publisher.PublishError):
                rollover.rollover(
                    artifacts, training, inference, 14, 171, TRAIN_COMMIT,
                    inference_commit=INFER_COMMIT, dedicated_training=True,
                    check_embedded_commit=False, runtime_resources=resources)
            self.assertEqual(registry_path.read_bytes(), original)
            restored = json.loads(registry_path.read_text())
            self.assertEqual(restored["current_layout"], 13)
            publisher.validate_existing_registry(artifacts, restored)
            for worker in restored["workers"]:
                self.assertTrue((artifacts / worker["executable"]).exists())

    def test_dedicated_train_compiled_contract_mismatch_fails_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            training = root / "lstm-train-worker"
            training.write_bytes(b"fixture")
            training.chmod(0o755)
            def identity(layout, width):
                return ("TRAIN_WORKER_BUILD_IDENTITY,identity_contract_version=1,artifact_role=lstm-train-worker,"
                        f"semantic_layout={layout},model_input_width={width},"
                        "source_commit=" + TRAIN_COMMIT)
            for reported_layout, reported_width in (
                    (13, 171), (14, 170), (None, None)):
                with self.subTest(layout=reported_layout, width=reported_width):
                    output = (identity(reported_layout, reported_width)
                              if reported_layout is not None
                              else "TRAIN_WORKER_BUILD_IDENTITY,identity_contract_version=1,artifact_role=lstm-train-worker")
                    completed = __import__("subprocess").CompletedProcess(
                        args=[], returncode=0, stdout=output, stderr="")
                    with patch.object(rollover.subprocess, "run", return_value=completed):
                        with self.assertRaisesRegex(
                                publisher.PublishError, "semantic layout or model input width"):
                            rollover.verify_train_semantic_contract(training, 14, 171)
            completed = __import__("subprocess").CompletedProcess(
                args=[], returncode=0, stdout=identity(14, 171), stderr="")
            with patch.object(rollover.subprocess, "run", return_value=completed):
                rollover.verify_train_semantic_contract(training, 14, 171)

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
