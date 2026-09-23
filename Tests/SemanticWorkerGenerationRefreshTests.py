#!/usr/bin/env python3
"""Disposable tests for atomic same-layout semantic worker generation refresh."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "Scripts" / "RefreshSemanticWorkerGeneration.py"
SPEC = importlib.util.spec_from_file_location("semantic_worker_refresh", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
refresh = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(refresh)
rollover = refresh.rollover
publisher = refresh.publisher


class SemanticWorkerGenerationRefreshTests(unittest.TestCase):
    commit6 = "6" * 40
    commit7 = "7" * 40
    old_commit8 = "8" * 40
    new_commit8 = "9" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea-refresh-test.")
        self.base = Path(self.temporary.name)
        self.root = (self.base / "SemanticWorkers")
        self.root.mkdir()
        self.root = self.root.resolve()
        self.sources = self.base / "products"
        self.sources.mkdir()
        self.runtime_resources = self.make_runtime("shared-runtime")
        self.seed_registry()
        self.training = self.executable("LSTM_Release", b"new-layout-eight-training")
        self.inference = self.executable("lstm-infer-worker", b"new-layout-eight-infer")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes, directory: Path | None = None) -> Path:
        path = (directory or self.sources) / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(contents)
        path.chmod(0o700)
        return path

    def make_runtime(self, label: str, directory: Path | None = None) -> dict[str, Path]:
        base = directory or self.sources
        result = {
            "default.metallib": base / f"{label}-default.metallib",
            "MetaNN_metal.metallib": base / f"{label}-metann.metallib",
        }
        result["default.metallib"].write_bytes((label + " default").encode())
        result["MetaNN_metal.metallib"].write_bytes((label + " metann").encode())
        return result

    def stage(self, executable: Path, layout: int, width: int, commit: str,
              role: str, schema: int, capabilities: list[str], runtime: dict) -> dict:
        digest = publisher.sha256(executable)
        relative, manifest, worker = rollover._worker_value(
            layout, width, commit, digest, role, schema, capabilities, runtime["identity"])
        rollover._stage_worker(self.root, executable, relative, manifest, digest, runtime)
        return worker

    def seed_registry(self) -> None:
        manifest, identity = publisher.runtime_manifest(dict(self.runtime_resources))
        runtime = publisher.stage_runtime_package(self.root, dict(self.runtime_resources), manifest, identity)
        layout6 = self.stage(self.executable("legacy6", b"layout6"), 6, 77, self.commit6,
                             "infer", publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION, ["infer"], runtime)
        layout6["worker_rule"] = "historical"
        layout7_train = self.stage(self.executable("legacy7-train", b"layout7 train"), 7, 77, self.commit7,
                                   "train", publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                                   rollover.TRAINING_CAPABILITIES, runtime)
        layout7_train["worker_rule"] = "historical"
        layout7_infer = self.stage(self.executable("legacy7-infer", b"layout7 infer"), 7, 77, self.commit7,
                                   "infer", publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                                   rollover.INFERENCE_CAPABILITIES, runtime)
        layout7_infer["worker_rule"] = "historical"
        old_train = self.stage(self.executable("old-layout8-train", b"old8 train"), 8, 80, self.old_commit8,
                               "train", publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
                               rollover.TRAINING_CAPABILITIES, runtime)
        old_infer = self.stage(self.executable("old-layout8-infer", b"old8 infer"), 8, 80, self.old_commit8,
                               "infer", publisher.WORKER_MANIFEST_SCHEMA_VERSION,
                               rollover.INFERENCE_CAPABILITIES, runtime)
        registry = {"schema_version": publisher.REGISTRY_SCHEMA_VERSION, "current_layout": 8,
                    "runtimes": [runtime],
                    "workers": [layout6, layout7_train, layout7_infer, old_train, old_infer]}
        publisher.validate_existing_registry(self.root, registry)
        (self.root / "registry.json").write_text(publisher.json_text(registry), encoding="utf-8")

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def registry_bytes(self) -> bytes:
        return (self.root / "registry.json").read_bytes()

    def immutable_bytes(self, worker: dict) -> tuple[bytes, bytes]:
        directory = self.root / worker["executable"]
        return directory.read_bytes(), (directory.parent / "manifest.json").read_bytes()

    def publish(self, **kwargs: object) -> tuple[Path, Path]:
        return refresh.refresh(self.root, self.training, self.inference, 8, 80, self.new_commit8,
                               check_embedded_commit=False,
                               runtime_resources=dict(self.runtime_resources), **kwargs)

    def test_successful_layout8_refresh_replaces_complete_current_pair_only(self) -> None:
        before = self.registry()
        old_current = [worker for worker in before["workers"] if worker["semantic_layout"] == 8]
        old_current_bytes = [self.immutable_bytes(worker) for worker in old_current]
        old_historical = [worker for worker in before["workers"] if worker["semantic_layout"] in {6, 7}]
        old_historical_bytes = [self.immutable_bytes(worker) for worker in old_historical]
        training, inference = self.publish()
        after = self.registry()
        self.assertEqual(after["current_layout"], 8)
        current = [worker for worker in after["workers"] if worker["semantic_layout"] == 8]
        self.assertEqual({worker["worker_role"] for worker in current}, {"train", "infer"})
        self.assertTrue(all(worker["worker_rule"] == "current" for worker in current))
        self.assertTrue(all(worker["model_input_width"] == 80 for worker in current))
        self.assertEqual({worker["source_commit"] for worker in current}, {self.new_commit8})
        self.assertEqual([worker for worker in after["workers"] if worker["semantic_layout"] in {6, 7}],
                         old_historical)
        self.assertEqual([self.immutable_bytes(worker) for worker in old_historical],
                         old_historical_bytes)
        self.assertEqual([self.immutable_bytes(worker) for worker in old_current], old_current_bytes)
        self.assertEqual(training.name, "LSTM_Release")
        self.assertEqual(inference.name, "lstm-infer-worker")
        self.assertEqual((self.root / "current").resolve(), inference.parent)

    def test_registry_replacement_failure_leaves_prior_registry_byte_identical(self) -> None:
        before = self.registry_bytes()
        with mock.patch.object(
            publisher, "atomic_write_json",
            side_effect=OSError("injected registry replacement failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected registry replacement failure"):
                self.publish()
        self.assertEqual(self.registry_bytes(), before)

    def test_convenience_link_failure_after_commit_reports_committed_registry(self) -> None:
        before = self.registry_bytes()
        original_replace = publisher.os.replace

        def fail_current_link(source: object, destination: object) -> None:
            if Path(destination) == self.root / "current":
                raise OSError("injected current link replacement failure")
            original_replace(source, destination)

        with mock.patch.object(publisher.os, "replace", side_effect=fail_current_link):
            with self.assertRaisesRegex(
                publisher.RegistryCommittedLinkUpdateError,
                "registry committed but current convenience link update failed",
            ):
                self.publish()
        self.assertNotEqual(self.registry_bytes(), before)
        committed = publisher.load_registry(self.root / "registry.json")
        current = [worker for worker in committed["workers"]
                   if worker["semantic_layout"] == committed["current_layout"]]
        self.assertEqual(committed["current_layout"], 8)
        self.assertEqual({worker["source_commit"] for worker in current}, {self.new_commit8})

    def test_convenience_link_fsync_failure_after_commit_reports_committed_registry(self) -> None:
        before = self.registry_bytes()
        original_fsync_directory = publisher.fsync_directory
        root_fsync_count = 0

        def fail_post_commit_fsync(path: Path) -> None:
            nonlocal root_fsync_count
            if path == self.root:
                root_fsync_count += 1
                if root_fsync_count == 2:
                    raise OSError("injected current link fsync failure")
            original_fsync_directory(path)

        with mock.patch.object(publisher, "fsync_directory", side_effect=fail_post_commit_fsync):
            with self.assertRaisesRegex(
                publisher.RegistryCommittedLinkUpdateError,
                "registry committed but current convenience link update failed",
            ):
                self.publish()
        self.assertEqual(root_fsync_count, 2)
        self.assertNotEqual(self.registry_bytes(), before)
        committed = publisher.load_registry(self.root / "registry.json")
        self.assertEqual(committed["current_layout"], 8)
        self.assertEqual(
            {worker["source_commit"] for worker in committed["workers"]
             if worker["semantic_layout"] == 8},
            {self.new_commit8},
        )

    def test_train_only_or_infer_only_candidate_is_impossible(self) -> None:
        before = self.registry_bytes()
        missing = self.sources / "missing"
        with self.assertRaises(publisher.PublishError):
            refresh.refresh(self.root, self.training, missing, 8, 80, self.new_commit8,
                            check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        wrong = self.executable("not-infer", b"not infer")
        with self.assertRaisesRegex(publisher.PublishError, "expected lstm-infer-worker"):
            refresh.refresh(self.root, self.training, wrong, 8, 80, self.new_commit8,
                            check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry_bytes(), before)

    def test_mismatched_requested_commit_and_wrong_layout_or_width_leave_registry_unchanged(self) -> None:
        before = self.registry_bytes()
        with mock.patch.object(publisher, "clean_source_commit", return_value=self.new_commit8):
            with self.assertRaisesRegex(publisher.PublishError, "disagrees with clean HEAD"):
                refresh.refresh_from_repository(REPOSITORY, self.training, self.inference, self.root,
                                                source_commit="a" * 40)
        with self.assertRaisesRegex(publisher.PublishError, "layout to match source contract"):
            refresh.refresh(self.root, self.training, self.inference, 9, 80, self.new_commit8,
                            check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        with self.assertRaisesRegex(publisher.PublishError, "registry width to match source contract"):
            refresh.refresh(self.root, self.training, self.inference, 8, 81, self.new_commit8,
                            check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry_bytes(), before)

    def test_incomplete_current_prestate_and_prospective_validation_failure_leave_registry_unchanged(self) -> None:
        before = self.registry_bytes()
        registry = self.registry()
        registry["workers"] = [worker for worker in registry["workers"]
                               if not (worker["semantic_layout"] == 8 and worker["worker_role"] == "train")]
        (self.root / "registry.json").write_text(publisher.json_text(registry), encoding="utf-8")
        incomplete = self.registry_bytes()
        with self.assertRaises(publisher.PublishError):
            self.publish()
        self.assertEqual(self.registry_bytes(), incomplete)
        (self.root / "registry.json").write_bytes(before)
        with mock.patch.object(publisher, "validate_existing_registry",
                               side_effect=publisher.PublishError("injected prospective failure")):
            with self.assertRaisesRegex(publisher.PublishError, "injected prospective failure"):
                self.publish()
        self.assertEqual(self.registry_bytes(), before)

    def test_identity_and_runtime_mismatch_leave_registry_unchanged(self) -> None:
        before = self.registry_bytes()
        with mock.patch.object(
            publisher, "verify_embedded_commit",
            side_effect=publisher.PublishError("built executable does not contain exact source commit"),
        ):
            with self.assertRaisesRegex(publisher.PublishError, "exact source commit"):
                refresh.refresh(self.root, self.training, self.inference, 8, 80, self.new_commit8)
        self.assertEqual(self.registry_bytes(), before)
        with mock.patch.object(publisher, "verify_embedded_commit") as embedded:
            with mock.patch.object(publisher, "verify_inference_build_identity",
                                   side_effect=publisher.PublishError("inference worker build identity mismatch")):
                with self.assertRaisesRegex(publisher.PublishError, "build identity mismatch"):
                    refresh.refresh(self.root, self.training, self.inference, 8, 80, self.new_commit8)
        self.assertGreaterEqual(embedded.call_count, 2)
        training_dir = self.base / "runtime-training"
        inference_dir = self.base / "runtime-inference"
        training = self.executable("LSTM_Release", b"training", training_dir)
        inference = self.executable("lstm-infer-worker", b"inference", inference_dir)
        self.executable("default.metallib", b"same", training_dir)
        self.executable("MetaNN_metal.metallib", b"same", training_dir)
        self.executable("default.metallib", b"different", inference_dir)
        self.executable("MetaNN_metal.metallib", b"same", inference_dir)
        with mock.patch.object(publisher, "verify_embedded_commit"):
            with mock.patch.object(publisher, "verify_inference_build_identity"):
                with self.assertRaisesRegex(publisher.PublishError, "runtime resource mismatch"):
                    refresh.refresh(self.root, training, inference, 8, 80, self.new_commit8)
        self.assertEqual(self.registry_bytes(), before)

    def test_immutable_artifact_conflict_leaves_registry_unchanged(self) -> None:
        before = self.registry_bytes()
        digest = publisher.sha256(self.training)
        conflict = self.root / "layout8" / self.new_commit8 / digest
        conflict.mkdir(parents=True)
        (conflict / "LSTM_Release").write_bytes(b"wrong immutable artifact")
        (conflict / "LSTM_Release").chmod(0o555)
        (conflict / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "immutable artifact hash conflict"):
            self.publish()
        self.assertEqual(self.registry_bytes(), before)


if __name__ == "__main__":
    unittest.main()
