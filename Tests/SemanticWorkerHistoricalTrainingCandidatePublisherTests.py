#!/usr/bin/env python3
"""Disposable v5 historical TRAIN-candidate publication tests."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "Scripts" / "PublishHistoricalTrainingCandidate.py"
SPEC = importlib.util.spec_from_file_location("historical_training_candidate", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
candidate_publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(candidate_publisher)
publisher = candidate_publisher.publisher
rollover = candidate_publisher.rollover


class HistoricalTrainingCandidatePublisherTests(unittest.TestCase):
    old_commit = "6" * 40
    new_commit = "7" * 40
    current_commit = "8" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea-train-candidate-test.")
        self.base = Path(self.temporary.name)
        self.root = (self.base / "SemanticWorkers")
        self.root.mkdir()
        self.root = self.root.resolve()
        self.sources = self.base / "sources"
        self.sources.mkdir()
        self.runtime_resources = {
            "default.metallib": self.sources / "default.metallib",
            "MetaNN_metal.metallib": self.sources / "metann.metallib",
        }
        for path in self.runtime_resources.values():
            path.write_bytes(path.name.encode())
        self.seed_registry()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(contents)
        path.chmod(0o700)
        return path

    def stage(self, executable: Path, layout: int, width: int, commit: str,
              role: str, capabilities: list[str], runtime: dict) -> dict:
        digest = publisher.sha256(executable)
        relative, manifest, worker = rollover._worker_value(
            layout, width, commit, digest, role,
            publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
            capabilities, runtime["identity"])
        rollover._stage_worker(self.root, executable, relative, manifest, digest, runtime)
        return worker

    def seed_registry(self) -> None:
        manifest, identity = publisher.runtime_manifest(dict(self.runtime_resources))
        runtime = publisher.stage_runtime_package(
            self.root, dict(self.runtime_resources), manifest, identity)
        old = self.stage(self.executable("old", b"old train"), 8, 80,
                         self.old_commit, "train", ["train"], runtime)
        old["worker_rule"] = "historical"
        old["selection_priority"] = 0
        current_train = self.stage(self.executable("current-train", b"current train"),
                                   9, 103, self.current_commit, "train",
                                   ["train"], runtime)
        current_infer = self.stage(self.executable("current-infer", b"current infer"),
                                   9, 103, "9" * 40, "infer", ["infer"], runtime)
        current_infer["artifact_manifest_schema_version"] = (
            publisher.WORKER_MANIFEST_SCHEMA_VERSION)
        # Re-stage the inference artifact under its role-aware manifest path.
        digest = publisher.sha256(self.sources / "current-infer")
        relative, manifest_value, current_infer = rollover._worker_value(
            9, 103, "9" * 40, digest, "infer",
            publisher.WORKER_MANIFEST_SCHEMA_VERSION, ["infer"], runtime["identity"])
        rollover._stage_worker(self.root, self.sources / "current-infer",
                              relative, manifest_value, digest, runtime)
        registry = {"schema_version": publisher.REGISTRY_SCHEMA_VERSION,
                    "current_layout": 9, "runtimes": [runtime],
                    "workers": [old, current_train, current_infer]}
        publisher.validate_existing_registry(self.root, registry)
        (self.root / "registry.json").write_text(publisher.json_text(registry), encoding="utf-8")

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def test_append_preserves_old_binding_and_requires_explicit_qualification(self) -> None:
        before = self.registry()
        old = next(worker for worker in before["workers"]
                   if worker["semantic_layout"] == 8)
        candidate = self.executable("LSTM_Release", b"qualified train")
        with self.assertRaisesRegex(
                publisher.PublishError, "requires explicit qualification"):
            candidate_publisher.append_historical_training_candidate(
                self.root, candidate, 8, 80, self.new_commit,
                ["train", "train_feature_ablation_v1"], 1,
                check_embedded_commit=False,
                runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry(), before)

        staged = candidate_publisher.append_historical_training_candidate(
            self.root, candidate, 8, 80, self.new_commit,
            ["train", "train_feature_ablation_v1"], 1,
            feature_ablation_qualified=True, check_embedded_commit=False,
            runtime_resources=dict(self.runtime_resources))
        after = self.registry()
        self.assertTrue(staged.is_file())
        preserved = next(worker for worker in after["workers"]
                         if worker["semantic_layout"] == 8 and
                         worker["selection_priority"] == 0)
        self.assertEqual(preserved, old)
        qualified = next(worker for worker in after["workers"]
                         if worker["semantic_layout"] == 8 and
                         worker["selection_priority"] == 1)
        self.assertEqual(qualified["capabilities"],
                         ["train", "train_feature_ablation_v1"])

    def test_append_refuses_priority_and_identity_collision(self) -> None:
        candidate = self.executable("LSTM_Release", b"candidate")
        with self.assertRaisesRegex(publisher.PublishError, "priority already registered"):
            candidate_publisher.append_historical_training_candidate(
                self.root, candidate, 8, 80, self.new_commit, ["train"], 0,
                check_embedded_commit=False,
                runtime_resources=dict(self.runtime_resources))
        candidate_publisher.append_historical_training_candidate(
            self.root, candidate, 8, 80, self.new_commit, ["train"], 2,
            check_embedded_commit=False,
            runtime_resources=dict(self.runtime_resources))
        later_directory = self.sources / "later"
        later_directory.mkdir()
        later = later_directory / "LSTM_Release"
        later.write_bytes(b"later candidate")
        later.chmod(0o700)
        with self.assertRaisesRegex(publisher.PublishError, "priority must append"):
            candidate_publisher.append_historical_training_candidate(
                self.root, later, 8, 80, "a" * 40, ["train"], 1,
                check_embedded_commit=False,
                runtime_resources=dict(self.runtime_resources))

    def test_post_staging_registry_failure_is_idempotent_and_collision_safe(self) -> None:
        before_registry = (self.root / "registry.json").read_bytes()
        old = next(worker for worker in self.registry()["workers"]
                   if worker["semantic_layout"] == 8)
        old_executable = self.root / old["executable"]
        old_manifest = old_executable.parent / "manifest.json"
        old_bytes = (old_executable.read_bytes(), old_manifest.read_bytes())
        candidate = self.executable("LSTM_Release", b"retry-safe candidate")
        arguments = dict(
            artifact_root=self.root,
            executable=candidate,
            layout=8,
            width=80,
            commit=self.new_commit,
            capabilities=["train"],
            selection_priority=1,
            check_embedded_commit=False,
            runtime_resources=dict(self.runtime_resources),
        )
        with mock.patch.object(
                publisher, "atomic_write_json",
                side_effect=OSError("injected registry replacement failure")):
            with self.assertRaisesRegex(OSError, "injected registry replacement failure"):
                candidate_publisher.append_historical_training_candidate(**arguments)
        self.assertEqual((self.root / "registry.json").read_bytes(), before_registry)
        self.assertEqual((old_executable.read_bytes(), old_manifest.read_bytes()), old_bytes)

        digest = publisher.sha256(candidate)
        staged = self.root / "layout8" / self.new_commit / digest / "LSTM_Release"
        self.assertTrue(staged.is_file())

        # Same content address plus a different immutable manifest cannot be
        # adopted from the failed publication attempt.
        conflicting = dict(arguments)
        conflicting.update(
            capabilities=["train", "train_feature_ablation_v1"],
            selection_priority=2,
            feature_ablation_qualified=True,
        )
        with self.assertRaisesRegex(publisher.PublishError, "immutable artifact manifest conflict"):
            candidate_publisher.append_historical_training_candidate(**conflicting)
        self.assertEqual((self.root / "registry.json").read_bytes(), before_registry)

        # Exact retry verifies/reuses the staged artifact and atomically binds
        # it, without modifying the old immutable artifact.
        retried = candidate_publisher.append_historical_training_candidate(**arguments)
        self.assertEqual(retried, staged)
        self.assertEqual((old_executable.read_bytes(), old_manifest.read_bytes()), old_bytes)
        published = next(worker for worker in self.registry()["workers"]
                         if worker["semantic_layout"] == 8 and
                         worker["selection_priority"] == 1)
        self.assertEqual(published["sha256"], digest)


if __name__ == "__main__":
    unittest.main()
