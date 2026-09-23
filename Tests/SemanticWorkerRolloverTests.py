#!/usr/bin/env python3

"""Disposable failure-injection tests for coordinated semantic layout rollover."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "Scripts" / "RollSemanticWorkerLayout.py"
SPEC = importlib.util.spec_from_file_location("semantic_worker_rollover", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
rollover = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rollover)
publisher = rollover.publisher


class SemanticWorkerRolloverTests(unittest.TestCase):
    commit6 = "6" * 40
    commit7_train = "7" * 40
    commit7_infer = "8" * 40
    commit8 = "9" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea-rollover-test.")
        self.base = Path(self.temporary.name)
        self.root = self.base / "SemanticWorkers"
        self.sources = self.base / "products"
        self.sources.mkdir()
        self.runtime_resources = {
            "default.metallib": self.sources / "default.metallib",
            "MetaNN_metal.metallib": self.sources / "MetaNN_metal.metallib",
        }
        self.runtime_resources["default.metallib"].write_bytes(b"default-metal")
        self.runtime_resources["MetaNN_metal.metallib"].write_bytes(b"metann-metal")
        self.root.mkdir()
        self.root = self.root.resolve()
        self.seed_layout7_registry()
        self.training8 = self.executable("LSTM_Release", b"layout-eight-training\n")
        self.inference8 = self.executable("lstm-infer-worker", b"layout-eight-infer\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes) -> Path:
        executable = self.sources / name
        executable.write_bytes(contents)
        executable.chmod(0o700)
        return executable

    def stage(self, executable: Path, layout: int, width: int, commit: str,
              role: str, manifest_schema: int, capabilities: list[str],
              runtime: dict) -> dict:
        digest = publisher.sha256(executable)
        relative, manifest, worker = rollover._worker_value(
            layout, width, commit, digest, role, manifest_schema,
            capabilities, runtime["identity"])
        rollover._stage_worker(self.root, executable, relative, manifest, digest, runtime)
        return worker

    def seed_layout7_registry(self) -> None:
        manifest, identity = publisher.runtime_manifest(dict(self.runtime_resources))
        runtime = publisher.stage_runtime_package(
            self.root, dict(self.runtime_resources), manifest, identity)
        layout6 = self.executable("legacy-layout6", b"layout-six-infer\n")
        layout7_train = self.executable("legacy-layout7-train", b"layout-seven-train\n")
        layout7_infer = self.executable("legacy-layout7-infer", b"layout-seven-infer\n")
        worker6 = self.stage(
            layout6, 6, 77, self.commit6, "infer",
            publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION, ["infer"], runtime)
        worker6["worker_rule"] = "historical"
        worker7_train = self.stage(
            layout7_train, 7, 77, self.commit7_train, "train",
            publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
            rollover.TRAINING_CAPABILITIES, runtime)
        worker7_infer = self.stage(
            layout7_infer, 7, 77, self.commit7_infer, "infer",
            publisher.WORKER_MANIFEST_SCHEMA_VERSION,
            rollover.INFERENCE_CAPABILITIES, runtime)
        registry = {
            "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
            "current_layout": 7,
            "runtimes": [runtime],
            "workers": [worker6, worker7_train, worker7_infer],
        }
        publisher.validate_existing_registry(self.root, registry)
        (self.root / "registry.json").write_text(
            publisher.json_text(registry), encoding="utf-8")

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def registry_bytes(self) -> bytes:
        return (self.root / "registry.json").read_bytes()

    def publish_rollover(self) -> tuple[Path, Path]:
        return rollover.rollover(
            self.root, self.training8, self.inference8, 8, 80, self.commit8,
            check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))

    def test_successful_rollover_has_exact_role_layout_matrix(self) -> None:
        training, inference = self.publish_rollover()
        registry = self.registry()
        self.assertEqual(registry["current_layout"], 8)
        by_role = {(entry["semantic_layout"], entry["worker_role"]): entry
                   for entry in registry["workers"]}
        self.assertEqual(set(by_role), {(6, "infer"), (7, "train"),
                                        (7, "infer"), (8, "train"),
                                        (8, "infer")})
        self.assertEqual(by_role[6, "infer"]["worker_rule"], "historical")
        self.assertEqual(by_role[7, "train"]["worker_rule"], "historical")
        self.assertEqual(by_role[7, "infer"]["worker_rule"], "historical")
        self.assertEqual(by_role[8, "train"]["worker_rule"], "current")
        self.assertEqual(by_role[8, "infer"]["worker_rule"], "current")
        self.assertEqual(by_role[7, "train"]["model_input_width"], 77)
        self.assertEqual(by_role[8, "train"]["model_input_width"], 80)
        self.assertEqual(by_role[8, "train"]["capabilities"],
                         rollover.TRAINING_CAPABILITIES)
        self.assertEqual(by_role[8, "infer"]["capabilities"],
                         rollover.INFERENCE_CAPABILITIES)
        self.assertEqual(training.name, "LSTM_Release")
        self.assertEqual(inference.name, "lstm-infer-worker")
        self.assertEqual((self.root / "current").resolve(), inference.parent)
        for entry in by_role.values():
            directory = self.root / entry["executable"]
            self.assertTrue(directory.is_file())
            self.assertTrue((directory.parent / "default.metallib").is_symlink())
            self.assertTrue((directory.parent / "MetaNN.metallib").is_symlink())

    def test_inference_only_current_advance_is_still_rejected(self) -> None:
        before = self.registry_bytes()
        candidate = self.executable("infer-only", b"infer-only\n")
        with self.assertRaisesRegex(
            publisher.PublishError,
            "inference publication cannot change current layout without a training/reference binding",
        ):
            publisher.publish(
                self.root, candidate, "current", 8, 80, self.commit8,
                ["infer"], check_embedded_commit=False,
                runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry_bytes(), before)

    def test_registry_replace_failure_leaves_layout7_byte_identical(self) -> None:
        before = self.registry_bytes()
        with mock.patch.object(
            publisher, "atomic_write_json",
            side_effect=OSError("injected registry replacement failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected registry replacement failure"):
                self.publish_rollover()
        self.assertEqual(self.registry_bytes(), before)
        self.assertEqual(self.registry()["current_layout"], 7)

    def test_missing_or_wrong_role_candidate_never_stages_registry_change(self) -> None:
        before = self.registry_bytes()
        missing = self.sources / "missing"
        with self.assertRaises(publisher.PublishError):
            rollover.rollover(
                self.root, self.training8, missing, 8, 80, self.commit8,
                check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        wrong_role = self.executable("wrong-worker", b"wrong\n")
        with self.assertRaisesRegex(publisher.PublishError, "expected lstm-infer-worker"):
            rollover.rollover(
                self.root, self.training8, wrong_role, 8, 80, self.commit8,
                check_embedded_commit=False, runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry_bytes(), before)

    def test_outgoing_artifact_and_runtime_corruption_block_rollover(self) -> None:
        before = self.registry_bytes()
        old = self.registry()["workers"][0]
        (self.root / old["executable"]).unlink()
        with self.assertRaisesRegex(publisher.PublishError, "immutable artifact path is incomplete"):
            self.publish_rollover()
        self.assertEqual(self.registry_bytes(), before)

    def test_corrupt_prior_runtime_blocks_rollover(self) -> None:
        before = self.registry_bytes()
        runtime = self.registry()["runtimes"][0]
        resource = self.root / runtime["directory"] / "default.metallib"
        resource.chmod(0o600)
        resource.write_bytes(b"tampered-runtime")
        with self.assertRaisesRegex(
            publisher.PublishError, "existing semantic worker runtime resource hash conflict",
        ):
            self.publish_rollover()
        self.assertEqual(self.registry_bytes(), before)

    def test_staged_artifact_conflict_cannot_change_registry(self) -> None:
        before = self.registry_bytes()
        digest = publisher.sha256(self.training8)
        conflict = self.root / "layout8" / self.commit8 / digest
        conflict.mkdir(parents=True)
        (conflict / "LSTM_Release").write_bytes(b"not-the-candidate")
        (conflict / "LSTM_Release").chmod(0o555)
        (conflict / "manifest.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "immutable artifact hash conflict"):
            self.publish_rollover()
        self.assertEqual(self.registry_bytes(), before)

    def test_target_layout_collision_and_malformed_prior_registry_are_rejected(self) -> None:
        before = self.registry_bytes()
        registry = self.registry()
        duplicate = dict(registry["workers"][0])
        duplicate["semantic_layout"] = 8
        registry["workers"].append(duplicate)
        (self.root / "registry.json").write_text(publisher.json_text(registry), encoding="utf-8")
        with self.assertRaises(publisher.PublishError):
            self.publish_rollover()
        (self.root / "registry.json").write_bytes(before)
        (self.root / "registry.json").write_text("{malformed", encoding="utf-8")
        with self.assertRaisesRegex(publisher.PublishError, "existing semantic worker registry is malformed"):
            self.publish_rollover()

    def test_clean_source_and_build_identity_preflight_are_required(self) -> None:
        with mock.patch.object(
            publisher, "clean_source_commit",
            side_effect=publisher.PublishError("semantic worker publication requires a clean source tree"),
        ):
            with self.assertRaisesRegex(publisher.PublishError, "clean source tree"):
                rollover.rollover_from_repository(
                    REPOSITORY, self.training8, self.inference8, self.root)
        with mock.patch.object(publisher, "verify_embedded_commit") as embedded:
            with mock.patch.object(
                publisher, "verify_inference_build_identity",
                side_effect=publisher.PublishError("inference worker build identity mismatch"),
            ):
                with self.assertRaisesRegex(publisher.PublishError, "build identity mismatch"):
                    rollover.rollover(
                        self.root, self.training8, self.inference8, 8, 80,
                        self.commit8, check_embedded_commit=True)
        self.assertGreaterEqual(embedded.call_count, 2)

    def test_explicit_source_commit_disagreement_is_rejected_before_staging(self) -> None:
        before = self.registry_bytes()
        with mock.patch.object(publisher, "clean_source_commit", return_value=self.commit8):
            with self.assertRaisesRegex(
                publisher.PublishError, "explicit source commit disagrees with clean HEAD",
            ):
                rollover.rollover_from_repository(
                    REPOSITORY, self.training8, self.inference8, self.root,
                    source_commit="a" * 40)
        self.assertEqual(self.registry_bytes(), before)

    def test_cpp_registry_resolves_historical_and_current_roles_exactly(self) -> None:
        self.publish_rollover()
        binary = self.base / "SemanticWorkerRegistryTests"
        subprocess.run([
            "clang++", "-std=c++20", "-Wall", "-Wextra", "-Werror",
            "-I", str(REPOSITORY / "Headers"), "-I", str(REPOSITORY / "Sources"),
            str(REPOSITORY / "Tests" / "SemanticWorkerRegistryTests.cpp"),
            str(REPOSITORY / "Sources" / "SchedulerCore" / "SemanticWorkerRegistry.cpp"),
            "-o", str(binary),
        ], check=True)
        environment = dict(os.environ)
        environment["EA_SEMANTIC_REGISTRY_ROLLOVER_UNDER_TEST"] = str(
            self.root / "registry.json")
        subprocess.run([str(binary)], check=True, env=environment)


if __name__ == "__main__":
    unittest.main()
