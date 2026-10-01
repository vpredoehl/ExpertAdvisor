#!/usr/bin/env python3

"""Disposable contract tests for legacy historical INFER publication."""

from __future__ import annotations

import argparse
import contextlib
import importlib.util
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock


REPOSITORY = Path(__file__).resolve().parents[1]
SCRIPT = REPOSITORY / "Scripts" / "PublishHistoricalInferenceWorker.py"
SPEC = importlib.util.spec_from_file_location("historical_inference_publisher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
historical = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(historical)
publisher = historical.publisher
rollover = historical.rollover


class SemanticWorkerHistoricalInferenceWorkerPublisherTests(unittest.TestCase):
    current_commit = "a" * 40
    historical_commit = "b" * 40

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea-historical-infer-test.")
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
        self.seed_current_registry()
        self.historical_inference = self.executable(
            "LSTM_Release", b"historical inference\n" + self.historical_commit.encode() + b"\n")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(contents)
        path.chmod(0o700)
        return path

    def stage(self, executable: Path, layout: int, width: int, commit: str,
              role: str, schema: int, capabilities: list[str], runtime: dict) -> dict:
        digest = publisher.sha256(executable)
        relative, manifest, worker = rollover._worker_value(
            layout, width, commit, digest, role, schema, capabilities,
            runtime["identity"])
        rollover._stage_worker(self.root, executable, relative, manifest, digest, runtime)
        return worker

    def seed_current_registry(self) -> None:
        manifest, identity = publisher.runtime_manifest(dict(self.runtime_resources))
        runtime = publisher.stage_runtime_package(
            self.root, dict(self.runtime_resources), manifest, identity)
        train = self.executable("current-train", b"current train\n")
        infer = self.executable("current-infer", b"current infer\n")
        train_worker = self.stage(
            train, 7, 77, self.current_commit, "train",
            publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
            ["train", "infer", "analyze"], runtime)
        infer_worker = self.stage(
            infer, 7, 77, "c" * 40, "infer",
            publisher.WORKER_MANIFEST_SCHEMA_VERSION, ["infer"], runtime)
        registry = {
            "schema_version": publisher.REGISTRY_SCHEMA_VERSION,
            "current_layout": 7,
            "runtimes": [runtime],
            "workers": [train_worker, infer_worker],
        }
        publisher.validate_existing_registry(self.root, registry)
        (self.root / "registry.json").write_text(
            publisher.json_text(registry), encoding="utf-8")

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def registry_bytes(self) -> bytes:
        return (self.root / "registry.json").read_bytes()

    def publish(self, **kwargs: object) -> Path:
        return historical.publish_historical_inference_worker(
            self.root, self.historical_inference, 5, 64, self.historical_commit,
            runtime_resources=dict(self.runtime_resources), **kwargs)

    def test_bootstraps_legacy_historical_inference_without_build_identity(self) -> None:
        before = self.registry()
        staged = self.publish()
        registry = self.registry()
        worker = next(item for item in registry["workers"]
                      if item["semantic_layout"] == 5 and
                      item["worker_role"] == "infer")
        digest = publisher.sha256(self.historical_inference)
        relative = Path("layout5") / self.historical_commit / digest
        self.assertEqual(staged, self.root / relative / "LSTM_Release")
        self.assertEqual(staged.name, "LSTM_Release")
        self.assertEqual(worker["artifact_manifest_schema_version"],
                         publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(worker["worker_rule"], "historical")
        self.assertEqual(worker["capabilities"], ["infer"])
        self.assertEqual(worker["selection_priority"], 0)
        self.assertEqual(worker["executable"], str(relative / "LSTM_Release"))
        self.assertEqual(registry["current_layout"], before["current_layout"])
        runtime_directory = self.root / "runtime" / worker["runtime_identity"]
        self.assertTrue((staged.parent / "default.metallib").is_symlink())
        self.assertTrue((staged.parent / "MetaNN.metallib").is_symlink())
        self.assertEqual(
            os.readlink(staged.parent / "default.metallib"),
            os.path.relpath(runtime_directory / "default.metallib", staged.parent))
        self.assertEqual(
            os.readlink(staged.parent / "MetaNN.metallib"),
            os.path.relpath(runtime_directory / "MetaNN.metallib", staged.parent))
        self.assertEqual((staged.parent / "default.metallib").resolve(),
                         runtime_directory / "default.metallib")
        self.assertEqual((staged.parent / "MetaNN.metallib").resolve(),
                         runtime_directory / "MetaNN.metallib")
        current_bindings = {
            entry["worker_role"]: entry for entry in registry["workers"]
            if entry["semantic_layout"] == 7
        }
        before_bindings = {
            entry["worker_role"]: entry for entry in before["workers"]
            if entry["semantic_layout"] == 7
        }
        self.assertEqual(current_bindings, before_bindings)
        publisher.validate_existing_registry(self.root, registry)

    def test_duplicate_target_binding_fails_closed(self) -> None:
        self.publish()
        before = self.registry_bytes()
        with self.assertRaisesRegex(publisher.PublishError, "binding already exists"):
            self.publish()
        self.assertEqual(self.registry_bytes(), before)

    def test_cli_summary_records_immutable_publication_and_preserved_current_layout(self) -> None:
        output = io.StringIO()
        with mock.patch.object(
            historical, "parse_arguments",
            return_value=argparse.Namespace(
                artifact_root=self.root,
                inference_executable=self.historical_inference,
                semantic_layout=5,
                model_input_width=64,
                source_commit=self.historical_commit,
            ),
        ), contextlib.redirect_stdout(output):
            self.assertEqual(historical.main(), 0)
        summary = output.getvalue()
        worker = next(entry for entry in self.registry()["workers"]
                      if entry["semantic_layout"] == 5)
        self.assertIn("semantic_layout=5", summary)
        self.assertIn("model_input_width=64", summary)
        self.assertIn("worker_role=infer", summary)
        self.assertIn("worker_rule=historical", summary)
        self.assertIn(f"source_commit={self.historical_commit}", summary)
        self.assertIn(f"executable_sha256={worker['sha256']}", summary)
        self.assertIn(f"runtime_identity={worker['runtime_identity']}", summary)
        self.assertIn("current_layout=7", summary)
        self.assertIn("current_layout_preserved=true", summary)

    def test_invalid_contract_or_artifact_never_changes_registry(self) -> None:
        before = self.registry_bytes()
        cases = [
            (self.historical_inference, 0, 64, self.historical_commit),
            (self.historical_inference, 5, 0, self.historical_commit),
            (self.historical_inference, 5, 64, "B" * 40),
            (Path("LSTM_Release"), 5, 64, self.historical_commit),
            (self.executable("wrong-name", self.historical_commit.encode()), 5, 64,
             self.historical_commit),
            (self.sources / "missing", 5, 64, self.historical_commit),
        ]
        non_executable_directory = self.sources / "non-executable"
        non_executable_directory.mkdir()
        non_executable = non_executable_directory / "LSTM_Release"
        non_executable.write_bytes(self.historical_commit.encode())
        non_executable.chmod(0o600)
        cases.append((non_executable, 5, 64, self.historical_commit))
        for executable, layout, width, commit in cases:
            with self.subTest(executable=executable, layout=layout, width=width):
                with self.assertRaises(publisher.PublishError):
                    historical.publish_historical_inference_worker(
                        self.root, executable, layout, width, commit,
                        runtime_resources=dict(self.runtime_resources))
                self.assertEqual(self.registry_bytes(), before)

    def test_embedded_commit_mismatch_and_registry_failure_preserve_authority(self) -> None:
        before = self.registry_bytes()
        mismatch_directory = self.sources / "mismatch"
        mismatch_directory.mkdir()
        mismatched = mismatch_directory / "LSTM_Release"
        mismatched.write_bytes(b"no matching commit\n")
        mismatched.chmod(0o700)
        with self.assertRaises(publisher.PublishError):
            historical.publish_historical_inference_worker(
                self.root, mismatched, 5, 64, self.historical_commit,
                runtime_resources=dict(self.runtime_resources))
        self.assertEqual(self.registry_bytes(), before)
        with mock.patch.object(
            publisher, "validate_existing_registry",
            side_effect=[None, publisher.PublishError("injected prospective validation failure")],
        ):
            with self.assertRaisesRegex(publisher.PublishError,
                                        "injected prospective validation failure"):
                self.publish()
        self.assertEqual(self.registry_bytes(), before)
        with mock.patch.object(publisher, "atomic_write_json",
                               side_effect=OSError("injected registry failure")):
            with self.assertRaisesRegex(OSError, "injected registry failure"):
                self.publish()
        self.assertEqual(self.registry_bytes(), before)
        staged = self.publish()
        self.assertEqual(staged.read_bytes(), self.historical_inference.read_bytes())

    def test_missing_or_mismatched_runtime_resources_fail_closed(self) -> None:
        before = self.registry_bytes()
        missing_resources = dict(self.runtime_resources)
        missing_resources.pop("MetaNN_metal.metallib")
        with self.assertRaisesRegex(publisher.PublishError, "runtime resource is missing"):
            historical.publish_historical_inference_worker(
                self.root, self.historical_inference, 5, 64,
                self.historical_commit, runtime_resources=missing_resources)
        self.assertEqual(self.registry_bytes(), before)

        isolated_resources_directory = self.sources / "isolated-runtime"
        isolated_resources_directory.mkdir()
        isolated_resources = {
            "default.metallib": isolated_resources_directory / "default.metallib",
            "MetaNN_metal.metallib": isolated_resources_directory / "MetaNN_metal.metallib",
        }
        isolated_resources["default.metallib"].write_bytes(b"isolated-default")
        isolated_resources["MetaNN_metal.metallib"].write_bytes(b"isolated-metann")
        manifest, identity = publisher.runtime_manifest(dict(isolated_resources))
        corrupt_runtime = self.root / "runtime" / identity
        corrupt_runtime.mkdir()
        (corrupt_runtime / "manifest.json").write_text(
            publisher.json_text(manifest), encoding="utf-8")
        (corrupt_runtime / "default.metallib").write_bytes(b"corrupt")
        (corrupt_runtime / "MetaNN.metallib").write_bytes(
            isolated_resources["MetaNN_metal.metallib"].read_bytes())
        with self.assertRaisesRegex(publisher.PublishError, "runtime resource hash conflict"):
            historical.publish_historical_inference_worker(
                self.root, self.historical_inference, 5, 64,
                self.historical_commit, runtime_resources=isolated_resources)
        self.assertEqual(self.registry_bytes(), before)


if __name__ == "__main__":
    unittest.main()
