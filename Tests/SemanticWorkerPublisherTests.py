#!/usr/bin/env python3

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


SCRIPT = Path(__file__).resolve().parents[1] / "Scripts" / "PublishSemanticWorker.py"
SPEC = importlib.util.spec_from_file_location("semantic_worker_publisher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
publisher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(publisher)


class SemanticWorkerPublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea-publisher-test.")
        self.root = Path(self.temporary.name) / "SemanticWorkers"
        self.sources = Path(self.temporary.name) / "sources"
        self.sources.mkdir()
        self.runtime_resources = {
            "default.metallib": self.sources / "default.metallib",
            "MetaNN_metal.metallib": self.sources / "MetaNN_metal.metallib",
        }
        self.runtime_resources["default.metallib"].write_bytes(b"default-metal")
        self.runtime_resources["MetaNN_metal.metallib"].write_bytes(b"metann-metal")
        self.seed_legacy_training_reference()

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(contents)
        path.chmod(0o700)
        return path

    def seed_legacy_training_reference(self) -> None:
        self.root.mkdir(parents=True)
        self.root = self.root.resolve()
        executable = self.executable("legacy-training", b"legacy-training")
        commit = "a" * 40
        digest = publisher.sha256(executable)
        directory = self.root / "layout7" / commit / digest
        directory.mkdir(parents=True)
        archived = directory / "LSTM_Release"
        archived.write_bytes(executable.read_bytes())
        archived.chmod(0o555)
        (directory / "manifest.json").write_text(json.dumps({
            "schema_version": 1,
            "semantic_layout": 7,
            "storage": "immutable",
            "model_input_width": 77,
            "source_commit": commit,
            "sha256": digest,
            "executable_identity": "LSTM_Release",
            "capabilities": ["train", "infer", "analyze"],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        manifest, identity = publisher.runtime_manifest(dict(self.runtime_resources))
        runtime = publisher.stage_runtime_package(
            self.root, dict(self.runtime_resources), manifest, identity)
        publisher.install_runtime_links(self.root, directory, runtime)
        (self.root / "registry.json").write_text(json.dumps({
            "schema_version": 2,
            "current_layout": 7,
            "runtimes": [runtime],
            "workers": [{
                "semantic_layout": 7,
                "worker_rule": "current",
                "model_input_width": 77,
                "source_commit": commit,
                "sha256": digest,
                "executable": str(Path("layout7") / commit / digest / "LSTM_Release"),
                "manifest": str(Path("layout7") / commit / digest / "manifest.json"),
                "runtime_identity": identity,
                "capabilities": ["train", "infer", "analyze"],
            }],
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    def publish(self, executable: Path, rule: str, layout: int, commit: str) -> Path:
        return publisher.publish(
            artifact_root=self.root,
            executable=executable,
            worker_rule=rule,
            layout=layout,
            width=77,
            commit=commit,
            capabilities=["infer"],
            check_embedded_commit=False,
            runtime_resources=dict(self.runtime_resources),
        )

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def test_role_publication_preserves_training_reference_and_never_overwrites(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        commit7 = "7" * 40
        archived7 = self.publish(worker7, "current", 7, commit7)
        original7 = archived7.read_bytes()

        worker6 = self.executable("worker6", b"layout-six")
        archived6 = self.publish(worker6, "historical", 6, "6" * 40)
        self.assertEqual(archived6.read_bytes(), b"layout-six")

        worker8 = self.executable("worker8", b"layout-eight")
        archived8 = self.publish(worker8, "current", 7, "8" * 40)
        registry = self.registry()
        self.assertEqual(registry["current_layout"], 7)
        by_role = {(entry["semantic_layout"], entry["worker_role"]): entry
                   for entry in registry["workers"]}
        self.assertEqual(by_role[7, "infer"]["worker_rule"], "current")
        self.assertEqual(by_role[7, "train"]["worker_rule"], "current")
        self.assertEqual(archived7.read_bytes(), original7)
        self.assertEqual(archived8.read_bytes(), b"layout-eight")
        self.assertEqual((self.root / "current").resolve(), archived8.parent)
        self.assertEqual(registry["schema_version"], 4)
        self.assertEqual(len(registry["runtimes"]), 1)
        runtime_identity = by_role[7, "infer"]["runtime_identity"]
        self.assertEqual(by_role[6, "infer"]["runtime_identity"], runtime_identity)
        self.assertEqual(by_role[7, "train"]["runtime_identity"], runtime_identity)
        for archived in (archived6, archived7, archived8):
            self.assertTrue((archived.parent / "default.metallib").is_symlink())
            self.assertTrue((archived.parent / "MetaNN.metallib").is_symlink())
            self.assertEqual(
                (archived.parent / "default.metallib").resolve(),
                (self.root / "runtime" / runtime_identity /
                 "default.metallib").resolve(),
            )

        archived7.chmod(0o700)
        archived7.write_bytes(b"tampered-in-place")
        archived7.chmod(0o700)
        with self.assertRaisesRegex(publisher.PublishError, "immutable artifact hash conflict"):
            self.publish(worker7, "historical", 7, commit7)
        self.assertEqual(archived7.read_bytes(), b"tampered-in-place")

    def test_registry_replacement_failure_leaves_previous_registry_valid(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        self.publish(worker7, "current", 7, "7" * 40)
        before = (self.root / "registry.json").read_bytes()
        worker8 = self.executable("worker8", b"layout-eight")
        with mock.patch.object(
            publisher,
            "atomic_write_json",
            side_effect=OSError("injected registry replace failure"),
        ):
            with self.assertRaisesRegex(OSError, "injected registry replace failure"):
                self.publish(worker8, "current", 7, "8" * 40)
        self.assertEqual((self.root / "registry.json").read_bytes(), before)
        self.assertEqual(self.registry()["current_layout"], 7)

    def test_invalid_candidate_never_becomes_current(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        self.publish(worker7, "current", 7, "7" * 40)
        before = (self.root / "registry.json").read_bytes()
        missing = self.sources / "missing"
        with self.assertRaises(publisher.PublishError):
            self.publish(missing, "current", 8, "8" * 40)
        self.assertEqual((self.root / "registry.json").read_bytes(), before)

    def test_invalid_outgoing_registry_artifact_blocks_rollover(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        archived7 = self.publish(worker7, "current", 7, "7" * 40)
        before = (self.root / "registry.json").read_bytes()
        archived7.unlink()
        worker8 = self.executable("worker8", b"layout-eight")
        with self.assertRaisesRegex(
            publisher.PublishError, "immutable artifact path is incomplete"
        ):
            self.publish(worker8, "current", 7, "8" * 40)
        self.assertEqual((self.root / "registry.json").read_bytes(), before)

    def test_aliased_outgoing_registry_artifact_blocks_rollover(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        archived7 = self.publish(worker7, "current", 7, "7" * 40)
        before = (self.root / "registry.json").read_bytes()
        aliased_target = archived7.with_name("worker-alias-target")
        archived7.rename(aliased_target)
        archived7.symlink_to(aliased_target.name)
        worker8 = self.executable("worker8", b"layout-eight")
        with self.assertRaisesRegex(
            publisher.PublishError, "immutable artifact path is not canonical"
        ):
            self.publish(worker8, "current", 8, "8" * 40)
        self.assertEqual((self.root / "registry.json").read_bytes(), before)

    def test_missing_runtime_resource_never_publishes_worker(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        self.runtime_resources["MetaNN_metal.metallib"].unlink()
        with self.assertRaisesRegex(
            publisher.PublishError,
            "required semantic worker runtime resource is missing: MetaNN_metal.metallib",
        ):
            self.publish(worker7, "current", 7, "7" * 40)
        self.assertEqual(self.registry()["schema_version"], 2)

    def test_runtime_tamper_blocks_rollover(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        self.publish(worker7, "current", 7, "7" * 40)
        registry = self.registry()
        identity = registry["workers"][0]["runtime_identity"]
        resource = self.root / "runtime" / identity / "default.metallib"
        resource.chmod(0o600)
        resource.write_bytes(b"tampered")
        worker8 = self.executable("worker8", b"layout-eight")
        with self.assertRaisesRegex(
            publisher.PublishError,
            "runtime resource hash conflict: default.metallib",
        ):
            self.publish(worker8, "current", 7, "8" * 40)

    def test_v1_registry_upgrade_keeps_existing_executable_identity(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        commit7 = "7" * 40
        digest7 = publisher.sha256(worker7)
        relative7 = Path("layout7") / commit7 / digest7
        archived7 = self.root / relative7 / "LSTM_Release"
        archived7.parent.mkdir(parents=True)
        archived7.write_bytes(worker7.read_bytes())
        archived7.chmod(0o555)
        manifest7 = {
            "schema_version": 1,
            "semantic_layout": 7,
            "storage": "immutable",
            "model_input_width": 77,
            "source_commit": commit7,
            "sha256": digest7,
            "executable_identity": "LSTM_Release",
            "capabilities": ["train", "infer", "analyze"],
        }
        (archived7.parent / "manifest.json").write_text(
            json.dumps(manifest7, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        registry_v1 = {
            "schema_version": 1,
            "current_layout": 7,
            "workers": [{
                "semantic_layout": 7,
                "worker_rule": "current",
                "model_input_width": 77,
                "source_commit": commit7,
                "sha256": digest7,
                "executable": str(relative7 / "LSTM_Release"),
                "manifest": str(relative7 / "manifest.json"),
                "capabilities": ["train", "infer", "analyze"],
            }],
        }
        (self.root / "registry.json").write_text(
            json.dumps(registry_v1, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

        original = archived7.read_bytes()
        worker8 = self.executable("worker8", b"layout-eight")
        self.publish(worker8, "current", 7, "8" * 40)
        upgraded = self.registry()
        by_layout = {entry["semantic_layout"]: entry
                     for entry in upgraded["workers"]}
        self.assertEqual(upgraded["schema_version"], 4)
        self.assertEqual(archived7.read_bytes(), original)
        self.assertEqual(by_layout[7]["sha256"], digest7)
        self.assertTrue((archived7.parent / "default.metallib").is_symlink())
        self.assertTrue((archived7.parent / "MetaNN.metallib").is_symlink())


if __name__ == "__main__":
    unittest.main()
