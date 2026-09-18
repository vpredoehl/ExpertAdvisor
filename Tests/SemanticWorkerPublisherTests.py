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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def executable(self, name: str, contents: bytes) -> Path:
        path = self.sources / name
        path.write_bytes(contents)
        path.chmod(0o700)
        return path

    def publish(self, executable: Path, rule: str, layout: int, commit: str) -> Path:
        return publisher.publish(
            artifact_root=self.root,
            executable=executable,
            worker_rule=rule,
            layout=layout,
            width=77,
            commit=commit,
            capabilities=(
                ["train", "infer", "analyze"] if rule == "current"
                else ["infer"]
            ),
            check_embedded_commit=False,
        )

    def registry(self) -> dict:
        return json.loads((self.root / "registry.json").read_text(encoding="utf-8"))

    def test_rollover_preserves_prior_layout_and_never_overwrites(self) -> None:
        worker7 = self.executable("worker7", b"layout-seven")
        commit7 = "7" * 40
        archived7 = self.publish(worker7, "current", 7, commit7)
        original7 = archived7.read_bytes()

        worker6 = self.executable("worker6", b"layout-six")
        archived6 = self.publish(worker6, "historical", 6, "6" * 40)
        self.assertEqual(archived6.read_bytes(), b"layout-six")

        worker8 = self.executable("worker8", b"layout-eight")
        archived8 = self.publish(worker8, "current", 8, "8" * 40)
        registry = self.registry()
        self.assertEqual(registry["current_layout"], 8)
        by_layout = {entry["semantic_layout"]: entry for entry in registry["workers"]}
        self.assertEqual(by_layout[7]["worker_rule"], "historical")
        self.assertEqual(by_layout[8]["worker_rule"], "current")
        self.assertEqual(archived7.read_bytes(), original7)
        self.assertEqual(archived8.read_bytes(), b"layout-eight")
        self.assertEqual((self.root / "current").resolve(), archived8.parent)

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
                self.publish(worker8, "current", 8, "8" * 40)
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
            self.publish(worker8, "current", 8, "8" * 40)
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


if __name__ == "__main__":
    unittest.main()
