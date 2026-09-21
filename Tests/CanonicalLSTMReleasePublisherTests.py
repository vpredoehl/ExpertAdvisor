#!/usr/bin/env python3
"""Focused contract tests for ordinary canonical LSTM_Release publication."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import os
import subprocess
import tempfile
import unittest


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PUBLISHER = REPOSITORY_ROOT / "Scripts" / "PublishCanonicalLSTMRelease.py"


class CanonicalLSTMReleasePublisherTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory(prefix="ea_canonical_release.")
        self.root = Path(self.temporary.name) / "source"
        self.root.mkdir()
        self.command("git", "init", "-q")
        self.command("git", "config", "user.email", "test@example.invalid")
        self.command("git", "config", "user.name", "Publisher Test")
        (self.root / ".gitignore").write_text("Build/\nDerivedData/\nBuilds/\n", encoding="utf-8")
        (self.root / "source.txt").write_text("source\n", encoding="utf-8")
        self.command("git", "add", ".gitignore", "source.txt")
        self.command("git", "commit", "-qm", "source")
        self.commit = self.output("git", "rev-parse", "HEAD").strip()
        self.binary = self.root / "Build" / "Products" / "Release" / "LSTM_Release"
        self.write_binary(self.binary, self.commit, "ordinary")
        self.artifacts = self.root / "Builds" / "Canonical" / "LSTM_Release"
        self.canonical = self.root / "DerivedData" / "Canonical" / "LSTM_Release"

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def command(self, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
        return subprocess.run(arguments, cwd=self.root, check=check, text=True,
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def output(self, *arguments: str) -> str:
        return self.command(*arguments).stdout

    @staticmethod
    def write_binary(path: Path, commit: str, label: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        source = path.with_suffix(".c")
        source.write_text(
            "__attribute__((used)) static const char source_commit[] = "
            f"\"{commit}\";\n"
            "__attribute__((used)) static const char build_label[] = "
            f"\"{label}\";\n"
            "int main(void) { return 0; }\n", encoding="utf-8")
        subprocess.run(["/usr/bin/clang", str(source), "-o", str(path)], check=True)

    def publish(self, executable: Path | None = None) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            ["/usr/bin/python3", str(PUBLISHER), "--repository-root", str(self.root),
             "--built-executable", str(executable or self.binary),
             "--artifact-root", str(self.artifacts), "--canonical-path", str(self.canonical)],
            check=False, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def test_publish_retain_republish_and_leave_semantic_registry_untouched(self) -> None:
        registry = self.root / "Builds" / "SemanticWorkers" / "registry.json"
        registry.parent.mkdir(parents=True)
        registry.write_text('{"current_layout":7}\n', encoding="utf-8")
        registry_hash = hashlib.sha256(registry.read_bytes()).hexdigest()

        result = self.publish()
        self.assertEqual(result.returncode, 0, result.stderr)
        digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        retained = (self.artifacts / self.commit / digest / "LSTM_Release").resolve()
        self.assertEqual(self.canonical.resolve(), retained)
        self.assertEqual(hashlib.sha256(retained.read_bytes()).hexdigest(), digest)
        self.assertEqual(
            json.loads((retained.parent / "manifest.json").read_text(encoding="utf-8")),
            {"artifact_role": "ordinary-canonical-lstm-release", "executable": "LSTM_Release",
             "manifest_schema_version": 1, "sha256": digest, "source_commit": self.commit})
        self.assertEqual(self.publish().returncode, 0)
        self.assertEqual(hashlib.sha256(registry.read_bytes()).hexdigest(), registry_hash)

    def test_rejects_dirty_source_and_missing_embedded_identity(self) -> None:
        (self.root / "source.txt").write_text("dirty\n", encoding="utf-8")
        dirty = self.publish()
        self.assertNotEqual(dirty.returncode, 0)
        self.assertIn("clean source tree", dirty.stderr)
        self.command("git", "checkout", "--", "source.txt")
        invalid = self.root / "Build" / "Products" / "Release" / "invalid"
        self.write_binary(invalid, "0" * 40, "invalid")
        malformed = self.publish(invalid)
        self.assertNotEqual(malformed.returncode, 0)
        self.assertIn("does not contain exact source commit", malformed.stderr)

    def test_rejects_retained_collision_and_ambiguous_canonical_path(self) -> None:
        digest = hashlib.sha256(self.binary.read_bytes()).hexdigest()
        collision = self.artifacts / self.commit / digest
        collision.mkdir(parents=True)
        (collision / "LSTM_Release").write_text("different bytes\n", encoding="utf-8")
        (collision / "LSTM_Release").chmod(0o755)
        collision_result = self.publish()
        self.assertNotEqual(collision_result.returncode, 0)
        self.assertIn("hash conflict", collision_result.stderr)
        for child in collision.iterdir():
            child.unlink()
        collision.rmdir()
        self.canonical.parent.mkdir(parents=True, exist_ok=True)
        self.canonical.write_text("not a symlink\n", encoding="utf-8")
        ambiguous = self.publish()
        self.assertNotEqual(ambiguous.returncode, 0)
        self.assertIn("canonical path is ambiguous", ambiguous.stderr)


if __name__ == "__main__":
    unittest.main()
