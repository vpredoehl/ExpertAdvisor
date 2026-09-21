#!/usr/bin/env python3
"""Publish an ordinary, provenance-verified canonical LSTM_Release.

This publisher deliberately owns only Builds/Canonical/LSTM_Release and the
DerivedData/Canonical/LSTM_Release convenience symlink.  Semantic-worker
artifacts and registry.json are outside its responsibility and are never read,
written, or invoked by this script.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile


COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
MANIFEST_SCHEMA_VERSION = 1


class PublishError(RuntimeError):
    pass


def git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repository_root), *arguments],
        check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        text=True)
    if result.returncode:
        raise PublishError("git command failed: " + " ".join(arguments))
    return result.stdout.rstrip("\r\n")


def clean_source_commit(repository_root: Path) -> str:
    if git_output(repository_root, "status", "--porcelain", "--untracked-files=all"):
        raise PublishError("canonical publication requires a clean source tree")
    commit = git_output(repository_root, "rev-parse", "--verify", "HEAD")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise PublishError("HEAD is not an exact lowercase 40-hex commit")
    return commit


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def embedded_commit(executable: Path, commit: str) -> None:
    result = subprocess.run(
        ["/usr/bin/strings", str(executable)], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        errors="replace")
    if result.returncode or commit not in result.stdout.splitlines():
        raise PublishError("built executable does not contain exact source commit " + commit)


def harmless_help(executable: Path) -> None:
    result = subprocess.run(
        [str(executable), "--help"], check=False, stdout=subprocess.PIPE,
        stderr=subprocess.PIPE, text=True, errors="replace")
    if result.returncode:
        raise PublishError("built executable failed harmless --help validation")


def ordinary_manifest(commit: str, digest: str) -> dict[str, object]:
    return {
        "artifact_role": "ordinary-canonical-lstm-release",
        "executable": "LSTM_Release",
        "manifest_schema_version": MANIFEST_SCHEMA_VERSION,
        "sha256": digest,
        "source_commit": commit,
    }


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def write_new_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json_text(value))
        output.flush()
        os.fsync(output.fileno())


def verify_retained_artifact(directory: Path, commit: str, digest: str) -> Path:
    executable = directory / "LSTM_Release"
    manifest = directory / "manifest.json"
    try:
        canonical_directory = directory.resolve(strict=True)
        canonical_executable = executable.resolve(strict=True)
    except OSError as error:
        raise PublishError("retained canonical artifact is incomplete: " + str(directory)) from error
    if (canonical_directory != directory or canonical_executable != executable or
            not executable.is_file() or
            not os.access(executable, os.X_OK)):
        raise PublishError("retained canonical artifact is not canonical: " + str(directory))
    if sha256(executable) != digest:
        raise PublishError("retained canonical artifact hash conflict: " + str(directory))
    try:
        canonical_manifest = manifest.resolve(strict=True)
    except OSError as error:
        raise PublishError("retained canonical artifact is incomplete: " + str(directory)) from error
    if canonical_manifest != manifest:
        raise PublishError("retained canonical artifact is not canonical: " + str(directory))
    try:
        manifest_value = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublishError("retained canonical artifact manifest is invalid") from error
    if manifest_value != ordinary_manifest(commit, digest):
        raise PublishError("retained canonical artifact manifest conflict")
    embedded_commit(executable, commit)
    harmless_help(executable)
    return executable


def validate_existing_canonical_path(canonical_path: Path) -> None:
    if not os.path.lexists(canonical_path):
        return
    if not canonical_path.is_symlink():
        raise PublishError("canonical path is ambiguous: expected a symlink")
    try:
        target = canonical_path.resolve(strict=True)
    except OSError as error:
        raise PublishError("canonical path has a missing target") from error
    if not target.is_file() or not os.access(target, os.X_OK):
        raise PublishError("canonical path target is not an executable file")


def atomically_update_canonical_path(canonical_path: Path, executable: Path) -> None:
    canonical_parent = canonical_path.parent.resolve(strict=True)
    if canonical_path.parent != canonical_parent:
        raise PublishError("canonical path parent is not canonical")
    validate_existing_canonical_path(canonical_path)
    temporary = canonical_parent / ("." + canonical_path.name + ".publish.tmp")
    try:
        temporary.unlink(missing_ok=True)
        temporary.symlink_to(executable)
        os.replace(temporary, canonical_path)
        fsync_directory(canonical_parent)
    finally:
        temporary.unlink(missing_ok=True)


def publish(repository_root: Path, built_executable: Path, artifact_root: Path,
            canonical_path: Path, requested_commit: str | None) -> tuple[Path, str, str]:
    repository_root = repository_root.resolve(strict=True)
    commit = clean_source_commit(repository_root)
    if requested_commit is not None and requested_commit != commit:
        raise PublishError("explicit source commit disagrees with clean HEAD")

    built_executable = built_executable.resolve(strict=True)
    if not built_executable.is_file() or not os.access(built_executable, os.X_OK):
        raise PublishError("built executable must be an executable regular file")
    embedded_commit(built_executable, commit)
    harmless_help(built_executable)
    digest = sha256(built_executable)
    if not SHA256_PATTERN.fullmatch(digest):
        raise PublishError("computed SHA-256 is malformed")

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    canonical_path = canonical_path.absolute()
    canonical_path.parent.mkdir(parents=True, exist_ok=True)
    canonical_path = canonical_path.parent.resolve(strict=True) / canonical_path.name
    final_directory = artifact_root / commit / digest
    final_executable = final_directory / "LSTM_Release"
    lock_path = artifact_root / ".publish.lock"
    with lock_path.open("a+", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if final_directory.exists():
            verify_retained_artifact(final_directory, commit, digest)
        else:
            commit_directory = artifact_root / commit
            commit_directory.mkdir(mode=0o755, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".publish.", dir=commit_directory))
            try:
                staged_executable = stage / "LSTM_Release"
                shutil.copyfile(built_executable, staged_executable)
                os.chmod(staged_executable, 0o555)
                if sha256(staged_executable) != digest:
                    raise PublishError("staged canonical artifact hash mismatch")
                embedded_commit(staged_executable, commit)
                harmless_help(staged_executable)
                write_new_json(stage / "manifest.json", ordinary_manifest(commit, digest))
                os.chmod(stage / "manifest.json", 0o444)
                fsync_directory(stage)
                try:
                    os.rename(stage, final_directory)
                except FileExistsError:
                    verify_retained_artifact(final_directory, commit, digest)
                else:
                    os.chmod(final_directory, 0o555)
                    fsync_directory(commit_directory)
            finally:
                if stage.exists():
                    shutil.rmtree(stage)
        retained = verify_retained_artifact(final_directory, commit, digest)
        atomically_update_canonical_path(canonical_path, retained)
        resolved = canonical_path.resolve(strict=True)
        if resolved != retained or sha256(resolved) != digest:
            raise PublishError("canonical path verification failed")
        embedded_commit(resolved, commit)
        harmless_help(resolved)
    return retained, commit, digest


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", type=Path, required=True)
    parser.add_argument("--built-executable", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--canonical-path", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    repository_root = arguments.repository_root.resolve(strict=True)
    artifact_root = arguments.artifact_root or (
        repository_root / "Builds" / "Canonical" / "LSTM_Release")
    canonical_path = arguments.canonical_path or (
        repository_root / "DerivedData" / "Canonical" / "LSTM_Release")
    retained, commit, digest = publish(
        repository_root, arguments.built_executable, artifact_root,
        canonical_path, arguments.source_commit)
    print("Canonical LSTM_Release published: " + str(retained))
    print("source_commit=" + commit)
    print("sha256=" + digest)
    print("canonical_path=" + str(canonical_path.absolute()))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, PublishError, subprocess.SubprocessError) as error:
        raise SystemExit("PublishCanonicalLSTMRelease.py: " + str(error))
