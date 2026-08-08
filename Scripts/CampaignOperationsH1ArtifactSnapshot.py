#!/usr/bin/env python3
"""Descriptor-bound immutable snapshots for Phase H H1 evidence artifacts.

The evidence root is opened once.  Every directory component and file is then
opened relative to an already trusted directory descriptor with no-follow
semantics.  Metadata, bytes, and the digest come from the same file descriptor;
all later consumers use the private in-memory snapshot rather than a pathname.
"""
from __future__ import annotations

import csv
import hashlib
import io
import os
import stat
import unicodedata
from dataclasses import dataclass
from pathlib import PurePosixPath
from pathlib import Path
from typing import Callable, Iterator


class SnapshotError(RuntimeError):
    def __init__(self, code: str, key: str, detail: str, stage: str):
        super().__init__(detail)
        self.code, self.key, self.detail, self.stage = code, key, detail, stage


def _error(code: str, key: str, detail: str, stage: str) -> None:
    raise SnapshotError(code, key, detail, stage)


def canonical_relative(value: str, artifact_id: str) -> PurePosixPath:
    if (not value or value.startswith("/") or "\\" in value or "//" in value or
            unicodedata.normalize("NFC", value) != value):
        _error("301", artifact_id, "unsupported-path-normalization", "filesystem-representation-validation")
    pure = PurePosixPath(value)
    if str(pure) != value or any(part in {"", ".", ".."} for part in pure.parts):
        _error("301", artifact_id, "non-canonical-relative-path", "filesystem-representation-validation")
    if any(part.endswith((" ", ".")) for part in pure.parts):
        _error("301", artifact_id, "trailing-space-or-dot-alias", "filesystem-representation-validation")
    return pure


@dataclass(frozen=True)
class ArtifactSnapshot:
    artifact_id: str
    lexical_path: str
    run_id: str
    device: int
    inode: int
    file_type: str
    link_count: int
    size: int
    digest: str
    snapshot_id: str
    data: bytes

    def text(self, errors: str = "strict") -> str:
        return self.data.decode("utf-8", errors=errors)

    def tsv(self) -> tuple[list[str], list[dict[str, str]]]:
        rows = list(csv.reader(io.StringIO(self.text()), delimiter="\t"))
        if not rows or len(set(rows[0])) != len(rows[0]):
            _error("312", self.artifact_id, "invalid-snapshot-tsv-header", "snapshot-parse-validation")
        result: list[dict[str, str]] = []
        for number, values in enumerate(rows[1:], 2):
            if len(values) != len(rows[0]):
                _error("312", f"{self.artifact_id}:{number}", "snapshot-tsv-field-count", "snapshot-parse-validation")
            result.append(dict(zip(rows[0], values)))
        return rows[0], result


class ArtifactSnapshotSet:
    """One immutable capture of all registered evidence files present in a root."""

    def __init__(self, root: os.PathLike[str] | str, artifacts: dict[str, dict[str, str]], run_id: str,
                 generation: bool = False, opaque_directories: set[str] | None = None,
                 before_read_hook: Callable[[str], None] | None = None):
        self.root = os.fspath(root)
        self.run_id = run_id
        self.generation = generation
        self.opaque_directories = opaque_directories or set()
        self.before_read_hook = before_read_hook
        self._snapshots_by_id: dict[str, ArtifactSnapshot] = {}
        self._snapshots_by_path: dict[str, ArtifactSnapshot] = {}
        self._present: set[str] = set()
        self._artifacts = artifacts
        self._capture()

    def __contains__(self, lexical_path: object) -> bool:
        return isinstance(lexical_path, str) and lexical_path in self._present

    def __iter__(self) -> Iterator[str]:
        return iter(self._present)

    @property
    def present(self) -> set[str]:
        return set(self._present)

    def by_id(self, artifact_id: str) -> ArtifactSnapshot:
        try:
            return self._snapshots_by_id[artifact_id]
        except KeyError:
            _error("313", artifact_id, "artifact-not-in-snapshot", "snapshot-read-validation")

    def by_path(self, lexical_path: str) -> ArtifactSnapshot:
        try:
            return self._snapshots_by_path[lexical_path]
        except KeyError:
            _error("313", lexical_path, "path-not-in-snapshot", "snapshot-read-validation")

    def inventory_rows(self) -> list[dict[str, str]]:
        return [{
            "artifact_id": item.artifact_id, "lexical_path": item.lexical_path,
            "device": str(item.device), "inode": str(item.inode), "file_type": item.file_type,
            "link_count": str(item.link_count), "size": str(item.size), "digest": item.digest,
            "snapshot_id": item.snapshot_id, "run_id": item.run_id,
        } for item in sorted(self._snapshots_by_id.values(), key=lambda value: value.artifact_id)]

    def assert_run_id(self, run_id: str) -> None:
        if run_id != self.run_id or any(item.run_id != run_id for item in self._snapshots_by_id.values()):
            _error("315", run_id, "stale-snapshot-run-identity", "snapshot-identity-validation")

    @staticmethod
    def _flags(directory: bool = False) -> int:
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        return flags | (getattr(os, "O_DIRECTORY", 0) if directory else 0)

    @staticmethod
    def _identity(metadata: os.stat_result) -> tuple[int, int]:
        return metadata.st_dev, metadata.st_ino

    def _verify_binding(self, root_fd: int, parts: tuple[str, ...], directory_identities: list[tuple[int, int]],
                        file_identity: tuple[int, int], artifact_id: str) -> None:
        current = os.dup(root_fd)
        try:
            for index, part in enumerate(parts[:-1]):
                next_fd = os.open(part, self._flags(directory=True), dir_fd=current)
                os.close(current)
                current = next_fd
                if self._identity(os.fstat(current)) != directory_identities[index]:
                    _error("314", artifact_id, "parent-directory-identity-changed", "snapshot-identity-validation")
            metadata = os.stat(parts[-1], dir_fd=current, follow_symlinks=False)
            if not stat.S_ISREG(metadata.st_mode) or self._identity(metadata) != file_identity:
                _error("314", artifact_id, "registered-path-identity-changed", "snapshot-identity-validation")
        except SnapshotError:
            raise
        except OSError:
            _error("314", artifact_id, "registered-path-binding-lost", "snapshot-identity-validation")
        finally:
            os.close(current)

    def _capture_file(self, root_fd: int, parent_fd: int, relative: str, artifact_id: str,
                      parts: tuple[str, ...], directory_identities: list[tuple[int, int]],
                      identities: dict[tuple[int, int], tuple[str, str]]) -> None:
        try:
            descriptor = os.open(parts[-1], self._flags(), dir_fd=parent_fd)
        except OSError:
            _error("305", relative, "artifact-open-no-follow-failed", "filesystem-representation-validation")
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                _error("308", relative, "unsupported-entry-type", "unsupported-entry-validation")
            identity = self._identity(before)
            if before.st_nlink != 1:
                _error("310", artifact_id, f"unsupported-hard-link-count:{before.st_nlink}",
                       "physical-file-identity-validation")
            if identity in identities:
                other_id, other_path = identities[identity]
                _error("310", artifact_id, f"duplicate-physical-file:{other_id}:{other_path}",
                       "physical-file-identity-validation")
            if self.before_read_hook is not None:
                self.before_read_hook(relative)
            chunks: list[bytes] = []
            while True:
                chunk = os.read(descriptor, 1024 * 1024)
                if not chunk:
                    break
                chunks.append(chunk)
            data = b"".join(chunks)
            after = os.fstat(descriptor)
            if (self._identity(after) != identity or after.st_size != before.st_size or
                    after.st_mtime_ns != before.st_mtime_ns or len(data) != before.st_size):
                _error("314", artifact_id, "descriptor-identity-or-size-changed", "snapshot-identity-validation")
            self._verify_binding(root_fd, parts, directory_identities, identity, artifact_id)
            digest = hashlib.sha256(data).hexdigest()
            snapshot_id = hashlib.sha256(
                f"{self.run_id}\0{artifact_id}\0{relative}\0{before.st_dev}\0{before.st_ino}\0{digest}".encode()
            ).hexdigest()
            item = ArtifactSnapshot(artifact_id, relative, self.run_id, before.st_dev, before.st_ino,
                                    "regular", before.st_nlink, before.st_size, digest, snapshot_id, data)
            identities[identity] = (artifact_id, relative)
            self._snapshots_by_id[artifact_id] = item
            self._snapshots_by_path[relative] = item
            self._present.add(relative)
        finally:
            os.close(descriptor)

    def _capture(self) -> None:
        by_path: dict[str, str] = {}
        expected_dirs = set(self.opaque_directories)
        casefold: dict[str, str] = {}
        for artifact_id, row in self._artifacts.items():
            pure = canonical_relative(row["path"], artifact_id)
            value = pure.as_posix()
            if value in by_path:
                _error("303", artifact_id, f"duplicate-canonical-path:{by_path[value]}",
                       "filesystem-representation-validation")
            folded = value.casefold()
            if folded in casefold and casefold[folded] != value:
                _error("303", artifact_id, f"case-collision:{casefold[folded]}",
                       "filesystem-representation-validation")
            by_path[value] = artifact_id
            casefold[folded] = value
            parent = pure.parent
            while str(parent) != ".":
                expected_dirs.add(parent.as_posix())
                parent = parent.parent

        try:
            root_fd = os.open(self.root, self._flags(directory=True))
        except OSError:
            _error("302", self.root, "invalid-evidence-root", "filesystem-representation-validation")
        identities: dict[tuple[int, int], tuple[str, str]] = {}

        def visit(directory_fd: int, relative_directory: str, ancestor_ids: list[tuple[int, int]]) -> None:
            try:
                names = sorted(os.listdir(directory_fd), key=lambda value: unicodedata.normalize("NFC", value))
            except OSError:
                _error("304", relative_directory, "directory-scan-failed", "directory-policy-validation")
            for name in names:
                relative = name if relative_directory == "." else f"{relative_directory}/{name}"
                if unicodedata.normalize("NFC", relative) != relative:
                    _error("301", relative, "non-canonical-unicode", "filesystem-representation-validation")
                try:
                    metadata = os.stat(name, dir_fd=directory_fd, follow_symlinks=False)
                except OSError:
                    _error("304", relative, "entry-stat-failed", "filesystem-representation-validation")
                if stat.S_ISLNK(metadata.st_mode):
                    detail = "symbolic-link-artifact" if relative in by_path else "symbolic-link-directory-or-entry"
                    _error("305", relative, detail, "filesystem-representation-validation")
                if stat.S_ISDIR(metadata.st_mode):
                    if relative not in expected_dirs:
                        _error("306", relative, "unregistered-directory", "directory-policy-validation")
                    child = os.open(name, self._flags(directory=True), dir_fd=directory_fd)
                    try:
                        if relative not in self.opaque_directories:
                            visit(child, relative, ancestor_ids + [self._identity(os.fstat(child))])
                    finally:
                        os.close(child)
                    continue
                artifact_id = by_path.get(relative)
                if artifact_id is None:
                    _error("309", relative, "unknown-evidence-file", "filesystem-representation-validation")
                parts = tuple(PurePosixPath(relative).parts)
                self._capture_file(root_fd, directory_fd, relative, artifact_id, parts,
                                   ancestor_ids, identities)

        try:
            visit(root_fd, ".", [])
            required = {row["path"] for row in self._artifacts.values()
                        if row["required_phase"] == "base" or
                        (not self.generation and row["required_phase"] == "generated")}
            missing = sorted(required - self._present)
            if missing:
                _error("311", by_path[missing[0]], f"missing-registered-artifact:{missing[0]}",
                       "filesystem-representation-validation")
        finally:
            os.close(root_fd)


def capture_regular_file(path: os.PathLike[str] | str, artifact_id: str, run_id: str,
                         lexical_path: str | None = None) -> ArtifactSnapshot:
    """Capture one regular file without a pathname-based digest or later reopen."""
    value = os.fspath(path)
    flags = ArtifactSnapshotSet._flags()
    try:
        descriptor = os.open(value, flags)
    except OSError:
        _error("305", artifact_id, "artifact-open-no-follow-failed", "filesystem-representation-validation")
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            _error("308", artifact_id, "unsupported-entry-type", "unsupported-entry-validation")
        chunks: list[bytes] = []
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            chunks.append(chunk)
        data = b"".join(chunks)
        after = os.fstat(descriptor)
        if (ArtifactSnapshotSet._identity(before) != ArtifactSnapshotSet._identity(after) or
                before.st_size != after.st_size or before.st_mtime_ns != after.st_mtime_ns or
                len(data) != before.st_size):
            _error("314", artifact_id, "descriptor-identity-or-size-changed", "snapshot-identity-validation")
        try:
            pathname = os.stat(value, follow_symlinks=False)
        except OSError:
            _error("314", artifact_id, "registered-path-binding-lost", "snapshot-identity-validation")
        if (not stat.S_ISREG(pathname.st_mode) or
                ArtifactSnapshotSet._identity(pathname) != ArtifactSnapshotSet._identity(before)):
            _error("314", artifact_id, "registered-path-identity-changed", "snapshot-identity-validation")
        data_digest = hashlib.sha256(data).hexdigest()
        lexical = lexical_path if lexical_path is not None else Path(value).name
        snapshot_id = hashlib.sha256(
            f"{run_id}\0{artifact_id}\0{lexical}\0{before.st_dev}\0{before.st_ino}\0{data_digest}".encode()
        ).hexdigest()
        return ArtifactSnapshot(artifact_id, lexical, run_id, before.st_dev, before.st_ino,
                                "regular", before.st_nlink, before.st_size, data_digest,
                                snapshot_id, data)
    finally:
        os.close(descriptor)
