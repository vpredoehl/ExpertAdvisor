"""Startup-selected read-only reader; the shared production module is unchanged."""
from __future__ import annotations

import os
from pathlib import Path
from collections.abc import Mapping


ROOT_ENVIRONMENT_VARIABLE = "EXPERTADVISOR_REPOSITORY_ROOT"
DEFAULT_REPOSITORY_ROOT = Path("/Volumes/Developer SSD/ExpertAdvisor")
SOURCE_DIRECTORIES = ("Headers", "Sources", "LSTM")
SOURCE_SUFFIXES = frozenset({".cpp", ".cc", ".cxx", ".h", ".hpp", ".metal"})
# Phase 24B needs these exact operational assertions and target memberships.
# This is not a general Tests, Scripts, or Xcode-project directory permission.
DEVELOPMENT_FILES = frozenset({
    "Tests/DedicatedTrainingWorkerArchitectureTests.sh",
    "Tests/ReleaseWorkerBuildConfigurationTests.sh",
    "Scripts/tests/test_dedicated_train_rollover.py",
    "ExpertAdvisor.xcodeproj/project.pbxproj",
})


MAX_DIRECTORY_ENTRIES = 20_000
MAX_DIRECTORY_DEPTH = 20
MAX_SEARCH_FILES = 2_000
MAX_SEARCH_LINES = 200_000
MAX_SEARCH_BYTES = 32 * 1024 * 1024
MAX_READ_BYTES = 256 * 1024
MAX_LINE_BYTES = 64 * 1024


def resolve_repository_root(environ: Mapping[str, str] | None = None) -> Path:
    environ = os.environ if environ is None else environ
    if ROOT_ENVIRONMENT_VARIABLE not in environ:
        return DEFAULT_REPOSITORY_ROOT
    value = environ[ROOT_ENVIRONMENT_VARIABLE]
    if not value or not Path(value).is_absolute():
        raise ValueError("EXPERTADVISOR_REPOSITORY_ROOT must be an absolute directory")
    root = Path(value).resolve(strict=True)
    if not root.is_dir():
        raise ValueError("EXPERTADVISOR_REPOSITORY_ROOT must be a directory")
    return root


def _relative_path(name: str) -> Path:
    if (not isinstance(name, str) or not name or "\\" in name or "\x00" in name
            or Path(name).is_absolute()
            or any(part in {"", ".", ".."} for part in name.split("/"))):
        raise ValueError("file must be a normalized repository-relative path")
    return Path(name)


class RepositorySourceReader:
    def __init__(self, root: Path):
        self.root = root.resolve(strict=True)
        if not self.root.is_dir():
            raise ValueError("repository root must be a directory")
        production_root = DEFAULT_REPOSITORY_ROOT.resolve(strict=False)
        self.development_files = (
            DEVELOPMENT_FILES if self.root != production_root else frozenset()
        )

    def _permitted(self, relative: Path) -> bool:
        return (relative.as_posix() in self.development_files or
                (relative.parts[0] in SOURCE_DIRECTORIES and
                 relative.suffix.lower() in SOURCE_SUFFIXES))

    def resolve_file(self, name: str) -> Path:
        relative = _relative_path(name)
        if not self._permitted(relative):
            raise ValueError(f"File is outside the read-only source boundary: {name}")
        try:
            path = (self.root / relative).resolve(strict=True)
            resolved_relative = path.relative_to(self.root)
        except (OSError, ValueError) as error:
            raise ValueError(f"File is outside the read-only source boundary: {name}") from error
        # Check both the requested identity and resolved target: an allowed
        # symlink must not expose a disallowed directory within the same root.
        if not path.is_file() or not self._permitted(resolved_relative):
            raise ValueError(f"File is outside the read-only source boundary: {name}")
        return path

    def _bounded_lines(self, path: Path):
        """Read a validated regular file without following a replacement symlink."""
        import stat

        try:
            expected = path.stat()

            flags = os.O_RDONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)

            descriptor = os.open(path, flags)
        except OSError as exc:
            raise ValueError(
                f"cannot open validated source file: {path.name}"
            ) from exc

        try:
            actual = os.fstat(descriptor)

            if (
                not stat.S_ISREG(actual.st_mode)
                or (actual.st_dev, actual.st_ino)
                != (expected.st_dev, expected.st_ino)
            ):
                raise ValueError(
                    f"source file changed during validation: {path.name}"
                )

            with os.fdopen(descriptor, "rb") as source:
                descriptor = -1
                number = 0
                while True:
                    raw = source.readline(MAX_LINE_BYTES + 1)
                    if not raw:
                        break

                    if len(raw) > MAX_LINE_BYTES:
                        raise ValueError(
                            f"source line exceeds {MAX_LINE_BYTES} bytes: {path.name}"
                        )

                    number += 1
                    yield number, raw.decode("utf-8", errors="replace"), len(raw)
        finally:
            if descriptor >= 0:
                os.close(descriptor)

    def list_files(self, prefix: str = "") -> str:
        if prefix:
            _relative_path(prefix)

        candidates = set(self.development_files)
        entries_seen = 0

        for directory in SOURCE_DIRECTORIES:
            base = self.root / directory

            try:
                resolved_path = base.resolve(strict=True)
            except FileNotFoundError:
                continue
            except OSError as exc:
                raise ValueError(
                    f"cannot resolve source directory: {base}"
                ) from exc

            try:
                resolved = resolved_path.relative_to(self.root)
            except ValueError:
                # A symlinked source directory escaping the selected root
                # is not part of the permitted source tree.
                continue

            if (
                not resolved.parts
                or resolved.parts[0] not in SOURCE_DIRECTORIES
                or not base.is_dir()
            ):
                continue

            pending = [(base, 0)]

            while pending:
                current, depth = pending.pop()

                try:
                    with os.scandir(current) as iterator:
                        for entry in iterator:
                            entries_seen += 1

                            if entries_seen > MAX_DIRECTORY_ENTRIES:
                                raise ValueError(
                                    "repository directory-entry budget exceeded"
                                )

                            relative = Path(entry.path).relative_to(self.root)

                            if entry.is_symlink():
                                continue

                            if entry.is_dir(follow_symlinks=False):
                                if depth >= MAX_DIRECTORY_DEPTH:
                                    raise ValueError(
                                        "repository directory-depth budget exceeded"
                                    )

                                pending.append((Path(entry.path), depth + 1))
                                continue

                            if (
                                entry.is_file(follow_symlinks=False)
                                and relative.suffix.lower() in SOURCE_SUFFIXES
                            ):
                                candidates.add(relative.as_posix())

                except OSError as exc:
                    raise ValueError(
                        f"cannot enumerate source directory: {current}"
                    ) from exc

        allowed = []

        for name in sorted(candidates):
            if prefix and not name.startswith(prefix):
                continue

            try:
                self.resolve_file(name)
            except ValueError:
                continue

            allowed.append(name)

        return "\n".join(allowed)

    def read_file(self, name: str, start: int = 1, end: int = 200) -> str:
        path = self.resolve_file(name)

        start = max(1, int(start))
        end = min(max(start, int(end)), start + 499)

        output = []
        bytes_read = 0
        bytes_returned = 0

        for number, line, size in self._bounded_lines(path):
            bytes_read += size

            if bytes_read > MAX_SEARCH_BYTES:
                raise ValueError("read-file input byte budget exceeded")

            if number >= start:
                rendered = f"{number:6d} | {line.rstrip()}"
                encoded_size = len(rendered.encode("utf-8"))

                if output:
                    encoded_size += 1

                bytes_returned += encoded_size

                if bytes_returned > MAX_READ_BYTES:
                    raise ValueError("read-file output byte budget exceeded")

                output.append(rendered)

            if number >= end:
                break

        return "\n".join(output)

    def search(self, pattern: str, max_results: int = 100) -> str:
        """Bounded case-insensitive literal search of permitted source files."""
        if not isinstance(pattern, str) or not pattern or len(pattern) > 256:
            raise ValueError("search pattern must contain 1-256 characters")

        if not isinstance(max_results, int) or isinstance(max_results, bool):
            raise ValueError("max_results must be an integer")

        if not 1 <= max_results <= 100:
            raise ValueError("max_results must be between 1 and 100")

        needle = pattern.lower()
        results = []

        scanned_files = 0
        scanned_lines = 0
        scanned_bytes = 0

        names = self.list_files().splitlines()

        for name in names:
            if scanned_files >= MAX_SEARCH_FILES:
                raise ValueError("search file budget exceeded")

            path = self.resolve_file(name)
            scanned_files += 1

            for number, line, size in self._bounded_lines(path):
                scanned_lines += 1
                scanned_bytes += size

                if scanned_lines > MAX_SEARCH_LINES:
                    raise ValueError("search line budget exceeded")

                if scanned_bytes > MAX_SEARCH_BYTES:
                    raise ValueError("search input byte budget exceeded")

                if needle in line.lower():
                    results.append(
                        f"{name}:{number}: {line.rstrip()[:500]}"
                    )

                    if len(results) >= max_results:
                        return "\n".join(results)

        return "\n".join(results)


# Select once at startup. Every controller/index consumer imports these same
# bound functions; no request can switch the repository or widen the boundary.
REPOSITORY_ROOT = resolve_repository_root()
USES_CONFIGURED_ROOT = ROOT_ENVIRONMENT_VARIABLE in os.environ
if USES_CONFIGURED_ROOT:
    _reader = RepositorySourceReader(REPOSITORY_ROOT)
    list_files = _reader.list_files
    read_file = _reader.read_file
    search = _reader.search
else:
    # Preserve the existing production reader's behavior and test-fixture API.
    from expertadvisor_agent import list_files, read_file, search
