#!/usr/bin/env python3
"""Publish immutable semantic workers and atomically replace their registry."""

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


REGISTRY_SCHEMA_VERSION = 4
ROLE_AWARE_REGISTRY_SCHEMA_VERSION = 3
LEGACY_REGISTRY_SCHEMA_VERSION = 2
LEGACY_WORKER_MANIFEST_SCHEMA_VERSION = 1
WORKER_MANIFEST_SCHEMA_VERSION = 2
RUNTIME_MANIFEST_SCHEMA_VERSION = 1
COMMIT_PATTERN = re.compile(r"[0-9a-f]{40}\Z")
SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
VALID_CAPABILITIES = frozenset({"train", "infer", "analyze"})
RUNTIME_RESOURCE_SPECS = (
    ("MetaNN_metal.metallib", "MetaNN.metallib"),
    ("default.metallib", "default.metallib"),
)


class PublishError(RuntimeError):
    pass


def git_output(repository_root: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["/usr/bin/git", "-C", str(repository_root), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout.rstrip("\r\n")


def clean_source_commit(repository_root: Path) -> str:
    if git_output(repository_root, "status", "--porcelain"):
        raise PublishError("semantic worker publication requires a clean source tree")
    commit = git_output(repository_root, "rev-parse", "--verify", "HEAD")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise PublishError("HEAD is not an exact lowercase 40-hex commit")
    return commit


def current_semantic_contract(repository_root: Path) -> tuple[int, int]:
    source = (
        '#include "ModelInputExpansion.hpp"\n'
        '#include <iostream>\n'
        'int main() { std::cout << EA::kModelInputSemanticLayoutVersion << " " '
        '<< EA::kCurrentModelInputWidth << "\\n"; }\n'
    )
    with tempfile.TemporaryDirectory(prefix="ea-semantic-contract.") as temporary:
        temporary_path = Path(temporary)
        source_path = temporary_path / "contract.cpp"
        executable_path = temporary_path / "contract"
        source_path.write_text(source, encoding="utf-8")
        subprocess.run(
            ["/usr/bin/xcrun", "clang++", "-std=c++20",
             "-I", str(repository_root / "Headers"),
             str(source_path), "-o", str(executable_path)],
            check=True,
        )
        fields = subprocess.check_output([str(executable_path)], text=True).split()
    if len(fields) != 2:
        raise PublishError("current semantic contract probe returned malformed output")
    layout, width = (int(value) for value in fields)
    if layout <= 0 or width <= 0:
        raise PublishError("current semantic contract is invalid")
    return layout, width


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def fsync_file(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json(path: Path, value: object) -> None:
    with path.open("x", encoding="utf-8") as output:
        output.write(json_text(value))
        output.flush()
        os.fsync(output.fileno())


def atomic_write_json(path: Path, value: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(json_text(value))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def json_text(value: object) -> str:
    return json.dumps(value, indent=2, sort_keys=True) + "\n"


def verify_embedded_commit(executable: Path, commit: str) -> None:
    result = subprocess.run(
        ["/usr/bin/strings", str(executable)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        errors="replace",
    )
    if commit not in result.stdout.splitlines():
        raise PublishError(
            f"built executable does not contain exact source commit {commit}")


def verify_inference_build_identity(executable: Path, commit: str, digest: str) -> None:
    result = subprocess.run(
        [str(executable), "--build-identity"], check=False,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    if result.returncode != 0:
        raise PublishError("inference worker build identity is unavailable")
    fields = {}
    for field in result.stdout.strip().split(","):
        key, separator, value = field.partition("=")
        if separator:
            fields[key] = value
    if (fields.get("artifact_role") != "lstm-infer-worker" or
            fields.get("source_commit") != commit or
            fields.get("executable_sha256") != f"sha256:{digest}"):
        raise PublishError("inference worker build identity mismatch")


def validate_inputs(
    executable: Path,
    worker_rule: str,
    layout: int,
    width: int,
    commit: str,
    capabilities: list[str],
    worker_role: str,
) -> None:
    if not executable.is_absolute() or not executable.is_file():
        raise PublishError("worker executable must be an absolute existing regular file")
    if not os.access(executable, os.X_OK):
        raise PublishError("worker executable is not executable")
    if worker_rule not in {"current", "historical"}:
        raise PublishError("worker rule must be current or historical")
    if layout <= 0 or width <= 0:
        raise PublishError("semantic layout and model input width must be positive")
    if not COMMIT_PATTERN.fullmatch(commit):
        raise PublishError("source commit must be exact lowercase 40-hex")
    if not capabilities or len(capabilities) != len(set(capabilities)):
        raise PublishError("capabilities must be nonempty and unique")
    if not set(capabilities) <= VALID_CAPABILITIES:
        raise PublishError("capabilities contain an unsupported phase")
    if worker_role != "infer":
        raise PublishError("Phase 22E publication supports only the infer worker role")
    if set(capabilities) != {"infer"}:
        raise PublishError("role-aware inference workers must support only infer")


def load_registry(path: Path) -> dict:
    if not path.exists():
        return {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "current_layout": None,
            "runtimes": [],
            "workers": [],
        }
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublishError(f"existing semantic worker registry is malformed: {error}") from error
    if not isinstance(value, dict):
        raise PublishError("existing semantic worker registry has an invalid shape")
    schema_version = value.get("schema_version")
    expected_fields = {"schema_version", "current_layout", "workers"}
    if schema_version in {REGISTRY_SCHEMA_VERSION,
                          ROLE_AWARE_REGISTRY_SCHEMA_VERSION,
                          LEGACY_REGISTRY_SCHEMA_VERSION}:
        expected_fields.add("runtimes")
    elif schema_version != 1:
        raise PublishError("existing semantic worker registry schema is unsupported")
    if set(value) != expected_fields or not isinstance(value.get("workers"), list):
        raise PublishError("existing semantic worker registry has an invalid shape")
    keys = [(worker.get("semantic_layout"), worker.get("worker_role", "infer"))
            for worker in value["workers"] if isinstance(worker, dict)]
    if len(keys) != len(value["workers"]) or len(keys) != len(set(keys)):
        raise PublishError("existing semantic worker registry has duplicate/malformed layout roles")
    validate_existing_registry(path.parent.resolve(), value)
    if schema_version != REGISTRY_SCHEMA_VERSION:
        workers = []
        for worker in value["workers"]:
            roles = ["infer"]
            if "train" in worker["capabilities"]:
                roles.append("train")
            for role in roles:
                upgraded = dict(worker)
                upgraded["worker_role"] = role
                upgraded["artifact_manifest_schema_version"] = (
                    worker.get("artifact_manifest_schema_version",
                               LEGACY_WORKER_MANIFEST_SCHEMA_VERSION))
                workers.append(upgraded)
        value = {
            "schema_version": REGISTRY_SCHEMA_VERSION,
            "current_layout": value["current_layout"],
            "runtimes": value.get("runtimes", []),
            "workers": workers,
        }
    return value


def validate_existing_registry(artifact_root: Path, registry: dict) -> None:
    required_fields = {
        "semantic_layout", "worker_rule", "model_input_width", "source_commit",
        "sha256", "executable", "manifest", "capabilities",
    }
    if registry["schema_version"] in {REGISTRY_SCHEMA_VERSION,
                                      ROLE_AWARE_REGISTRY_SCHEMA_VERSION,
                                      LEGACY_REGISTRY_SCHEMA_VERSION}:
        required_fields.add("runtime_identity")
        runtimes = registry.get("runtimes")
        if not isinstance(runtimes, list) or not runtimes:
            raise PublishError("existing semantic worker registry runtimes are invalid")
        identities: set[str] = set()
        for runtime in runtimes:
            if not isinstance(runtime, dict) or set(runtime) != {
                "identity", "directory", "manifest"
            }:
                raise PublishError("existing semantic worker runtime shape is invalid")
            identity = runtime["identity"]
            if (not isinstance(identity, str) or
                    not SHA256_PATTERN.fullmatch(identity) or
                    identity in identities):
                raise PublishError("existing semantic worker runtime identity is invalid")
            identities.add(identity)
            verify_runtime_package(artifact_root, runtime)
    else:
        identities = set()
    current_layout = registry["current_layout"]
    if (not isinstance(current_layout, int) or isinstance(current_layout, bool) or
            current_layout <= 0):
        raise PublishError("existing semantic worker registry current layout is invalid")
    if not registry["workers"]:
        raise PublishError("existing semantic worker registry has no workers")
    for worker in registry["workers"]:
        expected_worker_fields = set(required_fields)
        if registry["schema_version"] >= ROLE_AWARE_REGISTRY_SCHEMA_VERSION:
            expected_worker_fields.add("worker_role")
            expected_worker_fields.add("artifact_manifest_schema_version")
        if set(worker) != expected_worker_fields:
            raise PublishError("existing semantic worker registry worker shape is invalid")
        layout = worker["semantic_layout"]
        width = worker["model_input_width"]
        commit = worker["source_commit"]
        digest = worker["sha256"]
        rule = worker["worker_rule"]
        capabilities = worker["capabilities"]
        runtime_identity = worker.get("runtime_identity")
        if (not isinstance(layout, int) or isinstance(layout, bool) or layout <= 0 or
                not isinstance(width, int) or isinstance(width, bool) or width <= 0 or
                not isinstance(rule, str) or rule not in {"current", "historical"} or
                not isinstance(commit, str) or not COMMIT_PATTERN.fullmatch(commit) or
                not isinstance(digest, str) or not SHA256_PATTERN.fullmatch(digest) or
                not isinstance(capabilities, list) or not capabilities or
                not all(isinstance(capability, str) for capability in capabilities) or
                len(capabilities) != len(set(capabilities)) or
                not set(capabilities) <= VALID_CAPABILITIES or
                (registry["schema_version"] in {REGISTRY_SCHEMA_VERSION,
                                                ROLE_AWARE_REGISTRY_SCHEMA_VERSION,
                                                LEGACY_REGISTRY_SCHEMA_VERSION} and
                 runtime_identity not in identities)):
            raise PublishError("existing semantic worker registry worker contract is invalid")
        role = worker.get("worker_role", "infer")
        manifest_schema = worker.get(
            "artifact_manifest_schema_version",
            LEGACY_WORKER_MANIFEST_SCHEMA_VERSION)
        if (registry["schema_version"] >= ROLE_AWARE_REGISTRY_SCHEMA_VERSION and
                role not in {"infer", "train"}):
            raise PublishError("existing semantic worker role is invalid")
        if role not in capabilities:
            raise PublishError("existing semantic worker role capability is invalid")
        if (registry["schema_version"] >= ROLE_AWARE_REGISTRY_SCHEMA_VERSION and
                role == "infer" and manifest_schema == WORKER_MANIFEST_SCHEMA_VERSION and
                set(capabilities) != {"infer"}):
            raise PublishError("existing role-aware inference worker capabilities are invalid")
        if (registry["schema_version"] != REGISTRY_SCHEMA_VERSION and
                rule == "current" and set(capabilities) != VALID_CAPABILITIES):
            raise PublishError("existing current semantic worker capabilities are invalid")
        relative_directory = (Path(f"layout{layout}") / role / commit / digest
                              if manifest_schema == WORKER_MANIFEST_SCHEMA_VERSION
                              else Path(f"layout{layout}") / commit / digest)
        executable_name = ("lstm-infer-worker"
                           if manifest_schema == WORKER_MANIFEST_SCHEMA_VERSION and role == "infer"
                           else "LSTM_Release")
        expected_executable = str(relative_directory / executable_name)
        expected_manifest_path = str(relative_directory / "manifest.json")
        if (worker["executable"] != expected_executable or
                worker["manifest"] != expected_manifest_path):
            raise PublishError("existing semantic worker content-addressed path is invalid")
        directory = artifact_root / relative_directory
        try:
            if directory.resolve(strict=True) != directory:
                raise PublishError("existing semantic worker artifact path is not canonical")
        except OSError as error:
            raise PublishError(
                f"existing semantic worker artifact is missing: {directory}") from error
        expected_manifest = {
            "schema_version": (manifest_schema if role else LEGACY_WORKER_MANIFEST_SCHEMA_VERSION),
            "semantic_layout": layout,
            "storage": "immutable",
            "model_input_width": width,
            "source_commit": commit,
            "sha256": digest,
            "executable_identity": executable_name,
            "capabilities": capabilities,
        }
        if manifest_schema == WORKER_MANIFEST_SCHEMA_VERSION:
            expected_manifest["worker_role"] = role
        verify_existing_artifact(directory, expected_manifest, digest)
        if registry["schema_version"] == REGISTRY_SCHEMA_VERSION:
            runtime = next(item for item in registry["runtimes"]
                           if item["identity"] == runtime_identity)
            verify_runtime_links(artifact_root, directory, runtime)

    current_roles = {
        worker.get("worker_role", "infer")
        for worker in registry["workers"]
        if worker["worker_rule"] == "current" and
        worker["semantic_layout"] == current_layout
    }
    if registry["schema_version"] == REGISTRY_SCHEMA_VERSION:
        if current_roles != {"infer", "train"}:
            raise PublishError("existing semantic worker registry current role bindings are invalid")
    elif current_roles != {"infer"}:
        raise PublishError("existing semantic worker registry current rule is invalid")


def runtime_manifest(runtime_resources: dict[str, Path]) -> tuple[dict, str]:
    resources = []
    for built_identity, runtime_name in RUNTIME_RESOURCE_SPECS:
        source = runtime_resources.get(built_identity)
        if source is None:
            raise PublishError(
                f"required semantic worker runtime resource is missing: {built_identity}")
        try:
            source = source.resolve(strict=True)
        except OSError as error:
            raise PublishError(
                f"required semantic worker runtime resource is missing: {built_identity}"
            ) from error
        if not source.is_file():
            raise PublishError(
                f"required semantic worker runtime resource is not a file: {built_identity}")
        runtime_resources[built_identity] = source
        resources.append({
            "built_identity": built_identity,
            "runtime_name": runtime_name,
            "sha256": sha256(source),
        })
    manifest = {
        "schema_version": RUNTIME_MANIFEST_SCHEMA_VERSION,
        "storage": "immutable",
        "resources": resources,
    }
    identity = hashlib.sha256(json_text(manifest).encode("utf-8")).hexdigest()
    return manifest, identity


def verify_runtime_package(artifact_root: Path, runtime: dict) -> None:
    identity = runtime["identity"]
    relative_directory = Path("runtime") / identity
    if (runtime["directory"] != str(relative_directory) or
            runtime["manifest"] != str(relative_directory / "manifest.json")):
        raise PublishError("existing semantic worker runtime path is invalid")
    directory = artifact_root / relative_directory
    manifest_path = directory / "manifest.json"
    try:
        if (directory.resolve(strict=True) != directory or
                manifest_path.resolve(strict=True) != manifest_path):
            raise PublishError("existing semantic worker runtime path is not canonical")
        raw_manifest = manifest_path.read_bytes()
    except OSError as error:
        raise PublishError(
            f"existing semantic worker runtime is incomplete: {directory}") from error
    if hashlib.sha256(raw_manifest).hexdigest() != identity:
        raise PublishError("existing semantic worker runtime manifest hash conflict")
    try:
        manifest = json.loads(raw_manifest)
    except json.JSONDecodeError as error:
        raise PublishError("existing semantic worker runtime manifest is invalid") from error
    if (not isinstance(manifest, dict) or set(manifest) != {
            "schema_version", "storage", "resources"} or
            manifest["schema_version"] != RUNTIME_MANIFEST_SCHEMA_VERSION or
            manifest["storage"] != "immutable" or
            not isinstance(manifest["resources"], list)):
        raise PublishError("existing semantic worker runtime manifest is invalid")
    expected_names = {runtime_name for _, runtime_name in RUNTIME_RESOURCE_SPECS}
    observed_names: set[str] = set()
    for resource in manifest["resources"]:
        if not isinstance(resource, dict) or set(resource) != {
            "built_identity", "runtime_name", "sha256"
        }:
            raise PublishError("existing semantic worker runtime manifest is invalid")
        built_identity = resource["built_identity"]
        runtime_name = resource["runtime_name"]
        digest = resource["sha256"]
        if ((built_identity, runtime_name) not in RUNTIME_RESOURCE_SPECS or
                runtime_name in observed_names or
                not isinstance(digest, str) or
                not SHA256_PATTERN.fullmatch(digest)):
            raise PublishError("existing semantic worker runtime manifest is invalid")
        resource_path = directory / runtime_name
        try:
            if resource_path.resolve(strict=True) != resource_path:
                raise PublishError("existing semantic worker runtime resource is aliased")
        except OSError as error:
            raise PublishError(
                f"existing semantic worker runtime resource is missing: {runtime_name}"
            ) from error
        if sha256(resource_path) != digest:
            raise PublishError(
                f"existing semantic worker runtime resource hash conflict: {runtime_name}")
        observed_names.add(runtime_name)
    if observed_names != expected_names:
        raise PublishError("existing semantic worker runtime required resources are invalid")


def verify_runtime_links(
    artifact_root: Path, worker_directory: Path, runtime: dict
) -> None:
    runtime_directory = artifact_root / runtime["directory"]
    for _, runtime_name in RUNTIME_RESOURCE_SPECS:
        link = worker_directory / runtime_name
        expected_target = os.path.relpath(runtime_directory / runtime_name,
                                          worker_directory)
        if not link.is_symlink() or os.readlink(link) != expected_target:
            raise PublishError(
                f"semantic worker runtime dependency is unresolvable: {runtime_name}")
        try:
            if link.resolve(strict=True) != (runtime_directory / runtime_name):
                raise PublishError(
                    f"semantic worker runtime dependency is unresolvable: {runtime_name}")
        except OSError as error:
            raise PublishError(
                f"semantic worker runtime dependency is missing: {runtime_name}"
            ) from error


def verify_existing_artifact(directory: Path, expected_manifest: dict, digest: str) -> None:
    executable = directory / expected_manifest["executable_identity"]
    manifest = directory / "manifest.json"
    try:
        canonical_executable = executable.resolve(strict=True)
        canonical_manifest = manifest.resolve(strict=True)
    except OSError as error:
        raise PublishError(f"immutable artifact path is incomplete: {directory}") from error
    if canonical_executable != executable or canonical_manifest != manifest:
        raise PublishError(f"immutable artifact path is not canonical: {directory}")
    if not executable.is_file() or not os.access(executable, os.X_OK):
        raise PublishError(f"immutable artifact path is incomplete: {directory}")
    if sha256(executable) != digest:
        raise PublishError(f"immutable artifact hash conflict: {directory}")
    try:
        observed_manifest = json.loads(manifest.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise PublishError(f"immutable artifact manifest is invalid: {directory}") from error
    if observed_manifest != expected_manifest:
        raise PublishError(f"immutable artifact manifest conflict: {directory}")


def stage_runtime_package(
    artifact_root: Path,
    runtime_resources: dict[str, Path],
    manifest_value: dict,
    identity: str,
) -> dict:
    relative_directory = Path("runtime") / identity
    final_directory = artifact_root / relative_directory
    runtime_value = {
        "identity": identity,
        "directory": str(relative_directory),
        "manifest": str(relative_directory / "manifest.json"),
    }
    if final_directory.exists():
        verify_runtime_package(artifact_root, runtime_value)
        return runtime_value

    final_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{identity}.", suffix=".stage", dir=final_directory.parent))
    try:
        for resource in manifest_value["resources"]:
            built_identity = resource["built_identity"]
            staged_resource = staging / resource["runtime_name"]
            shutil.copy2(runtime_resources[built_identity], staged_resource)
            staged_resource.chmod(0o444)
            if sha256(staged_resource) != resource["sha256"]:
                raise PublishError(
                    f"staged semantic worker runtime hash mismatch: {built_identity}")
            fsync_file(staged_resource)
        staged_manifest = staging / "manifest.json"
        write_json(staged_manifest, manifest_value)
        staged_manifest.chmod(0o444)
        if sha256(staged_manifest) != identity:
            raise PublishError("staged semantic worker runtime identity mismatch")
        fsync_directory(staging)
        os.rename(staging, final_directory)
        fsync_directory(final_directory.parent)
    finally:
        if staging.exists():
            shutil.rmtree(staging)
    verify_runtime_package(artifact_root, runtime_value)
    return runtime_value


def install_runtime_links(
    artifact_root: Path, worker_directory: Path, runtime: dict
) -> None:
    runtime_directory = artifact_root / runtime["directory"]
    for _, runtime_name in RUNTIME_RESOURCE_SPECS:
        link = worker_directory / runtime_name
        target = os.path.relpath(runtime_directory / runtime_name, worker_directory)
        if os.path.lexists(link):
            if not link.is_symlink() or os.readlink(link) != target:
                raise PublishError(
                    f"semantic worker runtime link conflict: {link}")
            continue
        os.symlink(target, link)
    fsync_directory(worker_directory)
    verify_runtime_links(artifact_root, worker_directory, runtime)


def publish(
    artifact_root: Path,
    executable: Path,
    worker_rule: str,
    layout: int,
    width: int,
    commit: str,
    capabilities: list[str],
    worker_role: str = "infer",
    check_embedded_commit: bool = True,
    runtime_resources: dict[str, Path] | None = None,
) -> Path:
    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve()
    try:
        executable = executable.resolve(strict=True)
    except OSError as error:
        raise PublishError(
            "worker executable must be an absolute existing regular file"
        ) from error
    validate_inputs(executable, worker_rule, layout, width, commit, capabilities, worker_role)
    if check_embedded_commit:
        verify_embedded_commit(executable, commit)
    digest = sha256(executable)
    if not SHA256_PATTERN.fullmatch(digest):
        raise PublishError("computed SHA-256 is malformed")
    if check_embedded_commit:
        verify_inference_build_identity(executable, commit, digest)
    if runtime_resources is None:
        runtime_resources = {
            built_identity: executable.parent / built_identity
            for built_identity, _ in RUNTIME_RESOURCE_SPECS
        }
    runtime_manifest_value, runtime_identity = runtime_manifest(runtime_resources)

    relative_directory = Path(f"layout{layout}") / worker_role / commit / digest
    final_directory = artifact_root / relative_directory
    manifest_value = {
        "schema_version": WORKER_MANIFEST_SCHEMA_VERSION,
        "semantic_layout": layout,
        "storage": "immutable",
        "model_input_width": width,
        "source_commit": commit,
        "sha256": digest,
        "executable_identity": "lstm-infer-worker",
        "worker_role": worker_role,
        "capabilities": capabilities,
    }
    worker_value = {
        "semantic_layout": layout,
        "worker_role": worker_role,
        "artifact_manifest_schema_version": WORKER_MANIFEST_SCHEMA_VERSION,
        "worker_rule": worker_rule,
        "model_input_width": width,
        "source_commit": commit,
        "sha256": digest,
        "executable": str(relative_directory / "lstm-infer-worker"),
        "manifest": str(relative_directory / "manifest.json"),
        "runtime_identity": runtime_identity,
        "capabilities": capabilities,
    }

    lock_path = artifact_root / ".publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry_path = artifact_root / "registry.json"
        registry = load_registry(registry_path)
        if registry["current_layout"] is None:
            raise PublishError(
                "inference publication requires an existing training/reference binding")
        if worker_rule == "current" and registry["current_layout"] != layout:
            raise PublishError(
                "inference publication cannot change current layout without a training/reference binding")
        runtime_value = stage_runtime_package(
            artifact_root,
            runtime_resources,
            runtime_manifest_value,
            runtime_identity,
        )
        if not any(item["identity"] == runtime_identity
                   for item in registry["runtimes"]):
            registry["runtimes"].append(runtime_value)
            registry["runtimes"].sort(key=lambda item: item["identity"])

        # Schema-v1 archives retain their executable and manifest identity.
        # Publication attaches only deterministic links to the separately
        # immutable runtime package, then binds those links in registry v2.
        runtime_by_identity = {
            item["identity"]: item for item in registry["runtimes"]
        }
        for existing_worker in registry["workers"]:
            if "runtime_identity" not in existing_worker:
                existing_worker["runtime_identity"] = runtime_identity
            existing_directory = (
                artifact_root / existing_worker["executable"]
            ).parent
            install_runtime_links(
                artifact_root,
                existing_directory,
                runtime_by_identity[existing_worker["runtime_identity"]],
            )

        if final_directory.exists():
            verify_existing_artifact(final_directory, manifest_value, digest)
        else:
            final_directory.parent.mkdir(parents=True, exist_ok=True)
            staging = Path(tempfile.mkdtemp(
                prefix=f".{digest}.", suffix=".stage", dir=final_directory.parent))
            try:
                staged_executable = staging / "lstm-infer-worker"
                shutil.copy2(executable, staged_executable)
                staged_executable.chmod(0o555)
                if sha256(staged_executable) != digest:
                    raise PublishError("staged semantic worker hash mismatch")
                fsync_file(staged_executable)
                staged_manifest = staging / "manifest.json"
                write_json(staged_manifest, manifest_value)
                staged_manifest.chmod(0o444)
                fsync_directory(staging)
                os.rename(staging, final_directory)
                fsync_directory(final_directory.parent)
            finally:
                if staging.exists():
                    shutil.rmtree(staging)

        install_runtime_links(artifact_root, final_directory, runtime_value)

        workers = [worker for worker in registry["workers"]
                   if (worker["semantic_layout"], worker.get("worker_role", "infer")) !=
                   (layout, worker_role)]
        if worker_rule == "current":
            for worker in workers:
                if (worker.get("worker_rule") == "current" and
                        worker["worker_role"] == worker_role):
                    worker["worker_rule"] = "historical"
            registry["current_layout"] = layout
        elif registry["current_layout"] == layout:
            raise PublishError("cannot replace the current layout with a historical rule")
        workers.append(worker_value)
        workers.sort(key=lambda worker: (worker["semantic_layout"],
                                         worker["worker_role"]))
        registry["workers"] = workers
        if registry["current_layout"] is None:
            raise PublishError("historical publication requires an existing current worker")

        atomic_write_json(registry_path, registry)

        if worker_rule == "current":
            current_link = artifact_root / "current"
            temporary_link = artifact_root / f".current.{os.getpid()}.tmp"
            temporary_link.unlink(missing_ok=True)
            os.symlink(relative_directory, temporary_link)
            os.replace(temporary_link, current_link)
            fsync_directory(artifact_root)
    return final_directory / "lstm-infer-worker"


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--built-executable", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--worker-rule", choices=("current", "historical"), default="current")
    parser.add_argument("--semantic-layout", type=int)
    parser.add_argument("--model-input-width", type=int)
    parser.add_argument("--source-commit")
    parser.add_argument("--capability", action="append", dest="capabilities")
    parser.add_argument("--worker-role", choices=("infer",), default="infer")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    repository_root = arguments.repository_root.resolve(strict=True)
    artifact_root = (arguments.artifact_root or
                     repository_root / "Builds" / "SemanticWorkers")
    if arguments.semantic_layout is None or arguments.model_input_width is None:
        if arguments.worker_rule != "current":
            raise PublishError("historical import requires explicit semantic layout and width")
        source_layout, source_width = current_semantic_contract(repository_root)
        layout = arguments.semantic_layout or source_layout
        width = arguments.model_input_width or source_width
        if (layout, width) != (source_layout, source_width):
            raise PublishError("explicit current contract disagrees with source")
    else:
        layout, width = arguments.semantic_layout, arguments.model_input_width
    if arguments.worker_rule == "current":
        clean_commit = clean_source_commit(repository_root)
        if (arguments.source_commit is not None and
                arguments.source_commit != clean_commit):
            raise PublishError("explicit current source commit disagrees with clean HEAD")
        commit = clean_commit
    else:
        if arguments.source_commit is None:
            raise PublishError("historical import requires an explicit source commit")
        commit = arguments.source_commit
    capabilities = arguments.capabilities or ["infer"]
    published = publish(
        artifact_root=artifact_root,
        executable=arguments.built_executable,
        worker_rule=arguments.worker_rule,
        layout=layout,
        width=width,
        commit=commit,
        capabilities=capabilities,
        worker_role=arguments.worker_role,
        check_embedded_commit=True,
    )
    print(f"Semantic worker published: {published}")
    print(f"Semantic worker registry: {(artifact_root / 'registry.json').resolve()}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, subprocess.CalledProcessError, PublishError) as error:
        raise SystemExit(f"PublishSemanticWorker.py: {error}")
