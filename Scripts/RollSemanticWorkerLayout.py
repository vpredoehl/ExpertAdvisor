#!/usr/bin/env python3
"""Atomically roll the current semantic-worker layout with both role bindings.

This deliberately does not extend PublishSemanticWorker.py's infer-only CLI.
An advance of current_layout has a different safety contract: the new layout
must gain its LSTM_Release training/reference binding and its dedicated
lstm-infer-worker binding in one registry replacement.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import tempfile


_PUBLISHER_PATH = Path(__file__).with_name("PublishSemanticWorker.py")
_SPEC = importlib.util.spec_from_file_location("semantic_worker_publisher",
                                               _PUBLISHER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
publisher = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(publisher)


TRAINING_CAPABILITIES = ["train", "infer", "analyze"]
INFERENCE_CAPABILITIES = ["infer"]


def _resolve_executable(path: Path, expected_name: str) -> Path:
    if not path.is_absolute():
        raise publisher.PublishError("worker executable must be an absolute existing regular file")
    try:
        resolved = path.resolve(strict=True)
    except OSError as error:
        raise publisher.PublishError(
            "worker executable must be an absolute existing regular file") from error
    if (resolved.name != expected_name or not resolved.is_file() or
            not os.access(resolved, os.X_OK)):
        raise publisher.PublishError(
            f"expected {expected_name} executable regular file")
    return resolved


def _runtime_resources_match(training: Path, inference: Path) -> dict[str, Path]:
    resources: dict[str, Path] = {}
    for built_identity, _ in publisher.RUNTIME_RESOURCE_SPECS:
        training_resource = training.parent / built_identity
        inference_resource = inference.parent / built_identity
        try:
            training_resource = training_resource.resolve(strict=True)
            inference_resource = inference_resource.resolve(strict=True)
        except OSError as error:
            raise publisher.PublishError(
                f"required semantic worker runtime resource is missing: {built_identity}") from error
        if (not training_resource.is_file() or not inference_resource.is_file() or
                publisher.sha256(training_resource) != publisher.sha256(inference_resource)):
            raise publisher.PublishError(
                f"training and inference runtime resource mismatch: {built_identity}")
        resources[built_identity] = inference_resource
    return resources


def _worker_value(
    layout: int, width: int, commit: str, digest: str, role: str,
    manifest_schema: int, capabilities: list[str], runtime_identity: str,
) -> tuple[Path, dict, dict]:
    if manifest_schema == publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION:
        relative_directory = Path(f"layout{layout}") / commit / digest
        executable_name = "LSTM_Release"
    elif manifest_schema == publisher.WORKER_MANIFEST_SCHEMA_VERSION:
        relative_directory = Path(f"layout{layout}") / role / commit / digest
        executable_name = "lstm-infer-worker"
    else:
        raise publisher.PublishError("unsupported semantic worker artifact manifest schema")
    manifest = {
        "schema_version": manifest_schema,
        "semantic_layout": layout,
        "storage": "immutable",
        "model_input_width": width,
        "source_commit": commit,
        "sha256": digest,
        "executable_identity": executable_name,
        "capabilities": capabilities,
    }
    if manifest_schema == publisher.WORKER_MANIFEST_SCHEMA_VERSION:
        manifest["worker_role"] = role
    worker = {
        "semantic_layout": layout,
        "worker_role": role,
        "artifact_manifest_schema_version": manifest_schema,
        "worker_rule": "current",
        "model_input_width": width,
        "source_commit": commit,
        "sha256": digest,
        "executable": str(relative_directory / executable_name),
        "manifest": str(relative_directory / "manifest.json"),
        "runtime_identity": runtime_identity,
        "capabilities": capabilities,
    }
    return relative_directory, manifest, worker


def _stage_worker(
    artifact_root: Path, executable: Path, relative_directory: Path,
    manifest: dict, digest: str, runtime: dict,
) -> Path:
    final_directory = artifact_root / relative_directory
    executable_name = manifest["executable_identity"]
    if final_directory.exists():
        publisher.verify_existing_artifact(final_directory, manifest, digest)
    else:
        final_directory.parent.mkdir(parents=True, exist_ok=True)
        staging = Path(tempfile.mkdtemp(
            prefix=f".{digest}.", suffix=".stage", dir=final_directory.parent))
        try:
            staged_executable = staging / executable_name
            shutil.copy2(executable, staged_executable)
            staged_executable.chmod(0o555)
            if publisher.sha256(staged_executable) != digest:
                raise publisher.PublishError("staged semantic worker hash mismatch")
            publisher.fsync_file(staged_executable)
            staged_manifest = staging / "manifest.json"
            publisher.write_json(staged_manifest, manifest)
            staged_manifest.chmod(0o444)
            publisher.fsync_directory(staging)
            try:
                os.rename(staging, final_directory)
            except FileExistsError:
                publisher.verify_existing_artifact(final_directory, manifest, digest)
            else:
                publisher.fsync_directory(final_directory.parent)
        finally:
            if staging.exists():
                shutil.rmtree(staging)
    publisher.install_runtime_links(artifact_root, final_directory, runtime)
    return final_directory / executable_name


def _validate_rollover_prestate(registry: dict, layout: int) -> None:
    if registry["schema_version"] != publisher.REGISTRY_SCHEMA_VERSION:
        raise publisher.PublishError("current semantic layout rollover requires registry schema 4")
    if registry["current_layout"] == layout:
        raise publisher.PublishError("current semantic layout rollover requires a new layout")
    if any(worker["semantic_layout"] == layout for worker in registry["workers"]):
        raise publisher.PublishError("target semantic layout is already registered")
    current_roles = {
        worker["worker_role"] for worker in registry["workers"]
        if worker["worker_rule"] == "current" and
        worker["semantic_layout"] == registry["current_layout"]
    }
    if current_roles != {"train", "infer"}:
        raise publisher.PublishError(
            "current semantic layout rollover requires both prior role bindings")


def rollover(
    artifact_root: Path,
    training_executable: Path,
    inference_executable: Path,
    layout: int,
    width: int,
    commit: str,
    *,
    check_embedded_commit: bool = True,
    runtime_resources: dict[str, Path] | None = None,
) -> tuple[Path, Path]:
    """Stage both immutable artifacts, then atomically advance current_layout."""
    training = _resolve_executable(training_executable, "LSTM_Release")
    inference = _resolve_executable(inference_executable, "lstm-infer-worker")
    if layout <= 0 or width <= 0 or not publisher.COMMIT_PATTERN.fullmatch(commit):
        raise publisher.PublishError("rollover semantic contract or source commit is invalid")
    if check_embedded_commit:
        publisher.verify_embedded_commit(training, commit)
        training_digest = publisher.sha256(training)
        publisher.verify_embedded_commit(inference, commit)
        inference_digest = publisher.sha256(inference)
        publisher.verify_inference_build_identity(inference, commit, inference_digest)
        runtime_resources = _runtime_resources_match(training, inference)
    else:
        training_digest = publisher.sha256(training)
        inference_digest = publisher.sha256(inference)
        if runtime_resources is None:
            raise publisher.PublishError("test rollover requires explicit runtime resources")
    assert runtime_resources is not None
    runtime_manifest, runtime_identity = publisher.runtime_manifest(dict(runtime_resources))
    training_relative, training_manifest, training_worker = _worker_value(
        layout, width, commit, training_digest, "train",
        publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
        TRAINING_CAPABILITIES, runtime_identity)
    inference_relative, inference_manifest, inference_worker = _worker_value(
        layout, width, commit, inference_digest, "infer",
        publisher.WORKER_MANIFEST_SCHEMA_VERSION,
        INFERENCE_CAPABILITIES, runtime_identity)

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    lock_path = artifact_root / ".publish.lock"
    with lock_path.open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry_path = artifact_root / "registry.json"
        registry = publisher.load_registry(registry_path)
        _validate_rollover_prestate(registry, layout)
        runtime = publisher.stage_runtime_package(
            artifact_root, dict(runtime_resources), runtime_manifest, runtime_identity)
        if not any(item["identity"] == runtime_identity for item in registry["runtimes"]):
            registry["runtimes"].append(runtime)
            registry["runtimes"].sort(key=lambda item: item["identity"])
        runtime_by_identity = {item["identity"]: item for item in registry["runtimes"]}
        for worker in registry["workers"]:
            publisher.install_runtime_links(
                artifact_root, (artifact_root / worker["executable"]).parent,
                runtime_by_identity[worker["runtime_identity"]])

        staged_training = _stage_worker(
            artifact_root, training, training_relative, training_manifest,
            training_digest, runtime)
        staged_inference = _stage_worker(
            artifact_root, inference, inference_relative, inference_manifest,
            inference_digest, runtime)

        for worker in registry["workers"]:
            if worker["worker_rule"] == "current":
                worker["worker_rule"] = "historical"
        registry["workers"].extend([training_worker, inference_worker])
        registry["workers"].sort(
            key=lambda worker: (worker["semantic_layout"], worker["worker_role"]))
        registry["current_layout"] = layout
        # This validates the complete prospective state, including both role
        # bindings and all old artifacts, before registry replacement.
        publisher.validate_existing_registry(artifact_root, registry)
        publisher.atomic_write_json(registry_path, registry)

        # Registry authority precedes this convenience link.  It intentionally
        # continues to name the current infer-worker directory, as the old
        # inference-only publisher does.
        current_link = artifact_root / "current"
        temporary_link = artifact_root / f".current.{os.getpid()}.tmp"
        temporary_link.unlink(missing_ok=True)
        try:
            os.symlink(inference_relative, temporary_link)
            os.replace(temporary_link, current_link)
            publisher.fsync_directory(artifact_root)
        finally:
            temporary_link.unlink(missing_ok=True)
    return staged_training, staged_inference


def rollover_from_repository(
    repository_root: Path,
    training_executable: Path,
    inference_executable: Path,
    artifact_root: Path | None = None,
    source_commit: str | None = None,
) -> tuple[Path, Path, int, int, str]:
    repository_root = repository_root.resolve(strict=True)
    commit = publisher.clean_source_commit(repository_root)
    if source_commit is not None and source_commit != commit:
        raise publisher.PublishError("explicit source commit disagrees with clean HEAD")
    layout, width = publisher.current_semantic_contract(repository_root)
    training, inference = rollover(
        artifact_root or repository_root / "Builds" / "SemanticWorkers",
        training_executable, inference_executable, layout, width, commit)
    return training, inference, layout, width, commit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--training-executable", required=True, type=Path)
    parser.add_argument("--inference-executable", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--source-commit")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    training, inference, layout, width, commit = rollover_from_repository(
        arguments.repository_root, arguments.training_executable,
        arguments.inference_executable, arguments.artifact_root,
        arguments.source_commit)
    print(f"Semantic layout rollover published: layout={layout}, width={width}")
    print(f"source_commit={commit}")
    print(f"training_reference={training}")
    print(f"inference_worker={inference}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, publisher.PublishError, subprocess.SubprocessError) as error:
        raise SystemExit(f"RollSemanticWorkerLayout.py: {error}")
