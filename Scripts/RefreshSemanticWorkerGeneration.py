#!/usr/bin/env python3
"""Atomically refresh both role bindings for the current semantic layout.

Unlike RollSemanticWorkerLayout.py, this operation deliberately cannot change
the semantic layout.  It is for a code-only generation replacement where the
clean source contract and registry already agree on layout and input width.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
import os
from pathlib import Path
import subprocess


_ROLLOVER_PATH = Path(__file__).with_name("RollSemanticWorkerLayout.py")
_SPEC = importlib.util.spec_from_file_location("semantic_worker_rollover",
                                               _ROLLOVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
rollover = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollover)
publisher = rollover.publisher


def _refreshed_training_capabilities(
    feature_ablation_qualified: bool,
) -> list[str]:
    """Return the narrow immutable claim for a refreshed TRAIN artifact.

    A current-layout refresh that explicitly qualifies ablation is commonly
    used for controlled ablation work. Its artifact need only claim the TRAIN
    implementation and that qualified superset capability; INFER has its
    dedicated role artifact and ANALYZE is scheduler-owned. Keeping this claim
    narrow also makes append-only priority a stable tie-breaker among future
    ablation-qualified TRAIN candidates.
    """
    if feature_ablation_qualified:
        return ["train", "train_feature_ablation_v1"]
    return ["train"]


def _validate_refresh_prestate(registry: dict, layout: int, width: int) -> None:
    if registry["schema_version"] != publisher.REGISTRY_SCHEMA_VERSION:
        raise publisher.PublishError(
            "current semantic worker refresh requires registry schema "
            f"{publisher.REGISTRY_SCHEMA_VERSION}")
    if registry["current_layout"] != layout:
        raise publisher.PublishError(
            "current semantic worker refresh requires registry layout to match source contract")
    current_layout_workers = [
        worker for worker in registry["workers"]
        if worker["semantic_layout"] == layout and
        worker["worker_rule"] == "current"
    ]
    if (len(current_layout_workers) != 2 or
            {worker["worker_role"] for worker in current_layout_workers} != {"train", "infer"} or
            any(worker["worker_rule"] != "current" for worker in current_layout_workers)):
        raise publisher.PublishError(
            "current semantic worker refresh requires exactly current train and infer bindings")
    if any(worker["model_input_width"] != width for worker in current_layout_workers):
        raise publisher.PublishError(
            "current semantic worker refresh requires registry width to match source contract")


def refresh(
    artifact_root: Path,
    training_executable: Path,
    inference_executable: Path,
    layout: int,
    width: int,
    commit: str,
    *,
    inference_commit: str | None = None,
    check_embedded_commit: bool = True,
    runtime_resources: dict[str, Path] | None = None,
    feature_ablation_qualified: bool = False,
) -> tuple[Path, Path]:
    """Stage a matched generation and atomically replace current role bindings."""
    training = rollover._resolve_executable(training_executable, "lstm-train-worker")
    inference = rollover._resolve_executable(inference_executable, "lstm-infer-worker")
    inference_commit = inference_commit or commit
    if (layout <= 0 or width <= 0 or
            not publisher.COMMIT_PATTERN.fullmatch(commit) or
            not publisher.COMMIT_PATTERN.fullmatch(inference_commit)):
        raise publisher.PublishError("refresh semantic contract or source commit is invalid")
    if check_embedded_commit:
        publisher.verify_embedded_commit(training, commit)
        training_digest = publisher.sha256(training)
        publisher.verify_worker_build_identity(training, "train", commit, training_digest)
        publisher.verify_embedded_commit(inference, inference_commit)
        inference_digest = publisher.sha256(inference)
        publisher.verify_worker_build_identity(
            inference, "infer", inference_commit, inference_digest)
        runtime_resources = rollover._runtime_resources_match(training, inference)
    else:
        training_digest = publisher.sha256(training)
        inference_digest = publisher.sha256(inference)
        if runtime_resources is None:
            raise publisher.PublishError("test refresh requires explicit runtime resources")
    assert runtime_resources is not None
    runtime_manifest, runtime_identity = publisher.runtime_manifest(dict(runtime_resources))
    training_relative, training_manifest, training_worker = rollover._worker_value(
        layout, width, commit, training_digest, "train",
        publisher.WORKER_MANIFEST_SCHEMA_VERSION,
        _refreshed_training_capabilities(feature_ablation_qualified), runtime_identity)
    inference_relative, inference_manifest, inference_worker = rollover._worker_value(
        layout, width, inference_commit, inference_digest, "infer",
        publisher.WORKER_MANIFEST_SCHEMA_VERSION,
        rollover.INFERENCE_CAPABILITIES, runtime_identity)

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    with (artifact_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry_path = artifact_root / "registry.json"
        registry = publisher.load_registry(registry_path)
        _validate_refresh_prestate(registry, layout, width)
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

        staged_training = rollover._stage_worker(
            artifact_root, training, training_relative, training_manifest,
            training_digest, runtime)
        staged_inference = rollover._stage_worker(
            artifact_root, inference, inference_relative, inference_manifest,
            inference_digest, runtime)

        # Retain the outgoing TRAIN artifact as an immutable historical
        # candidate. The registry permits several TRAIN candidates, whereas
        # INFER intentionally remains a singleton per layout; the outgoing
        # infer artifact stays immutable on disk but is retired from its
        # singleton binding. No executable or artifact manifest is rewritten.
        for worker in registry["workers"]:
            if (worker["semantic_layout"] == layout and
                    worker["worker_rule"] == "current" and
                    worker["worker_role"] == "train"):
                worker["worker_rule"] = "historical"
        registry["workers"] = [
            worker for worker in registry["workers"]
            if not (worker["semantic_layout"] == layout and
                    worker["worker_rule"] == "current" and
                    worker["worker_role"] == "infer")
        ]
        training_worker["selection_priority"] = 1 + max(
            worker["selection_priority"] for worker in registry["workers"]
            if worker["semantic_layout"] == layout and
            worker["worker_role"] == "train" and
            worker["model_input_width"] == width)
        registry["workers"].extend([training_worker, inference_worker])
        publisher.validate_existing_registry(artifact_root, registry)
        publisher.atomic_write_json(registry_path, registry)

        # The registry is already authoritative. A link/fsync failure is
        # explicitly surfaced as committed-but-link-update-failed.
        publisher.update_current_link_after_registry_commit(
            artifact_root, inference_relative)
    return staged_training, staged_inference


def refresh_from_repository(
    repository_root: Path,
    training_executable: Path,
    inference_executable: Path,
    artifact_root: Path | None = None,
    source_commit: str | None = None,
    inference_source_commit: str | None = None,
    feature_ablation_qualified: bool = False,
) -> tuple[Path, Path, int, int, str]:
    repository_root = repository_root.resolve(strict=True)
    clean_commit = publisher.clean_source_commit(repository_root)
    commit = source_commit or clean_commit
    layout, width = publisher.current_semantic_contract(repository_root)
    training, inference = refresh(
        artifact_root or repository_root / "Builds" / "SemanticWorkers",
        training_executable, inference_executable, layout, width, commit,
        inference_commit=inference_source_commit,
        feature_ablation_qualified=feature_ablation_qualified)
    return training, inference, layout, width, commit


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--training-executable", required=True, type=Path)
    parser.add_argument("--inference-executable", required=True, type=Path)
    parser.add_argument("--artifact-root", type=Path)
    parser.add_argument("--source-commit")
    parser.add_argument("--inference-source-commit")
    parser.add_argument("--train-feature-ablation-qualified", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    training, inference, layout, width, commit = refresh_from_repository(
        arguments.repository_root, arguments.training_executable,
        arguments.inference_executable, arguments.artifact_root,
        source_commit=arguments.source_commit,
        inference_source_commit=arguments.inference_source_commit,
        feature_ablation_qualified=arguments.train_feature_ablation_qualified)
    print(f"Semantic worker generation refreshed: layout={layout}, width={width}")
    print(f"source_commit={commit}")
    print(f"training_worker={training}")
    print(f"inference_worker={inference}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, publisher.PublishError, subprocess.SubprocessError) as error:
        raise SystemExit(f"RefreshSemanticWorkerGeneration.py: {error}")
