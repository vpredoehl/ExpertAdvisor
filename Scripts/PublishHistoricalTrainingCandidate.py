#!/usr/bin/env python3
"""Append a qualified immutable historical TRAIN candidate to registry v5.

This deliberately does not advance ``current_layout`` or publish inference.
Its explicit capability list is an artifact claim; in particular the ablation
capability is accepted only with an explicit qualification switch.
"""

from __future__ import annotations

import argparse
import fcntl
import importlib.util
from pathlib import Path
import subprocess


_ROLLOVER_PATH = Path(__file__).with_name("RollSemanticWorkerLayout.py")
_SPEC = importlib.util.spec_from_file_location("semantic_worker_rollover", _ROLLOVER_PATH)
assert _SPEC is not None and _SPEC.loader is not None
rollover = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(rollover)
publisher = rollover.publisher


def append_historical_training_candidate(
    artifact_root: Path,
    executable: Path,
    layout: int,
    width: int,
    commit: str,
    capabilities: list[str],
    selection_priority: int,
    *,
    feature_ablation_qualified: bool = False,
    check_embedded_commit: bool = True,
    runtime_resources: dict[str, Path] | None = None,
) -> Path:
    """Stage and append one historical TRAIN candidate without replacement."""
    training = rollover._resolve_executable(executable, "LSTM_Release")
    if (layout <= 0 or width <= 0 or
            not publisher.COMMIT_PATTERN.fullmatch(commit) or
            not isinstance(selection_priority, int) or
            isinstance(selection_priority, bool) or selection_priority < 0):
        raise publisher.PublishError("historical training candidate contract is invalid")
    if (not capabilities or len(capabilities) != len(set(capabilities)) or
            not set(capabilities) <= publisher.VALID_CAPABILITIES or
            "train" not in capabilities):
        raise publisher.PublishError("historical training candidate capabilities are invalid")
    advertises_ablation = "train_feature_ablation_v1" in capabilities
    if advertises_ablation != feature_ablation_qualified:
        raise publisher.PublishError(
            "train feature ablation capability requires explicit qualification")
    if check_embedded_commit:
        publisher.verify_embedded_commit(training, commit)
    digest = publisher.sha256(training)
    if runtime_resources is None:
        runtime_resources = {
            built_identity: training.parent / built_identity
            for built_identity, _ in publisher.RUNTIME_RESOURCE_SPECS
        }
    runtime_manifest, runtime_identity = publisher.runtime_manifest(runtime_resources)
    relative, manifest, worker = rollover._worker_value(
        layout, width, commit, digest, "train",
        publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION,
        capabilities, runtime_identity)
    worker["worker_rule"] = "historical"
    worker["selection_priority"] = selection_priority

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    with (artifact_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry_path = artifact_root / "registry.json"
        registry = publisher.load_registry(registry_path)
        if registry["current_layout"] is None:
            raise publisher.PublishError(
                "historical training candidate requires an existing current registry")
        role_layout_candidates = [entry for entry in registry["workers"]
                                  if entry["semantic_layout"] == layout and
                                  entry.get("worker_role") == "train"]
        matching = [entry for entry in role_layout_candidates
                    if entry["model_input_width"] == width]
        if not any(entry["worker_rule"] == "historical" for entry in matching):
            raise publisher.PublishError(
                "historical training candidate requires an existing historical binding")
        # Keep this preflight identity scope identical to final registry
        # validation.  Width is intentionally not part of artifact identity:
        # the immutable manifest already binds it.
        if any((entry["source_commit"], entry["sha256"]) == (commit, digest)
               for entry in role_layout_candidates):
            raise publisher.PublishError("historical training candidate identity already registered")
        if any(entry["executable"] == worker["executable"] for entry in matching):
            raise publisher.PublishError("historical training candidate executable already registered")
        if any(entry["selection_priority"] == selection_priority for entry in matching):
            raise publisher.PublishError("historical training candidate priority already registered")
        if matching and selection_priority <= max(
                entry["selection_priority"] for entry in matching):
            raise publisher.PublishError(
                "historical training candidate priority must append after existing candidates")

        # Content-addressed staging is deliberately commit-before-registry.
        # If prospective validation or atomic registry replacement fails, the
        # prior registry remains authoritative and an exact retry verifies and
        # reuses this immutable runtime/artifact.  A different manifest at the
        # same content address fails closed in _stage_worker.
        runtime = publisher.stage_runtime_package(
            artifact_root, runtime_resources, runtime_manifest, runtime_identity)
        if not any(item["identity"] == runtime_identity for item in registry["runtimes"]):
            registry["runtimes"].append(runtime)
            registry["runtimes"].sort(key=lambda item: item["identity"])
        runtime_by_identity = {item["identity"]: item for item in registry["runtimes"]}
        for existing in registry["workers"]:
            publisher.install_runtime_links(
                artifact_root, (artifact_root / existing["executable"]).parent,
                runtime_by_identity[existing["runtime_identity"]])
        staged = rollover._stage_worker(
            artifact_root, training, relative, manifest, digest, runtime)
        registry["workers"].append(worker)
        registry["workers"].sort(
            key=lambda entry: (entry["semantic_layout"], entry["worker_role"],
                               entry["model_input_width"], entry["selection_priority"]))
        publisher.validate_existing_registry(artifact_root, registry)
        publisher.atomic_write_json(registry_path, registry)
    return staged


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", required=True, type=Path)
    parser.add_argument("--training-executable", required=True, type=Path)
    parser.add_argument("--semantic-layout", required=True, type=int)
    parser.add_argument("--model-input-width", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--capability", action="append", required=True)
    parser.add_argument("--selection-priority", required=True, type=int)
    parser.add_argument("--train-feature-ablation-qualified", action="store_true")
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    staged = append_historical_training_candidate(
        arguments.artifact_root, arguments.training_executable,
        arguments.semantic_layout, arguments.model_input_width,
        arguments.source_commit, arguments.capability,
        arguments.selection_priority,
        feature_ablation_qualified=arguments.train_feature_ablation_qualified)
    print(f"Historical TRAIN candidate published: {staged}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, publisher.PublishError, subprocess.SubprocessError) as error:
        raise SystemExit(f"PublishHistoricalTrainingCandidate.py: {error}")
