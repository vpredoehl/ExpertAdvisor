#!/usr/bin/env python3
"""Bootstrap one immutable legacy historical INFER worker binding.

This intentionally cannot advance ``current_layout`` or alter any existing
worker rule.  It publishes only the runtime-supported schema-v1 LSTM_Release
inference artifact used for a historical semantic layout.
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


def publish_historical_inference_worker(
    artifact_root: Path,
    executable: Path,
    layout: int,
    width: int,
    commit: str,
    *,
    runtime_resources: dict[str, Path] | None = None,
) -> Path:
    """Stage one schema-v1 historical LSTM_Release INFER binding.

    The caller supplies a legacy executable, so embedded-commit evidence is
    required but the schema-v2 dedicated-worker build identity is not used.
    """
    inference = rollover._resolve_executable(executable, "LSTM_Release")
    if (layout <= 0 or width <= 0 or
            not publisher.COMMIT_PATTERN.fullmatch(commit)):
        raise publisher.PublishError(
            "historical inference semantic contract or source commit is invalid")
    publisher.verify_embedded_commit(inference, commit)
    digest = publisher.sha256(inference)
    if runtime_resources is None:
        runtime_resources = {
            built_identity: inference.parent / built_identity
            for built_identity, _ in publisher.RUNTIME_RESOURCE_SPECS
        }
    runtime_manifest, runtime_identity = publisher.runtime_manifest(runtime_resources)
    relative, manifest, worker = rollover._worker_value(
        layout, width, commit, digest, "infer",
        publisher.LEGACY_WORKER_MANIFEST_SCHEMA_VERSION, ["infer"],
        runtime_identity)
    worker["worker_rule"] = "historical"
    worker["selection_priority"] = 0

    artifact_root.mkdir(parents=True, exist_ok=True)
    artifact_root = artifact_root.resolve(strict=True)
    with (artifact_root / ".publish.lock").open("a+b") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        registry_path = artifact_root / "registry.json"
        if not registry_path.exists():
            raise publisher.PublishError(
                "historical inference worker requires an existing current registry")
        registry = publisher.load_registry(registry_path)
        if registry["current_layout"] is None:
            raise publisher.PublishError(
                "historical inference worker requires an existing current registry")
        if any(entry["semantic_layout"] == layout and
               entry["model_input_width"] == width and
               entry["worker_role"] == "infer"
               for entry in registry["workers"]):
            raise publisher.PublishError(
                "historical inference binding already exists for target layout and width")

        # Immutable staging precedes registry replacement.  Thus a failed
        # prospective validation/replacement leaves the old registry
        # authoritative, while a retry verifies and reuses its exact address.
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
            artifact_root, inference, relative, manifest, digest, runtime)
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
    parser.add_argument("--inference-executable", required=True, type=Path)
    parser.add_argument("--semantic-layout", required=True, type=int)
    parser.add_argument("--model-input-width", required=True, type=int)
    parser.add_argument("--source-commit", required=True)
    return parser.parse_args()


def main() -> int:
    arguments = parse_arguments()
    artifact_root = arguments.artifact_root.resolve()
    previous_current_layout = publisher.load_registry(
        artifact_root / "registry.json")["current_layout"]
    staged = publish_historical_inference_worker(
        artifact_root, arguments.inference_executable,
        arguments.semantic_layout, arguments.model_input_width,
        arguments.source_commit)
    registry = publisher.load_registry(artifact_root / "registry.json")
    worker = next(entry for entry in registry["workers"]
                  if entry["executable"] == str(staged.relative_to(artifact_root)))
    print("Historical INFER worker published: "
          f"semantic_layout={worker['semantic_layout']},"
          f"model_input_width={worker['model_input_width']},"
          f"worker_role={worker['worker_role']},"
          f"worker_rule={worker['worker_rule']},"
          f"source_commit={worker['source_commit']},"
          f"executable_sha256={worker['sha256']},"
          f"runtime_identity={worker['runtime_identity']},"
          f"current_layout={registry['current_layout']},"
          "current_layout_preserved="
          f"{str(registry['current_layout'] == previous_current_layout).lower()}")
    print(f"Historical INFER worker path: {staged}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, publisher.PublishError, subprocess.SubprocessError) as error:
        raise SystemExit(f"PublishHistoricalInferenceWorker.py: {error}")
