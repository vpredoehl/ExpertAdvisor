#!/usr/bin/env python3
"""Create disposable semantic workers for scheduler recovery integration tests."""

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

if len(sys.argv) != 2:
    raise SystemExit("usage: make_scheduler_recovery_registry.py OUTPUT_DIRECTORY")

root = Path(sys.argv[1]).resolve()
root.mkdir(parents=True, exist_ok=True)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)


commit = subprocess.check_output(
    ["git", "rev-parse", "HEAD"], text=True
).strip()

resources = []

for built, name in (
    ("MetaNN_metal.metallib", "MetaNN.metallib"),
    ("default.metallib", "default.metallib"),
):
    data = f"phase24g-disposable-{name}\n".encode()

    resources.append({
        "built_identity": built,
        "runtime_name": name,
        "sha256": sha(data),
    })

runtime_manifest = {
    "resources": resources,
    "schema_version": 1,
    "storage": "immutable",
}

runtime_bytes = json.dumps(
    runtime_manifest,
    sort_keys=True,
    separators=(",", ":"),
).encode()

runtime_id = sha(runtime_bytes)
runtime_dir = root / "runtime" / runtime_id

for resource in resources:
    name = resource["runtime_name"]
    write(runtime_dir / name, f"phase24g-disposable-{name}\n".encode())

write(runtime_dir / "manifest.json", runtime_bytes)

workers = []

for role, filename, capabilities in (
    ("train", "LSTM_Release", ["train"]),
    ("infer", "lstm-infer-worker", ["infer"]),
):
    executable_bytes = (
        b"#!/bin/sh\n"
        b"echo 'PHASE24G_TEST_WORKER_MUST_NOT_LAUNCH' >&2\n"
        b"exit 99\n"
    )

    digest = sha(executable_bytes)
    relative = Path("layout9") / role / commit / digest
    directory = root / relative
    executable = directory / filename

    write(executable, executable_bytes)
    executable.chmod(0o700)

    for resource in resources:
        name = resource["runtime_name"]
        target = os.path.relpath(runtime_dir / name, directory)
        (directory / name).symlink_to(target)

    manifest = {
        "schema_version": 2,
        "semantic_layout": 9,
        "storage": "immutable",
        "model_input_width": 103,
        "source_commit": commit,
        "sha256": digest,
        "executable_identity": filename,
        "worker_role": role,
        "capabilities": capabilities,
    }

    write(
        directory / "manifest.json",
        json.dumps(manifest, sort_keys=True).encode(),
    )

    workers.append({
        "semantic_layout": 9,
        "worker_role": role,
        "artifact_manifest_schema_version": 2,
        "worker_rule": "current",
        "model_input_width": 103,
        "source_commit": commit,
        "sha256": digest,
        "executable": str(relative / filename),
        "manifest": str(relative / "manifest.json"),
        "runtime_identity": runtime_id,
        "capabilities": capabilities,
    })

registry = {
    "schema_version": 4,
    "current_layout": 9,
    "runtimes": [{
        "identity": runtime_id,
        "directory": f"runtime/{runtime_id}",
        "manifest": f"runtime/{runtime_id}/manifest.json",
    }],
    "workers": workers,
}

write(
    root / "registry.json",
    (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode(),
)

print(root / "registry.json")
