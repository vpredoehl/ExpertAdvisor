#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
expected_commit=7645265bca0c2529523e1d2cdb37e7d023dfd559
expected_sha256=945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7
artifact_root="${repo_root}/Builds/SemanticWorkers"
legacy_binary="${artifact_root}/layout6/${expected_commit}/${expected_sha256}/LSTM_Release"
manifest="$(dirname "${legacy_binary}")/manifest.json"
registry="${artifact_root}/registry.json"

test -x "${legacy_binary}"
test "$(shasum -a 256 "${legacy_binary}" | awk '{print $1}')" = \
    "${expected_sha256}"
test -f "${manifest}"
test -f "${registry}"

/usr/bin/python3 - "${registry}" "${manifest}" "${legacy_binary}" \
    "${expected_commit}" "${expected_sha256}" <<'PY'
import json
from pathlib import Path
import sys

registry_path, manifest_path, binary_path, commit, digest = sys.argv[1:]
registry = json.loads(Path(registry_path).read_text(encoding="utf-8"))
manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
matches = [entry for entry in registry["workers"]
           if entry["semantic_layout"] == 6]
assert len(matches) == 1
entry = matches[0]
assert entry["worker_rule"] == "historical"
assert entry["model_input_width"] == 77
assert entry["source_commit"] == commit
assert entry["sha256"] == digest
assert entry["capabilities"] == ["infer"]
assert (Path(registry_path).parent / entry["executable"]).resolve() == Path(binary_path).resolve()
assert manifest["semantic_layout"] == 6
assert manifest["model_input_width"] == 77
assert manifest["source_commit"] == commit
assert manifest["sha256"] == digest
assert manifest["storage"] == "immutable"
PY

rg -q -- '--scheduler-worker-attempt-id' < <(strings "${legacy_binary}")
rg -q 'worker_attempt_exact_active_identity_mismatch' \
    < <(strings "${legacy_binary}")
rg -q 'SCHEDULER_WORKER_REGISTERED' < <(strings "${legacy_binary}")
rg -q "inference_scope = 'final'" < <(strings "${legacy_binary}")

if [[ $# -ne 0 ]]; then
    if [[ $# -ne 2 ]]; then
        echo "usage: $0 [/path/to/layout6/worktree /path/to/source/LSTM_Release]" >&2
        exit 2
    fi
    legacy_worktree="$1"
    source_binary="$2"
    test "$(git -C "${legacy_worktree}" rev-parse HEAD)" = "${expected_commit}"
    test -z "$(git -C "${legacy_worktree}" status --porcelain --untracked-files=no)"
    test "$(shasum -a 256 "${source_binary}" | awk '{print $1}')" = \
        "${expected_sha256}"
    cmp -s "${legacy_binary}" "${source_binary}"
fi

printf '%s\n' \
    'LegacyLayout6WorkerCompatibilityTests passed' \
    "commit=${expected_commit}" \
    "sha256=${expected_sha256}" \
    'semantic_layout=6' \
    "archived_executable=${legacy_binary}" \
    'registry_route=verified' \
    'exact_worker_attempt_registration=verified' \
    'final_and_checkpoint_inference_persistence=verified'
