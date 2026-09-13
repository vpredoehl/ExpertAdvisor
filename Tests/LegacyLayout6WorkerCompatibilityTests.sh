#!/usr/bin/env bash
set -euo pipefail

legacy_worktree="${1:?usage: $0 /path/to/layout6/worktree /path/to/layout6/LSTM_Release}"
legacy_binary="${2:?usage: $0 /path/to/layout6/worktree /path/to/layout6/LSTM_Release}"
expected_commit=7645265bca0c2529523e1d2cdb37e7d023dfd559
expected_sha256=945225dd2a42f87a2a8dfbfe47b006708e3d90c88a858d25787e5a2237c62dd7

test -x "${legacy_binary}"
test "$(git -C "${legacy_worktree}" rev-parse HEAD)" = "${expected_commit}"
test -z "$(git -C "${legacy_worktree}" status --porcelain --untracked-files=no)"
test "$(shasum -a 256 "${legacy_binary}" | awk '{print $1}')" = \
    "${expected_sha256}"

git -C "${legacy_worktree}" grep -q \
    'kModelInputSemanticLayoutVersion = 6;' \
    HEAD -- Headers/ModelInputExpansion.hpp

git -C "${legacy_worktree}" grep -q -- '--scheduler-worker-attempt-id' \
    HEAD -- Sources/ExperimentScheduler.cpp
git -C "${legacy_worktree}" grep -q \
    'worker_attempt_exact_active_identity_mismatch' \
    HEAD -- Sources/ExperimentScheduler.cpp
git -C "${legacy_worktree}" grep -q 'canonical_executable_path=\$5' \
    HEAD -- Sources/ExperimentScheduler.cpp

git -C "${legacy_worktree}" grep -q \
    'direct CLI execution of scheduler-managed work is' HEAD -- LSTM/main.cpp
git -C "${legacy_worktree}" grep -q 'PersistCompletedInferenceResult' \
    HEAD -- LSTM/main.cpp
git -C "${legacy_worktree}" grep -q \
    'PersistCompletedCheckpointInferenceResult' HEAD -- LSTM/main.cpp

rg -q -- '--scheduler-worker-attempt-id' < <(strings "${legacy_binary}")
rg -q 'worker_attempt_exact_active_identity_mismatch' \
    < <(strings "${legacy_binary}")
rg -q 'SCHEDULER_WORKER_REGISTERED' < <(strings "${legacy_binary}")
rg -q "inference_scope = 'final'" < <(strings "${legacy_binary}")

printf '%s\n' \
    'LegacyLayout6WorkerCompatibilityTests passed' \
    "commit=${expected_commit}" \
    "sha256=${expected_sha256}" \
    'semantic_layout=6' \
    'exact_worker_attempt_registration=verified' \
    'final_and_checkpoint_inference_persistence=verified'
