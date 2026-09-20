#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
scheduler="${products}/lstm-scheduler"
analyzer="${products}/lstm-analyze-worker"

test -x "${scheduler}"
test -x "${analyzer}"
test ! -L "${scheduler}"
test ! -L "${analyzer}"

identity_output="$("${scheduler}" --build-identity)"
expected_scheduler="$(realpath "${scheduler}")"
expected_sha256="sha256:$(/usr/bin/shasum -a 256 -- "${scheduler}" | awk '{print $1}')"

printf '%s\n' "${identity_output}" | rg -Fq 'SCHEDULER_BUILD_IDENTITY,identity_contract_version=1,artifact_role=lstm-scheduler,'
printf '%s\n' "${identity_output}" | rg -Fq "source_commit=$(git -C "${repo_root}" rev-parse HEAD)"
printf '%s\n' "${identity_output}" | rg -Fq "canonical_executable=${expected_scheduler}"
printf '%s\n' "${identity_output}" | rg -Fq "executable_sha256=${expected_sha256}"

set +e
analyzer_output="$("${analyzer}" 2>&1)"
analyzer_status=$?
set -e
test "${analyzer_status}" = 1
printf '%s\n' "${analyzer_output}" | rg -Fq -- '--analyze-experiment=EXPERIMENT_ID'

printf '%s\n' \
    "StandaloneSchedulerReleaseBundleTests passed" \
    "scheduler=${expected_scheduler}" \
    "analyzer=$(realpath "${analyzer}")"
