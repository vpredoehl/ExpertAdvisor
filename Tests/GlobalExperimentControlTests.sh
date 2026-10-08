#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_area="${TMPDIR:-${repo_root}/DerivedData/ExpertAdvisor/Phase24T/tmp}"
mkdir -p "${test_area}"
test_dir="$(mktemp -d "${test_area}/ea-control-unit.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT
export TMPDIR="${test_dir}"
read -r -a compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a link_flags <<<"$(pkg-config --libs libpqxx)"
# GlobalExperimentControl contains an existing unused private helper. The
# targeted control unit build keeps all other warnings fatal.
/usr/bin/clang++ -std=c++20 -O0 -Wall -Wextra -Werror \
    -Wno-unused-function -Wno-deprecated-declarations \
    -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${compile_flags[@]}" \
    "${repo_root}/Tests/GlobalExperimentControlTests.cpp" \
    "${repo_root}/Sources/GlobalExperimentControl.cpp" \
    "${repo_root}/Sources/CheckpointPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/CheckpointEvaluationService.cpp" \
    "${repo_root}/Sources/SchedulerCore/PostgresSchedulerRepository.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerOperationalObservation.cpp" \
    "${link_flags[@]}" -o "${test_dir}/GlobalExperimentControlTests"
"${test_dir}/GlobalExperimentControlTests"
