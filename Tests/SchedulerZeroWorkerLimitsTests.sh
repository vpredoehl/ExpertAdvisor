#!/bin/bash

set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_zero_worker_limits_tests.XXXXXX")"
trap 'rm -rf "${test_dir}"' EXIT

clang++ \
    -std=c++20 \
    -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" \
    "${repo_root}/Tests/SchedulerZeroWorkerLimitsTests.cpp" \
    -o "${test_dir}/SchedulerZeroWorkerLimitsTests"

"${test_dir}/SchedulerZeroWorkerLimitsTests"

# Keep unrelated positive-only scheduler parsing isolated from the worker helper.
grep -Fq \
    'options.schedulerPollSeconds = ParsePositiveInt(arg, RequireNextArg(argc, argv, i, arg));' \
    "${repo_root}/Sources/ExperimentScheduler.cpp"
grep -Fq \
    'options.schedulerPollSeconds = ParsePositiveInt("--scheduler-poll-seconds", value);' \
    "${repo_root}/Sources/ExperimentScheduler.cpp"

if [[ $# -eq 1 ]]; then
    scheduler_binary="$1"
    test -x "${scheduler_binary}"
    safe_db_name="ea_zero_worker_limits_cli_no_connection"
    [[ "${safe_db_name}" =~ ^ea_zero_worker_limits_[A-Za-z0-9_]+$ ]]

    expect_valid() {
        LSTM_DB_NAME="${safe_db_name}" \
            "${scheduler_binary}" --help "$@" >/dev/null 2>&1
    }

    expect_invalid() {
        if LSTM_DB_NAME="${safe_db_name}" \
            "${scheduler_binary}" --help "$@" >"${test_dir}/invalid.out" 2>&1
        then
            echo "expected invalid scheduler arguments: $*" >&2
            return 1
        fi
        grep -q "Argument error:" "${test_dir}/invalid.out"
    }

    expect_valid --max-train-procs=0
    expect_valid --max-infer-procs=0
    expect_valid --max-analyze-procs=0
    expect_valid --max-train-procs 0
    expect_valid --max-infer-procs 0
    expect_valid --max-analyze-procs 0
    expect_valid --max-train-procs=2147483647

    expect_invalid --max-train-procs=-1
    expect_invalid --max-infer-procs=-2
    expect_invalid --max-analyze-procs=1x
    expect_invalid --max-train-procs=
    expect_invalid --max-infer-procs=2147483648

    # This unrelated option remains on ParsePositiveInt and still rejects zero.
    expect_invalid --scheduler-poll-seconds=0
fi
