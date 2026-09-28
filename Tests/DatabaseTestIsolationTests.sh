#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d /tmp/ea_database_test_isolation.XXXXXX)"
trap 'rm -rf -- "${test_dir}"' EXIT
source "${repo_root}/Tests/DatabaseTestIsolation.sh"

if LSTM_TEST_DB_NAME= require_disposable_test_database 2>/dev/null; then
    printf '%s\n' 'shell guard accepted missing target unexpectedly' >&2
    exit 1
fi
if LSTM_TEST_DB_NAME=LSTM require_disposable_test_database 2>/dev/null; then
    printf '%s\n' 'shell guard accepted LSTM unexpectedly' >&2
    exit 1
fi
LSTM_TEST_DB_NAME=ea_isolation_test_123 require_disposable_test_database

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Tests" \
    -I/opt/homebrew/opt/libpqxx@7.10.1/include \
    "${repo_root}/Tests/DatabaseTestIsolationTests.cpp" \
    -L/opt/homebrew/opt/libpqxx@7.10.1/lib \
    -L/opt/homebrew/opt/libpq/lib -lpqxx -lpq \
    -o "${test_dir}/DatabaseTestIsolationTests"

"${test_dir}/DatabaseTestIsolationTests"

if output=$(LSTM_TEST_DB_NAME=LSTM \
    bash "${repo_root}/Tests/ExperimentRecommendationPhase3ARepositoryTests.sh" \
    2>&1); then
    printf '%s\n' 'Phase3A repository test accepted LSTM unexpectedly' >&2
    exit 1
fi
grep -q 'clearly_disposable_non_LSTM_database_required' <<<"${output}"

if output=$(env -u LSTM_TEST_DB_NAME \
    bash "${repo_root}/Tests/ExperimentRecommendationPhase3ARepositoryTests.sh" \
    2>&1); then
    printf '%s\n' 'Phase3A repository test accepted missing target unexpectedly' >&2
    exit 1
fi
grep -q 'LSTM_TEST_DB_NAME_required' <<<"${output}"

printf '%s\n' 'DATABASE_TEST_ISOLATION_TESTS=passed'
