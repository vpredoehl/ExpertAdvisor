#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_db="ea_scheduler_repository_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_scheduler_repository.XXXXXX")"

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

createdb "${test_db}"
read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"

clang++ -std=c++20 -Wall -Wextra -Werror \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" \
    "${pqxx_compile_flags[@]}" \
    "${repo_root}/Sources/SchedulerCore/SchedulerPolicy.cpp" \
    "${repo_root}/Sources/SchedulerCore/SchedulerRepository.cpp" \
    "${repo_root}/Sources/SchedulerCore/PostgresSchedulerRepository.cpp" \
    "${repo_root}/Tests/PostgresSchedulerRepositoryTests.cpp" \
    "${pqxx_link_flags[@]}" \
    -o "${test_dir}/PostgresSchedulerRepositoryTests"

"${test_dir}/PostgresSchedulerRepositoryTests" "dbname=${test_db}"
printf '%s\n' "PostgresSchedulerRepositoryTests passed"
