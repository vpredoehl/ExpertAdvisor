#!/usr/bin/env bash
set -euo pipefail
repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase_pg.XXXXXX")"
test_db="ea_phase_priority_${$}"
trap 'dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true; rm -rf "${test_dir}"' EXIT
createdb "${test_db}"
read -r -a compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a link_flags <<<"$(pkg-config --libs libpqxx)"
"${CXX:-clang++}" -std=c++20 -Wall -Wextra -Werror -Wno-deprecated-declarations \
    -I"${repo_root}/Sources" "${compile_flags[@]}" \
    "${repo_root}/Tests/SchedulerPhasePriorityPostgresTests.cpp" \
    "${link_flags[@]}" -o "${test_dir}/phase_pg"
"${test_dir}/phase_pg" "dbname=${test_db}" unmigrated
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -f "${repo_root}/Database/migrations/099_scheduler_phase_priority.sql"
# Runtime role grants are part of the migration contract.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "SET ROLE pqxx; SELECT phase_priority FROM experiment_scheduler_phase_policy; UPDATE experiment_scheduler_phase_policy SET phase_priority='concurrent';"
"${test_dir}/phase_pg" "dbname=${test_db}" migrated
# Migration replay preserves a live operator setting.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -f "${repo_root}/Database/migrations/099_scheduler_phase_priority.sql"
test "$(psql -X -Atq -d "${test_db}" -c 'SELECT phase_priority FROM experiment_scheduler_phase_policy')" = 'infer:analyze:train'
printf '%s\n' 'Phase policy migration replay passed; production rows changed=0'
