#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_db="ea_training_objective_migration_${$}"

case "${test_db}" in
    ea_training_objective_migration_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
(
    cd "${repo_root}/Tests"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f TrainingObjectiveMigrationTests.sql
)

printf '%s\n' 'TrainingObjectiveMigrationTests passed'
