#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
source "${repo_root}/Tests/DatabaseTestIsolation.sh"
require_disposable_test_database
db_host="${LSTM_TEST_DB_HOST:-127.0.0.1}"
db_name="${LSTM_TEST_DB_NAME}"
db_admin_user="${LSTM_TEST_DB_ADMIN_USER:-${USER}}"

psql -X -v ON_ERROR_STOP=1 -h "${db_host}" -U "${db_admin_user}" \
    -d "${db_name}" \
    -f "${repo_root}/Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql"
