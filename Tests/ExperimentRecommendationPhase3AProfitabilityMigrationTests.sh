#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_name="${LSTM_DB_NAME:-LSTM}"
db_admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"

psql -X -v ON_ERROR_STOP=1 -h "${db_host}" -U "${db_admin_user}" \
    -d "${db_name}" \
    -f "${repo_root}/Tests/ExperimentRecommendationPhase3AProfitabilityMigrationTests.sql"
