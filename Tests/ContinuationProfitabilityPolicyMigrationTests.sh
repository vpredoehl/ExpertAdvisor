#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
schema="continuation_profitability_policy_test_${$}"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_name="${LSTM_DB_NAME:-LSTM}"
db_admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
psql_admin=(psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -U "${db_admin_user}" -d "${db_name}")

cleanup() {
    "${psql_admin[@]}" -c "DROP SCHEMA IF EXISTS \"${schema}\" CASCADE;" >/dev/null
}
trap cleanup EXIT

"${psql_admin[@]}" <<SQL
CREATE SCHEMA "${schema}";
SET search_path TO "${schema}", public;
CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    continuation_policy_last_decision text,
    CONSTRAINT experiment_continuation_policy_last_decision_check
        CHECK (
            continuation_policy_last_decision IS NULL
            OR continuation_policy_last_decision IN (
                'eligible', 'insufficient_evidence', 'rejected_threshold',
                'rejected_rank', 'rejected_trend', 'already_continued',
                'continuation_queued', 'skipped', 'error'
            )
        )
);
CREATE TABLE experiment_continuation_decision (
    decision text NOT NULL,
    CONSTRAINT experiment_continuation_decision_value_check
        CHECK (
            decision IN (
                'eligible', 'insufficient_evidence', 'rejected_threshold',
                'rejected_rank', 'rejected_trend', 'already_continued',
                'continuation_queued', 'skipped', 'error'
            )
        )
);
\i '${repo_root}/Database/migrations/074_continuation_profitability_policy.sql'
\i '${repo_root}/Database/migrations/074_continuation_profitability_policy.sql'
\i '${repo_root}/Tests/ContinuationProfitabilityPolicyMigrationTests.sql'
SQL

echo "ContinuationProfitabilityPolicyMigrationTests passed"
