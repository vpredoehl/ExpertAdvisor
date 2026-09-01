#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_db="ea_model_input_identity_migration_${$}"

case "${test_db}" in
    ea_model_input_identity_migration_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"

# This row exists before 089 and must retain the legacy NULL/NULL identity.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(
    symbol,prediction_horizon,c_next_threshold,target_epochs,
    checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce)
VALUES(
    'phase20legacy',4,0.0008,1,0,'2020-01-01','2020-01-02',
    'completed','done',2089000);
SQL

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/089_lstm_model_input_identity.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/LSTMModelInputIdentityMigrationTests.sql"

test "$(psql -X -At -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE symbol='phase20legacy' AND model_input_width IS NULL AND model_input_semantic_layout_version IS NULL")" = 1

printf '%s\n' 'LSTMModelInputIdentityMigrationTests passed'
