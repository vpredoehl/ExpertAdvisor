#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
fixture_root="${repo_root}/Tests/fixtures/migration_runner"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
admin_user="${LSTM_DB_ADMIN_USER:-${USER:-vjp}}"
maintenance_db="${LSTM_DB_MAINTENANCE_DB:-postgres}"
tmp_root="$(mktemp -d "${TMPDIR:-/tmp}/ea-migration-runner.XXXXXX")"
databases=()

cleanup() {
    local database=""
    for database in "${databases[@]}"; do
        dropdb -h "${db_host}" -U "${admin_user}" \
            --maintenance-db="${maintenance_db}" --if-exists \
            "${database}" >/dev/null 2>&1 || true
    done
    rm -rf "${tmp_root}"
}
trap cleanup EXIT

new_database() {
    local label="$1"
    database_created="ea_migration_runner_${label}_${$}"
    createdb -h "${db_host}" -U "${admin_user}" \
        --maintenance-db="${maintenance_db}" --template=template0 \
        "${database_created}"
    databases+=("${database_created}")
}

psql_database() {
    local database="$1"
    shift
    psql -X -q -v ON_ERROR_STOP=1 -h "${db_host}" \
        -U "${admin_user}" -d "${database}" "$@"
}

make_runner() {
    local label="$1"
    shift
    local runner_root="${tmp_root}/${label}"
    mkdir -p "${runner_root}/Database/migrations"
    cp "${repo_root}/migrate_lstm_db.sh" "${runner_root}/migrate_lstm_db.sh"
    cp "$@" "${runner_root}/Database/migrations/"
    printf '%s\n' "${runner_root}"
}

run_runner() {
    local database="$1"
    local runner_root="$2"
    LSTM_DB_HOST="${db_host}" LSTM_DB_NAME="${database}" \
        LSTM_DB_ADMIN_USER="${admin_user}" \
        bash "${runner_root}/migrate_lstm_db.sh"
}

table_exists() {
    local database="$1" table_name="$2"
    psql_database "${database}" -Atc \
        "SELECT to_regclass('public.${table_name}') IS NOT NULL;"
}

ledger_count() {
    local database="$1" version="$2"
    psql_database "${database}" -Atc \
        "SELECT count(*) FROM schema_migrations WHERE version='${version}';"
}

# A, E, F: success records the exact checksum, matching replay skips, and a
# changed applied migration fails closed.
new_database success
success_db="${database_created}"
success_runner="$(make_runner success "${fixture_root}/success/010_success.sql")"
success_log="${tmp_root}/success.log"
run_runner "${success_db}" "${success_runner}" | tee "${success_log}"
success_checksum="$(shasum -a 256 \
    "${fixture_root}/success/010_success.sql" | awk '{print $1}')"
[[ "$(table_exists "${success_db}" migration_runner_success_effect)" == t ]]
[[ "$(psql_database "${success_db}" -Atc \
    "SELECT filename || '|' || checksum FROM schema_migrations WHERE version='010';")" == \
    "010_success.sql|${success_checksum}" ]]
run_runner "${success_db}" "${success_runner}" > "${tmp_root}/success-replay.log"
grep -Fq 'MIGRATION_SKIP,version=010,filename=010_success.sql' \
    "${tmp_root}/success-replay.log"
printf '\n-- deliberate checksum mismatch\n' >> \
    "${success_runner}/Database/migrations/010_success.sql"
if run_runner "${success_db}" "${success_runner}" \
    > "${tmp_root}/checksum-mismatch.log" 2>&1; then
    echo "checksum mismatch unexpectedly succeeded" >&2
    exit 1
fi
grep -Fq 'reason=checksum_mismatch' "${tmp_root}/checksum-mismatch.log"

# B: a migration error after a schema statement rolls back both schema and
# ledger state.
new_database failure
failure_db="${database_created}"
failure_runner="$(make_runner failure \
    "${fixture_root}/failure/020_failure.sql")"
if run_runner "${failure_db}" "${failure_runner}" \
    > "${tmp_root}/failure.log" 2>&1; then
    echo "failing migration unexpectedly succeeded" >&2
    exit 1
fi
[[ "$(table_exists "${failure_db}" migration_runner_failure_effect)" == f ]]
[[ "$(ledger_count "${failure_db}" 020)" == 0 ]]

# C: a historical migration-level BEGIN/COMMIT wrapper is normalized while
# the original bytes still determine the ledger checksum.
new_database internal
internal_db="${database_created}"
internal_runner="$(make_runner internal \
    "${fixture_root}/internal/030_internal_transaction.sql")"
run_runner "${internal_db}" "${internal_runner}" > "${tmp_root}/internal.log"
internal_checksum="$(shasum -a 256 \
    "${fixture_root}/internal/030_internal_transaction.sql" | awk '{print $1}')"
[[ "$(table_exists "${internal_db}" migration_runner_internal_effect)" == t ]]
[[ "$(psql_database "${internal_db}" -Atc \
    "SELECT count(*) FROM schema_migrations WHERE version='030' AND checksum='${internal_checksum}';")" == 1 ]]

# D: force the ledger insert itself to fail after all migration schema
# statements. The migration's internal COMMIT must not let either effect
# survive.
new_database ledger_failure
ledger_failure_db="${database_created}"
ledger_failure_runner="$(make_runner ledger_failure \
    "${fixture_root}/ledger_failure/040_ledger_failure.sql")"
if run_runner "${ledger_failure_db}" "${ledger_failure_runner}" \
    > "${tmp_root}/ledger-failure.log" 2>&1; then
    echo "ledger-insert failure unexpectedly succeeded" >&2
    exit 1
fi
[[ "$(table_exists "${ledger_failure_db}" migration_runner_pre_ledger_effect)" == f ]]
[[ "$(ledger_count "${ledger_failure_db}" 040)" == 0 ]]
[[ "$(psql_database "${ledger_failure_db}" -Atc \
    "SELECT count(*) FROM pg_constraint WHERE conname='migration_runner_reject_040_check';")" == 0 ]]

# Any top-level transaction control that is not a matched whole-migration
# wrapper fails before the migration body is sent to PostgreSQL.
new_database transaction_control
transaction_control_db="${database_created}"
transaction_control_runner="$(make_runner transaction_control \
    "${fixture_root}/transaction_control/050_unmatched_commit.sql")"
if run_runner "${transaction_control_db}" "${transaction_control_runner}" \
    > "${tmp_root}/transaction-control.log" 2>&1; then
    echo "unsupported transaction control unexpectedly succeeded" >&2
    exit 1
fi
grep -Fq 'reason=unsupported_transaction_control' \
    "${tmp_root}/transaction-control.log"
[[ "$(table_exists "${transaction_control_db}" migration_runner_unsupported_transaction_effect)" == f ]]
[[ "$(ledger_count "${transaction_control_db}" 050)" == 0 ]]

# G: filename ordering remains deterministic.
new_database order
order_db="${database_created}"
order_runner="$(make_runner order "${fixture_root}"/order/*.sql)"
run_runner "${order_db}" "${order_runner}" > "${tmp_root}/order.log"
[[ "$(psql_database "${order_db}" -Atc \
    "SELECT string_agg(migration_version, ',' ORDER BY order_id) FROM migration_runner_order;")" == \
    "003,010,020" ]]
[[ "$(psql_database "${order_db}" -Atc \
    "SELECT string_agg(version, ',' ORDER BY applied_at, version) FROM schema_migrations;")" == \
    "003,010,020" ]]

# Exercise the checked-in 086 and 087 bytes against a deterministic minimal
# predecessor. Both contain historical transaction wrappers.
new_database real
real_db="${database_created}"
psql_database "${real_db}" -f "${fixture_root}/086_087_predecessor.sql"
real_runner="$(make_runner real \
    "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql" \
    "${repo_root}/Database/migrations/087_campaign_materialization_pause_resume.sql")"
run_runner "${real_db}" "${real_runner}" > "${tmp_root}/real.log"
psql_database "${real_db}" -f \
    "${repo_root}/Tests/CampaignMaterializationControlMigrationTests.sql"
checksum_086="$(shasum -a 256 \
    "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql" | awk '{print $1}')"
checksum_087="$(shasum -a 256 \
    "${repo_root}/Database/migrations/087_campaign_materialization_pause_resume.sql" | awk '{print $1}')"
[[ "$(psql_database "${real_db}" -Atc \
    "SELECT count(*) FROM schema_migrations WHERE (version='086' AND checksum='${checksum_086}') OR (version='087' AND checksum='${checksum_087}');")" == 2 ]]
[[ "$(psql_database "${real_db}" -Atc \
    "SELECT attnotnull AND pg_get_expr(adbin, adrelid) = '''normal''::text' FROM pg_attribute JOIN pg_attrdef ON adrelid=attrelid AND adnum=attnum WHERE attrelid='experiment'::regclass AND attname='scheduler_priority';")" == t ]]
[[ "$(table_exists "${real_db}" experiment_campaign_materialization_control_operation)" == t ]]

# Force the exact 086 ledger insert to fail and prove its schema changes do
# not escape. Then force the exact 087 ledger insert to fail after a separately
# committed 086 and prove only 087 rolls back.
new_database real086failure
real_086_failure_db="${database_created}"
psql_database "${real_086_failure_db}" -f \
    "${fixture_root}/086_087_predecessor.sql"
psql_database "${real_086_failure_db}" <<'SQL'
CREATE TABLE schema_migrations (
    version text PRIMARY KEY,
    filename text NOT NULL,
    checksum text NOT NULL,
    applied_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT reject_086 CHECK (version <> '086')
);
SQL
real_086_runner="$(make_runner real086failure \
    "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql")"
if run_runner "${real_086_failure_db}" "${real_086_runner}" \
    > "${tmp_root}/real-086-failure.log" 2>&1; then
    echo "real migration 086 ledger failure unexpectedly succeeded" >&2
    exit 1
fi
[[ "$(psql_database "${real_086_failure_db}" -Atc \
    "SELECT count(*) FROM pg_attribute WHERE attrelid='experiment'::regclass AND attname IN ('scheduler_priority','resume_requested') AND NOT attisdropped;")" == 0 ]]
[[ "$(ledger_count "${real_086_failure_db}" 086)" == 0 ]]

new_database real087failure
real_087_failure_db="${database_created}"
psql_database "${real_087_failure_db}" -f \
    "${fixture_root}/086_087_predecessor.sql"
real_086_success_runner="$(make_runner real087base \
    "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql")"
run_runner "${real_087_failure_db}" "${real_086_success_runner}" \
    > "${tmp_root}/real-087-base.log"
psql_database "${real_087_failure_db}" -c \
    "ALTER TABLE schema_migrations ADD CONSTRAINT reject_087 CHECK (version <> '087');"
real_087_runner="$(make_runner real087failure \
    "${repo_root}/Database/migrations/087_campaign_materialization_pause_resume.sql")"
if run_runner "${real_087_failure_db}" "${real_087_runner}" \
    > "${tmp_root}/real-087-failure.log" 2>&1; then
    echo "real migration 087 ledger failure unexpectedly succeeded" >&2
    exit 1
fi
[[ "$(table_exists "${real_087_failure_db}" experiment_campaign_materialization_control_operation)" == f ]]
[[ "$(ledger_count "${real_087_failure_db}" 086)" == 1 ]]
[[ "$(ledger_count "${real_087_failure_db}" 087)" == 0 ]]

echo "MIGRATION_RUNNER_ATOMICITY_TESTS_PASS success=PASS rollback=PASS internal_wrapper=PASS ledger_failure=PASS transaction_control=FAIL_CLOSED replay=PASS checksum_mismatch=FAIL_CLOSED order=003,010,020 migration086=PASS migration087=PASS"
