#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/Release/Phase16CorrectedCausalSurpriseReplicationMaterializationHardening/Build/Products/Release/LSTM_Release}"
test_root="${EA_PHASE16_TEST_ROOT:-${repo_root}/DerivedData/Validation/Phase16CorrectedCausalSurpriseReplicationMaterializationHardening/Tests}"
build_dir="${test_root}/MaterializationConcurrency"
fixture_binary="${build_dir}/fixture"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-phase16-materialization.XXXXXX")"
database_name="ea_phase16_materialization_${$}_${RANDOM}"
database_created=false
db_host=127.0.0.1
db_port="${LSTM_DB_PORT:-5432}"
db_admin_user="${LSTM_DB_ADMIN_USER:-${USER}}"
source_database="${LSTM_DB_NAME:-LSTM}"
plan_hash=fnv1a64:cbbf9367c12dd722

if [[ ! "${database_name}" =~ ^ea_phase16_materialization_[0-9]+_[0-9]+$ ]]; then
    echo "unsafe disposable database name: ${database_name}" >&2
    exit 1
fi
if [[ ! -x "${scheduler_binary}" ]]; then
    echo "missing scheduler binary: ${scheduler_binary}" >&2
    exit 1
fi

createdb_cmd=(createdb -h "${db_host}" -p "${db_port}" -U "${db_admin_user}")
dropdb_cmd=(dropdb -h "${db_host}" -p "${db_port}" -U "${db_admin_user}")
psql_admin=(psql -X -v ON_ERROR_STOP=1 -q -h "${db_host}" -p "${db_port}" -U "${db_admin_user}")
pg_dump_admin=(pg_dump -s -h "${db_host}" -p "${db_port}" -U "${db_admin_user}")
psql_fixture=(psql -X -v ON_ERROR_STOP=1 -Atq -h "${db_host}" -p "${db_port}" -U pqxx)

diagnose() {
    local status="$1"
    if [[ "${status}" -ne 0 ]]; then
        for output in "${test_dir}"/*.out; do
            [[ -f "${output}" ]] || continue
            echo "--- ${output} ---" >&2
            tail -100 "${output}" >&2 || true
        done
        if [[ "${database_created}" == true ]]; then
            "${psql_fixture[@]}" -d "${database_name}" -c \
                "SELECT experiment_id,symbol,prediction_horizon,feature_ablation_mask,scheduler_priority,continuation_policy_enabled,invocation_mode FROM experiment ORDER BY experiment_id;" \
                >&2 || true
        fi
    fi
    return "${status}"
}

cleanup() {
    local status=$?
    diagnose "${status}" || true
    if [[ "${database_created}" == true ]]; then
        "${dropdb_cmd[@]}" --if-exists "${database_name}" >/dev/null
        database_created=false
    fi
    rm -rf -- "${test_dir}"
    exit "${status}"
}
trap cleanup EXIT

mkdir -p "${build_dir}"
if command -v pkg-config >/dev/null 2>&1 && pkg-config --exists libpqxx; then
    read -r -a pqxx_cflags <<<"$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_libs <<<"$(pkg-config --libs libpqxx)"
else
    pqxx_prefix="$(brew --prefix libpqxx)"
    libpq_prefix="$(brew --prefix libpq)"
    pqxx_cflags=("-I${pqxx_prefix}/include" "-I${libpq_prefix}/include")
    pqxx_libs=("-L${pqxx_prefix}/lib" "-L${libpq_prefix}/lib" -lpqxx -lpq)
fi

clang++ -std=c++20 -Wall -Wextra -Werror \
    "${pqxx_cflags[@]}" -I "${repo_root}/Sources" -I "${repo_root}/Headers" \
    "${repo_root}/Tests/CorrectedCausalSurpriseReplicationMaterializationFixture.cpp" \
    "${repo_root}/Sources/InferenceProfitability.cpp" \
    "${repo_root}/Sources/InferenceProfitabilityRepository.cpp" \
    "${pqxx_libs[@]}" -o "${fixture_binary}"

"${createdb_cmd[@]}" "${database_name}"
database_created=true
"${pg_dump_admin[@]}" -d "${source_database}" | \
    "${psql_admin[@]}" -d "${database_name}"

PGOPTIONS= PGHOST="${db_host}" PGPORT="${db_port}" \
    LSTM_DB_HOST="${db_host}" LSTM_DB_PORT="${db_port}" \
    LSTM_DB_USER="${db_admin_user}" LSTM_DB_NAME="${database_name}" \
    "${fixture_binary}" seed-anchor \
    >"${test_dir}/fixture.out" 2>&1

materialize=(
    "${scheduler_binary}"
    "--materialize-corrected-causal-surprise-replication=624:625"
    "--corrected-causal-surprise-anchor-pair=624:625"
    "--expected-corrected-replication-plan-hash=${plan_hash}"
    --yes
)

# Hold the same table lock acquired immediately before the mutating identity
# checks.  Both serializable materializers must finish their authoritative
# reads and block behind this disposable-database lock before either may write.
# Releasing it then deterministically exercises the competing-snapshot path
# that PostgreSQL resolves by aborting one transaction with SQLSTATE 40001.
"${psql_fixture[@]}" -d "${database_name}" -c \
    "BEGIN; LOCK TABLE economic_calendar_snapshot IN SHARE ROW EXCLUSIVE MODE; SELECT pg_sleep(5); COMMIT;" \
    >"${test_dir}/lock-holder.out" 2>&1 &
lock_holder_pid=$!
holder_ready=false
for _ in {1..100}; do
    granted_locks="$("${psql_fixture[@]}" -d "${database_name}" -c \
        "SELECT count(*) FROM pg_locks locks
          JOIN pg_class relation ON relation.oid=locks.relation
         WHERE relation.relname='economic_calendar_snapshot'
           AND locks.mode='ShareRowExclusiveLock' AND locks.granted;")"
    if [[ "${granted_locks}" -eq 1 ]]; then
        holder_ready=true
        break
    fi
    sleep 0.05
done
[[ "${holder_ready}" == true ]]

set +e
PGOPTIONS= PGHOST="${db_host}" PGPORT="${db_port}" \
    LSTM_DB_HOST="${db_host}" LSTM_DB_PORT="${db_port}" \
    LSTM_DB_NAME="${database_name}" "${materialize[@]}" \
    >"${test_dir}/materializer-1.out" 2>&1 &
pid_one=$!
PGOPTIONS= PGHOST="${db_host}" PGPORT="${db_port}" \
    LSTM_DB_HOST="${db_host}" LSTM_DB_PORT="${db_port}" \
    LSTM_DB_NAME="${database_name}" "${materialize[@]}" \
    >"${test_dir}/materializer-2.out" 2>&1 &
pid_two=$!
both_blocked=false
for _ in {1..100}; do
    blocked_materializers="$("${psql_fixture[@]}" -d "${database_name}" -c \
        "SELECT count(*) FROM pg_stat_activity
         WHERE datname=current_database()
           AND wait_event_type='Lock'
           AND query LIKE 'LOCK TABLE economic_calendar_snapshot%';")"
    if [[ "${blocked_materializers}" -eq 2 ]]; then
        both_blocked=true
        break
    fi
    sleep 0.05
done
wait "${pid_one}"
rc_one=$?
wait "${pid_two}"
rc_two=$?
wait "${lock_holder_pid}"
rc_holder=$?
set -e

[[ "${both_blocked}" == true ]]
[[ "${rc_holder}" -eq 0 ]]
[[ "${rc_one}" -eq 0 ]]
[[ "${rc_two}" -eq 0 ]]
if grep -Eiq \
    'could not serialize|SQLSTATE[^[:alnum:]]*40001|EXPERIMENT_DATABASE_ERROR' \
    "${test_dir}"/materializer-*.out; then
    echo "raw serialization failure leaked to materialization output" >&2
    exit 1
fi

materialized_count="$(grep -h -c \
    '^CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZED,' \
    "${test_dir}"/materializer-*.out | awk '{sum += $1} END {print sum + 0}')"
already_count="$(grep -h -c \
    '^CORRECTED_CAUSAL_SURPRISE_REPLICATION_ALREADY_MATERIALIZED,' \
    "${test_dir}"/materializer-*.out | awk '{sum += $1} END {print sum + 0}')"
[[ "${materialized_count}" -eq 1 ]]
[[ "${already_count}" -eq 1 ]]

materialized_output="$(grep -h \
    '^CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZED,' \
    "${test_dir}"/materializer-*.out)"
already_output="$(grep -h \
    '^CORRECTED_CAUSAL_SURPRISE_REPLICATION_ALREADY_MATERIALIZED,' \
    "${test_dir}"/materializer-*.out)"
[[ "${materialized_output}" == *",experiment_rows_created=2" ]]
[[ "${already_output}" == *",experiment_rows_created=0" ]]

row_count="$("${psql_fixture[@]}" -d "${database_name}" -c \
    "SELECT count(*) FROM experiment WHERE experiment_id > 625;")"
[[ "${row_count}" -eq 2 ]]

actual_rows="$("${psql_fixture[@]}" -d "${database_name}" -F '|' -c \
    "SELECT symbol,prediction_horizon,feature_ablation_mask,
            model_input_width,model_input_semantic_layout_version,
            economic_calendar_snapshot_id,economic_calendar_snapshot_hash,
            scheduler_priority,continuation_policy_enabled,invocation_mode
       FROM experiment
      WHERE experiment_id > 625
      ORDER BY feature_ablation_mask,experiment_id;")"
control_provenance="corrected_causal_surprise_replication_materialization_v1;plan_hash=${plan_hash};plan_semantic_version=1;scientific_policy_version=1;outcome_blind=true;pair_ordinal=1;replication_unit_hash=fnv1a64:5ffbab7aec47d361;arm_role=control;arm_identity_hash=fnv1a64:7ef5cbe8b77a9d80;"
treatment_provenance="corrected_causal_surprise_replication_materialization_v1;plan_hash=${plan_hash};plan_semantic_version=1;scientific_policy_version=1;outcome_blind=true;pair_ordinal=1;replication_unit_hash=fnv1a64:5ffbab7aec47d361;arm_role=treatment;arm_identity_hash=fnv1a64:028e04da36b5bf55;"
expected_rows=$(printf '%s\n%s' \
    "gbpusdrmp|6||77|7|1|fnv1a64:67610f94f5c8e7cc|high|f|${control_provenance}" \
    "gbpusdrmp|6|causal_first_release_surprise_available,causal_first_release_surprise|77|7|1|fnv1a64:67610f94f5c8e7cc|high|f|${treatment_provenance}")
[[ "${actual_rows}" == "${expected_rows}" ]]

echo "CORRECTED_CAUSAL_SURPRISE_REPLICATION_MATERIALIZATION_CONCURRENCY_TEST_PASS,materialized=1,already_materialized=1,experiment_rows=2,plan_hash=${plan_hash},replication_unit_hash=fnv1a64:5ffbab7aec47d361"
