#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ $# -ne 1 ]]; then
    echo "Usage: $0 /absolute/path/to/isolated/LSTM_Release" >&2
    exit 2
fi

scheduler_binary="$1"

if [[ "${scheduler_binary}" != /* || ! -x "${scheduler_binary}" ]]; then
    echo "ERROR: An absolute path to an executable Release binary is required." >&2
    exit 2
fi

scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"

case "${scheduler_binary}" in
    "${repo_root}"/DerivedData/*/Build/Products/Release/LSTM_Release) ;;
    *)
        echo "ERROR: Scheduler must be an isolated RepositoryAgent Release build." >&2
        exit 2
        ;;
esac

for command in createdb dropdb psql python3 awk sed grep; do
    command -v "${command}" >/dev/null || {
        echo "ERROR: Missing command: ${command}" >&2
        exit 2
    }
done

test_db="ea_phase24g_recovery_test_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea_phase24g_recovery.XXXXXX")"
db_created=0

cleanup() {
    local status=$?

    if [[ "${status}" -ne 0 ]]; then
        echo "Phase 24G failed. Diagnostic output:" >&2
        for output in "${test_dir}"/scheduler_*.log; do
            [[ -f "${output}" ]] || continue
            echo "=== ${output} ===" >&2
            cat "${output}" >&2
        done
    fi

    if [[ "${db_created}" -eq 1 ]]; then
        dropdb --if-exists "${test_db}" >/dev/null 2>&1 || \
            echo "WARNING: Could not drop disposable database ${test_db}" >&2
    fi

    rm -rf -- "${test_dir}"
    return "${status}"
}

trap cleanup EXIT

scalar() {
    psql -X -v ON_ERROR_STOP=1 -At -q \
        -d "${test_db}" -c "$1"
}

assert_equal() {
    local expected="$1"
    local actual="$2"
    local description="$3"

    if [[ "${actual}" != "${expected}" ]]; then
        echo "FAIL: ${description}" >&2
        echo "Expected: ${expected}" >&2
        echo "Actual:   ${actual}" >&2
        exit 1
    fi

    echo "PASS: ${description}"
}

run_scheduler() {
    local label="$1"
    local output="${test_dir}/scheduler_${label}.log"

    LSTM_DB_NAME="${test_db}" \
        "${scheduler_binary}" \
        --schedule-experiments \
        --scheduler-once \
        --recover-orphans-only \
        "--semantic-worker-registry=${registry}" \
        "--scheduler-log-dir=${test_dir}/logs" \
        >"${output}" 2>&1

    grep -q 'SCHEDULER_STOP,exit_code=0' "${output}" || {
        echo "FAIL: Scheduler did not stop successfully." >&2
        exit 1
    }

    grep -q 'SCHEDULER_OWNERSHIP_RELEASED' "${output}" || {
        echo "FAIL: Scheduler ownership release not observed." >&2
        exit 1
    }

    grep -q 'recover_orphans_only=1' "${output}" || {
        echo "FAIL: Scheduler did not confirm recovery-only mode." >&2
        exit 1
    }

    if grep -q 'PHASE24G_TEST_WORKER_MUST_NOT_LAUNCH' "${output}"; then
        echo "FAIL: Disposable worker was unexpectedly launched." >&2
        exit 1
    fi

    local lease_state
    lease_state="$(scalar         "SELECT authority_state FROM experiment_scheduler_lease
         WHERE singleton=true")"

    assert_equal "released" "${lease_state}"         "Scheduler lease released after ${label}"
}

echo "=== Phase 24G: Interrupted launch recovery integration ==="
echo "Repository: ${repo_root}"
echo "Scheduler:  ${scheduler_binary}"
echo "Database:   ${test_db}"

# The database name is generated internally and cannot name production.
case "${test_db}" in
    ea_phase24g_recovery_test_[0-9]*) ;;
    *) echo "ERROR: Unsafe database name." >&2; exit 90 ;;
esac

createdb "${test_db}"
db_created=1

# Load only the repository snapshot. Never inspect production PostgreSQL.
# The schema dump deliberately clears search_path; restore it for fixtures.
awk '
    /^CREATE TABLE controlled_experiment_family \(/ {
        print "SET search_path TO public;"
    }
    { print }
' "${repo_root}/Database/LSTM_schema.sql" |
    psql -X -v ON_ERROR_STOP=1 -q \
        -d "${test_db}" \
        >"${test_dir}/schema.log" 2>&1 || {
        echo "FAIL: Repository schema initialization." >&2
        tail -80 "${test_dir}/schema.log" >&2
        exit 1
    }

registry="$(python3 \
    "${repo_root}/Tests/fixtures/make_scheduler_recovery_registry.py" \
    "${test_dir}/registry")"

test -f "${registry}"

# The scheduler connects over TCP as pqxx. Grants affect this disposable DB only.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SET search_path TO public;

GRANT USAGE ON SCHEMA public TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE
    ON ALL TABLES IN SCHEMA public TO pqxx;
GRANT USAGE, SELECT, UPDATE
    ON ALL SEQUENCES IN SCHEMA public TO pqxx;

INSERT INTO experiment_scheduler_protocol (
    singleton,
    required_generation,
    cutover_state,
    cutover_completed_at,
    cutover_completed_by,
    cutover_executable_path,
    cutover_process_evidence,
    legacy_no_pid_grace_seconds
)
VALUES (
    true,
    52,
    'complete',
    clock_timestamp() - interval '2 minutes',
    'phase24g-disposable-test',
    '/tmp/phase24g-test-scheduler',
    'isolated-database-fixture',
    30
);

INSERT INTO experiment_global_control(singleton, desired_state)
VALUES (true, 'running');

INSERT INTO experiment_scheduler_lease(singleton)
VALUES (true)
ON CONFLICT(singleton) DO NOTHING;
SQL

# First invocation establishes a real scheduler invocation for the fixture.
run_scheduler "initial"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
BEGIN;

SET LOCAL expertadvisor.scheduler_protocol_generation = '52';

INSERT INTO experiment (
    experiment_id,
    symbol,
    prediction_horizon,
    c_next_threshold,
    target_epochs,
    train_start,
    train_end,
    status,
    phase,
    current_operation,
    duplicate_nonce,
    model_input_width,
    model_input_semantic_layout_version
)
VALUES (
    920056,
    'phase24gtest',
    4,
    0.0008,
    1,
    '2020-01-01',
    '2020-02-01',
    'running',
    'train',
    'train',
    920056,
    103,
    9
);

WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,
        scheduler_invocation_id,
        scheduler_fencing_token,
        experiment_id,
        worker_kind,
        lifecycle_phase,
        capacity_class,
        ownership_origin,
        lifecycle_state,
        canonical_executable_path,
        command_identity,
        reserved_at
    )
    SELECT
        'phase24g-never-spawned-920056',
        scheduler_invocation_id,
        1,
        920056,
        'experiment',
        'train',
        'train',
        'scheduler_launch',
        'reserved',
        '/tmp/phase24g-never-spawned',
        'experiment:920056:train',
        clock_timestamp()
    FROM experiment_scheduler_invocation
    ORDER BY started_at DESC
    LIMIT 1
    RETURNING worker_attempt_id
)
UPDATE experiment
SET active_scheduler_worker_attempt_id = (
    SELECT worker_attempt_id FROM attempt
)
WHERE experiment_id = 920056;

COMMIT;
SQL

attempt_id="$(scalar \
    "SELECT active_scheduler_worker_attempt_id
     FROM experiment WHERE experiment_id=920056")"

[[ "${attempt_id}" =~ ^[0-9]+$ ]] || {
    echo "FAIL: Fixture has no active attempt." >&2
    exit 1
}

# Fresh reservation: recovery must preserve the active binding.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -c "UPDATE experiment_scheduler_worker_attempt
        SET reserved_at=clock_timestamp()
        WHERE worker_attempt_id=${attempt_id}
          AND lifecycle_state='reserved'"

run_scheduler "within_grace"

state="$(scalar \
    "SELECT lifecycle_state || '|' || diagnostic
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${attempt_id}")"

assert_equal \
    "identity_ambiguous|launch_reservation_within_recovery_grace" \
    "${state}" \
    "Reservation remains ambiguous during grace period"

binding="$(scalar \
    "SELECT status || '|' || active_scheduler_worker_attempt_id
     FROM experiment WHERE experiment_id=920056")"

assert_equal \
    "running|${attempt_id}" \
    "${binding}" \
    "Experiment retains active binding during grace period"

# Expired reservation: previously ambiguous attempt must terminalize.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -c "UPDATE experiment_scheduler_worker_attempt
        SET reserved_at=clock_timestamp()-interval '2 minutes'
        WHERE worker_attempt_id=${attempt_id}
          AND lifecycle_state='identity_ambiguous'
          AND diagnostic='launch_reservation_within_recovery_grace'"

run_scheduler "after_grace"

state="$(scalar \
    "SELECT lifecycle_state || '|' ||
            reconciliation_result || '|' || diagnostic
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${attempt_id}")"

assert_equal \
    "launch_failed|never_spawned|interrupted_launch_never_spawned" \
    "${state}" \
    "Expired ambiguous reservation reconciles as never spawned"

binding="$(scalar \
    "SELECT status || '|' ||
            COALESCE(active_scheduler_worker_attempt_id::text, 'NULL')
     FROM experiment WHERE experiment_id=920056")"

assert_equal \
    "failed|NULL" \
    "${binding}" \
    "Experiment fails and clears active binding"

assert_equal \
    "1" \
    "$(scalar \
        "SELECT count(*) FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=920056 AND worker_pid IS NULL")" \
    "No worker was spawned"

assert_equal \
    "released" \
    "$(scalar \
        "SELECT authority_state FROM experiment_scheduler_lease
         WHERE singleton=true")" \
    "Scheduler lease released"

echo
echo "PASS: Phase 24G interrupted-launch recovery integration"
