#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
process_binary="${2:-${repo_root}/DerivedData/SchedulerOwnership/Tests/GlobalExperimentControlProcessTests}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"
process_binary="$(cd "$(dirname "${process_binary}")" && pwd)/$(basename "${process_binary}")"
test_db="ea_worker_attempt_reconcile_test_${$}"
test_dir="$(mktemp -d /tmp/ea_worker_attempt_reconcile_test.XXXXXX)"
live_pid=""
live_pgid=""
live_start=""
live_executable=""

cleanup() {
    if [[ -n "${live_pid}" ]] && kill -0 "${live_pid}" >/dev/null 2>&1; then
        local identity=""
        local observed_pid="" observed_pgid="" observed_start=""
        local observed_executable="" observed_command=""
        identity="$("${process_binary}" \
            "--inspect-managed-test-process=${live_pid}" 2>/dev/null || true)"
        IFS='|' read -r observed_pid observed_pgid observed_start \
            observed_executable observed_command <<<"${identity}"
        if [[ "${observed_pid}" = "${live_pid}" ]] &&
           [[ "${observed_pgid}" = "${live_pgid}" ]] &&
           [[ "${observed_start}" = "${live_start}" ]] &&
           [[ "${observed_executable}" = "${live_executable}" ]] &&
           [[ "${observed_command}" == *"--managed-test-worker" ]]; then
            kill -TERM -- "-${live_pgid}" >/dev/null 2>&1 || true
        fi
    fi
    if [[ -n "${live_pid}" ]]; then
        wait "${live_pid}" 2>/dev/null || true
    fi
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING;
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='worker-attempt-reconciliation-test',
    cutover_executable_path='/test/LSTM_Release',
    cutover_process_evidence='isolated-test-database',
    updated_at=clock_timestamp()
WHERE singleton=true;
SQL

insert_fixture() {
    local experiment_id="$1"
    local attempt_id="$2"
    local pid="$3"
    local pgid="$4"
    local start_identity="$5"
    local executable="$6"
    local command_line="$7"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" \
        -v attempt_id="${attempt_id}" \
        -v pid="${pid}" \
        -v pgid="${pgid}" \
        -v start_identity="${start_identity}" \
        -v executable="${executable}" \
        -v command_line="${command_line}" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,target_epochs,
    train_start,train_end,status,phase,current_operation,worker_pid,
    worker_process_group_id,worker_process_start_identity,worker_executable,
    worker_command_line,worker_started_at,duplicate_nonce
) VALUES(
    :experiment_id,'reconcile-fixture',4,0.0008,20,
    '2020-01-01','2021-01-01','running','train','train',
    :pid,:pgid,:'start_identity',:'executable',:'command_line',
    clock_timestamp(),:experiment_id
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
) VALUES(
    :attempt_id,'test:reconcile:'||:attempt_id,:experiment_id,'experiment',
    'train','train','scheduler_launch','identity_ambiguous',
    :pid,:pgid,:'start_identity',:'executable',:'command_line',
    'experiment:'||:experiment_id||':train',clock_timestamp(),
    clock_timestamp(),clock_timestamp()
);
UPDATE experiment
SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=:experiment_id;
SQL
}

fresh_snapshot() {
    psql -X -At -d "${test_db}" -c \
        "SELECT a.lifecycle_state||':'||COALESCE(a.reconciliation_result,'NULL')||':'||
                e.status||':'||e.phase||':'||
                (e.worker_pid IS NULL)::text||':'||
                (e.worker_process_group_id IS NULL)::text||':'||
                (e.active_scheduler_worker_attempt_id IS NULL)::text||':'||
                (a.completed_at IS NOT NULL)::text||':'||
                (e.completed_at IS NOT NULL)::text
         FROM experiment_scheduler_worker_attempt a
         JOIN experiment e ON e.experiment_id=a.experiment_id
         WHERE a.worker_attempt_id=$1"
}

assert_absent_terminal_state() {
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT a.lifecycle_state||':'||a.exit_code||':'||
                a.reconciliation_result||':'||a.diagnostic||':'||
                (a.completed_at IS NOT NULL)::text||':'||e.status||':'||
                e.phase||':'||(e.worker_pid IS NULL)::text||':'||
                (e.worker_process_group_id IS NULL)::text||':'||
                (e.active_scheduler_worker_attempt_id IS NULL)::text||':'||
                e.exit_code||':'||e.error_message||':'||
                (e.completed_at IS NOT NULL)::text
         FROM experiment_scheduler_worker_attempt a
         JOIN experiment e ON e.experiment_id=a.experiment_id
         WHERE a.worker_attempt_id=$1")" = \
        "failed:-1:process_missing_no_result:exact_process_identity_absent_no_result:true:failed:train:true:true:true:-1:worker_process_missing_no_result:true"
}

absent_pid=2147480000
absent_executable="/missing/LSTM_Release"
absent_command="/missing/LSTM_Release --managed-test-worker --self-session --scheduler-experiment-id=930101 --ready-fd=9 --train --scheduler-worker-attempt-id=991101"
insert_fixture 930101 991101 "${absent_pid}" "${absent_pid}" \
    absent-process-start-991101 "${absent_executable}" "${absent_command}"

# Every psql invocation below opens a new connection. Dry-run must not mutate.
test "$(fresh_snapshot 991101)" = "identity_ambiguous:NULL:running:train:false:false:false:false:false"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991101 --dry-run \
    >"${test_dir}/absent-dry-run.out" 2>&1
grep -q 'outcome=eligible,worker_attempt_id=991101' "${test_dir}/absent-dry-run.out"
grep -q 'process_presence=absent' "${test_dir}/absent-dry-run.out"
test "$(fresh_snapshot 991101)" = "identity_ambiguous:NULL:running:train:false:false:false:false:false"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991101 --yes \
    >"${test_dir}/absent-apply.out" 2>&1
test "$(awk '/outcome=applying/{print NR; exit}' "${test_dir}/absent-apply.out")" -lt \
    "$(awk '/outcome=applied/{print NR; exit}' "${test_dir}/absent-apply.out")"
grep -q 'outcome=applied,worker_attempt_id=991101' "${test_dir}/absent-apply.out"
assert_absent_terminal_state 991101

# A terminal attempt is not replayable through this administrative command.
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991101 --dry-run \
    >"${test_dir}/terminal-replay.out" 2>&1
terminal_replay_result=$?
set -e
test "${terminal_replay_result}" = "1"
grep -q 'outcome=rejected' "${test_dir}/terminal-replay.out"
! grep -q 'outcome=applied' "${test_dir}/terminal-replay.out"
assert_absent_terminal_state 991101

# A stale worker identity rejects before either mutation; this is the exact
# lifecycle-binding fence used to prevent a replaced binding from applying.
insert_fixture 930102 991102 "${absent_pid}" "${absent_pid}" \
    absent-process-start-991102 "${absent_executable}" "${absent_command}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET worker_command_line=worker_command_line || ' --stale' WHERE experiment_id=930102"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991102 --yes \
    >"${test_dir}/stale-binding.out" 2>&1
stale_result=$?
set -e
test "${stale_result}" = "1"
grep -q 'outcome=rejected' "${test_dir}/stale-binding.out"
! grep -q 'outcome=applied' "${test_dir}/stale-binding.out"
test "$(fresh_snapshot 991102)" = "identity_ambiguous:NULL:running:train:false:false:false:false:false"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET worker_command_line=replace(worker_command_line, ' --stale', '') WHERE experiment_id=930102"

# A present, identity-matching process must not enter the absent branch.
live_link="${test_dir}/LSTM_Release_train"
live_ready="${test_dir}/live.ready"
ln -sf "${process_binary}" "${live_link}"
"${live_link}" --managed-test-worker --self-session \
    --scheduler-experiment-id=930103 --ready-fd=9 --train \
    --scheduler-worker-attempt-id=991103 9>"${live_ready}" &
live_pid=$!
observed_live_pid=""
live_command=""
for _ in {1..100}; do
    live_identity="$("${process_binary}" \
        "--inspect-managed-test-process=${live_pid}" 2>/dev/null || true)"
    IFS='|' read -r observed_live_pid live_pgid live_start \
        live_executable live_command <<<"${live_identity}"
    [[ -s "${live_ready}" && "${observed_live_pid}" = "${live_pid}" ]] && break
    sleep 0.02
done
test "${observed_live_pid}" = "${live_pid}"
test "${live_pgid}" = "${live_pid}"
test -n "${live_start}"
test -n "${live_executable}"
live_command="${live_link} --managed-test-worker --self-session --scheduler-experiment-id=930103 --ready-fd=9 --train --scheduler-worker-attempt-id=991103"
insert_fixture 930103 991103 "${live_pid}" "${live_pgid}" \
    "${live_start}" "${live_executable}" "${live_command}"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991103 --yes \
    >"${test_dir}/live-apply.out" 2>&1
grep -q 'outcome=applied,worker_attempt_id=991103' "${test_dir}/live-apply.out"
grep -q 'reason=exact_identity_verified' "${test_dir}/live-apply.out"
! grep -q 'reason=exact_process_absent_no_result' "${test_dir}/live-apply.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result||':'||e.status||':'||
            (e.worker_pid IS NOT NULL)::text
     FROM experiment_scheduler_worker_attempt a JOIN experiment e
       ON e.experiment_id=a.experiment_id WHERE a.worker_attempt_id=991103")" = \
    "observed:valid_process_observed:running:true"
kill -TERM -- "${live_pgid}"
wait "${live_pid}"
live_pid=""

# Force the database commit to fail. The command may report applying before
# the commit attempt, but it must never report applied and both rows must roll
# back as observed through a fresh connection.
commit_command="${absent_command/930101/930104}"
commit_command="${commit_command/991101/991104}"
insert_fixture 930104 991104 "${absent_pid}" "${absent_pid}" \
    absent-process-start-991104 "${absent_executable}" "${commit_command}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
CREATE OR REPLACE FUNCTION reconciliation_test_commit_failure()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.experiment_id = 930104 AND NEW.status = 'failed' THEN
        RAISE EXCEPTION 'forced reconciliation commit failure';
    END IF;
    RETURN NEW;
END;
$$;
CREATE CONSTRAINT TRIGGER reconciliation_test_commit_failure_trigger
AFTER UPDATE ON experiment
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION reconciliation_test_commit_failure();
SQL
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt=991104 --yes \
    >"${test_dir}/commit-failure.out" 2>&1
commit_failure_result=$?
set -e
test "${commit_failure_result}" = "2"
grep -q 'WORKER_ATTEMPT_RECONCILIATION_ERROR' "${test_dir}/commit-failure.out"
! grep -q 'outcome=applied' "${test_dir}/commit-failure.out"
test "$(fresh_snapshot 991104)" = "identity_ambiguous:NULL:running:train:false:false:false:false:false"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
DROP TRIGGER reconciliation_test_commit_failure_trigger ON experiment;
DROP FUNCTION reconciliation_test_commit_failure();
SQL

printf '%s\n' "WorkerAttemptReconciliationIntegrationTests passed"
