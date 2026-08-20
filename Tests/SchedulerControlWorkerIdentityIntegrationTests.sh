#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
test_db="ea_scheduler_control_identity_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_control_identity_test.XXXXXX)"

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

case "${test_db}" in
    ea_scheduler_control_identity_test_[0-9]*) ;;
    *) exit 90 ;;
esac

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton, desired_state)
VALUES (true, 'running')
ON CONFLICT (singleton) DO UPDATE SET desired_state='running';

UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='scheduler-control-identity-test',
    cutover_executable_path='/isolated/LSTM_Release',
    cutover_process_evidence='isolated-disposable-database',
    updated_at=clock_timestamp()
WHERE singleton;

INSERT INTO experiment_admin_request(
    action, invocation_identity, status, previous_global_state,
    resulting_global_state, completed_at
) VALUES (
    'pause_all', 'scheduler-control-identity-test', 'completed',
    'running', 'paused', clock_timestamp()
);

INSERT INTO model(model_id, name)
VALUES (941010, 'scheduler-control-identity-analysis-model');

INSERT INTO experiment(
    experiment_id, symbol, prediction_horizon, c_next_threshold,
    target_epochs, checkpoint_interval, train_start, train_end,
    infer_start, infer_end, status, phase, last_model_id,
    worker_pid, worker_process_group_id, worker_process_start_identity,
    worker_executable, worker_command_line, worker_started_at,
    current_operation, worker_control_state, worker_global_pause_request_id,
    started_at, completed_at, exit_code, error_message, updated_at,
    duplicate_nonce
) VALUES
(
    941001, 'eurusd', 4, 0.0008, 1, 1,
    '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03',
    'failed', 'train', NULL,
    941001, 941001, 'old-start-941001', '/old/LSTM_Release',
    '/old/LSTM_Release --train --scheduler-experiment-id=941001 --scheduler-worker-attempt-id=941101',
    clock_timestamp(), 'train', 'paused',
    (SELECT request_id FROM experiment_admin_request
     WHERE invocation_identity='scheduler-control-identity-test'),
    clock_timestamp(), clock_timestamp(), 17, 'old failure', clock_timestamp(), 941001
),
(
    941002, 'eurusd', 4, 0.0008, 1, 1,
    '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03',
    'completed', 'done', 941010,
    941002, 941002, 'old-start-941002', '/old/LSTM_Release',
    '/old/LSTM_Release --infer --scheduler-experiment-id=941002 --scheduler-worker-attempt-id=941102',
    clock_timestamp(), 'infer', 'paused',
    (SELECT request_id FROM experiment_admin_request
     WHERE invocation_identity='scheduler-control-identity-test'),
    clock_timestamp(), clock_timestamp(), 18, 'old inference failure', clock_timestamp(), 941002
),
(
    941003, 'eurusd', 4, 0.0008, 1, 1,
    '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03',
    'completed', 'done', 941010,
    941003, 941003, 'old-start-941003', '/old/LSTM_Release',
    '/old/LSTM_Release --analyze-experiment=941003 --scheduler-worker-attempt-id=941103',
    clock_timestamp(), 'analyze', 'paused',
    (SELECT request_id FROM experiment_admin_request
     WHERE invocation_identity='scheduler-control-identity-test'),
    clock_timestamp(), clock_timestamp(), 19, 'old analysis failure', clock_timestamp(), 941003
),
(
    941004, 'eurusd', 4, 0.0008, 1, 1,
    '2020-01-01', '2020-01-02', '2020-01-02', '2020-01-03',
    'cancelled', 'done', NULL,
    941004, 941004, 'unrelated-start-941004', '/unrelated/LSTM_Release',
    '/unrelated/LSTM_Release --train --scheduler-experiment-id=941004',
    clock_timestamp(), 'train', 'running', NULL,
    clock_timestamp(), clock_timestamp(), 99, 'unrelated', clock_timestamp(), 941004
);

INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id, launch_attempt_identity, experiment_id,
    worker_kind, lifecycle_phase, capacity_class, ownership_origin,
    lifecycle_state, worker_pid, worker_process_group_id,
    worker_process_start_identity, canonical_executable_path, command_line,
    command_identity, reserved_at, completed_at, exit_code, diagnostic
) VALUES
(
    941101, 'historical:941001:941101', 941001, 'experiment', 'train',
    'train', 'legacy_unverified', 'failed', 941001, 941001,
    'old-start-941001', '/old/LSTM_Release',
    '/old/LSTM_Release --train --scheduler-experiment-id=941001 --scheduler-worker-attempt-id=941101',
    'experiment:941001:train', clock_timestamp(), clock_timestamp(), 17,
    'immutable historical attempt fixture'
),
(
    941102, 'historical:941002:941102', 941002, 'experiment', 'infer',
    'infer', 'legacy_unverified', 'completed', 941002, 941002,
    'old-start-941002', '/old/LSTM_Release',
    '/old/LSTM_Release --infer --scheduler-experiment-id=941002 --scheduler-worker-attempt-id=941102',
    'experiment:941002:infer', clock_timestamp(), clock_timestamp(), 0,
    'immutable historical attempt fixture'
),
(
    941103, 'historical:941003:941103', 941003, 'experiment', 'analyze',
    'analyze', 'legacy_unverified', 'failed', 941003, 941003,
    'old-start-941003', '/old/LSTM_Release',
    '/old/LSTM_Release --analyze-experiment=941003 --scheduler-worker-attempt-id=941103',
    'experiment:941003:analyze', clock_timestamp(), clock_timestamp(), 17,
    'immutable historical attempt fixture'
);
SQL

snapshot() {
    psql -X -At -d "${test_db}" -c "SELECT experiment_id||':'||status||':'||phase||':'||COALESCE(worker_pid::text,'NULL')||':'||COALESCE(worker_process_group_id::text,'NULL')||':'||COALESCE(worker_process_start_identity,'NULL')||':'||COALESCE(worker_executable,'NULL')||':'||COALESCE(worker_command_line,'NULL')||':'||COALESCE(active_scheduler_worker_attempt_id::text,'NULL')||':'||COALESCE(worker_control_state,'NULL')||':'||COALESCE(worker_global_pause_request_id::text,'NULL')||':'||COALESCE(current_operation,'NULL') FROM experiment WHERE experiment_id BETWEEN 941001 AND 941004 ORDER BY experiment_id"
}

historical_snapshot() {
    psql -X -At -d "${test_db}" -c "SELECT worker_attempt_id||':'||lifecycle_state||':'||worker_pid||':'||worker_process_group_id||':'||worker_process_start_identity||':'||canonical_executable_path||':'||command_line||':'||exit_code||':'||(completed_at IS NOT NULL)::text FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id BETWEEN 941101 AND 941103 ORDER BY worker_attempt_id"
}

before_dry_run="$(snapshot)"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --retry-failed-experiment=941001 --dry-run >"${test_dir}/retry-dry-run.out" 2>&1
grep -q 'SCHEDULER_CONTROL_DRY_RUN,action=retry_failed,experiment_id=941001' "${test_dir}/retry-dry-run.out"
test "${before_dry_run}" = "$(snapshot)"

historical_before="$(historical_snapshot)"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --retry-failed-experiment=941001 --yes >"${test_dir}/retry.out" 2>&1
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --requeue-inference=941002 --yes >"${test_dir}/requeue-inference.out" 2>&1
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --requeue-analysis=941003 --yes >"${test_dir}/requeue-analysis.out" 2>&1

test "$(psql -X -At -d "${test_db}" -c "SELECT count(*) FROM experiment WHERE experiment_id IN (941001,941002,941003) AND status='pending' AND worker_pid IS NULL AND worker_process_group_id IS NULL AND worker_process_start_identity IS NULL AND worker_executable IS NULL AND worker_command_line IS NULL AND active_scheduler_worker_attempt_id IS NULL AND worker_control_state='running' AND worker_global_pause_request_id IS NULL AND started_at IS NULL AND completed_at IS NULL AND exit_code IS NULL AND error_message IS NULL AND current_operation IS NULL")" = "3"
test "$(psql -X -At -d "${test_db}" -c "SELECT phase FROM experiment WHERE experiment_id=941001")" = "train"
test "$(psql -X -At -d "${test_db}" -c "SELECT phase FROM experiment WHERE experiment_id=941002")" = "infer"
test "$(psql -X -At -d "${test_db}" -c "SELECT phase FROM experiment WHERE experiment_id=941003")" = "analyze"
test "${historical_before}" = "$(historical_snapshot)"
test "$(snapshot | sed -n '4p')" = "941004:cancelled:done:941004:941004:unrelated-start-941004:/unrelated/LSTM_Release:/unrelated/LSTM_Release --train --scheduler-experiment-id=941004:NULL:running:NULL:train"

set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --schedule-experiments --scheduler-once --max-analyze-procs=1 --scheduler-log-dir="${test_dir}/logs" >"${test_dir}/replacement-dispatch.out" 2>&1
dispatch_result=$?
set -e
grep -q 'SCHEDULER_QUEUE_PHASE,phase=analyze,.*launched=1' "${test_dir}/replacement-dispatch.out"

replacement="$(psql -X -At -F $'\t' -d "${test_db}" -c "SELECT a.worker_attempt_id,a.lifecycle_state,a.worker_pid,a.worker_process_group_id,a.worker_process_start_identity,a.canonical_executable_path,a.command_line,COALESCE(e.worker_pid::text,'NULL'),COALESCE(e.worker_process_group_id::text,'NULL'),COALESCE(e.worker_process_start_identity,'NULL'),COALESCE(e.worker_executable,'NULL'),COALESCE(e.worker_command_line,'NULL'),COALESCE(e.active_scheduler_worker_attempt_id::text,'NULL') FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id WHERE a.experiment_id=941003 AND a.worker_attempt_id<>941103 ORDER BY a.worker_attempt_id DESC LIMIT 1")"
IFS=$'\t' read -r replacement_id replacement_state replacement_pid replacement_pgid replacement_start replacement_executable replacement_command lifecycle_pid lifecycle_pgid lifecycle_start lifecycle_executable lifecycle_command lifecycle_attempt <<<"${replacement}"
test -n "${replacement_id}"
test "${replacement_id}" != "941103"
test -n "${replacement_pid}"
test -n "${replacement_pgid}"
test "${replacement_pid}" != "941003"
test "${replacement_pgid}" != "941003"
test "${replacement_start}" != "old-start-941003"
test "${replacement_executable}" != "/old/LSTM_Release"
[[ "${replacement_command}" == *"--scheduler-worker-attempt-id=${replacement_id}"* ]]
test "${lifecycle_start}" != "old-start-941003"
test "${lifecycle_executable}" != "/old/LSTM_Release"
[[ "${lifecycle_command}" == *"--scheduler-worker-attempt-id=${replacement_id}"* ]]
if [[ "${lifecycle_attempt}" != "NULL" ]]; then
    test "${lifecycle_attempt}" = "${replacement_id}"
    test -n "${lifecycle_pid}"
    test -n "${lifecycle_pgid}"
fi
test "${dispatch_result}" -eq 0 -o "${dispatch_result}" -eq 1

printf '%s\n' "SchedulerControlWorkerIdentityIntegrationTests passed"
