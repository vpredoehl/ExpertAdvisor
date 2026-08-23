#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"
process_binary="${2:-}"
test_db="ea_scheduler_forced_final_infer_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_forced_final_infer_test.XXXXXX)"
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
           [[ "${observed_command}" == *"--managed-test-worker"* ]] &&
           [[ "${observed_command}" == *"--scheduler-experiment-id=978004"* ]]; then
            kill -TERM -- "-${live_pgid}" >/dev/null 2>&1 || true
        fi
    fi
    if [[ -n "${live_pid}" ]]; then
        wait "${live_pid}" 2>/dev/null || true
    fi
    local pid=""
    while IFS= read -r pid; do
        [[ -z "${pid}" ]] && continue
        if kill -0 "${pid}" >/dev/null 2>&1; then
            local command=""
            command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
            if [[ "${command}" == *"${scheduler_binary}"* &&
                  "${command}" == *"--scheduler-experiment-id=978002"* ]]; then
                kill -TERM -- "-${pid}" >/dev/null 2>&1 || true
            fi
        fi
    done < <(psql -X -At -d "${test_db}" -c \
        "SELECT worker_pid FROM experiment_scheduler_worker_attempt
         WHERE experiment_id BETWEEN 978001 AND 978004
           AND worker_pid IS NOT NULL
           AND lifecycle_state IN
               ('reserved','spawned','running','observed','identity_ambiguous')" \
        2>/dev/null || true)
    if [[ "${EA_KEEP_FORCED_FINAL_INFER_TEST:-0}" = "1" ]]; then
        printf '%s\n' "preserved test database=${test_db} directory=${test_dir}" >&2
        return
    fi
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

case "${test_db}" in
    ea_scheduler_forced_final_infer_test_[0-9]*) ;;
    *) exit 90 ;;
esac

if [[ -z "${process_binary}" ]]; then
    process_binary="${test_dir}/GlobalExperimentControlProcessTests"
    read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"
    "${CXX:-clang++}" -std=c++20 -O0 -g \
        -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
        -I"${repo_root}/Headers" \
        "${pqxx_compile_flags[@]}" \
        "${repo_root}/Tests/GlobalExperimentControlProcessTests.cpp" \
        "${repo_root}/Sources/GlobalExperimentControl.cpp" \
        "${pqxx_link_flags[@]}" \
        -o "${process_binary}"
else
    process_binary="$(cd "$(dirname "${process_binary}")" && pwd)/$(basename "${process_binary}")"
fi

scalar() {
    psql -X -At -d "${test_db}" -c "$1"
}

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running')
ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='forced-final-inference-rerun-test',
    cutover_executable_path='/isolated/LSTM_Release',
    cutover_process_evidence='disposable-database',
    updated_at=clock_timestamp()
WHERE singleton=true;

INSERT INTO model(model_id,name) VALUES
    (978011,'ordinary-final-inference-model'),
    (978012,'forced-final-inference-model'),
    (978013,'analysis-control-model'),
    (978014,'active-forced-final-inference-model');

INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,last_model_id,duplicate_nonce
) VALUES
    (978001,'ordinaryrerunfixture',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'pending','infer',978011,978001),
    (978002,'forcedrerunfixture',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'completed','done',978012,978002),
    (978003,'analysiscontrolfixture',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'completed','done',978013,978003),
    (978004,'activeforcedrerunfixture',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'completed','done',978014,978004);

UPDATE model SET experiment_id=978001 WHERE model_id=978011;
UPDATE model SET experiment_id=978002 WHERE model_id=978012;
UPDATE model SET experiment_id=978003 WHERE model_id=978013;
UPDATE model SET experiment_id=978004 WHERE model_id=978014;

INSERT INTO inference_eval_result(
    id,model_id,symbol,prediction_horizon,threshold_logret,
    window_size,label_rule_id,target_type,from_date,to_date,
    completed_epochs,accuracy,accept_model,reject_reason,
    pred_down,pred_neutral,pred_up,status,completed_at,
    inference_scope,checkpoint_eval_id,parent_experiment_id,
    checkpoint_epoch
) VALUES
    (978021,978011,'ordinaryrerunfixture',4,0.0008,
     30,1,0,'2020-02-01','2020-03-01',20,0.5,false,'fixture',
     0.3,0.4,0.3,'completed',clock_timestamp()-interval '1 day',
     'final',NULL,NULL,NULL),
    (978022,978012,'forcedrerunfixture',4,0.0008,
     30,1,0,'2020-02-01','2020-03-01',20,0.5,false,'fixture',
     0.3,0.4,0.3,'completed',clock_timestamp()-interval '1 day',
     'final',NULL,NULL,NULL),
    (978023,978014,'activeforcedrerunfixture',4,0.0008,
     30,1,0,'2020-02-01','2020-03-01',20,0.5,false,'fixture',
     0.3,0.4,0.3,'completed',clock_timestamp()-interval '1 day',
     'final',NULL,NULL,NULL);

INSERT INTO experiment_analysis_result(
    experiment_id,model_id,symbol,prediction_horizon,
    analysis_scope,analysis_status,created_at,updated_at
) VALUES
    (978001,978011,'ordinaryrerunfixture',4,
     'final','completed',clock_timestamp(),clock_timestamp());
SQL

# Migration replay is non-destructive and preserves pre-existing inference
# rows, including completed FINAL semantic identities.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/078_operator_forced_final_inference_rerun.sql"

test "$(scalar "SELECT count(*) FROM inference_eval_result")" = "3"
test "$(scalar "SELECT count(*) FROM experiment WHERE operator_forced_final_inference_rerun_requested")" = "0"

# Ordinary train->infer progression keeps exact-result deduplication.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/ordinary-logs" \
    >"${test_dir}/ordinary.out" 2>&1
grep -q 'SCHEDULER_SKIP_EXISTING_INFERENCE,experiment_id=978001' \
    "${test_dir}/ordinary.out"
! grep -q 'SCHEDULER_CHILD_LAUNCHED,experiment_id=978001' \
    "${test_dir}/ordinary.out"
test "$(scalar "SELECT status||':'||phase FROM experiment WHERE experiment_id=978001")" = \
    "completed:done"

# The control command durably records operator intent; analysis requeue does not.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --requeue-inference=978002 --yes \
    >"${test_dir}/requeue-infer.out" 2>&1
grep -q 'SCHEDULER_CONTROL_APPLIED,action=requeue_inference,experiment_id=978002,new_status=pending,new_phase=infer,operator_forced_final_inference_rerun_requested=1' \
    "${test_dir}/requeue-infer.out"
test "$(scalar "SELECT status||':'||phase||':'||operator_forced_final_inference_rerun_requested::text FROM experiment WHERE experiment_id=978002")" = \
    "pending:infer:true"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --requeue-analysis=978003 --yes \
    >"${test_dir}/requeue-analysis.out" 2>&1
test "$(scalar "SELECT status||':'||phase||':'||operator_forced_final_inference_rerun_requested::text FROM experiment WHERE experiment_id=978003")" = \
    "pending:analyze:false"
# Keep the control-path assertion isolated from the later inference dispatch.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET status='completed', phase='done' WHERE experiment_id=978003"

# A scheduler restart before launch sees the durable request and plans inference.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/dry-run-logs" \
    >"${test_dir}/forced-dry-run.out" 2>&1
grep -q 'SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_DISPATCH,experiment_id=978002,model_id=978012' \
    "${test_dir}/forced-dry-run.out"
grep -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=978002,phase=infer,dry_run=1' \
    "${test_dir}/forced-dry-run.out"
! grep -q 'SCHEDULER_SKIP_EXISTING_INFERENCE,experiment_id=978002' \
    "${test_dir}/forced-dry-run.out"
test "$(scalar "SELECT status||':'||phase||':'||operator_forced_final_inference_rerun_requested::text FROM experiment WHERE experiment_id=978002")" = \
    "pending:infer:true"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --scheduler-status >"${test_dir}/status.out" 2>&1
grep -q 'SCHEDULER_STATUS_JOB,experiment_id=978002,phase=infer,status=pending,.*operator_forced_final_inference_rerun_requested=1' \
    "${test_dir}/status.out"

# A real scheduler reservation/exec occurs despite the pre-existing result.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/launch-logs" \
    >"${test_dir}/forced-launch.out" 2>&1
grep -q 'SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_DISPATCH,experiment_id=978002,model_id=978012' \
    "${test_dir}/forced-launch.out"
grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=978002,.*phase=infer' \
    "${test_dir}/forced-launch.out"
! grep -q 'SCHEDULER_SKIP_EXISTING_INFERENCE,experiment_id=978002' \
    "${test_dir}/forced-launch.out"
launched_attempt="$(scalar "SELECT worker_attempt_id FROM experiment_scheduler_worker_attempt WHERE experiment_id=978002 ORDER BY worker_attempt_id DESC LIMIT 1")"
test -n "${launched_attempt}"

# The disposable model intentionally lacks inference matrices. Once that exact
# child exits, ownership reconciliation fails deterministically and retains intent.
launched_pid="$(scalar "SELECT worker_pid FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${launched_attempt}")"
for _ in {1..300}; do
    ! kill -0 "${launched_pid}" >/dev/null 2>&1 && break
    sleep 0.02
done
! kill -0 "${launched_pid}" >/dev/null 2>&1
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/failed-recovery-logs" \
    >"${test_dir}/failed-recovery.out" 2>&1
test "$(scalar "SELECT status||':'||operator_forced_final_inference_rerun_requested::text||':'||(error_message LIKE 'forced_final_inference_rerun_missing_attempt_result;%')::text FROM experiment WHERE experiment_id=978002")" = \
    "failed:true:true"

# Requeueing later works, and exact attempt-relative completed_at evidence on
# the same semantic row authoritatively consumes the second request.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --requeue-inference=978002 --yes \
    >"${test_dir}/second-requeue.out" 2>&1
second_attempt="$(psql -X -qAt -d "${test_db}" <<SQL
INSERT INTO experiment_scheduler_worker_attempt(
    launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
    capacity_class,ownership_origin,lifecycle_state,worker_pid,
    worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
) VALUES(
    'forced-final-infer-success-${$}',978002,'experiment','infer','infer',
    'legacy_unverified','running',2147480002,2147480002,
    'absent-forced-final-infer-success','${scheduler_binary}',
    '${scheduler_binary} --infer --scheduler-experiment-id=978002 --scheduler-worker-attempt-id=999999',
    'experiment:978002:infer',clock_timestamp()-interval '1 second',
    clock_timestamp()-interval '1 second',clock_timestamp()-interval '1 second'
) RETURNING worker_attempt_id;
SQL
)"
PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v attempt_id="${second_attempt}" <<'SQL'
UPDATE experiment
SET status='running',phase='infer',current_operation='infer',
    worker_pid=2147480002,worker_process_group_id=2147480002,
    worker_process_start_identity='absent-forced-final-infer-success',
    active_scheduler_worker_attempt_id=:attempt_id,
    worker_started_at=clock_timestamp()-interval '1 second',
    updated_at=clock_timestamp()
WHERE experiment_id=978002 AND status='pending' AND phase='infer'
  AND operator_forced_final_inference_rerun_requested;
UPDATE inference_eval_result
SET completed_at=clock_timestamp()
WHERE id=978022;
INSERT INTO inference_profitability_observation(
    experiment_id,model_id,inference_eval_result_id,inference_scope,
    checkpoint_eval_id,inference_start,inference_end,prediction_count,
    actionable_count,winning_actionable_count,losing_actionable_count,
    gross_positive_terminal_horizon_log_return_sum,
    gross_negative_terminal_horizon_log_return_sum,
    aggregate_terminal_horizon_log_return_sum,
    average_terminal_horizon_log_return_per_actionable_prediction,
    metric_definition_canonical,metric_definition_hash,source_content_hash,
    observation_identity_canonical,observation_identity_hash
) VALUES(
    978002,978012,978022,'final',NULL,'2020-02-01','2020-03-01',
    3,2,1,1,0.02,-0.01,0.01,0.005,
    'forced-final-rerun-fixture-v1','fnv1a64:1111111111111111',
    'fnv1a64:2222222222222222','forced-final-rerun-fixture',
    'fnv1a64:3333333333333333');
SQL
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/success-recovery-logs" \
    >"${test_dir}/success-recovery.out" 2>&1
grep -q "SCHEDULER_FORCED_FINAL_INFERENCE_RERUN_CONSUMED,experiment_id=978002,worker_attempt_id=${second_attempt}" \
    "${test_dir}/success-recovery.out"
test "$(scalar "SELECT status||':'||phase||':'||operator_forced_final_inference_rerun_requested::text||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment WHERE experiment_id=978002")" = \
    "pending:analyze:false:true"
test "$(scalar "SELECT count(*)||':'||min(id)::text||':'||max(id)::text FROM inference_eval_result WHERE model_id=978012 AND symbol='forcedrerunfixture' AND prediction_horizon=4 AND threshold_logret=0.0008 AND from_date='2020-02-01' AND to_date='2020-03-01' AND status='completed' AND inference_scope='final'")" = \
    "1:978022:978022"
test "$(scalar "SELECT count(*)||':'||min(inference_eval_result_id)::text||':'||max(inference_eval_result_id)::text FROM inference_profitability_observation WHERE experiment_id=978002 AND inference_scope='final'")" = \
    "1:978022:978022"

# The database still rejects a duplicate completed FINAL semantic identity.
set +e
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO inference_eval_result(model_id,symbol,prediction_horizon,threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,status,inference_scope) VALUES(978012,'forcedrerunfixture',4,0.0008,30,1,0,'2020-02-01','2020-03-01','completed','final')" \
    >"${test_dir}/duplicate-final.out" 2>&1
duplicate_result=$?
set -e
test "${duplicate_result}" != "0"
grep -q 'inference_eval_result_final_completed_uidx' \
    "${test_dir}/duplicate-final.out"

# A scheduler restart while the exact forced worker attempt is live observes
# that attempt and does not reserve or launch a duplicate.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --requeue-inference=978004 --yes \
    >"${test_dir}/active-requeue.out" 2>&1
ready_file="${test_dir}/active-worker.ready"
"${process_binary}" --managed-test-worker --self-session \
    --infer --scheduler-experiment-id=978004 --ready-fd=9 \
    9>"${ready_file}" &
live_pid=$!
live_identity=""
for _ in {1..150}; do
    live_identity="$("${process_binary}" \
        "--inspect-managed-test-process=${live_pid}" 2>/dev/null || true)"
    [[ -s "${ready_file}" && -n "${live_identity}" ]] && break
    sleep 0.02
done
live_command=""
observed_live_pid=""
IFS='|' read -r observed_live_pid live_pgid live_start live_executable \
    live_command <<<"${live_identity}"
test "${observed_live_pid}" = "${live_pid}"
test "${live_pgid}" = "${live_pid}"
test -n "${live_start}"
test -n "${live_executable}"
[[ "${live_command}" == *"--scheduler-experiment-id=978004"* ]]

PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v pid="${live_pid}" -v pgid="${live_pgid}" \
    -v start_identity="${live_start}" \
    -v executable="${live_executable}" \
    -v command_line="${live_command}" <<'SQL'
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt(
        launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,worker_pid,
        worker_process_group_id,worker_process_start_identity,
        canonical_executable_path,command_line,command_identity,
        reserved_at,spawned_at,registered_at
    ) VALUES(
        'forced-final-infer-live-restart',978004,'experiment','infer','infer',
        'legacy_unverified','running',:pid,:pgid,:'start_identity',
        :'executable',:'command_line','experiment:978004:infer',
        clock_timestamp(),clock_timestamp(),clock_timestamp()
    ) RETURNING worker_attempt_id
)
UPDATE experiment e
SET status='running',phase='infer',current_operation='infer',
    worker_pid=:pid,worker_process_group_id=:pgid,
    worker_process_start_identity=:'start_identity',
    worker_executable=:'executable',worker_command_line=:'command_line',
    worker_started_at=clock_timestamp(),
    active_scheduler_worker_attempt_id=attempt.worker_attempt_id,
    updated_at=clock_timestamp()
FROM attempt
WHERE e.experiment_id=978004 AND e.status='pending'
  AND e.phase='infer'
  AND e.operator_forced_final_inference_rerun_requested;
SQL
for restart in 1 2; do
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once --recover-orphans-only \
        --scheduler-log-dir="${test_dir}/active-restart-${restart}-logs" \
        >"${test_dir}/active-restart-${restart}.out" 2>&1
    ! grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=978004' \
        "${test_dir}/active-restart-${restart}.out"
    test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=978004")" = "1"
    test "$(scalar "SELECT status||':'||phase||':'||operator_forced_final_inference_rerun_requested::text FROM experiment WHERE experiment_id=978004")" = \
        "running:infer:true"
done
kill -TERM -- "-${live_pgid}"
wait "${live_pid}"
live_pid=""

# Checkpoint dispatch remains a separate path and does not consult this flag.
checkpoint_body="$(sed -n '/std::vector<std::string> BuildCheckpointEvalInferCommand/,/^}/p' "${repo_root}/Sources/ExperimentScheduler.cpp")"
[[ "${checkpoint_body}" != *"operator_forced_final_inference_rerun_requested"* ]]

printf '%s\n' "SchedulerForcedFinalInferenceRerunIntegrationTests passed"
