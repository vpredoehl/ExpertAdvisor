#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
test_db="ea_scheduler_analyze_orphan_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_analyze_orphan.XXXXXX)"

cleanup() {
    local status=$?
    if [[ "${status}" -ne 0 ]]; then
        for output in "${test_dir}"/*.out; do
            [[ -f "${output}" ]] || continue
            printf '%s\n' "--- ${output}" >&2
            sed -n '1,240p' "${output}" >&2
        done
    fi
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
    return "${status}"
}
trap cleanup EXIT

scalar() { psql -X -At -q -d "${test_db}" -c "$1"; }

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
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment_scheduler_invocation(
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status,protocol_generation
) VALUES(
    'analyze-orphan-test-owner',2147480000,2147480000,
    'absent-analyze-owner','/missing/LSTM_Release',
    'LSTM_Release --schedule-experiments',
    'analyze-orphan-test-owner-nonce','lost',52
);
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='analyze-orphan-test',
    cutover_executable_path='/missing/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;

INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,current_operation,duplicate_nonce
) VALUES
    (940101,'analyzerecovered',4,0.0008,20,20,
     '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
     'running','analyze','analyze',940101),
    (940102,'analyzemissing',4,0.0008,20,20,
     '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
     'running','analyze','analyze',940102),
    (940103,'checkpointpending',4,0.0008,20,20,
     '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
     'completed','done','analyze',940103);

INSERT INTO model(experiment_id,name,comment) VALUES
    (940101,'analyze-recovered-model','analysis recovery fixture'),
    (940102,'analyze-missing-model','analysis missing fixture'),
    (940103,'checkpoint-pending-model','periodic training checkpoint');

UPDATE experiment e SET last_model_id=m.model_id
FROM model m WHERE m.experiment_id=e.experiment_id
  AND e.experiment_id BETWEEN 940101 AND 940103;

INSERT INTO experiment_checkpoint_eval(
    checkpoint_eval_id,experiment_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,symbol,prediction_horizon,status,phase
)
SELECT 949103,e.experiment_id,e.experiment_id,20,m.model_id,
       e.symbol,e.prediction_horizon,'pending','analyze'
FROM experiment e JOIN model m USING(experiment_id)
WHERE e.experiment_id=940103;

INSERT INTO inference_eval_result(
    model_id,symbol,prediction_horizon,threshold_logret,window_size,
    label_rule_id,target_type,from_date,to_date,completed_epochs,
    accuracy,accept_model,status,inference_scope,checkpoint_eval_id,
    parent_experiment_id,checkpoint_epoch
)
SELECT m.model_id,e.symbol,e.prediction_horizon,e.c_next_threshold,20,
       1,2,e.infer_start::text,e.infer_end::text,20,
       0.75,true,'completed','checkpoint',949103,940103,20
FROM experiment e JOIN model m USING(experiment_id)
WHERE e.experiment_id=940103;
SQL

insert_attempt() {
    local experiment_id="$1" attempt_identity="$2"
    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -At -q -d "${test_db}" -c \
        "INSERT INTO experiment_scheduler_worker_attempt(
             launch_attempt_identity,scheduler_invocation_id,
             scheduler_fencing_token,experiment_id,worker_kind,
             lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
             worker_pid,worker_process_group_id,worker_process_start_identity,
             canonical_executable_path,command_line,command_identity,
             reserved_at,spawned_at,registered_at
         ) VALUES(
             '${attempt_identity}','analyze-orphan-test-owner',52,
             ${experiment_id},'experiment','analyze','analyze',
             'scheduler_launch','running',2147480000,2147480000,
             'absent-analyze-process','/missing/LSTM_Release',
             '/missing/LSTM_Release --analyze-experiment=${experiment_id}',
             'experiment:${experiment_id}:analyze',
             clock_timestamp()-interval '2 minutes',
             clock_timestamp()-interval '2 minutes',
             clock_timestamp()-interval '2 minutes'
         ) RETURNING worker_attempt_id"
}

recovered_attempt="$(insert_attempt 940101 analyze-recovered-attempt)"
missing_attempt="$(insert_attempt 940102 analyze-missing-attempt)"

PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment e SET active_scheduler_worker_attempt_id=a.worker_attempt_id,
         worker_pid=a.worker_pid,worker_process_group_id=a.worker_process_group_id
     FROM experiment_scheduler_worker_attempt a
     WHERE a.experiment_id=e.experiment_id
       AND e.experiment_id IN (940101,940102)"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
BEGIN;
INSERT INTO experiment_analysis_result(
    experiment_id,model_id,symbol,prediction_horizon,
    analysis_scope,analysis_status,created_at,updated_at)
SELECT e.experiment_id,e.last_model_id,e.symbol,e.prediction_horizon,
       'final','completed',clock_timestamp(),clock_timestamp()
FROM experiment e WHERE e.experiment_id=940101;
UPDATE experiment
SET status='completed',phase='done',completed_at=clock_timestamp(),
    updated_at=clock_timestamp()
WHERE experiment_id=940101 AND status='running' AND phase='analyze';
COMMIT;
SQL

test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE capacity_class='analyze' AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = "2"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/recovery-logs" \
    >"${test_dir}/recovery.out" 2>&1

grep -q 'SCHEDULER_ORPHAN_RECOVERY_DONE,recovered_or_failed=2' \
    "${test_dir}/recovery.out"
test "$(scalar "SELECT lifecycle_state||':'||(completed_at IS NOT NULL)::text||':'||reconciliation_result||':'||diagnostic FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${recovered_attempt}")" = \
    "completed:true:process_missing_result_recovered:exact_process_identity_absent"
test "$(scalar "SELECT status||':'||phase||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment WHERE experiment_id=940101")" = \
    "completed:done:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${missing_attempt}")" = \
    "failed:process_missing_no_result"
test "$(scalar "SELECT status||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment WHERE experiment_id=940102")" = \
    "failed:true"
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE capacity_class='analyze' AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = "0"
test "$(scalar "SELECT status||':'||phase||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=949103")" = \
    "pending:analyze:true"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/selection-logs" \
    >"${test_dir}/selection.out" 2>&1

grep -q 'CHECKPOINT_ANALYSIS_CLAIMED,checkpoint_eval_id=949103' \
    "${test_dir}/selection.out"
test "$(scalar "SELECT status||':'||phase||':'||(analysis_id IS NOT NULL)::text||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=949103")" = \
    "completed:done:true:true"
test "$(scalar "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE checkpoint_eval_id=949103 AND worker_kind='checkpoint_analyze'")" = \
    "completed"

printf '%s\n' "SchedulerAnalyzeOrphanRecoveryTests passed"
