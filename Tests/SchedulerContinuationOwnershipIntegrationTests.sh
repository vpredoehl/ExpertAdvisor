#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/SchedulerOwnershipCorrection/Build/Products/Release/LSTM_Release}"
test_db="ea_scheduler_continuation_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_continuation_test.XXXXXX)"
source_experiment_id=930070
source_model_id=935070
foreign_invocation_id="continuation-test-foreign-owner"

case "${test_db}" in
    ea_scheduler_continuation_test_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/078_operator_forced_final_inference_rerun.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/092_economic_calendar_snapshot.sql"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING;
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='continuation_ownership_test',
    cutover_executable_path='/tmp/LSTM_Release',
    cutover_process_evidence=
        'disposable_database;no_external_scheduler_can_connect',
    updated_at=clock_timestamp()
WHERE singleton=true AND required_generation=52
  AND cutover_state='pending';
INSERT INTO experiment_scheduler_invocation(
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status,protocol_generation
) VALUES(
    'continuation-test-foreign-owner',999970,999970,'foreign-start',
    '/missing/LSTM_Release','LSTM_Release --schedule-experiments',
    'continuation-test-foreign-owner-nonce','lost',52
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    last_model_id,current_operation,duplicate_nonce,
    continuation_policy_enabled,continuation_policy_target_epochs,
    continuation_policy_min_evals,continuation_policy_patience,
    continuation_policy_min_leader_score,continuation_policy_scope,
    continuation_policy_trend_mode,continuation_policy_source_mode,
    continuation_policy_include_excluded,
    continuation_candidate_excluded,continuation_policy_revision,
    model_input_width,model_input_semantic_layout_version
) VALUES(
    930070,'continuationfixture',4,0.0008,1.0,1.0,20,20,
    '2020-01-01','2021-01-01','2021-02-01','2021-03-01',
    'completed','done',935070,'analyze',930070,
    true,40,1,2,0.1,'symbol_horizon','none','final_model',
    false,false,1,77,6
);
INSERT INTO model(model_id,name,experiment_id)
VALUES(935070,'continuation-source-model',930070);
WITH valueset(values) AS (
    VALUES(ARRAY[
        1.0,4.0,0.0008,0.0,0.0,0.0,0.0,
        0.0,0.0,0.0,20.0,1.0,1.0,0.0
    ]::double precision[])
)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'train_config_meta',1,14,0,index-1,values[index]
FROM valueset,generate_series(1,14) AS index;
WITH encoded(value) AS (
    VALUES('continuationfixture'::text)
)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'train_symbol_meta',1,length(value),0,index-1,
       ascii(substr(value,index,1))
FROM encoded,generate_series(1,length(value)) AS index;
WITH encoded(value) AS (
    VALUES('2020-01-01|2021-01-01'::text)
)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'train_range_meta',1,length(value),0,index-1,
       ascii(substr(value,index,1))
FROM encoded,generate_series(1,length(value)) AS index;

WITH v AS (SELECT ARRAY[1.0,77.0,1.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'model_meta',1,3,0,i-1,a[i]
FROM v,generate_series(1,3)i;

WITH v AS (SELECT ARRAY[1.0,6.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'model_input_semantics_meta',1,2,0,i-1,a[i]
FROM v,generate_series(1,2)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'param',78,4,(i-1)/4,(i-1)%4,0.0
FROM generate_series(1,312)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'bias',1,4,0,i-1,0.0
FROM generate_series(1,4)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
VALUES
    (935070,'returnHeadWeight',1,1,0,0,0.0),
    (935070,'returnHeadBias',1,1,0,0,0.0);

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'returnHeadDirWeight',1,3,0,i-1,0.0
FROM generate_series(1,3)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'returnHeadDirBias',1,3,0,i-1,0.0
FROM generate_series(1,3)i;

WITH v AS (
    SELECT ARRAY[2.0,1.0,0.0,0.0,0.0,1.0]::double precision[] a
)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'target_meta',1,6,0,i-1,a[i]
FROM v,generate_series(1,6)i;

WITH v AS (
    SELECT ARRAY[1.0,1.0,20.0,0.0,0.0]::double precision[] a
)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 935070,'optimizer_meta',1,5,0,i-1,a[i]
FROM v,generate_series(1,5)i;
INSERT INTO experiment_analysis_result(
    experiment_id,model_id,symbol,prediction_horizon,
    target_epochs,completed_epochs,infer_accuracy,leader_score,
    analysis_status,analysis_scope
) VALUES(
    930070,935070,'continuationfixture',4,20,20,0.9,0.8,
    'completed','final'
);
SQL

run_scheduler() {
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-train-procs=1 --max-infer-procs=1 \
        --max-analyze-procs=1 \
        --scheduler-log-dir="${test_dir}/logs" "$@"
}

# Positive baseline: automatic evaluation queues exactly one child, replay
# creates no duplicate, and the child enters the ordinary durable attempt path.
set +e
run_scheduler --auto-queue-continuations \
    >"${test_dir}/baseline-queue.out" 2>&1
baseline_queue_result=$?
set -e
if [[ "${baseline_queue_result}" -ne 0 ]]; then
    sed -n '1,240p' "${test_dir}/baseline-queue.out" >&2
    exit "${baseline_queue_result}"
fi
grep -q 'CONTINUATION_AUTO_QUEUED' \
    "${test_dir}/baseline-queue.out"
child_experiment_id="$(psql -X -At -d "${test_db}" -c \
    "SELECT continuation_policy_queued_experiment_id
     FROM experiment WHERE experiment_id=${source_experiment_id}")"
test -n "${child_experiment_id}"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment
     WHERE parent_experiment_id=${source_experiment_id}")" = "1"
run_scheduler --auto-queue-continuations \
    >"${test_dir}/baseline-replay.out" 2>&1
grep -Eq 'CONTINUATION_AUTO_(ALREADY_SATISFIED|SKIPPED)' \
    "${test_dir}/baseline-replay.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment
     WHERE parent_experiment_id=${source_experiment_id}")" = "1"
run_scheduler >"${test_dir}/baseline-child-dispatch.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=${child_experiment_id}
       AND ownership_origin='scheduler_launch'")" = "1"

# Restore only the disposable continuation fixture. The terminal child attempt
# is removed before its lifecycle row, preserving FK and exact-binding order.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v child_id="${child_experiment_id}" <<'SQL'
BEGIN;
UPDATE experiment
SET continuation_policy_queued_experiment_id=NULL,
    continuation_policy_last_decision=NULL,
    continuation_policy_last_decision_at=NULL,
    continuation_policy_last_reason=NULL,
    continuation_policy_selected_model_id=NULL
WHERE experiment_id=930070;
UPDATE experiment_continuation_decision
SET queued_experiment_id=NULL
WHERE source_experiment_id=930070;
UPDATE experiment
SET parent_experiment_id=NULL,
    continuation_source_experiment_id=NULL,
    continuation_source_model_id=NULL,
    continuation_source_epoch=NULL,
    continuation_decision_id=NULL
WHERE experiment_id=:child_id;
DELETE FROM experiment_scheduler_worker_attempt
WHERE experiment_id=:child_id;
DELETE FROM experiment WHERE experiment_id=:child_id;
DELETE FROM experiment_continuation_decision
WHERE source_experiment_id=930070;
COMMIT;
SQL

for boundary in \
    continuation_before_evaluation_mutation \
    continuation_after_evaluation_before_queue \
    continuation_before_child_creation \
    continuation_before_final_commit
do
    decision_count_before="$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment_continuation_decision
         WHERE source_experiment_id=${source_experiment_id}")"
    set +e
    EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
    EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY="${boundary}" \
    EA_SCHEDULER_OWNERSHIP_TEST_FOREIGN_INVOCATION_ID="${foreign_invocation_id}" \
    run_scheduler --auto-queue-continuations \
        >"${test_dir}/${boundary}.out" 2>&1
    boundary_result=$?
    set -e
    test "${boundary_result}" = "4"
    grep -q 'SCHEDULER_OWNERSHIP_LOST' \
        "${test_dir}/${boundary}.out"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment
         WHERE parent_experiment_id=${source_experiment_id}")" = "0"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT continuation_policy_queued_experiment_id IS NULL
         FROM experiment WHERE experiment_id=${source_experiment_id}")" = "t"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment_continuation_decision
         WHERE source_experiment_id=${source_experiment_id}
           AND (decision='continuation_queued'
                OR queued_experiment_id IS NOT NULL)")" = "0"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment_scheduler_worker_attempt
         WHERE experiment_id<>${source_experiment_id}")" = "0"
    if [[ "${boundary}" = \
          continuation_before_evaluation_mutation ]]; then
        test "$(psql -X -At -d "${test_db}" -c \
            "SELECT count(*) FROM experiment_continuation_decision
             WHERE source_experiment_id=${source_experiment_id}")" = \
            "${decision_count_before}"
    fi
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
DELETE FROM experiment_continuation_decision
WHERE source_experiment_id=930070;
UPDATE experiment
SET continuation_policy_last_decision=NULL,
    continuation_policy_last_decision_at=NULL,
    continuation_policy_last_reason=NULL,
    continuation_policy_selected_model_id=NULL,
    continuation_policy_queued_experiment_id=NULL
WHERE experiment_id=930070;
SQL
done

printf '%s\n' \
    "SchedulerContinuationOwnershipIntegrationTests passed"
