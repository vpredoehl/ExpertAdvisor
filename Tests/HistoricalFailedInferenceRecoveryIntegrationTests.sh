#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/lstm-scheduler}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"

test_db="ea_historical_inference_recovery_test_${$}"
test_dir="$(mktemp -d /tmp/ea_historical_inference_recovery_test.XXXXXX)"

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

createdb "${test_db}"

pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"

for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql \
    086_scheduler_pause_resume_priority.sql
do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO experiment_global_control(singleton,desired_state)
     VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_protocol
     SET cutover_state='complete',
         cutover_completed_at=clock_timestamp(),
         cutover_completed_by='historical-failed-inference-recovery-test',
         cutover_executable_path='/test/LSTM_Release',
         cutover_process_evidence='isolated-test-database',
         updated_at=clock_timestamp()
     WHERE singleton=true"

insert_fixture() {
    local experiment_id="$1"
    local attempt_id="$2"
    local model_id="$3"
    local result_id="$4"
    local threshold="${5:-0.00079999998}"
    local error_message="${6:-child_exit_code_0;phase=infer;exit_code=0}"
    local reconciliation="${7:-parent_observed_exit}"

    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" \
        -v attempt_id="${attempt_id}" \
        -v model_id="${model_id}" \
        -v result_id="${result_id}" \
        -v threshold="${threshold}" \
        -v error_message="${error_message}" \
        -v reconciliation="${reconciliation}" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);

INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,target_epochs,
    train_start,train_end,infer_start,infer_end,status,phase,current_epoch,
    exit_code,error_message,duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    worker_started_at
) VALUES(
    :experiment_id,'fixturefx',4,0.0008,80,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'failed','infer',80,0,:'error_message',:experiment_id,77,7,
    7654321,'historical-process-start','/test/lstm-infer-worker',
    '/test/lstm-infer-worker --infer --scheduler-experiment-id='||:experiment_id||
        ' --scheduler-worker-attempt-id='||:attempt_id,
    clock_timestamp() - interval '2 hours'
);

INSERT INTO model(model_id,name,experiment_id)
VALUES(:model_id,'historical-recovery-model-'||:model_id,:experiment_id);

UPDATE experiment
SET last_model_id=:model_id
WHERE experiment_id=:experiment_id;

INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,checkpoint_eval_id,
    worker_kind,lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at,completed_at,exit_code,reconciliation_result,
    diagnostic,reconciled_at
) VALUES(
    :attempt_id,'test:historical-recovery:'||:attempt_id,:experiment_id,NULL,
    'experiment','infer','infer','scheduler_launch','failed',
    NULL,7654321,'historical-process-start','/test/lstm-infer-worker',
    '/test/lstm-infer-worker --infer --scheduler-experiment-id='||:experiment_id||
        ' --scheduler-worker-attempt-id='||:attempt_id,
    'experiment:'||:experiment_id||':infer',
    clock_timestamp() - interval '90 minutes',
    clock_timestamp() - interval '90 minutes',
    clock_timestamp() - interval '90 minutes',
    clock_timestamp() - interval '60 minutes',
    0,:'reconciliation','historical fixture',clock_timestamp() - interval '60 minutes'
);

INSERT INTO inference_eval_result(
    id,model_id,symbol,prediction_horizon,threshold_logret,window_size,
    label_rule_id,target_type,from_date,to_date,status,completed_at,
    inference_scope,checkpoint_eval_id
) VALUES(
    :result_id,:model_id,'fixturefx',4,:threshold,20,
    1,0,'2021-01-01','2022-01-01','completed',
    clock_timestamp() - interval '30 minutes','final',NULL
);
SQL
}

snapshot() {
    local experiment_id="$1"
    psql -X -At -d "${test_db}" -c \
        "SELECT e.status||':'||e.phase||':'||COALESCE(e.exit_code::text,'NULL')||':'||
                COALESCE(e.error_message,'NULL')||':'||
                (e.active_scheduler_worker_attempt_id IS NULL)::text||':'||
                e.worker_control_state||':'||e.operator_forced_final_inference_rerun_requested::text||':'||
                e.resume_requested::text||':'||e.scheduler_resume_origin||':'||
                a.lifecycle_state||':'||COALESCE(a.exit_code::text,'NULL')||':'||
                COALESCE(a.reconciliation_result,'NULL')
         FROM experiment e
         JOIN experiment_scheduler_worker_attempt a ON a.experiment_id=e.experiment_id
         WHERE e.experiment_id=${experiment_id}
         ORDER BY a.worker_attempt_id
         LIMIT 1"
}

counts() {
    local experiment_id="$1"
    psql -X -At -d "${test_db}" -c \
        "SELECT
             (SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=${experiment_id})||':'||
             (SELECT count(*) FROM inference_eval_result r JOIN model m ON m.model_id=r.model_id
                WHERE m.experiment_id=${experiment_id})"
}

run_recovery() {
    local experiment_id="$1"
    local mode="$2"
    local output="$3"
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        "--recover-failed-inference=${experiment_id}" "${mode}" >"${output}" 2>&1
}

# Happy path: threshold differs only by the known float representation delta.
insert_fixture 940101 992101 994101 996101
before="$(snapshot 940101)"
before_counts="$(counts 940101)"

run_recovery 940101 --dry-run "${test_dir}/happy-dry-run.out"
grep -q 'HISTORICAL_FAILED_INFERENCE_RECOVERY' "${test_dir}/happy-dry-run.out"
grep -q 'outcome=eligible' "${test_dir}/happy-dry-run.out"
grep -q 'worker_attempt_id=992101' "${test_dir}/happy-dry-run.out"
grep -q 'model_id=994101' "${test_dir}/happy-dry-run.out"
grep -q 'inference_result_id=996101' "${test_dir}/happy-dry-run.out"
test "$(snapshot 940101)" = "${before}"
test "$(counts 940101)" = "${before_counts}"

run_recovery 940101 --yes "${test_dir}/happy-apply.out"
grep -q 'outcome=applied' "${test_dir}/happy-apply.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||exit_code||':'||
            (error_message IS NULL)::text||':'||
            (active_scheduler_worker_attempt_id IS NULL)::text||':'||
            worker_control_state||':'||
            operator_forced_final_inference_rerun_requested::text||':'||
            resume_requested::text||':'||scheduler_resume_origin
     FROM experiment WHERE experiment_id=940101")" = \
    "pending:analyze:0:true:true:paused:false:false:none"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||exit_code||':'||reconciliation_result||':'||
            diagnostic||':'||(completed_at IS NOT NULL)::text
     FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=992101")" = \
    "completed:0:historical_completed_inference_result_recovered:historical_failed_final_inference_recovered_from_durable_result:true"
test "$(counts 940101)" = "${before_counts}"

# Recovery clears retained experiment worker identity while preserving the
# historical attempt/result rows.
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT (worker_pid IS NULL)::text||':'||
            (worker_process_group_id IS NULL)::text||':'||
            (worker_process_start_identity IS NULL)::text||':'||
            (worker_executable IS NULL)::text||':'||
            (worker_command_line IS NULL)::text||':'||
            (worker_started_at IS NULL)::text FROM experiment WHERE experiment_id=940101")" = \
    "true:true:true:true:true:true"

# Replay must reject.
set +e
run_recovery 940101 --dry-run "${test_dir}/replay.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/replay.out"
! grep -q 'outcome=applied' "${test_dir}/replay.out"

# Parser/command XOR: supplying both confirmation modes is invalid and immutable.
insert_fixture 940102 992102 994102 996102
xor_before="$(snapshot 940102)"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --recover-failed-inference=940102 --dry-run --yes \
    >"${test_dir}/xor.out" 2>&1
rc=$?
set -e
test "${rc}" = "1"
grep -q 'Argument error: --recover-failed-inference requires exactly one of --dry-run or --yes' \
    "${test_dir}/xor.out"
test "$(snapshot 940102)" = "${xor_before}"

# Missing durable result.
insert_fixture 940103 992103 994103 996103
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM inference_eval_result WHERE id=996103"
missing_before="$(snapshot 940103)"
set +e
run_recovery 940103 --dry-run "${test_dir}/missing-result.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/missing-result.out"
test "$(snapshot 940103)" = "${missing_before}"

# Wrong historical reconciliation identity.
insert_fixture 940104 992104 994104 996104 0.00079999998 \
    'child_exit_code_0;phase=infer;exit_code=0' process_missing_no_result
wrong_recon_before="$(snapshot 940104)"
set +e
run_recovery 940104 --dry-run "${test_dir}/wrong-reconciliation.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/wrong-reconciliation.out"
test "$(snapshot 940104)" = "${wrong_recon_before}"

# Threshold outside the 1e-7 tolerance.
insert_fixture 940105 992105 994105 996105 0.0006
threshold_before="$(snapshot 940105)"
set +e
run_recovery 940105 --dry-run "${test_dir}/threshold.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/threshold.out"
test "$(snapshot 940105)" = "${threshold_before}"

# Wrong experiment error signature must reject.
insert_fixture 940106 992106 994106 996106 0.00079999998 \
    'some_other_failure'
error_before="$(snapshot 940106)"
set +e
run_recovery 940106 --dry-run "${test_dir}/wrong-error.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/wrong-error.out"
test "$(snapshot 940106)" = "${error_before}"

# Historical experiment worker identity is evidence, not an active-worker fence.
# The base fixture deliberately retains PGID/start/executable/command/start time
# while worker_pid and active_scheduler_worker_attempt_id remain NULL.
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT (worker_pid IS NULL)::text||':'||
            (worker_process_group_id IS NOT NULL)::text||':'||
            (worker_process_start_identity IS NOT NULL)::text||':'||
            (worker_executable IS NOT NULL)::text||':'||
            (worker_command_line IS NOT NULL)::text||':'||
            (worker_started_at IS NOT NULL)::text
     FROM experiment WHERE experiment_id=940102")" = \
    "true:true:true:true:true:true"

# Ambiguous evidence: a second otherwise qualifying failed inference attempt
# paired to the same durable result must fail closed.
insert_fixture 940107 992107 994107 996107
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO experiment_scheduler_worker_attempt(
         worker_attempt_id,launch_attempt_identity,experiment_id,checkpoint_eval_id,
         worker_kind,lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
         command_identity,reserved_at,completed_at,exit_code,reconciliation_result)
     VALUES(
         992108,'test:historical-recovery:992108',940107,NULL,
         'experiment','infer','infer','scheduler_launch','failed',
         'experiment:940107:infer',
         clock_timestamp()-interval '80 minutes',
         clock_timestamp()-interval '50 minutes',0,'parent_observed_exit')"
ambiguous_before="$(snapshot 940107)"
set +e
run_recovery 940107 --dry-run "${test_dir}/ambiguous.out"
rc=$?
set -e
test "${rc}" = "1"
grep -q 'outcome=rejected' "${test_dir}/ambiguous.out"
test "$(snapshot 940107)" = "${ambiguous_before}"

# Deferred commit failure must roll back both the attempt and experiment updates.
insert_fixture 940108 992109 994108 996108
commit_before="$(snapshot 940108)"
commit_counts="$(counts 940108)"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
CREATE OR REPLACE FUNCTION historical_recovery_test_commit_failure()
RETURNS trigger LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.experiment_id = 940108 AND NEW.status = 'pending' AND NEW.phase = 'analyze' THEN
        RAISE EXCEPTION 'forced historical recovery commit failure';
    END IF;
    RETURN NEW;
END;
$$;

CREATE CONSTRAINT TRIGGER historical_recovery_test_commit_failure_trigger
AFTER UPDATE ON experiment
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION historical_recovery_test_commit_failure();
SQL

set +e
run_recovery 940108 --yes "${test_dir}/commit-failure.out"
rc=$?
set -e
test "${rc}" = "2"
grep -q 'HISTORICAL_FAILED_INFERENCE_RECOVERY_ERROR' "${test_dir}/commit-failure.out"
! grep -q 'outcome=applied' "${test_dir}/commit-failure.out"
test "$(snapshot 940108)" = "${commit_before}"
test "$(counts 940108)" = "${commit_counts}"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
DROP TRIGGER historical_recovery_test_commit_failure_trigger ON experiment;
DROP FUNCTION historical_recovery_test_commit_failure();
SQL

printf '%s\n' "HistoricalFailedInferenceRecoveryIntegrationTests passed"
