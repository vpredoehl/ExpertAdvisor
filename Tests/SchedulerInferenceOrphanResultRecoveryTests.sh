#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"
test_db="ea_scheduler_infer_orphan_result_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_infer_orphan_result.XXXXXX)"

cleanup() {
    local status=$?
    if [[ "${status}" -ne 0 ]]; then
        for output in "${test_dir}"/*.out; do
            [[ -f "${output}" ]] || continue
            sed -n '1,260p' "${output}" >&2
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
for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql \
    086_scheduler_pause_resume_priority.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running')
ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment_scheduler_invocation(
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status,protocol_generation
) VALUES(
    'phase17d-infer-orphan-owner',2147480000,2147480000,
    'absent-phase17d-owner','/missing/LSTM_Release',
    'LSTM_Release --schedule-experiments',
    'phase17d-infer-orphan-owner-nonce','lost',52
);
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='phase17d-infer-orphan-test',
    cutover_executable_path='/missing/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;

INSERT INTO model(model_id,name) VALUES
    (918011,'phase17d-recovered-model'),
    (918012,'phase17d-mismatch-model'),
    (918013,'phase17d-missing-model'),
    (918014,'phase17d-profitability-only-model'),
    (918015,'phase17d-administrative-model');
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,current_operation,last_model_id,
    duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,worker_control_state
) VALUES
    (918001,'recoverexact',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'running','infer','infer',918011,918001,77,7,'running'),
    (918002,'mismatchresult',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'running','infer','infer',918012,918002,77,7,'running'),
    (918003,'missingresult',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'running','infer','infer',918013,918003,77,7,'running'),
    (918004,'profitabilityonly',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'running','infer','infer',918014,918004,77,7,'running');
UPDATE model SET experiment_id=918001 WHERE model_id=918011;
UPDATE model SET experiment_id=918002 WHERE model_id=918012;
UPDATE model SET experiment_id=918003 WHERE model_id=918013;
UPDATE model SET experiment_id=918004 WHERE model_id=918014;
SQL

insert_attempt() {
    local experiment_id="$1" state="$2" attempt_identity="$3"
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
             '${attempt_identity}','phase17d-infer-orphan-owner',52,
             ${experiment_id},'experiment','infer','infer',
             'scheduler_launch','${state}',2147480000,2147480000,
             'absent-infer-process-${experiment_id}','/missing/LSTM_Release',
             '/missing/LSTM_Release --infer --scheduler-experiment-id=${experiment_id}',
             'experiment:${experiment_id}:infer',
             clock_timestamp()-interval '2 minutes',
             clock_timestamp()-interval '2 minutes',
             clock_timestamp()-interval '2 minutes'
         ) RETURNING worker_attempt_id"
}

bind_attempt() {
    local experiment_id="$1" attempt_id="$2"
    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
        "UPDATE experiment e SET
             active_scheduler_worker_attempt_id=${attempt_id},
             worker_pid=a.worker_pid,
             worker_process_group_id=a.worker_process_group_id,
             worker_process_start_identity=a.worker_process_start_identity,
             worker_executable=a.canonical_executable_path,
             worker_command_line=a.command_line,
             worker_started_at=clock_timestamp()-interval '2 minutes'
         FROM experiment_scheduler_worker_attempt a
         WHERE e.experiment_id=${experiment_id}
           AND a.worker_attempt_id=${attempt_id}"
}

recovered_attempt="$(insert_attempt 918001 running phase17d-recovered-attempt)"
mismatch_attempt="$(insert_attempt 918002 running phase17d-mismatch-attempt)"
missing_attempt="$(insert_attempt 918003 running phase17d-missing-attempt)"
profitability_attempt="$(insert_attempt 918004 running phase17d-profitability-attempt)"
bind_attempt 918001 "${recovered_attempt}"
bind_attempt 918002 "${mismatch_attempt}"
bind_attempt 918003 "${missing_attempt}"
bind_attempt 918004 "${profitability_attempt}"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO inference_eval_result(
    id,model_id,symbol,prediction_horizon,threshold_logret,
    window_size,label_rule_id,target_type,from_date,to_date,
    completed_epochs,accuracy,accept_model,status,completed_at,
    inference_scope,checkpoint_eval_id,parent_experiment_id,checkpoint_epoch
) VALUES
    -- Regression fixture from production experiment 625:
    -- c_next_threshold=0.0008 but inference persistence produced
    -- threshold_logret=0.0007999999797903001. Exact float8 equality is false;
    -- the values match under the established 1e-7 threshold tolerance.
    (918021,918011,'recoverexact',4,0.0007999999797903001,
     30,1,0,'2020-02-01','2020-03-01',20,0.5,false,'completed',
     clock_timestamp(),'final',NULL,NULL,NULL),
    (918022,918012,'wrong-symbol',4,0.0008,
     30,1,0,'2020-02-01','2020-03-01',20,0.5,false,'completed',
     clock_timestamp(),'final',NULL,NULL,NULL),
    (918024,918014,'profitabilityonly',4,0.0008,
     30,1,0,'2019-02-01','2019-03-01',20,0.5,false,'completed',
     clock_timestamp(),'final',NULL,NULL,NULL);

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
    918004,918014,918024,'final',NULL,'2019-02-01','2019-03-01',
    3,2,1,1,0.02,-0.01,0.01,0.005,
    'phase17d-profitability-only','fnv1a64:1111111111111111',
    'fnv1a64:2222222222222222','phase17d-profitability-only',
    'fnv1a64:3333333333333333');
SQL

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/recovery-logs" \
    >"${test_dir}/recovery.out" 2>&1

grep -q "SCHEDULER_WORKER_RESULT_RECOVERED,experiment_id=918001,model_id=918011,phase=infer,reason=completed_inference_result" \
    "${test_dir}/recovery.out"
test "$(scalar "SELECT status||':'||phase||':'||exit_code||':'||
    worker_control_state||':'||(worker_pid IS NULL)::text||':'||
    (worker_process_group_id IS NULL)::text||':'||
    (worker_process_start_identity IS NULL)::text||':'||
    (worker_executable IS NULL)::text||':'||
    (worker_command_line IS NULL)::text||':'||
    (current_operation IS NULL)::text||':'||
    (active_scheduler_worker_attempt_id IS NULL)::text
    FROM experiment WHERE experiment_id=918001")" = \
    "pending:analyze:0:paused:true:true:true:true:true:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result||':'||
    (completed_at IS NOT NULL)::text FROM experiment_scheduler_worker_attempt
    WHERE worker_attempt_id=${recovered_attempt}")" = \
    "completed:process_missing_result_recovered:true"

for fixture in \
    "918002:${mismatch_attempt}" \
    "918003:${missing_attempt}" \
    "918004:${profitability_attempt}"; do
    experiment_id="${fixture%%:*}"
    attempt_id="${fixture##*:}"
    test "$(scalar "SELECT status||':'||phase||':'||error_message||':'||
        worker_control_state||':'||(worker_pid IS NULL)::text||':'||
        (worker_process_group_id IS NULL)::text||':'||
        (worker_process_start_identity IS NULL)::text||':'||
        (worker_executable IS NULL)::text||':'||
        (worker_command_line IS NULL)::text||':'||
        (current_operation IS NULL)::text||':'||
        (active_scheduler_worker_attempt_id IS NULL)::text
        FROM experiment WHERE experiment_id=${experiment_id}")" = \
        "failed:infer:worker_process_missing_no_result:paused:true:true:true:true:true:true:true"
    test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result
        FROM experiment_scheduler_worker_attempt
        WHERE worker_attempt_id=${attempt_id}")" = \
        "failed:process_missing_no_result"
done

test "$(scalar "SELECT count(*) FROM inference_eval_result")" = "3"
test "$(scalar "SELECT count(*) FROM inference_profitability_observation")" = "1"
first_snapshot="$(scalar "SELECT md5(string_agg(row_to_json(x)::text,'|' ORDER BY x.experiment_id))
    FROM (SELECT e.experiment_id,e.status,e.phase,e.error_message,
                 e.worker_control_state,e.active_scheduler_worker_attempt_id,
                 a.lifecycle_state,a.reconciliation_result
          FROM experiment e
          JOIN experiment_scheduler_worker_attempt a USING(experiment_id)
          WHERE e.experiment_id BETWEEN 918001 AND 918004) x")"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/second-recovery-logs" \
    >"${test_dir}/second-recovery.out" 2>&1
second_snapshot="$(scalar "SELECT md5(string_agg(row_to_json(x)::text,'|' ORDER BY x.experiment_id))
    FROM (SELECT e.experiment_id,e.status,e.phase,e.error_message,
                 e.worker_control_state,e.active_scheduler_worker_attempt_id,
                 a.lifecycle_state,a.reconciliation_result
          FROM experiment e
          JOIN experiment_scheduler_worker_attempt a USING(experiment_id)
          WHERE e.experiment_id BETWEEN 918001 AND 918004) x")"
test "${first_snapshot}" = "${second_snapshot}"
test "$(scalar "SELECT count(*) FROM inference_eval_result")" = "3"
test "$(scalar "SELECT count(*) FROM inference_profitability_observation")" = "1"

# The separately reachable administrative exact-attempt path shares the same
# durable result rule and must not recreate the false failure.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,current_operation,last_model_id,
    duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,worker_control_state
) VALUES(
    918005,'adminrecover',4,0.0008,20,20,
    '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
    'running','infer','infer',918015,918005,77,7,'running');
UPDATE model SET experiment_id=918005 WHERE model_id=918015;
SQL
admin_attempt="$(insert_attempt 918005 identity_ambiguous phase17d-admin-attempt)"
bind_attempt 918005 "${admin_attempt}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO inference_eval_result(
         id,model_id,symbol,prediction_horizon,threshold_logret,
         window_size,label_rule_id,target_type,from_date,to_date,
         completed_epochs,accuracy,accept_model,status,completed_at,
         inference_scope,checkpoint_eval_id,parent_experiment_id,
         checkpoint_epoch)
     VALUES(918025,918015,'adminrecover',4,0.0008,30,1,0,
         '2020-02-01','2020-03-01',20,0.5,false,'completed',
         clock_timestamp(),'final',NULL,NULL,NULL)"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${admin_attempt}" --yes \
    >"${test_dir}/administrative-recovery.out" 2>&1
grep -q "outcome=applied,worker_attempt_id=${admin_attempt},.*proposed_lifecycle_state=completed,.*reason=exact_process_absent_completed_inference_result" \
    "${test_dir}/administrative-recovery.out"
test "$(scalar "SELECT status||':'||phase||':'||exit_code||':'||
    worker_control_state||':'||(worker_pid IS NULL)::text||':'||
    (worker_process_group_id IS NULL)::text||':'||
    (worker_process_start_identity IS NULL)::text||':'||
    (worker_executable IS NULL)::text||':'||
    (worker_command_line IS NULL)::text||':'||
    (current_operation IS NULL)::text||':'||
    (active_scheduler_worker_attempt_id IS NULL)::text
    FROM experiment WHERE experiment_id=918005")" = \
    "pending:analyze:0:paused:true:true:true:true:true:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result
    FROM experiment_scheduler_worker_attempt
    WHERE worker_attempt_id=${admin_attempt}")" = \
    "completed:process_missing_result_recovered"

set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${admin_attempt}" --yes \
    >"${test_dir}/administrative-replay.out" 2>&1
replay_status=$?
set -e
test "${replay_status}" = "1"
test "$(scalar "SELECT count(*) FROM inference_eval_result WHERE id=918025")" = "1"

printf '%s\n' "SchedulerInferenceOrphanResultRecoveryTests passed"
