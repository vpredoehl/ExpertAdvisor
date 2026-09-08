#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/Development/SchedulerRecoveryRequeuePreemption/Build/Products/Release/LSTM_Release}"
test_db="ea_scheduler_training_requeue_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-scheduler-training-requeue.XXXXXX")"
# The development workspace defaults PostgreSQL sessions to read-only. This
# fixture targets only its uniquely named disposable database and needs writes.
export PGOPTIONS=

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

diagnose_error() {
    local result=$?
    for diagnostic in "${test_dir}"/*.out; do
        [[ -f "${diagnostic}" ]] || continue
        printf '%s\n' "--- ${diagnostic} ---" >&2
        tail -80 "${diagnostic}" >&2 || true
    done
    psql -X -Atq -d "${test_db}" -c \
        "SELECT experiment_id,status,phase,current_epoch,target_epochs,
                last_model_id,resume_model_id,resume_requested,
                scheduler_resume_origin,scheduler_priority,
                active_scheduler_worker_attempt_id
         FROM experiment ORDER BY experiment_id" >&2 || true
    return "${result}"
}
trap diagnose_error ERR

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql \
    086_scheduler_pause_resume_priority.sql \
    093_scheduler_priority_preemption.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING;
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment_scheduler_invocation(
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status,protocol_generation
) VALUES(
    'training-requeue-test-scheduler',999991,999991,'test-start',
    '/test/LSTM_Release','LSTM_Release --schedule-experiments',
    'training-requeue-test-nonce-0001','lost',52
);
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='training-requeue-integration',
    cutover_executable_path='/test/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;
SQL

scalar() { psql -X -Atq -d "${test_db}" -c "$1"; }
run_cli() { LSTM_DB_NAME="${test_db}" "${scheduler_binary}" "$@"; }

insert_experiment() {
    local experiment_id="$1" symbol="$2" status="$3" phase="$4"
    local current_epoch="$5" priority="$6"
    local current_operation="${phase}"
    [[ "${phase}" != done ]] || current_operation="NULL"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_epoch,current_operation,duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,scheduler_priority,
    exit_code,error_message,completed_at
) VALUES(
    ${experiment_id},'${symbol}',4,0.0008,1.0,1.0,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    '${status}','${phase}',${current_epoch},
    $(if [[ "${current_operation}" = NULL ]]; then printf 'NULL'; else printf "'%s'" "${current_operation}"; fi),
    ${experiment_id},36,5,'${priority}',
    $(if [[ "${status}" = failed ]]; then printf '9'; else printf 'NULL'; fi),
    $(if [[ "${status}" = failed ]]; then printf "'fixture failure'"; else printf 'NULL'; fi),
    $(if [[ "${status}" = completed ]]; then printf 'clock_timestamp()'; else printf 'NULL'; fi)
);
SQL
}

insert_model() {
    local experiment_id="$1" symbol="$2" completed_epochs="$3"
    local comment="${4:-periodic training checkpoint}" model_id
    model_id="$(psql -X -Atq -d "${test_db}" -c \
        "INSERT INTO model(experiment_id,name,comment)
         VALUES(${experiment_id},'requeue-${experiment_id}','${comment}')
         RETURNING model_id")"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_config_meta',1,14,0,col_idx,
       CASE col_idx WHEN 0 THEN 1 WHEN 1 THEN 4 WHEN 2 THEN 0.0008
           WHEN 10 THEN ${completed_epochs} WHEN 11 THEN 1.0
           WHEN 12 THEN 1.0 WHEN 13 THEN 1.0 ELSE 1.0 END
FROM generate_series(0,13) AS columns(col_idx);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_symbol_meta',1,length('${symbol}'),0,position-1,
       ascii(substr('${symbol}',position,1))
FROM generate_series(1,length('${symbol}')) AS chars(position);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_range_meta',1,length('2020-01-01|2021-01-01'),0,
       position-1,ascii(substr('2020-01-01|2021-01-01',position,1))
FROM generate_series(1,length('2020-01-01|2021-01-01')) AS chars(position);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'model_meta',1,3,0,i-1,v[i]
FROM (SELECT ARRAY[1.0,36.0,1.0] v) data,generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'param',37,4,(i-1)/4,(i-1)%4,0.0
FROM generate_series(1,148) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'bias',1,4,0,i-1,0.0 FROM generate_series(1,4) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
VALUES
    (${model_id},'returnHeadWeight',1,1,0,0,0.0),
    (${model_id},'returnHeadBias',1,1,0,0,0.0);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'returnHeadDirWeight',1,3,0,i-1,0.0
FROM generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'returnHeadDirBias',1,3,0,i-1,0.0
FROM generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'target_meta',1,6,0,i-1,v[i]
FROM (SELECT ARRAY[2.0,1.0,0.0,0.0,0.0,1.0] v) data,
     generate_series(1,6) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'optimizer_meta',1,5,0,i-1,v[i]
FROM (SELECT ARRAY[1.0,1.0,${completed_epochs}::double precision,0.0,0.0] v) data,
     generate_series(1,5) i;
SQL
    echo "${model_id}"
}

insert_attempt() {
    local experiment_id="$1" phase="$2" lifecycle="$3" bind="$4" attempt_id
    attempt_id="$(PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -Atq -d "${test_db}" -c \
        "INSERT INTO experiment_scheduler_worker_attempt(
             launch_attempt_identity,scheduler_invocation_id,scheduler_fencing_token,
             experiment_id,worker_kind,lifecycle_phase,capacity_class,
             ownership_origin,lifecycle_state,worker_pid,worker_process_group_id,
             worker_process_start_identity,canonical_executable_path,command_line,
             command_identity,reserved_at,spawned_at,registered_at
         ) VALUES(
             'requeue-attempt-${experiment_id}','training-requeue-test-scheduler',52,
             ${experiment_id},'experiment','${phase}','${phase}',
             'scheduler_launch','${lifecycle}',987654,987654,'missing-test-process',
             '/test/LSTM_Release','LSTM_Release --${phase}',
             'experiment:${experiment_id}:${phase}',clock_timestamp(),
             clock_timestamp(),clock_timestamp()) RETURNING worker_attempt_id")"
    if [[ "${bind}" = true ]]; then
        PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
            psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
            "UPDATE experiment SET active_scheduler_worker_attempt_id=${attempt_id}
             WHERE experiment_id=${experiment_id}"
    fi
    echo "${attempt_id}"
}

# B1/B10/B11: pending infer is returned to train under the same ID and priority.
insert_experiment 930001 successhigh pending infer 35 high
success_model="$(insert_model 930001 successhigh 40)"
experiment_count_before="$(scalar 'SELECT count(*) FROM experiment')"
run_cli --requeue-training=930001 --yes >"${test_dir}/success.out"
grep -q "SCHEDULER_REQUEUE_TRAINING_SELECTION,experiment_id=930001,resume_model_id=${success_model},completed_epoch=40,scheduler_priority=high" "${test_dir}/success.out"
test "$(scalar "SELECT status||':'||phase||':'||current_operation||':'||last_model_id||':'||resume_model_id||':'||resume_requested||':'||scheduler_resume_origin||':'||scheduler_priority FROM experiment WHERE experiment_id=930001")" = "pending:train:train:${success_model}:${success_model}:false:none:high"
test "${experiment_count_before}" = "$(scalar 'SELECT count(*) FROM experiment')"

# B2: an unusable newest candidate falls back through the Phase A selector.
insert_experiment 930002 fallbacknormal pending infer 20 normal
fallback_model="$(insert_model 930002 fallbacknormal 20)"
invalid_newest="$(insert_model 930002 fallbacknormal 60)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${invalid_newest} AND param_name='optimizer_meta'"
run_cli --requeue-training 930002 --yes >"${test_dir}/fallback.out"
grep -q "SCHEDULER_CHECKPOINT_CANDIDATE_REJECTED,experiment_id=930002,model_id=${invalid_newest}" "${test_dir}/fallback.out"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=930002,model_id=${fallback_model}" "${test_dir}/fallback.out"

expect_rejected_unchanged() {
    local experiment_id="$1" expected_reason="$2" output="$3"
    shift 3
    local before result
    before="$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=${experiment_id}")"
    if run_cli "$@" >"${test_dir}/${output}.out" 2>&1; then
        result=0
    else
        result=$?
    fi
    test "${result}" -ne 0
    grep -q "reason=${expected_reason}" "${test_dir}/${output}.out"
    test "${before}" = "$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=${experiment_id}")"
}

# B3/B5: no candidate and incompatible contents fail closed.
insert_experiment 930003 nocheckpoint pending infer 10 normal
expect_rejected_unchanged 930003 no_valid_checkpoint no-checkpoint --requeue-training=930003 --yes
insert_experiment 930005 compatible pending infer 10 normal
insert_model 930005 wrongsymbol 20 >/dev/null
expect_rejected_unchanged 930005 no_valid_checkpoint incompatible --requeue-training=930005 --yes

# B6: equal-epoch intermediate checkpoints use the authoritative deterministic
# ordering and requeue the same experiment ID without creating a new row.
insert_experiment 930006 ambiguous pending infer 10 normal
older_tied_model="$(insert_model 930006 ambiguous 20)"
newer_tied_model="$(insert_model 930006 ambiguous 20)"
tied_count_before="$(scalar 'SELECT count(*) FROM experiment')"
run_cli --requeue-training=930006 --yes >"${test_dir}/tied-checkpoints.out"
grep -q "SCHEDULER_CHECKPOINT_CANDIDATE_TIE_BREAK,experiment_id=930006,completed_epoch=20,candidate_count=2,candidate_order=periodic_flag_asc_model_id_desc,selected_model_id=${newer_tied_model}" "${test_dir}/tied-checkpoints.out"
grep -q "SCHEDULER_REQUEUE_TRAINING_SELECTION,experiment_id=930006,resume_model_id=${newer_tied_model},completed_epoch=20,scheduler_priority=normal" "${test_dir}/tied-checkpoints.out"
test "$(scalar "SELECT status||':'||phase||':'||current_operation||':'||resume_model_id FROM experiment WHERE experiment_id=930006")" = "pending:train:train:${newer_tied_model}"
test "${tied_count_before}" = "$(scalar 'SELECT count(*) FROM experiment')"
test "${older_tied_model}" -lt "${newer_tied_model}"

# B4: completed/final work never reopens.
insert_experiment 930004 final completed done 80 normal
insert_model 930004 final 80 'final training model' >/dev/null
expect_rejected_unchanged 930004 terminal_experiment_cannot_be_requeued final --requeue-training=930004 --yes

# B7: dry-run reports selection but changes neither rows nor counts.
insert_experiment 930007 dryrun pending infer 10 low
insert_model 930007 dryrun 20 >/dev/null
dry_before="$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=930007")"
attempts_before="$(scalar 'SELECT count(*) FROM experiment_scheduler_worker_attempt')"
models_before="$(scalar 'SELECT count(*) FROM model')"
run_cli --requeue-training=930007 --dry-run >"${test_dir}/dry-run.out"
grep -q 'SCHEDULER_CONTROL_DRY_RUN,action=requeue_training,experiment_id=930007' "${test_dir}/dry-run.out"
test "${dry_before}" = "$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=930007")"
test "${attempts_before}" = "$(scalar 'SELECT count(*) FROM experiment_scheduler_worker_attempt')"
test "${models_before}" = "$(scalar 'SELECT count(*) FROM model')"

# B8/B9: missing IDs reject and unconfirmed valid requests do not mutate.
if run_cli --requeue-training=939999 --yes >"${test_dir}/missing.out" 2>&1; then
    missing_result=0
else
    missing_result=$?
fi
test "${missing_result}" -ne 0
grep -q 'reason=experiment_not_found' "${test_dir}/missing.out"
insert_experiment 930009 unconfirmed pending infer 10 normal
insert_model 930009 unconfirmed 20 >/dev/null
unconfirmed_before="$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=930009")"
run_cli --requeue-training=930009 >"${test_dir}/unconfirmed.out"
grep -q 'Use --yes to apply.' "${test_dir}/unconfirmed.out"
test "${unconfirmed_before}" = "$(scalar "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=930009")"

# B10: all persistent priority values survive exactly.
for entry in '930010 high' '930011 normal' '930012 low'; do
    read -r experiment_id priority <<<"${entry}"
    insert_experiment "${experiment_id}" "priority${experiment_id}" failed train 10 "${priority}"
    insert_model "${experiment_id}" "priority${experiment_id}" 20 >/dev/null
    run_cli --requeue-training="${experiment_id}" --yes >/dev/null
    test "$(scalar "SELECT scheduler_priority FROM experiment WHERE experiment_id=${experiment_id}")" = "${priority}"
done

# B12/B13: active and stopped attempts are rejected without detachment.
insert_experiment 930013 attachedrunning pending infer 10 high
running_attempt="$(insert_attempt 930013 infer running true)"
expect_rejected_unchanged 930013 requeue_training_worker_attempt_attached attached-running --requeue-training=930013 --yes
test "$(scalar "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${running_attempt}")" = running
insert_experiment 930014 attachedstopped pending train 10 low
stopped_attempt="$(insert_attempt 930014 train stopped true)"
expect_rejected_unchanged 930014 requeue_training_worker_attempt_attached attached-stopped --requeue-training=930014 --yes
test "$(scalar "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${stopped_attempt}")" = stopped

# B14: historical checkpoint-stop evidence remains intact on success.
insert_experiment 930015 history pending infer 20 normal
history_model="$(insert_model 930015 history 20)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET stopped_at_checkpoint_epoch=20,
     stopped_at_checkpoint_model_id=${history_model} WHERE experiment_id=930015"
run_cli --requeue-training=930015 --yes >/dev/null
test "$(scalar "SELECT stopped_at_checkpoint_epoch||':'||stopped_at_checkpoint_model_id FROM experiment WHERE experiment_id=930015")" = "20:${history_model}"

printf '%s\n' 'SchedulerTrainingRequeueIntegrationTests passed'
