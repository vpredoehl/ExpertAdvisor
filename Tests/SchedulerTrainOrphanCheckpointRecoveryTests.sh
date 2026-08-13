#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
test_db="ea_scheduler_train_orphan_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_train_orphan.XXXXXX)"

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
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
    'orphan-test-scheduler',999991,999991,'orphan-test-start',
    '/test/LSTM_Release','LSTM_Release --schedule-experiments',
    'orphan-test-scheduler-nonce-0001','lost',52
);
UPDATE experiment_scheduler_protocol
SET required_generation=52, cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='scheduler-train-orphan-test',
    cutover_executable_path='/test/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL, updated_at=clock_timestamp()
WHERE singleton=true;
SQL

insert_experiment() {
    local experiment_id="$1" symbol="$2" target_epochs="$3"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_epoch,current_operation,stop_after_checkpoint_epoch,
    duplicate_nonce
) VALUES(
    ${experiment_id},'${symbol}',4,0.0008,1.0,1.0,${target_epochs},20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'running','train',57,'train',60,${experiment_id}
);
SQL
}

insert_model() {
    local experiment_id="$1" symbol="$2" completed_epochs="$3" model_id
    model_id="$(psql -X -At -q -d "${test_db}" -c \
        "INSERT INTO model(experiment_id,name,comment)
         VALUES(${experiment_id},'orphan-${experiment_id}','recovery fixture')
         RETURNING model_id")"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_config_meta',1,14,0,col_idx,
       CASE col_idx WHEN 0 THEN 1 WHEN 1 THEN 4 WHEN 2 THEN 0.0008
           WHEN 10 THEN ${completed_epochs} WHEN 11 THEN 1.0
           WHEN 12 THEN 1.0 WHEN 13 THEN 1.0 ELSE 1.0 END
FROM generate_series(0,13) AS columns(col_idx);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_symbol_meta',1,length('${symbol}'),0,position - 1,
       ascii(substr('${symbol}',position,1))
FROM generate_series(1,length('${symbol}')) AS chars(position);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_range_meta',1,length('2020-01-01|2021-01-01'),0,
       position - 1,ascii(substr('2020-01-01|2021-01-01',position,1))
FROM generate_series(1,length('2020-01-01|2021-01-01')) AS chars(position);
SQL
    echo "${model_id}"
}

insert_attempt() {
    local experiment_id="$1" attempt_identity="$2" bind_experiment="$3" attempt_id
    attempt_id="$(PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -At -q -d "${test_db}" -c \
        "INSERT INTO experiment_scheduler_worker_attempt(
             launch_attempt_identity,scheduler_invocation_id,scheduler_fencing_token,
             experiment_id,worker_kind,lifecycle_phase,capacity_class,
             ownership_origin,lifecycle_state,worker_pid,worker_process_group_id,
             worker_process_start_identity,canonical_executable_path,command_line,
             command_identity,reserved_at,spawned_at,registered_at
         ) VALUES(
             '${attempt_identity}','orphan-test-scheduler',52,${experiment_id},
             'experiment','train','train','scheduler_launch','identity_ambiguous',
             987654,987654,'missing-process-identity','/missing/LSTM_Release',
             'LSTM_Release --train --scheduler-experiment-id=${experiment_id}',
             'experiment:${experiment_id}:train',clock_timestamp()-interval '1 minute',
             clock_timestamp()-interval '1 minute',clock_timestamp()-interval '1 minute'
         ) RETURNING worker_attempt_id")"
    if [[ "${bind_experiment}" = true ]]; then
        PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
            "UPDATE experiment SET active_scheduler_worker_attempt_id=${attempt_id}
             WHERE experiment_id=${experiment_id}"
    fi
    echo "${attempt_id}"
}

insert_experiment 920070 orphanfixture 80
intermediate_model="$(insert_model 920070 orphanfixture 40)"
intermediate_attempt="$(insert_attempt 920070 orphan-attempt-920070 true)"
insert_experiment 920071 finalfixture 80
final_model="$(insert_model 920071 finalfixture 80)"
final_attempt="$(insert_attempt 920071 orphan-attempt-920071 true)"
insert_experiment 920072 mismatchfixture 80
mismatch_model="$(insert_model 920072 wrongfixture 40)"
mismatch_attempt="$(insert_attempt 920072 orphan-attempt-920072 true)"
insert_experiment 920073 unboundfixture 80
unbound_model="$(insert_model 920073 unboundfixture 40)"
unbound_attempt="$(insert_attempt 920073 orphan-attempt-920073 false)"

attempts_before="$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt")"
dry_run_output="${test_dir}/dry-run.out"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only --dry-run \
    >"${dry_run_output}" 2>&1
grep -q 'SCHEDULER_ORPHAN_RECOVERY_SKIPPED,dry_run=1' "${dry_run_output}"
! grep -q 'SCHEDULER_ORPHAN_RECOVERY_DONE' "${dry_run_output}"
test "${attempts_before}" = "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt")"

recovery_output="${test_dir}/recovery.out"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    >"${recovery_output}" 2>&1
grep -q 'SCHEDULER_ORPHAN_RECOVERY_DONE,recovered_or_failed=4' "${recovery_output}"
grep -q "SCHEDULER_ORPHAN_RECOVERED,experiment_id=920070,resume_model_id=${intermediate_model}" "${recovery_output}"
grep -q "SCHEDULER_ORPHAN_ADVANCED,experiment_id=920071,model_id=${final_model}" "${recovery_output}"

test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text||':'||resume_model_id::text||':'||current_epoch::text||':'||stop_after_checkpoint_epoch::text||':'||current_operation FROM experiment WHERE experiment_id=920070")" = "pending:train:${intermediate_model}:${intermediate_model}:57:60:train"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result||':'||(completed_at IS NOT NULL)::text FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${intermediate_attempt}")" = "completed:process_missing_result_recovered:true"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text||':'||(resume_model_id IS NULL)::text||':'||current_operation FROM experiment WHERE experiment_id=920071")" = "pending:infer:${final_model}:true:train"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${final_attempt}")" = "completed:process_missing_result_recovered"

test "$(scalar "SELECT status||':'||phase||':'||(last_model_id IS NULL)::text||':'||(resume_model_id IS NULL)::text FROM experiment WHERE experiment_id=920072")" = "failed:train:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${mismatch_attempt}")" = "failed:process_missing_no_result"
test "$(scalar "SELECT status||':'||phase||':'||(last_model_id IS NULL)::text||':'||(resume_model_id IS NULL)::text FROM experiment WHERE experiment_id=920073")" = "running:train:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${unbound_attempt}")" = "failed:lifecycle_predicate_changed"

snapshot="$(scalar "SELECT string_agg(worker_attempt_id::text||':'||lifecycle_state||':'||COALESCE(reconciliation_result,'NULL'),',' ORDER BY worker_attempt_id) FROM experiment_scheduler_worker_attempt")"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    >"${test_dir}/recovery-replay.out" 2>&1
test "${snapshot}" = "$(scalar "SELECT string_agg(worker_attempt_id::text||':'||lifecycle_state||':'||COALESCE(reconciliation_result,'NULL'),',' ORDER BY worker_attempt_id) FROM experiment_scheduler_worker_attempt")"
test "${attempts_before}" = "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt")"

printf '%s\n' "SchedulerTrainOrphanCheckpointRecoveryTests passed"
