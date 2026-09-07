#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
test_db="ea_scheduler_train_orphan_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_train_orphan.XXXXXX)"
export PGOPTIONS=

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
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/078_operator_forced_final_inference_rerun.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/093_scheduler_priority_preemption.sql"
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
    duplicate_nonce,model_input_width,
    model_input_semantic_layout_version
) VALUES(
    ${experiment_id},'${symbol}',4,0.0008,1.0,1.0,${target_epochs},20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'running','train',57,'train',60,${experiment_id},36,5
);
SQL
}

insert_model() {
    local experiment_id="$1" symbol="$2" completed_epochs="$3"
    local comment="${4:-periodic training checkpoint}" model_id
    model_id="$(psql -X -At -q -d "${test_db}" -c \
        "INSERT INTO model(experiment_id,name,comment)
         VALUES(${experiment_id},'orphan-${experiment_id}','${comment}')
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
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'model_meta',1,3,0,i-1,v[i]
FROM (SELECT ARRAY[1.0,36.0,1.0] v) data,
     generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'param',37,4,(i-1)/4,(i-1)%4,0.0
FROM generate_series(1,148) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'bias',1,4,0,i-1,0.0
FROM generate_series(1,4) i;
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

insert_experiment 920070 newestvalidfixture 80
older_model="$(insert_model 920070 newestvalidfixture 20)"
newest_model="$(insert_model 920070 newestvalidfixture 40)"
newest_attempt="$(insert_attempt 920070 orphan-attempt-920070 true)"

insert_experiment 920071 fallbackfixture 80
fallback_model="$(insert_model 920071 fallbackfixture 40)"
invalid_newest_model="$(insert_model 920071 fallbackfixture 60)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${invalid_newest_model} AND param_name='optimizer_meta'"
fallback_attempt="$(insert_attempt 920071 orphan-attempt-920071 true)"

insert_experiment 920072 thirdfixture 80
third_model="$(insert_model 920072 thirdfixture 20)"
invalid_second_model="$(insert_model 920072 thirdfixture 40)"
invalid_first_model="$(insert_model 920072 thirdfixture 60)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${invalid_second_model} AND param_name='returnHeadDirWeight'"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${invalid_first_model} AND param_name='returnHeadDirBias'"
third_attempt="$(insert_attempt 920072 orphan-attempt-920072 true)"

insert_experiment 920073 unusablefixture 80
unusable_model="$(insert_model 920073 unusablefixture 40)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${unusable_model} AND param_name='param'"
unusable_attempt="$(insert_attempt 920073 orphan-attempt-920073 true)"

insert_experiment 920074 ambiguousfixture 80
ambiguous_fallback_model="$(insert_model 920074 ambiguousfixture 20)"
ambiguous_model_one="$(insert_model 920074 ambiguousfixture 40)"
ambiguous_model_two="$(insert_model 920074 ambiguousfixture 40)"
ambiguous_attempt="$(insert_attempt 920074 orphan-attempt-920074 true)"

insert_experiment 920075 intermediatefixture 80
intermediate_model="$(insert_model 920075 intermediatefixture 40)"
intermediate_attempt="$(insert_attempt 920075 orphan-attempt-920075 true)"

insert_experiment 920076 finalfixture 80
final_model="$(insert_model 920076 finalfixture 80 'final training model')"
final_attempt="$(insert_attempt 920076 orphan-attempt-920076 true)"

insert_experiment 920077 mismatchfixture 80
mismatch_model="$(insert_model 920077 wrongfixture 40)"
mismatch_attempt="$(insert_attempt 920077 orphan-attempt-920077 true)"

insert_experiment 920078 unboundfixture 80
unbound_model="$(insert_model 920078 unboundfixture 40)"
unbound_attempt="$(insert_attempt 920078 orphan-attempt-920078 false)"

# Exact-attempt and durable-start fencing. The epoch-60 model predates the
# current attempt and must lose to the attributable epoch-40 checkpoint.
insert_experiment 920082 attemptfencefixture 80
stale_attempt_model="$(insert_model 920082 attemptfencefixture 60)"
current_attempt_model="$(insert_model 920082 attemptfencefixture 40)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE model SET created_at=clock_timestamp()-interval '5 minutes'
     WHERE model_id=${stale_attempt_model}"
attempt_fence_attempt="$(insert_attempt 920082 orphan-attempt-920082 true)"

# All evidence before the durable attempt-start boundary is ineligible.
insert_experiment 920083 boundaryfixture 80
pre_attempt_model="$(insert_model 920083 boundaryfixture 60)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE model SET created_at=clock_timestamp()-interval '5 minutes'
     WHERE model_id=${pre_attempt_model}"
boundary_attempt="$(insert_attempt 920083 orphan-attempt-920083 true)"

# A higher model from another experiment is never a candidate.
insert_experiment 920084 experimentfencefixture 80
experiment_fence_model="$(insert_model 920084 experimentfencefixture 40)"
insert_experiment 920085 otherexperimentfixture 80
wrong_experiment_model="$(insert_model 920085 otherexperimentfixture 80 'final training model')"
experiment_fence_attempt="$(insert_attempt 920084 orphan-attempt-920084 true)"

# A uniquely identified final artifact can itself be invalid. Qualification
# must not mutate, and fallback continues at the earlier valid checkpoint.
insert_experiment 920086 invalidfinalfallbackfixture 80
invalid_final_fallback_model="$(insert_model 920086 invalidfinalfallbackfixture 40)"
invalid_final_model="$(insert_model 920086 invalidfinalfallbackfixture 80 'final training model')"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM matrix WHERE model_id=${invalid_final_model}
     AND param_name='optimizer_meta'"
invalid_final_attempt="$(insert_attempt 920086 orphan-attempt-920086 true)"

# ----------------------------------------------------------------------
# Continuation orphan regression fixtures.
#
# These deliberately use the production continuation lineage constraint:
#
#   parent_experiment_id = continuation_source_experiment_id
#   resume_model_id = continuation_source_model_id
#   target_epochs > continuation_source_epoch
#
# Operational recovery must therefore NEVER replace resume_model_id with an
# own-experiment checkpoint.
# ----------------------------------------------------------------------

create_continuation_source() {
    local source_experiment_id="$1"
    local symbol="$2"
    local source_epoch="$3"
    local child_target_epoch="$4"

    insert_experiment \
        "${source_experiment_id}" \
        "${symbol}" \
        "${source_epoch}"

    local source_model_id
    source_model_id="$(
        insert_model \
            "${source_experiment_id}" \
            "${symbol}" \
            "${source_epoch}" \
            "continuation source final model"
    )"

    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE experiment
SET status='completed',
    phase='done',
    current_epoch=${source_epoch},
    last_model_id=${source_model_id},
    resume_model_id=NULL,
    current_operation='analyze',
    stop_after_checkpoint_epoch=NULL,
    completed_at=clock_timestamp()
WHERE experiment_id=${source_experiment_id};

INSERT INTO experiment_analysis_result(
    experiment_id,model_id,symbol,prediction_horizon,
    target_epochs,completed_epochs,infer_accuracy,leader_score,
    analysis_status,analysis_scope
) VALUES(
    ${source_experiment_id},${source_model_id},'${symbol}',4,
    ${source_epoch},${source_epoch},0.90,0.80,
    'completed','final'
);
SQL

    local source_analysis_id
    source_analysis_id="$(
        scalar "
            SELECT analysis_id
            FROM experiment_analysis_result
            WHERE experiment_id=${source_experiment_id}
              AND model_id=${source_model_id}
              AND analysis_scope='final'
            ORDER BY analysis_id DESC
            LIMIT 1
        "
    )"

    local decision_id
    decision_id="$(
        psql -X -At -q -d "${test_db}" -c "
            INSERT INTO experiment_continuation_decision(
                source_experiment_id,
                source_model_id,
                source_analysis_id,
                source_epoch,
                target_epochs,
                decision,
                reason,
                observed_eval_count,
                patience_window,
                policy_revision,
                policy_hash,
                evidence_watermark
            ) VALUES(
                ${source_experiment_id},
                ${source_model_id},
                ${source_analysis_id},
                ${source_epoch},
                ${child_target_epoch},
                'eligible',
                'scheduler_orphan_continuation_regression',
                1,
                1,
                1,
                '0123456789abcdef',
                'fedcba9876543210'
            )
            RETURNING continuation_decision_id
        "
    )"

    printf '%s:%s\n' "${source_model_id}" "${decision_id}"
}

bind_continuation_child() {
    local child_experiment_id="$1"
    local source_experiment_id="$2"
    local source_model_id="$3"
    local source_epoch="$4"
    local decision_id="$5"

    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE experiment
SET resume_model_id=${source_model_id},
    parent_experiment_id=${source_experiment_id},
    continuation_source_experiment_id=${source_experiment_id},
    continuation_source_model_id=${source_model_id},
    continuation_source_epoch=${source_epoch},
    continuation_generation=1,
    continuation_decision_id=${decision_id}
WHERE experiment_id=${child_experiment_id};
SQL
}

# ----------------------------------------------------------------------
# 920080: exact 584 failure shape.
#
# Original scientific source epoch = 60
# child target                  = 120
#
# child artifacts:
#   epoch 100 periodic checkpoint
#   epoch 120 periodic checkpoint
#   epoch 120 ordinary final model
#
# Old behavior rejected epoch 120 as ambiguous, selected epoch 100, then
# attempted resume_model_id=<epoch100>, violating continuation lineage.
#
# Correct behavior selects the unique non-periodic target artifact and
# advances directly to inference.
# ----------------------------------------------------------------------

source_584_fixture="$(
    create_continuation_source \
        920060 \
        continuation584source \
        60 \
        120
)"
source_584_model="${source_584_fixture%%:*}"
source_584_decision="${source_584_fixture##*:}"

insert_experiment 920080 continuation584child 120
bind_continuation_child \
    920080 \
    920060 \
    "${source_584_model}" \
    60 \
    "${source_584_decision}"

continuation_584_epoch100="$(
    insert_model 920080 continuation584child 100
)"
continuation_584_epoch120_checkpoint="$(
    insert_model 920080 continuation584child 120
)"
continuation_584_final="$(
    insert_model \
        920080 \
        continuation584child \
        120 \
        "resumed trained parameters"
)"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE model
SET parent_model_id=${source_584_model}
WHERE model_id IN (
    ${continuation_584_epoch100},
    ${continuation_584_epoch120_checkpoint},
    ${continuation_584_final}
);
SQL

continuation_584_attempt="$(
    insert_attempt 920080 orphan-attempt-920080 true
)"

# ----------------------------------------------------------------------
# 920081: intermediate continuation orphan.
#
# Scientific source remains immutable in resume_model_id.
# Operational epoch-100 checkpoint must be persisted only as last_model_id.
# ----------------------------------------------------------------------

source_intermediate_fixture="$(
    create_continuation_source \
        920061 \
        continuationrecoverysource \
        60 \
        120
)"
source_intermediate_model="${source_intermediate_fixture%%:*}"
source_intermediate_decision="${source_intermediate_fixture##*:}"

insert_experiment 920081 continuationrecoverychild 120
bind_continuation_child \
    920081 \
    920061 \
    "${source_intermediate_model}" \
    60 \
    "${source_intermediate_decision}"

continuation_intermediate_checkpoint="$(
    insert_model 920081 continuationrecoverychild 100
)"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE model
SET parent_model_id=${source_intermediate_model}
WHERE model_id=${continuation_intermediate_checkpoint};
SQL

continuation_intermediate_attempt="$(
    insert_attempt 920081 orphan-attempt-920081 true
)"

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
grep -q 'SCHEDULER_ORPHAN_RECOVERY_DONE,recovered_or_failed=15' "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920070,model_id=${newest_model},completed_epoch=40" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_CANDIDATE_REJECTED,experiment_id=920071,model_id=${invalid_newest_model}" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920071,model_id=${fallback_model},completed_epoch=40" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920072,model_id=${third_model},completed_epoch=20" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTION_FAILED,experiment_id=920073.*reason=no_valid_checkpoint" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_CANDIDATE_TIE_BREAK,experiment_id=920074,completed_epoch=40,candidate_count=2,candidate_order=periodic_flag_asc_model_id_desc,selected_model_id=${ambiguous_model_two}" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920074,model_id=${ambiguous_model_two},completed_epoch=40" "${recovery_output}"
grep -q "SCHEDULER_ORPHAN_RECOVERED,experiment_id=920075,restart_model_id=${intermediate_model}" "${recovery_output}"
grep -q "SCHEDULER_ORPHAN_ADVANCED,experiment_id=920076,model_id=${final_model}" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920082,model_id=${current_attempt_model},completed_epoch=40" "${recovery_output}"
! grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920082,model_id=${stale_attempt_model}" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTION_FAILED,experiment_id=920083,candidate_count=0,reason=no_valid_checkpoint" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920084,model_id=${experiment_fence_model},completed_epoch=40" "${recovery_output}"
! grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920084,model_id=${wrong_experiment_model}" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_CANDIDATE_REJECTED,experiment_id=920086,model_id=${invalid_final_model},completed_epoch=80" "${recovery_output}"
grep -q "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920086,model_id=${invalid_final_fallback_model},completed_epoch=40" "${recovery_output}"

grep -q \
    "SCHEDULER_CHECKPOINT_FINAL_CANDIDATE_RESOLVED,experiment_id=920080,completed_epoch=120,candidate_count=2,model_id=${continuation_584_final},reason=unique_non_periodic_target_model" \
    "${recovery_output}"

grep -q \
    "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920080,model_id=${continuation_584_final},completed_epoch=120,result=final_model" \
    "${recovery_output}"

grep -q \
    "SCHEDULER_ORPHAN_ADVANCED,experiment_id=920080,model_id=${continuation_584_final},completed_epochs=120,target_epochs=120,next_phase=infer" \
    "${recovery_output}"

! grep -q \
    "SCHEDULER_CHECKPOINT_SELECTED,experiment_id=920080,model_id=${continuation_584_epoch100}" \
    "${recovery_output}"

grep -q \
    "SCHEDULER_ORPHAN_RECOVERED,experiment_id=920081,restart_model_id=${continuation_intermediate_checkpoint},completed_epochs=100,target_epochs=120" \
    "${recovery_output}"

test "$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=920070")" = "${newest_model}"
test "$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=920071")" = "${fallback_model}"
test "$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=920072")" = "${third_model}"
test "$(scalar "SELECT status||':'||phase FROM experiment WHERE experiment_id=920073")" = "failed:train"
test "$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=920074")" = "${ambiguous_model_two}"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text||':'||resume_model_id::text||':'||current_epoch::text||':'||stop_after_checkpoint_epoch::text||':'||current_operation FROM experiment WHERE experiment_id=920075")" = "pending:train:${intermediate_model}:${intermediate_model}:57:60:train"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result||':'||(completed_at IS NOT NULL)::text FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${intermediate_attempt}")" = "completed:process_missing_result_recovered:true"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text||':'||(resume_model_id IS NULL)::text||':'||current_operation FROM experiment WHERE experiment_id=920076")" = "pending:infer:${final_model}:true:train"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${final_attempt}")" = "completed:process_missing_result_recovered"

# 584-equivalent target completion:
# scientific continuation identity is unchanged and the final model advances.
test "$(
    scalar "
        SELECT
            status||':'||
            phase||':'||
            last_model_id::text||':'||
            resume_model_id::text||':'||
            continuation_source_model_id::text||':'||
            continuation_source_epoch::text
        FROM experiment
        WHERE experiment_id=920080
    "
)" = "pending:infer:${continuation_584_final}:${source_584_model}:${source_584_model}:60"

test "$(
    scalar "
        SELECT
            lifecycle_state||':'||
            reconciliation_result||':'||
            (completed_at IS NOT NULL)::text
        FROM experiment_scheduler_worker_attempt
        WHERE worker_attempt_id=${continuation_584_attempt}
    "
)" = "completed:process_missing_result_recovered:true"

# Intermediate continuation recovery:
# resume_model_id is the immutable scientific source;
# last_model_id is the newer operational restart checkpoint.
test "$(
    scalar "
        SELECT
            status||':'||
            phase||':'||
            last_model_id::text||':'||
            resume_model_id::text||':'||
            continuation_source_model_id::text||':'||
            continuation_source_epoch::text
        FROM experiment
        WHERE experiment_id=920081
    "
)" = "pending:train:${continuation_intermediate_checkpoint}:${source_intermediate_model}:${source_intermediate_model}:60"

test "$(
    scalar "
        SELECT
            lifecycle_state||':'||
            reconciliation_result||':'||
            (completed_at IS NOT NULL)::text
        FROM experiment_scheduler_worker_attempt
        WHERE worker_attempt_id=${continuation_intermediate_attempt}
    "
)" = "completed:process_missing_result_recovered:true"

# Explicitly prove PostgreSQL still considers both continuation rows valid
# under the production constraint.
test "$(
    scalar "
        SELECT count(*)
        FROM experiment
        WHERE experiment_id IN (920080,920081)
          AND resume_model_id=continuation_source_model_id
          AND parent_experiment_id=continuation_source_experiment_id
          AND target_epochs>continuation_source_epoch
    "
)" = "2"

test "$(scalar "SELECT status||':'||phase||':'||(last_model_id IS NULL)::text||':'||(resume_model_id IS NULL)::text FROM experiment WHERE experiment_id=920077")" = "failed:train:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${mismatch_attempt}")" = "failed:process_missing_no_result"
test "$(scalar "SELECT status||':'||phase||':'||(last_model_id IS NULL)::text||':'||(resume_model_id IS NULL)::text FROM experiment WHERE experiment_id=920078")" = "running:train:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${unbound_attempt}")" = "failed:lifecycle_predicate_changed"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text FROM experiment WHERE experiment_id=920082")" = "pending:train:${current_attempt_model}"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${attempt_fence_attempt}")" = "completed:process_missing_result_recovered"
test "$(scalar "SELECT status||':'||phase||':'||(last_model_id IS NULL)::text FROM experiment WHERE experiment_id=920083")" = "failed:train:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${boundary_attempt}")" = "failed:process_missing_no_result"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text FROM experiment WHERE experiment_id=920084")" = "pending:train:${experiment_fence_model}"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${experiment_fence_attempt}")" = "completed:process_missing_result_recovered"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id::text FROM experiment WHERE experiment_id=920086")" = "pending:train:${invalid_final_fallback_model}"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${invalid_final_attempt}")" = "completed:process_missing_result_recovered"

# Rejected candidates never become durable restart state.
test "$(scalar "SELECT count(*) FROM experiment WHERE last_model_id IN (${invalid_newest_model},${unusable_model},${stale_attempt_model},${pre_attempt_model},${wrong_experiment_model},${invalid_final_model}) OR resume_model_id IN (${invalid_newest_model},${unusable_model},${stale_attempt_model},${pre_attempt_model},${wrong_experiment_model},${invalid_final_model})")" = "0"

snapshot="$(scalar "SELECT string_agg(worker_attempt_id::text||':'||lifecycle_state||':'||COALESCE(reconciliation_result,'NULL'),',' ORDER BY worker_attempt_id) FROM experiment_scheduler_worker_attempt")"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    >"${test_dir}/recovery-replay.out" 2>&1
test "${snapshot}" = "$(scalar "SELECT string_agg(worker_attempt_id::text||':'||lifecycle_state||':'||COALESCE(reconciliation_result,'NULL'),',' ORDER BY worker_attempt_id) FROM experiment_scheduler_worker_attempt")"
test "${attempts_before}" = "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt")"

printf '%s\n' "SchedulerTrainOrphanCheckpointRecoveryTests passed"
