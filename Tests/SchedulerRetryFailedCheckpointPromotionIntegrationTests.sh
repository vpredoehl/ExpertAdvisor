#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
test_db="ea_scheduler_retry_checkpoint_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_retry_checkpoint_test.XXXXXX)"
cleanup() { local status=$?; dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true; rm -rf "${test_dir}"; exit "${status}"; }
trap cleanup EXIT
case "${test_db}" in ea_scheduler_retry_checkpoint_test_[0-9]*) ;; *) exit 90 ;; esac

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM | psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state) VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
UPDATE experiment_scheduler_protocol SET cutover_state='complete',cutover_completed_at=clock_timestamp(),cutover_completed_by='retry-checkpoint-test',cutover_executable_path='/isolated/LSTM_Release',cutover_process_evidence='disposable database',updated_at=clock_timestamp() WHERE singleton;
INSERT INTO experiment_admin_request(action,invocation_identity,status,previous_global_state,resulting_global_state,completed_at) VALUES('pause_all','retry-checkpoint-test','completed','running','paused',clock_timestamp());
SQL

add_experiment() {
    local id="$1" phase="$2" resume="$3" mode="$4"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,infer_start,infer_end,status,phase,resume_model_id,current_epoch,current_operation,worker_control_state,started_at,completed_at,exit_code,error_message,duplicate_nonce,donchian20_mode) VALUES(${id},'cadchfrmp',4,0.0008,120,25,80,20,'2020-01-01','2021-01-01','2021-01-01','2022-01-01','failed','${phase}',${resume},79,'${phase}','running',clock_timestamp(),clock_timestamp(),17,'fixture failure',${id},'${mode}')"
}

add_model() {
    local model_id="$1" experiment_id="$2" epoch="$3" mode="$4" symbol="${5:-cadchfrmp}" horizon="${6:-4}" threshold="${7:-0.0008}" range="${8:-2020-01-01|2021-01-01}" core="${9:-120}" head="${10:-25}" parent="${11:-NULL}"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO model(model_id,experiment_id,parent_model_id,name,comment) VALUES(${model_id},${experiment_id},${parent},'retry-${model_id}','periodic training checkpoint');
WITH v AS (SELECT ARRAY[1.0,${horizon}::double precision,${threshold}::double precision,0.0,0.0,0.0,0.0,0.0,0.0,0.0,${epoch}::double precision,${core}::double precision,${head}::double precision,0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'train_config_meta',1,14,0,i-1,a[i] FROM v,generate_series(1,14)i;
WITH v AS (SELECT '${symbol}'::text a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'train_symbol_meta',1,length(a),0,i-1,ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT '${range}'::text a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'train_range_meta',1,length(a),0,i-1,ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
SQL
    if [[ "${mode}" != "legacy" ]]; then
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "WITH v AS (SELECT '${mode}'::text a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'donchian20_mode_meta',1,length(a),0,i-1,ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i"
    fi
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
WITH v AS (SELECT ARRAY[1.0,36.0,1.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'model_meta',1,3,0,i-1,a[i] FROM v,generate_series(1,3)i;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'param',37,4,(i-1)/4,(i-1)%4,a[1] FROM v,generate_series(1,148)i;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'bias',1,4,0,i-1,a[1] FROM v,generate_series(1,4)i;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'returnHeadWeight',1,1,0,0,a[1] FROM v;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'returnHeadBias',1,1,0,0,a[1] FROM v;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'returnHeadDirWeight',1,3,0,(i-1)%3,a[1] FROM v,generate_series(1,3)i;
WITH v AS (SELECT ARRAY[0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'returnHeadDirBias',1,3,0,(i-1)%3,a[1] FROM v,generate_series(1,3)i;
WITH v AS (SELECT ARRAY[2.0,1.0,0.0,0.0,0.0,1.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'target_meta',1,6,0,i-1,a[i] FROM v,generate_series(1,6)i;
WITH v AS (SELECT ARRAY[1.0,1.0,${epoch}::double precision,0.0,0.0] a) INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value) SELECT ${model_id},'optimizer_meta',1,5,0,i-1,a[i] FROM v,generate_series(1,5)i;
SQL
}

add_experiment 960554 train 961001 enabled
add_experiment 960549 train NULL enabled
add_experiment 960546 train 961062 enabled
add_experiment 960560 train NULL enabled
add_experiment 960570 train 961070 enabled
add_experiment 960580 train NULL enabled
add_experiment 960581 train NULL zero_ablation
add_experiment 960590 train 961090 enabled
add_experiment 960600 infer 961001 enabled
add_experiment 960561 train 961001 enabled
add_experiment 960555 train NULL enabled
add_experiment 960610 train NULL enabled
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,status,phase,duplicate_nonce) SELECT id,'cadchfrmp',4,0.0008,120,25,80,20,'2020-01-01','2021-01-01','completed','done',id FROM unnest(ARRAY[960551::bigint,960562,960563,960569]) AS id"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "INSERT INTO experiment(experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,status,phase,worker_pid,worker_process_group_id,worker_process_start_identity,worker_executable,worker_command_line,current_operation,worker_control_state,duplicate_nonce) VALUES(960699,'unrelatedrmp',4,0.0008,120,25,80,20,'2020-01-01','2021-01-01','cancelled','done',960699,960699,'unrelated','/unrelated/LSTM_Release','unrelated command','train','running',960699)"

# Sources and durable same-experiment checkpoints.  961001 deliberately has
# no Donchian metadata, matching the legacy enabled source behavior.
add_model 961001 960551 40 legacy
add_model 961005 960554 60 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 120 25 961001
add_model 961020 960549 20 enabled; add_model 961030 960549 40 enabled; add_model 961040 960549 60 enabled
add_model 961062 960562 40 enabled; add_model 961063 960546 20 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 120 25 961062
add_model 961560 960560 40 enabled; add_model 961561 960560 60 enabled; add_model 961562 960560 60 enabled
add_model 961070 960569 40 enabled
add_model 961571 960570 60 enabled wrongrmp
add_model 961572 960570 60 enabled cadchfrmp 8
add_model 961573 960570 60 enabled cadchfrmp 4 0.0009
add_model 961574 960570 60 enabled cadchfrmp 4 0.0008 '2021-01-01|2022-01-01'
add_model 961575 960570 60 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 121
add_model 961576 960570 60 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 120 24
add_model 961577 960570 60 zero_ablation
add_model 961580 960580 60 legacy; add_model 961581 960581 60 legacy
add_model 961090 960563 40 enabled; add_model 961590 960590 80 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 120 25 961090
add_model 961591 960561 60 enabled cadchfrmp 4 0.0008 '2020-01-01|2021-01-01' 120 25 961001
add_model 961510 960555 60 enabled
add_model 961511 960555 70 enabled
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "DELETE FROM matrix WHERE model_id=961511 AND param_name='optimizer_meta'"
# 961510 is complete.  Every newer candidate is metadata-compatible but
# missing a tensor required by the actual UpNeutralDownReturn resume path.
add_model 961512 960555 80 enabled
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "DELETE FROM matrix WHERE model_id=961512 AND param_name='returnHeadDirWeight'"
add_model 961513 960555 90 enabled
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c "DELETE FROM matrix WHERE model_id=961513 AND param_name='returnHeadDirBias'"
add_model 961610 960610 60 enabled
# Regression targets do not consume the directional head, so their valid
# legacy resume state remains accepted without those optional tensors.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE matrix SET value=0.0
WHERE model_id=961610 AND param_name='target_meta' AND row_idx=0 AND col_idx=0;
DELETE FROM matrix WHERE model_id=961610
AND param_name IN ('returnHeadDirWeight','returnHeadDirBias');
SQL
add_model 961600 960600 60 enabled
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_scheduler_worker_attempt(worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,worker_pid,worker_process_group_id,worker_process_start_identity,canonical_executable_path,command_line,command_identity,reserved_at,completed_at,exit_code,diagnostic) VALUES(961554,'historical:960554',960554,'experiment','train','train','legacy_unverified','failed',960554,960554,'old-start','/old/LSTM_Release','old command','experiment:960554:train',clock_timestamp(),clock_timestamp(),17,'immutable');
SQL

attempt_before="$(psql -X -At -d "${test_db}" -c "SELECT md5(string_agg(row_to_json(a)::text,'' ORDER BY worker_attempt_id)) FROM experiment_scheduler_worker_attempt a")"
unrelated_before="$(psql -X -At -d "${test_db}" -c "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=960699")"
dry_before="$(psql -X -At -d "${test_db}" -c "SELECT status||':'||phase||':'||COALESCE(resume_model_id::text,'NULL') FROM experiment WHERE experiment_id=960561")"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --retry-failed-experiment=960561 --dry-run >"${test_dir}/dry.out" 2>&1
grep -q 'SCHEDULER_RETRY_CHECKPOINT_SELECTION,experiment_id=960561,previous_resume_model_id=961001,selected_resume_model_id=961591,selected_completed_epoch=60,promotion=1,reason=newer_compatible_checkpoint' "${test_dir}/dry.out"
test "${dry_before}" = "$(psql -X -At -d "${test_db}" -c "SELECT status||':'||phase||':'||COALESCE(resume_model_id::text,'NULL') FROM experiment WHERE experiment_id=960561")"

for id in 960554 960549 960546 960560 960570 960580 960581 960590 960600 960555 960610; do
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --retry-failed-experiment="${id}" --yes >"${test_dir}/${id}.out" 2>&1
done
value() { psql -X -At -d "${test_db}" -c "$1"; }
test "$(value "SELECT status||':'||phase||':'||resume_model_id FROM experiment WHERE experiment_id=960554")" = 'pending:train:961005'
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960549")" = 961040
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960546")" = 961062
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960560")" = 961562
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960570")" = 961070
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960580")" = 961580
test "$(value "SELECT COALESCE(resume_model_id::text,'NULL') FROM experiment WHERE experiment_id=960581")" = NULL
test "$(value "SELECT resume_model_id FROM experiment WHERE experiment_id=960590")" = 961090
test "$(value "SELECT status||':'||phase||':'||resume_model_id FROM experiment WHERE experiment_id=960555")" = 'pending:train:961510'
test "$(value "SELECT status||':'||phase||':'||resume_model_id FROM experiment WHERE experiment_id=960610")" = 'pending:train:961610'
test "$(value "SELECT status||':'||phase||':'||resume_model_id FROM experiment WHERE experiment_id=960600")" = 'pending:infer:961001'
test "$(value "SELECT count(*) FROM experiment WHERE experiment_id=960554 AND worker_pid IS NULL AND worker_process_group_id IS NULL AND worker_process_start_identity IS NULL AND worker_executable IS NULL AND worker_command_line IS NULL AND active_scheduler_worker_attempt_id IS NULL AND worker_control_state='running' AND started_at IS NULL AND completed_at IS NULL AND exit_code IS NULL AND error_message IS NULL AND current_operation IS NULL")" = 1
grep -q 'reason=no_compatible_checkpoint' "${test_dir}/960570.out"
grep -q 'reason=no_compatible_checkpoint' "${test_dir}/960590.out"
grep -q 'selected_resume_model_id=961510,selected_completed_epoch=60,promotion=1,reason=newer_compatible_checkpoint' "${test_dir}/960555.out"
! grep -q SCHEDULER_RETRY_CHECKPOINT_SELECTION "${test_dir}/960600.out"
test "${attempt_before}" = "$(psql -X -At -d "${test_db}" -c "SELECT md5(string_agg(row_to_json(a)::text,'' ORDER BY worker_attempt_id)) FROM experiment_scheduler_worker_attempt a")"
test "${unrelated_before}" = "$(psql -X -At -d "${test_db}" -c "SELECT row_to_json(e)::text FROM experiment e WHERE experiment_id=960699")"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --schedule-experiments --scheduler-once --dry-run --max-train-procs=20 --max-infer-procs=1 --max-analyze-procs=1 --scheduler-log-dir="${test_dir}/logs" >"${test_dir}/dispatch.out" 2>&1
grep -E -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=960554,phase=train,dry_run=1,argv=.*--donchian20-mode=enabled.*--resume-model-id=961005.*--target-epochs=80' "${test_dir}/dispatch.out"
printf '%s\n' 'SchedulerRetryFailedCheckpointPromotionIntegrationTests passed'
