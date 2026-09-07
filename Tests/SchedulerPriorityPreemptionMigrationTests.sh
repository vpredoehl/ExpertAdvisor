#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_db="ea_scheduler_preemption_migration_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-scheduler-preemption-migration.XXXXXX")"
export PGOPTIONS=

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

scalar() { psql -X -Atq -d "${test_db}" -c "$1"; }

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,status,phase,current_operation,duplicate_nonce,
    scheduler_priority,resume_requested,updated_at,model_input_width,
    model_input_semantic_layout_version
) VALUES
    (993001,'legacyfalse',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993001,'normal',false,'2026-01-01',75,5),
    (993002,'legacytrue',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993002,'normal',true,'2026-01-02',75,5);
SQL

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/093_scheduler_priority_preemption.sql"

test "$(scalar "SELECT string_agg(experiment_id||':'||resume_requested::text||':'||scheduler_resume_origin,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id IN (993001,993002)")" = \
    '993001:false:none,993002:true:operator'
test "$(scalar "SELECT is_nullable||':'||column_default FROM information_schema.columns WHERE table_schema='public' AND table_name='experiment' AND column_name='scheduler_resume_origin'")" = \
    "NO:'none'::text"
test "$(scalar "SELECT count(*) FROM pg_constraint WHERE conrelid='experiment'::regclass AND conname IN ('experiment_scheduler_resume_origin_check','experiment_scheduler_resume_state_check') AND contype='c'")" = 2

set +e
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET scheduler_resume_origin='invalid' WHERE experiment_id=993001" \
    >"${test_dir}/invalid-origin.out" 2>&1
invalid_origin_result=$?
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET resume_requested=false,scheduler_resume_origin='operator' WHERE experiment_id=993001" \
    >"${test_dir}/invalid-state.out" 2>&1
invalid_state_result=$?
set -e
test "${invalid_origin_result}" -ne 0
test "${invalid_state_result}" -ne 0

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE experiment
SET scheduler_priority='low',resume_requested=true,
    scheduler_resume_origin='preemption',updated_at='2026-02-01'
WHERE experiment_id=993001;
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,status,phase,current_operation,duplicate_nonce,
    scheduler_priority,resume_requested,scheduler_resume_origin,updated_at,
    model_input_width,model_input_semantic_layout_version
) VALUES
    (993003,'highordinary',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993003,'high',false,'none','2026-02-03',75,5),
    (993004,'normalordinary',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993004,'normal',false,'none','2026-02-04',75,5),
    (993005,'normaloperator',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993005,'normal',true,'operator','2026-02-05',75,5),
    (993006,'normalpreempted',1,0.0008,1,1,2,1,'2020-01-01','2020-02-01',
     'pending','train','train',993006,'normal',true,'preemption','2026-02-06',75,5);
SQL

admission_order="$(scalar "SELECT string_agg(experiment_id::text,',' ORDER BY CASE scheduler_priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,CASE scheduler_resume_origin WHEN 'operator' THEN 0 WHEN 'preemption' THEN 1 ELSE 2 END,updated_at,experiment_id) FROM experiment WHERE experiment_id IN (993001,993003,993004,993005,993006)")"
test "${admission_order}" = '993003,993005,993006,993004,993001'
index_definition="$(scalar "SELECT pg_get_indexdef(indexrelid) FROM pg_index WHERE indexrelid='experiment_scheduler_pending_priority_idx'::regclass")"
[[ "${index_definition}" == *"scheduler_resume_origin"* ]]
[[ "${index_definition}" != *"resume_requested DESC"* ]]

# Replay must preserve a durable scheduler-created preemption origin.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/093_scheduler_priority_preemption.sql" \
    >"${test_dir}/migration-replay.out"
test "$(scalar "SELECT resume_requested::text||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=993001")" = \
    'true:preemption'
test "$(scalar "SELECT count(*) FROM experiment WHERE resume_requested <> (scheduler_resume_origin <> 'none')")" = 0

printf '%s\n' \
    'SchedulerPriorityPreemptionMigrationTests passed' \
    'legacy_mapping=false:none,true:operator' \
    'replay_preserves_preemption=PASS' \
    'priority_then_origin_order=PASS' \
    'production_database_mutations=0'
