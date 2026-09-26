#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
test_db="ea_input_width_expansion_test_${$}"
test_dir="$(mktemp -d /tmp/ea_input_width_expansion_scheduler.XXXXXX)"
expected_source_commit="$(git -C "${repo_root}" rev-parse --verify HEAD)"

test -x "${scheduler_binary}"
if ! strings "${scheduler_binary}" | grep -Fx -- "${expected_source_commit}" \
    >/dev/null; then
    printf '%s\n' \
        "scheduler binary does not embed current source commit: ${expected_source_commit}" \
        >&2
    exit 2
fi

case "${test_db}" in
    ea_input_width_expansion_test_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
# The production database is a read-only schema source.  This test exercises
# the current scheduler fixture, not historical migration replay: applying
# migrations to this already-current clone can replay incompatible DDL.
PGOPTIONS='-c default_transaction_read_only=on' \
    pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
# The production schema source predates the current queue binary's additive
# fresh-initialization-seed column and its identity index. Apply only their
# idempotent owner migrations to the disposable clone; do not replay unrelated
# historical DDL.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/094_fresh_initialization_seed.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/095_fresh_initialization_seed_identity.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/InputWidthExpansionMigrationTests.sql"
# LSTMModelInputIdentityMigrationTests.sql asserts migration 089's exact
# pre-092 index shape.  Its dedicated harness reconstructs that predecessor
# schema before applying 089; it is intentionally not a current-schema
# scheduler-fixture assertion.

# This historical layout-8/width-80 fixture proves that an ordinary
# source-derived child retains the persisted source layout while an explicit
# input-width expansion deliberately transitions to the current layout.
source_experiment_id="$(psql -X -At -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(
    symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,
    target_epochs,checkpoint_interval,train_start,train_end,infer_start,
    infer_end,status,phase,duplicate_nonce,donchian20_mode,
    feature_warmup_scope,donchian_lookback,feature_ablation_mask,
    model_input_width,model_input_semantic_layout_version
) VALUES(
    'expansionfixture',4,0.0008,1.0,1.0,40,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'completed','done',900001,'enabled','legacy_cold_boundary',20,'',80,8
) RETURNING experiment_id;
SQL
)"

source_model_id="$(psql -X -At -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v source_experiment_id="${source_experiment_id}" <<'SQL'
INSERT INTO model(name,comment,experiment_id)
VALUES('width-51-source','isolated expansion fixture',:source_experiment_id)
RETURNING model_id;
SQL
)"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v source_model_id="${source_model_id}" <<'SQL'
WITH v AS (SELECT ARRAY[1.0,4.0,0.0008,20.0,1.0,1.0,1.0,1.0,
                            1.0,1.0,40.0,1.0,1.0,1.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'train_config_meta',1,14,0,i-1,a[i]
FROM v,generate_series(1,14)i;
WITH v AS (SELECT 'expansionfixture'::text a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'train_symbol_meta',1,length(a),0,i-1,
       ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT '2020-01-01|2021-01-01'::text a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'train_range_meta',1,length(a),0,i-1,
       ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT 'enabled'::text a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'donchian20_mode_meta',1,length(a),0,i-1,
       ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT 'legacy_cold_boundary'::text a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'feature_warmup_scope_meta',1,length(a),0,i-1,
       ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT '20'::text a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'donchian_lookback_meta',1,length(a),0,i-1,
       ascii(substr(a,i,1)) FROM v,generate_series(1,length(a))i;
WITH v AS (SELECT ARRAY[1.0,80.0,1.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'model_meta',1,3,0,i-1,a[i]
FROM v,generate_series(1,3)i;
WITH v AS (SELECT ARRAY[1.0,8.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'model_input_semantics_meta',1,2,0,i-1,a[i]
FROM v,generate_series(1,2)i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'param',81,4,(i-1)/4,(i-1)%4,i::double precision/1024.0
FROM generate_series(1,324)i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'bias',1,4,0,i-1,0.0 FROM generate_series(1,4)i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
VALUES(:source_model_id,'returnHeadWeight',1,1,0,0,0.25),
      (:source_model_id,'returnHeadBias',1,1,0,0,-0.5);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'returnHeadDirWeight',1,3,0,i-1,0.1*i
FROM generate_series(1,3)i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'returnHeadDirBias',1,3,0,i-1,-0.1*i
FROM generate_series(1,3)i;
WITH v AS (SELECT ARRAY[2.0,1.0,0.0,0.0,0.0,1.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'target_meta',1,6,0,i-1,a[i]
FROM v,generate_series(1,6)i;
WITH v AS (SELECT ARRAY[1.0,1.0,17.0,0.0,0.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'optimizer_meta',1,5,0,i-1,a[i]
FROM v,generate_series(1,5)i;
SQL

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --target-epochs=80 >"${test_dir}/ordinary.out" 2>&1
grep -q 'resume_expand_input_width=0' "${test_dir}/ordinary.out"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --resume-expand-input-width --target-epochs=80 \
    >"${test_dir}/expanded.out" 2>&1
grep -q 'resume_expand_input_width=1' "${test_dir}/expanded.out"
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = 2
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(resume_expand_input_width::text,',' ORDER BY resume_expand_input_width) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = 'false,true'
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(model_input_width::text,',' ORDER BY resume_expand_input_width) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = '80,103'
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(model_input_semantic_layout_version::text,',' ORDER BY resume_expand_input_width) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = '8,9'

# A caller-supplied fresh seed is invalid for a resume even though the
# SchedulerOptions default remains present for schema compatibility.
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --target-epochs=82 --fresh-initialization-seed=43 \
    >"${test_dir}/queue-resume-seed.out" 2>&1
queue_resume_seed_status=$?
set -e
test "${queue_resume_seed_status}" -ne 0
grep -q -- '--fresh-initialization-seed is not valid with --resume-model-id' \
    "${test_dir}/queue-resume-seed.out"

# A stored resumed-row seed is non-operative. Generic queue equivalence must
# still find this child after its compatibility value differs from the default.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v source_model_id="${source_model_id}" <<'SQL'
UPDATE experiment SET fresh_initialization_seed=43
WHERE resume_model_id=:source_model_id
  AND target_epochs=80
  AND resume_expand_input_width=false;
SQL
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --target-epochs=80 >"${test_dir}/queue-resume-equivalent.out" 2>&1
queue_resume_equivalent_status=$?
set -e
test "${queue_resume_equivalent_status}" -eq 3
grep -q 'QUEUE_ALREADY_EXISTS' "${test_dir}/queue-resume-equivalent.out"
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80 AND NOT resume_expand_input_width")" = 1

# --enqueue-experiment has independently parsed required arguments, but a
# resume must still persist the source model's semantic identity rather than
# the runtime's current layout.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --enqueue-experiment --resume-model-id="${source_model_id}" \
    --symbol=expansionfixture --prediction-horizon=4 \
    --c-next-threshold=0.0008 --target-epochs=83 \
    --train-start=2020-01-01 --train-end=2021-01-01 \
    >"${test_dir}/enqueue.out" 2>&1
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT model_input_width::text || ':' || model_input_semantic_layout_version::text FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=83")" = '80:8'

set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --enqueue-experiment --resume-model-id="${source_model_id}" \
    --symbol=expansionfixture --prediction-horizon=4 \
    --c-next-threshold=0.0008 --target-epochs=84 \
    --train-start=2020-01-01 --train-end=2021-01-01 \
    --fresh-initialization-seed=43 \
    >"${test_dir}/enqueue-resume-seed.out" 2>&1
enqueue_resume_seed_status=$?
set -e
test "${enqueue_resume_seed_status}" -ne 0
grep -q -- '--fresh-initialization-seed is not valid with --resume-model-id' \
    "${test_dir}/enqueue-resume-seed.out"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v source_model_id="${source_model_id}" <<'SQL'
UPDATE experiment SET fresh_initialization_seed=43
WHERE resume_model_id=:source_model_id
  AND target_epochs=83;
SQL
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --enqueue-experiment --resume-model-id="${source_model_id}" \
    --symbol=expansionfixture --prediction-horizon=4 \
    --c-next-threshold=0.0008 --target-epochs=83 \
    --train-start=2020-01-01 --train-end=2021-01-01 \
    >"${test_dir}/enqueue-resume-equivalent.out" 2>&1
enqueue_resume_equivalent_status=$?
set -e
test "${enqueue_resume_equivalent_status}" -ne 0
grep -q 'SCHEDULER_DUPLICATE_REJECTED' \
    "${test_dir}/enqueue-resume-equivalent.out"
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=83")" = 1

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
    --target-epochs=1 --fresh-initialization-seed=43 \
    >"${test_dir}/fresh-queue.out" 2>&1
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
    --target-epochs=1 --fresh-initialization-seed=44 \
    >"${test_dir}/fresh-queue-second-seed.out" 2>&1
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(fresh_initialization_seed::text,',' ORDER BY fresh_initialization_seed) FROM experiment WHERE symbol='eurusdrmp' AND target_epochs=1")" = '43,44'
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(model_input_width::text || ':' || model_input_semantic_layout_version::text,',' ORDER BY fresh_initialization_seed) FROM experiment WHERE symbol='eurusdrmp' AND target_epochs=1")" = '103:9,103:9'

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --enqueue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
    --c-next-threshold=0.0008 --target-epochs=2 \
    --train-start=2010-01-01 --train-end=2025-01-01 \
    --fresh-initialization-seed=45 >"${test_dir}/fresh-enqueue.out" 2>&1
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT fresh_initialization_seed FROM experiment WHERE symbol='eurusdrmp' AND target_epochs=2")" = 45

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --resume-expand-input-width --target-epochs=81 --dry-run \
    --ablate-features=historical_level_proximity \
    >"${test_dir}/ablation.out" 2>&1
grep -q 'feature_ablation_mask=historical_level_proximity' \
    "${test_dir}/ablation.out"

# A semantic marker is optional only for models that predate the marker. Once
# present, incompatible semantics fail closed for ordinary resume and explicit
# expansion alike.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v source_model_id="${source_model_id}" <<'SQL'
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
VALUES(:source_model_id,'model_input_semantics_meta',1,2,0,0,1.0),
      (:source_model_id,'model_input_semantics_meta',1,2,0,1,999.0);
SQL
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --target-epochs=82 --dry-run >"${test_dir}/ordinary-semantic.out" 2>&1
ordinary_incompatible_status=$?
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --resume-model-id="${source_model_id}" \
    --resume-expand-input-width --target-epochs=82 --dry-run \
    >"${test_dir}/incompatible-semantic.out" 2>&1
incompatible_status=$?
set -e
test "${ordinary_incompatible_status}" -ne 0
test "${incompatible_status}" -ne 0
grep -q 'MODEL_INPUT_EXPANSION_SEMANTIC_METADATA_INCOMPATIBLE' \
    "${test_dir}/ordinary-semantic.out"
grep -q 'MODEL_INPUT_EXPANSION_SEMANTIC_METADATA_INCOMPATIBLE' \
    "${test_dir}/incompatible-semantic.out"

printf '%s\n' 'LSTMInputWidthExpansionSchedulerIntegrationTests passed'
