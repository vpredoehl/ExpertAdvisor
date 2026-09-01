#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
test_db="ea_input_width_expansion_test_${$}"
test_dir="$(mktemp -d /tmp/ea_input_width_expansion_scheduler.XXXXXX)"

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
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/078_operator_forced_final_inference_rerun.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/089_lstm_model_input_identity.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/InputWidthExpansionMigrationTests.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/LSTMModelInputIdentityMigrationTests.sql"

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
    'completed','done',900001,'enabled','legacy_cold_boundary',20,'',51,5
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
WITH v AS (SELECT ARRAY[1.0,51.0,1.0] a)
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'model_meta',1,3,0,i-1,a[i]
FROM v,generate_series(1,3)i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT :source_model_id,'param',52,4,(i-1)/4,(i-1)%4,i::double precision/1024.0
FROM generate_series(1,208)i;
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
    "SELECT string_agg(model_input_width::text,',' ORDER BY resume_expand_input_width) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = '51,75'
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT string_agg(model_input_semantic_layout_version::text,',' ORDER BY resume_expand_input_width) FROM experiment WHERE resume_model_id=${source_model_id} AND target_epochs=80")" = '5,5'

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --queue-experiment --symbol=eurusdrmp --prediction-horizon=4 \
    --target-epochs=1 >"${test_dir}/fresh.out" 2>&1
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT model_input_width::text || ':' || model_input_semantic_layout_version::text FROM experiment WHERE symbol='eurusdrmp' AND target_epochs=1 ORDER BY experiment_id DESC LIMIT 1")" = '75:5'

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
