#!/usr/bin/env bash
set -euo pipefail

MODE="preview"
if [[ "${1:-}" == "--apply" ]]; then
    MODE="apply"
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--apply]" >&2
    exit 2
fi

OLD_SYMBOL="usdjpyrmp"
NEW_SYMBOL="usdpjyrmp"

FOREX_DB_HOST="${FOREX_DB_HOST:-${LSTM_DB_HOST:-127.0.0.1}}"
FOREX_DB_NAME="${FOREX_DB_NAME:-forex}"
FOREX_DB_ADMIN_USER="${FOREX_DB_ADMIN_USER:-${LSTM_DB_ADMIN_USER:-$(whoami)}}"

LSTM_DB_HOST="${LSTM_DB_HOST:-127.0.0.1}"
LSTM_DB_NAME="${LSTM_DB_NAME:-LSTM}"
LSTM_DB_ADMIN_USER="${LSTM_DB_ADMIN_USER:-$(whoami)}"

psql_forex=(psql -v ON_ERROR_STOP=1 -X -h "$FOREX_DB_HOST" -U "$FOREX_DB_ADMIN_USER" -d "$FOREX_DB_NAME")
psql_lstm=(psql -v ON_ERROR_STOP=1 -X -h "$LSTM_DB_HOST" -U "$LSTM_DB_ADMIN_USER" -d "$LSTM_DB_NAME")

run_preview() {
    echo "USDJPY_SYMBOL_REPAIR_PREVIEW,old=${OLD_SYMBOL},new=${NEW_SYMBOL}"

    "${psql_forex[@]}" <<SQL
\\pset pager off
\\echo 'PREVIEW_FOREX_TABLES'
SELECT c.oid::regclass AS relation,
       n.nspname AS schema,
       c.relname AS table_name,
       c.relkind,
       c.reltuples::bigint AS estimated_rows,
       pg_get_userbyid(c.relowner) AS owner
FROM pg_class c
JOIN pg_namespace n ON n.oid = c.relnamespace
WHERE n.nspname = 'public'
  AND c.relname IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY c.relname;
\\echo 'PREVIEW_FOREX_PARTITIONS'
SELECT parent.relname AS parent_table,
       child.relname AS child_table
FROM pg_inherits i
JOIN pg_class parent ON parent.oid = i.inhparent
JOIN pg_namespace pn ON pn.oid = parent.relnamespace
JOIN pg_class child ON child.oid = i.inhrelid
WHERE pn.nspname = 'public'
  AND parent.relname IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY parent.relname, child.relname;
\\echo 'PREVIEW_FOREX_DEPENDENT_VIEWS'
SELECT dependent_ns.nspname AS dependent_schema,
       dependent_view.relname AS dependent_object,
       dependent_view.relkind AS dependent_kind,
       source_table.relname AS source_table
FROM pg_depend d
JOIN pg_rewrite r ON r.oid = d.objid
JOIN pg_class dependent_view ON dependent_view.oid = r.ev_class
JOIN pg_namespace dependent_ns ON dependent_ns.oid = dependent_view.relnamespace
JOIN pg_class source_table ON source_table.oid = d.refobjid
JOIN pg_namespace source_ns ON source_ns.oid = source_table.relnamespace
WHERE source_ns.nspname = 'public'
  AND source_table.relname IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY dependent_schema, dependent_object;
\\echo 'PREVIEW_FOREX_INDEXES'
SELECT schemaname, tablename, indexname, indexdef
FROM pg_indexes
WHERE schemaname = 'public'
  AND tablename IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY tablename, indexname;
SQL

    "${psql_lstm[@]}" <<SQL
\\pset pager off
\\echo 'PREVIEW_LSTM_REFERENCE_COUNTS'
WITH decoded AS (
    SELECT model_id,
           string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS symbol
    FROM matrix
    WHERE param_name = 'train_symbol_meta'
      AND row_idx = 0
    GROUP BY model_id
)
SELECT 'model.name_bad' AS location, count(*) FROM model WHERE name ILIKE '%${OLD_SYMBOL}%'
UNION ALL SELECT 'model.name_good', count(*) FROM model WHERE name ILIKE '%${NEW_SYMBOL}%'
UNION ALL SELECT 'model.comment_bad', count(*) FROM model WHERE comment ILIKE '%${OLD_SYMBOL}%'
UNION ALL SELECT 'model.comment_good', count(*) FROM model WHERE comment ILIKE '%${NEW_SYMBOL}%'
UNION ALL SELECT 'matrix.train_symbol_meta_bad', count(*) FROM decoded WHERE symbol = '${OLD_SYMBOL}'
UNION ALL SELECT 'matrix.train_symbol_meta_good', count(*) FROM decoded WHERE symbol = '${NEW_SYMBOL}'
UNION ALL SELECT 'experiment.symbol_bad', count(*) FROM experiment WHERE symbol = '${OLD_SYMBOL}'
UNION ALL SELECT 'experiment.symbol_good', count(*) FROM experiment WHERE symbol = '${NEW_SYMBOL}'
UNION ALL SELECT 'inference_eval_result.symbol_bad', count(*) FROM inference_eval_result WHERE symbol = '${OLD_SYMBOL}'
UNION ALL SELECT 'inference_eval_result.symbol_good', count(*) FROM inference_eval_result WHERE symbol = '${NEW_SYMBOL}'
UNION ALL SELECT 'experiment_analysis_result.symbol_bad', count(*) FROM experiment_analysis_result WHERE symbol = '${OLD_SYMBOL}'
UNION ALL SELECT 'experiment_analysis_result.symbol_good', count(*) FROM experiment_analysis_result WHERE symbol = '${NEW_SYMBOL}'
ORDER BY location;
\\echo 'PREVIEW_LSTM_MODEL_ROWS'
SELECT model_id, name, comment
FROM model
WHERE name ILIKE '%${OLD_SYMBOL}%' OR name ILIKE '%${NEW_SYMBOL}%'
   OR comment ILIKE '%${OLD_SYMBOL}%' OR comment ILIKE '%${NEW_SYMBOL}%'
ORDER BY model_id;
\\echo 'PREVIEW_LSTM_TRAIN_SYMBOL_META'
WITH decoded AS (
    SELECT model_id,
           string_agg(chr(round(value)::int), '' ORDER BY col_idx) AS symbol,
           count(*) AS encoded_chars
    FROM matrix
    WHERE param_name = 'train_symbol_meta'
      AND row_idx = 0
    GROUP BY model_id
)
SELECT d.model_id, m.name, d.symbol, d.encoded_chars
FROM decoded d
LEFT JOIN model m USING (model_id)
WHERE d.symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY d.model_id;
\\echo 'PREVIEW_LSTM_MODEL_NAME_COLLISIONS'
WITH renamed AS (
    SELECT model_id,
           name AS old_name,
           replace(name, '${OLD_SYMBOL}', '${NEW_SYMBOL}') AS new_name
    FROM model
    WHERE name LIKE '%${OLD_SYMBOL}%'
)
SELECT r.model_id, r.old_name, r.new_name,
       existing.model_id AS colliding_model_id,
       existing.name AS colliding_name
FROM renamed r
JOIN model existing ON existing.name = r.new_name
                   AND existing.model_id <> r.model_id
ORDER BY r.model_id;
\\echo 'PREVIEW_LSTM_EXPERIMENT_ROWS'
SELECT experiment_id, symbol, last_model_id, resume_model_id,
       train_log_path, infer_log_path, analysis_log_path, status, phase
FROM experiment
WHERE symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
   OR train_log_path ILIKE '%${OLD_SYMBOL}%' OR train_log_path ILIKE '%${NEW_SYMBOL}%'
   OR infer_log_path ILIKE '%${OLD_SYMBOL}%' OR infer_log_path ILIKE '%${NEW_SYMBOL}%'
   OR analysis_log_path ILIKE '%${OLD_SYMBOL}%' OR analysis_log_path ILIKE '%${NEW_SYMBOL}%'
ORDER BY experiment_id;
\\echo 'PREVIEW_LSTM_INFERENCE_EVAL_ROWS'
SELECT id, model_id, symbol, prediction_horizon, threshold_logret,
       window_size, from_date, to_date, status, accuracy, accept_model, reject_reason
FROM inference_eval_result
WHERE symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
ORDER BY id;
\\echo 'PREVIEW_LSTM_ANALYSIS_ROWS'
SELECT analysis_id, experiment_id, model_id, symbol, prediction_horizon,
       target_epochs, source_train_log_path, source_infer_log_path, analysis_status
FROM experiment_analysis_result
WHERE symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
   OR source_train_log_path ILIKE '%${OLD_SYMBOL}%' OR source_train_log_path ILIKE '%${NEW_SYMBOL}%'
   OR source_infer_log_path ILIKE '%${OLD_SYMBOL}%' OR source_infer_log_path ILIKE '%${NEW_SYMBOL}%'
ORDER BY analysis_id;
SQL
}

apply_repair() {
    echo "USDJPY_SYMBOL_REPAIR_APPLY_BEGIN,old=${OLD_SYMBOL},new=${NEW_SYMBOL}"

    "${psql_forex[@]}" <<SQL
BEGIN;
DO \$\$
BEGIN
    IF to_regclass('public.${OLD_SYMBOL}') IS NOT NULL
       AND to_regclass('public.${NEW_SYMBOL}') IS NULL THEN
        EXECUTE 'ALTER TABLE public.${OLD_SYMBOL} RENAME TO ${NEW_SYMBOL}';
    ELSIF to_regclass('public.${OLD_SYMBOL}') IS NULL
       AND to_regclass('public.${NEW_SYMBOL}') IS NOT NULL THEN
        RAISE NOTICE 'FOREX table already repaired: public.${NEW_SYMBOL} exists';
    ELSE
        RAISE EXCEPTION 'Unsafe forex table state: public.${OLD_SYMBOL}=%, public.${NEW_SYMBOL}=%',
            to_regclass('public.${OLD_SYMBOL}'),
            to_regclass('public.${NEW_SYMBOL}');
    END IF;
END
\$\$;
GRANT SELECT ON TABLE public.${NEW_SYMBOL} TO pqxx;
COMMIT;
SQL

    "${psql_lstm[@]}" <<SQL
BEGIN;
LOCK TABLE model, matrix, experiment, inference_eval_result, experiment_analysis_result IN EXCLUSIVE MODE;

DO \$\$
BEGIN
    IF EXISTS (
        WITH renamed AS (
            SELECT model_id,
                   replace(name, '${OLD_SYMBOL}', '${NEW_SYMBOL}') AS new_name
            FROM model
            WHERE name LIKE '%${OLD_SYMBOL}%'
        )
        SELECT 1
        FROM renamed r
        JOIN model existing ON existing.name = r.new_name
                           AND existing.model_id <> r.model_id
    ) THEN
        RAISE EXCEPTION 'Repair would create duplicate model.name values';
    END IF;

    IF EXISTS (
        WITH projected AS (
            SELECT experiment_id,
                   CASE WHEN symbol = '${OLD_SYMBOL}' THEN '${NEW_SYMBOL}' ELSE symbol END AS symbol,
                   prediction_horizon,
                   c_next_threshold,
                   core_lr_mult,
                   head_lr_mult,
                   target_epochs,
                   checkpoint_interval,
                   train_start,
                   train_end,
                   infer_start,
                   infer_end,
                   resume_model_id,
                   duplicate_nonce
            FROM experiment
            WHERE status <> 'cancelled'
              AND symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
        )
        SELECT 1
        FROM projected
        GROUP BY symbol,
                 prediction_horizon,
                 c_next_threshold,
                 COALESCE(core_lr_mult, '-Infinity'::double precision),
                 COALESCE(head_lr_mult, '-Infinity'::double precision),
                 target_epochs,
                 checkpoint_interval,
                 train_start,
                 train_end,
                 COALESCE(infer_start, '-infinity'::timestamp with time zone),
                 COALESCE(infer_end, '-infinity'::timestamp with time zone),
                 COALESCE(resume_model_id, -1),
                 duplicate_nonce
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'Repair would create duplicate experiment identity values';
    END IF;

    IF EXISTS (
        WITH projected AS (
            SELECT id,
                   model_id,
                   CASE WHEN symbol = '${OLD_SYMBOL}' THEN '${NEW_SYMBOL}' ELSE symbol END AS symbol,
                   prediction_horizon,
                   threshold_logret,
                   window_size,
                   label_rule_id,
                   target_type,
                   from_date,
                   to_date
            FROM inference_eval_result
            WHERE status = 'completed'
              AND symbol IN ('${OLD_SYMBOL}', '${NEW_SYMBOL}')
        )
        SELECT 1
        FROM projected
        GROUP BY model_id, symbol, prediction_horizon, threshold_logret,
                 window_size, label_rule_id, target_type, from_date, to_date
        HAVING count(*) > 1
    ) THEN
        RAISE EXCEPTION 'Repair would create duplicate completed inference_eval_result identity values';
    END IF;
END
\$\$;

UPDATE model
SET name = replace(name, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE name LIKE '%${OLD_SYMBOL}%';

UPDATE model
SET comment = replace(comment, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE comment LIKE '%${OLD_SYMBOL}%';

WITH bad_models AS (
    SELECT model_id
    FROM matrix
    WHERE param_name = 'train_symbol_meta'
      AND row_idx = 0
    GROUP BY model_id
    HAVING string_agg(chr(round(value)::int), '' ORDER BY col_idx) = '${OLD_SYMBOL}'
)
UPDATE matrix
SET value = CASE col_idx
    WHEN 3 THEN ascii('p')
    WHEN 4 THEN ascii('j')
    ELSE value
END
WHERE param_name = 'train_symbol_meta'
  AND row_idx = 0
  AND col_idx IN (3, 4)
  AND model_id IN (SELECT model_id FROM bad_models);

UPDATE experiment
SET symbol = '${NEW_SYMBOL}'
WHERE symbol = '${OLD_SYMBOL}';

UPDATE experiment
SET train_log_path = replace(train_log_path, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE train_log_path LIKE '%${OLD_SYMBOL}%';

UPDATE experiment
SET infer_log_path = replace(infer_log_path, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE infer_log_path LIKE '%${OLD_SYMBOL}%';

UPDATE experiment
SET analysis_log_path = replace(analysis_log_path, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE analysis_log_path LIKE '%${OLD_SYMBOL}%';

UPDATE inference_eval_result
SET symbol = '${NEW_SYMBOL}'
WHERE symbol = '${OLD_SYMBOL}';

UPDATE experiment_analysis_result
SET symbol = '${NEW_SYMBOL}'
WHERE symbol = '${OLD_SYMBOL}';

UPDATE experiment_analysis_result
SET source_train_log_path = replace(source_train_log_path, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE source_train_log_path LIKE '%${OLD_SYMBOL}%';

UPDATE experiment_analysis_result
SET source_infer_log_path = replace(source_infer_log_path, '${OLD_SYMBOL}', '${NEW_SYMBOL}')
WHERE source_infer_log_path LIKE '%${OLD_SYMBOL}%';

COMMIT;
SQL

    echo "USDJPY_SYMBOL_REPAIR_APPLY_DONE,old=${OLD_SYMBOL},new=${NEW_SYMBOL}"
}

run_preview

if [[ "$MODE" == "apply" ]]; then
    apply_repair
    echo "USDJPY_SYMBOL_REPAIR_VERIFY_AFTER_APPLY"
    run_preview
else
    echo "USDJPY_SYMBOL_REPAIR_PREVIEW_ONLY,use=./Database/repair_usdpjyrmp_symbol.sh --apply"
fi
