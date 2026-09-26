-- A fresh initialization seed is scientific identity only when a new model is
-- initialized.  A resumed experiment restores its initialization from
-- resume_model_id; its required non-null seed is a schema-compatibility value
-- and must not split duplicate identity.
-- The zero resume sentinel is outside migration 094's valid seed range, and
-- avoids PostgreSQL unique-index NULL-distinctness for resumed rows.
--
-- Keep the outer transaction so direct psql use retains the prior index if
-- existing history contains resumed rows that collide under this correction.
-- migrate_lstm_db.sh owns the equivalent transaction in production.
BEGIN;

DROP INDEX IF EXISTS experiment_unique_identity_uidx;

CREATE UNIQUE INDEX experiment_unique_identity_uidx
    ON experiment (
        symbol, prediction_horizon, c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs, checkpoint_interval, train_start, train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1), donchian20_mode,
        donchian_lookback, feature_warmup_scope, feature_ablation_mask,
        resume_expand_input_width, training_objective_hash,
        COALESCE(model_input_width, -1),
        COALESCE(model_input_semantic_layout_version, -1),
        COALESCE(economic_calendar_snapshot_id, -1),
        COALESCE(economic_calendar_snapshot_hash, ''),
        (CASE WHEN resume_model_id IS NULL THEN fresh_initialization_seed
              ELSE 0::bigint END),
        duplicate_nonce
    ) WHERE status <> 'cancelled';

COMMIT;
