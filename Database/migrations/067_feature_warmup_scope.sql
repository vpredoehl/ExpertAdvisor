-- Rows before this migration used the cold logical-boundary query. New work
-- names full-history feature warmup explicitly.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS feature_warmup_scope text
        NOT NULL DEFAULT 'legacy_cold_boundary';

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_feature_warmup_scope_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_feature_warmup_scope_check
        CHECK (feature_warmup_scope IN (
            'legacy_cold_boundary',
            'full_history_warmup'));

DROP INDEX IF EXISTS experiment_unique_identity_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_unique_identity_uidx
    ON experiment (
        symbol, prediction_horizon, c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs, checkpoint_interval, train_start, train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1), donchian20_mode,
        feature_warmup_scope, duplicate_nonce
    )
    WHERE status <> 'cancelled';
