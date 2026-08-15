-- Configurable Donchian lookback.  Rows preceding this migration used the
-- closed 20-bar definition and are preserved by the forward-only default.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS donchian_lookback integer
        NOT NULL DEFAULT 20;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_donchian_lookback_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_donchian_lookback_check
        CHECK (donchian_lookback BETWEEN 1 AND 10000);

DROP INDEX IF EXISTS experiment_unique_identity_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_unique_identity_uidx
    ON experiment (
        symbol,
        prediction_horizon,
        c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs,
        checkpoint_interval,
        train_start,
        train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1),
        donchian20_mode,
        donchian_lookback,
        feature_warmup_scope,
        duplicate_nonce
    )
    WHERE status <> 'cancelled';
