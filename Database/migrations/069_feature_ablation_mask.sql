-- Experiment-control metadata: Tensor layout and model input widths are not
-- changed by ablation. Empty text is the immutable compatibility default.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS feature_ablation_mask text NOT NULL DEFAULT '';

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
        donchian_lookback, feature_warmup_scope, feature_ablation_mask,
        duplicate_nonce
    ) WHERE status <> 'cancelled';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
