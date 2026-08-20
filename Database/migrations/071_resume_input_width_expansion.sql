-- Explicit scientific provenance for append-only model input-width expansion.
-- Ordinary resumes remain false and retain their historical-width behavior.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS resume_expand_input_width boolean NOT NULL
        DEFAULT false;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_resume_expand_input_width_source_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_resume_expand_input_width_source_check CHECK (
        NOT resume_expand_input_width OR resume_model_id IS NOT NULL
    );

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
        resume_expand_input_width, duplicate_nonce
    ) WHERE status <> 'cancelled';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
