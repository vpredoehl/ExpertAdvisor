-- Durable Donchian-20 experiment mode.  Existing experiments are the
-- production-enabled arm and are backfilled by the column default.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS donchian20_mode text
        NOT NULL DEFAULT 'enabled';

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_donchian20_mode_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_donchian20_mode_check
        CHECK (donchian20_mode IN ('enabled', 'zero_ablation'));

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
        duplicate_nonce
    )
    WHERE status <> 'cancelled';
