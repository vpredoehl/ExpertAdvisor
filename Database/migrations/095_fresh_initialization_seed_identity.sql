-- Fresh initialization seed is part of fresh experiment identity.
-- Migration 094 persisted the seed and application duplicate detection includes it;
-- extend the database uniqueness contract to match.
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
        fresh_initialization_seed,
        duplicate_nonce
    ) WHERE status <> 'cancelled';
