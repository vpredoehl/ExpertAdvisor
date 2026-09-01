-- Bind every newly materialized experiment to the model-input width and
-- append-only semantic-layout generation selected by the queueing binary.
-- Historical rows remain NULL/NULL so completed and in-flight pre-089 work is
-- not reinterpreted. New application writes always supply both values.

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS model_input_width integer,
    ADD COLUMN IF NOT EXISTS model_input_semantic_layout_version integer;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_model_input_identity_shape_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_model_input_identity_shape_check CHECK (
        (model_input_width IS NULL AND
         model_input_semantic_layout_version IS NULL)
        OR
        (model_input_width IS NOT NULL AND
         model_input_semantic_layout_version IS NOT NULL AND
         model_input_width > 4 AND
         model_input_semantic_layout_version > 0)
    );

CREATE OR REPLACE FUNCTION enforce_experiment_model_input_identity()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'INSERT' AND
       (NEW.model_input_width IS NULL OR
        NEW.model_input_semantic_layout_version IS NULL)
    THEN
        RAISE EXCEPTION 'new experiment requires complete model input identity';
    END IF;
    IF TG_OP = 'UPDATE' THEN
        IF OLD.model_input_width IS DISTINCT FROM NEW.model_input_width OR
           OLD.model_input_semantic_layout_version IS DISTINCT FROM
               NEW.model_input_semantic_layout_version
        THEN
            RAISE EXCEPTION 'experiment model input identity is immutable';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS experiment_model_input_identity_trigger
    ON experiment;
CREATE TRIGGER experiment_model_input_identity_trigger
BEFORE INSERT OR UPDATE ON experiment
FOR EACH ROW EXECUTE FUNCTION
    enforce_experiment_model_input_identity();

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
        duplicate_nonce
    ) WHERE status <> 'cancelled';

COMMENT ON COLUMN experiment.model_input_width IS
    'Immutable expected total LSTM model input width. NULL identifies a legacy pre-089 experiment.';
COMMENT ON COLUMN experiment.model_input_semantic_layout_version IS
    'Immutable append-only ModelInputExpansion semantic layout generation selected at experiment materialization. NULL identifies a legacy pre-089 experiment.';

GRANT EXECUTE ON FUNCTION
    enforce_experiment_model_input_identity() TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
