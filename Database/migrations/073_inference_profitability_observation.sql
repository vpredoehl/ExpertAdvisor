-- Profitability Phase 1 is additive. Historical inference rows are left
-- without observations because their exact per-window source content cannot
-- be reconstructed from inference_eval_result.
CREATE SEQUENCE inference_profitability_observation_id_seq;

CREATE TABLE inference_profitability_observation (
    profitability_observation_id bigint PRIMARY KEY DEFAULT
        nextval('inference_profitability_observation_id_seq'),
    experiment_id bigint REFERENCES experiment(experiment_id),
    model_id bigint NOT NULL REFERENCES model(model_id),
    inference_eval_result_id bigint NOT NULL REFERENCES inference_eval_result(id),
    inference_scope text NOT NULL
        CHECK (inference_scope IN ('final', 'checkpoint')),
    checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    inference_start text NOT NULL CHECK (inference_start <> ''),
    inference_end text NOT NULL CHECK (inference_end <> ''),

    prediction_count bigint NOT NULL CHECK (prediction_count >= 0),
    actionable_count bigint NOT NULL CHECK (
        actionable_count >= 0 AND actionable_count <= prediction_count),
    winning_actionable_count bigint NOT NULL CHECK (
        winning_actionable_count >= 0 AND
        winning_actionable_count <= actionable_count),
    losing_actionable_count bigint NOT NULL CHECK (
        losing_actionable_count >= 0 AND
        losing_actionable_count <= actionable_count),
    gross_positive_terminal_horizon_log_return_sum double precision NOT NULL
        CHECK (gross_positive_terminal_horizon_log_return_sum >= 0),
    gross_negative_terminal_horizon_log_return_sum double precision NOT NULL
        CHECK (gross_negative_terminal_horizon_log_return_sum <= 0),
    aggregate_terminal_horizon_log_return_sum double precision NOT NULL,
    average_terminal_horizon_log_return_per_actionable_prediction
        double precision,

    metric_definition_canonical text NOT NULL
        CHECK (metric_definition_canonical <> ''),
    metric_definition_hash text NOT NULL
        CHECK (metric_definition_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    source_content_hash text NOT NULL
        CHECK (source_content_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    observation_identity_canonical text NOT NULL UNIQUE
        CHECK (observation_identity_canonical <> ''),
    observation_identity_hash text NOT NULL
        CHECK (observation_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT inference_profitability_scope_identity_check CHECK (
        (inference_scope = 'final' AND checkpoint_eval_id IS NULL) OR
        (inference_scope = 'checkpoint' AND checkpoint_eval_id IS NOT NULL AND
         experiment_id IS NOT NULL)),
    CONSTRAINT inference_profitability_win_loss_count_check CHECK (
        winning_actionable_count + losing_actionable_count <= actionable_count),
    CONSTRAINT inference_profitability_finite_returns_check CHECK (
        gross_positive_terminal_horizon_log_return_sum NOT IN (
            'NaN'::double precision, 'Infinity'::double precision,
            '-Infinity'::double precision) AND
        gross_negative_terminal_horizon_log_return_sum NOT IN (
            'NaN'::double precision, 'Infinity'::double precision,
            '-Infinity'::double precision) AND
        aggregate_terminal_horizon_log_return_sum NOT IN (
            'NaN'::double precision, 'Infinity'::double precision,
            '-Infinity'::double precision) AND
        (average_terminal_horizon_log_return_per_actionable_prediction IS NULL OR
         average_terminal_horizon_log_return_per_actionable_prediction NOT IN (
            'NaN'::double precision, 'Infinity'::double precision,
            '-Infinity'::double precision))),
    CONSTRAINT inference_profitability_average_presence_check CHECK (
        (actionable_count = 0 AND
         average_terminal_horizon_log_return_per_actionable_prediction IS NULL) OR
        (actionable_count > 0 AND
         average_terminal_horizon_log_return_per_actionable_prediction IS NOT NULL))
);

ALTER SEQUENCE inference_profitability_observation_id_seq OWNED BY
    inference_profitability_observation.profitability_observation_id;

CREATE INDEX inference_profitability_final_lookup_idx
    ON inference_profitability_observation (
        experiment_id, model_id, inference_eval_result_id,
        metric_definition_hash, source_content_hash)
    WHERE inference_scope = 'final' AND checkpoint_eval_id IS NULL;

CREATE INDEX inference_profitability_checkpoint_lookup_idx
    ON inference_profitability_observation (
        checkpoint_eval_id, inference_eval_result_id,
        metric_definition_hash, source_content_hash)
    WHERE inference_scope = 'checkpoint';

CREATE INDEX inference_profitability_observation_hash_idx
    ON inference_profitability_observation (observation_identity_hash);

CREATE OR REPLACE FUNCTION validate_inference_profitability_observation_provenance()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM inference_eval_result ier
        JOIN model m ON m.model_id = NEW.model_id
        WHERE ier.id = NEW.inference_eval_result_id
          AND ier.model_id = NEW.model_id
          AND ier.status = 'completed'
          AND ier.inference_scope = NEW.inference_scope
          AND ier.checkpoint_eval_id IS NOT DISTINCT FROM NEW.checkpoint_eval_id
          AND ier.from_date = NEW.inference_start
          AND ier.to_date = NEW.inference_end
          AND (
              (NEW.inference_scope = 'final' AND
               m.experiment_id IS NOT DISTINCT FROM NEW.experiment_id AND
               ier.parent_experiment_id IS NULL) OR
              (NEW.inference_scope = 'checkpoint' AND
               ier.parent_experiment_id = NEW.experiment_id)
          )
    ) THEN
        RAISE EXCEPTION
            'inference profitability provenance does not match completed inference evidence';
    END IF;

    IF NEW.inference_scope = 'checkpoint' AND NOT EXISTS (
        SELECT 1
        FROM experiment_checkpoint_eval ce
        WHERE ce.checkpoint_eval_id = NEW.checkpoint_eval_id
          AND ce.parent_experiment_id = NEW.experiment_id
          AND ce.checkpoint_model_id = NEW.model_id
    ) THEN
        RAISE EXCEPTION
            'checkpoint profitability provenance does not match checkpoint evaluation';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER inference_profitability_validate_provenance_trigger
BEFORE INSERT ON inference_profitability_observation
FOR EACH ROW EXECUTE FUNCTION
    validate_inference_profitability_observation_provenance();

CREATE OR REPLACE FUNCTION reject_inference_profitability_observation_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'inference profitability observations are immutable';
END;
$$;

CREATE TRIGGER inference_profitability_immutable_trigger
BEFORE UPDATE OR DELETE ON inference_profitability_observation
FOR EACH ROW EXECUTE FUNCTION
    reject_inference_profitability_observation_mutation();

COMMENT ON TABLE inference_profitability_observation IS
    'Immutable terminal-horizon directional log-return observations produced by completed inference; not portfolio P&L.';
COMMENT ON COLUMN inference_profitability_observation.source_content_hash IS
    'FNV-1a identity of the ordered predicted-class and terminal-price inputs consumed by the metric calculation.';
COMMENT ON COLUMN inference_profitability_observation.metric_definition_canonical IS
    'Versioned executable metric semantics, including explicit absent cost, sizing, leverage, and overlap models.';
COMMENT ON COLUMN inference_profitability_observation.average_terminal_horizon_log_return_per_actionable_prediction IS
    'NULL when actionable_count is zero; otherwise aggregate return divided by actionable_count.';

GRANT SELECT, INSERT ON inference_profitability_observation TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE
    inference_profitability_observation_id_seq TO pqxx;
GRANT EXECUTE ON FUNCTION
    validate_inference_profitability_observation_provenance() TO pqxx;
GRANT EXECUTE ON FUNCTION
    reject_inference_profitability_observation_mutation() TO pqxx;
