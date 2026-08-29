-- Phase 12: advisory prospective outcomes for a precommitted ranking cohort.
-- This table is deliberately separate from inference_eval_result and
-- inference_profitability_observation so authoritative FINAL/checkpoint
-- evidence cannot be overwritten or satisfy this contract.

CREATE SEQUENCE IF NOT EXISTS
    campaign_profitability_prospective_outcome_result_id_seq;

CREATE TABLE IF NOT EXISTS campaign_profitability_prospective_outcome_result (
    prospective_outcome_result_id bigint PRIMARY KEY DEFAULT nextval(
        'campaign_profitability_prospective_outcome_result_id_seq'),
    validation_cohort_identity_hash text NOT NULL CHECK (
        validation_cohort_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    ranking_snapshot_id bigint NOT NULL REFERENCES
        experiment_recommendation_ranking_snapshot(
            recommendation_ranking_snapshot_id) ON DELETE RESTRICT,
    source_evaluation_run_id bigint NOT NULL REFERENCES
        experiment_recommendation_evaluation_run(
            recommendation_evaluation_run_id) ON DELETE RESTRICT,
    source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    source_model_id bigint NOT NULL REFERENCES model(model_id)
        ON DELETE RESTRICT,
    symbol text NOT NULL CHECK (btrim(symbol) <> ''),
    prediction_horizon integer NOT NULL CHECK (prediction_horizon > 0),
    threshold_logret double precision NOT NULL,
    window_size integer NOT NULL CHECK (window_size > 0),
    label_rule_id integer NOT NULL,
    target_type integer NOT NULL,
    input_width integer NOT NULL CHECK (input_width > 0),
    outcome_start date NOT NULL,
    outcome_end date NOT NULL CHECK (outcome_end > outcome_start),
    job_identity_hash text NOT NULL UNIQUE CHECK (
        job_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    feature_semantic_hash text NOT NULL CHECK (
        feature_semantic_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    model_lineage_hash text NOT NULL CHECK (
        model_lineage_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    model_artifact_content_hash text NOT NULL CHECK (
        model_artifact_content_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    metric_definition_canonical text NOT NULL CHECK (
        btrim(metric_definition_canonical) <> ''),
    metric_definition_hash text NOT NULL CHECK (
        metric_definition_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    source_content_hash text NOT NULL CHECK (
        source_content_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
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
    inference_accuracy double precision NOT NULL CHECK (
        inference_accuracy >= 0 AND inference_accuracy <= 1),
    outcome_identity_canonical text NOT NULL UNIQUE CHECK (
        btrim(outcome_identity_canonical) <> ''),
    outcome_identity_hash text NOT NULL UNIQUE CHECK (
        outcome_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT campaign_profitability_prospective_counts_check CHECK (
        winning_actionable_count + losing_actionable_count <= actionable_count),
    CONSTRAINT campaign_profitability_prospective_average_check CHECK (
        (actionable_count = 0 AND
         average_terminal_horizon_log_return_per_actionable_prediction IS NULL)
        OR
        (actionable_count > 0 AND
         average_terminal_horizon_log_return_per_actionable_prediction IS NOT NULL))
);

ALTER SEQUENCE campaign_profitability_prospective_outcome_result_id_seq
    OWNED BY campaign_profitability_prospective_outcome_result.
        prospective_outcome_result_id;

CREATE INDEX IF NOT EXISTS
    campaign_profitability_prospective_outcome_cohort_idx
ON campaign_profitability_prospective_outcome_result(
    validation_cohort_identity_hash, source_model_id, outcome_start,
    outcome_end);

CREATE OR REPLACE FUNCTION
validate_campaign_profitability_prospective_outcome_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM experiment_recommendation_ranking_snapshot snapshot
        JOIN experiment_recommendation_ranking_member member
          ON member.recommendation_ranking_snapshot_id =
             snapshot.recommendation_ranking_snapshot_id
        JOIN experiment experiment_row
          ON experiment_row.experiment_id = NEW.source_experiment_id
        JOIN model source_model
          ON source_model.model_id = NEW.source_model_id
        WHERE snapshot.recommendation_ranking_snapshot_id =
                  NEW.ranking_snapshot_id
          AND snapshot.status = 'completed'
          AND snapshot.evaluation_run_filter = NEW.source_evaluation_run_id
          AND member.source_experiment_id = NEW.source_experiment_id
          AND member.source_model_id = NEW.source_model_id
          AND source_model.experiment_id = NEW.source_experiment_id
          AND experiment_row.last_model_id = NEW.source_model_id
          AND experiment_row.symbol = NEW.symbol
          AND experiment_row.prediction_horizon = NEW.prediction_horizon
          AND abs(experiment_row.c_next_threshold - NEW.threshold_logret) <=
              0.0000001
          AND experiment_row.infer_end::date < NEW.outcome_start
    ) THEN
        RAISE EXCEPTION
            'prospective outcome provenance does not match an exact frozen final source model';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS
    campaign_profitability_prospective_outcome_validate_trigger
ON campaign_profitability_prospective_outcome_result;
CREATE TRIGGER campaign_profitability_prospective_outcome_validate_trigger
BEFORE INSERT ON campaign_profitability_prospective_outcome_result
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_profitability_prospective_outcome_v1();

CREATE OR REPLACE FUNCTION
reject_campaign_profitability_prospective_outcome_mutation_v1()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'prospective campaign profitability outcomes are immutable';
END;
$$;

DROP TRIGGER IF EXISTS
    campaign_profitability_prospective_outcome_immutable_trigger
ON campaign_profitability_prospective_outcome_result;
CREATE TRIGGER campaign_profitability_prospective_outcome_immutable_trigger
BEFORE UPDATE OR DELETE ON campaign_profitability_prospective_outcome_result
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_profitability_prospective_outcome_mutation_v1();

REVOKE ALL ON campaign_profitability_prospective_outcome_result FROM PUBLIC;
GRANT SELECT, INSERT ON campaign_profitability_prospective_outcome_result TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE
    campaign_profitability_prospective_outcome_result_id_seq TO pqxx;
