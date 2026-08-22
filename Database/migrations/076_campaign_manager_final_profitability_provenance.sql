-- Campaign Manager Phase 3A: immutable observational provenance for the exact
-- authoritative FINAL inference profitability seen by a recommendation.
-- Profitability is deliberately absent from all decision-policy, score,
-- eligibility, ranking, tie-break, and campaign-selection identities.

ALTER TABLE experiment_recommendation
    ADD COLUMN IF NOT EXISTS final_profitability_provenance_version integer,
    ADD COLUMN IF NOT EXISTS source_final_inference_eval_result_id bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_observation_id bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_unavailable_reason text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_scope text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_start text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_end text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_actionable_count bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_aggregate_return double precision,
    ADD COLUMN IF NOT EXISTS source_final_profitability_average_return double precision,
    ADD COLUMN IF NOT EXISTS source_final_profitability_metric_definition_hash text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_source_content_hash text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_observation_identity_hash text;

ALTER TABLE experiment_recommendation_evaluation_result
    ADD COLUMN IF NOT EXISTS final_profitability_provenance_version integer,
    ADD COLUMN IF NOT EXISTS source_final_inference_eval_result_id bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_observation_id bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_unavailable_reason text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_scope text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_start text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_inference_end text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_actionable_count bigint,
    ADD COLUMN IF NOT EXISTS source_final_profitability_aggregate_return double precision,
    ADD COLUMN IF NOT EXISTS source_final_profitability_average_return double precision,
    ADD COLUMN IF NOT EXISTS source_final_profitability_metric_definition_hash text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_source_content_hash text,
    ADD COLUMN IF NOT EXISTS source_final_profitability_observation_identity_hash text,
    ADD COLUMN IF NOT EXISTS profitability_evidence_canonical text,
    ADD COLUMN IF NOT EXISTS profitability_evidence_hash text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'recommendation_final_profitability_result_fkey'
          AND conrelid = 'experiment_recommendation'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT recommendation_final_profitability_result_fkey
            FOREIGN KEY (source_final_inference_eval_result_id)
            REFERENCES inference_eval_result(id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'recommendation_final_profitability_observation_fkey'
          AND conrelid = 'experiment_recommendation'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT recommendation_final_profitability_observation_fkey
            FOREIGN KEY (source_final_profitability_observation_id)
            REFERENCES inference_profitability_observation(
                profitability_observation_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'evaluation_final_profitability_result_fkey'
          AND conrelid =
              'experiment_recommendation_evaluation_result'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation_evaluation_result
            ADD CONSTRAINT evaluation_final_profitability_result_fkey
            FOREIGN KEY (source_final_inference_eval_result_id)
            REFERENCES inference_eval_result(id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'evaluation_final_profitability_observation_fkey'
          AND conrelid =
              'experiment_recommendation_evaluation_result'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation_evaluation_result
            ADD CONSTRAINT evaluation_final_profitability_observation_fkey
            FOREIGN KEY (source_final_profitability_observation_id)
            REFERENCES inference_profitability_observation(
                profitability_observation_id);
    END IF;
END $$;

-- Legacy recommendation rows have every Phase-3A field NULL. A Phase-3A row
-- is either explicitly unavailable, or contains the complete frozen
-- observation. Zero actionable predictions are available evidence with a
-- NULL average; this is distinct from unavailable evidence.
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'recommendation_final_profitability_shape_check'
          AND conrelid = 'experiment_recommendation'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation
        ADD CONSTRAINT recommendation_final_profitability_shape_check CHECK (
            (final_profitability_provenance_version IS NULL
             AND source_final_inference_eval_result_id IS NULL
             AND source_final_profitability_observation_id IS NULL
             AND source_final_profitability_unavailable_reason IS NULL
             AND source_final_profitability_inference_scope IS NULL
             AND source_final_profitability_inference_start IS NULL
             AND source_final_profitability_inference_end IS NULL
             AND source_final_profitability_actionable_count IS NULL
             AND source_final_profitability_aggregate_return IS NULL
             AND source_final_profitability_average_return IS NULL
             AND source_final_profitability_metric_definition_hash IS NULL
             AND source_final_profitability_source_content_hash IS NULL
             AND source_final_profitability_observation_identity_hash IS NULL)
            OR
            (final_profitability_provenance_version = 1
             AND source_final_profitability_inference_scope = 'final'
             AND source_final_profitability_observation_id IS NULL
             AND btrim(source_final_profitability_unavailable_reason) <> ''
             AND source_final_profitability_inference_start IS NULL
             AND source_final_profitability_inference_end IS NULL
             AND source_final_profitability_actionable_count IS NULL
             AND source_final_profitability_aggregate_return IS NULL
             AND source_final_profitability_average_return IS NULL
             AND source_final_profitability_metric_definition_hash IS NULL
             AND source_final_profitability_source_content_hash IS NULL
             AND source_final_profitability_observation_identity_hash IS NULL)
            OR
            (final_profitability_provenance_version = 1
             AND source_final_inference_eval_result_id IS NOT NULL
             AND source_final_profitability_observation_id IS NOT NULL
             AND source_final_profitability_unavailable_reason IS NULL
             AND source_final_profitability_inference_scope = 'final'
             AND btrim(source_final_profitability_inference_start) <> ''
             AND btrim(source_final_profitability_inference_end) <> ''
             AND source_final_profitability_actionable_count >= 0
             AND source_final_profitability_aggregate_return NOT IN
                 ('NaN'::double precision,'Infinity'::double precision,
                  '-Infinity'::double precision)
             AND ((source_final_profitability_actionable_count = 0
                   AND source_final_profitability_average_return IS NULL)
                  OR
                  (source_final_profitability_actionable_count > 0
                   AND source_final_profitability_average_return IS NOT NULL
                   AND source_final_profitability_average_return NOT IN
                       ('NaN'::double precision,'Infinity'::double precision,
                        '-Infinity'::double precision)))
             AND source_final_profitability_metric_definition_hash ~
                 '^fnv1a64:[0-9a-f]{16}$'
             AND source_final_profitability_source_content_hash ~
                 '^fnv1a64:[0-9a-f]{16}$'
             AND source_final_profitability_observation_identity_hash ~
                 '^fnv1a64:[0-9a-f]{16}$')
        );
    END IF;
END $$;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'evaluation_final_profitability_shape_check'
          AND conrelid =
              'experiment_recommendation_evaluation_result'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation_evaluation_result
        ADD CONSTRAINT evaluation_final_profitability_shape_check CHECK (
            ((profitability_evidence_canonical IS NULL) =
             (profitability_evidence_hash IS NULL))
            AND (profitability_evidence_hash IS NULL OR
                 profitability_evidence_hash ~ '^fnv1a64:[0-9a-f]{16}$')
            AND (
                (final_profitability_provenance_version IS NULL
                 AND source_final_inference_eval_result_id IS NULL
                 AND source_final_profitability_observation_id IS NULL
                 AND source_final_profitability_unavailable_reason IS NULL
                 AND source_final_profitability_inference_scope IS NULL
                 AND source_final_profitability_inference_start IS NULL
                 AND source_final_profitability_inference_end IS NULL
                 AND source_final_profitability_actionable_count IS NULL
                 AND source_final_profitability_aggregate_return IS NULL
                 AND source_final_profitability_average_return IS NULL
                 AND source_final_profitability_metric_definition_hash IS NULL
                 AND source_final_profitability_source_content_hash IS NULL
                 AND source_final_profitability_observation_identity_hash IS NULL)
                OR
                (final_profitability_provenance_version = 1
                 AND source_final_profitability_inference_scope = 'final'
                 AND source_final_profitability_observation_id IS NULL
                 AND btrim(source_final_profitability_unavailable_reason) <> ''
                 AND source_final_profitability_inference_start IS NULL
                 AND source_final_profitability_inference_end IS NULL
                 AND source_final_profitability_actionable_count IS NULL
                 AND source_final_profitability_aggregate_return IS NULL
                 AND source_final_profitability_average_return IS NULL
                 AND source_final_profitability_metric_definition_hash IS NULL
                 AND source_final_profitability_source_content_hash IS NULL
                 AND source_final_profitability_observation_identity_hash IS NULL)
                OR
                (final_profitability_provenance_version = 1
                 AND source_final_inference_eval_result_id IS NOT NULL
                 AND source_final_profitability_observation_id IS NOT NULL
                 AND source_final_profitability_unavailable_reason IS NULL
                 AND source_final_profitability_inference_scope = 'final'
                 AND btrim(source_final_profitability_inference_start) <> ''
                 AND btrim(source_final_profitability_inference_end) <> ''
                 AND source_final_profitability_actionable_count >= 0
                 AND source_final_profitability_aggregate_return NOT IN
                     ('NaN'::double precision,'Infinity'::double precision,
                      '-Infinity'::double precision)
                 AND ((source_final_profitability_actionable_count = 0
                       AND source_final_profitability_average_return IS NULL)
                      OR
                      (source_final_profitability_actionable_count > 0
                       AND source_final_profitability_average_return IS NOT NULL
                       AND source_final_profitability_average_return NOT IN
                           ('NaN'::double precision,'Infinity'::double precision,
                            '-Infinity'::double precision)))
                 AND source_final_profitability_metric_definition_hash ~
                     '^fnv1a64:[0-9a-f]{16}$'
                 AND source_final_profitability_source_content_hash ~
                     '^fnv1a64:[0-9a-f]{16}$'
                 AND source_final_profitability_observation_identity_hash ~
                     '^fnv1a64:[0-9a-f]{16}$')
            )
        );
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS recommendation_final_profitability_observation_idx
    ON experiment_recommendation(
        source_final_profitability_observation_id)
    WHERE source_final_profitability_observation_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS evaluation_final_profitability_observation_idx
    ON experiment_recommendation_evaluation_result(
        source_final_profitability_observation_id)
    WHERE source_final_profitability_observation_id IS NOT NULL;

CREATE OR REPLACE FUNCTION
    validate_recommendation_final_profitability_provenance()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF NEW.final_profitability_provenance_version IS NULL THEN
        RETURN NEW;
    END IF;

    -- Mirror ResolveExactFinalInferenceResult(): the result reference is valid
    -- on its own, including when no profitability observation is available.
    IF NEW.source_final_inference_eval_result_id IS NOT NULL AND NOT EXISTS (
        WITH model_config AS (
            SELECT model_id,
                round(max(value) FILTER (WHERE col_idx = 1))::bigint AS
                    prediction_horizon,
                max(value) FILTER (WHERE col_idx = 2) AS threshold_logret,
                round(max(value) FILTER (WHERE col_idx = 3))::bigint AS
                    window_size,
                round(max(value) FILTER (WHERE col_idx = 4))::integer AS
                    label_rule_id,
                round(max(value) FILTER (WHERE col_idx = 10))::bigint AS
                    completed_epochs
            FROM matrix
            WHERE param_name = 'train_config_meta'
              AND row_idx = 0
              AND model_id = NEW.source_model_id
            GROUP BY model_id
            HAVING count(DISTINCT col_idx) FILTER (
                WHERE col_idx BETWEEN 0 AND 13) >= 14
        ), target_config AS (
            SELECT model_id,
                round(max(value) FILTER (WHERE col_idx = 0))::integer AS
                    target_type
            FROM matrix
            WHERE param_name = 'target_meta'
              AND row_idx = 0
              AND model_id = NEW.source_model_id
            GROUP BY model_id
        ), source_context AS (
            SELECT experiment.experiment_id,
                model.model_id,
                experiment.symbol,
                model_config.prediction_horizon,
                model_config.threshold_logret,
                model_config.window_size,
                model_config.label_rule_id,
                COALESCE(target_config.target_type, 1) AS target_type,
                experiment.infer_start::date::text AS inference_start,
                experiment.infer_end::date::text AS inference_end,
                model_config.completed_epochs
            FROM experiment
            JOIN model
              ON model.model_id = experiment.last_model_id
             AND model.experiment_id = experiment.experiment_id
            JOIN model_config ON model_config.model_id = model.model_id
            LEFT JOIN target_config ON target_config.model_id = model.model_id
            WHERE experiment.experiment_id = NEW.source_experiment_id
              AND experiment.last_model_id = NEW.source_model_id
              AND experiment.infer_start IS NOT NULL
              AND experiment.infer_end IS NOT NULL
              AND model_config.prediction_horizon =
                    experiment.prediction_horizon
              AND abs(model_config.threshold_logret -
                      experiment.c_next_threshold) <= 1e-7
        ), exact_result AS (
            SELECT result.id
            FROM source_context
            JOIN inference_eval_result result
              ON result.model_id = source_context.model_id
             AND result.symbol = source_context.symbol
             AND result.prediction_horizon =
                    source_context.prediction_horizon
             AND result.threshold_logret = source_context.threshold_logret
             AND result.window_size = source_context.window_size
             AND result.label_rule_id = source_context.label_rule_id
             AND result.target_type = source_context.target_type
             AND result.from_date = source_context.inference_start
             AND result.to_date = source_context.inference_end
             AND result.completed_epochs = source_context.completed_epochs
             AND result.status = 'completed'
             AND result.inference_scope = 'final'
             AND result.checkpoint_eval_id IS NULL
             AND result.parent_experiment_id IS NULL
        )
        SELECT 1
        FROM exact_result
        WHERE exact_result.id = NEW.source_final_inference_eval_result_id
          AND (SELECT count(*) FROM exact_result) = 1
    ) THEN
        RAISE EXCEPTION
            'recommendation FINAL inference provenance mismatch'
            USING ERRCODE = '23514';
    END IF;

    IF NEW.source_final_profitability_observation_id IS NOT NULL AND NOT EXISTS (
        SELECT 1
        FROM inference_profitability_observation observation
        WHERE observation.profitability_observation_id =
                  NEW.source_final_profitability_observation_id
          AND observation.experiment_id = NEW.source_experiment_id
          AND observation.model_id = NEW.source_model_id
          AND observation.inference_eval_result_id =
                  NEW.source_final_inference_eval_result_id
          AND observation.inference_scope = 'final'
          AND observation.checkpoint_eval_id IS NULL
          AND observation.inference_start =
                  NEW.source_final_profitability_inference_start
          AND observation.inference_end =
                  NEW.source_final_profitability_inference_end
          AND observation.actionable_count =
                  NEW.source_final_profitability_actionable_count
          AND observation.aggregate_terminal_horizon_log_return_sum =
                  NEW.source_final_profitability_aggregate_return
          AND observation.average_terminal_horizon_log_return_per_actionable_prediction
                  IS NOT DISTINCT FROM
                  NEW.source_final_profitability_average_return
          AND observation.metric_definition_hash =
                  NEW.source_final_profitability_metric_definition_hash
          AND observation.source_content_hash =
                  NEW.source_final_profitability_source_content_hash
          AND observation.observation_identity_hash =
                  NEW.source_final_profitability_observation_identity_hash
    ) THEN
        RAISE EXCEPTION
            'recommendation FINAL profitability observation provenance mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recommendation_final_profitability_validate
    ON experiment_recommendation;
CREATE TRIGGER recommendation_final_profitability_validate
BEFORE INSERT OR UPDATE ON experiment_recommendation
FOR EACH ROW EXECUTE FUNCTION
    validate_recommendation_final_profitability_provenance();

DROP TRIGGER IF EXISTS evaluation_final_profitability_validate
    ON experiment_recommendation_evaluation_result;
CREATE TRIGGER evaluation_final_profitability_validate
BEFORE INSERT OR UPDATE ON experiment_recommendation_evaluation_result
FOR EACH ROW EXECUTE FUNCTION
    validate_recommendation_final_profitability_provenance();

CREATE OR REPLACE FUNCTION
    reject_recommendation_profitability_snapshot_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF OLD.final_profitability_provenance_version IS DISTINCT FROM
           NEW.final_profitability_provenance_version
       OR OLD.source_final_inference_eval_result_id IS DISTINCT FROM
           NEW.source_final_inference_eval_result_id
       OR OLD.source_final_profitability_observation_id IS DISTINCT FROM
           NEW.source_final_profitability_observation_id
       OR OLD.source_final_profitability_unavailable_reason IS DISTINCT FROM
           NEW.source_final_profitability_unavailable_reason
       OR OLD.source_final_profitability_inference_scope IS DISTINCT FROM
           NEW.source_final_profitability_inference_scope
       OR OLD.source_final_profitability_inference_start IS DISTINCT FROM
           NEW.source_final_profitability_inference_start
       OR OLD.source_final_profitability_inference_end IS DISTINCT FROM
           NEW.source_final_profitability_inference_end
       OR OLD.source_final_profitability_actionable_count IS DISTINCT FROM
           NEW.source_final_profitability_actionable_count
       OR OLD.source_final_profitability_aggregate_return IS DISTINCT FROM
           NEW.source_final_profitability_aggregate_return
       OR OLD.source_final_profitability_average_return IS DISTINCT FROM
           NEW.source_final_profitability_average_return
       OR OLD.source_final_profitability_metric_definition_hash IS DISTINCT FROM
           NEW.source_final_profitability_metric_definition_hash
       OR OLD.source_final_profitability_source_content_hash IS DISTINCT FROM
           NEW.source_final_profitability_source_content_hash
       OR OLD.source_final_profitability_observation_identity_hash IS DISTINCT FROM
           NEW.source_final_profitability_observation_identity_hash
    THEN
        RAISE EXCEPTION
            'recommendation profitability snapshot is immutable'
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recommendation_profitability_snapshot_immutable
    ON experiment_recommendation;
DROP TRIGGER IF EXISTS recommendation_00_profitability_snapshot_immutable
    ON experiment_recommendation;
CREATE TRIGGER recommendation_00_profitability_snapshot_immutable
BEFORE UPDATE ON experiment_recommendation
FOR EACH ROW EXECUTE FUNCTION
    reject_recommendation_profitability_snapshot_mutation();

COMMENT ON COLUMN experiment_recommendation.final_profitability_provenance_version IS
    'NULL means a pre-Phase-3A legacy snapshot; 1 means FINAL profitability was explicitly observed as available or unavailable.';
COMMENT ON COLUMN experiment_recommendation.source_final_profitability_observation_id IS
    'Exact immutable FINAL inference profitability observation; never a checkpoint observation.';
COMMENT ON COLUMN experiment_recommendation_evaluation_result.profitability_evidence_hash IS
    'Observational-only hash, deliberately excluded from scoring policy, decision evidence, ranking, and tie-break identities.';

GRANT EXECUTE ON FUNCTION
    validate_recommendation_final_profitability_provenance() TO pqxx;
GRANT EXECUTE ON FUNCTION
    reject_recommendation_profitability_snapshot_mutation() TO pqxx;
