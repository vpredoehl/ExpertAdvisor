-- Phase 4B Step 1: deterministic, persisted advisory recommendation
-- evaluation evidence. These tables do not update recommendation review state,
-- create experiments, or participate in scheduler polling.

CREATE TABLE IF NOT EXISTS experiment_recommendation_evaluation_run (
    recommendation_evaluation_run_id bigserial PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('running','completed','failed')),
    evaluation_run_identity_canonical text NOT NULL
        CHECK (btrim(evaluation_run_identity_canonical) <> ''),
    evaluation_run_identity_hash text NOT NULL
        CHECK (btrim(evaluation_run_identity_hash) <> ''),
    evaluation_policy_canonical text NOT NULL
        CHECK (btrim(evaluation_policy_canonical) <> ''),
    evaluation_policy_hash text NOT NULL
        CHECK (btrim(evaluation_policy_hash) <> ''),
    evaluation_version integer NOT NULL CHECK (evaluation_version > 0),
    evaluator_version integer NOT NULL CHECK (evaluator_version > 0),
    scoring_policy_canonical text NOT NULL
        CHECK (btrim(scoring_policy_canonical) <> ''),
    scoring_policy_hash text NOT NULL
        CHECK (btrim(scoring_policy_hash) <> ''),
    scoring_version integer NOT NULL CHECK (scoring_version > 0),
    recommendation_scan_filter bigint
        REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    recommendation_id_filter bigint
        REFERENCES experiment_recommendation(recommendation_id),
    requested_limit integer CHECK (requested_limit IS NULL OR requested_limit > 0),
    evidence_snapshot_canonical text NOT NULL
        CHECK (btrim(evidence_snapshot_canonical) <> ''),
    evidence_snapshot_hash text NOT NULL
        CHECK (btrim(evidence_snapshot_hash) <> ''),
    recommendations_considered integer NOT NULL DEFAULT 0
        CHECK (recommendations_considered >= 0),
    recommendations_evaluated integer NOT NULL DEFAULT 0
        CHECK (recommendations_evaluated >= 0),
    recommendations_eligible integer NOT NULL DEFAULT 0
        CHECK (recommendations_eligible >= 0),
    recommendations_blocked integer NOT NULL DEFAULT 0
        CHECK (recommendations_blocked >= 0),
    evaluation_errors integer NOT NULL DEFAULT 0
        CHECK (evaluation_errors >= 0),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT experiment_recommendation_evaluation_run_identity_uidx
        UNIQUE (evaluation_run_identity_canonical),
    CONSTRAINT experiment_recommendation_evaluation_run_lifecycle_check CHECK (
        (status = 'running' AND completed_at IS NULL AND error_message IS NULL)
        OR (status = 'completed' AND completed_at IS NOT NULL AND error_message IS NULL)
        OR (status = 'failed' AND completed_at IS NOT NULL
            AND error_message IS NOT NULL AND btrim(error_message) <> '')
    )
);

CREATE TABLE IF NOT EXISTS experiment_recommendation_evaluation_result (
    recommendation_evaluation_result_id bigserial PRIMARY KEY,
    recommendation_evaluation_run_id bigint NOT NULL
        REFERENCES experiment_recommendation_evaluation_run(
            recommendation_evaluation_run_id),
    recommendation_id bigint NOT NULL
        REFERENCES experiment_recommendation(recommendation_id),
    evaluation_identity_canonical text NOT NULL
        CHECK (btrim(evaluation_identity_canonical) <> ''),
    evaluation_identity_hash text NOT NULL
        CHECK (btrim(evaluation_identity_hash) <> ''),
    recommendation_semantic_canonical text NOT NULL
        CHECK (btrim(recommendation_semantic_canonical) <> ''),
    recommendation_semantic_hash text NOT NULL
        CHECK (btrim(recommendation_semantic_hash) <> ''),
    recommendation_policy_canonical text NOT NULL
        CHECK (btrim(recommendation_policy_canonical) <> ''),
    recommendation_policy_hash text NOT NULL
        CHECK (btrim(recommendation_policy_hash) <> ''),
    recommendation_scan_id bigint NOT NULL
        REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    source_model_id bigint REFERENCES model(model_id),
    source_analysis_id bigint REFERENCES experiment_analysis_result(analysis_id),
    evidence_canonical text NOT NULL CHECK (btrim(evidence_canonical) <> ''),
    evidence_hash text NOT NULL CHECK (btrim(evidence_hash) <> ''),
    eligibility text NOT NULL CHECK (eligibility IN ('eligible','ineligible')),
    disposition text NOT NULL CHECK (disposition IN (
        'advisory_ready','insufficient_evidence',
        'blocked_pending_duplicate','blocked_active_duplicate',
        'completed_duplicate','stale_source_evidence',
        'unsupported_recommendation_family','invalid_persisted_evidence')),
    reason_code text NOT NULL CHECK (btrim(reason_code) <> ''),
    explanation text NOT NULL CHECK (btrim(explanation) <> ''),
    final_score double precision,
    raw_positive_score double precision,
    raw_penalty_score double precision,
    raw_total_score double precision,
    component_count integer NOT NULL CHECK (component_count >= 0),
    missing_evidence_count integer NOT NULL CHECK (missing_evidence_count >= 0),
    ranking_ordinal integer NOT NULL CHECK (ranking_ordinal > 0),
    result_status text NOT NULL DEFAULT 'evaluated' CHECK (result_status='evaluated'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_evaluation_result_run_recommendation_uidx
        UNIQUE (recommendation_evaluation_run_id, recommendation_id),
    CONSTRAINT experiment_recommendation_evaluation_result_run_ordinal_uidx
        UNIQUE (recommendation_evaluation_run_id, ranking_ordinal),
    CONSTRAINT experiment_recommendation_evaluation_result_identity_uidx
        UNIQUE (recommendation_evaluation_run_id, evaluation_identity_canonical),
    CONSTRAINT experiment_recommendation_evaluation_result_shape_check CHECK (
        (
            eligibility = 'eligible'
            AND disposition = 'advisory_ready'
            AND final_score IS NOT NULL
            AND final_score >= 0.0 AND final_score <= 1.0
            AND raw_positive_score IS NOT NULL
            AND raw_penalty_score IS NOT NULL
            AND raw_total_score IS NOT NULL
            AND component_count > 0
        )
        OR (
            eligibility = 'ineligible'
            AND disposition <> 'advisory_ready'
            AND final_score IS NULL
            AND raw_positive_score IS NULL
            AND raw_penalty_score IS NULL
            AND raw_total_score IS NULL
            AND component_count = 0
        )
    ),
    CONSTRAINT experiment_recommendation_evaluation_result_finite_check CHECK (
        (final_score IS NULL OR final_score NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
        AND (raw_positive_score IS NULL OR raw_positive_score NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
        AND (raw_penalty_score IS NULL OR raw_penalty_score NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
        AND (raw_total_score IS NULL OR raw_total_score NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
    )
);

CREATE TABLE IF NOT EXISTS experiment_recommendation_evaluation_component (
    recommendation_evaluation_component_id bigserial PRIMARY KEY,
    recommendation_evaluation_result_id bigint NOT NULL
        REFERENCES experiment_recommendation_evaluation_result(
            recommendation_evaluation_result_id),
    component_ordinal integer NOT NULL CHECK (component_ordinal > 0),
    component_name text NOT NULL CHECK (component_name IN (
        'leader_quality','inference_accuracy','evidence_strength',
        'neutral_balance','structural_proximity','parameter_preference',
        'source_rank','horizon_change_penalty','relative_mutation_penalty')),
    reason_code text NOT NULL CHECK (btrim(reason_code) <> ''),
    input_canonical text NOT NULL,
    normalized_value double precision NOT NULL CHECK (
        normalized_value >= 0.0 AND normalized_value <= 1.0),
    weight double precision NOT NULL CHECK (
        weight >= 0.0 AND weight NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision)),
    weighted_contribution double precision NOT NULL CHECK (
        weighted_contribution >= 0.0 AND weighted_contribution NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision)),
    is_penalty boolean NOT NULL,
    is_missing boolean NOT NULL DEFAULT false,
    explanation text NOT NULL CHECK (btrim(explanation) <> ''),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT experiment_recommendation_evaluation_component_ordinal_uidx
        UNIQUE (recommendation_evaluation_result_id, component_ordinal),
    CONSTRAINT experiment_recommendation_evaluation_component_name_uidx
        UNIQUE (recommendation_evaluation_result_id, component_name)
);

CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_run_status_idx
    ON experiment_recommendation_evaluation_run(
        status, recommendation_evaluation_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_run_hash_idx
    ON experiment_recommendation_evaluation_run(
        evaluation_run_identity_hash, recommendation_evaluation_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_result_recommendation_idx
    ON experiment_recommendation_evaluation_result(
        recommendation_id, recommendation_evaluation_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_evaluation_result_disposition_idx
    ON experiment_recommendation_evaluation_result(
        disposition, recommendation_evaluation_run_id DESC, ranking_ordinal);

REVOKE UPDATE, DELETE ON experiment_recommendation_evaluation_run FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_evaluation_run TO pqxx;
GRANT UPDATE (status,recommendations_considered,recommendations_evaluated,
    recommendations_eligible,recommendations_blocked,evaluation_errors,
    completed_at,error_message,updated_at)
    ON experiment_recommendation_evaluation_run TO pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_evaluation_result TO pqxx;
REVOKE UPDATE, DELETE ON experiment_recommendation_evaluation_result FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_evaluation_component TO pqxx;
REVOKE UPDATE, DELETE ON experiment_recommendation_evaluation_component FROM pqxx;

DO $$
DECLARE
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence('experiment_recommendation_evaluation_run',
                               'recommendation_evaluation_run_id'),
        pg_get_serial_sequence('experiment_recommendation_evaluation_result',
                               'recommendation_evaluation_result_id'),
        pg_get_serial_sequence('experiment_recommendation_evaluation_component',
                               'recommendation_evaluation_component_id')
    ]
    LOOP
        EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO pqxx',
                       sequence_name);
    END LOOP;
END $$;
