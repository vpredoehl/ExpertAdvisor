-- Phase 4A Step 4: immutable, advisory recommendation score history.
-- Scoring never mutates experiment or recommendation state.

ALTER TABLE experiment_recommendation
    ADD COLUMN IF NOT EXISTS source_rank integer;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_recommendation_source_rank_positive_check'
    ) THEN
        ALTER TABLE experiment_recommendation
            ADD CONSTRAINT experiment_recommendation_source_rank_positive_check
            CHECK (recommendation_scan_id IS NULL OR source_rank > 0)
            NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS experiment_recommendation_score_run (
    recommendation_score_run_id bigserial PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('running','completed','failed')),
    scoring_policy_canonical text NOT NULL,
    scoring_policy_hash text NOT NULL,
    scoring_version integer NOT NULL CHECK (scoring_version > 0),
    recommendation_status_filter text
        CHECK (recommendation_status_filter IS NULL OR
               recommendation_status_filter IN ('proposed','rejected','expired','approved')),
    symbol_filter text,
    horizon_filter integer CHECK (horizon_filter IS NULL OR horizon_filter > 0),
    recommendation_scan_filter bigint
        REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    recommendation_id_filter bigint
        REFERENCES experiment_recommendation(recommendation_id),
    requested_limit integer CHECK (requested_limit IS NULL OR requested_limit > 0),
    recommendations_considered integer NOT NULL DEFAULT 0
        CHECK (recommendations_considered >= 0),
    recommendations_scored integer NOT NULL DEFAULT 0
        CHECK (recommendations_scored >= 0),
    recommendations_skipped integer NOT NULL DEFAULT 0
        CHECK (recommendations_skipped >= 0),
    scoring_errors integer NOT NULL DEFAULT 0 CHECK (scoring_errors >= 0),
    hash_collisions integer NOT NULL DEFAULT 0 CHECK (hash_collisions >= 0),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CHECK (
        (status = 'running' AND completed_at IS NULL AND error_message IS NULL)
        OR (status = 'completed' AND completed_at IS NOT NULL AND error_message IS NULL)
        OR (status = 'failed' AND completed_at IS NOT NULL AND
            error_message IS NOT NULL AND btrim(error_message) <> '')
    )
);

CREATE TABLE IF NOT EXISTS experiment_recommendation_score (
    recommendation_score_id bigserial PRIMARY KEY,
    recommendation_score_run_id bigint NOT NULL
        REFERENCES experiment_recommendation_score_run(recommendation_score_run_id),
    recommendation_id bigint NOT NULL
        REFERENCES experiment_recommendation(recommendation_id),
    scoring_policy_canonical text NOT NULL,
    scoring_policy_hash text NOT NULL,
    scoring_version integer NOT NULL CHECK (scoring_version > 0),
    recommendation_semantic_canonical text NOT NULL,
    recommendation_policy_canonical text NOT NULL,
    source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    final_score double precision NOT NULL CHECK (
        final_score >= 0.0 AND final_score <= 1.0),
    raw_positive_score double precision NOT NULL CHECK (
        raw_positive_score NOT IN ('NaN'::double precision,
                                   'Infinity'::double precision,
                                   '-Infinity'::double precision)),
    raw_penalty_score double precision NOT NULL CHECK (
        raw_penalty_score NOT IN ('NaN'::double precision,
                                  'Infinity'::double precision,
                                  '-Infinity'::double precision)),
    raw_total_score double precision NOT NULL CHECK (
        raw_total_score NOT IN ('NaN'::double precision,
                                'Infinity'::double precision,
                                '-Infinity'::double precision)),
    structural_distance double precision NOT NULL CHECK (
        structural_distance >= 0.0 AND
        structural_distance <> 'Infinity'::double precision),
    score_rank integer NOT NULL CHECK (score_rank > 0),
    tie_group integer NOT NULL CHECK (tie_group > 0),
    ranking_ordinal integer NOT NULL CHECK (ranking_ordinal > 0),
    score_status text NOT NULL CHECK (score_status = 'scored'),
    reason_code text NOT NULL CHECK (btrim(reason_code) <> ''),
    explanation text NOT NULL CHECK (btrim(explanation) <> ''),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recommendation_score_run_id, recommendation_id),
    UNIQUE (recommendation_score_run_id, ranking_ordinal)
);

CREATE TABLE IF NOT EXISTS experiment_recommendation_score_component (
    recommendation_score_component_id bigserial PRIMARY KEY,
    recommendation_score_id bigint NOT NULL
        REFERENCES experiment_recommendation_score(recommendation_score_id),
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
        weight >= 0.0 AND weight <> 'Infinity'::double precision),
    weighted_contribution double precision NOT NULL CHECK (
        weighted_contribution >= 0.0 AND
        weighted_contribution <> 'Infinity'::double precision),
    is_penalty boolean NOT NULL,
    explanation text NOT NULL CHECK (btrim(explanation) <> ''),
    created_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE (recommendation_score_id, component_ordinal),
    UNIQUE (recommendation_score_id, component_name)
);

CREATE INDEX IF NOT EXISTS experiment_recommendation_score_run_status_idx
    ON experiment_recommendation_score_run(status,
        recommendation_score_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_score_run_policy_idx
    ON experiment_recommendation_score_run(scoring_policy_hash,
        recommendation_score_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_score_recommendation_idx
    ON experiment_recommendation_score(recommendation_id,
        recommendation_score_run_id DESC);
CREATE INDEX IF NOT EXISTS experiment_recommendation_score_run_rank_idx
    ON experiment_recommendation_score(recommendation_score_run_id,
        score_rank, ranking_ordinal);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_recommendation_score_run TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_recommendation_score TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_recommendation_score_component TO pqxx;

-- PostgreSQL shortens generated serial sequence identifiers to NAMEDATALEN.
-- Resolve them authoritatively rather than repeating a possibly truncated name.
DO $$
DECLARE
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence('experiment_recommendation_score_run',
                               'recommendation_score_run_id'),
        pg_get_serial_sequence('experiment_recommendation_score',
                               'recommendation_score_id'),
        pg_get_serial_sequence('experiment_recommendation_score_component',
                               'recommendation_score_component_id')
    ]
    LOOP
        EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO pqxx',
                       sequence_name);
    END LOOP;
END $$;
