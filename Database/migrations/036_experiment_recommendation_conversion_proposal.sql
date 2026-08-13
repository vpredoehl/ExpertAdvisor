-- Phase 4C Step 2: immutable, manually prepared recommendation conversion
-- proposals. A proposal is audit evidence only; it neither creates nor queues
-- an experiment and is not consumed by the scheduler.

CREATE TABLE IF NOT EXISTS experiment_recommendation_conversion_proposal (
    recommendation_conversion_proposal_id bigserial PRIMARY KEY,
    recommendation_id bigint NOT NULL REFERENCES
        experiment_recommendation(recommendation_id) ON DELETE RESTRICT CHECK (
            recommendation_id > 0),
    source_experiment_id bigint NOT NULL REFERENCES
        experiment(experiment_id) ON DELETE RESTRICT CHECK (
            source_experiment_id > 0),
    conversion_contract_version integer NOT NULL CHECK (
        conversion_contract_version = 1),
    changed_parameter text NOT NULL CHECK (changed_parameter IN (
        'core_lr_mult','head_lr_mult','label_threshold','prediction_horizon')),
    source_value_canonical text NOT NULL CHECK (
        source_value_canonical <> '' AND
        octet_length(source_value_canonical) <= 128),
    proposed_value_canonical text NOT NULL CHECK (
        proposed_value_canonical <> '' AND
        octet_length(proposed_value_canonical) <= 128),
    recommendation_semantic_hash text NOT NULL CHECK (
        btrim(recommendation_semantic_hash) <> '' AND
        octet_length(recommendation_semantic_hash) <= 256),
    evaluation_identity_hash text NOT NULL CHECK (
        btrim(evaluation_identity_hash) <> '' AND
        octet_length(evaluation_identity_hash) <= 256),
    evaluation_policy_hash text NOT NULL CHECK (
        btrim(evaluation_policy_hash) <> '' AND
        octet_length(evaluation_policy_hash) <= 256),
    scoring_policy_hash text NOT NULL CHECK (
        btrim(scoring_policy_hash) <> '' AND
        octet_length(scoring_policy_hash) <= 256),
    review_authorization_hash text NOT NULL CHECK (
        btrim(review_authorization_hash) <> '' AND
        octet_length(review_authorization_hash) <= 256),
    ranking_snapshot_identity_hash text CHECK (
        ranking_snapshot_identity_hash IS NULL OR (
            btrim(ranking_snapshot_identity_hash) <> '' AND
            octet_length(ranking_snapshot_identity_hash) <= 256)),
    proposed_symbol text NOT NULL CHECK (
        btrim(proposed_symbol) <> '' AND octet_length(proposed_symbol) <= 64),
    proposed_prediction_horizon integer NOT NULL CHECK (
        proposed_prediction_horizon > 0),
    proposed_label_threshold double precision NOT NULL,
    proposed_core_lr_mult double precision,
    proposed_head_lr_mult double precision,
    proposed_target_epochs integer NOT NULL CHECK (proposed_target_epochs > 0),
    proposed_train_start_date date NOT NULL,
    proposed_train_end_date date NOT NULL,
    proposed_infer_start_date date,
    proposed_infer_end_date date,
    proposed_checkpoint_interval integer NOT NULL CHECK (
        proposed_checkpoint_interval > 0),
    proposed_resume_model_id bigint REFERENCES model(model_id) ON DELETE RESTRICT
        CHECK (proposed_resume_model_id IS NULL OR proposed_resume_model_id > 0),
    source_invocation_canonical text NOT NULL CHECK (
        btrim(source_invocation_canonical) <> '' AND
        octet_length(source_invocation_canonical) <= 1048576),
    proposed_invocation_canonical text NOT NULL CHECK (
        btrim(proposed_invocation_canonical) <> '' AND
        octet_length(proposed_invocation_canonical) <= 1048576),
    -- Canonical identity equality is byte-exact and must not inherit a
    -- database locale or a nondeterministic ICU collation.
    conversion_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(conversion_identity_canonical) <> '' AND
        octet_length(conversion_identity_canonical) <= 1048576),
    conversion_identity_hash text NOT NULL CHECK (
        btrim(conversion_identity_hash) <> '' AND
        octet_length(conversion_identity_hash) <= 256),
    conversion_hash_collision_ordinal integer NOT NULL CHECK (
        conversion_hash_collision_ordinal >= 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_conversion_proposal_hash_ordinal_uidx
        UNIQUE (conversion_identity_hash, conversion_hash_collision_ordinal),
    CONSTRAINT recommendation_conversion_proposal_finite_check CHECK (
        proposed_label_threshold NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision)
        AND (proposed_core_lr_mult IS NULL OR proposed_core_lr_mult NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
        AND (proposed_head_lr_mult IS NULL OR proposed_head_lr_mult NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
    ),
    CONSTRAINT recommendation_conversion_proposal_source_proposed_check CHECK (
        source_value_canonical <> proposed_value_canonical)
);

-- PostgreSQL btree keys cannot safely hold the potentially long authoritative
-- canonical text. The repository serializes a hash bucket, compares exact
-- canonical text, and allocates a collision ordinal. The pair is the narrow
-- concurrency constraint; neither hash nor ordinal establishes equality.
CREATE INDEX IF NOT EXISTS recommendation_conversion_proposal_canonical_idx
    ON experiment_recommendation_conversion_proposal USING hash(
        conversion_identity_canonical);
CREATE INDEX IF NOT EXISTS recommendation_conversion_proposal_recommendation_idx
    ON experiment_recommendation_conversion_proposal(
        recommendation_id, recommendation_conversion_proposal_id);
CREATE INDEX IF NOT EXISTS recommendation_conversion_proposal_source_idx
    ON experiment_recommendation_conversion_proposal(
        source_experiment_id, recommendation_conversion_proposal_id);

REVOKE UPDATE, DELETE ON experiment_recommendation_conversion_proposal FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_conversion_proposal TO pqxx;

DO $$
DECLARE
    sequence_name text := pg_get_serial_sequence(
        'experiment_recommendation_conversion_proposal',
        'recommendation_conversion_proposal_id');
BEGIN
    EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
