-- Phase 4C Step 5: explicit activation of one converted paused experiment.
-- Activation makes the existing experiment pending/train; it does not create
-- an experiment, start a worker, or bypass the scheduler.

CREATE UNIQUE INDEX IF NOT EXISTS conversion_execution_provenance_uidx
    ON experiment_recommendation_conversion_execution(
        recommendation_conversion_execution_id,
        recommendation_conversion_proposal_id,
        recommendation_conversion_review_decision_id,
        experiment_id);

CREATE TABLE IF NOT EXISTS
experiment_recommendation_conversion_activation (
    recommendation_conversion_activation_id bigserial PRIMARY KEY CHECK (
        recommendation_conversion_activation_id > 0),
    recommendation_conversion_execution_id bigint NOT NULL UNIQUE CHECK (
        recommendation_conversion_execution_id > 0),
    recommendation_conversion_proposal_id bigint NOT NULL CHECK (
        recommendation_conversion_proposal_id > 0),
    recommendation_conversion_review_decision_id bigint NOT NULL CHECK (
        recommendation_conversion_review_decision_id > 0),
    experiment_id bigint NOT NULL UNIQUE CHECK (experiment_id > 0),
    activation_contract_version integer NOT NULL CHECK (
        activation_contract_version = 1),
    previous_status text NOT NULL CHECK (previous_status = 'paused'),
    previous_phase text NOT NULL CHECK (previous_phase = 'train'),
    resulting_status text NOT NULL CHECK (resulting_status = 'pending'),
    resulting_phase text NOT NULL CHECK (resulting_phase = 'train'),
    activation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(activation_identity_canonical) <> '' AND
        octet_length(activation_identity_canonical) <= 4096),
    activation_identity_hash text NOT NULL CHECK (
        btrim(activation_identity_hash) <> '' AND
        octet_length(activation_identity_hash) <= 256),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT conversion_activation_execution_provenance_fkey FOREIGN KEY (
        recommendation_conversion_execution_id,
        recommendation_conversion_proposal_id,
        recommendation_conversion_review_decision_id,
        experiment_id) REFERENCES
        experiment_recommendation_conversion_execution(
            recommendation_conversion_execution_id,
            recommendation_conversion_proposal_id,
            recommendation_conversion_review_decision_id,
            experiment_id) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS conversion_activation_proposal_idx
    ON experiment_recommendation_conversion_activation(
        recommendation_conversion_proposal_id);
CREATE INDEX IF NOT EXISTS conversion_activation_review_idx
    ON experiment_recommendation_conversion_activation(
        recommendation_conversion_review_decision_id);
CREATE INDEX IF NOT EXISTS conversion_activation_hash_idx
    ON experiment_recommendation_conversion_activation(
        activation_identity_hash);

COMMENT ON TABLE experiment_recommendation_conversion_activation IS
    'Append-only explicit activation evidence; activation starts no worker.';
COMMENT ON COLUMN
    experiment_recommendation_conversion_activation.resulting_status IS
    'Scheduler-eligible pending state reached by explicit manual activation.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_activation FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_activation FROM pqxx;
GRANT SELECT ON
    experiment_recommendation_conversion_activation TO pqxx;
GRANT INSERT (
    recommendation_conversion_execution_id,
    recommendation_conversion_proposal_id,
    recommendation_conversion_review_decision_id,
    experiment_id,
    activation_contract_version,
    previous_status,
    previous_phase,
    resulting_status,
    resulting_phase,
    activation_identity_canonical,
    activation_identity_hash
) ON experiment_recommendation_conversion_activation TO pqxx;

DO $$
DECLARE
    sequence_name text := pg_get_serial_sequence(
        'experiment_recommendation_conversion_activation',
        'recommendation_conversion_activation_id');
BEGIN
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
