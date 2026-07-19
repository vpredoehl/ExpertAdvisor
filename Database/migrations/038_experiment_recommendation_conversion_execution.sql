-- Phase 4C Step 4: one explicit, manual proposal-to-experiment conversion.
-- The created experiment is paused and is neither queued nor scheduler-owned.

-- Enables a restrictive composite foreign key proving that the exact review
-- decision referenced by an execution belongs to the same proposal.
CREATE UNIQUE INDEX IF NOT EXISTS conversion_review_decision_proposal_action_uidx
    ON experiment_recommendation_conversion_review_decision(
        recommendation_conversion_review_decision_id,
        recommendation_conversion_proposal_id,
        decision);

CREATE TABLE IF NOT EXISTS
experiment_recommendation_conversion_execution (
    recommendation_conversion_execution_id bigserial PRIMARY KEY CHECK (
        recommendation_conversion_execution_id > 0),
    recommendation_conversion_proposal_id bigint NOT NULL UNIQUE CHECK (
        recommendation_conversion_proposal_id > 0),
    recommendation_conversion_review_decision_id bigint NOT NULL CHECK (
        recommendation_conversion_review_decision_id > 0),
    experiment_id bigint NOT NULL UNIQUE REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT CHECK (experiment_id > 0),
    execution_contract_version integer NOT NULL CHECK (
        execution_contract_version = 1),
    authorization_decision text NOT NULL CHECK (
        authorization_decision = 'approve'),
    execution_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(execution_identity_canonical) <> '' AND
        octet_length(execution_identity_canonical) <= 1048576),
    execution_identity_hash text NOT NULL CHECK (
        btrim(execution_identity_hash) <> '' AND
        octet_length(execution_identity_hash) <= 256),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT conversion_execution_proposal_fkey FOREIGN KEY (
        recommendation_conversion_proposal_id) REFERENCES
        experiment_recommendation_conversion_proposal(
            recommendation_conversion_proposal_id) ON DELETE RESTRICT,
    CONSTRAINT conversion_execution_review_proposal_fkey FOREIGN KEY (
        recommendation_conversion_review_decision_id,
        recommendation_conversion_proposal_id,
        authorization_decision) REFERENCES
        experiment_recommendation_conversion_review_decision(
            recommendation_conversion_review_decision_id,
            recommendation_conversion_proposal_id,
            decision) ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS conversion_execution_review_idx
    ON experiment_recommendation_conversion_execution(
        recommendation_conversion_review_decision_id);
CREATE INDEX IF NOT EXISTS conversion_execution_hash_idx
    ON experiment_recommendation_conversion_execution(
        execution_identity_hash);

COMMENT ON TABLE experiment_recommendation_conversion_execution IS
    'Immutable manual proposal-to-paused-experiment conversion evidence.';
COMMENT ON COLUMN
    experiment_recommendation_conversion_execution.experiment_id IS
    'Created paused experiment; conversion does not queue or execute it.';
COMMENT ON COLUMN
    experiment_recommendation_conversion_execution.authorization_decision IS
    'Exact approving review action protected by the composite foreign key.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_execution FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_execution FROM pqxx;
GRANT SELECT ON
    experiment_recommendation_conversion_execution TO pqxx;
GRANT INSERT (
    recommendation_conversion_proposal_id,
    recommendation_conversion_review_decision_id,
    experiment_id,
    execution_contract_version,
    authorization_decision,
    execution_identity_canonical,
    execution_identity_hash
) ON experiment_recommendation_conversion_execution TO pqxx;

DO $$
DECLARE
    sequence_name text := pg_get_serial_sequence(
        'experiment_recommendation_conversion_execution',
        'recommendation_conversion_execution_id');
BEGIN
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
