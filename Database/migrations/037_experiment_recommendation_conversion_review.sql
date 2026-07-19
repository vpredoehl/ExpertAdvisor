-- Phase 4C Step 3: append-only manual review decisions for exact persisted
-- conversion proposals. Approval is administrative evidence only and does not
-- create, queue, or modify an experiment.

CREATE TABLE IF NOT EXISTS
experiment_recommendation_conversion_review_decision (
    recommendation_conversion_review_decision_id bigserial PRIMARY KEY
        CHECK (recommendation_conversion_review_decision_id > 0),
    recommendation_conversion_proposal_id bigint NOT NULL REFERENCES
        experiment_recommendation_conversion_proposal(
            recommendation_conversion_proposal_id) ON DELETE RESTRICT CHECK (
                recommendation_conversion_proposal_id > 0),
    decision text NOT NULL CONSTRAINT conversion_review_decision_value_check
        CHECK (decision IN ('approve','reject')),
    decision_request_id text COLLATE "C" NOT NULL
        CONSTRAINT conversion_review_request_id_check CHECK (
        octet_length(decision_request_id) BETWEEN 1 AND 128 AND
        decision_request_id ~ '^[A-Za-z0-9][A-Za-z0-9._:-]*$'),
    operator_identity text CONSTRAINT conversion_review_operator_check CHECK (
        operator_identity IS NULL OR (
            operator_identity = btrim(operator_identity, E' \t\n\r\f\v') AND
            operator_identity <> '' AND
            octet_length(operator_identity) <= 200)),
    reason_text text CONSTRAINT conversion_review_reason_check CHECK (
        reason_text IS NULL OR (
            reason_text = btrim(reason_text, E' \t\n\r\f\v') AND
            reason_text <> '' AND octet_length(reason_text) <= 2000)),
    decided_at timestamptz NOT NULL DEFAULT now(),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT conversion_review_proposal_request_uidx UNIQUE (
        recommendation_conversion_proposal_id, decision_request_id)
);

CREATE INDEX IF NOT EXISTS conversion_review_proposal_history_idx
    ON experiment_recommendation_conversion_review_decision(
        recommendation_conversion_proposal_id,
        recommendation_conversion_review_decision_id DESC);
CREATE INDEX IF NOT EXISTS conversion_review_decision_idx
    ON experiment_recommendation_conversion_review_decision(
        decision, recommendation_conversion_review_decision_id DESC);

COMMENT ON TABLE experiment_recommendation_conversion_review_decision IS
    'Append-only manual decisions for conversion proposals; never execution.';
COMMENT ON COLUMN
    experiment_recommendation_conversion_review_decision.decision_request_id IS
    'Caller-visible idempotency token, unique within one exact proposal.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_review_decision FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_conversion_review_decision FROM pqxx;
GRANT SELECT ON
    experiment_recommendation_conversion_review_decision TO pqxx;
GRANT INSERT (
    recommendation_conversion_proposal_id,
    decision,
    decision_request_id,
    operator_identity,
    reason_text
) ON experiment_recommendation_conversion_review_decision TO pqxx;

DO $$
DECLARE
    sequence_name text := pg_get_serial_sequence(
        'experiment_recommendation_conversion_review_decision',
        'recommendation_conversion_review_decision_id');
BEGIN
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;

-- Restate the Step 2 audit boundary explicitly for the runtime role.
REVOKE UPDATE, DELETE, TRUNCATE ON
    experiment_recommendation_conversion_proposal FROM pqxx;
REVOKE UPDATE, DELETE, TRUNCATE ON
    experiment_recommendation_conversion_proposal FROM PUBLIC;
