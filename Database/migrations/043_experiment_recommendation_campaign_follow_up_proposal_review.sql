-- Phase 6C: append-only administrative operator-review decisions for exact
-- persisted Phase 6B proposals. Approval here grants no activation,
-- execution, follow-up, queue, scheduler, worker, or experiment authority.

CREATE TABLE IF NOT EXISTS
experiment_recommendation_campaign_follow_up_proposal_review_event (
    recommendation_campaign_follow_up_proposal_review_event_id bigserial
        PRIMARY KEY CHECK (
            recommendation_campaign_follow_up_proposal_review_event_id > 0),
    review_contract_version integer NOT NULL CHECK (
        review_contract_version = 1),
    recommendation_campaign_follow_up_proposal_id bigint NOT NULL,
    proposal_contract_version integer NOT NULL CHECK (
        proposal_contract_version = 1),
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        proposal_identity_canonical <> '' AND
        octet_length(proposal_identity_canonical) <= 1048576),
    proposal_identity_hash text NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    review_decision text NOT NULL CHECK (
        review_decision IN ('approved', 'rejected')),
    reviewer_identity text COLLATE "C" NOT NULL CHECK (
        reviewer_identity ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    reason_text text COLLATE "C" NOT NULL CHECK (
        reason_text <> '' AND octet_length(reason_text) <= 4096),
    review_identity_canonical text COLLATE "C" NOT NULL CHECK (
        review_identity_canonical <> '' AND
        octet_length(review_identity_canonical) <= 1056768),
    review_identity_hash text NOT NULL CHECK (
        review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_follow_up_proposal_review_proposal_fk
        FOREIGN KEY (recommendation_campaign_follow_up_proposal_id)
        REFERENCES experiment_recommendation_campaign_follow_up_proposal(
            recommendation_campaign_follow_up_proposal_id)
        ON DELETE RESTRICT,
    CONSTRAINT recommendation_campaign_follow_up_proposal_review_once_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id)
);

CREATE INDEX IF NOT EXISTS
recommendation_campaign_follow_up_proposal_review_identity_hash_idx
    ON experiment_recommendation_campaign_follow_up_proposal_review_event(
        review_identity_hash);

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_follow_up_proposal_review_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_follow_up_proposal proposal
    WHERE proposal.recommendation_campaign_follow_up_proposal_id =
              NEW.recommendation_campaign_follow_up_proposal_id
      AND proposal.proposal_contract_version =
              NEW.proposal_contract_version
      AND proposal.proposal_identity_canonical =
              NEW.proposal_identity_canonical
      AND proposal.proposal_identity_hash = NEW.proposal_identity_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'follow-up proposal review provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_follow_up_proposal_review_provenance_check';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE review_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_follow_up_proposal_review_provenance() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        review_schema, review_schema);
END $$;

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_follow_up_proposal_review_provenance_trigger
ON experiment_recommendation_campaign_follow_up_proposal_review_event;
CREATE TRIGGER
enforce_recommendation_campaign_follow_up_proposal_review_provenance_trigger
BEFORE INSERT ON
    experiment_recommendation_campaign_follow_up_proposal_review_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_review_provenance();

COMMENT ON TABLE
experiment_recommendation_campaign_follow_up_proposal_review_event IS
    'Immutable Phase 6C administrative review only; never activation, execution, scheduling, experiment mutation, or follow-up authorization.';
COMMENT ON COLUMN
experiment_recommendation_campaign_follow_up_proposal_review_event.review_identity_canonical IS
    'Authoritative byte-exact Phase 6C review identity; the tagged hash is only an accelerator.';
COMMENT ON COLUMN
experiment_recommendation_campaign_follow_up_proposal_review_event.review_decision IS
    'Administrative approved or rejected outcome only; approval grants no action authority.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_follow_up_proposal_review_event
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_review_provenance()
    FROM PUBLIC, pqxx;

GRANT SELECT ON
    experiment_recommendation_campaign_follow_up_proposal_review_event
    TO pqxx;
GRANT INSERT (
    review_contract_version,
    recommendation_campaign_follow_up_proposal_id,
    proposal_contract_version,
    proposal_identity_canonical,
    proposal_identity_hash,
    review_decision,
    reviewer_identity,
    reason_text,
    review_identity_canonical,
    review_identity_hash)
ON experiment_recommendation_campaign_follow_up_proposal_review_event
TO pqxx;

DO $$
DECLARE sequence_name text;
BEGIN
    sequence_name := pg_get_serial_sequence(
        'experiment_recommendation_campaign_follow_up_proposal_review_event',
        'recommendation_campaign_follow_up_proposal_review_event_id');
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
