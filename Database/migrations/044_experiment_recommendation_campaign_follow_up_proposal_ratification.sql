-- Phase 6D: one immutable governance ratification for advancement of one exact
-- approved Phase 6C review into the next separately controlled phase. The
-- ratifier must differ from the Phase 6C reviewer. This evidence grants no
-- Phase 6E capability or operational authority.

CREATE TABLE IF NOT EXISTS
experiment_recommendation_campaign_follow_up_ratification_event (
    recommendation_campaign_follow_up_ratification_event_id bigserial
        PRIMARY KEY CHECK (
            recommendation_campaign_follow_up_ratification_event_id > 0),
    ratification_contract_version integer NOT NULL CHECK (
        ratification_contract_version = 1),
    recommendation_campaign_follow_up_proposal_review_event_id bigint NOT NULL,
    review_contract_version integer NOT NULL CHECK (
        review_contract_version = 1),
    review_identity_canonical text COLLATE "C" NOT NULL CHECK (
        review_identity_canonical <> '' AND
        octet_length(review_identity_canonical) <= 1056768),
    review_identity_hash text NOT NULL CHECK (
        review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    review_decision text NOT NULL CHECK (review_decision = 'approved'),
    reviewer_identity text COLLATE "C" NOT NULL CHECK (
        reviewer_identity ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    recommendation_campaign_follow_up_proposal_id bigint NOT NULL,
    proposal_contract_version integer NOT NULL CHECK (
        proposal_contract_version = 1),
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        proposal_identity_canonical <> '' AND
        octet_length(proposal_identity_canonical) <= 1048576),
    proposal_identity_hash text NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    ratification_authority_role text COLLATE "C" NOT NULL CHECK (
        ratification_authority_role = 'follow_up_governance_ratifier'),
    ratification_decision text NOT NULL CHECK (
        ratification_decision = 'ratified'),
    ratifier_identity text COLLATE "C" NOT NULL CHECK (
        ratifier_identity ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    ratification_basis text COLLATE "C" NOT NULL CHECK (
        ratification_basis <> '' AND octet_length(ratification_basis) <= 4096 AND
        ratification_basis ~ E'[^ \t\r\n]' AND
        translate(ratification_basis, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    ratification_identity_canonical text COLLATE "C" NOT NULL CHECK (
        ratification_identity_canonical <> '' AND
        octet_length(ratification_identity_canonical) <= 2113536),
    ratification_identity_hash text NOT NULL CHECK (
        ratification_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT follow_up_ratification_separation_of_duties_check
        CHECK (ratifier_identity <> reviewer_identity),
    CONSTRAINT follow_up_ratification_review_fk
        FOREIGN KEY (
            recommendation_campaign_follow_up_proposal_review_event_id)
        REFERENCES
            experiment_recommendation_campaign_follow_up_proposal_review_event(
                recommendation_campaign_follow_up_proposal_review_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT follow_up_ratification_proposal_fk
        FOREIGN KEY (recommendation_campaign_follow_up_proposal_id)
        REFERENCES experiment_recommendation_campaign_follow_up_proposal(
            recommendation_campaign_follow_up_proposal_id)
        ON DELETE RESTRICT,
    CONSTRAINT follow_up_ratification_review_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_review_event_id),
    CONSTRAINT follow_up_ratification_proposal_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id)
);

CREATE INDEX IF NOT EXISTS
follow_up_ratification_identity_hash_idx
    ON experiment_recommendation_campaign_follow_up_ratification_event(
        ratification_identity_hash);

CREATE OR REPLACE FUNCTION
enforce_follow_up_ratification_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_follow_up_proposal_review_event
        review
    JOIN experiment_recommendation_campaign_follow_up_proposal proposal
      ON proposal.recommendation_campaign_follow_up_proposal_id =
             review.recommendation_campaign_follow_up_proposal_id
    WHERE review.recommendation_campaign_follow_up_proposal_review_event_id =
              NEW.recommendation_campaign_follow_up_proposal_review_event_id
      AND review.review_contract_version = NEW.review_contract_version
      AND review.review_identity_canonical = NEW.review_identity_canonical
      AND review.review_identity_hash = NEW.review_identity_hash
      AND review.review_decision = 'approved'
      AND NEW.review_decision = review.review_decision
      AND review.reviewer_identity = NEW.reviewer_identity
      AND NEW.ratifier_identity <> review.reviewer_identity
      AND review.recommendation_campaign_follow_up_proposal_id =
              NEW.recommendation_campaign_follow_up_proposal_id
      AND review.proposal_contract_version = NEW.proposal_contract_version
      AND review.proposal_identity_canonical =
              NEW.proposal_identity_canonical
      AND review.proposal_identity_hash = NEW.proposal_identity_hash
      AND proposal.proposal_contract_version = NEW.proposal_contract_version
      AND proposal.proposal_identity_canonical =
              NEW.proposal_identity_canonical
      AND proposal.proposal_identity_hash = NEW.proposal_identity_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'follow-up proposal ratification provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'follow_up_ratification_provenance_check';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE ratification_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_follow_up_ratification_provenance() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        ratification_schema, ratification_schema);
END $$;

DROP TRIGGER IF EXISTS
follow_up_ratification_provenance_trigger
ON experiment_recommendation_campaign_follow_up_ratification_event;
CREATE TRIGGER
follow_up_ratification_provenance_trigger
BEFORE INSERT ON
    experiment_recommendation_campaign_follow_up_ratification_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_follow_up_ratification_provenance();

COMMENT ON TABLE
experiment_recommendation_campaign_follow_up_ratification_event IS
    'Immutable Phase 6D governance ratification of advancement only; never Phase 6E capability, operational authorization, or action.';
COMMENT ON COLUMN
experiment_recommendation_campaign_follow_up_ratification_event.ratification_identity_canonical IS
    'Authoritative byte-exact Phase 6D ratification identity; the tagged hash is only an accelerator.';
COMMENT ON COLUMN
experiment_recommendation_campaign_follow_up_ratification_event.ratification_decision IS
    'Governance ratification of an eligible approved Phase 6C review for entry into the next separately controlled phase; grants no Phase 6E or action authority.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_follow_up_ratification_event
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_follow_up_ratification_provenance()
    FROM PUBLIC, pqxx;

GRANT SELECT ON
    experiment_recommendation_campaign_follow_up_ratification_event
    TO pqxx;
GRANT INSERT (
    ratification_contract_version,
    recommendation_campaign_follow_up_proposal_review_event_id,
    review_contract_version,
    review_identity_canonical,
    review_identity_hash,
    review_decision,
    reviewer_identity,
    recommendation_campaign_follow_up_proposal_id,
    proposal_contract_version,
    proposal_identity_canonical,
    proposal_identity_hash,
    ratification_authority_role,
    ratification_decision,
    ratifier_identity,
    ratification_basis,
    ratification_identity_canonical,
    ratification_identity_hash)
ON experiment_recommendation_campaign_follow_up_ratification_event
TO pqxx;

DO $$
DECLARE sequence_name text;
BEGIN
    sequence_name := pg_get_serial_sequence(
        'experiment_recommendation_campaign_follow_up_ratification_event',
        'recommendation_campaign_follow_up_ratification_event_id');
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC', sequence_name);
    EXECUTE format(
        'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx', sequence_name);
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
END $$;
