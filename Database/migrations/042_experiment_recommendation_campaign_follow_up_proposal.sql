-- Phase 6B: immutable persistence for exact Phase 6A advisory follow-up
-- proposals. These tables add no approval, activation, execution, queue,
-- scheduler, worker, experiment, or follow-up authorization capability.

CREATE TABLE IF NOT EXISTS
experiment_recommendation_campaign_follow_up_proposal (
    recommendation_campaign_follow_up_proposal_id bigserial PRIMARY KEY CHECK (
        recommendation_campaign_follow_up_proposal_id > 0),
    proposal_contract_version integer NOT NULL CHECK (
        proposal_contract_version = 1),
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        proposal_identity_canonical <> '' AND
        octet_length(proposal_identity_canonical) <= 1048576),
    proposal_identity_hash text NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    proposal_identity_hash_collision_ordinal integer NOT NULL CHECK (
        proposal_identity_hash_collision_ordinal >= 0),
    assessment_contract_version integer NOT NULL CHECK (
        assessment_contract_version = 2),
    assessment_identity_canonical text COLLATE "C" NOT NULL CHECK (
        assessment_identity_canonical <> '' AND
        octet_length(assessment_identity_canonical) <= 1048576),
    assessment_identity_hash text NOT NULL CHECK (
        assessment_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    policy_contract_version integer NOT NULL CHECK (
        policy_contract_version = 1),
    policy_identity_canonical text COLLATE "C" NOT NULL CHECK (
        policy_identity_canonical <> '' AND
        octet_length(policy_identity_canonical) <= 1048576),
    policy_identity_hash text NOT NULL CHECK (
        policy_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    policy_decision_contract_version integer NOT NULL CHECK (
        policy_decision_contract_version = 1),
    policy_decision_identity_canonical text COLLATE "C" NOT NULL CHECK (
        policy_decision_identity_canonical <> '' AND
        octet_length(policy_decision_identity_canonical) <= 1048576),
    policy_decision_identity_hash text NOT NULL CHECK (
        policy_decision_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    campaign_approval_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_approval(
            recommendation_campaign_approval_id) ON DELETE RESTRICT CHECK (
                campaign_approval_id > 0),
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 1048576),
    campaign_identity_hash text NOT NULL CHECK (
        campaign_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    materialization_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id) ON DELETE RESTRICT
        CHECK (materialization_id > 0),
    materialization_campaign_approval_id bigint NOT NULL CHECK (
        materialization_campaign_approval_id > 0),
    materialization_campaign_identity_hash text NOT NULL CHECK (
        materialization_campaign_identity_hash ~
            '^fnv1a64:[0-9a-f]{16}$'),
    materialization_contract_version integer NOT NULL CHECK (
        materialization_contract_version = 1),
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        materialization_identity_canonical <> '' AND
        octet_length(materialization_identity_canonical) <= 1048576),
    materialization_identity_hash text NOT NULL CHECK (
        materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    evidence_sufficiency text NOT NULL CHECK (
        evidence_sufficiency = 'sufficient'),
    campaign_interpretation text NOT NULL CHECK (
        campaign_interpretation = 'favorable'),
    follow_up_eligibility text NOT NULL CHECK (
        follow_up_eligibility = 'eligible_for_operator_review'),
    follow_up_authorized boolean NOT NULL CHECK (NOT follow_up_authorized),
    proposal_reason text NOT NULL CHECK (
        proposal_reason = 'eligible_favorable_policy_decision'),
    member_count integer NOT NULL CHECK (
        member_count > 0 AND member_count <= 1000),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_follow_up_proposal_hash_ordinal_uidx
        UNIQUE (proposal_identity_hash,
                proposal_identity_hash_collision_ordinal),
    CONSTRAINT recommendation_campaign_follow_up_proposal_alignment_check
        CHECK (
            materialization_campaign_approval_id = campaign_approval_id AND
            materialization_campaign_identity_hash =
                campaign_identity_hash)
);

CREATE TABLE IF NOT EXISTS
experiment_recommendation_campaign_follow_up_proposal_member (
    recommendation_campaign_follow_up_proposal_member_id bigserial PRIMARY KEY
        CHECK (recommendation_campaign_follow_up_proposal_member_id > 0),
    recommendation_campaign_follow_up_proposal_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_follow_up_proposal(
            recommendation_campaign_follow_up_proposal_id)
        ON DELETE RESTRICT,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    materialization_member_id bigint NOT NULL CHECK (
        materialization_member_id > 0),
    ranking_member_id bigint NOT NULL CHECK (ranking_member_id > 0),
    recommendation_id bigint NOT NULL CHECK (recommendation_id > 0),
    source_experiment_id bigint NOT NULL CHECK (source_experiment_id > 0),
    conversion_proposal_id bigint NOT NULL CHECK (
        conversion_proposal_id > 0),
    expected_experiment_id bigint CHECK (expected_experiment_id > 0),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_follow_up_member_ordinal_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id,
                member_ordinal),
    CONSTRAINT recommendation_campaign_follow_up_member_materialized_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id,
                materialization_member_id),
    CONSTRAINT recommendation_campaign_follow_up_member_ranking_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id,
                ranking_member_id),
    CONSTRAINT recommendation_campaign_follow_up_member_recommendation_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id,
                recommendation_id),
    CONSTRAINT recommendation_campaign_follow_up_member_conversion_uidx
        UNIQUE (recommendation_campaign_follow_up_proposal_id,
                conversion_proposal_id)
);

CREATE INDEX IF NOT EXISTS
recommendation_campaign_follow_up_proposal_identity_hash_idx
    ON experiment_recommendation_campaign_follow_up_proposal(
        proposal_identity_hash);
-- Canonical text is authoritative. A PostgreSQL hash index keeps exact
-- equality lookup practical without a btree tuple-size dependency.
CREATE INDEX IF NOT EXISTS
recommendation_campaign_follow_up_proposal_identity_canonical_idx
    ON experiment_recommendation_campaign_follow_up_proposal USING hash(
        proposal_identity_canonical);
CREATE INDEX IF NOT EXISTS
recommendation_campaign_follow_up_proposal_campaign_idx
    ON experiment_recommendation_campaign_follow_up_proposal(
        campaign_approval_id,
        recommendation_campaign_follow_up_proposal_id);
CREATE INDEX IF NOT EXISTS
recommendation_campaign_follow_up_proposal_materialization_idx
    ON experiment_recommendation_campaign_follow_up_proposal(
        materialization_id,
        recommendation_campaign_follow_up_proposal_id);

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_follow_up_proposal_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_approval approval
    JOIN experiment_recommendation_campaign_materialization materialization
      ON materialization.recommendation_campaign_approval_id =
             approval.recommendation_campaign_approval_id
    WHERE approval.recommendation_campaign_approval_id =
              NEW.campaign_approval_id
      AND approval.approval_identity_canonical =
              NEW.campaign_identity_canonical
      AND approval.approval_identity_hash = NEW.campaign_identity_hash
      AND materialization.recommendation_campaign_materialization_id =
              NEW.materialization_id
      AND materialization.materialization_contract_version =
              NEW.materialization_contract_version
      AND materialization.approval_identity_hash =
              NEW.materialization_campaign_identity_hash
      AND materialization.selected_member_count = NEW.member_count
      AND materialization.materialization_identity_canonical =
              NEW.materialization_identity_canonical
      AND materialization.materialization_identity_hash =
              NEW.materialization_identity_hash;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'follow-up proposal upstream provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_follow_up_proposal_provenance_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_follow_up_proposal_member_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_follow_up_proposal proposal
    JOIN experiment_recommendation_campaign_materialization_member member
      ON member.recommendation_campaign_materialization_id =
             proposal.materialization_id
    WHERE proposal.recommendation_campaign_follow_up_proposal_id =
              NEW.recommendation_campaign_follow_up_proposal_id
      AND member.recommendation_campaign_materialization_member_id =
              NEW.materialization_member_id
      AND member.member_ordinal = NEW.member_ordinal
      AND member.recommendation_ranking_member_id = NEW.ranking_member_id
      AND member.recommendation_id = NEW.recommendation_id
      AND member.source_experiment_id = NEW.source_experiment_id
      AND member.recommendation_conversion_proposal_id =
              NEW.conversion_proposal_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'follow-up proposal member provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_follow_up_proposal_member_provenance_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_follow_up_proposal_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    persisted_count integer;
    minimum_ordinal integer;
    maximum_ordinal integer;
BEGIN
    SELECT count(*)::integer, min(member_ordinal), max(member_ordinal)
    INTO persisted_count, minimum_ordinal, maximum_ordinal
    FROM experiment_recommendation_campaign_follow_up_proposal_member
    WHERE recommendation_campaign_follow_up_proposal_id =
          NEW.recommendation_campaign_follow_up_proposal_id;
    IF persisted_count <> NEW.member_count
       OR minimum_ordinal <> 1
       OR maximum_ordinal <> NEW.member_count THEN
        RAISE EXCEPTION 'follow-up proposal member set incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_follow_up_proposal_complete_check';
    END IF;
    RETURN NULL;
END;
$$;

DO $$
DECLARE proposal_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_follow_up_proposal_provenance() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        proposal_schema, proposal_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_follow_up_proposal_member_provenance() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        proposal_schema, proposal_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_follow_up_proposal_complete() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        proposal_schema, proposal_schema);
END $$;

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_follow_up_proposal_provenance_trigger
ON experiment_recommendation_campaign_follow_up_proposal;
CREATE TRIGGER
enforce_recommendation_campaign_follow_up_proposal_provenance_trigger
BEFORE INSERT ON experiment_recommendation_campaign_follow_up_proposal
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_provenance();

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_follow_up_proposal_member_provenance_trigger
ON experiment_recommendation_campaign_follow_up_proposal_member;
CREATE TRIGGER
enforce_recommendation_campaign_follow_up_proposal_member_provenance_trigger
BEFORE INSERT ON experiment_recommendation_campaign_follow_up_proposal_member
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_member_provenance();

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_follow_up_proposal_complete_trigger
ON experiment_recommendation_campaign_follow_up_proposal;
CREATE CONSTRAINT TRIGGER
enforce_recommendation_campaign_follow_up_proposal_complete_trigger
AFTER INSERT ON experiment_recommendation_campaign_follow_up_proposal
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_complete();

COMMENT ON TABLE
experiment_recommendation_campaign_follow_up_proposal IS
    'Immutable exact Phase 6A advisory proposal for read-only operator preview; never approval or authorization.';
COMMENT ON TABLE
experiment_recommendation_campaign_follow_up_proposal_member IS
    'Immutable ordered Phase 6A follow-up proposal membership; never a queue or schedule.';
COMMENT ON COLUMN
experiment_recommendation_campaign_follow_up_proposal.proposal_identity_canonical IS
    'Authoritative byte-exact Phase 6A identity; the tagged hash is only an accelerator.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_follow_up_proposal FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_follow_up_proposal_member
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_provenance()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_member_provenance()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_follow_up_proposal_complete()
    FROM PUBLIC, pqxx;

GRANT SELECT ON
    experiment_recommendation_campaign_follow_up_proposal TO pqxx;
GRANT SELECT ON
    experiment_recommendation_campaign_follow_up_proposal_member TO pqxx;
GRANT INSERT (
    proposal_contract_version,proposal_identity_canonical,
    proposal_identity_hash,proposal_identity_hash_collision_ordinal,
    assessment_contract_version,assessment_identity_canonical,
    assessment_identity_hash,policy_contract_version,
    policy_identity_canonical,policy_identity_hash,
    policy_decision_contract_version,policy_decision_identity_canonical,
    policy_decision_identity_hash,campaign_approval_id,
    campaign_identity_canonical,campaign_identity_hash,materialization_id,
    materialization_campaign_approval_id,
    materialization_campaign_identity_hash,materialization_contract_version,
    materialization_identity_canonical,materialization_identity_hash,
    evidence_sufficiency,campaign_interpretation,follow_up_eligibility,
    follow_up_authorized,proposal_reason,member_count)
ON experiment_recommendation_campaign_follow_up_proposal TO pqxx;
GRANT INSERT (
    recommendation_campaign_follow_up_proposal_id,member_ordinal,
    materialization_member_id,ranking_member_id,recommendation_id,
    source_experiment_id,conversion_proposal_id,expected_experiment_id)
ON experiment_recommendation_campaign_follow_up_proposal_member TO pqxx;

DO $$
DECLARE sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
            'experiment_recommendation_campaign_follow_up_proposal',
            'recommendation_campaign_follow_up_proposal_id'),
        pg_get_serial_sequence(
            'experiment_recommendation_campaign_follow_up_proposal_member',
            'recommendation_campaign_follow_up_proposal_member_id')]
    LOOP
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC',
                       sequence_name);
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx',
                       sequence_name);
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
    END LOOP;
END $$;
