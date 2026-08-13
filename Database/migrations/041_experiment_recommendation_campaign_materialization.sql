-- Phase 4D Step 4: one immutable materialization of an approved campaign into
-- ordered Phase 4C conversion-proposal links. This migration creates no
-- experiment and grants no experiment, scheduler, or worker capability.

-- These narrow unique indexes support restrictive composite foreign keys that
-- prove copied provenance belongs to the exact immutable source rows.
CREATE UNIQUE INDEX IF NOT EXISTS recommendation_campaign_approval_id_decision_uidx
    ON experiment_recommendation_campaign_approval(
        recommendation_campaign_approval_id, decision);
CREATE UNIQUE INDEX IF NOT EXISTS recommendation_ranking_member_provenance_uidx
    ON experiment_recommendation_ranking_member(
        recommendation_ranking_member_id, recommendation_id,
        source_experiment_id);
CREATE UNIQUE INDEX IF NOT EXISTS recommendation_conversion_proposal_provenance_uidx
    ON experiment_recommendation_conversion_proposal(
        recommendation_conversion_proposal_id, recommendation_id,
        source_experiment_id);

CREATE TABLE IF NOT EXISTS experiment_recommendation_campaign_materialization (
    recommendation_campaign_materialization_id bigserial PRIMARY KEY CHECK (
        recommendation_campaign_materialization_id > 0),
    recommendation_campaign_approval_id bigint NOT NULL UNIQUE CHECK (
        recommendation_campaign_approval_id > 0),
    materialization_contract_version integer NOT NULL CHECK (
        materialization_contract_version = 1),
    approval_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(approval_identity_canonical) <> '' AND
        octet_length(approval_identity_canonical) <= 67108864),
    approval_identity_hash text NOT NULL CHECK (
        approval_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_ranking_snapshot_id bigint NOT NULL REFERENCES
        experiment_recommendation_ranking_snapshot(
            recommendation_ranking_snapshot_id) ON DELETE RESTRICT CHECK (
                recommendation_ranking_snapshot_id > 0),
    ranking_snapshot_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(ranking_snapshot_identity_canonical) <> '' AND
        octet_length(ranking_snapshot_identity_canonical) <= 4194304),
    ranking_snapshot_identity_hash text NOT NULL CHECK (
        ranking_snapshot_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    planning_policy_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(planning_policy_canonical) <> '' AND
        octet_length(planning_policy_canonical) <= 1048576),
    planning_policy_hash text NOT NULL CHECK (
        planning_policy_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    planning_scope_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(planning_scope_canonical) <> '' AND
        octet_length(planning_scope_canonical) <= 65536),
    campaign_plan_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(campaign_plan_identity_canonical) <> '' AND
        octet_length(campaign_plan_identity_canonical) <= 16777216),
    campaign_plan_identity_hash text NOT NULL CHECK (
        campaign_plan_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    campaign_review_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(campaign_review_identity_canonical) <> '' AND
        octet_length(campaign_review_identity_canonical) <= 33554432),
    campaign_review_identity_hash text NOT NULL CHECK (
        campaign_review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    approval_decision text NOT NULL CHECK (approval_decision = 'approved'),
    approval_reviewer_identity text COLLATE "C" NOT NULL CHECK (
        approval_reviewer_identity <> '' AND
        octet_length(approval_reviewer_identity) <= 200),
    approval_reason_text text NOT NULL CHECK (
        approval_reason_text <> '' AND
        octet_length(approval_reason_text) <= 2000),
    materialized_by text COLLATE "C" NOT NULL CHECK (
        materialized_by = btrim(materialized_by, E' \t\n\r\f\v') AND
        materialized_by <> '' AND octet_length(materialized_by) <= 200 AND
        materialized_by !~ '[[:cntrl:]]'),
    materialization_reason_text text NOT NULL CHECK (
        materialization_reason_text = btrim(
            materialization_reason_text, E' \t\n\r\f\v') AND
        materialization_reason_text <> '' AND
        octet_length(materialization_reason_text) <= 2000 AND
        materialization_reason_text !~ '[[:cntrl:]]'),
    selected_member_count integer NOT NULL CHECK (selected_member_count > 0),
    initially_created_proposal_count integer NOT NULL CHECK (
        initially_created_proposal_count >= 0),
    initially_reused_proposal_count integer NOT NULL CHECK (
        initially_reused_proposal_count >= 0),
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(materialization_identity_canonical) <> '' AND
        octet_length(materialization_identity_canonical) <= 67108864),
    materialization_identity_hash text NOT NULL CHECK (
        materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_materialization_approval_fkey
        FOREIGN KEY (recommendation_campaign_approval_id, approval_decision)
        REFERENCES experiment_recommendation_campaign_approval(
            recommendation_campaign_approval_id, decision)
        ON DELETE RESTRICT,
    CONSTRAINT recommendation_campaign_materialization_counts_check CHECK (
        selected_member_count = initially_created_proposal_count +
            initially_reused_proposal_count)
);

CREATE TABLE IF NOT EXISTS
experiment_recommendation_campaign_materialization_member (
    recommendation_campaign_materialization_member_id bigserial PRIMARY KEY
        CHECK (recommendation_campaign_materialization_member_id > 0),
    recommendation_campaign_materialization_id bigint NOT NULL REFERENCES
        experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id) ON DELETE RESTRICT,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    recommendation_ranking_member_id bigint NOT NULL,
    recommendation_id bigint NOT NULL REFERENCES
        experiment_recommendation(recommendation_id) ON DELETE RESTRICT,
    source_experiment_id bigint NOT NULL REFERENCES
        experiment(experiment_id) ON DELETE RESTRICT,
    ranking_position integer NOT NULL CHECK (ranking_position > 0),
    selected_member_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(selected_member_identity_canonical) <> '' AND
        octet_length(selected_member_identity_canonical) <= 4194304),
    selected_member_identity_hash text NOT NULL CHECK (
        selected_member_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_conversion_proposal_id bigint NOT NULL,
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        btrim(proposal_identity_canonical) <> '' AND
        octet_length(proposal_identity_canonical) <= 1048576),
    proposal_identity_hash text NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_campaign_materialization_member_rank_fkey
        FOREIGN KEY (recommendation_ranking_member_id, recommendation_id,
                     source_experiment_id)
        REFERENCES experiment_recommendation_ranking_member(
            recommendation_ranking_member_id, recommendation_id,
            source_experiment_id) ON DELETE RESTRICT,
    CONSTRAINT recommendation_campaign_materialization_member_proposal_fkey
        FOREIGN KEY (recommendation_conversion_proposal_id,
                     recommendation_id, source_experiment_id)
        REFERENCES experiment_recommendation_conversion_proposal(
            recommendation_conversion_proposal_id, recommendation_id,
            source_experiment_id) ON DELETE RESTRICT,
    CONSTRAINT recommendation_campaign_materialization_member_ordinal_uidx
        UNIQUE (recommendation_campaign_materialization_id, member_ordinal),
    CONSTRAINT recommendation_campaign_materialization_member_rank_uidx
        UNIQUE (recommendation_campaign_materialization_id,
                recommendation_ranking_member_id),
    CONSTRAINT recommendation_campaign_materialization_member_proposal_uidx
        UNIQUE (recommendation_campaign_materialization_id,
                recommendation_conversion_proposal_id)
);

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_materialization_manifest_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_approval approval
    WHERE approval.recommendation_campaign_approval_id =
              NEW.recommendation_campaign_approval_id
      AND approval.decision = 'approved'
      AND NEW.approval_decision = approval.decision
      AND NEW.approval_identity_canonical =
              approval.approval_identity_canonical
      AND NEW.approval_identity_hash = approval.approval_identity_hash
      AND NEW.recommendation_ranking_snapshot_id =
              approval.recommendation_ranking_snapshot_id
      AND NEW.ranking_snapshot_identity_canonical =
              approval.ranking_snapshot_identity_canonical
      AND NEW.ranking_snapshot_identity_hash =
              approval.ranking_snapshot_identity_hash
      AND NEW.planning_policy_canonical = approval.planning_policy_canonical
      AND NEW.planning_policy_hash = approval.planning_policy_hash
      AND NEW.planning_scope_canonical = approval.planning_scope_canonical
      AND NEW.campaign_plan_identity_canonical =
              approval.campaign_plan_identity_canonical
      AND NEW.campaign_plan_identity_hash =
              approval.campaign_plan_identity_hash
      AND NEW.campaign_review_identity_canonical =
              approval.campaign_review_identity_canonical
      AND NEW.campaign_review_identity_hash =
              approval.campaign_review_identity_hash
      AND NEW.approval_reviewer_identity = approval.reviewer_identity
      AND NEW.approval_reason_text = approval.reason_text
      AND NEW.selected_member_count = approval.selected_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign materialization approval provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_materialization_authority_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_materialization_member_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_materialization materialization
    JOIN experiment_recommendation_ranking_member ranking_member
      ON ranking_member.recommendation_ranking_member_id =
             NEW.recommendation_ranking_member_id
     AND ranking_member.recommendation_ranking_snapshot_id =
             materialization.recommendation_ranking_snapshot_id
     AND ranking_member.recommendation_id = NEW.recommendation_id
     AND ranking_member.source_experiment_id = NEW.source_experiment_id
     AND ranking_member.global_ordinal = NEW.ranking_position
    JOIN experiment_recommendation_conversion_proposal proposal
      ON proposal.recommendation_conversion_proposal_id =
             NEW.recommendation_conversion_proposal_id
     AND proposal.recommendation_id = NEW.recommendation_id
     AND proposal.source_experiment_id = NEW.source_experiment_id
     AND proposal.conversion_identity_canonical =
             NEW.proposal_identity_canonical
     AND proposal.conversion_identity_hash = NEW.proposal_identity_hash
    WHERE materialization.recommendation_campaign_materialization_id =
              NEW.recommendation_campaign_materialization_id
      AND NEW.member_ordinal <= materialization.selected_member_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign materialization member provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_materialization_member_authority_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_recommendation_campaign_materialization_complete()
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
    FROM experiment_recommendation_campaign_materialization_member
    WHERE recommendation_campaign_materialization_id =
          NEW.recommendation_campaign_materialization_id;
    IF persisted_count <> NEW.selected_member_count
       OR minimum_ordinal <> 1
       OR maximum_ordinal <> NEW.selected_member_count THEN
        RAISE EXCEPTION 'campaign materialization member set incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'recommendation_campaign_materialization_complete_check';
    END IF;
    RETURN NULL;
END;
$$;

DO $$
DECLARE materialization_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_materialization_manifest_insert() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        materialization_schema, materialization_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_materialization_member_insert() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        materialization_schema, materialization_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_campaign_materialization_complete() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        materialization_schema, materialization_schema);
END $$;

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_materialization_manifest_insert_trigger
ON experiment_recommendation_campaign_materialization;
CREATE TRIGGER
enforce_recommendation_campaign_materialization_manifest_insert_trigger
BEFORE INSERT ON experiment_recommendation_campaign_materialization
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_materialization_manifest_insert();

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_materialization_member_insert_trigger
ON experiment_recommendation_campaign_materialization_member;
CREATE TRIGGER
enforce_recommendation_campaign_materialization_member_insert_trigger
BEFORE INSERT ON experiment_recommendation_campaign_materialization_member
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_materialization_member_insert();

DROP TRIGGER IF EXISTS
enforce_recommendation_campaign_materialization_complete_trigger
ON experiment_recommendation_campaign_materialization;
CREATE CONSTRAINT TRIGGER
enforce_recommendation_campaign_materialization_complete_trigger
AFTER INSERT ON experiment_recommendation_campaign_materialization
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_recommendation_campaign_materialization_complete();

COMMENT ON TABLE experiment_recommendation_campaign_materialization IS
    'Immutable all-or-nothing linkage from one approved campaign to Phase 4C proposals; never experiment execution.';
COMMENT ON TABLE experiment_recommendation_campaign_materialization_member IS
    'Immutable ordered selected-member linkage to an existing Phase 4C conversion proposal.';

REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_materialization FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_materialization_member FROM PUBLIC;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_materialization FROM pqxx;
REVOKE ALL PRIVILEGES ON
    experiment_recommendation_campaign_materialization_member FROM pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_materialization_manifest_insert()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_materialization_member_insert()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_recommendation_campaign_materialization_complete()
    FROM PUBLIC, pqxx;
GRANT SELECT ON experiment_recommendation_campaign_materialization TO pqxx;
GRANT SELECT ON experiment_recommendation_campaign_materialization_member TO pqxx;
GRANT INSERT (
    recommendation_campaign_approval_id,materialization_contract_version,
    approval_identity_canonical,approval_identity_hash,
    recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,planning_policy_canonical,
    planning_policy_hash,planning_scope_canonical,
    campaign_plan_identity_canonical,campaign_plan_identity_hash,
    campaign_review_identity_canonical,campaign_review_identity_hash,
    approval_decision,approval_reviewer_identity,approval_reason_text,
    materialized_by,materialization_reason_text,selected_member_count,
    initially_created_proposal_count,initially_reused_proposal_count,
    materialization_identity_canonical,materialization_identity_hash)
ON experiment_recommendation_campaign_materialization TO pqxx;
GRANT INSERT (
    recommendation_campaign_materialization_id,member_ordinal,
    recommendation_ranking_member_id,recommendation_id,source_experiment_id,
    ranking_position,selected_member_identity_canonical,
    selected_member_identity_hash,recommendation_conversion_proposal_id,
    proposal_identity_canonical,proposal_identity_hash)
ON experiment_recommendation_campaign_materialization_member TO pqxx;

DO $$
DECLARE sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence(
            'experiment_recommendation_campaign_materialization',
            'recommendation_campaign_materialization_id'),
        pg_get_serial_sequence(
            'experiment_recommendation_campaign_materialization_member',
            'recommendation_campaign_materialization_member_id')]
    LOOP
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC',
                       sequence_name);
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx',
                       sequence_name);
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO pqxx', sequence_name);
    END LOOP;
END $$;
