-- Campaign Operations Phase 1 foundation.
--
-- This additive migration establishes only the immutable operational-campaign,
-- optional governance-provenance, serialized authorization-evidence, and audit
-- persistence boundary. It creates no budget, reservation, request, dispatch,
-- cancellation, completion, lifecycle, scheduler, worker, or CLI behavior.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_owner') THEN
        CREATE ROLE campaign_operations_owner NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_campaign_creator') THEN
        CREATE ROLE campaign_operations_campaign_creator NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_authorizer') THEN
        CREATE ROLE campaign_operations_authorizer NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_auditor') THEN
        CREATE ROLE campaign_operations_auditor NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_reader') THEN
        CREATE ROLE campaign_operations_reader NOLOGIN;
    END IF;
END $$;

ALTER ROLE campaign_operations_owner
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_campaign_creator
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_authorizer
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_auditor
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_reader
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

DO $$
DECLARE capability_role text;
BEGIN
    FOREACH capability_role IN ARRAY ARRAY[
        'campaign_operations_campaign_creator',
        'campaign_operations_authorizer',
        'campaign_operations_auditor',
        'campaign_operations_reader',
        'campaign_operations_owner']
    LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_auth_members
            WHERE roleid = to_regrole(capability_role)
              AND member = to_regrole('pqxx')) THEN
            EXECUTE format('REVOKE %I FROM pqxx', capability_role);
        END IF;
        IF pg_has_role('pqxx', capability_role, 'MEMBER') THEN
            RAISE EXCEPTION
                'pqxx retains indirect membership in Campaign Operations role %',
                capability_role;
        END IF;
    END LOOP;
END $$;

CREATE TABLE IF NOT EXISTS campaign_operations_campaign (
    operational_campaign_id bigserial PRIMARY KEY CHECK (
        operational_campaign_id > 0),
    recommendation_campaign_materialization_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL CHECK (
        materialization_contract_version = 1),
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        materialization_identity_canonical <> '' AND
        octet_length(materialization_identity_canonical) <= 134217728),
    materialization_identity_hash text COLLATE "C" NOT NULL CHECK (
        materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    materialization_member_count integer NOT NULL CHECK (
        materialization_member_count > 0),
    origin_kind text COLLATE "C" NOT NULL CHECK (
        origin_kind = 'phase4d_materialization_v1'),
    action_kind text COLLATE "C" NOT NULL CHECK (
        action_kind = 'dispatch_full_materialization'),
    action_contract_version integer NOT NULL CHECK (
        action_contract_version = 1),
    scope_kind text COLLATE "C" NOT NULL CHECK (
        scope_kind = 'complete_materialization'),
    scope_contract_version integer NOT NULL CHECK (
        scope_contract_version = 1),
    campaign_contract_version integer NOT NULL CHECK (
        campaign_contract_version = 1),
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    campaign_identity_hash text COLLATE "C" NOT NULL CHECK (
        campaign_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_campaign_materialization_fk
        FOREIGN KEY (recommendation_campaign_materialization_id)
        REFERENCES experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_campaign_materialization_uidx
        UNIQUE (recommendation_campaign_materialization_id)
);

ALTER TABLE campaign_operations_campaign
    DROP CONSTRAINT IF EXISTS campaign_operations_campaign_identity_uidx;

CREATE INDEX IF NOT EXISTS campaign_operations_campaign_identity_hash_idx
    ON campaign_operations_campaign(campaign_identity_hash);

CREATE TABLE IF NOT EXISTS
campaign_operations_governance_provenance_event (
    governance_provenance_event_id bigserial PRIMARY KEY CHECK (
        governance_provenance_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    recommendation_campaign_follow_up_ratification_event_id bigint NOT NULL,
    ratification_contract_version integer NOT NULL CHECK (
        ratification_contract_version = 1),
    ratification_identity_canonical text COLLATE "C" NOT NULL CHECK (
        ratification_identity_canonical <> '' AND
        octet_length(ratification_identity_canonical) <= 4194304),
    ratification_identity_hash text COLLATE "C" NOT NULL CHECK (
        ratification_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_campaign_follow_up_proposal_review_event_id bigint NOT NULL,
    review_contract_version integer NOT NULL CHECK (
        review_contract_version = 1),
    review_identity_canonical text COLLATE "C" NOT NULL CHECK (
        review_identity_canonical <> '' AND
        octet_length(review_identity_canonical) <= 2097152),
    review_identity_hash text COLLATE "C" NOT NULL CHECK (
        review_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_campaign_follow_up_proposal_id bigint NOT NULL,
    proposal_contract_version integer NOT NULL CHECK (
        proposal_contract_version = 1),
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        proposal_identity_canonical <> '' AND
        octet_length(proposal_identity_canonical) <= 1048576),
    proposal_identity_hash text COLLATE "C" NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    prerequisite_policy text COLLATE "C" NOT NULL CHECK (
        prerequisite_policy IN (
            'phase4d_materialization_only_v1',
            'phase4d_materialization_plus_exact_phase6d_ratification_v1')),
    provenance_contract_version integer NOT NULL CHECK (
        provenance_contract_version = 1),
    provenance_identity_canonical text COLLATE "C" NOT NULL CHECK (
        provenance_identity_canonical <> '' AND
        octet_length(provenance_identity_canonical) <= 134217728),
    provenance_identity_hash text COLLATE "C" NOT NULL CHECK (
        provenance_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_provenance_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_provenance_ratification_fk
        FOREIGN KEY (
            recommendation_campaign_follow_up_ratification_event_id)
        REFERENCES
            experiment_recommendation_campaign_follow_up_ratification_event(
                recommendation_campaign_follow_up_ratification_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_provenance_review_fk
        FOREIGN KEY (
            recommendation_campaign_follow_up_proposal_review_event_id)
        REFERENCES
            experiment_recommendation_campaign_follow_up_proposal_review_event(
                recommendation_campaign_follow_up_proposal_review_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_provenance_proposal_fk
        FOREIGN KEY (recommendation_campaign_follow_up_proposal_id)
        REFERENCES experiment_recommendation_campaign_follow_up_proposal(
            recommendation_campaign_follow_up_proposal_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_provenance_ratification_uidx
        UNIQUE (operational_campaign_id,
            recommendation_campaign_follow_up_ratification_event_id)
);

ALTER TABLE campaign_operations_governance_provenance_event
    DROP CONSTRAINT IF EXISTS campaign_operations_provenance_identity_uidx;

CREATE INDEX IF NOT EXISTS
campaign_operations_governance_provenance_identity_hash_idx
    ON campaign_operations_governance_provenance_event(
        provenance_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_authorization_event (
    authorization_event_id bigserial PRIMARY KEY CHECK (
        authorization_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    previous_event_id bigint,
    previous_event_identity_canonical text COLLATE "C",
    previous_event_identity_hash text COLLATE "C",
    chain_version integer NOT NULL CHECK (chain_version > 0),
    event_kind text COLLATE "C" NOT NULL CHECK (
        event_kind IN ('granted', 'revoked', 'expiry_observed')),
    action_kind text COLLATE "C" NOT NULL CHECK (
        action_kind IN (
            'dispatch_full_materialization',
            'adopt_existing_pending_and_control')),
    action_contract_version integer NOT NULL CHECK (
        action_contract_version = 1),
    scope_kind text COLLATE "C" NOT NULL CHECK (
        scope_kind = 'complete_materialization'),
    scope_contract_version integer NOT NULL CHECK (
        scope_contract_version = 1),
    prerequisite_policy text COLLATE "C" NOT NULL CHECK (
        prerequisite_policy IN (
            'phase4d_materialization_only_v1',
            'phase4d_materialization_plus_exact_phase6d_ratification_v1')),
    governance_provenance_event_id bigint,
    provenance_identity_canonical text COLLATE "C",
    provenance_identity_hash text COLLATE "C",
    authorization_role text COLLATE "C" NOT NULL CHECK (
        authorization_role = 'campaign_operations_authorizer'),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    not_before timestamptz NOT NULL,
    expires_at timestamptz,
    authorization_contract_version integer NOT NULL CHECK (
        authorization_contract_version = 1),
    authorization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        authorization_identity_canonical <> '' AND
        octet_length(authorization_identity_canonical) <= 134217728),
    authorization_identity_hash text COLLATE "C" NOT NULL CHECK (
        authorization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_authorization_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_authorization_previous_fk
        FOREIGN KEY (previous_event_id)
        REFERENCES campaign_operations_authorization_event(
            authorization_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_authorization_provenance_fk
        FOREIGN KEY (governance_provenance_event_id)
        REFERENCES campaign_operations_governance_provenance_event(
            governance_provenance_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_authorization_validity_check CHECK (
        expires_at IS NULL OR expires_at > not_before),
    CONSTRAINT campaign_operations_authorization_previous_shape_check CHECK (
        (chain_version = 1 AND previous_event_id IS NULL AND
            previous_event_identity_canonical IS NULL AND
            previous_event_identity_hash IS NULL) OR
        (chain_version > 1 AND previous_event_id IS NOT NULL AND
            previous_event_identity_canonical IS NOT NULL AND
            previous_event_identity_canonical <> '' AND
            previous_event_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$')),
    CONSTRAINT campaign_operations_authorization_initial_kind_check CHECK (
        chain_version > 1 OR event_kind = 'granted'),
    CONSTRAINT campaign_operations_authorization_provenance_shape_check CHECK (
        (governance_provenance_event_id IS NULL AND
            provenance_identity_canonical IS NULL AND
            provenance_identity_hash IS NULL AND
            prerequisite_policy = 'phase4d_materialization_only_v1') OR
        (governance_provenance_event_id IS NOT NULL AND
            provenance_identity_canonical IS NOT NULL AND
            provenance_identity_canonical <> '' AND
            provenance_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$')),
    CONSTRAINT campaign_operations_authorization_chain_uidx UNIQUE (
        operational_campaign_id, action_kind, action_contract_version,
        scope_kind, scope_contract_version, chain_version),
    CONSTRAINT campaign_operations_authorization_previous_uidx UNIQUE (
        previous_event_id)
);

ALTER TABLE campaign_operations_authorization_event
    DROP CONSTRAINT IF EXISTS campaign_operations_authorization_identity_uidx;

CREATE INDEX IF NOT EXISTS campaign_operations_authorization_head_idx
    ON campaign_operations_authorization_event(
        operational_campaign_id, action_kind, action_contract_version,
        scope_kind, scope_contract_version, chain_version DESC);
CREATE INDEX IF NOT EXISTS campaign_operations_authorization_identity_hash_idx
    ON campaign_operations_authorization_event(authorization_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_audit_reference_event (
    audit_reference_event_id bigserial PRIMARY KEY CHECK (
        audit_reference_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    governance_provenance_event_id bigint,
    authorization_event_id bigint,
    cause_kind text COLLATE "C" NOT NULL CHECK (
        cause_kind IN (
            'campaign_created',
            'governance_provenance_recorded',
            'authorization_recorded')),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability IN (
            'campaign_operations_campaign_creator',
            'campaign_operations_authorizer')),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    prior_version integer CHECK (prior_version IS NULL OR prior_version > 0),
    resulting_version integer NOT NULL CHECK (resulting_version > 0),
    outcome text COLLATE "C" NOT NULL CHECK (outcome = 'recorded'),
    replay_disposition text COLLATE "C" NOT NULL CHECK (
        replay_disposition = 'recorded'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_audit_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_audit_provenance_fk
        FOREIGN KEY (governance_provenance_event_id)
        REFERENCES campaign_operations_governance_provenance_event(
            governance_provenance_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_audit_authorization_fk
        FOREIGN KEY (authorization_event_id)
        REFERENCES campaign_operations_authorization_event(
            authorization_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_audit_cause_shape_check CHECK (
        (cause_kind = 'campaign_created' AND
            governance_provenance_event_id IS NULL AND
            authorization_event_id IS NULL AND prior_version IS NULL AND
            resulting_version = 1) OR
        (cause_kind = 'governance_provenance_recorded' AND
            governance_provenance_event_id IS NOT NULL AND
            authorization_event_id IS NULL AND prior_version IS NULL AND
            resulting_version = 1) OR
        (cause_kind = 'authorization_recorded' AND
            authorization_event_id IS NOT NULL AND
            resulting_version >= 1 AND
            ((resulting_version = 1 AND prior_version IS NULL) OR
             (resulting_version > 1 AND
              prior_version = resulting_version - 1))))
);

CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_audit_campaign_creation_uidx
    ON campaign_operations_audit_reference_event(operational_campaign_id)
    WHERE cause_kind = 'campaign_created';
CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_audit_provenance_uidx
    ON campaign_operations_audit_reference_event(
        governance_provenance_event_id)
    WHERE governance_provenance_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_audit_authorization_uidx
    ON campaign_operations_audit_reference_event(authorization_event_id)
    WHERE authorization_event_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS campaign_operations_audit_campaign_idx
    ON campaign_operations_audit_reference_event(
        operational_campaign_id, audit_reference_event_id);

CREATE OR REPLACE FUNCTION enforce_campaign_operations_campaign_binding()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_campaign_materialization materialization
    WHERE materialization.recommendation_campaign_materialization_id =
              NEW.recommendation_campaign_materialization_id
      AND materialization.materialization_contract_version =
              NEW.materialization_contract_version
      AND materialization.materialization_identity_canonical =
              NEW.materialization_identity_canonical
      AND materialization.materialization_identity_hash =
              NEW.materialization_identity_hash
      AND materialization.selected_member_count =
              NEW.materialization_member_count
      AND (SELECT count(*)::integer
           FROM experiment_recommendation_campaign_materialization_member member
           WHERE member.recommendation_campaign_materialization_id =
                 materialization.recommendation_campaign_materialization_id) =
              NEW.materialization_member_count
      AND (SELECT min(member_ordinal)
           FROM experiment_recommendation_campaign_materialization_member member
           WHERE member.recommendation_campaign_materialization_id =
                 materialization.recommendation_campaign_materialization_id) = 1
      AND (SELECT max(member_ordinal)
           FROM experiment_recommendation_campaign_materialization_member member
           WHERE member.recommendation_campaign_materialization_id =
                 materialization.recommendation_campaign_materialization_id) =
              NEW.materialization_member_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations materialization mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_campaign_materialization_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_governance_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign campaign
    JOIN experiment_recommendation_campaign_follow_up_ratification_event
        ratification
      ON ratification.recommendation_campaign_follow_up_ratification_event_id =
             NEW.recommendation_campaign_follow_up_ratification_event_id
    JOIN experiment_recommendation_campaign_follow_up_proposal proposal
      ON proposal.recommendation_campaign_follow_up_proposal_id =
             NEW.recommendation_campaign_follow_up_proposal_id
    WHERE campaign.operational_campaign_id = NEW.operational_campaign_id
      AND campaign.campaign_identity_canonical =
              NEW.campaign_identity_canonical
      AND ratification.ratification_contract_version =
              NEW.ratification_contract_version
      AND ratification.ratification_identity_canonical =
              NEW.ratification_identity_canonical
      AND ratification.ratification_identity_hash =
              NEW.ratification_identity_hash
      AND ratification.recommendation_campaign_follow_up_proposal_review_event_id =
              NEW.recommendation_campaign_follow_up_proposal_review_event_id
      AND ratification.review_contract_version = NEW.review_contract_version
      AND ratification.review_identity_canonical =
              NEW.review_identity_canonical
      AND ratification.review_identity_hash = NEW.review_identity_hash
      AND ratification.recommendation_campaign_follow_up_proposal_id =
              NEW.recommendation_campaign_follow_up_proposal_id
      AND ratification.proposal_contract_version =
              NEW.proposal_contract_version
      AND ratification.proposal_identity_canonical =
              NEW.proposal_identity_canonical
      AND ratification.proposal_identity_hash = NEW.proposal_identity_hash
      AND proposal.materialization_id =
              campaign.recommendation_campaign_materialization_id
      AND proposal.materialization_contract_version =
              campaign.materialization_contract_version
      AND proposal.materialization_identity_canonical =
              campaign.materialization_identity_canonical
      AND proposal.materialization_identity_hash =
              campaign.materialization_identity_hash
      AND proposal.member_count = campaign.materialization_member_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations governance provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_governance_provenance_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_authorization_chain()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign campaign
    WHERE campaign.operational_campaign_id = NEW.operational_campaign_id
      AND campaign.campaign_identity_canonical =
              NEW.campaign_identity_canonical;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations authorization campaign mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_authorization_campaign_check';
    END IF;

    IF NEW.chain_version > 1 THEN
        PERFORM 1
        FROM campaign_operations_authorization_event predecessor
        WHERE predecessor.authorization_event_id = NEW.previous_event_id
          AND predecessor.operational_campaign_id =
                  NEW.operational_campaign_id
          AND predecessor.action_kind = NEW.action_kind
          AND predecessor.action_contract_version =
                  NEW.action_contract_version
          AND predecessor.scope_kind = NEW.scope_kind
          AND predecessor.scope_contract_version =
                  NEW.scope_contract_version
          AND predecessor.chain_version = NEW.chain_version - 1
          AND predecessor.authorization_identity_canonical =
                  NEW.previous_event_identity_canonical
          AND predecessor.authorization_identity_hash =
                  NEW.previous_event_identity_hash;
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations authorization predecessor mismatch'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                          'campaign_operations_authorization_previous_check';
        END IF;
    END IF;

    IF NEW.governance_provenance_event_id IS NOT NULL THEN
        PERFORM 1
        FROM campaign_operations_governance_provenance_event provenance
        WHERE provenance.governance_provenance_event_id =
                  NEW.governance_provenance_event_id
          AND provenance.operational_campaign_id =
                  NEW.operational_campaign_id
          AND provenance.provenance_identity_canonical =
                  NEW.provenance_identity_canonical
          AND provenance.provenance_identity_hash =
                  NEW.provenance_identity_hash
          AND provenance.prerequisite_policy =
                  NEW.prerequisite_policy;
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations authorization provenance mismatch'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                          'campaign_operations_authorization_provenance_check';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE foundation_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_campaign_binding() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        foundation_schema, foundation_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_governance_provenance() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        foundation_schema, foundation_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_authorization_chain() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        foundation_schema, foundation_schema);
END $$;

DROP TRIGGER IF EXISTS campaign_operations_campaign_binding_trigger
ON campaign_operations_campaign;
CREATE TRIGGER campaign_operations_campaign_binding_trigger
BEFORE INSERT ON campaign_operations_campaign
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_campaign_binding();

DROP TRIGGER IF EXISTS campaign_operations_governance_provenance_trigger
ON campaign_operations_governance_provenance_event;
CREATE TRIGGER campaign_operations_governance_provenance_trigger
BEFORE INSERT ON campaign_operations_governance_provenance_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_governance_provenance();

DROP TRIGGER IF EXISTS campaign_operations_authorization_chain_trigger
ON campaign_operations_authorization_event;
CREATE TRIGGER campaign_operations_authorization_chain_trigger
BEFORE INSERT ON campaign_operations_authorization_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_authorization_chain();

COMMENT ON TABLE campaign_operations_campaign IS
    'Immutable V1 operational campaign bound to exactly one Phase 4D materialization; row existence derives awaiting_operational_authorization.';
COMMENT ON TABLE campaign_operations_governance_provenance_event IS
    'Optional immutable exact Phase 6D governance provenance; never campaign origin or operational authority.';
COMMENT ON TABLE campaign_operations_authorization_event IS
    'Immutable serialized operational-authorization chain; persisted kinds are granted, revoked, and expiry_observed only.';
COMMENT ON TABLE campaign_operations_audit_reference_event IS
    'Append-only same-transaction reference to Campaign Operations foundation authority; never current-state authority.';

REVOKE ALL PRIVILEGES ON campaign_operations_campaign
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON
    campaign_operations_governance_provenance_event FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON campaign_operations_authorization_event
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON campaign_operations_audit_reference_event
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_campaign_binding() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_governance_provenance() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_authorization_chain() FROM PUBLIC, pqxx;

DO $$
DECLARE foundation_schema text := current_schema();
BEGIN
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO '
        'campaign_operations_owner, campaign_operations_campaign_creator, '
        'campaign_operations_authorizer, campaign_operations_auditor, '
        'campaign_operations_reader', foundation_schema);
END $$;

GRANT SELECT ON campaign_operations_campaign
    TO campaign_operations_campaign_creator,
       campaign_operations_authorizer,
       campaign_operations_auditor,
       campaign_operations_reader;
GRANT SELECT ON campaign_operations_governance_provenance_event,
    campaign_operations_authorization_event
    TO campaign_operations_authorizer,
       campaign_operations_auditor,
       campaign_operations_reader;
GRANT SELECT ON campaign_operations_audit_reference_event
    TO campaign_operations_auditor, campaign_operations_reader;

GRANT SELECT ON experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member
    TO campaign_operations_campaign_creator,
       campaign_operations_authorizer,
       campaign_operations_auditor;
GRANT SELECT ON
    experiment_recommendation_campaign_follow_up_ratification_event,
    experiment_recommendation_campaign_follow_up_proposal_review_event,
    experiment_recommendation_campaign_follow_up_proposal,
    experiment_recommendation_campaign_follow_up_proposal_member
    TO campaign_operations_authorizer, campaign_operations_auditor;

GRANT INSERT (
    recommendation_campaign_materialization_id,
    materialization_contract_version,materialization_identity_canonical,
    materialization_identity_hash,materialization_member_count,origin_kind,
    action_kind,action_contract_version,scope_kind,scope_contract_version,
    campaign_contract_version,campaign_identity_canonical,
    campaign_identity_hash)
ON campaign_operations_campaign
TO campaign_operations_campaign_creator;

GRANT INSERT (
    operational_campaign_id,campaign_identity_canonical,
    recommendation_campaign_follow_up_ratification_event_id,
    ratification_contract_version,ratification_identity_canonical,
    ratification_identity_hash,
    recommendation_campaign_follow_up_proposal_review_event_id,
    review_contract_version,review_identity_canonical,review_identity_hash,
    recommendation_campaign_follow_up_proposal_id,proposal_contract_version,
    proposal_identity_canonical,proposal_identity_hash,prerequisite_policy,
    provenance_contract_version,provenance_identity_canonical,
    provenance_identity_hash)
ON campaign_operations_governance_provenance_event
TO campaign_operations_authorizer;

GRANT INSERT (
    operational_campaign_id,campaign_identity_canonical,previous_event_id,
    previous_event_identity_canonical,previous_event_identity_hash,
    chain_version,event_kind,action_kind,action_contract_version,scope_kind,
    scope_contract_version,prerequisite_policy,
    governance_provenance_event_id,provenance_identity_canonical,
    provenance_identity_hash,authorization_role,actor_identity,reason,
    not_before,expires_at,authorization_contract_version,
    authorization_identity_canonical,authorization_identity_hash)
ON campaign_operations_authorization_event
TO campaign_operations_authorizer;

GRANT INSERT (
    operational_campaign_id,governance_provenance_event_id,
    authorization_event_id,cause_kind,actor_identity,capability,reason,
    prior_version,resulting_version,outcome,replay_disposition)
ON campaign_operations_audit_reference_event
TO campaign_operations_campaign_creator, campaign_operations_authorizer;

DO $$
DECLARE
    campaign_sequence text := pg_get_serial_sequence(
        'campaign_operations_campaign', 'operational_campaign_id');
    provenance_sequence text := pg_get_serial_sequence(
        'campaign_operations_governance_provenance_event',
        'governance_provenance_event_id');
    authorization_sequence text := pg_get_serial_sequence(
        'campaign_operations_authorization_event',
        'authorization_event_id');
    audit_sequence text := pg_get_serial_sequence(
        'campaign_operations_audit_reference_event',
        'audit_reference_event_id');
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        campaign_sequence, provenance_sequence,
        authorization_sequence, audit_sequence]
    LOOP
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC',
            sequence_name);
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx',
            sequence_name);
    END LOOP;
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_campaign_creator', campaign_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_authorizer', provenance_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_authorizer', authorization_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_campaign_creator, '
        'campaign_operations_authorizer', audit_sequence);
END $$;

ALTER TABLE campaign_operations_campaign
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_governance_provenance_event
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_authorization_event
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_audit_reference_event
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_campaign_binding()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_governance_provenance()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_authorization_chain()
    OWNER TO campaign_operations_owner;

-- The capability roles are intentionally not granted to pqxx by this
-- foundation migration. Runtime principal assignment is a separate deployment
-- decision; repository code is present but no operational workflow is enabled.
