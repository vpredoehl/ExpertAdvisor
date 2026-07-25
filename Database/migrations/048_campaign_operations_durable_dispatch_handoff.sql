-- Campaign Operations Phase 3 (architectural Phase E): durable dispatch and
-- atomic lifecycle handoff.  Production dispatch remains structurally
-- disabled.  This migration creates no scheduler claim/attempt capability.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_dispatcher') THEN
        CREATE ROLE campaign_operations_dispatcher NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname =
                       'campaign_operations_phase5_transactional') THEN
        CREATE ROLE campaign_operations_phase5_transactional NOLOGIN;
    END IF;
END $$;

ALTER ROLE campaign_operations_dispatcher
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_phase5_transactional
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
REVOKE campaign_operations_dispatcher FROM pqxx;
REVOKE campaign_operations_phase5_transactional FROM pqxx;

DO $$
BEGIN
    IF pg_has_role('pqxx', 'campaign_operations_dispatcher', 'MEMBER') OR
       pg_has_role('pqxx', 'campaign_operations_phase5_transactional',
                   'MEMBER') THEN
        RAISE EXCEPTION
            'pqxx must not inherit a Campaign Operations dispatch capability';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS campaign_operations_dispatch_attempt (
    dispatch_attempt_id bigserial PRIMARY KEY CHECK (dispatch_attempt_id > 0),
    operational_request_id bigint NOT NULL,
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    attempt_ordinal integer NOT NULL CHECK (attempt_ordinal > 0),
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    resulting_request_version integer NOT NULL CHECK (
        resulting_request_version = expected_request_version + 1),
    lease_token_digest text COLLATE "C" NOT NULL CHECK (
        lease_token_digest ~ '^fnv1a64:[0-9a-f]{16}$'),
    lease_expires_at timestamptz NOT NULL,
    dispatcher_identity text COLLATE "C" NOT NULL CHECK (
        dispatcher_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    attempt_contract_version integer NOT NULL CHECK (
        attempt_contract_version = 1),
    attempt_identity_canonical text COLLATE "C" NOT NULL CHECK (
        attempt_identity_canonical <> ''),
    attempt_identity_hash text COLLATE "C" NOT NULL CHECK (
        attempt_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    acquired_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_dispatch_attempt_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_dispatch_attempt_request_ordinal_uidx
        UNIQUE (operational_request_id, attempt_ordinal),
    CONSTRAINT campaign_operations_dispatch_attempt_request_version_uidx
        UNIQUE (operational_request_id, resulting_request_version)
);

CREATE INDEX IF NOT EXISTS campaign_operations_dispatch_attempt_hash_idx
    ON campaign_operations_dispatch_attempt(attempt_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_dispatch_attempt_lease_idx
    ON campaign_operations_dispatch_attempt(
        operational_request_id, lease_expires_at DESC);

CREATE TABLE IF NOT EXISTS campaign_operations_request_binding (
    request_binding_id bigserial PRIMARY KEY CHECK (request_binding_id > 0),
    operational_request_id bigint NOT NULL,
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    recommendation_campaign_materialization_id bigint NOT NULL,
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        materialization_identity_canonical <> ''),
    recommendation_campaign_materialization_member_id bigint NOT NULL,
    member_ordinal integer NOT NULL CHECK (member_ordinal > 0),
    selected_member_identity_canonical text COLLATE "C" NOT NULL CHECK (
        selected_member_identity_canonical <> ''),
    selected_member_identity_hash text COLLATE "C" NOT NULL CHECK (
        selected_member_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_conversion_proposal_id bigint NOT NULL,
    proposal_identity_canonical text COLLATE "C" NOT NULL CHECK (
        proposal_identity_canonical <> ''),
    proposal_identity_hash text COLLATE "C" NOT NULL CHECK (
        proposal_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommendation_conversion_review_decision_id bigint NOT NULL,
    recommendation_conversion_execution_id bigint NOT NULL,
    execution_identity_canonical text COLLATE "C" NOT NULL CHECK (
        execution_identity_canonical <> ''),
    execution_identity_hash text COLLATE "C" NOT NULL,
    recommendation_conversion_activation_id bigint NOT NULL,
    activation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        activation_identity_canonical <> ''),
    activation_identity_hash text COLLATE "C" NOT NULL,
    experiment_id bigint NOT NULL,
    binding_disposition text COLLATE "C" NOT NULL CHECK (
        binding_disposition IN ('created', 'adopted_existing_pending')),
    execution_disposition text COLLATE "C" NOT NULL CHECK (
        execution_disposition IN ('created', 'reused')),
    activation_disposition text COLLATE "C" NOT NULL CHECK (
        activation_disposition IN ('created', 'reused')),
    binding_contract_version integer NOT NULL CHECK (
        binding_contract_version = 1),
    binding_identity_canonical text COLLATE "C" NOT NULL CHECK (
        binding_identity_canonical <> ''),
    binding_identity_hash text COLLATE "C" NOT NULL CHECK (
        binding_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_request_binding_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_materialization_fk
        FOREIGN KEY (recommendation_campaign_materialization_id) REFERENCES
        experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_member_fk FOREIGN KEY (
        recommendation_campaign_materialization_member_id) REFERENCES
        experiment_recommendation_campaign_materialization_member(
            recommendation_campaign_materialization_member_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_execution_fk FOREIGN KEY (
        recommendation_conversion_execution_id) REFERENCES
        experiment_recommendation_conversion_execution(
            recommendation_conversion_execution_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_activation_fk FOREIGN KEY (
        recommendation_conversion_activation_id) REFERENCES
        experiment_recommendation_conversion_activation(
            recommendation_conversion_activation_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_experiment_fk FOREIGN KEY (
        experiment_id) REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_binding_member_uidx UNIQUE (
        operational_request_id,
        recommendation_campaign_materialization_member_id),
    CONSTRAINT campaign_operations_request_binding_ordinal_uidx UNIQUE (
        operational_request_id, member_ordinal),
    CONSTRAINT campaign_operations_request_binding_experiment_uidx UNIQUE (
        experiment_id),
    CONSTRAINT campaign_operations_request_binding_disposition_check CHECK (
        (binding_disposition = 'created' AND
            execution_disposition = 'created' AND
            activation_disposition = 'created') OR
        (binding_disposition = 'adopted_existing_pending' AND
            execution_disposition = 'reused' AND
            activation_disposition = 'reused'))
);

CREATE INDEX IF NOT EXISTS campaign_operations_request_binding_hash_idx
    ON campaign_operations_request_binding(binding_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_request_binding_lookup_idx
    ON campaign_operations_request_binding(
        operational_request_id, member_ordinal);

CREATE TABLE IF NOT EXISTS campaign_operations_downstream_control_owner (
    downstream_control_owner_id bigserial PRIMARY KEY CHECK (
        downstream_control_owner_id > 0),
    request_binding_id bigint NOT NULL UNIQUE,
    operational_request_id bigint NOT NULL,
    binding_identity_canonical text COLLATE "C" NOT NULL CHECK (
        binding_identity_canonical <> ''),
    experiment_id bigint NOT NULL UNIQUE,
    control_mode text COLLATE "C" NOT NULL CHECK (
        control_mode IN ('created_control',
                         'authorized_adoption_control')),
    adoption_authorization_event_id bigint,
    adoption_authorization_identity_canonical text COLLATE "C",
    adoption_authorization_identity_hash text COLLATE "C",
    owner_contract_version integer NOT NULL CHECK (
        owner_contract_version = 1),
    owner_identity_canonical text COLLATE "C" NOT NULL CHECK (
        owner_identity_canonical <> ''),
    owner_identity_hash text COLLATE "C" NOT NULL CHECK (
        owner_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_control_owner_binding_fk FOREIGN KEY (
        request_binding_id) REFERENCES
        campaign_operations_request_binding(request_binding_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_owner_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_owner_experiment_fk FOREIGN KEY (
        experiment_id) REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_owner_adoption_auth_fk FOREIGN KEY (
        adoption_authorization_event_id) REFERENCES
        campaign_operations_authorization_event(authorization_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_owner_adoption_shape_check CHECK (
        (control_mode = 'created_control' AND
            adoption_authorization_event_id IS NULL AND
            adoption_authorization_identity_canonical IS NULL AND
            adoption_authorization_identity_hash IS NULL) OR
        (control_mode = 'authorized_adoption_control' AND
            adoption_authorization_event_id IS NOT NULL AND
            adoption_authorization_identity_canonical IS NOT NULL AND
            adoption_authorization_identity_hash ~
                '^fnv1a64:[0-9a-f]{16}$'))
);

CREATE INDEX IF NOT EXISTS campaign_operations_control_owner_hash_idx
    ON campaign_operations_downstream_control_owner(owner_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_control_owner_request_idx
    ON campaign_operations_downstream_control_owner(operational_request_id);

CREATE TABLE IF NOT EXISTS campaign_operations_reservation_commitment (
    reservation_commitment_id bigserial PRIMARY KEY CHECK (
        reservation_commitment_id > 0),
    reservation_id bigint NOT NULL UNIQUE,
    reservation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_identity_canonical <> ''),
    operational_request_id bigint NOT NULL UNIQUE,
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    expected_reservation_version integer NOT NULL CHECK (
        expected_reservation_version > 0),
    resulting_reservation_version integer NOT NULL CHECK (
        resulting_reservation_version =
            expected_reservation_version + 1),
    amount bigint NOT NULL CHECK (amount > 0),
    request_binding_set_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_binding_set_identity_canonical <> ''),
    request_binding_set_identity_hash text COLLATE "C" NOT NULL CHECK (
        request_binding_set_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    commitment_contract_version integer NOT NULL CHECK (
        commitment_contract_version = 1),
    commitment_identity_canonical text COLLATE "C" NOT NULL CHECK (
        commitment_identity_canonical <> ''),
    commitment_identity_hash text COLLATE "C" NOT NULL CHECK (
        commitment_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    committed_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_commitment_reservation_fk FOREIGN KEY (
        reservation_id) REFERENCES
        campaign_operations_reservation(reservation_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_commitment_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS campaign_operations_commitment_hash_idx
    ON campaign_operations_reservation_commitment(
        commitment_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_dispatch_attempt_outcome (
    dispatch_attempt_outcome_id bigserial PRIMARY KEY CHECK (
        dispatch_attempt_outcome_id > 0),
    dispatch_attempt_id bigint NOT NULL UNIQUE,
    attempt_identity_canonical text COLLATE "C" NOT NULL CHECK (
        attempt_identity_canonical <> ''),
    result_classification text COLLATE "C" NOT NULL CHECK (
        result_classification IN (
            'created_and_bound', 'adopted_existing_pending_and_bound',
            'existing_identical', 'rejected', 'semantic_conflict',
            'reconciliation_required')),
    downstream_evidence_classification text COLLATE "C" NOT NULL CHECK (
        downstream_evidence_classification IN (
            'no_phase5_evidence', 'exact_complete_pending_train',
            'partial_phase5_evidence', 'paused_only_evidence',
            'progressed_unbound_evidence', 'causally_ambiguous',
            'complete_campaign_operations_binding')),
    semantic_conflict_classification text COLLATE "C" NOT NULL CHECK (
        semantic_conflict_classification IN (
            'none', 'state_version_mismatch', 'lease_unavailable',
            'authorization_inactive', 'budget_inactive',
            'reservation_mismatch', 'request_mismatch',
            'control_owner_collision', 'partial_downstream_evidence',
            'paused_only_evidence', 'progressed_unbound_evidence',
            'causality_mismatch', 'binding_mismatch')),
    uncertain_commit_recovery_classification text COLLATE "C" NOT NULL CHECK (
        uncertain_commit_recovery_classification IN (
            'complete_authoritative_binding', 'proven_no_commit',
            'ambiguous_evidence')),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code ~ '^[a-z0-9_]{1,128}$'),
    request_binding_set_identity_canonical text COLLATE "C",
    request_binding_set_identity_hash text COLLATE "C",
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    resulting_request_version integer NOT NULL CHECK (
        resulting_request_version >= expected_request_version),
    expected_reservation_version integer NOT NULL CHECK (
        expected_reservation_version > 0),
    resulting_reservation_version integer NOT NULL CHECK (
        resulting_reservation_version >= expected_reservation_version),
    outcome_contract_version integer NOT NULL CHECK (
        outcome_contract_version = 1),
    outcome_identity_canonical text COLLATE "C" NOT NULL CHECK (
        outcome_identity_canonical <> ''),
    outcome_identity_hash text COLLATE "C" NOT NULL CHECK (
        outcome_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_attempt_outcome_attempt_fk FOREIGN KEY (
        dispatch_attempt_id) REFERENCES
        campaign_operations_dispatch_attempt(dispatch_attempt_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_attempt_outcome_binding_shape_check CHECK (
        (request_binding_set_identity_canonical IS NULL AND
            request_binding_set_identity_hash IS NULL) OR
        (request_binding_set_identity_canonical IS NOT NULL AND
            request_binding_set_identity_hash ~
                '^fnv1a64:[0-9a-f]{16}$'))
);

CREATE INDEX IF NOT EXISTS campaign_operations_attempt_outcome_hash_idx
    ON campaign_operations_dispatch_attempt_outcome(outcome_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_dispatch_audit_reference_event (
    dispatch_audit_reference_event_id bigserial PRIMARY KEY CHECK (
        dispatch_audit_reference_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    operational_request_id bigint NOT NULL,
    dispatch_attempt_id bigint NOT NULL,
    dispatch_attempt_outcome_id bigint,
    cause_kind text COLLATE "C" NOT NULL CHECK (
        cause_kind IN ('dispatch_lease_acquired',
                       'dispatch_handoff_completed',
                       'dispatch_handoff_failed')),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability IN ('campaign_operations_dispatcher',
                       'campaign_operations_phase5_transactional')),
    prior_version integer NOT NULL CHECK (prior_version > 0),
    resulting_version integer NOT NULL CHECK (
        resulting_version >= prior_version),
    outcome text COLLATE "C" NOT NULL CHECK (
        outcome IN ('recorded', 'rejected', 'reconciliation_required')),
    replay_disposition text COLLATE "C" NOT NULL CHECK (
        replay_disposition IN (
            'new_operation', 'authoritative_existing', 'proven_absent',
            'changed_payload_conflict', 'reconciliation_required')),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code ~ '^[a-z0-9_]{1,128}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_dispatch_audit_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_dispatch_audit_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_dispatch_audit_attempt_fk FOREIGN KEY (
        dispatch_attempt_id) REFERENCES
        campaign_operations_dispatch_attempt(dispatch_attempt_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_dispatch_audit_outcome_fk FOREIGN KEY (
        dispatch_attempt_outcome_id) REFERENCES
        campaign_operations_dispatch_attempt_outcome(
            dispatch_attempt_outcome_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_dispatch_audit_shape_check CHECK (
        (cause_kind = 'dispatch_lease_acquired' AND
            dispatch_attempt_outcome_id IS NULL AND
            capability = 'campaign_operations_dispatcher' AND
            outcome = 'recorded' AND
            resulting_version = prior_version + 1) OR
        (cause_kind IN (
                'dispatch_handoff_completed', 'dispatch_handoff_failed') AND
            dispatch_attempt_outcome_id IS NOT NULL AND
            capability = 'campaign_operations_phase5_transactional'))
);

CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_dispatch_audit_acquisition_uidx
    ON campaign_operations_dispatch_audit_reference_event(
        dispatch_attempt_id)
    WHERE cause_kind = 'dispatch_lease_acquired';
CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_dispatch_audit_outcome_uidx
    ON campaign_operations_dispatch_audit_reference_event(
        dispatch_attempt_outcome_id)
    WHERE dispatch_attempt_outcome_id IS NOT NULL;

CREATE OR REPLACE FUNCTION lock_campaign_operations_reservation(
    target_reservation_id bigint)
RETURNS campaign_operations_reservation
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE locked_row campaign_operations_reservation%ROWTYPE;
BEGIN
    SELECT * INTO STRICT locked_row
    FROM campaign_operations_reservation
    WHERE reservation_id = target_reservation_id
    FOR UPDATE;
    RETURN locked_row;
END;
$$;

CREATE OR REPLACE FUNCTION lock_campaign_operations_authorization_head(
    target_campaign_id bigint, target_action_kind text)
RETURNS campaign_operations_authorization_event
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE locked_row campaign_operations_authorization_event%ROWTYPE;
BEGIN
    SELECT * INTO locked_row
    FROM campaign_operations_authorization_event
    WHERE operational_campaign_id = target_campaign_id
      AND action_kind = target_action_kind
      AND action_contract_version = 1
      AND scope_kind = 'complete_materialization'
      AND scope_contract_version = 1
    ORDER BY chain_version DESC LIMIT 1
    FOR UPDATE;
    RETURN locked_row;
END;
$$;

CREATE OR REPLACE FUNCTION lock_campaign_operations_budget_head(
    target_campaign_id bigint)
RETURNS campaign_operations_budget_ledger_entry
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE locked_row campaign_operations_budget_ledger_entry%ROWTYPE;
BEGIN
    SELECT * INTO locked_row
    FROM campaign_operations_budget_ledger_entry
    WHERE operational_campaign_id = target_campaign_id
    ORDER BY ledger_version DESC LIMIT 1
    FOR UPDATE;
    RETURN locked_row;
END;
$$;

CREATE OR REPLACE FUNCTION lock_campaign_operations_request(
    target_request_id bigint)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE locked_row campaign_operations_operational_request%ROWTYPE;
BEGIN
    SELECT * INTO STRICT locked_row
    FROM campaign_operations_operational_request
    WHERE operational_request_id = target_request_id
    FOR UPDATE;
    RETURN locked_row;
END;
$$;

CREATE OR REPLACE FUNCTION transition_campaign_operations_request_dispatching(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, dispatcher_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_operational_request%ROWTYPE;
BEGIN
    UPDATE campaign_operations_operational_request
       SET request_state = 'dispatching',
           state_version = state_version + 1,
           lease_token_hash = lease_digest_value,
           lease_expires_at = transaction_timestamp() + interval '5 minutes',
           dispatcher_identity = dispatcher_value,
           updated_at = transaction_timestamp()
     WHERE operational_request_id = target_request_id
       AND request_state = 'ready'
       AND state_version = expected_version_value
       AND lease_token_hash IS NULL
       AND lease_expires_at IS NULL
       AND dispatcher_identity IS NULL
       AND production_dispatch_enabled = false
     RETURNING * INTO changed;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations lease compare-and-set lost'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION transition_campaign_operations_reservation_committed(
    target_reservation_id bigint, expected_version_value integer)
RETURNS campaign_operations_reservation
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_reservation%ROWTYPE;
BEGIN
    UPDATE campaign_operations_reservation
       SET reservation_state = 'committed',
           state_version = state_version + 1,
           updated_at = transaction_timestamp()
     WHERE reservation_id = target_reservation_id
       AND reservation_state = 'held'
       AND state_version = expected_version_value
       AND (expires_at IS NULL OR expires_at > transaction_timestamp())
     RETURNING * INTO changed;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations reservation compare-and-set lost'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION transition_campaign_operations_request_bound(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_operational_request%ROWTYPE;
BEGIN
    UPDATE campaign_operations_operational_request
       SET request_state = 'bound',
           state_version = state_version + 1,
           lease_token_hash = NULL,
           lease_expires_at = NULL,
           dispatcher_identity = NULL,
           updated_at = transaction_timestamp()
     WHERE operational_request_id = target_request_id
       AND request_state = 'dispatching'
       AND state_version = expected_version_value
       AND lease_token_hash = lease_digest_value
       AND lease_expires_at > transaction_timestamp()
       AND production_dispatch_enabled = false
     RETURNING * INTO changed;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations bind compare-and-set lost'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_binding_provenance()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_operational_request request
    JOIN experiment_recommendation_campaign_materialization materialization
      ON materialization.recommendation_campaign_materialization_id =
         request.recommendation_campaign_materialization_id
    JOIN experiment_recommendation_campaign_materialization_member member
      ON member.recommendation_campaign_materialization_id =
         materialization.recommendation_campaign_materialization_id
    JOIN experiment_recommendation_conversion_execution execution
      ON execution.recommendation_conversion_proposal_id =
         member.recommendation_conversion_proposal_id
    JOIN experiment_recommendation_conversion_activation activation
      ON activation.recommendation_conversion_execution_id =
         execution.recommendation_conversion_execution_id
    JOIN experiment downstream
      ON downstream.experiment_id = execution.experiment_id
    WHERE request.operational_request_id = NEW.operational_request_id
      AND request.request_identity_canonical =
          NEW.request_identity_canonical
      AND request.recommendation_campaign_materialization_id =
          NEW.recommendation_campaign_materialization_id
      AND materialization.materialization_identity_canonical =
          NEW.materialization_identity_canonical
      AND member.recommendation_campaign_materialization_member_id =
          NEW.recommendation_campaign_materialization_member_id
      AND member.member_ordinal = NEW.member_ordinal
      AND member.selected_member_identity_canonical =
          NEW.selected_member_identity_canonical
      AND member.selected_member_identity_hash =
          NEW.selected_member_identity_hash
      AND member.recommendation_conversion_proposal_id =
          NEW.recommendation_conversion_proposal_id
      AND member.proposal_identity_canonical =
          NEW.proposal_identity_canonical
      AND member.proposal_identity_hash = NEW.proposal_identity_hash
      AND execution.recommendation_conversion_review_decision_id =
          NEW.recommendation_conversion_review_decision_id
      AND execution.recommendation_conversion_execution_id =
          NEW.recommendation_conversion_execution_id
      AND execution.execution_identity_canonical =
          NEW.execution_identity_canonical
      AND execution.execution_identity_hash = NEW.execution_identity_hash
      AND activation.recommendation_conversion_activation_id =
          NEW.recommendation_conversion_activation_id
      AND activation.activation_identity_canonical =
          NEW.activation_identity_canonical
      AND activation.activation_identity_hash =
          NEW.activation_identity_hash
      AND activation.experiment_id = NEW.experiment_id
      AND downstream.status = 'pending'
      AND downstream.phase = 'train';
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations binding provenance mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_binding_provenance_check';
    END IF;
    IF NEW.binding_disposition = 'adopted_existing_pending' AND
       (NEW.execution_disposition <> 'reused' OR
        NEW.activation_disposition <> 'reused') THEN
        RAISE EXCEPTION 'campaign operations adoption must reuse exact work'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_complete_binding()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE request_row campaign_operations_operational_request%ROWTYPE;
DECLARE reservation_row campaign_operations_reservation%ROWTYPE;
DECLARE binding_count integer;
DECLARE owner_count integer;
DECLARE target_request_id bigint;
BEGIN
    IF TG_TABLE_NAME = 'campaign_operations_reservation' THEN
        SELECT operational_request_id INTO target_request_id
        FROM campaign_operations_operational_request
        WHERE reservation_id = NEW.reservation_id;
    ELSE
        target_request_id := NEW.operational_request_id;
    END IF;
    SELECT * INTO request_row
    FROM campaign_operations_operational_request
    WHERE operational_request_id = target_request_id;
    IF NOT FOUND OR request_row.request_state <> 'bound' THEN
        RETURN NEW;
    END IF;
    SELECT * INTO STRICT reservation_row
    FROM campaign_operations_reservation
    WHERE reservation_id = request_row.reservation_id;
    SELECT count(*) INTO binding_count
    FROM campaign_operations_request_binding
    WHERE operational_request_id = request_row.operational_request_id;
    SELECT count(*) INTO owner_count
    FROM campaign_operations_downstream_control_owner
    WHERE operational_request_id = request_row.operational_request_id;
    IF reservation_row.reservation_state <> 'committed' OR
       binding_count <> request_row.materialization_member_count OR
       owner_count <> request_row.materialization_member_count OR
       NOT EXISTS (
           SELECT 1 FROM campaign_operations_reservation_commitment c
           WHERE c.reservation_id = reservation_row.reservation_id
             AND c.operational_request_id =
                 request_row.operational_request_id) OR
       NOT EXISTS (
           SELECT 1
           FROM campaign_operations_dispatch_attempt attempt
           JOIN campaign_operations_dispatch_attempt_outcome outcome
             ON outcome.dispatch_attempt_id = attempt.dispatch_attempt_id
           WHERE attempt.operational_request_id =
                 request_row.operational_request_id
             AND outcome.result_classification IN (
                 'created_and_bound',
                 'adopted_existing_pending_and_bound',
                 'existing_identical')) THEN
        RAISE EXCEPTION 'campaign operations bound request is incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_complete_binding_check';
    END IF;
    IF EXISTS (
        SELECT member.member_ordinal
        FROM experiment_recommendation_campaign_materialization_member member
        WHERE member.recommendation_campaign_materialization_id =
              request_row.recommendation_campaign_materialization_id
        EXCEPT
        SELECT binding.member_ordinal
        FROM campaign_operations_request_binding binding
        WHERE binding.operational_request_id =
              request_row.operational_request_id) THEN
        RAISE EXCEPTION 'campaign operations binding ordinals incomplete'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_control_owner()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_request_binding binding
    WHERE binding.request_binding_id = NEW.request_binding_id
      AND binding.operational_request_id = NEW.operational_request_id
      AND binding.binding_identity_canonical =
          NEW.binding_identity_canonical
      AND binding.experiment_id = NEW.experiment_id
      AND ((binding.binding_disposition = 'created' AND
            NEW.control_mode = 'created_control') OR
           (binding.binding_disposition = 'adopted_existing_pending' AND
            NEW.control_mode = 'authorized_adoption_control'));
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations control owner mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_control_owner_binding_check';
    END IF;
    IF NEW.control_mode = 'authorized_adoption_control' THEN
        PERFORM 1
        FROM campaign_operations_authorization_event authz
        WHERE authz.authorization_event_id =
              NEW.adoption_authorization_event_id
          AND authz.authorization_identity_canonical =
              NEW.adoption_authorization_identity_canonical
          AND authz.authorization_identity_hash =
              NEW.adoption_authorization_identity_hash
          AND authz.operational_campaign_id = (
              SELECT operational_campaign_id
              FROM campaign_operations_operational_request
              WHERE operational_request_id = NEW.operational_request_id)
          AND authz.action_kind =
              'adopt_existing_pending_and_control'
          AND authz.event_kind = 'granted'
          AND authz.not_before <= transaction_timestamp()
          AND (authz.expires_at IS NULL OR
               authz.expires_at > transaction_timestamp())
          AND authz.authorization_event_id = (
              SELECT authorization_event_id
              FROM campaign_operations_authorization_event
              WHERE operational_campaign_id =
                    authz.operational_campaign_id
                AND action_kind =
                    'adopt_existing_pending_and_control'
                AND action_contract_version = 1
                AND scope_kind = 'complete_materialization'
                AND scope_contract_version = 1
              ORDER BY chain_version DESC LIMIT 1);
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations adoption authorization mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_reservation_commitment()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_reservation reservation
    JOIN campaign_operations_operational_request request
      ON request.reservation_id = reservation.reservation_id
    WHERE reservation.reservation_id = NEW.reservation_id
      AND reservation.reservation_identity_canonical =
          NEW.reservation_identity_canonical
      AND reservation.reservation_state = 'committed'
      AND reservation.state_version =
          NEW.resulting_reservation_version
      AND reservation.amount = NEW.amount
      AND request.operational_request_id = NEW.operational_request_id
      AND request.request_identity_canonical =
          NEW.request_identity_canonical
      AND request.request_state = 'bound'
      AND NEW.resulting_reservation_version =
          NEW.expected_reservation_version + 1
      AND (SELECT count(*)
           FROM campaign_operations_request_binding binding
           WHERE binding.operational_request_id =
                 NEW.operational_request_id) =
          request.materialization_member_count;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations commitment mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_commitment_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_dispatch_acquisition_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF NEW.request_state <> 'dispatching' THEN
        RETURN NEW;
    END IF;
    IF OLD.request_state <> 'ready' OR
       NEW.state_version <> OLD.state_version + 1 OR
       NEW.lease_token_hash IS NULL OR
       NEW.lease_expires_at IS NULL OR
       NEW.dispatcher_identity IS NULL OR
       NEW.production_dispatch_enabled OR
       NEW.operational_campaign_id <> OLD.operational_campaign_id OR
       NEW.reservation_id <> OLD.reservation_id OR
       NEW.request_identity_canonical <> OLD.request_identity_canonical OR
       NOT EXISTS (
           SELECT 1
           FROM campaign_operations_dispatch_attempt attempt
           JOIN campaign_operations_dispatch_audit_reference_event audit
             ON audit.dispatch_attempt_id = attempt.dispatch_attempt_id
           WHERE attempt.operational_request_id =
                 NEW.operational_request_id
             AND attempt.request_identity_canonical =
                 NEW.request_identity_canonical
             AND attempt.expected_request_version = OLD.state_version
             AND attempt.resulting_request_version = NEW.state_version
             AND attempt.lease_token_digest = NEW.lease_token_hash
             AND attempt.lease_expires_at = NEW.lease_expires_at
             AND attempt.dispatcher_identity = NEW.dispatcher_identity
             AND audit.operational_campaign_id =
                 NEW.operational_campaign_id
             AND audit.operational_request_id =
                 NEW.operational_request_id
             AND audit.cause_kind = 'dispatch_lease_acquired'
             AND audit.actor_identity = NEW.dispatcher_identity
             AND audit.prior_version = OLD.state_version
             AND audit.resulting_version = NEW.state_version) THEN
        RAISE EXCEPTION
            'campaign operations dispatch acquisition evidence incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_dispatch_acquisition_complete';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_phase5_experiment_update()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF current_user = 'campaign_operations_phase5_transactional' AND (
       OLD.status <> 'paused' OR OLD.phase <> 'train' OR
       NEW.status <> 'pending' OR NEW.phase <> 'train' OR
       NEW.updated_at <> transaction_timestamp() OR
       (to_jsonb(NEW) - ARRAY['status','phase','updated_at']) <>
       (to_jsonb(OLD) - ARRAY['status','phase','updated_at']) OR
       NOT EXISTS (
           SELECT 1
           FROM experiment_recommendation_conversion_activation activation
           WHERE activation.experiment_id = NEW.experiment_id
             AND activation.previous_status = 'paused'
             AND activation.previous_phase = 'train'
             AND activation.resulting_status = 'pending'
             AND activation.resulting_phase = 'train'
             AND activation.xmin =
                 pg_current_xact_id()::text::xid)) THEN
        RAISE EXCEPTION
            'Campaign Operations Phase 5 experiment update denied'
            USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_phase5_mutation_bound()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE target_experiment_id bigint;
BEGIN
    IF current_user <> 'campaign_operations_phase5_transactional' THEN
        RETURN NEW;
    END IF;
    target_experiment_id := NEW.experiment_id;
    IF NOT EXISTS (
        SELECT 1
        FROM campaign_operations_request_binding binding
        JOIN campaign_operations_downstream_control_owner owner
          ON owner.request_binding_id = binding.request_binding_id
         AND owner.operational_request_id =
             binding.operational_request_id
         AND owner.experiment_id = binding.experiment_id
        JOIN campaign_operations_operational_request request
          ON request.operational_request_id =
             binding.operational_request_id
        JOIN campaign_operations_reservation reservation
          ON reservation.reservation_id = request.reservation_id
        JOIN campaign_operations_reservation_commitment commitment
          ON commitment.reservation_id = reservation.reservation_id
         AND commitment.operational_request_id =
             request.operational_request_id
        WHERE binding.experiment_id = target_experiment_id
          AND request.request_state = 'bound'
          AND reservation.reservation_state = 'committed') THEN
        RAISE EXCEPTION
            'Campaign Operations Phase 5 mutation lacks atomic binding'
            USING ERRCODE = '42501';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE phase3_schema text := current_schema();
DECLARE function_name text;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'lock_campaign_operations_authorization_head(bigint,text)',
        'lock_campaign_operations_budget_head(bigint)',
        'lock_campaign_operations_reservation(bigint)',
        'lock_campaign_operations_request(bigint)',
        'transition_campaign_operations_request_dispatching(bigint,integer,text,text)',
        'transition_campaign_operations_reservation_committed(bigint,integer)',
        'transition_campaign_operations_request_bound(bigint,integer,text)',
        'enforce_campaign_operations_binding_provenance()',
        'enforce_campaign_operations_complete_binding()',
        'enforce_campaign_operations_control_owner()',
        'enforce_campaign_operations_reservation_commitment()',
        'enforce_campaign_operations_dispatch_acquisition_complete()',
        'guard_campaign_operations_phase5_experiment_update()',
        'enforce_campaign_operations_phase5_mutation_bound()']
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%s SET search_path TO pg_catalog, %I, pg_temp',
            phase3_schema, function_name, phase3_schema);
    END LOOP;
END $$;

DROP TRIGGER IF EXISTS campaign_operations_binding_provenance_trigger
    ON campaign_operations_request_binding;
CREATE TRIGGER campaign_operations_binding_provenance_trigger
BEFORE INSERT ON campaign_operations_request_binding
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_binding_provenance();

DROP TRIGGER IF EXISTS campaign_operations_control_owner_trigger
    ON campaign_operations_downstream_control_owner;
CREATE TRIGGER campaign_operations_control_owner_trigger
BEFORE INSERT ON campaign_operations_downstream_control_owner
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_owner();

DROP TRIGGER IF EXISTS campaign_operations_reservation_commitment_trigger
    ON campaign_operations_reservation_commitment;
CREATE CONSTRAINT TRIGGER campaign_operations_reservation_commitment_trigger
AFTER INSERT ON campaign_operations_reservation_commitment
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_reservation_commitment();

DROP TRIGGER IF EXISTS
    campaign_operations_dispatch_acquisition_complete_trigger
    ON campaign_operations_operational_request;
CREATE CONSTRAINT TRIGGER
    campaign_operations_dispatch_acquisition_complete_trigger
AFTER UPDATE ON campaign_operations_operational_request
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_dispatch_acquisition_complete();

DROP TRIGGER IF EXISTS
    campaign_operations_phase5_experiment_update_guard_trigger
    ON experiment;
CREATE TRIGGER
    campaign_operations_phase5_experiment_update_guard_trigger
BEFORE UPDATE ON experiment
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_phase5_experiment_update();

DROP TRIGGER IF EXISTS
    campaign_operations_phase5_experiment_bound_trigger ON experiment;
CREATE CONSTRAINT TRIGGER
    campaign_operations_phase5_experiment_bound_trigger
AFTER INSERT OR UPDATE ON experiment
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_phase5_mutation_bound();
DROP TRIGGER IF EXISTS
    campaign_operations_phase5_execution_bound_trigger
    ON experiment_recommendation_conversion_execution;
CREATE CONSTRAINT TRIGGER
    campaign_operations_phase5_execution_bound_trigger
AFTER INSERT ON experiment_recommendation_conversion_execution
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_phase5_mutation_bound();
DROP TRIGGER IF EXISTS
    campaign_operations_phase5_activation_bound_trigger
    ON experiment_recommendation_conversion_activation;
CREATE CONSTRAINT TRIGGER
    campaign_operations_phase5_activation_bound_trigger
AFTER INSERT ON experiment_recommendation_conversion_activation
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_phase5_mutation_bound();

DROP TRIGGER IF EXISTS campaign_operations_complete_request_trigger
    ON campaign_operations_operational_request;
CREATE CONSTRAINT TRIGGER campaign_operations_complete_request_trigger
AFTER INSERT OR UPDATE ON campaign_operations_operational_request
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_complete_binding();
DROP TRIGGER IF EXISTS campaign_operations_complete_reservation_trigger
    ON campaign_operations_reservation;
CREATE CONSTRAINT TRIGGER campaign_operations_complete_reservation_trigger
AFTER UPDATE ON campaign_operations_reservation
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_complete_binding();
DROP TRIGGER IF EXISTS campaign_operations_complete_binding_trigger
    ON campaign_operations_request_binding;
CREATE CONSTRAINT TRIGGER campaign_operations_complete_binding_trigger
AFTER INSERT ON campaign_operations_request_binding
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_complete_binding();
DROP TRIGGER IF EXISTS campaign_operations_complete_owner_trigger
    ON campaign_operations_downstream_control_owner;
CREATE CONSTRAINT TRIGGER campaign_operations_complete_owner_trigger
AFTER INSERT ON campaign_operations_downstream_control_owner
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_complete_binding();

COMMENT ON TABLE campaign_operations_dispatch_attempt IS
    'Immutable lease-acquisition evidence; never current dispatch authority.';
COMMENT ON TABLE campaign_operations_request_binding IS
    'Immutable complete ordered Phase 4D-to-Phase 5 member binding.';
COMMENT ON TABLE campaign_operations_downstream_control_owner IS
    'Permanent V1 Campaign Operations control ownership; no transfer path.';
COMMENT ON TABLE campaign_operations_reservation_commitment IS
    'Immutable held-to-committed settlement evidence.';
COMMENT ON TABLE campaign_operations_dispatch_attempt_outcome IS
    'Immutable dispatch outcome and recovery classification evidence.';

REVOKE ALL PRIVILEGES ON
    campaign_operations_dispatch_attempt,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_reservation_commitment,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_dispatch_audit_reference_event
    FROM PUBLIC, pqxx,
         campaign_operations_dispatcher,
         campaign_operations_phase5_transactional;

DO $$
DECLARE sequence_name text;
BEGIN
    FOR sequence_name IN
        SELECT pg_get_serial_sequence(table_name, column_name)
        FROM (VALUES
            ('campaign_operations_dispatch_attempt',
             'dispatch_attempt_id'),
            ('campaign_operations_request_binding', 'request_binding_id'),
            ('campaign_operations_downstream_control_owner',
             'downstream_control_owner_id'),
            ('campaign_operations_reservation_commitment',
             'reservation_commitment_id'),
            ('campaign_operations_dispatch_attempt_outcome',
             'dispatch_attempt_outcome_id'),
            ('campaign_operations_dispatch_audit_reference_event',
             'dispatch_audit_reference_event_id'))
             AS sequences(table_name, column_name)
    LOOP
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC',
                       sequence_name);
        EXECUTE format('REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM pqxx',
                       sequence_name);
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM '
            'campaign_operations_dispatcher, '
            'campaign_operations_phase5_transactional', sequence_name);
    END LOOP;
END $$;

REVOKE ALL PRIVILEGES ON FUNCTION
    lock_campaign_operations_authorization_head(bigint, text),
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint),
    transition_campaign_operations_request_dispatching(
        bigint, integer, text, text),
    transition_campaign_operations_reservation_committed(bigint, integer),
    transition_campaign_operations_request_bound(bigint, integer, text),
    enforce_campaign_operations_binding_provenance(),
    enforce_campaign_operations_complete_binding(),
    enforce_campaign_operations_control_owner(),
    enforce_campaign_operations_reservation_commitment(),
    enforce_campaign_operations_dispatch_acquisition_complete(),
    guard_campaign_operations_phase5_experiment_update(),
    enforce_campaign_operations_phase5_mutation_bound()
    FROM PUBLIC, pqxx,
         campaign_operations_dispatcher,
         campaign_operations_phase5_transactional;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_governance_provenance_event,
    campaign_operations_authorization_event,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_reservation_commitment,
    campaign_operations_dispatch_audit_reference_event,
    experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member
    TO campaign_operations_dispatcher;
GRANT INSERT (
    operational_request_id, request_identity_canonical, attempt_ordinal,
    expected_request_version, resulting_request_version, lease_token_digest,
    lease_expires_at, dispatcher_identity, attempt_contract_version,
    attempt_identity_canonical, attempt_identity_hash)
    ON campaign_operations_dispatch_attempt
    TO campaign_operations_dispatcher;
GRANT INSERT (
    operational_campaign_id, operational_request_id, dispatch_attempt_id,
    dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
    prior_version, resulting_version, outcome, replay_disposition,
    diagnostic_code)
    ON campaign_operations_dispatch_audit_reference_event
    TO campaign_operations_dispatcher;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_governance_provenance_event,
    campaign_operations_authorization_event,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_reservation_commitment,
    campaign_operations_dispatch_audit_reference_event,
    experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_conversion_proposal,
    experiment_recommendation_conversion_review_decision,
    experiment_recommendation_conversion_execution,
    experiment_recommendation_conversion_activation,
    experiment
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    operational_request_id, request_identity_canonical,
    recommendation_campaign_materialization_id,
    materialization_identity_canonical,
    recommendation_campaign_materialization_member_id, member_ordinal,
    selected_member_identity_canonical, selected_member_identity_hash,
    recommendation_conversion_proposal_id, proposal_identity_canonical,
    proposal_identity_hash, recommendation_conversion_review_decision_id,
    recommendation_conversion_execution_id, execution_identity_canonical,
    execution_identity_hash, recommendation_conversion_activation_id,
    activation_identity_canonical, activation_identity_hash, experiment_id,
    binding_disposition, execution_disposition, activation_disposition,
    binding_contract_version, binding_identity_canonical,
    binding_identity_hash)
    ON campaign_operations_request_binding
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    request_binding_id, operational_request_id, binding_identity_canonical,
    experiment_id, control_mode, adoption_authorization_event_id,
    adoption_authorization_identity_canonical,
    adoption_authorization_identity_hash, owner_contract_version,
    owner_identity_canonical, owner_identity_hash)
    ON campaign_operations_downstream_control_owner
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    reservation_id, reservation_identity_canonical, operational_request_id,
    request_identity_canonical, expected_reservation_version,
    resulting_reservation_version, amount,
    request_binding_set_identity_canonical,
    request_binding_set_identity_hash, commitment_contract_version,
    commitment_identity_canonical, commitment_identity_hash)
    ON campaign_operations_reservation_commitment
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    dispatch_attempt_id, attempt_identity_canonical, result_classification,
    downstream_evidence_classification, semantic_conflict_classification,
    uncertain_commit_recovery_classification, diagnostic_code,
    request_binding_set_identity_canonical,
    request_binding_set_identity_hash, expected_request_version,
    resulting_request_version, expected_reservation_version,
    resulting_reservation_version, outcome_contract_version,
    outcome_identity_canonical, outcome_identity_hash)
    ON campaign_operations_dispatch_attempt_outcome
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    operational_campaign_id, operational_request_id, dispatch_attempt_id,
    dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
    prior_version, resulting_version, outcome, replay_disposition,
    diagnostic_code)
    ON campaign_operations_dispatch_audit_reference_event
    TO campaign_operations_phase5_transactional;

GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_authorization_head(bigint, text),
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint),
    transition_campaign_operations_request_dispatching(
        bigint, integer, text, text)
    TO campaign_operations_dispatcher;
GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_authorization_head(bigint, text),
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint),
    transition_campaign_operations_reservation_committed(bigint, integer),
    transition_campaign_operations_request_bound(bigint, integer, text)
    TO campaign_operations_phase5_transactional;

DO $$
DECLARE sequence_name text;
BEGIN
    FOR sequence_name IN
        SELECT pg_get_serial_sequence(table_name, column_name)
        FROM (VALUES
            ('campaign_operations_dispatch_attempt',
             'dispatch_attempt_id'),
            ('campaign_operations_dispatch_audit_reference_event',
             'dispatch_audit_reference_event_id'))
             AS sequences(table_name, column_name)
    LOOP
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
                       'campaign_operations_dispatcher', sequence_name);
    END LOOP;
    FOR sequence_name IN
        SELECT pg_get_serial_sequence(table_name, column_name)
        FROM (VALUES
            ('campaign_operations_request_binding', 'request_binding_id'),
            ('campaign_operations_downstream_control_owner',
             'downstream_control_owner_id'),
            ('campaign_operations_reservation_commitment',
             'reservation_commitment_id'),
            ('campaign_operations_dispatch_attempt_outcome',
             'dispatch_attempt_outcome_id'),
            ('campaign_operations_dispatch_audit_reference_event',
             'dispatch_audit_reference_event_id'),
            ('experiment_recommendation_conversion_execution',
             'recommendation_conversion_execution_id'),
            ('experiment_recommendation_conversion_activation',
             'recommendation_conversion_activation_id'),
            ('experiment', 'experiment_id'))
             AS sequences(table_name, column_name)
    LOOP
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
                       'campaign_operations_phase5_transactional',
                       sequence_name);
    END LOOP;
END $$;

-- Existing Phase 5 repository operations, column-scoped exactly.
GRANT INSERT (
    symbol, prediction_horizon, c_next_threshold, core_lr_mult, head_lr_mult,
    target_epochs, checkpoint_interval, train_start, train_end, infer_start,
    infer_end, status, phase, resume_model_id, duplicate_nonce,
    invocation_mode, updated_at)
    ON experiment TO campaign_operations_phase5_transactional;
GRANT UPDATE (status, phase, updated_at)
    ON experiment TO campaign_operations_phase5_transactional;
GRANT INSERT (
    recommendation_conversion_proposal_id,
    recommendation_conversion_review_decision_id, experiment_id,
    execution_contract_version, authorization_decision,
    execution_identity_canonical, execution_identity_hash)
    ON experiment_recommendation_conversion_execution
    TO campaign_operations_phase5_transactional;
GRANT INSERT (
    recommendation_conversion_execution_id,
    recommendation_conversion_proposal_id,
    recommendation_conversion_review_decision_id, experiment_id,
    activation_contract_version, previous_status, previous_phase,
    resulting_status, resulting_phase, activation_identity_canonical,
    activation_identity_hash)
    ON experiment_recommendation_conversion_activation
    TO campaign_operations_phase5_transactional;

-- The application schema is owned by the existing no-login owner.  Capability
-- roles are invokers only and are deliberately not granted to a login here.
ALTER TABLE campaign_operations_dispatch_attempt
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_request_binding
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_downstream_control_owner
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reservation_commitment
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_dispatch_attempt_outcome
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    OWNER TO campaign_operations_owner;

ALTER FUNCTION lock_campaign_operations_reservation(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION lock_campaign_operations_authorization_head(bigint, text)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION lock_campaign_operations_budget_head(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION lock_campaign_operations_request(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_request_dispatching(
    bigint, integer, text, text) OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_reservation_committed(
    bigint, integer) OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_request_bound(
    bigint, integer, text) OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_binding_provenance()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_complete_binding()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_control_owner()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_reservation_commitment()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_dispatch_acquisition_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION guard_campaign_operations_phase5_experiment_update()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_phase5_mutation_bound()
    OWNER TO campaign_operations_owner;

DO $$
DECLARE phase3_schema text := current_schema();
BEGIN
    EXECUTE format(
        'GRANT USAGE ON SCHEMA %I TO campaign_operations_dispatcher, '
        'campaign_operations_phase5_transactional', phase3_schema);
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner '
        'IN SCHEMA %I REVOKE ALL ON TABLES FROM PUBLIC',
        phase3_schema);
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner '
        'IN SCHEMA %I REVOKE ALL ON SEQUENCES FROM PUBLIC',
        phase3_schema);
    EXECUTE format(
        'ALTER DEFAULT PRIVILEGES FOR ROLE campaign_operations_owner '
        'IN SCHEMA %I REVOKE EXECUTE ON FUNCTIONS FROM PUBLIC',
        phase3_schema);
END $$;
