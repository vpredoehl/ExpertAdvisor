-- Campaign Operations Phase 4 (architectural Phase F): campaign-only
-- controls, cancellation intent/settlement, lifecycle delegation, and
-- detection-first reconciliation.  This migration creates no scheduler,
-- process-signal, completion, archival, reporting, or production-dispatch
-- authority.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_controller') THEN
        CREATE ROLE campaign_operations_controller NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname =
                       'campaign_operations_cancellation_coordinator') THEN
        CREATE ROLE campaign_operations_cancellation_coordinator NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_reconciler') THEN
        CREATE ROLE campaign_operations_reconciler NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_recovery') THEN
        CREATE ROLE campaign_operations_recovery NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'experiment_lifecycle_cancellation') THEN
        CREATE ROLE experiment_lifecycle_cancellation NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname =
                       'experiment_lifecycle_cancellation_owner') THEN
        CREATE ROLE experiment_lifecycle_cancellation_owner NOLOGIN;
    END IF;
END $$;

ALTER ROLE campaign_operations_controller
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_cancellation_coordinator
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_reconciler
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_recovery
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE experiment_lifecycle_cancellation
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE experiment_lifecycle_cancellation_owner
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

REVOKE campaign_operations_controller,
       campaign_operations_cancellation_coordinator,
       campaign_operations_reconciler,
       campaign_operations_recovery,
       experiment_lifecycle_cancellation,
       experiment_lifecycle_cancellation_owner
    FROM pqxx;

CREATE TABLE IF NOT EXISTS campaign_operations_control_event (
    control_event_id bigserial PRIMARY KEY CHECK (control_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> ''),
    previous_control_event_id bigint,
    previous_control_event_identity_canonical text COLLATE "C",
    control_version integer NOT NULL CHECK (control_version > 0),
    event_kind text COLLATE "C" NOT NULL CHECK (
        event_kind IN ('pause', 'resume')),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability = 'campaign_operations_controller'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096),
    control_contract_version integer NOT NULL CHECK (
        control_contract_version = 1),
    control_identity_canonical text COLLATE "C" NOT NULL CHECK (
        control_identity_canonical <> ''),
    control_identity_hash text COLLATE "C" NOT NULL CHECK (
        control_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_control_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_previous_fk FOREIGN KEY (
        previous_control_event_id) REFERENCES
        campaign_operations_control_event(control_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_version_uidx UNIQUE (
        operational_campaign_id, control_version),
    CONSTRAINT campaign_operations_control_previous_uidx UNIQUE (
        previous_control_event_id),
    CONSTRAINT campaign_operations_control_chain_shape_check CHECK (
        (control_version = 1 AND previous_control_event_id IS NULL AND
            previous_control_event_identity_canonical IS NULL) OR
        (control_version > 1 AND previous_control_event_id IS NOT NULL AND
            previous_control_event_identity_canonical IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS campaign_operations_control_head_idx
    ON campaign_operations_control_event(
        operational_campaign_id, control_version DESC);
CREATE INDEX IF NOT EXISTS campaign_operations_control_hash_idx
    ON campaign_operations_control_event(control_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_cancellation_request (
    cancellation_request_id bigserial PRIMARY KEY CHECK (
        cancellation_request_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> ''),
    operational_request_id bigint,
    request_identity_canonical text COLLATE "C",
    expected_request_state text COLLATE "C" CHECK (
        expected_request_state IN (
            'ready', 'dispatching', 'bound', 'permanently_failed',
            'cancelled', 'reconciliation_required')),
    expected_request_version integer CHECK (expected_request_version > 0),
    cancellation_scope text COLLATE "C" NOT NULL CHECK (
        cancellation_scope = 'complete_materialization'),
    operation_key text COLLATE "C" NOT NULL CHECK (
        operation_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    capability text COLLATE "C" NOT NULL CHECK (
        capability = 'campaign_operations_cancellation_coordinator'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096),
    cancellation_contract_version integer NOT NULL CHECK (
        cancellation_contract_version = 1),
    cancellation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        cancellation_identity_canonical <> ''),
    cancellation_identity_hash text COLLATE "C" NOT NULL CHECK (
        cancellation_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_cancellation_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_cancellation_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_cancellation_operation_uidx UNIQUE (
        operational_campaign_id, cancellation_scope, operation_key),
    CONSTRAINT campaign_operations_cancellation_target_shape_check CHECK (
        (operational_request_id IS NULL AND
            request_identity_canonical IS NULL AND
            expected_request_state IS NULL AND
            expected_request_version IS NULL) OR
        (operational_request_id IS NOT NULL AND
            request_identity_canonical IS NOT NULL AND
            expected_request_state IS NOT NULL AND
            expected_request_version IS NOT NULL))
);

CREATE INDEX IF NOT EXISTS campaign_operations_cancellation_request_hash_idx
    ON campaign_operations_cancellation_request(
        cancellation_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_cancellation_unsettled_idx
    ON campaign_operations_cancellation_request(
        operational_request_id, cancellation_request_id);
CREATE UNIQUE INDEX IF NOT EXISTS
    campaign_operations_cancellation_request_target_uidx
    ON campaign_operations_cancellation_request(operational_request_id)
    WHERE operational_request_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS
    campaign_operations_cancellation_campaign_target_uidx
    ON campaign_operations_cancellation_request(operational_campaign_id)
    WHERE operational_request_id IS NULL;

CREATE TABLE IF NOT EXISTS experiment_lifecycle_cancellation_event (
    lifecycle_cancellation_event_id bigserial PRIMARY KEY CHECK (
        lifecycle_cancellation_event_id > 0),
    cancellation_request_id bigint NOT NULL,
    cancellation_request_identity_canonical text COLLATE "C" NOT NULL,
    downstream_control_owner_id bigint NOT NULL,
    experiment_id bigint NOT NULL,
    expected_status text COLLATE "C" NOT NULL,
    expected_phase text COLLATE "C" NOT NULL,
    observed_status text COLLATE "C" NOT NULL,
    observed_phase text COLLATE "C" NOT NULL,
    resulting_status text COLLATE "C" NOT NULL,
    disposition text COLLATE "C" NOT NULL CHECK (
        disposition IN (
            'accepted', 'already_terminal', 'running_not_supported')),
    actor_identity text COLLATE "C" NOT NULL CHECK (
        actor_identity ~ '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    event_contract_version integer NOT NULL CHECK (
        event_contract_version = 1),
    event_identity_canonical text COLLATE "C" NOT NULL CHECK (
        event_identity_canonical <> ''),
    event_identity_hash text COLLATE "C" NOT NULL CHECK (
        event_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT lifecycle_cancellation_request_fk FOREIGN KEY (
        cancellation_request_id) REFERENCES
        campaign_operations_cancellation_request(cancellation_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT lifecycle_cancellation_owner_fk FOREIGN KEY (
        downstream_control_owner_id) REFERENCES
        campaign_operations_downstream_control_owner(
            downstream_control_owner_id) ON DELETE RESTRICT,
    CONSTRAINT lifecycle_cancellation_experiment_fk FOREIGN KEY (
        experiment_id) REFERENCES experiment(experiment_id)
        ON DELETE RESTRICT,
    CONSTRAINT lifecycle_cancellation_owner_uidx UNIQUE (
        cancellation_request_id, downstream_control_owner_id),
    CONSTRAINT lifecycle_cancellation_result_shape_check CHECK (
        (disposition = 'accepted' AND
            observed_status IN ('pending', 'paused') AND
            resulting_status = 'cancelled') OR
        (disposition = 'already_terminal' AND
            observed_status IN ('completed', 'failed', 'cancelled') AND
            resulting_status = observed_status) OR
        (disposition = 'running_not_supported' AND
            observed_status = 'running' AND
            resulting_status = observed_status))
);

CREATE INDEX IF NOT EXISTS lifecycle_cancellation_event_hash_idx
    ON experiment_lifecycle_cancellation_event(event_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_cancellation_settlement (
    cancellation_settlement_id bigserial PRIMARY KEY CHECK (
        cancellation_settlement_id > 0),
    cancellation_request_id bigint NOT NULL UNIQUE,
    cancellation_request_identity_canonical text COLLATE "C" NOT NULL,
    disposition text COLLATE "C" NOT NULL CHECK (
        disposition IN (
            'unbound_cancelled', 'lifecycle_request_accepted',
            'already_terminal', 'running_cancellation_not_supported',
            'inconsistent')),
    reservation_event_id bigint,
    reservation_event_identity_canonical text COLLATE "C",
    resulting_request_version integer CHECK (
        resulting_request_version > 0),
    lifecycle_evidence_identity_canonical text COLLATE "C",
    lifecycle_evidence_identity_hash text COLLATE "C",
    settlement_contract_version integer NOT NULL CHECK (
        settlement_contract_version = 1),
    settlement_identity_canonical text COLLATE "C" NOT NULL CHECK (
        settlement_identity_canonical <> ''),
    settlement_identity_hash text COLLATE "C" NOT NULL CHECK (
        settlement_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_settlement_request_fk FOREIGN KEY (
        cancellation_request_id) REFERENCES
        campaign_operations_cancellation_request(cancellation_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_settlement_reservation_event_fk
        FOREIGN KEY (reservation_event_id) REFERENCES
        campaign_operations_reservation_event(reservation_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_settlement_shape_check CHECK (
        (disposition = 'unbound_cancelled' AND
            reservation_event_id IS NOT NULL AND
            reservation_event_identity_canonical IS NOT NULL AND
            resulting_request_version IS NOT NULL AND
            lifecycle_evidence_identity_canonical IS NULL AND
            lifecycle_evidence_identity_hash IS NULL) OR
        (disposition <> 'unbound_cancelled' AND
            reservation_event_id IS NULL AND
            reservation_event_identity_canonical IS NULL AND
            lifecycle_evidence_identity_canonical IS NOT NULL AND
            lifecycle_evidence_identity_hash ~
                '^fnv1a64:[0-9a-f]{16}$'))
);

CREATE INDEX IF NOT EXISTS campaign_operations_settlement_hash_idx
    ON campaign_operations_cancellation_settlement(
        settlement_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_reconciliation_observation (
    reconciliation_observation_id bigserial PRIMARY KEY CHECK (
        reconciliation_observation_id > 0),
    run_key text COLLATE "C" NOT NULL CHECK (
        run_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    operational_campaign_id bigint NOT NULL,
    operational_request_id bigint NOT NULL,
    request_identity_canonical text COLLATE "C" NOT NULL,
    expected_request_state text COLLATE "C" NOT NULL CHECK (
        expected_request_state IN (
            'ready', 'dispatching', 'bound', 'permanently_failed',
            'cancelled', 'reconciliation_required')),
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    reason_code text COLLATE "C" NOT NULL CHECK (
        reason_code IN (
            'ready_request_not_dispatched',
            'dispatch_lease_expired_no_downstream_evidence',
            'dispatch_outcome_unknown',
            'binding_projection_missing',
            'reservation_projection_missing_commit',
            'held_reservation_terminal_unbound_request',
            'reservation_expired_no_downstream_evidence',
            'cancellation_settlement_pending',
            'terminal_lifecycle_completion_ready',
            'progressed_unbound_evidence',
            'partial_downstream_evidence',
            'binding_cardinality_mismatch',
            'control_owner_conflict',
            'budget_accounting_mismatch',
            'post_completion_lifecycle_changed',
            'causality_ambiguous')),
    evidence_identity_canonical text COLLATE "C" NOT NULL,
    evidence_identity_hash text COLLATE "C" NOT NULL CHECK (
        evidence_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    recommended_service text COLLATE "C" NOT NULL CHECK (
        recommended_service ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    recommended_action text COLLATE "C" NOT NULL CHECK (
        recommended_action ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code ~ '^[a-z0-9_]{1,128}$'),
    observation_contract_version integer NOT NULL CHECK (
        observation_contract_version = 1),
    observation_identity_canonical text COLLATE "C" NOT NULL,
    observation_identity_hash text COLLATE "C" NOT NULL CHECK (
        observation_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    observed_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_observation_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_observation_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT
);

CREATE INDEX IF NOT EXISTS campaign_operations_observation_candidate_idx
    ON campaign_operations_reconciliation_observation(
        operational_request_id, reconciliation_observation_id);
CREATE INDEX IF NOT EXISTS campaign_operations_observation_hash_idx
    ON campaign_operations_reconciliation_observation(
        observation_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_reconciliation_resolution (
    reconciliation_resolution_id bigserial PRIMARY KEY CHECK (
        reconciliation_resolution_id > 0),
    reconciliation_observation_id bigint NOT NULL UNIQUE,
    observation_identity_canonical text COLLATE "C" NOT NULL,
    owning_service text COLLATE "C" NOT NULL,
    owning_capability text COLLATE "C" NOT NULL CHECK (
        owning_capability IN (
            'campaign_operations_recovery',
            'campaign_operations_cancellation_coordinator')),
    transition_identity_canonical text COLLATE "C" NOT NULL,
    transition_identity_hash text COLLATE "C" NOT NULL CHECK (
        transition_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    cancellation_settlement_id bigint,
    dispatch_attempt_outcome_id bigint,
    resolution_disposition text COLLATE "C" NOT NULL CHECK (
        resolution_disposition IN (
            'request_returned_ready', 'cancellation_settled',
            'already_resolved')),
    resolution_contract_version integer NOT NULL CHECK (
        resolution_contract_version = 1),
    resolution_identity_canonical text COLLATE "C" NOT NULL,
    resolution_identity_hash text COLLATE "C" NOT NULL CHECK (
        resolution_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_resolution_observation_fk FOREIGN KEY (
        reconciliation_observation_id) REFERENCES
        campaign_operations_reconciliation_observation(
            reconciliation_observation_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_resolution_cancellation_cause_fk
        FOREIGN KEY (cancellation_settlement_id) REFERENCES
        campaign_operations_cancellation_settlement(
            cancellation_settlement_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_resolution_recovery_cause_fk
        FOREIGN KEY (dispatch_attempt_outcome_id) REFERENCES
        campaign_operations_dispatch_attempt_outcome(
            dispatch_attempt_outcome_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_resolution_causal_shape_check CHECK (
        (owning_capability =
             'campaign_operations_cancellation_coordinator' AND
         cancellation_settlement_id IS NOT NULL AND
         dispatch_attempt_outcome_id IS NULL AND
         resolution_disposition = 'cancellation_settled') OR
        (owning_capability = 'campaign_operations_recovery' AND
         cancellation_settlement_id IS NULL AND
         dispatch_attempt_outcome_id IS NOT NULL AND
         resolution_disposition IN (
             'request_returned_ready', 'already_resolved')))
);

CREATE TABLE IF NOT EXISTS campaign_operations_reconciliation_cursor_event (
    reconciliation_cursor_event_id bigserial PRIMARY KEY CHECK (
        reconciliation_cursor_event_id > 0),
    run_key text COLLATE "C" NOT NULL CHECK (
        run_key ~ '^[A-Za-z0-9][A-Za-z0-9._:/\-]{0,127}$'),
    prior_target_id bigint NOT NULL CHECK (prior_target_id >= 0),
    last_target_id bigint NOT NULL CHECK (
        last_target_id >= prior_target_id),
    requested_limit integer NOT NULL CHECK (
        requested_limit > 0 AND requested_limit <= 1000),
    selected_count integer NOT NULL CHECK (
        selected_count >= 0 AND selected_count <= requested_limit),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_reconciliation_cursor_uidx UNIQUE (
        run_key, prior_target_id)
);

CREATE INDEX IF NOT EXISTS campaign_operations_reconciliation_cursor_head_idx
    ON campaign_operations_reconciliation_cursor_event(
        run_key, reconciliation_cursor_event_id DESC);

ALTER TABLE campaign_operations_reconciliation_observation
    ADD COLUMN IF NOT EXISTS reconciliation_cursor_event_id bigint;

-- The superseded staged design did not persist enough information to
-- reconstruct exact membership.  Never guess from run_key or request ranges
-- during an upgrade: fail atomically if legacy rows exist without membership.
DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM campaign_operations_reconciliation_observation
        WHERE reconciliation_cursor_event_id IS NULL) THEN
        RAISE EXCEPTION
            'migration 049 cannot infer exact reconciliation batch membership from legacy observations';
    END IF;
END $$;

ALTER TABLE campaign_operations_reconciliation_observation
    ALTER COLUMN reconciliation_cursor_event_id SET NOT NULL;
DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'campaign_operations_observation_cursor_fk'
          AND conrelid =
              'campaign_operations_reconciliation_observation'::regclass) THEN
        ALTER TABLE campaign_operations_reconciliation_observation
            ADD CONSTRAINT campaign_operations_observation_cursor_fk
            FOREIGN KEY (reconciliation_cursor_event_id) REFERENCES
                campaign_operations_reconciliation_cursor_event(
                    reconciliation_cursor_event_id) ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'campaign_operations_observation_cursor_request_uidx'
          AND conrelid =
              'campaign_operations_reconciliation_observation'::regclass) THEN
        ALTER TABLE campaign_operations_reconciliation_observation
            ADD CONSTRAINT
                campaign_operations_observation_cursor_request_uidx
            UNIQUE (
                reconciliation_cursor_event_id,
                operational_request_id);
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS campaign_operations_observation_cursor_idx
    ON campaign_operations_reconciliation_observation(
        reconciliation_cursor_event_id, operational_request_id);

CREATE TABLE IF NOT EXISTS campaign_operations_control_audit_reference_event (
    control_audit_reference_event_id bigserial PRIMARY KEY CHECK (
        control_audit_reference_event_id > 0),
    operational_campaign_id bigint NOT NULL,
    operational_request_id bigint,
    control_event_id bigint,
    cancellation_request_id bigint,
    cancellation_settlement_id bigint,
    reconciliation_observation_id bigint,
    reconciliation_resolution_id bigint,
    cause_kind text COLLATE "C" NOT NULL CHECK (
        cause_kind IN (
            'campaign_paused', 'campaign_resumed',
            'cancellation_requested', 'cancellation_settled',
            'reconciliation_observed', 'reconciliation_resolved')),
    actor_identity text COLLATE "C" NOT NULL,
    capability text COLLATE "C" NOT NULL,
    reason text COLLATE "C" NOT NULL,
    prior_version integer,
    resulting_version integer,
    outcome text COLLATE "C" NOT NULL CHECK (
        outcome IN (
            'recorded', 'existing_identical',
            'waiting_for_lease_expiry', 'reconciliation_required')),
    diagnostic_code text COLLATE "C" NOT NULL CHECK (
        diagnostic_code ~ '^[a-z0-9_]{1,128}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_control_audit_campaign_fk FOREIGN KEY (
        operational_campaign_id) REFERENCES
        campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_request_fk FOREIGN KEY (
        operational_request_id) REFERENCES
        campaign_operations_operational_request(operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_control_fk FOREIGN KEY (
        control_event_id) REFERENCES
        campaign_operations_control_event(control_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_cancel_request_fk
        FOREIGN KEY (cancellation_request_id) REFERENCES
        campaign_operations_cancellation_request(cancellation_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_cancel_settlement_fk
        FOREIGN KEY (cancellation_settlement_id) REFERENCES
        campaign_operations_cancellation_settlement(
            cancellation_settlement_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_observation_fk FOREIGN KEY (
        reconciliation_observation_id) REFERENCES
        campaign_operations_reconciliation_observation(
            reconciliation_observation_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_control_audit_resolution_fk FOREIGN KEY (
        reconciliation_resolution_id) REFERENCES
        campaign_operations_reconciliation_resolution(
            reconciliation_resolution_id) ON DELETE RESTRICT
);

ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_reference_e_cause_kind_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_reference_event_cause_kind_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS campaign_ops_dispatch_audit_cause_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    ADD CONSTRAINT
        campaign_ops_dispatch_audit_cause_check
    CHECK (cause_kind IN (
        'dispatch_lease_acquired', 'dispatch_handoff_completed',
        'dispatch_handoff_failed', 'dispatch_lease_recovered'));
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_reference_e_capability_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_reference_event_capability_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_ops_dispatch_audit_capability_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    ADD CONSTRAINT
        campaign_ops_dispatch_audit_capability_check
    CHECK (capability IN (
        'campaign_operations_dispatcher',
        'campaign_operations_phase5_transactional',
        'campaign_operations_cancellation_coordinator',
        'campaign_operations_recovery'));
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    DROP CONSTRAINT IF EXISTS
        campaign_operations_dispatch_audit_shape_check;
ALTER TABLE campaign_operations_dispatch_audit_reference_event
    ADD CONSTRAINT campaign_operations_dispatch_audit_shape_check CHECK (
        (cause_kind = 'dispatch_lease_acquired' AND
            dispatch_attempt_outcome_id IS NULL AND
            capability = 'campaign_operations_dispatcher' AND
            outcome = 'recorded' AND
            resulting_version = prior_version + 1) OR
        (cause_kind IN (
                'dispatch_handoff_completed', 'dispatch_handoff_failed') AND
            dispatch_attempt_outcome_id IS NOT NULL AND
            capability = 'campaign_operations_phase5_transactional') OR
        (cause_kind = 'dispatch_lease_recovered' AND
            dispatch_attempt_outcome_id IS NOT NULL AND
            capability IN (
                'campaign_operations_cancellation_coordinator',
                'campaign_operations_recovery') AND
            outcome = 'recorded' AND
            resulting_version = prior_version + 1));

ALTER TABLE campaign_operations_reservation_event
    ADD COLUMN IF NOT EXISTS cancellation_request_id bigint;
ALTER TABLE campaign_operations_reservation_event
    ADD COLUMN IF NOT EXISTS reconciliation_observation_id bigint;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_reservation_event_cancellation_fk') THEN
        ALTER TABLE campaign_operations_reservation_event
            ADD CONSTRAINT
                campaign_operations_reservation_event_cancellation_fk
            FOREIGN KEY (cancellation_request_id) REFERENCES
                campaign_operations_cancellation_request(
                    cancellation_request_id) ON DELETE RESTRICT;
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_reservation_event_observation_fk') THEN
        ALTER TABLE campaign_operations_reservation_event
            ADD CONSTRAINT
                campaign_operations_reservation_event_observation_fk
            FOREIGN KEY (reconciliation_observation_id) REFERENCES
                campaign_operations_reconciliation_observation(
                    reconciliation_observation_id) ON DELETE RESTRICT;
    END IF;
END $$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_control_chain()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    predecessor campaign_operations_control_event%ROWTYPE;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(
        NEW.campaign_identity_canonical || ';control', 1179402835030005));
    PERFORM 1 FROM campaign_operations_campaign
    WHERE operational_campaign_id = NEW.operational_campaign_id
      AND campaign_identity_canonical = NEW.campaign_identity_canonical
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations control campaign mismatch'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.control_version = 1 THEN
        IF NEW.event_kind <> 'pause' THEN
            RAISE EXCEPTION 'first campaign control must pause'
                USING ERRCODE = '23514';
        END IF;
    ELSE
        SELECT * INTO STRICT predecessor
        FROM campaign_operations_control_event
        WHERE control_event_id = NEW.previous_control_event_id
        FOR UPDATE;
        IF predecessor.operational_campaign_id <>
                NEW.operational_campaign_id OR
           predecessor.control_version <> NEW.control_version - 1 OR
           predecessor.control_identity_canonical <>
                NEW.previous_control_event_identity_canonical OR
           predecessor.event_kind = NEW.event_kind THEN
            RAISE EXCEPTION 'campaign operations control chain mismatch'
                USING ERRCODE = '23514';
        END IF;
    END IF;
    IF EXISTS (
        SELECT 1 FROM campaign_operations_cancellation_request
        WHERE operational_campaign_id = NEW.operational_campaign_id) THEN
        RAISE EXCEPTION 'campaign operations control blocked by cancellation'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.event_kind = 'resume' AND EXISTS (
        SELECT 1
        FROM campaign_operations_reconciliation_observation observation
        LEFT JOIN campaign_operations_reconciliation_resolution resolution
          ON resolution.reconciliation_observation_id =
              observation.reconciliation_observation_id
        WHERE observation.operational_campaign_id =
              NEW.operational_campaign_id
          AND resolution.reconciliation_resolution_id IS NULL) THEN
        RAISE EXCEPTION
            'campaign operations resume blocked by unresolved reconciliation'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
EXCEPTION WHEN no_data_found THEN
    RAISE EXCEPTION 'campaign operations control predecessor missing'
        USING ERRCODE = '23514';
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_future_action_gate()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = NEW.operational_campaign_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations campaign missing'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1 FROM campaign_operations_control_event
        WHERE operational_campaign_id = NEW.operational_campaign_id
        ORDER BY control_version DESC LIMIT 1
    ) AND (
        SELECT event_kind = 'pause'
        FROM campaign_operations_control_event
        WHERE operational_campaign_id = NEW.operational_campaign_id
        ORDER BY control_version DESC LIMIT 1
    ) THEN
        RAISE EXCEPTION 'campaign operations campaign paused'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1 FROM campaign_operations_cancellation_request
        WHERE operational_campaign_id = NEW.operational_campaign_id) THEN
        RAISE EXCEPTION 'campaign operations campaign cancellation requested'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_cancellation_target()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = NEW.operational_campaign_id
      AND campaign_identity_canonical =
          NEW.campaign_identity_canonical
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations cancellation campaign mismatch'
            USING ERRCODE = '23514';
    END IF;
    IF NEW.operational_request_id IS NOT NULL THEN
        PERFORM 1
        FROM campaign_operations_operational_request
        WHERE operational_request_id = NEW.operational_request_id
          AND operational_campaign_id = NEW.operational_campaign_id
          AND request_identity_canonical =
              NEW.request_identity_canonical
          AND request_state = NEW.expected_request_state
          AND state_version = NEW.expected_request_version
        FOR UPDATE;
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations cancellation target mismatch'
                USING ERRCODE = '40001';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
campaign_operations_future_actions_allowed(target_campaign_id bigint)
RETURNS boolean
LANGUAGE sql
STABLE
SECURITY DEFINER
AS $$
    SELECT NOT EXISTS(
        SELECT 1 FROM campaign_operations_cancellation_request
        WHERE operational_campaign_id = target_campaign_id)
    AND NOT COALESCE((
        SELECT event_kind = 'pause'
        FROM campaign_operations_control_event
        WHERE operational_campaign_id = target_campaign_id
        ORDER BY control_version DESC
        LIMIT 1), false);
$$;

CREATE OR REPLACE FUNCTION
guard_campaign_operations_dispatch_control()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE latest_control text;
BEGIN
    IF NEW.request_state NOT IN ('dispatching', 'bound') OR
       NEW.request_state = OLD.request_state THEN
        RETURN NEW;
    END IF;
    PERFORM 1
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = NEW.operational_campaign_id
    FOR UPDATE;
    SELECT event_kind INTO latest_control
    FROM campaign_operations_control_event
    WHERE operational_campaign_id = NEW.operational_campaign_id
    ORDER BY control_version DESC
    LIMIT 1;
    IF latest_control = 'pause' THEN
        RAISE EXCEPTION 'campaign operations dispatch blocked by pause'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1 FROM campaign_operations_cancellation_request
        WHERE operational_campaign_id = NEW.operational_campaign_id) THEN
        RAISE EXCEPTION
            'campaign operations dispatch blocked by cancellation'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_control_audit_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    IF TG_TABLE_NAME = 'campaign_operations_control_event' THEN
        PERFORM 1
        FROM campaign_operations_control_audit_reference_event audit
        WHERE audit.control_event_id = NEW.control_event_id
          AND audit.operational_campaign_id =
              NEW.operational_campaign_id
          AND audit.cause_kind =
              CASE WHEN NEW.event_kind = 'pause'
                   THEN 'campaign_paused'
                   ELSE 'campaign_resumed'
              END
          AND audit.actor_identity = NEW.actor_identity
          AND audit.capability = NEW.capability
          AND audit.reason = NEW.reason
          AND audit.prior_version = NEW.control_version - 1
          AND audit.resulting_version = NEW.control_version
          AND audit.outcome = 'recorded';
    ELSIF TG_TABLE_NAME =
            'campaign_operations_cancellation_request' THEN
        PERFORM 1
        FROM campaign_operations_control_audit_reference_event audit
        WHERE audit.cancellation_request_id =
              NEW.cancellation_request_id
          AND audit.operational_campaign_id =
              NEW.operational_campaign_id
          AND audit.operational_request_id IS NOT DISTINCT FROM
              NEW.operational_request_id
          AND audit.cause_kind = 'cancellation_requested'
          AND audit.actor_identity = NEW.actor_identity
          AND audit.capability = NEW.capability
          AND audit.reason = NEW.reason
          AND audit.outcome = 'recorded';
    ELSIF TG_TABLE_NAME =
            'campaign_operations_cancellation_settlement' THEN
        PERFORM 1
        FROM campaign_operations_control_audit_reference_event audit
        JOIN campaign_operations_cancellation_request request
          ON request.cancellation_request_id =
              NEW.cancellation_request_id
        WHERE audit.cancellation_settlement_id =
              NEW.cancellation_settlement_id
          AND audit.cancellation_request_id =
              NEW.cancellation_request_id
          AND audit.operational_campaign_id =
              request.operational_campaign_id
          AND audit.operational_request_id IS NOT DISTINCT FROM
              request.operational_request_id
          AND audit.cause_kind = 'cancellation_settled'
          AND audit.outcome = 'recorded';
    ELSIF TG_TABLE_NAME =
            'campaign_operations_reconciliation_observation' THEN
        PERFORM 1
        FROM campaign_operations_control_audit_reference_event audit
        WHERE audit.reconciliation_observation_id =
              NEW.reconciliation_observation_id
          AND audit.operational_campaign_id =
              NEW.operational_campaign_id
          AND audit.operational_request_id =
              NEW.operational_request_id
          AND audit.cause_kind = 'reconciliation_observed'
          AND audit.outcome = 'recorded';
    ELSIF TG_TABLE_NAME =
            'campaign_operations_reconciliation_resolution' THEN
        PERFORM 1
        FROM campaign_operations_control_audit_reference_event audit
        JOIN campaign_operations_reconciliation_observation observation
          ON observation.reconciliation_observation_id =
              NEW.reconciliation_observation_id
        WHERE audit.reconciliation_resolution_id =
              NEW.reconciliation_resolution_id
          AND audit.reconciliation_observation_id =
              NEW.reconciliation_observation_id
          AND audit.operational_campaign_id =
              observation.operational_campaign_id
          AND audit.operational_request_id =
              observation.operational_request_id
          AND audit.cause_kind = 'reconciliation_resolved'
          AND audit.actor_identity = NEW.owning_capability
          AND audit.capability = NEW.owning_capability
          AND audit.reason = observation.diagnostic_code
          AND audit.diagnostic_code = observation.diagnostic_code
          AND audit.outcome = 'recorded';
    ELSE
        RAISE EXCEPTION
            'unsupported campaign operations control audit source'
            USING ERRCODE = '23514';
    END IF;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations control audit evidence incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_control_audit_complete';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_phase4_request_transition_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    IF NEW.request_state = 'cancelled' AND
       OLD.request_state IN ('ready', 'dispatching') THEN
        PERFORM 1
        FROM campaign_operations_cancellation_request cancellation
        JOIN campaign_operations_cancellation_settlement settlement
          ON settlement.cancellation_request_id =
             cancellation.cancellation_request_id
        JOIN campaign_operations_reservation_event reservation_event
          ON reservation_event.reservation_event_id =
             settlement.reservation_event_id
         AND reservation_event.cancellation_request_id =
             cancellation.cancellation_request_id
        WHERE cancellation.operational_request_id =
                  NEW.operational_request_id
          AND cancellation.request_identity_canonical =
                  NEW.request_identity_canonical
          AND cancellation.expected_request_state = OLD.request_state
          AND cancellation.expected_request_version = OLD.state_version
          AND settlement.disposition = 'unbound_cancelled'
          AND settlement.resulting_request_version = NEW.state_version
          AND reservation_event.operational_request_id =
                  NEW.operational_request_id
          AND reservation_event.resulting_state = 'released';
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations cancellation transition incomplete'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                        'campaign_operations_phase4_request_transition_complete';
        END IF;
    ELSIF OLD.request_state = 'dispatching' AND
          NEW.request_state = 'ready' THEN
        PERFORM 1
        FROM campaign_operations_dispatch_attempt attempt
        JOIN campaign_operations_dispatch_attempt_outcome outcome
          ON outcome.dispatch_attempt_id = attempt.dispatch_attempt_id
        JOIN campaign_operations_dispatch_audit_reference_event audit
          ON audit.dispatch_attempt_outcome_id =
             outcome.dispatch_attempt_outcome_id
         AND audit.dispatch_attempt_id = attempt.dispatch_attempt_id
        JOIN campaign_operations_reconciliation_observation observation
          ON observation.operational_request_id =
             NEW.operational_request_id
         AND observation.expected_request_state = OLD.request_state
         AND observation.expected_request_version = OLD.state_version
         AND observation.reason_code =
             'dispatch_lease_expired_no_downstream_evidence'
        JOIN campaign_operations_reconciliation_resolution resolution
          ON resolution.reconciliation_observation_id =
             observation.reconciliation_observation_id
         AND resolution.transition_identity_canonical =
             outcome.outcome_identity_canonical
         AND resolution.transition_identity_hash =
             outcome.outcome_identity_hash
        WHERE attempt.operational_request_id =
                  NEW.operational_request_id
          AND attempt.attempt_ordinal = (
              SELECT max(latest.attempt_ordinal)
              FROM campaign_operations_dispatch_attempt latest
              WHERE latest.operational_request_id =
                    NEW.operational_request_id)
          AND outcome.expected_request_version = OLD.state_version
          AND outcome.resulting_request_version = NEW.state_version
          AND outcome.diagnostic_code =
              'dispatch_lease_expired_no_downstream_evidence'
          AND audit.cause_kind = 'dispatch_lease_recovered'
          AND audit.capability = 'campaign_operations_recovery'
          AND audit.prior_version = OLD.state_version
          AND audit.resulting_version = NEW.state_version
          AND resolution.owning_capability =
              'campaign_operations_recovery'
          AND resolution.resolution_disposition =
              'request_returned_ready';
        IF NOT FOUND THEN
            RAISE EXCEPTION
                'campaign operations recovery transition incomplete'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                        'campaign_operations_phase4_request_transition_complete';
        END IF;
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
append_campaign_operations_cancellation_resolution(
    target_observation_id bigint,
    target_transition_canonical text,
    target_transition_hash text,
    target_resolution_canonical text,
    target_resolution_hash text)
RETURNS campaign_operations_reconciliation_resolution
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    observation campaign_operations_reconciliation_observation%ROWTYPE;
    settlement campaign_operations_cancellation_settlement%ROWTYPE;
    existing campaign_operations_reconciliation_resolution%ROWTYPE;
    result campaign_operations_reconciliation_resolution%ROWTYPE;
BEGIN
    SELECT * INTO STRICT observation
    FROM campaign_operations_reconciliation_observation
    WHERE reconciliation_observation_id = target_observation_id
    FOR UPDATE;

    SELECT * INTO existing
    FROM campaign_operations_reconciliation_resolution
    WHERE reconciliation_observation_id = target_observation_id;
    IF existing.reconciliation_resolution_id IS NOT NULL THEN
        IF existing.cancellation_settlement_id IS NULL OR
           existing.dispatch_attempt_outcome_id IS NOT NULL OR
           existing.transition_identity_canonical <>
               target_transition_canonical OR
           existing.transition_identity_hash <> target_transition_hash OR
           existing.resolution_identity_canonical <>
               target_resolution_canonical OR
           existing.resolution_identity_hash <> target_resolution_hash THEN
            RAISE EXCEPTION
                'campaign operations cancellation resolution conflict'
                USING ERRCODE = '23514';
        END IF;
        RETURN existing;
    END IF;

    IF observation.reason_code <> 'cancellation_settlement_pending' OR
       observation.recommended_service <>
           'campaign_operations_cancellation_coordinator' OR
       observation.recommended_action <> 'replay_cancellation' OR
       observation.diagnostic_code <>
           'cancellation_settlement_pending' THEN
        RAISE EXCEPTION
            'campaign operations cancellation observation is not actionable'
            USING ERRCODE = '23514';
    END IF;

    BEGIN
        SELECT causal_settlement.* INTO STRICT settlement
        FROM campaign_operations_cancellation_settlement causal_settlement
        JOIN campaign_operations_cancellation_request cancellation
          ON cancellation.cancellation_request_id =
             causal_settlement.cancellation_request_id
        WHERE causal_settlement.settlement_identity_canonical =
                  target_transition_canonical
          AND causal_settlement.settlement_identity_hash =
                  target_transition_hash
          AND cancellation.operational_campaign_id =
                  observation.operational_campaign_id
          AND cancellation.operational_request_id =
                  observation.operational_request_id
          AND cancellation.request_identity_canonical =
                  observation.request_identity_canonical
          AND cancellation.expected_request_state =
                  observation.expected_request_state
          AND cancellation.expected_request_version =
                  observation.expected_request_version
          AND EXISTS (
              SELECT 1
              FROM campaign_operations_control_audit_reference_event audit
              WHERE audit.cancellation_request_id =
                        cancellation.cancellation_request_id
                AND audit.cancellation_settlement_id =
                        causal_settlement.cancellation_settlement_id
                AND audit.operational_campaign_id =
                        observation.operational_campaign_id
                AND audit.operational_request_id =
                        observation.operational_request_id
                AND audit.cause_kind = 'cancellation_settled'
                AND audit.capability =
                        'campaign_operations_cancellation_coordinator'
                AND audit.outcome = 'recorded')
        FOR UPDATE OF causal_settlement;
    EXCEPTION WHEN no_data_found OR too_many_rows THEN
        RAISE EXCEPTION
            'campaign operations cancellation causal evidence mismatch'
            USING ERRCODE = '23514';
    END;

    INSERT INTO campaign_operations_reconciliation_resolution(
        reconciliation_observation_id, observation_identity_canonical,
        owning_service, owning_capability, transition_identity_canonical,
        transition_identity_hash, cancellation_settlement_id,
        resolution_disposition, resolution_contract_version,
        resolution_identity_canonical, resolution_identity_hash)
    VALUES(target_observation_id,
        observation.observation_identity_canonical,
        'campaign_operations_cancellation_coordinator',
        'campaign_operations_cancellation_coordinator',
        settlement.settlement_identity_canonical,
        settlement.settlement_identity_hash,
        settlement.cancellation_settlement_id,
        'cancellation_settled', 1,
        target_resolution_canonical, target_resolution_hash)
    RETURNING * INTO result;
    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION
append_campaign_operations_recovery_resolution(
    target_observation_id bigint,
    target_transition_canonical text,
    target_transition_hash text,
    target_resolution_disposition text,
    target_resolution_canonical text,
    target_resolution_hash text)
RETURNS campaign_operations_reconciliation_resolution
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    observation campaign_operations_reconciliation_observation%ROWTYPE;
    outcome campaign_operations_dispatch_attempt_outcome%ROWTYPE;
    existing campaign_operations_reconciliation_resolution%ROWTYPE;
    derived_disposition text;
    result campaign_operations_reconciliation_resolution%ROWTYPE;
BEGIN
    SELECT * INTO STRICT observation
    FROM campaign_operations_reconciliation_observation
    WHERE reconciliation_observation_id = target_observation_id
    FOR UPDATE;

    SELECT * INTO existing
    FROM campaign_operations_reconciliation_resolution
    WHERE reconciliation_observation_id = target_observation_id;
    IF existing.reconciliation_resolution_id IS NOT NULL THEN
        IF existing.cancellation_settlement_id IS NOT NULL OR
           existing.dispatch_attempt_outcome_id IS NULL OR
           existing.transition_identity_canonical <>
               target_transition_canonical OR
           existing.transition_identity_hash <> target_transition_hash OR
           existing.resolution_disposition <>
               target_resolution_disposition OR
           existing.resolution_identity_canonical <>
               target_resolution_canonical OR
           existing.resolution_identity_hash <> target_resolution_hash THEN
            RAISE EXCEPTION
                'campaign operations recovery resolution conflict'
                USING ERRCODE = '23514';
        END IF;
        RETURN existing;
    END IF;

    IF observation.reason_code <>
           'dispatch_lease_expired_no_downstream_evidence' OR
       observation.recommended_service <>
           'campaign_operations_dispatch_recovery' OR
       observation.recommended_action <> 'clear_stale_dispatch_lease' OR
       observation.diagnostic_code <>
           'dispatch_lease_expired_no_downstream_evidence' THEN
        RAISE EXCEPTION
            'campaign operations recovery observation is not actionable'
            USING ERRCODE = '23514';
    END IF;

    BEGIN
        SELECT causal_outcome.* INTO STRICT outcome
        FROM campaign_operations_dispatch_attempt_outcome causal_outcome
        JOIN campaign_operations_dispatch_attempt attempt
          ON attempt.dispatch_attempt_id =
             causal_outcome.dispatch_attempt_id
        WHERE causal_outcome.outcome_identity_canonical =
                  target_transition_canonical
          AND causal_outcome.outcome_identity_hash =
                  target_transition_hash
          AND attempt.operational_request_id =
                  observation.operational_request_id
          AND causal_outcome.result_classification = 'rejected'
          AND causal_outcome.downstream_evidence_classification =
                  'no_phase5_evidence'
          AND causal_outcome.semantic_conflict_classification = 'none'
          AND causal_outcome.uncertain_commit_recovery_classification =
                  'proven_no_commit'
          AND causal_outcome.diagnostic_code =
                  observation.diagnostic_code
          AND causal_outcome.expected_request_version =
                  observation.expected_request_version
          AND causal_outcome.resulting_request_version =
                  observation.expected_request_version + 1
          AND EXISTS (
              SELECT 1
              FROM campaign_operations_dispatch_audit_reference_event audit
              WHERE audit.dispatch_attempt_id =
                        attempt.dispatch_attempt_id
                AND audit.dispatch_attempt_outcome_id =
                        causal_outcome.dispatch_attempt_outcome_id
                AND audit.operational_campaign_id =
                        observation.operational_campaign_id
                AND audit.operational_request_id =
                        observation.operational_request_id
                AND audit.cause_kind = 'dispatch_lease_recovered'
                AND audit.capability = 'campaign_operations_recovery'
                AND audit.prior_version =
                        observation.expected_request_version
                AND audit.resulting_version =
                        observation.expected_request_version + 1
                AND audit.outcome = 'recorded')
        FOR UPDATE OF causal_outcome;
    EXCEPTION WHEN no_data_found OR too_many_rows THEN
        RAISE EXCEPTION
            'campaign operations recovery causal evidence mismatch'
            USING ERRCODE = '23514';
    END;

    IF EXISTS (
        SELECT 1
        FROM campaign_operations_reconciliation_resolution resolution
        WHERE resolution.dispatch_attempt_outcome_id =
              outcome.dispatch_attempt_outcome_id) THEN
        derived_disposition := 'already_resolved';
    ELSE
        derived_disposition := 'request_returned_ready';
    END IF;
    IF target_resolution_disposition <> derived_disposition THEN
        RAISE EXCEPTION
            'campaign operations recovery disposition mismatch'
            USING ERRCODE = '23514';
    END IF;

    INSERT INTO campaign_operations_reconciliation_resolution(
        reconciliation_observation_id, observation_identity_canonical,
        owning_service, owning_capability, transition_identity_canonical,
        transition_identity_hash, dispatch_attempt_outcome_id,
        resolution_disposition, resolution_contract_version,
        resolution_identity_canonical, resolution_identity_hash)
    VALUES(target_observation_id,
        observation.observation_identity_canonical,
        'campaign_operations_dispatch_recovery',
        'campaign_operations_recovery',
        outcome.outcome_identity_canonical,
        outcome.outcome_identity_hash,
        outcome.dispatch_attempt_outcome_id,
        derived_disposition, 1,
        target_resolution_canonical, target_resolution_hash)
    RETURNING * INTO result;
    RETURN result;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_reconciliation_cursor_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    cursor_id bigint;
    cursor_row campaign_operations_reconciliation_cursor_event%ROWTYPE;
    observation_count bigint;
    distinct_request_count bigint;
    minimum_request_id bigint;
    maximum_request_id bigint;
    mismatched_run_count bigint;
    out_of_bounds_count bigint;
BEGIN
    cursor_id := CASE
        WHEN TG_TABLE_NAME =
             'campaign_operations_reconciliation_cursor_event'
        THEN NEW.reconciliation_cursor_event_id
        ELSE NEW.reconciliation_cursor_event_id
    END;
    SELECT * INTO STRICT cursor_row
    FROM campaign_operations_reconciliation_cursor_event
    WHERE reconciliation_cursor_event_id = cursor_id;
    SELECT count(*), count(DISTINCT operational_request_id),
           min(operational_request_id), max(operational_request_id),
           count(*) FILTER (WHERE run_key <> cursor_row.run_key),
           count(*) FILTER (
               WHERE operational_request_id <= cursor_row.prior_target_id OR
                     operational_request_id > cursor_row.last_target_id)
      INTO observation_count, distinct_request_count,
           minimum_request_id, maximum_request_id,
           mismatched_run_count, out_of_bounds_count
    FROM campaign_operations_reconciliation_observation
    WHERE reconciliation_cursor_event_id = cursor_id;

    IF observation_count <> cursor_row.selected_count OR
       distinct_request_count <> observation_count OR
       mismatched_run_count <> 0 OR out_of_bounds_count <> 0 OR
       (observation_count = 0 AND
            cursor_row.last_target_id <> cursor_row.prior_target_id) OR
       (observation_count > 0 AND
            (minimum_request_id <= cursor_row.prior_target_id OR
             maximum_request_id <> cursor_row.last_target_id)) THEN
        RAISE EXCEPTION
            'campaign operations reconciliation cursor batch incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                    'campaign_operations_reconciliation_cursor_complete';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
transition_campaign_operations_request_cancelled(
    target_request_id bigint, expected_version integer)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_operational_request%ROWTYPE;
BEGIN
    UPDATE campaign_operations_operational_request
       SET request_state = 'cancelled',
           state_version = state_version + 1,
           lease_token_hash = NULL,
           lease_expires_at = NULL,
           dispatcher_identity = NULL,
           updated_at = transaction_timestamp()
     WHERE operational_request_id = target_request_id
       AND state_version = expected_version
       AND request_state IN ('ready', 'dispatching')
       AND NOT EXISTS (
           SELECT 1 FROM campaign_operations_request_binding
           WHERE operational_request_id = target_request_id)
       AND NOT EXISTS (
           SELECT 1
           FROM campaign_operations_operational_request request
           JOIN experiment_recommendation_campaign_materialization_member
               member
             ON member.recommendation_campaign_materialization_id =
                request.recommendation_campaign_materialization_id
           JOIN experiment_recommendation_conversion_execution execution
             ON execution.recommendation_conversion_proposal_id =
                member.recommendation_conversion_proposal_id
           WHERE request.operational_request_id = target_request_id)
       AND (
           request_state = 'ready' OR NOT EXISTS (
               SELECT 1
               FROM campaign_operations_dispatch_attempt attempt
               JOIN campaign_operations_dispatch_attempt_outcome outcome
                 ON outcome.dispatch_attempt_id =
                    attempt.dispatch_attempt_id
               WHERE attempt.operational_request_id =
                    target_request_id
                 AND attempt.attempt_ordinal = (
                     SELECT max(latest.attempt_ordinal)
                     FROM campaign_operations_dispatch_attempt latest
                     WHERE latest.operational_request_id =
                         target_request_id)))
    RETURNING * INTO changed;
    IF changed.operational_request_id IS NULL THEN
        RAISE EXCEPTION 'campaign operations request cancellation conflict'
            USING ERRCODE = '40001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION
transition_campaign_operations_reservation_released(
    target_reservation_id bigint, expected_version integer)
RETURNS campaign_operations_reservation
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_reservation%ROWTYPE;
BEGIN
    UPDATE campaign_operations_reservation
       SET reservation_state = 'released',
           state_version = state_version + 1,
           updated_at = transaction_timestamp()
     WHERE reservation_id = target_reservation_id
       AND state_version = expected_version
       AND reservation_state = 'held'
    RETURNING * INTO changed;
    IF changed.reservation_id IS NULL THEN
        RAISE EXCEPTION 'campaign operations reservation release conflict'
            USING ERRCODE = '40001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION
transition_campaign_operations_request_ready_recovered(
    target_request_id bigint, expected_version integer)
RETURNS campaign_operations_operational_request
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE changed campaign_operations_operational_request%ROWTYPE;
BEGIN
    UPDATE campaign_operations_operational_request
       SET request_state = 'ready',
           state_version = state_version + 1,
           lease_token_hash = NULL,
           lease_expires_at = NULL,
           dispatcher_identity = NULL,
           updated_at = transaction_timestamp()
     WHERE operational_request_id = target_request_id
       AND state_version = expected_version
       AND request_state = 'dispatching'
       AND lease_expires_at <= transaction_timestamp()
       AND NOT EXISTS (
           SELECT 1 FROM campaign_operations_request_binding
           WHERE operational_request_id = target_request_id)
       AND NOT EXISTS (
           SELECT 1
           FROM campaign_operations_operational_request request
           JOIN experiment_recommendation_campaign_materialization_member
               member
             ON member.recommendation_campaign_materialization_id =
                request.recommendation_campaign_materialization_id
           JOIN experiment_recommendation_conversion_execution execution
             ON execution.recommendation_conversion_proposal_id =
                member.recommendation_conversion_proposal_id
           WHERE request.operational_request_id = target_request_id)
       AND NOT EXISTS (
           SELECT 1
           FROM campaign_operations_dispatch_attempt attempt
           JOIN campaign_operations_dispatch_attempt_outcome outcome
             ON outcome.dispatch_attempt_id = attempt.dispatch_attempt_id
           WHERE attempt.operational_request_id = target_request_id
             AND attempt.attempt_ordinal = (
                 SELECT max(latest.attempt_ordinal)
                 FROM campaign_operations_dispatch_attempt latest
                 WHERE latest.operational_request_id =
                     target_request_id))
    RETURNING * INTO changed;
    IF changed.operational_request_id IS NULL THEN
        RAISE EXCEPTION 'campaign operations lease recovery conflict'
            USING ERRCODE = '40001';
    END IF;
    RETURN changed;
END;
$$;

CREATE OR REPLACE FUNCTION apply_experiment_lifecycle_cancellation(
    target_cancellation_request_id bigint,
    target_control_owner_id bigint,
    target_experiment_id bigint,
    expected_status_value text,
    expected_phase_value text,
    actor_value text,
    event_canonical_value text,
    event_hash_value text)
RETURNS experiment_lifecycle_cancellation_event
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    current_experiment experiment%ROWTYPE;
    disposition_value text;
    resulting_status_value text;
    result experiment_lifecycle_cancellation_event%ROWTYPE;
BEGIN
    PERFORM 1
    FROM campaign_operations_cancellation_request cancellation
    JOIN campaign_operations_downstream_control_owner control_owner
      ON control_owner.operational_request_id =
         cancellation.operational_request_id
    WHERE cancellation.cancellation_request_id =
              target_cancellation_request_id
      AND cancellation.actor_identity = actor_value
      AND control_owner.downstream_control_owner_id =
              target_control_owner_id
      AND control_owner.experiment_id = target_experiment_id;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'lifecycle cancellation target mismatch'
            USING ERRCODE = '23514';
    END IF;
    SELECT * INTO result
    FROM experiment_lifecycle_cancellation_event
    WHERE cancellation_request_id = target_cancellation_request_id
      AND downstream_control_owner_id = target_control_owner_id;
    IF result.lifecycle_cancellation_event_id IS NOT NULL THEN
        RETURN result;
    END IF;
    SELECT * INTO STRICT current_experiment
    FROM experiment WHERE experiment_id = target_experiment_id FOR UPDATE;
    IF current_experiment.status <> expected_status_value OR
       current_experiment.phase <> expected_phase_value THEN
        RAISE EXCEPTION 'experiment lifecycle changed during cancellation'
            USING ERRCODE = '40001';
    END IF;
    IF current_experiment.status IN ('pending', 'paused') THEN
        disposition_value := 'accepted';
        resulting_status_value := 'cancelled';
        UPDATE experiment
           SET status = 'cancelled', completed_at = transaction_timestamp(),
               updated_at = transaction_timestamp()
         WHERE experiment_id = target_experiment_id;
    ELSIF current_experiment.status IN (
            'completed', 'failed', 'cancelled') THEN
        disposition_value := 'already_terminal';
        resulting_status_value := current_experiment.status;
    ELSIF current_experiment.status = 'running' THEN
        disposition_value := 'running_not_supported';
        resulting_status_value := current_experiment.status;
    ELSE
        RAISE EXCEPTION 'unsupported experiment lifecycle state'
            USING ERRCODE = '23514';
    END IF;
    INSERT INTO experiment_lifecycle_cancellation_event(
        cancellation_request_id,
        cancellation_request_identity_canonical,
        downstream_control_owner_id, experiment_id,
        expected_status, expected_phase, observed_status, observed_phase,
        resulting_status, disposition, actor_identity,
        event_contract_version, event_identity_canonical,
        event_identity_hash)
    SELECT target_cancellation_request_id,
           cancellation_identity_canonical,
           target_control_owner_id, target_experiment_id,
           expected_status_value, expected_phase_value,
           current_experiment.status, current_experiment.phase,
           resulting_status_value, disposition_value, actor_value,
           1, event_canonical_value, event_hash_value
    FROM campaign_operations_cancellation_request
    WHERE cancellation_request_id = target_cancellation_request_id
    RETURNING * INTO result;
    RETURN result;
EXCEPTION WHEN unique_violation THEN
    SELECT * INTO STRICT result
    FROM experiment_lifecycle_cancellation_event
    WHERE cancellation_request_id = target_cancellation_request_id
      AND downstream_control_owner_id = target_control_owner_id;
    RETURN result;
END;
$$;

DO $$
DECLARE phase4_schema text := current_schema();
DECLARE function_name text;
BEGIN
    FOREACH function_name IN ARRAY ARRAY[
        'enforce_campaign_operations_control_chain()',
        'enforce_campaign_operations_future_action_gate()',
        'enforce_campaign_operations_cancellation_target()',
        'campaign_operations_future_actions_allowed(bigint)',
        'guard_campaign_operations_dispatch_control()',
        'enforce_campaign_operations_control_audit_complete()',
        'enforce_campaign_operations_phase4_request_transition_complete()',
        'append_campaign_operations_cancellation_resolution(bigint,text,text,text,text)',
        'append_campaign_operations_recovery_resolution(bigint,text,text,text,text,text)',
        'enforce_campaign_operations_reconciliation_cursor_complete()',
        'transition_campaign_operations_request_cancelled(bigint,integer)',
        'transition_campaign_operations_reservation_released(bigint,integer)',
        'transition_campaign_operations_request_ready_recovered(bigint,integer)',
        'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)']
    LOOP
        EXECUTE format(
            'ALTER FUNCTION %I.%s SET search_path TO pg_catalog, %I, pg_temp',
            phase4_schema, function_name, phase4_schema);
    END LOOP;
END $$;

DROP TRIGGER IF EXISTS campaign_operations_control_chain_trigger
    ON campaign_operations_control_event;
CREATE TRIGGER campaign_operations_control_chain_trigger
BEFORE INSERT ON campaign_operations_control_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_chain();

DROP TRIGGER IF EXISTS
    campaign_operations_cancellation_target_trigger
    ON campaign_operations_cancellation_request;
CREATE TRIGGER campaign_operations_cancellation_target_trigger
BEFORE INSERT ON campaign_operations_cancellation_request
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_cancellation_target();

DROP TRIGGER IF EXISTS campaign_operations_future_action_gate_trigger
    ON campaign_operations_reservation;
CREATE TRIGGER campaign_operations_future_action_gate_trigger
BEFORE INSERT ON campaign_operations_reservation
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_future_action_gate();

DROP TRIGGER IF EXISTS
    campaign_operations_dispatch_control_gate_trigger
    ON campaign_operations_operational_request;
CREATE TRIGGER campaign_operations_dispatch_control_gate_trigger
BEFORE UPDATE ON campaign_operations_operational_request
FOR EACH ROW EXECUTE FUNCTION
    guard_campaign_operations_dispatch_control();

DROP TRIGGER IF EXISTS
    campaign_operations_control_audit_complete_trigger
    ON campaign_operations_control_event;
CREATE CONSTRAINT TRIGGER
    campaign_operations_control_audit_complete_trigger
AFTER INSERT ON campaign_operations_control_event
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_audit_complete();
DROP TRIGGER IF EXISTS
    campaign_operations_cancellation_request_audit_complete_trigger
    ON campaign_operations_cancellation_request;
CREATE CONSTRAINT TRIGGER
    campaign_operations_cancellation_request_audit_complete_trigger
AFTER INSERT ON campaign_operations_cancellation_request
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_audit_complete();
DROP TRIGGER IF EXISTS
    campaign_operations_cancellation_settlement_audit_complete_trigger
    ON campaign_operations_cancellation_settlement;
CREATE CONSTRAINT TRIGGER
    campaign_operations_cancellation_settlement_audit_complete_trigger
AFTER INSERT ON campaign_operations_cancellation_settlement
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_audit_complete();
DROP TRIGGER IF EXISTS
    campaign_operations_reconciliation_observation_audit_complete_trigger
    ON campaign_operations_reconciliation_observation;
CREATE CONSTRAINT TRIGGER
    campaign_operations_reconciliation_observation_audit_complete_trigger
AFTER INSERT ON campaign_operations_reconciliation_observation
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_audit_complete();
DROP TRIGGER IF EXISTS
    campaign_operations_reconciliation_resolution_audit_complete_trigger
    ON campaign_operations_reconciliation_resolution;
CREATE CONSTRAINT TRIGGER
    campaign_operations_reconciliation_resolution_audit_complete_trigger
AFTER INSERT ON campaign_operations_reconciliation_resolution
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_control_audit_complete();

DROP TRIGGER IF EXISTS
    campaign_operations_reconciliation_cursor_complete_trigger
    ON campaign_operations_reconciliation_cursor_event;
CREATE CONSTRAINT TRIGGER
    campaign_operations_reconciliation_cursor_complete_trigger
AFTER INSERT ON campaign_operations_reconciliation_cursor_event
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_reconciliation_cursor_complete();
DROP TRIGGER IF EXISTS
    campaign_operations_reconciliation_membership_complete_trigger
    ON campaign_operations_reconciliation_observation;
CREATE CONSTRAINT TRIGGER
    campaign_operations_reconciliation_membership_complete_trigger
AFTER INSERT ON campaign_operations_reconciliation_observation
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_reconciliation_cursor_complete();

DROP TRIGGER IF EXISTS
    campaign_operations_phase4_request_transition_complete_trigger
    ON campaign_operations_operational_request;
CREATE CONSTRAINT TRIGGER
    campaign_operations_phase4_request_transition_complete_trigger
AFTER UPDATE ON campaign_operations_operational_request
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_phase4_request_transition_complete();

COMMENT ON TABLE campaign_operations_control_event IS
    'Append-only campaign-only pause/resume chain; never worker control.';
COMMENT ON TABLE campaign_operations_cancellation_request IS
    'Immutable cancellation intent, separate from settlement.';
COMMENT ON TABLE campaign_operations_cancellation_settlement IS
    'Immutable authoritative cancellation disposition.';
COMMENT ON TABLE campaign_operations_reconciliation_observation IS
    'Detection-only immutable evidence linked to one exact durable cursor batch; grants no repair authority.';
COMMENT ON COLUMN
    campaign_operations_reconciliation_observation.
        reconciliation_cursor_event_id IS
    'Exact durable batch membership; replay never infers membership from ranges or current state.';
COMMENT ON TABLE campaign_operations_reconciliation_resolution IS
    'Immutable typed edge to PostgreSQL-validated owning-service causal evidence.';

REVOKE ALL PRIVILEGES ON
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution,
    campaign_operations_reconciliation_cursor_event,
    campaign_operations_control_audit_reference_event,
    experiment_lifecycle_cancellation_event
    FROM PUBLIC, pqxx,
         campaign_operations_reader,
         campaign_operations_controller,
         campaign_operations_cancellation_coordinator,
         campaign_operations_reconciler,
         campaign_operations_recovery,
         experiment_lifecycle_cancellation,
         experiment_lifecycle_cancellation_owner;

REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_control_chain(),
    enforce_campaign_operations_future_action_gate(),
    enforce_campaign_operations_cancellation_target(),
    campaign_operations_future_actions_allowed(bigint),
    guard_campaign_operations_dispatch_control(),
    enforce_campaign_operations_control_audit_complete(),
    enforce_campaign_operations_phase4_request_transition_complete(),
    append_campaign_operations_cancellation_resolution(
        bigint, text, text, text, text),
    append_campaign_operations_recovery_resolution(
        bigint, text, text, text, text, text),
    enforce_campaign_operations_reconciliation_cursor_complete(),
    transition_campaign_operations_request_cancelled(bigint, integer),
    transition_campaign_operations_reservation_released(bigint, integer),
    transition_campaign_operations_request_ready_recovered(bigint, integer),
    apply_experiment_lifecycle_cancellation(
        bigint, bigint, bigint, text, text, text, text, text)
    FROM PUBLIC, pqxx,
         campaign_operations_reader,
         campaign_operations_controller,
         campaign_operations_cancellation_coordinator,
         campaign_operations_reconciler,
         campaign_operations_recovery,
         experiment_lifecycle_cancellation,
         experiment_lifecycle_cancellation_owner;

GRANT EXECUTE ON FUNCTION
    campaign_operations_future_actions_allowed(bigint)
    TO campaign_operations_campaign_creator,
       campaign_operations_authorizer,
       campaign_operations_budget_administrator,
       campaign_operations_request_acceptor,
       campaign_operations_dispatcher,
       campaign_operations_phase5_transactional,
       campaign_operations_controller,
       campaign_operations_cancellation_coordinator,
       campaign_operations_reconciler,
       campaign_operations_recovery;

GRANT SELECT ON
    campaign_operations_control_event,
    campaign_operations_cancellation_request
    TO campaign_operations_request_acceptor,
       campaign_operations_dispatcher,
       campaign_operations_phase5_transactional;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution
    TO campaign_operations_reader;

GRANT SELECT ON
    experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_conversion_execution
    TO campaign_operations_owner,
       campaign_operations_controller,
       campaign_operations_cancellation_coordinator,
       campaign_operations_reconciler,
       campaign_operations_recovery;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution
    TO campaign_operations_controller;
GRANT INSERT (
    operational_campaign_id, campaign_identity_canonical,
    previous_control_event_id, previous_control_event_identity_canonical,
    control_version, event_kind, actor_identity, capability, reason,
    control_contract_version, control_identity_canonical,
    control_identity_hash)
    ON campaign_operations_control_event
    TO campaign_operations_controller;
GRANT INSERT (
    operational_campaign_id, control_event_id, cause_kind, actor_identity,
    capability, reason, prior_version, resulting_version, outcome,
    diagnostic_code)
    ON campaign_operations_control_audit_reference_event
    TO campaign_operations_controller;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    experiment_lifecycle_cancellation_event,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution,
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_conversion_execution
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    operational_campaign_id, campaign_identity_canonical,
    operational_request_id, request_identity_canonical,
    expected_request_state, expected_request_version, cancellation_scope,
    operation_key, actor_identity, capability, reason,
    cancellation_contract_version, cancellation_identity_canonical,
    cancellation_identity_hash)
    ON campaign_operations_cancellation_request
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    cancellation_request_id, cancellation_request_identity_canonical,
    disposition, reservation_event_id,
    reservation_event_identity_canonical, resulting_request_version,
    lifecycle_evidence_identity_canonical,
    lifecycle_evidence_identity_hash, settlement_contract_version,
    settlement_identity_canonical, settlement_identity_hash)
    ON campaign_operations_cancellation_settlement
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    reservation_id, reservation_identity_canonical, transition_kind,
    expected_state, resulting_state, expected_version, resulting_version,
    operational_request_id, request_identity_canonical, amount,
    reservation_event_contract_version,
    reservation_event_identity_canonical,
    reservation_event_identity_hash, cancellation_request_id,
    reconciliation_observation_id)
    ON campaign_operations_reservation_event
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    operational_campaign_id, operational_request_id,
    cancellation_request_id, cancellation_settlement_id, cause_kind,
    actor_identity, capability, reason, prior_version, resulting_version,
    outcome, diagnostic_code)
    ON campaign_operations_control_audit_reference_event
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    operational_campaign_id, operational_request_id,
    reconciliation_observation_id, reconciliation_resolution_id,
    cause_kind, actor_identity, capability, reason, prior_version,
    resulting_version, outcome, diagnostic_code)
    ON campaign_operations_control_audit_reference_event
    TO campaign_operations_cancellation_coordinator;
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
    TO campaign_operations_cancellation_coordinator;
GRANT INSERT (
    operational_campaign_id, operational_request_id, dispatch_attempt_id,
    dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
    prior_version, resulting_version, outcome, replay_disposition,
    diagnostic_code)
    ON campaign_operations_dispatch_audit_reference_event
    TO campaign_operations_cancellation_coordinator;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_request_binding,
    campaign_operations_downstream_control_owner,
    campaign_operations_control_event,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution,
    campaign_operations_reconciliation_cursor_event,
    experiment_lifecycle_cancellation_event,
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_conversion_execution,
    experiment_recommendation_conversion_activation,
    experiment
    TO campaign_operations_reconciler;
GRANT INSERT (
    reconciliation_cursor_event_id, run_key, operational_campaign_id,
    operational_request_id,
    request_identity_canonical, expected_request_state,
    expected_request_version, reason_code, evidence_identity_canonical,
    evidence_identity_hash, recommended_service, recommended_action,
    diagnostic_code, observation_contract_version,
    observation_identity_canonical, observation_identity_hash)
    ON campaign_operations_reconciliation_observation
    TO campaign_operations_reconciler;
GRANT INSERT (
    run_key, prior_target_id, last_target_id, requested_limit,
    selected_count)
    ON campaign_operations_reconciliation_cursor_event
    TO campaign_operations_reconciler;
GRANT INSERT (
    operational_campaign_id, operational_request_id,
    reconciliation_observation_id, cause_kind, actor_identity, capability,
    reason, prior_version, resulting_version, outcome, diagnostic_code)
    ON campaign_operations_control_audit_reference_event
    TO campaign_operations_reconciler;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_dispatch_attempt,
    campaign_operations_dispatch_attempt_outcome,
    campaign_operations_dispatch_audit_reference_event,
    campaign_operations_request_binding,
    campaign_operations_cancellation_request,
    campaign_operations_cancellation_settlement,
    campaign_operations_reconciliation_observation,
    campaign_operations_reconciliation_resolution
    TO campaign_operations_recovery;
GRANT SELECT ON
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_conversion_execution
    TO campaign_operations_recovery;
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
    TO campaign_operations_recovery;
GRANT INSERT (
    operational_campaign_id, operational_request_id, dispatch_attempt_id,
    dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
    prior_version, resulting_version, outcome, replay_disposition,
    diagnostic_code)
    ON campaign_operations_dispatch_audit_reference_event
    TO campaign_operations_recovery;
GRANT INSERT (
    operational_campaign_id, operational_request_id,
    reconciliation_observation_id, reconciliation_resolution_id,
    cause_kind, actor_identity, capability, reason, prior_version,
    resulting_version, outcome, diagnostic_code)
    ON campaign_operations_control_audit_reference_event
    TO campaign_operations_recovery;

GRANT SELECT ON
    campaign_operations_cancellation_request,
    campaign_operations_downstream_control_owner,
    experiment_lifecycle_cancellation_event,
    experiment
    TO experiment_lifecycle_cancellation;
GRANT SELECT ON
    campaign_operations_cancellation_request,
    campaign_operations_downstream_control_owner,
    experiment_lifecycle_cancellation_event,
    experiment
    TO experiment_lifecycle_cancellation_owner;
GRANT INSERT (
    cancellation_request_id,
    cancellation_request_identity_canonical,
    downstream_control_owner_id, experiment_id, expected_status,
    expected_phase, observed_status, observed_phase, resulting_status,
    disposition, actor_identity, event_contract_version,
    event_identity_canonical, event_identity_hash)
    ON experiment_lifecycle_cancellation_event
    TO experiment_lifecycle_cancellation_owner;
GRANT UPDATE (status, completed_at, updated_at)
    ON experiment TO experiment_lifecycle_cancellation_owner;

GRANT EXECUTE ON FUNCTION
    transition_campaign_operations_request_cancelled(bigint, integer),
    transition_campaign_operations_reservation_released(bigint, integer),
    append_campaign_operations_cancellation_resolution(
        bigint, text, text, text, text)
    TO campaign_operations_cancellation_coordinator;
GRANT EXECUTE ON FUNCTION
    transition_campaign_operations_request_ready_recovered(bigint, integer),
    append_campaign_operations_recovery_resolution(
        bigint, text, text, text, text, text)
    TO campaign_operations_recovery;
GRANT EXECUTE ON FUNCTION
    apply_experiment_lifecycle_cancellation(
        bigint, bigint, bigint, text, text, text, text, text)
    TO experiment_lifecycle_cancellation;
GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_campaign(bigint)
    TO campaign_operations_controller;
GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_request(bigint)
    TO campaign_operations_reconciler;
GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_budget_head(bigint),
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint)
    TO campaign_operations_cancellation_coordinator;
GRANT EXECUTE ON FUNCTION
    lock_campaign_operations_campaign(bigint),
    lock_campaign_operations_reservation(bigint),
    lock_campaign_operations_request(bigint)
    TO campaign_operations_recovery;

DO $$
DECLARE sequence_name text;
DECLARE role_name text;
BEGIN
    FOR sequence_name, role_name IN
        SELECT pg_get_serial_sequence(table_name, column_name), capability
        FROM (VALUES
            ('campaign_operations_control_event', 'control_event_id',
             'campaign_operations_controller'),
            ('campaign_operations_cancellation_request',
             'cancellation_request_id',
             'campaign_operations_cancellation_coordinator'),
            ('campaign_operations_cancellation_settlement',
             'cancellation_settlement_id',
             'campaign_operations_cancellation_coordinator'),
            ('campaign_operations_reservation_event',
             'reservation_event_id',
             'campaign_operations_cancellation_coordinator'),
            ('experiment_lifecycle_cancellation_event',
             'lifecycle_cancellation_event_id',
             'experiment_lifecycle_cancellation_owner'),
            ('campaign_operations_reconciliation_observation',
             'reconciliation_observation_id',
             'campaign_operations_reconciler'),
            ('campaign_operations_reconciliation_cursor_event',
             'reconciliation_cursor_event_id',
             'campaign_operations_reconciler'),
            ('campaign_operations_control_audit_reference_event',
             'control_audit_reference_event_id',
             'campaign_operations_controller'),
            ('campaign_operations_control_audit_reference_event',
             'control_audit_reference_event_id',
             'campaign_operations_cancellation_coordinator'),
            ('campaign_operations_control_audit_reference_event',
             'control_audit_reference_event_id',
             'campaign_operations_reconciler'),
            ('campaign_operations_control_audit_reference_event',
             'control_audit_reference_event_id',
             'campaign_operations_recovery'))
            AS sequences(table_name, column_name, capability)
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC, pqxx',
            sequence_name);
        EXECUTE format('GRANT USAGE ON SEQUENCE %s TO %I',
            sequence_name, role_name);
    END LOOP;
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_cancellation_coordinator, '
        'campaign_operations_recovery',
        pg_get_serial_sequence(
            'campaign_operations_dispatch_attempt_outcome',
            'dispatch_attempt_outcome_id'));
    EXECUTE format(
        'GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_cancellation_coordinator, '
        'campaign_operations_recovery',
        pg_get_serial_sequence(
            'campaign_operations_dispatch_audit_reference_event',
            'dispatch_audit_reference_event_id'));
END $$;

ALTER TABLE campaign_operations_control_event
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_cancellation_request
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_cancellation_settlement
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reconciliation_observation
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reconciliation_resolution
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reconciliation_cursor_event
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_control_audit_reference_event
    OWNER TO campaign_operations_owner;
ALTER TABLE experiment_lifecycle_cancellation_event
    OWNER TO campaign_operations_owner;

ALTER FUNCTION enforce_campaign_operations_control_chain()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_future_action_gate()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_cancellation_target()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION campaign_operations_future_actions_allowed(bigint)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION guard_campaign_operations_dispatch_control()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_control_audit_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION
    enforce_campaign_operations_phase4_request_transition_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION append_campaign_operations_cancellation_resolution(
    bigint, text, text, text, text)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION append_campaign_operations_recovery_resolution(
    bigint, text, text, text, text, text)
    OWNER TO campaign_operations_owner;
ALTER FUNCTION
    enforce_campaign_operations_reconciliation_cursor_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_request_cancelled(
    bigint, integer) OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_reservation_released(
    bigint, integer) OWNER TO campaign_operations_owner;
ALTER FUNCTION transition_campaign_operations_request_ready_recovered(
    bigint, integer) OWNER TO campaign_operations_owner;
ALTER FUNCTION apply_experiment_lifecycle_cancellation(
    bigint, bigint, bigint, text, text, text, text, text)
    OWNER TO experiment_lifecycle_cancellation_owner;

DO $$
DECLARE phase4_schema text := current_schema();
BEGIN
    EXECUTE format(
        'GRANT USAGE ON SCHEMA %I TO campaign_operations_controller, '
        'campaign_operations_cancellation_coordinator, '
        'campaign_operations_reconciler, campaign_operations_recovery, '
        'experiment_lifecycle_cancellation, '
        'experiment_lifecycle_cancellation_owner', phase4_schema);
END $$;
