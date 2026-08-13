-- Campaign Operations Phase 2: budget reservation and durable request
-- acceptance.
--
-- This additive migration implements only the accepted ADR-0012 budget and
-- reservation authority and the request-acceptance portion of ADR-0013.
-- It adds no dispatch lease, binding, Phase 5 invocation, experiment mutation,
-- scheduler access, worker launch, campaign control, or automatic progression.

DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_budget_administrator') THEN
        CREATE ROLE campaign_operations_budget_administrator NOLOGIN;
    END IF;
    IF NOT EXISTS (SELECT 1 FROM pg_roles
                   WHERE rolname = 'campaign_operations_request_acceptor') THEN
        CREATE ROLE campaign_operations_request_acceptor NOLOGIN;
    END IF;
END $$;

ALTER ROLE campaign_operations_budget_administrator
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;
ALTER ROLE campaign_operations_request_acceptor
    NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS;

DO $$
DECLARE capability_role text;
BEGIN
    FOREACH capability_role IN ARRAY ARRAY[
        'campaign_operations_budget_administrator',
        'campaign_operations_request_acceptor']
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

-- Migration 045's repository path takes this authorization-domain lock, but
-- the capability role also has the column-scoped INSERT required by that
-- repository. Enforce the same lock in PostgreSQL so a direct capability-role
-- successor cannot race Phase 2 request acceptance around the repository.
CREATE OR REPLACE FUNCTION enforce_campaign_operations_authorization_chain()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(
        NEW.campaign_identity_canonical ||
        ';action=' || NEW.action_kind ||
        ';scope=' || NEW.scope_kind, 1179402835030003));

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

CREATE TABLE IF NOT EXISTS campaign_operations_budget_ledger_entry (
    budget_ledger_entry_id bigserial PRIMARY KEY CHECK (
        budget_ledger_entry_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    previous_entry_id bigint,
    previous_entry_identity_canonical text COLLATE "C",
    previous_entry_identity_hash text COLLATE "C",
    ledger_version integer NOT NULL CHECK (ledger_version > 0),
    entry_kind text COLLATE "C" NOT NULL CHECK (
        entry_kind IN ('grant', 'amend', 'revoke', 'supersede')),
    ledger_status text COLLATE "C" NOT NULL CHECK (
        ledger_status IN ('active', 'revoked')),
    budget_unit text COLLATE "C" NOT NULL CHECK (
        budget_unit = 'materialized_member_dispatch'),
    delta bigint NOT NULL,
    prior_total bigint NOT NULL CHECK (prior_total >= 0),
    resulting_total bigint NOT NULL CHECK (resulting_total >= 0),
    administrator_identity text COLLATE "C" NOT NULL CHECK (
        administrator_identity ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    budget_contract_version integer NOT NULL CHECK (
        budget_contract_version = 1),
    budget_identity_canonical text COLLATE "C" NOT NULL CHECK (
        budget_identity_canonical <> '' AND
        octet_length(budget_identity_canonical) <= 134217728),
    budget_identity_hash text COLLATE "C" NOT NULL CHECK (
        budget_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_budget_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_budget_previous_fk
        FOREIGN KEY (previous_entry_id)
        REFERENCES campaign_operations_budget_ledger_entry(
            budget_ledger_entry_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_budget_previous_shape_check CHECK (
        (ledger_version = 1 AND previous_entry_id IS NULL AND
            previous_entry_identity_canonical IS NULL AND
            previous_entry_identity_hash IS NULL) OR
        (ledger_version > 1 AND previous_entry_id IS NOT NULL AND
            previous_entry_identity_canonical IS NOT NULL AND
            previous_entry_identity_canonical <> '' AND
            previous_entry_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$')),
    CONSTRAINT campaign_operations_budget_arithmetic_check CHECK (
        prior_total + delta = resulting_total),
    CONSTRAINT campaign_operations_budget_kind_shape_check CHECK (
        (entry_kind = 'grant' AND ledger_version = 1 AND prior_total = 0 AND
            delta > 0 AND ledger_status = 'active') OR
        (entry_kind = 'amend' AND ledger_version > 1 AND delta <> 0 AND
            ledger_status = 'active') OR
        (entry_kind = 'revoke' AND ledger_version > 1 AND
            ledger_status = 'revoked') OR
        (entry_kind = 'supersede' AND ledger_version > 1 AND
            ledger_status = 'active')),
    CONSTRAINT campaign_operations_budget_chain_uidx UNIQUE (
        operational_campaign_id, ledger_version),
    CONSTRAINT campaign_operations_budget_previous_uidx UNIQUE (
        previous_entry_id)
);

CREATE INDEX IF NOT EXISTS campaign_operations_budget_head_idx
    ON campaign_operations_budget_ledger_entry(
        operational_campaign_id, ledger_version DESC);
CREATE INDEX IF NOT EXISTS campaign_operations_budget_identity_hash_idx
    ON campaign_operations_budget_ledger_entry(budget_identity_hash);

CREATE TABLE IF NOT EXISTS campaign_operations_reservation (
    reservation_id bigserial PRIMARY KEY CHECK (reservation_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    logical_operation_contract_version integer NOT NULL CHECK (
        logical_operation_contract_version = 1),
    logical_operation_canonical text COLLATE "C" NOT NULL CHECK (
        logical_operation_canonical <> '' AND
        octet_length(logical_operation_canonical) <= 134217728),
    logical_operation_hash text COLLATE "C" NOT NULL CHECK (
        logical_operation_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    authorization_event_id bigint NOT NULL,
    authorization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        authorization_identity_canonical <> '' AND
        octet_length(authorization_identity_canonical) <= 134217728),
    authorization_identity_hash text COLLATE "C" NOT NULL CHECK (
        authorization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    budget_ledger_entry_id bigint NOT NULL,
    budget_ledger_version integer NOT NULL CHECK (budget_ledger_version > 0),
    budget_identity_canonical text COLLATE "C" NOT NULL CHECK (
        budget_identity_canonical <> '' AND
        octet_length(budget_identity_canonical) <= 134217728),
    budget_identity_hash text COLLATE "C" NOT NULL CHECK (
        budget_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    action_kind text COLLATE "C" NOT NULL CHECK (
        action_kind = 'dispatch_full_materialization'),
    action_contract_version integer NOT NULL CHECK (
        action_contract_version = 1),
    recommendation_campaign_materialization_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL CHECK (
        materialization_contract_version = 1),
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        materialization_identity_canonical <> '' AND
        octet_length(materialization_identity_canonical) <= 134217728),
    materialization_identity_hash text COLLATE "C" NOT NULL CHECK (
        materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    scope_kind text COLLATE "C" NOT NULL CHECK (
        scope_kind = 'complete_materialization'),
    scope_contract_version integer NOT NULL CHECK (
        scope_contract_version = 1),
    materialization_member_count integer NOT NULL CHECK (
        materialization_member_count > 0),
    amount bigint NOT NULL CHECK (amount > 0),
    budget_unit text COLLATE "C" NOT NULL CHECK (
        budget_unit = 'materialized_member_dispatch'),
    expires_at timestamptz,
    reservation_contract_version integer NOT NULL CHECK (
        reservation_contract_version = 1),
    reservation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_identity_canonical <> '' AND
        octet_length(reservation_identity_canonical) <= 134217728),
    reservation_identity_hash text COLLATE "C" NOT NULL CHECK (
        reservation_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    reservation_state text COLLATE "C" NOT NULL CHECK (
        reservation_state IN (
            'held', 'committed', 'released', 'expired',
            'reconciliation_required')),
    state_version integer NOT NULL CHECK (state_version > 0),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_reservation_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_authorization_fk
        FOREIGN KEY (authorization_event_id)
        REFERENCES campaign_operations_authorization_event(
            authorization_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_budget_fk
        FOREIGN KEY (budget_ledger_entry_id)
        REFERENCES campaign_operations_budget_ledger_entry(
            budget_ledger_entry_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_materialization_fk
        FOREIGN KEY (recommendation_campaign_materialization_id)
        REFERENCES experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_member_count_check CHECK (
        amount = materialization_member_count),
    CONSTRAINT campaign_operations_reservation_operation_uidx UNIQUE (
        operational_campaign_id, action_kind, action_contract_version)
);

ALTER TABLE campaign_operations_reservation
    DROP CONSTRAINT IF EXISTS
        campaign_operations_reservation_initial_state_check;

CREATE INDEX IF NOT EXISTS
campaign_operations_reservation_logical_operation_hash_idx
    ON campaign_operations_reservation(logical_operation_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_reservation_identity_hash_idx
    ON campaign_operations_reservation(reservation_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_reservation_recovery_idx
    ON campaign_operations_reservation(
        reservation_state, expires_at, reservation_id);

CREATE TABLE IF NOT EXISTS campaign_operations_operational_request (
    operational_request_id bigserial PRIMARY KEY CHECK (
        operational_request_id > 0),
    operational_campaign_id bigint NOT NULL,
    campaign_identity_canonical text COLLATE "C" NOT NULL CHECK (
        campaign_identity_canonical <> '' AND
        octet_length(campaign_identity_canonical) <= 134217728),
    logical_operation_contract_version integer NOT NULL CHECK (
        logical_operation_contract_version = 1),
    logical_operation_canonical text COLLATE "C" NOT NULL CHECK (
        logical_operation_canonical <> '' AND
        octet_length(logical_operation_canonical) <= 134217728),
    logical_operation_hash text COLLATE "C" NOT NULL CHECK (
        logical_operation_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    authorization_event_id bigint NOT NULL,
    authorization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        authorization_identity_canonical <> '' AND
        octet_length(authorization_identity_canonical) <= 134217728),
    authorization_identity_hash text COLLATE "C" NOT NULL CHECK (
        authorization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    reservation_id bigint NOT NULL,
    reservation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_identity_canonical <> '' AND
        octet_length(reservation_identity_canonical) <= 134217728),
    reservation_identity_hash text COLLATE "C" NOT NULL CHECK (
        reservation_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    action_kind text COLLATE "C" NOT NULL CHECK (
        action_kind = 'dispatch_full_materialization'),
    action_contract_version integer NOT NULL CHECK (
        action_contract_version = 1),
    recommendation_campaign_materialization_id bigint NOT NULL,
    materialization_contract_version integer NOT NULL CHECK (
        materialization_contract_version = 1),
    materialization_identity_canonical text COLLATE "C" NOT NULL CHECK (
        materialization_identity_canonical <> '' AND
        octet_length(materialization_identity_canonical) <= 134217728),
    materialization_identity_hash text COLLATE "C" NOT NULL CHECK (
        materialization_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    ordered_scope_digest text COLLATE "C" NOT NULL CHECK (
        ordered_scope_digest ~ '^fnv1a64:[0-9a-f]{16}$'),
    materialization_member_count integer NOT NULL CHECK (
        materialization_member_count > 0),
    accepting_actor_identity text COLLATE "C" NOT NULL CHECK (
        accepting_actor_identity ~
            '^[A-Za-z0-9][A-Za-z0-9._@:/+\-]{0,127}$'),
    reason text COLLATE "C" NOT NULL CHECK (
        reason <> '' AND octet_length(reason) <= 4096 AND
        reason ~ E'[^ \t\r\n]' AND
        translate(reason, E'\t\r\n', '') !~ '[[:cntrl:]]'),
    prerequisite_policy text COLLATE "C" NOT NULL CHECK (
        prerequisite_policy IN (
            'phase4d_materialization_only_v1',
            'phase4d_materialization_plus_exact_phase6d_ratification_v1')),
    provenance_identity_canonical text COLLATE "C",
    provenance_identity_hash text COLLATE "C",
    request_contract_version integer NOT NULL CHECK (
        request_contract_version = 1),
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> '' AND
        octet_length(request_identity_canonical) <= 134217728),
    request_identity_hash text COLLATE "C" NOT NULL CHECK (
        request_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    request_state text COLLATE "C" NOT NULL CHECK (
        request_state IN (
            'ready', 'dispatching', 'bound', 'permanently_failed',
            'cancelled', 'reconciliation_required')),
    state_version integer NOT NULL CHECK (state_version > 0),
    lease_token_hash text COLLATE "C",
    lease_expires_at timestamptz,
    dispatcher_identity text COLLATE "C",
    production_dispatch_enabled boolean NOT NULL DEFAULT false CHECK (
        production_dispatch_enabled = false),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_request_campaign_fk
        FOREIGN KEY (operational_campaign_id)
        REFERENCES campaign_operations_campaign(operational_campaign_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_authorization_fk
        FOREIGN KEY (authorization_event_id)
        REFERENCES campaign_operations_authorization_event(
            authorization_event_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_reservation_fk
        FOREIGN KEY (reservation_id)
        REFERENCES campaign_operations_reservation(reservation_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_materialization_fk
        FOREIGN KEY (recommendation_campaign_materialization_id)
        REFERENCES experiment_recommendation_campaign_materialization(
            recommendation_campaign_materialization_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_request_provenance_shape_check CHECK (
        (provenance_identity_canonical IS NULL AND
            provenance_identity_hash IS NULL) OR
        (provenance_identity_canonical IS NOT NULL AND
            provenance_identity_canonical <> '' AND
            provenance_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$')),
    CONSTRAINT campaign_operations_request_operation_uidx UNIQUE (
        operational_campaign_id, action_kind, action_contract_version),
    CONSTRAINT campaign_operations_request_reservation_uidx UNIQUE (
        reservation_id)
);

ALTER TABLE campaign_operations_operational_request
    DROP CONSTRAINT IF EXISTS
        campaign_operations_request_initial_state_check;

CREATE INDEX IF NOT EXISTS
campaign_operations_request_logical_operation_hash_idx
    ON campaign_operations_operational_request(logical_operation_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_request_identity_hash_idx
    ON campaign_operations_operational_request(request_identity_hash);
CREATE INDEX IF NOT EXISTS campaign_operations_request_ready_idx
    ON campaign_operations_operational_request(
        request_state, production_dispatch_enabled,
        operational_request_id);

CREATE TABLE IF NOT EXISTS campaign_operations_reservation_event (
    reservation_event_id bigserial PRIMARY KEY CHECK (
        reservation_event_id > 0),
    reservation_id bigint NOT NULL,
    reservation_identity_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_identity_canonical <> '' AND
        octet_length(reservation_identity_canonical) <= 134217728),
    transition_kind text COLLATE "C" NOT NULL CHECK (
        transition_kind IN (
            'acquired', 'committed', 'released', 'expired',
            'reconciliation_required')),
    expected_state text COLLATE "C" CHECK (
        expected_state IN (
            'held', 'committed', 'released', 'expired',
            'reconciliation_required')),
    resulting_state text COLLATE "C" NOT NULL CHECK (
        resulting_state IN (
            'held', 'committed', 'released', 'expired',
            'reconciliation_required')),
    expected_version integer NOT NULL CHECK (expected_version >= 0),
    resulting_version integer NOT NULL CHECK (resulting_version > 0),
    operational_request_id bigint NOT NULL,
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> '' AND
        octet_length(request_identity_canonical) <= 134217728),
    amount bigint NOT NULL CHECK (amount > 0),
    reservation_event_contract_version integer NOT NULL CHECK (
        reservation_event_contract_version = 1),
    reservation_event_identity_canonical text COLLATE "C" NOT NULL CHECK (
        reservation_event_identity_canonical <> '' AND
        octet_length(reservation_event_identity_canonical) <= 134217728),
    reservation_event_identity_hash text COLLATE "C" NOT NULL CHECK (
        reservation_event_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_reservation_event_reservation_fk
        FOREIGN KEY (reservation_id)
        REFERENCES campaign_operations_reservation(reservation_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_event_request_fk
        FOREIGN KEY (operational_request_id)
        REFERENCES campaign_operations_operational_request(
            operational_request_id)
        ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_reservation_event_version_check CHECK (
        resulting_version = expected_version + 1),
    CONSTRAINT campaign_operations_reservation_event_acquisition_check CHECK (
        transition_kind <> 'acquired' OR (
            expected_state IS NULL AND resulting_state = 'held' AND
            expected_version = 0 AND resulting_version = 1)),
    CONSTRAINT campaign_operations_reservation_event_version_uidx UNIQUE (
        reservation_id, resulting_version)
);

CREATE INDEX IF NOT EXISTS
campaign_operations_reservation_event_identity_hash_idx
    ON campaign_operations_reservation_event(
        reservation_event_identity_hash);

CREATE OR REPLACE FUNCTION enforce_campaign_operations_budget_entry()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    predecessor campaign_operations_budget_ledger_entry%ROWTYPE;
    committed_units bigint;
    held_units bigint;
BEGIN
    PERFORM pg_advisory_xact_lock(hashtextextended(
        NEW.campaign_identity_canonical, 1179402835030004));

    PERFORM 1
    FROM campaign_operations_campaign campaign
    WHERE campaign.operational_campaign_id = NEW.operational_campaign_id
      AND campaign.campaign_identity_canonical =
              NEW.campaign_identity_canonical
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations budget campaign mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'campaign_operations_budget_campaign_check';
    END IF;

    SELECT
        coalesce(sum(amount) FILTER (
            WHERE reservation_state = 'committed'), 0),
        coalesce(sum(amount) FILTER (
            WHERE reservation_state IN (
                'held', 'reconciliation_required')), 0)
    INTO committed_units, held_units
    FROM campaign_operations_reservation
    WHERE operational_campaign_id = NEW.operational_campaign_id;

    IF NEW.ledger_version = 1 THEN
        IF NEW.entry_kind <> 'grant' THEN
            RAISE EXCEPTION 'campaign operations initial budget must grant'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                          'campaign_operations_budget_initial_check';
        END IF;
    ELSE
        SELECT * INTO STRICT predecessor
        FROM campaign_operations_budget_ledger_entry
        WHERE budget_ledger_entry_id = NEW.previous_entry_id
        FOR UPDATE;
        IF predecessor.operational_campaign_id <>
                NEW.operational_campaign_id OR
           predecessor.ledger_version <> NEW.ledger_version - 1 OR
           predecessor.budget_identity_canonical <>
                NEW.previous_entry_identity_canonical OR
           predecessor.budget_identity_hash <>
                NEW.previous_entry_identity_hash OR
           predecessor.resulting_total <> NEW.prior_total OR
           (NEW.entry_kind IN ('amend', 'revoke') AND
                predecessor.ledger_status <> 'active') OR
           (NEW.entry_kind = 'supersede' AND
                predecessor.ledger_status <> 'revoked') THEN
            RAISE EXCEPTION
                'campaign operations budget predecessor mismatch'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                          'campaign_operations_budget_previous_check';
        END IF;
    END IF;

    IF NEW.resulting_total < committed_units + held_units OR
       (NEW.entry_kind = 'revoke' AND
            NEW.resulting_total <> committed_units + held_units) THEN
        RAISE EXCEPTION 'campaign operations budget below obligations'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_budget_obligation_check';
    END IF;
    RETURN NEW;
EXCEPTION
    WHEN no_data_found THEN
        RAISE EXCEPTION 'campaign operations budget predecessor missing'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_budget_previous_check';
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_reservation()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
DECLARE
    current_budget_id bigint;
    current_budget_total bigint;
    committed_units bigint;
    held_units bigint;
BEGIN
    IF NEW.reservation_state <> 'held' OR NEW.state_version <> 1 THEN
        RAISE EXCEPTION
            'campaign operations reservation must begin held at version one'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_initial_state_check';
    END IF;
    IF NEW.expires_at IS NOT NULL AND
       NEW.expires_at <= transaction_timestamp() THEN
        RAISE EXCEPTION
            'campaign operations reservation expiry must be in the future'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_expiry_check';
    END IF;

    PERFORM pg_advisory_xact_lock(hashtextextended(
        NEW.campaign_identity_canonical ||
        ';action=' || NEW.action_kind ||
        ';scope=' || NEW.scope_kind, 1179402835030003));
    PERFORM pg_advisory_xact_lock(hashtextextended(
        NEW.campaign_identity_canonical, 1179402835030004));

    PERFORM 1
    FROM campaign_operations_campaign campaign
    WHERE campaign.operational_campaign_id = NEW.operational_campaign_id
      AND campaign.campaign_identity_canonical =
              NEW.campaign_identity_canonical
      AND campaign.action_kind = NEW.action_kind
      AND campaign.action_contract_version = NEW.action_contract_version
      AND campaign.scope_kind = NEW.scope_kind
      AND campaign.scope_contract_version = NEW.scope_contract_version
      AND campaign.recommendation_campaign_materialization_id =
              NEW.recommendation_campaign_materialization_id
      AND campaign.materialization_contract_version =
              NEW.materialization_contract_version
      AND campaign.materialization_identity_canonical =
              NEW.materialization_identity_canonical
      AND campaign.materialization_identity_hash =
              NEW.materialization_identity_hash
      AND campaign.materialization_member_count =
              NEW.materialization_member_count
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations reservation campaign mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_campaign_check';
    END IF;

    PERFORM 1
    FROM campaign_operations_authorization_event authz
    WHERE authz.authorization_event_id = NEW.authorization_event_id
      AND authz.operational_campaign_id =
              NEW.operational_campaign_id
      AND authz.authorization_identity_canonical =
              NEW.authorization_identity_canonical
      AND authz.authorization_identity_hash =
              NEW.authorization_identity_hash
      AND authz.event_kind = 'granted'
      AND authz.action_kind = NEW.action_kind
      AND authz.action_contract_version =
              NEW.action_contract_version
      AND authz.scope_kind = NEW.scope_kind
      AND authz.scope_contract_version =
              NEW.scope_contract_version
      AND transaction_timestamp() >= authz.not_before
      AND (authz.expires_at IS NULL OR
           transaction_timestamp() < authz.expires_at)
      AND NOT EXISTS (
          SELECT 1
          FROM campaign_operations_authorization_event successor
          WHERE successor.operational_campaign_id =
                    authz.operational_campaign_id
            AND successor.action_kind = authz.action_kind
            AND successor.action_contract_version =
                    authz.action_contract_version
            AND successor.scope_kind = authz.scope_kind
            AND successor.scope_contract_version =
                    authz.scope_contract_version
            AND successor.chain_version > authz.chain_version);
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations reservation authorization denied'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_authorization_check';
    END IF;

    SELECT budget_ledger_entry_id, resulting_total
    INTO current_budget_id, current_budget_total
    FROM campaign_operations_budget_ledger_entry
    WHERE operational_campaign_id = NEW.operational_campaign_id
    ORDER BY ledger_version DESC
    LIMIT 1
    FOR UPDATE;
    IF current_budget_id IS NULL OR
       current_budget_id <> NEW.budget_ledger_entry_id OR
       NOT EXISTS (
           SELECT 1
           FROM campaign_operations_budget_ledger_entry budget
           WHERE budget.budget_ledger_entry_id = current_budget_id
             AND budget.ledger_version = NEW.budget_ledger_version
             AND budget.budget_identity_canonical =
                    NEW.budget_identity_canonical
             AND budget.budget_identity_hash =
                    NEW.budget_identity_hash
             AND budget.ledger_status = 'active') THEN
        RAISE EXCEPTION 'campaign operations reservation budget inactive'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_budget_check';
    END IF;

    SELECT
        coalesce(sum(amount) FILTER (
            WHERE reservation_state = 'committed'), 0),
        coalesce(sum(amount) FILTER (
            WHERE reservation_state IN (
                'held', 'reconciliation_required')), 0)
    INTO committed_units, held_units
    FROM campaign_operations_reservation
    WHERE operational_campaign_id = NEW.operational_campaign_id;
    IF NEW.amount > current_budget_total - committed_units - held_units THEN
        RAISE EXCEPTION 'insufficient materialized member dispatch units'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_available_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_request()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    IF NEW.request_state <> 'ready' OR NEW.state_version <> 1 OR
       NEW.lease_token_hash IS NOT NULL OR
       NEW.lease_expires_at IS NOT NULL OR
       NEW.dispatcher_identity IS NOT NULL OR
       NEW.production_dispatch_enabled THEN
        RAISE EXCEPTION
            'campaign operations request must begin undispatched and ready'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_request_initial_state_check';
    END IF;

    PERFORM 1
    FROM campaign_operations_reservation reservation
    JOIN campaign_operations_authorization_event authz
      ON authz.authorization_event_id =
             reservation.authorization_event_id
    LEFT JOIN campaign_operations_governance_provenance_event provenance
      ON provenance.governance_provenance_event_id =
             authz.governance_provenance_event_id
    WHERE reservation.reservation_id = NEW.reservation_id
      AND reservation.operational_campaign_id =
              NEW.operational_campaign_id
      AND reservation.campaign_identity_canonical =
              NEW.campaign_identity_canonical
      AND reservation.logical_operation_canonical =
              NEW.logical_operation_canonical
      AND reservation.logical_operation_hash =
              NEW.logical_operation_hash
      AND reservation.authorization_event_id =
              NEW.authorization_event_id
      AND reservation.authorization_identity_canonical =
              NEW.authorization_identity_canonical
      AND reservation.authorization_identity_hash =
              NEW.authorization_identity_hash
      AND authz.operational_campaign_id =
              NEW.operational_campaign_id
      AND authz.authorization_identity_canonical =
              NEW.authorization_identity_canonical
      AND authz.authorization_identity_hash =
              NEW.authorization_identity_hash
      AND authz.action_kind = NEW.action_kind
      AND authz.action_contract_version =
              NEW.action_contract_version
      AND authz.scope_kind = reservation.scope_kind
      AND authz.scope_contract_version =
              reservation.scope_contract_version
      AND authz.prerequisite_policy =
              NEW.prerequisite_policy
      AND authz.provenance_identity_canonical IS NOT DISTINCT FROM
              NEW.provenance_identity_canonical
      AND authz.provenance_identity_hash IS NOT DISTINCT FROM
              NEW.provenance_identity_hash
      AND (
          (authz.governance_provenance_event_id IS NULL AND
              provenance.governance_provenance_event_id IS NULL AND
              NEW.provenance_identity_canonical IS NULL AND
              NEW.provenance_identity_hash IS NULL) OR
          (authz.governance_provenance_event_id IS NOT NULL AND
              provenance.governance_provenance_event_id =
                  authz.governance_provenance_event_id AND
              provenance.operational_campaign_id =
                  NEW.operational_campaign_id AND
              provenance.prerequisite_policy =
                  NEW.prerequisite_policy AND
              provenance.provenance_identity_canonical =
                  NEW.provenance_identity_canonical AND
              provenance.provenance_identity_hash =
                  NEW.provenance_identity_hash))
      AND reservation.reservation_identity_canonical =
              NEW.reservation_identity_canonical
      AND reservation.reservation_identity_hash =
              NEW.reservation_identity_hash
      AND reservation.action_kind = NEW.action_kind
      AND reservation.action_contract_version =
              NEW.action_contract_version
      AND reservation.recommendation_campaign_materialization_id =
              NEW.recommendation_campaign_materialization_id
      AND reservation.materialization_contract_version =
              NEW.materialization_contract_version
      AND reservation.materialization_identity_canonical =
              NEW.materialization_identity_canonical
      AND reservation.materialization_identity_hash =
              NEW.materialization_identity_hash
      AND reservation.materialization_member_count =
              NEW.materialization_member_count
      AND reservation.amount = NEW.materialization_member_count
      AND reservation.reservation_state = 'held'
      AND reservation.state_version = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations request reservation mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_request_reservation_check';
    END IF;
    IF NEW.ordered_scope_digest <> NEW.materialization_identity_hash THEN
        RAISE EXCEPTION 'campaign operations request scope mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_request_scope_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_campaign_operations_reservation_event()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_reservation reservation
    JOIN campaign_operations_operational_request request
      ON request.operational_request_id = NEW.operational_request_id
    WHERE reservation.reservation_id = NEW.reservation_id
      AND reservation.reservation_identity_canonical =
              NEW.reservation_identity_canonical
      AND request.reservation_id = reservation.reservation_id
      AND request.request_identity_canonical =
              NEW.request_identity_canonical
      AND reservation.amount = NEW.amount
      AND reservation.reservation_state = NEW.resulting_state
      AND reservation.state_version = NEW.resulting_version;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations reservation event mismatch'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_event_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_request_acquisition_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_reservation_event event
    JOIN campaign_operations_reservation reservation
      ON reservation.reservation_id = event.reservation_id
    JOIN campaign_operations_authorization_event authz
      ON authz.authorization_event_id =
             NEW.authorization_event_id
    JOIN campaign_operations_budget_ledger_entry budget
      ON budget.budget_ledger_entry_id =
             reservation.budget_ledger_entry_id
    JOIN campaign_operations_audit_reference_event audit
      ON audit.reservation_event_id = event.reservation_event_id
    WHERE event.reservation_id = NEW.reservation_id
      AND event.operational_request_id = NEW.operational_request_id
      AND event.transition_kind = 'acquired'
      AND event.expected_state IS NULL
      AND event.resulting_state = 'held'
      AND event.expected_version = 0
      AND event.resulting_version = 1
      AND event.request_identity_canonical =
              NEW.request_identity_canonical
      AND reservation.operational_campaign_id =
              NEW.operational_campaign_id
      AND reservation.authorization_event_id =
              NEW.authorization_event_id
      AND reservation.authorization_identity_canonical =
              NEW.authorization_identity_canonical
      AND reservation.authorization_identity_hash =
              NEW.authorization_identity_hash
      AND reservation.reservation_identity_canonical =
              NEW.reservation_identity_canonical
      AND reservation.reservation_identity_hash =
              NEW.reservation_identity_hash
      AND authz.operational_campaign_id =
              NEW.operational_campaign_id
      AND authz.authorization_identity_canonical =
              NEW.authorization_identity_canonical
      AND authz.authorization_identity_hash =
              NEW.authorization_identity_hash
      AND budget.operational_campaign_id =
              NEW.operational_campaign_id
      AND budget.budget_ledger_entry_id =
              reservation.budget_ledger_entry_id
      AND budget.ledger_version = reservation.budget_ledger_version
      AND budget.budget_identity_canonical =
              reservation.budget_identity_canonical
      AND budget.budget_identity_hash =
              reservation.budget_identity_hash
      AND audit.cause_kind = 'reservation_request_accepted'
      AND audit.operational_campaign_id = NEW.operational_campaign_id
      AND audit.authorization_event_id = NEW.authorization_event_id
      AND audit.budget_ledger_entry_id =
              reservation.budget_ledger_entry_id
      AND audit.reservation_id = NEW.reservation_id
      AND audit.reservation_event_id = event.reservation_event_id
      AND audit.operational_request_id = NEW.operational_request_id
      AND audit.actor_identity = NEW.accepting_actor_identity
      AND audit.capability = 'campaign_operations_request_acceptor'
      AND audit.reason = NEW.reason
      AND audit.prior_version IS NULL
      AND audit.resulting_version = NEW.state_version
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'recorded';
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations request acquisition evidence incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_request_acquisition_check';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_reservation_acquisition_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_operational_request request
    JOIN campaign_operations_reservation_event event
      ON event.operational_request_id = request.operational_request_id
     AND event.reservation_id = request.reservation_id
    WHERE request.reservation_id = NEW.reservation_id
      AND request.logical_operation_canonical =
              NEW.logical_operation_canonical
      AND event.transition_kind = 'acquired'
      AND event.resulting_state = 'held'
      AND event.resulting_version = 1;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations reservation acquisition evidence incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_reservation_acquisition_check';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION
enforce_campaign_operations_budget_audit_complete()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_audit_reference_event audit
    WHERE audit.budget_ledger_entry_id = NEW.budget_ledger_entry_id
      AND audit.operational_campaign_id = NEW.operational_campaign_id
      AND audit.cause_kind = 'budget_ledger_recorded'
      AND audit.capability =
              'campaign_operations_budget_administrator'
      AND audit.actor_identity = NEW.administrator_identity
      AND audit.reason = NEW.reason
      AND audit.prior_version IS NOT DISTINCT FROM
              CASE WHEN NEW.ledger_version = 1
                   THEN NULL
                   ELSE NEW.ledger_version - 1
              END
      AND audit.resulting_version = NEW.ledger_version
      AND audit.outcome = 'recorded'
      AND audit.replay_disposition = 'recorded';
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations budget audit evidence incomplete'
            USING ERRCODE = '23514',
                  CONSTRAINT =
                      'campaign_operations_budget_audit_check';
    END IF;
    RETURN NULL;
END;
$$;

CREATE OR REPLACE FUNCTION lock_campaign_operations_campaign(
    target_operational_campaign_id bigint)
RETURNS void
LANGUAGE plpgsql
SECURITY DEFINER
AS $$
BEGIN
    PERFORM 1
    FROM campaign_operations_campaign
    WHERE operational_campaign_id = target_operational_campaign_id
    FOR UPDATE;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'campaign operations campaign not found'
            USING ERRCODE = 'P0002';
    END IF;
END;
$$;

DO $$
DECLARE phase_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_budget_entry() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_reservation() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_request() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_campaign_operations_reservation_event() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.'
        'enforce_campaign_operations_request_acquisition_complete() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.'
        'enforce_campaign_operations_reservation_acquisition_complete() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.'
        'enforce_campaign_operations_budget_audit_complete() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.lock_campaign_operations_campaign(bigint) '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.'
        'enforce_campaign_operations_authorization_chain() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        phase_schema, phase_schema);
END $$;

DROP TRIGGER IF EXISTS campaign_operations_budget_entry_trigger
ON campaign_operations_budget_ledger_entry;
CREATE TRIGGER campaign_operations_budget_entry_trigger
BEFORE INSERT ON campaign_operations_budget_ledger_entry
FOR EACH ROW EXECUTE FUNCTION enforce_campaign_operations_budget_entry();

DROP TRIGGER IF EXISTS campaign_operations_reservation_trigger
ON campaign_operations_reservation;
CREATE TRIGGER campaign_operations_reservation_trigger
BEFORE INSERT ON campaign_operations_reservation
FOR EACH ROW EXECUTE FUNCTION enforce_campaign_operations_reservation();

DROP TRIGGER IF EXISTS campaign_operations_request_trigger
ON campaign_operations_operational_request;
CREATE TRIGGER campaign_operations_request_trigger
BEFORE INSERT ON campaign_operations_operational_request
FOR EACH ROW EXECUTE FUNCTION enforce_campaign_operations_request();

DROP TRIGGER IF EXISTS campaign_operations_reservation_event_trigger
ON campaign_operations_reservation_event;
CREATE TRIGGER campaign_operations_reservation_event_trigger
BEFORE INSERT ON campaign_operations_reservation_event
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_reservation_event();

DROP TRIGGER IF EXISTS
campaign_operations_request_acquisition_complete_trigger
ON campaign_operations_operational_request;
CREATE CONSTRAINT TRIGGER
campaign_operations_request_acquisition_complete_trigger
AFTER INSERT ON campaign_operations_operational_request
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_request_acquisition_complete();

DROP TRIGGER IF EXISTS
campaign_operations_reservation_acquisition_complete_trigger
ON campaign_operations_reservation;
CREATE CONSTRAINT TRIGGER
campaign_operations_reservation_acquisition_complete_trigger
AFTER INSERT ON campaign_operations_reservation
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_reservation_acquisition_complete();

DROP TRIGGER IF EXISTS
campaign_operations_budget_audit_complete_trigger
ON campaign_operations_budget_ledger_entry;
CREATE CONSTRAINT TRIGGER
campaign_operations_budget_audit_complete_trigger
AFTER INSERT ON campaign_operations_budget_ledger_entry
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    enforce_campaign_operations_budget_audit_complete();

ALTER TABLE campaign_operations_audit_reference_event
    ADD COLUMN IF NOT EXISTS budget_ledger_entry_id bigint,
    ADD COLUMN IF NOT EXISTS reservation_id bigint,
    ADD COLUMN IF NOT EXISTS reservation_event_id bigint,
    ADD COLUMN IF NOT EXISTS operational_request_id bigint;

ALTER TABLE campaign_operations_audit_reference_event
    DROP CONSTRAINT IF EXISTS campaign_operations_audit_cause_shape_check,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_audit_budget_ledger_entry_fk,
    DROP CONSTRAINT IF EXISTS campaign_operations_audit_reservation_fk,
    DROP CONSTRAINT IF EXISTS campaign_operations_audit_reservation_event_fk,
    DROP CONSTRAINT IF EXISTS campaign_operations_audit_request_fk,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_audit_reference_event_cause_kind_check,
    DROP CONSTRAINT IF EXISTS
        campaign_operations_audit_reference_event_capability_check;

ALTER TABLE campaign_operations_audit_reference_event
    ADD CONSTRAINT campaign_operations_audit_budget_ledger_entry_fk
        FOREIGN KEY (budget_ledger_entry_id)
        REFERENCES campaign_operations_budget_ledger_entry(
            budget_ledger_entry_id) ON DELETE RESTRICT,
    ADD CONSTRAINT campaign_operations_audit_reservation_fk
        FOREIGN KEY (reservation_id)
        REFERENCES campaign_operations_reservation(reservation_id)
        ON DELETE RESTRICT,
    ADD CONSTRAINT campaign_operations_audit_reservation_event_fk
        FOREIGN KEY (reservation_event_id)
        REFERENCES campaign_operations_reservation_event(
            reservation_event_id) ON DELETE RESTRICT,
    ADD CONSTRAINT campaign_operations_audit_request_fk
        FOREIGN KEY (operational_request_id)
        REFERENCES campaign_operations_operational_request(
            operational_request_id) ON DELETE RESTRICT,
    ADD CONSTRAINT
        campaign_operations_audit_reference_event_cause_kind_check CHECK (
            cause_kind IN (
                'campaign_created',
                'governance_provenance_recorded',
                'authorization_recorded',
                'budget_ledger_recorded',
                'reservation_request_accepted')),
    ADD CONSTRAINT
        campaign_operations_audit_reference_event_capability_check CHECK (
            capability IN (
                'campaign_operations_campaign_creator',
                'campaign_operations_authorizer',
                'campaign_operations_budget_administrator',
                'campaign_operations_request_acceptor')),
    ADD CONSTRAINT campaign_operations_audit_cause_shape_check CHECK (
        (cause_kind = 'campaign_created' AND
            governance_provenance_event_id IS NULL AND
            authorization_event_id IS NULL AND
            budget_ledger_entry_id IS NULL AND reservation_id IS NULL AND
            reservation_event_id IS NULL AND operational_request_id IS NULL AND
            prior_version IS NULL AND resulting_version = 1) OR
        (cause_kind = 'governance_provenance_recorded' AND
            governance_provenance_event_id IS NOT NULL AND
            authorization_event_id IS NULL AND
            budget_ledger_entry_id IS NULL AND reservation_id IS NULL AND
            reservation_event_id IS NULL AND operational_request_id IS NULL AND
            prior_version IS NULL AND resulting_version = 1) OR
        (cause_kind = 'authorization_recorded' AND
            authorization_event_id IS NOT NULL AND
            budget_ledger_entry_id IS NULL AND reservation_id IS NULL AND
            reservation_event_id IS NULL AND operational_request_id IS NULL AND
            resulting_version >= 1 AND
            ((resulting_version = 1 AND prior_version IS NULL) OR
             (resulting_version > 1 AND
              prior_version = resulting_version - 1))) OR
        (cause_kind = 'budget_ledger_recorded' AND
            governance_provenance_event_id IS NULL AND
            authorization_event_id IS NULL AND
            budget_ledger_entry_id IS NOT NULL AND reservation_id IS NULL AND
            reservation_event_id IS NULL AND operational_request_id IS NULL AND
            resulting_version >= 1 AND
            ((resulting_version = 1 AND prior_version IS NULL) OR
             (resulting_version > 1 AND
              prior_version = resulting_version - 1))) OR
        (cause_kind = 'reservation_request_accepted' AND
            governance_provenance_event_id IS NULL AND
            authorization_event_id IS NOT NULL AND
            budget_ledger_entry_id IS NOT NULL AND
            reservation_id IS NOT NULL AND
            reservation_event_id IS NOT NULL AND
            operational_request_id IS NOT NULL AND
            prior_version IS NULL AND resulting_version = 1));

DROP INDEX IF EXISTS campaign_operations_audit_authorization_uidx;
CREATE UNIQUE INDEX campaign_operations_audit_authorization_uidx
    ON campaign_operations_audit_reference_event(authorization_event_id)
    WHERE cause_kind = 'authorization_recorded';

DROP INDEX IF EXISTS campaign_operations_audit_budget_entry_uidx;
CREATE UNIQUE INDEX
campaign_operations_audit_budget_entry_uidx
    ON campaign_operations_audit_reference_event(budget_ledger_entry_id)
    WHERE cause_kind = 'budget_ledger_recorded';
CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_audit_reservation_event_uidx
    ON campaign_operations_audit_reference_event(reservation_event_id)
    WHERE reservation_event_id IS NOT NULL;
CREATE UNIQUE INDEX IF NOT EXISTS
campaign_operations_audit_request_uidx
    ON campaign_operations_audit_reference_event(operational_request_id)
    WHERE operational_request_id IS NOT NULL;

CREATE OR REPLACE VIEW campaign_operations_budget_status_v1 AS
WITH reservation_totals AS (
    SELECT operational_campaign_id,
           coalesce(sum(amount), 0)::bigint AS ever_reserved,
           coalesce(sum(amount) FILTER (
               WHERE reservation_state = 'committed'), 0)::bigint
               AS committed,
           coalesce(sum(amount) FILTER (
               WHERE reservation_state IN ('released', 'expired')), 0)::bigint
               AS released_or_expired,
           count(*)::bigint AS reservation_count,
           count(*) FILTER (WHERE EXISTS (
               SELECT 1
               FROM campaign_operations_operational_request request
               WHERE request.reservation_id =
                       campaign_operations_reservation.reservation_id))::
               bigint AS request_count,
           count(*) FILTER (WHERE EXISTS (
               SELECT 1
               FROM campaign_operations_reservation_event event
               WHERE event.reservation_id =
                       campaign_operations_reservation.reservation_id
                 AND event.transition_kind = 'acquired'))::bigint
               AS acquisition_count
    FROM campaign_operations_reservation
    GROUP BY operational_campaign_id
), budget_head AS (
    SELECT DISTINCT ON (operational_campaign_id)
           operational_campaign_id, budget_ledger_entry_id, ledger_version,
           ledger_status, resulting_total, budget_identity_hash
    FROM campaign_operations_budget_ledger_entry
    ORDER BY operational_campaign_id, ledger_version DESC
)
SELECT campaign.operational_campaign_id,
       head.budget_ledger_entry_id,
       head.ledger_version,
       head.ledger_status,
       head.resulting_total AS granted,
       coalesce(totals.ever_reserved, 0)::bigint AS ever_reserved,
       coalesce(totals.committed, 0)::bigint AS committed,
       coalesce(totals.released_or_expired, 0)::bigint
           AS released_or_expired,
       (coalesce(totals.ever_reserved, 0) -
        coalesce(totals.committed, 0) -
        coalesce(totals.released_or_expired, 0))::bigint AS held,
       (coalesce(head.resulting_total, 0) -
        coalesce(totals.ever_reserved, 0) +
        coalesce(totals.released_or_expired, 0))::bigint AS unallocated,
       CASE WHEN head.ledger_status = 'active' THEN
           (head.resulting_total -
            coalesce(totals.ever_reserved, 0) +
            coalesce(totals.released_or_expired, 0))::bigint
       ELSE 0::bigint END AS reservable,
       head.budget_identity_hash,
       (
           coalesce(totals.ever_reserved, 0) >=
               coalesce(totals.committed, 0) +
               coalesce(totals.released_or_expired, 0) AND
           coalesce(head.resulting_total, 0) >=
               coalesce(totals.ever_reserved, 0) -
               coalesce(totals.released_or_expired, 0) AND
           coalesce(totals.reservation_count, 0) =
               coalesce(totals.request_count, 0) AND
           coalesce(totals.reservation_count, 0) =
               coalesce(totals.acquisition_count, 0)
       ) AS accounting_consistent
FROM campaign_operations_campaign campaign
LEFT JOIN budget_head head USING (operational_campaign_id)
LEFT JOIN reservation_totals totals USING (operational_campaign_id);

CREATE OR REPLACE VIEW campaign_operations_request_status_v1 AS
SELECT request.operational_request_id,
       request.operational_campaign_id,
       request.request_state,
       request.state_version AS request_state_version,
       request.request_identity_hash,
       request.production_dispatch_enabled,
       reservation.reservation_id,
       reservation.reservation_state,
       reservation.state_version AS reservation_state_version,
       reservation.amount,
       reservation.budget_unit,
       reservation.expires_at,
       reservation.reservation_identity_hash,
       request.authorization_event_id,
       reservation.budget_ledger_entry_id,
       reservation.budget_ledger_version,
       request.created_at,
       coalesce((
           reservation.operational_campaign_id =
               request.operational_campaign_id AND
           reservation.campaign_identity_canonical =
               request.campaign_identity_canonical AND
           reservation.logical_operation_contract_version =
               request.logical_operation_contract_version AND
           reservation.logical_operation_canonical =
               request.logical_operation_canonical AND
           reservation.logical_operation_hash =
               request.logical_operation_hash AND
           reservation.authorization_event_id =
               request.authorization_event_id AND
           reservation.authorization_identity_canonical =
               request.authorization_identity_canonical AND
           reservation.authorization_identity_hash =
               request.authorization_identity_hash AND
           reservation.reservation_identity_canonical =
               request.reservation_identity_canonical AND
           reservation.reservation_identity_hash =
               request.reservation_identity_hash AND
           reservation.action_kind = request.action_kind AND
           reservation.action_contract_version =
               request.action_contract_version AND
           reservation.recommendation_campaign_materialization_id =
               request.recommendation_campaign_materialization_id AND
           reservation.materialization_contract_version =
               request.materialization_contract_version AND
           reservation.materialization_identity_canonical =
               request.materialization_identity_canonical AND
           reservation.materialization_identity_hash =
               request.materialization_identity_hash AND
           reservation.materialization_member_count =
               request.materialization_member_count AND
           reservation.amount = request.materialization_member_count AND
           request.ordered_scope_digest =
               request.materialization_identity_hash AND
           authz.operational_campaign_id =
               request.operational_campaign_id AND
           authz.campaign_identity_canonical =
               request.campaign_identity_canonical AND
           authz.authorization_identity_canonical =
               request.authorization_identity_canonical AND
           authz.authorization_identity_hash =
               request.authorization_identity_hash AND
           authz.event_kind = 'granted' AND
           authz.action_kind = request.action_kind AND
           authz.action_contract_version =
               request.action_contract_version AND
           authz.scope_kind = reservation.scope_kind AND
           authz.scope_contract_version =
               reservation.scope_contract_version AND
           authz.prerequisite_policy =
               request.prerequisite_policy AND
           authz.provenance_identity_canonical IS NOT DISTINCT FROM
               request.provenance_identity_canonical AND
           authz.provenance_identity_hash IS NOT DISTINCT FROM
               request.provenance_identity_hash AND
           (
               (authz.governance_provenance_event_id IS NULL AND
                   provenance.governance_provenance_event_id IS NULL AND
                   request.provenance_identity_canonical IS NULL AND
                   request.provenance_identity_hash IS NULL) OR
               (authz.governance_provenance_event_id IS NOT NULL AND
                   provenance.governance_provenance_event_id =
                       authz.governance_provenance_event_id AND
                   provenance.operational_campaign_id =
                       request.operational_campaign_id AND
                   provenance.prerequisite_policy =
                       request.prerequisite_policy AND
                   provenance.provenance_identity_canonical =
                       request.provenance_identity_canonical AND
                   provenance.provenance_identity_hash =
                       request.provenance_identity_hash)
           ) AND
           budget.operational_campaign_id =
               request.operational_campaign_id AND
           budget.ledger_version = reservation.budget_ledger_version AND
           budget.budget_identity_canonical =
               reservation.budget_identity_canonical AND
           budget.budget_identity_hash =
               reservation.budget_identity_hash AND
           EXISTS (
               SELECT 1
               FROM campaign_operations_reservation_event event
               WHERE event.reservation_id = reservation.reservation_id
                 AND event.operational_request_id =
                         request.operational_request_id
                 AND event.reservation_identity_canonical =
                         reservation.reservation_identity_canonical
                 AND event.request_identity_canonical =
                         request.request_identity_canonical
                 AND event.transition_kind = 'acquired'
                 AND event.expected_state IS NULL
                 AND event.resulting_state = 'held'
                 AND event.expected_version = 0
                 AND event.resulting_version = 1
                 AND event.amount = reservation.amount) AND
           EXISTS (
               SELECT 1
               FROM campaign_operations_audit_reference_event audit
               WHERE audit.reservation_id = reservation.reservation_id
                 AND audit.operational_request_id =
                         request.operational_request_id
                 AND audit.operational_campaign_id =
                         request.operational_campaign_id
                 AND audit.authorization_event_id =
                         request.authorization_event_id
                 AND audit.budget_ledger_entry_id =
                         reservation.budget_ledger_entry_id
                 AND audit.reservation_event_id = (
                     SELECT event.reservation_event_id
                     FROM campaign_operations_reservation_event event
                     WHERE event.reservation_id = reservation.reservation_id
                       AND event.operational_request_id =
                               request.operational_request_id
                       AND event.transition_kind = 'acquired'
                       AND event.resulting_version = 1)
                 AND audit.cause_kind =
                         'reservation_request_accepted'
                 AND audit.actor_identity =
                         request.accepting_actor_identity
                 AND audit.capability =
                         'campaign_operations_request_acceptor'
                 AND audit.reason = request.reason
                 AND audit.prior_version IS NULL
                 AND audit.resulting_version = 1
                 AND audit.outcome = 'recorded'
                 AND audit.replay_disposition = 'recorded')
       ), false) AS evidence_consistent
FROM campaign_operations_operational_request request
JOIN campaign_operations_reservation reservation
  ON reservation.reservation_id = request.reservation_id
LEFT JOIN campaign_operations_authorization_event authz
  ON authz.authorization_event_id = request.authorization_event_id
LEFT JOIN campaign_operations_governance_provenance_event provenance
  ON provenance.governance_provenance_event_id =
         authz.governance_provenance_event_id
LEFT JOIN campaign_operations_budget_ledger_entry budget
  ON budget.budget_ledger_entry_id = reservation.budget_ledger_entry_id;

COMMENT ON TABLE campaign_operations_budget_ledger_entry IS
    'Append-only V1 materialized-member budget authority; balances derive from this head and reservation evidence.';
COMMENT ON TABLE campaign_operations_reservation IS
    'Guarded V1 full-materialization member-unit reservation; Phase 2 creates held reservations only.';
COMMENT ON TABLE campaign_operations_operational_request IS
    'Durable accepted V1 full-materialization request/outbox; Phase 2 persists ready requests but cannot dispatch them.';
COMMENT ON TABLE campaign_operations_reservation_event IS
    'Append-only reservation transition evidence; Phase 2 records acquisition atomically with held reservation and ready request.';

REVOKE ALL PRIVILEGES ON
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON
    campaign_operations_budget_status_v1,
    campaign_operations_request_status_v1
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_budget_entry() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_reservation() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_request() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_reservation_event() FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_request_acquisition_complete()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_reservation_acquisition_complete()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_budget_audit_complete()
    FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    lock_campaign_operations_campaign(bigint) FROM PUBLIC, pqxx;
REVOKE ALL PRIVILEGES ON FUNCTION
    enforce_campaign_operations_authorization_chain() FROM PUBLIC, pqxx;

DO $$
DECLARE phase_schema text := current_schema();
BEGIN
    EXECUTE format('GRANT USAGE ON SCHEMA %I TO '
        'campaign_operations_budget_administrator, '
        'campaign_operations_request_acceptor', phase_schema);
END $$;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event,
    experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member
    TO campaign_operations_budget_administrator;

GRANT SELECT ON
    campaign_operations_campaign,
    campaign_operations_governance_provenance_event,
    campaign_operations_authorization_event,
    campaign_operations_budget_ledger_entry,
    campaign_operations_reservation,
    campaign_operations_operational_request,
    campaign_operations_reservation_event,
    experiment_recommendation_campaign_materialization,
    experiment_recommendation_campaign_materialization_member,
    experiment_recommendation_campaign_follow_up_proposal,
    experiment_recommendation_campaign_follow_up_proposal_review_event,
    experiment_recommendation_campaign_follow_up_ratification_event
    TO campaign_operations_request_acceptor;

GRANT SELECT ON
    campaign_operations_budget_status_v1,
    campaign_operations_request_status_v1
    TO campaign_operations_budget_administrator,
       campaign_operations_request_acceptor,
       campaign_operations_auditor,
       campaign_operations_reader;

GRANT EXECUTE ON FUNCTION lock_campaign_operations_campaign(bigint)
    TO campaign_operations_budget_administrator,
       campaign_operations_request_acceptor;

GRANT INSERT (
    operational_campaign_id,campaign_identity_canonical,previous_entry_id,
    previous_entry_identity_canonical,previous_entry_identity_hash,
    ledger_version,entry_kind,ledger_status,budget_unit,delta,prior_total,
    resulting_total,administrator_identity,reason,budget_contract_version,
    budget_identity_canonical,budget_identity_hash)
ON campaign_operations_budget_ledger_entry
TO campaign_operations_budget_administrator;

GRANT INSERT (
    operational_campaign_id,campaign_identity_canonical,
    logical_operation_contract_version,logical_operation_canonical,
    logical_operation_hash,authorization_event_id,
    authorization_identity_canonical,authorization_identity_hash,
    budget_ledger_entry_id,budget_ledger_version,
    budget_identity_canonical,budget_identity_hash,action_kind,
    action_contract_version,recommendation_campaign_materialization_id,
    materialization_contract_version,materialization_identity_canonical,
    materialization_identity_hash,scope_kind,scope_contract_version,
    materialization_member_count,amount,budget_unit,expires_at,
    reservation_contract_version,reservation_identity_canonical,
    reservation_identity_hash,reservation_state,state_version)
ON campaign_operations_reservation
TO campaign_operations_request_acceptor;

GRANT INSERT (
    operational_campaign_id,campaign_identity_canonical,
    logical_operation_contract_version,logical_operation_canonical,
    logical_operation_hash,authorization_event_id,
    authorization_identity_canonical,authorization_identity_hash,
    reservation_id,reservation_identity_canonical,
    reservation_identity_hash,action_kind,action_contract_version,
    recommendation_campaign_materialization_id,
    materialization_contract_version,materialization_identity_canonical,
    materialization_identity_hash,ordered_scope_digest,
    materialization_member_count,accepting_actor_identity,reason,
    prerequisite_policy,provenance_identity_canonical,
    provenance_identity_hash,request_contract_version,
    request_identity_canonical,request_identity_hash,request_state,
    state_version)
ON campaign_operations_operational_request
TO campaign_operations_request_acceptor;

GRANT INSERT (
    reservation_id,reservation_identity_canonical,transition_kind,
    expected_state,resulting_state,expected_version,resulting_version,
    operational_request_id,request_identity_canonical,amount,
    reservation_event_contract_version,
    reservation_event_identity_canonical,reservation_event_identity_hash)
ON campaign_operations_reservation_event
TO campaign_operations_request_acceptor;

GRANT INSERT (
    operational_campaign_id,governance_provenance_event_id,
    authorization_event_id,budget_ledger_entry_id,reservation_id,
    reservation_event_id,operational_request_id,cause_kind,actor_identity,
    capability,reason,prior_version,resulting_version,outcome,
    replay_disposition)
ON campaign_operations_audit_reference_event
TO campaign_operations_budget_administrator,
   campaign_operations_request_acceptor;

DO $$
DECLARE
    budget_sequence text := pg_get_serial_sequence(
        'campaign_operations_budget_ledger_entry',
        'budget_ledger_entry_id');
    reservation_sequence text := pg_get_serial_sequence(
        'campaign_operations_reservation', 'reservation_id');
    request_sequence text := pg_get_serial_sequence(
        'campaign_operations_operational_request',
        'operational_request_id');
    reservation_event_sequence text := pg_get_serial_sequence(
        'campaign_operations_reservation_event', 'reservation_event_id');
    audit_sequence text := pg_get_serial_sequence(
        'campaign_operations_audit_reference_event',
        'audit_reference_event_id');
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        budget_sequence, reservation_sequence, request_sequence,
        reservation_event_sequence]
    LOOP
        EXECUTE format(
            'REVOKE ALL PRIVILEGES ON SEQUENCE %s FROM PUBLIC, pqxx',
            sequence_name);
    END LOOP;
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_budget_administrator', budget_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_request_acceptor', reservation_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_request_acceptor', request_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_request_acceptor',
        reservation_event_sequence);
    EXECUTE format('GRANT USAGE ON SEQUENCE %s TO '
        'campaign_operations_budget_administrator, '
        'campaign_operations_request_acceptor', audit_sequence);
END $$;

ALTER TABLE campaign_operations_budget_ledger_entry
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reservation
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_operational_request
    OWNER TO campaign_operations_owner;
ALTER TABLE campaign_operations_reservation_event
    OWNER TO campaign_operations_owner;
ALTER VIEW campaign_operations_budget_status_v1
    OWNER TO campaign_operations_owner;
ALTER VIEW campaign_operations_request_status_v1
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_budget_entry()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_reservation()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_request()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_reservation_event()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_request_acquisition_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_reservation_acquisition_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION enforce_campaign_operations_budget_audit_complete()
    OWNER TO campaign_operations_owner;
ALTER FUNCTION lock_campaign_operations_campaign(bigint)
    OWNER TO campaign_operations_owner;

-- Capability membership is intentionally not granted to pqxx. Enabling a
-- deployment principal remains a separate reviewed administrator action.
