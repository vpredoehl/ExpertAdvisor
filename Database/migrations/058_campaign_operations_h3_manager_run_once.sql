-- Campaign Operations Phase H3: bounded Manager run-once evidence.
--
-- H1/H2 Attempt V2 rows remain byte-for-byte immutable.  This additive
-- relation stores the complete deterministic Manager source canonical in a
-- one-to-one row attached to the exact Attempt V2 that acquired it.  It is
-- written in the same acquisition transaction and has no batch identity.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM public.schema_migrations
        WHERE version = '057'
          AND filename = '057_campaign_operations_h2_production_bind_state_contract.sql'
          AND checksum =
            'd0b9339fdac2366addf2ea5b322ca5a7cd08de39d35a38833d46692a5533399b') THEN
        RAISE EXCEPTION 'H3A001 migration 057 checksum or ledger mismatch'
            USING ERRCODE = '55000';
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS
campaign_operations_dispatch_manager_operation (
    dispatch_attempt_id bigint PRIMARY KEY CHECK (dispatch_attempt_id > 0),
    operational_request_id bigint NOT NULL,
    operation_key text COLLATE "C" NOT NULL CHECK (
        operation_key ~ '^mgr-v1:fnv1a64:[0-9a-f]{16}:[1-9][0-9]*$'),
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    source_canonical text COLLATE "C" NOT NULL CHECK (
        source_canonical <> '' AND octet_length(source_canonical) <= 134217728),
    source_hash text COLLATE "C" NOT NULL CHECK (
        source_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    contract_version integer NOT NULL CHECK (contract_version = 1),
    created_at timestamptz NOT NULL DEFAULT transaction_timestamp(),
    CONSTRAINT campaign_operations_manager_attempt_fk FOREIGN KEY
        (dispatch_attempt_id) REFERENCES campaign_operations_dispatch_attempt(
            dispatch_attempt_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_manager_request_fk FOREIGN KEY
        (operational_request_id) REFERENCES campaign_operations_operational_request(
            operational_request_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_manager_request_key_uidx UNIQUE
        (operational_request_id, operation_key)
);

ALTER TABLE campaign_operations_dispatch_manager_operation
    OWNER TO campaign_operations_owner;

-- H2 accepted caller keys before 058 used the frozen general operation-key
-- grammar, which included the mgr-v1: prefix.  Snapshot only the finite,
-- complete H2 evidence that already exists at this upgrade boundary.  The
-- relation is an immutable historical authority record, not a route for a
-- post-058 caller to claim the namespace.
CREATE TABLE IF NOT EXISTS campaign_operations_h2_manager_key_compatibility (
    dispatch_attempt_id bigint PRIMARY KEY CHECK (dispatch_attempt_id > 0),
    operational_request_id bigint NOT NULL CHECK (operational_request_id > 0),
    operation_key text COLLATE "C" NOT NULL CHECK (
        operation_key ~ '^mgr-v1:[A-Za-z0-9._:/-]{0,120}$'),
    request_identity_canonical text COLLATE "C" NOT NULL CHECK (
        request_identity_canonical <> ''),
    expected_request_version integer NOT NULL CHECK (
        expected_request_version > 0),
    requesting_actor text COLLATE "C" NOT NULL CHECK (
        requesting_actor <> 'campaign_operations_manager'),
    approved_build_contract_canonical text COLLATE "C" NOT NULL CHECK (
        approved_build_contract_canonical <> ''),
    attempt_identity_canonical text COLLATE "C" NOT NULL CHECK (
        attempt_identity_canonical <> ''),
    attempt_identity_hash text COLLATE "C" NOT NULL CHECK (
        attempt_identity_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    CONSTRAINT campaign_operations_h2_manager_key_compat_attempt_fk FOREIGN KEY
        (dispatch_attempt_id) REFERENCES campaign_operations_dispatch_attempt(
            dispatch_attempt_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_h2_manager_key_compat_request_fk FOREIGN KEY
        (operational_request_id) REFERENCES campaign_operations_operational_request(
            operational_request_id) ON DELETE RESTRICT,
    CONSTRAINT campaign_operations_h2_manager_key_compat_request_key_uidx UNIQUE
        (operational_request_id, operation_key)
);

ALTER TABLE campaign_operations_h2_manager_key_compatibility
    OWNER TO campaign_operations_owner;

INSERT INTO campaign_operations_h2_manager_key_compatibility(
    dispatch_attempt_id,operational_request_id,operation_key,
    request_identity_canonical,expected_request_version,requesting_actor,
    approved_build_contract_canonical,attempt_identity_canonical,
    attempt_identity_hash)
SELECT attempt.dispatch_attempt_id,attempt.operational_request_id,
       attempt.operation_key,attempt.request_identity_canonical,
       attempt.expected_request_version,attempt.requesting_actor,
       attempt.approved_build_contract_canonical,
       attempt.attempt_identity_canonical,attempt.attempt_identity_hash
FROM campaign_operations_dispatch_attempt attempt
JOIN campaign_operations_request_production_admission admission
  ON admission.request_production_admission_id =
     attempt.request_production_admission_id
JOIN campaign_operations_production_enablement_event enablement
  ON enablement.production_enablement_event_id =
     attempt.production_enablement_event_id
WHERE attempt.attempt_contract_version = 2
  AND attempt.operation_key LIKE 'mgr-v1:%'
  AND attempt.dispatcher_identity = attempt.requesting_actor
  AND attempt.dispatcher_identity <> 'campaign_operations_manager'
  AND attempt.requesting_actor <> 'campaign_operations_manager'
  AND attempt.operational_request_id = admission.operational_request_id
  AND attempt.request_identity_canonical = admission.request_identity_canonical
  AND attempt.expected_request_version = admission.expected_request_version
  AND attempt.operation_key = admission.dispatch_operation_key
  AND attempt.requesting_actor = admission.requesting_actor
  AND attempt.approved_build_contract_canonical =
      admission.approved_build_contract_canonical
  AND attempt.request_production_admission_canonical =
      public.campaign_operations_request_production_admission_canonical_v1(admission)
  AND attempt.request_production_admission_hash = public.campaign_operations_tagged_fnv1a64(
      attempt.request_production_admission_canonical)
  AND attempt.production_enablement_event_canonical =
      public.campaign_operations_production_enablement_canonical_v1(enablement)
  AND attempt.production_enablement_event_hash = public.campaign_operations_tagged_fnv1a64(
      attempt.production_enablement_event_canonical)
  AND attempt.attempt_identity_canonical =
      public.campaign_operations_dispatch_attempt_v2_canonical(attempt)
  AND attempt.attempt_identity_hash = public.campaign_operations_tagged_fnv1a64(
      attempt.attempt_identity_canonical)
  AND EXISTS (
      SELECT 1 FROM campaign_operations_dispatch_audit_reference_event audit
      WHERE audit.dispatch_attempt_id = attempt.dispatch_attempt_id
        AND audit.operational_request_id = attempt.operational_request_id
        AND audit.cause_kind = 'dispatch_lease_acquired'
        AND audit.actor_identity = attempt.requesting_actor
        AND audit.capability = 'campaign_operations_production_dispatcher'
        AND audit.outcome = 'recorded'
        AND audit.replay_disposition = 'new_operation'
        AND audit.diagnostic_code = 'dispatch_lease_acquired')
  AND NOT EXISTS (
      SELECT 1 FROM campaign_operations_dispatch_manager_operation evidence
      WHERE evidence.dispatch_attempt_id = attempt.dispatch_attempt_id)
ON CONFLICT DO NOTHING;

REVOKE ALL PRIVILEGES ON campaign_operations_h2_manager_key_compatibility
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher;
GRANT SELECT ON campaign_operations_h2_manager_key_compatibility
    TO campaign_operations_production_phase5_transactional;

CREATE OR REPLACE FUNCTION
reject_campaign_operations_h2_manager_key_compat_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'H2 Manager-key compatibility evidence is immutable'
        USING ERRCODE = '55000';
END;
$$;

ALTER FUNCTION reject_campaign_operations_h2_manager_key_compat_mutation()
    OWNER TO campaign_operations_owner;
REVOKE ALL PRIVILEGES ON FUNCTION
    reject_campaign_operations_h2_manager_key_compat_mutation()
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;
DROP TRIGGER IF EXISTS campaign_operations_h2_manager_key_compatibility_immutable
    ON campaign_operations_h2_manager_key_compatibility;
CREATE TRIGGER campaign_operations_h2_manager_key_compatibility_immutable
BEFORE UPDATE OR DELETE ON campaign_operations_h2_manager_key_compatibility
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_h2_manager_key_compat_mutation();
DROP TRIGGER IF EXISTS campaign_operations_h2_manager_key_compatibility_truncate
    ON campaign_operations_h2_manager_key_compatibility;
CREATE TRIGGER campaign_operations_h2_manager_key_compatibility_truncate
BEFORE TRUNCATE ON campaign_operations_h2_manager_key_compatibility
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_h2_manager_key_compat_mutation();
REVOKE UPDATE, DELETE, TRUNCATE
    ON campaign_operations_h2_manager_key_compatibility
    FROM campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;

CREATE INDEX IF NOT EXISTS campaign_operations_manager_source_hash_idx
    ON campaign_operations_dispatch_manager_operation(source_hash);

REVOKE ALL PRIVILEGES ON campaign_operations_dispatch_manager_operation
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher;
GRANT SELECT ON campaign_operations_dispatch_manager_operation
    TO campaign_operations_production_reader,
       campaign_operations_production_phase5_transactional;
GRANT INSERT (
    dispatch_attempt_id, operational_request_id, operation_key,
    request_identity_canonical, expected_request_version, source_canonical,
    source_hash, contract_version)
    ON campaign_operations_dispatch_manager_operation
    TO campaign_operations_production_phase5_transactional;

CREATE OR REPLACE FUNCTION
validate_campaign_operations_manager_operation_insert()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE expected_source text;
DECLARE expected_request_identity text;
BEGIN
    IF NEW.contract_version <> 1 OR
       NEW.operation_key !~ '^mgr-v1:fnv1a64:[0-9a-f]{16}:[1-9][0-9]*$' OR
       NEW.expected_request_version <= 0 THEN
        RAISE EXCEPTION 'H3 manager operation typed evidence invalid'
            USING ERRCODE = '23514';
    END IF;
    SELECT 'campaign_operations_manager_request_operation_v1' ||
           ';request_identity_canonical=' ||
           octet_length(attempt.request_identity_canonical)::text || ':' ||
           attempt.request_identity_canonical ||
           ';expected_request_version=' ||
           attempt.expected_request_version::text,
           attempt.request_identity_canonical
      INTO expected_source, expected_request_identity
    FROM public.campaign_operations_dispatch_attempt attempt
    WHERE attempt.dispatch_attempt_id = NEW.dispatch_attempt_id
      AND attempt.attempt_contract_version = 2
      AND attempt.operational_request_id = NEW.operational_request_id
      AND attempt.operation_key = NEW.operation_key
      AND attempt.expected_request_version = NEW.expected_request_version
      AND attempt.dispatcher_identity = 'campaign_operations_manager'
      AND attempt.requesting_actor = 'campaign_operations_manager';
    IF NOT FOUND OR NEW.request_identity_canonical <> expected_request_identity OR
       NEW.source_canonical <> expected_source OR
       NEW.source_hash <> public.campaign_operations_tagged_fnv1a64(
           NEW.source_canonical) OR
       NEW.operation_key <> 'mgr-v1:' || NEW.source_hash || ':' ||
           NEW.expected_request_version::text THEN
        RAISE EXCEPTION 'H3 manager operation source evidence mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

ALTER FUNCTION validate_campaign_operations_manager_operation_insert()
    OWNER TO campaign_operations_owner;
REVOKE ALL PRIVILEGES ON FUNCTION
    validate_campaign_operations_manager_operation_insert()
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher;
GRANT EXECUTE ON FUNCTION validate_campaign_operations_manager_operation_insert()
    TO campaign_operations_production_phase5_transactional;

DROP TRIGGER IF EXISTS campaign_operations_manager_operation_validate
    ON campaign_operations_dispatch_manager_operation;
CREATE TRIGGER campaign_operations_manager_operation_validate
BEFORE INSERT ON campaign_operations_dispatch_manager_operation
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_operations_manager_operation_insert();

-- A Manager key is a reserved production-dispatch namespace.  The evidence
-- row above proves the forward association when it is inserted; this deferred
-- constraint trigger proves the reverse association at COMMIT.  In
-- particular, an H2 caller cannot acquire a durable mgr-v1 Attempt V2 and
-- leave it available for a later Manager recovery to adopt.  The trigger is
-- deliberately deferred so the authoritative H3 acquisition can insert its
-- immutable Attempt V2 before inserting this dependent source row in the
-- same transaction.
CREATE OR REPLACE FUNCTION
validate_campaign_operations_manager_attempt_completeness()
RETURNS trigger
LANGUAGE plpgsql
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE attempt public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE evidence public.campaign_operations_dispatch_manager_operation%ROWTYPE;
DECLARE evidence_count integer;
DECLARE expected_source text;
BEGIN
    SELECT candidate.* INTO attempt
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.dispatch_attempt_id = NEW.dispatch_attempt_id;
    IF NOT FOUND OR attempt.attempt_contract_version <> 2 OR
       attempt.operation_key NOT LIKE 'mgr-v1:%' THEN
        RETURN NEW;
    END IF;

    -- The source relation is Manager-only.  Treat a caller which merely
    -- borrows the reserved key namespace as corrupt rather than permitting a
    -- source row for a non-Manager Attempt V2.
    IF attempt.dispatcher_identity <> 'campaign_operations_manager' OR
       attempt.requesting_actor <> 'campaign_operations_manager' OR
       attempt.operation_key !~ '^mgr-v1:fnv1a64:[0-9a-f]{16}:[1-9][0-9]*$' THEN
        RAISE EXCEPTION 'H3 Manager Attempt V2 identity is invalid'
            USING ERRCODE = '23514';
    END IF;

    SELECT count(*)::integer INTO evidence_count
    FROM public.campaign_operations_dispatch_manager_operation candidate
    WHERE candidate.dispatch_attempt_id = attempt.dispatch_attempt_id;
    IF evidence_count <> 1 THEN
        RAISE EXCEPTION 'H3 Manager Attempt V2 source evidence incomplete'
            USING ERRCODE = '23514';
    END IF;

    SELECT candidate.* INTO STRICT evidence
    FROM public.campaign_operations_dispatch_manager_operation candidate
    WHERE candidate.dispatch_attempt_id = attempt.dispatch_attempt_id;
    expected_source := 'campaign_operations_manager_request_operation_v1' ||
        ';request_identity_canonical=' ||
        octet_length(attempt.request_identity_canonical)::text || ':' ||
        attempt.request_identity_canonical ||
        ';expected_request_version=' || attempt.expected_request_version::text;
    IF evidence.operational_request_id <> attempt.operational_request_id OR
       evidence.operation_key <> attempt.operation_key OR
       evidence.request_identity_canonical <> attempt.request_identity_canonical OR
       evidence.expected_request_version <> attempt.expected_request_version OR
       evidence.source_canonical <> expected_source OR
       evidence.source_hash <> public.campaign_operations_tagged_fnv1a64(
           evidence.source_canonical) OR
       evidence.operation_key <> 'mgr-v1:' || evidence.source_hash || ':' ||
           evidence.expected_request_version::text OR
       evidence.contract_version <> 1 THEN
        RAISE EXCEPTION 'H3 Manager Attempt V2 source evidence mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

ALTER FUNCTION validate_campaign_operations_manager_attempt_completeness()
    OWNER TO campaign_operations_owner;
REVOKE ALL PRIVILEGES ON FUNCTION
    validate_campaign_operations_manager_attempt_completeness()
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;

DROP TRIGGER IF EXISTS campaign_operations_manager_attempt_complete
    ON campaign_operations_dispatch_attempt;
CREATE CONSTRAINT TRIGGER campaign_operations_manager_attempt_complete
AFTER INSERT OR UPDATE ON campaign_operations_dispatch_attempt
DEFERRABLE INITIALLY DEFERRED
FOR EACH ROW EXECUTE FUNCTION
    validate_campaign_operations_manager_attempt_completeness();

CREATE OR REPLACE FUNCTION reject_campaign_operations_manager_operation_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'H3 manager operation evidence is immutable'
        USING ERRCODE = '55000';
END;
$$;

ALTER FUNCTION reject_campaign_operations_manager_operation_mutation()
    OWNER TO campaign_operations_owner;
REVOKE ALL PRIVILEGES ON FUNCTION
    reject_campaign_operations_manager_operation_mutation()
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;

DROP TRIGGER IF EXISTS campaign_operations_manager_operation_immutable
    ON campaign_operations_dispatch_manager_operation;
CREATE TRIGGER campaign_operations_manager_operation_immutable
BEFORE UPDATE OR DELETE ON campaign_operations_dispatch_manager_operation
FOR EACH ROW EXECUTE FUNCTION
    reject_campaign_operations_manager_operation_mutation();
DROP TRIGGER IF EXISTS campaign_operations_manager_operation_truncate
    ON campaign_operations_dispatch_manager_operation;
CREATE TRIGGER campaign_operations_manager_operation_truncate
BEFORE TRUNCATE ON campaign_operations_dispatch_manager_operation
FOR EACH STATEMENT EXECUTE FUNCTION
    reject_campaign_operations_manager_operation_mutation();

REVOKE UPDATE, DELETE, TRUNCATE
    ON campaign_operations_dispatch_manager_operation
    FROM campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader;
