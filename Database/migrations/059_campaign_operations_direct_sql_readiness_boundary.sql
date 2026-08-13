-- Campaign Operations Phase H: direct-SQL readiness boundary correction.
--
-- Migration 055's production V2 transition remains the authoritative atomic
-- state transition.  This migration removes its direct dispatcher EXECUTE
-- edge and adds one sealed service boundary that must prove the same durable
-- readiness contract before entering that transition.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM public.schema_migrations
        WHERE version = '058'
          AND filename = '058_campaign_operations_h3_manager_run_once.sql'
          AND checksum =
            '983404ae310131af3ac42a7b0f3ac19f4a33597023518dc59e201d28d25385c9') THEN
        RAISE EXCEPTION 'H8A001 migration 058 checksum or ledger mismatch'
            USING ERRCODE = '55000';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'campaign_operations_h1_boundary_authority') THEN
        RAISE EXCEPTION 'H8A002 sealed H1 boundary role missing'
            USING ERRCODE = '42501';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_catalog.pg_roles
        WHERE rolname = 'campaign_operations_production_dispatch_service') THEN
        CREATE ROLE campaign_operations_production_dispatch_service NOLOGIN;
    END IF;
    ALTER ROLE campaign_operations_production_dispatch_service
        NOLOGIN NOSUPERUSER NOCREATEDB NOCREATEROLE NOREPLICATION NOBYPASSRLS
        INHERIT CONNECTION LIMIT -1;
END $$;

-- The gate is intentionally private.  It centralizes the database-observable
-- readiness contract instead of trusting a caller-settable GUC or token.  The
-- C++ adapter still performs the executable/build and deployment preflight;
-- this gate is the unavoidable database authorization boundary for the raw
-- atomic transition.
CREATE OR REPLACE FUNCTION
campaign_operations_production_dispatch_readiness_gate_v1(
    approved_build_contract_canonical_value text)
RETURNS void
LANGUAGE plpgsql
STABLE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE readiness record;
DECLARE current_head public.campaign_operations_production_enablement_event%ROWTYPE;
BEGIN
    SELECT snapshot.* INTO STRICT readiness
    FROM public.campaign_operations_production_readiness_snapshot_v1() snapshot;

    IF readiness.migration_version IS DISTINCT FROM '055' OR
       readiness.migration_filename IS DISTINCT FROM
           '055_campaign_operations_production_admission_foundation.sql' OR
       readiness.migration_checksum IS DISTINCT FROM
           '1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe' THEN
        RAISE EXCEPTION 'production dispatch readiness migration identity invalid'
            USING ERRCODE = '23514';
    END IF;
    IF readiness.scheduler_evidence_contract_version IS DISTINCT FROM '1' OR
       readiness.manager_build_contract_version IS DISTINCT FROM '1' OR
       readiness.enablement_contract_version IS DISTINCT FROM '1' OR
       (readiness.admission_contract_version IS NOT NULL AND
        readiness.admission_contract_version IS DISTINCT FROM '1') OR
       (readiness.production_attempt_contract_version IS NOT NULL AND
        readiness.production_attempt_contract_version IS DISTINCT FROM '2') THEN
        RAISE EXCEPTION 'production dispatch readiness contract version invalid'
            USING ERRCODE = '23514';
    END IF;
    IF NOT readiness.scheduler_evidence_complete OR
       readiness.scheduler_required_generation IS DISTINCT FROM 52 OR
       readiness.scheduler_cutover_state IS DISTINCT FROM 'complete' THEN
        RAISE EXCEPTION 'production dispatch readiness scheduler blocker'
            USING ERRCODE = '23514';
    END IF;
    IF readiness.production_enablement_event_id IS NULL OR
       readiness.enablement_kind IS DISTINCT FROM 'enable' OR
       readiness.enablement_effective IS DISTINCT FROM true OR
       readiness.independent_verification_reference IS NULL OR
       readiness.independent_verification_reference = '' THEN
        RAISE EXCEPTION 'production dispatch readiness enablement blocker'
            USING ERRCODE = '23514';
    END IF;
    IF readiness.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value THEN
        RAISE EXCEPTION 'production dispatch readiness build mismatch'
            USING ERRCODE = '23514';
    END IF;
    -- The Manager login is the read-only/readiness principal.  The final
    -- wrapper is callable only by the separate service capability, whose
    -- LOGIN receives this capability plus the existing Phase-5 transactional
    -- capability, but deliberately does not receive the Manager dispatcher
    -- capability.  The wrapper therefore needs an equivalent, narrower
    -- service graph branch here; requiring dispatcher_member unconditionally
    -- would make the trusted service unusable while restoring the original
    -- direct-SQL overlap.
    IF readiness.session_principal IS DISTINCT FROM session_user::text OR
       (
         (
           NOT readiness.reader_member OR
           NOT readiness.dispatcher_member OR
           NOT readiness.phase5_transactional_member OR
           NOT readiness.scheduler_evidence_reader_member OR
           readiness.enabler_member OR readiness.disabler_member
         ) AND
         (
           NOT public.campaign_operations_has_explicit_role_v1(
             session_user::name,
             'campaign_operations_production_dispatch_service'::name) OR
           NOT readiness.phase5_transactional_member OR
           readiness.enabler_member OR readiness.disabler_member OR
           readiness.prohibited_test_dispatcher_member OR
           readiness.prohibited_test_phase5_member
         )
       ) OR
       readiness.prohibited_test_dispatcher_member OR
       readiness.prohibited_test_phase5_member THEN
        RAISE EXCEPTION 'production dispatch readiness principal graph invalid'
            USING ERRCODE = '42501';
    END IF;
    IF readiness.completion_nested_v2_proof_valid IS DISTINCT FROM true OR
       readiness.completion_nested_v2_proof_version IS DISTINCT FROM '1' THEN
        RAISE EXCEPTION 'production dispatch readiness completion proof invalid'
            USING ERRCODE = '23514';
    END IF;
    IF readiness.reconciliation_required_count <> 0 THEN
        RAISE EXCEPTION 'production dispatch readiness reconciliation blocker'
            USING ERRCODE = '23514';
    END IF;

    SELECT event.* INTO STRICT current_head
    FROM public.campaign_operations_production_enablement_event event
    WHERE event.production_enablement_event_id =
          readiness.production_enablement_event_id;
    IF NOT public.campaign_operations_production_enablement_history_valid_v1(
           current_head.production_enablement_event_id) THEN
        RAISE EXCEPTION 'production dispatch readiness enablement evidence corrupt'
            USING ERRCODE = '23514';
    END IF;

    -- Validate all already-materialized production evidence before admitting
    -- another row.  Genesis-empty admission/Attempt V2 families naturally
    -- produce no rows here and remain valid.
    IF EXISTS (
        SELECT 1
        FROM public.campaign_operations_request_production_admission admission
        WHERE admission.admission_contract_version IS DISTINCT FROM 1 OR
              admission.admission_identity_canonical IS DISTINCT FROM
                  public.campaign_operations_request_production_admission_canonical_v1(
                      admission) OR
              admission.admission_identity_hash IS DISTINCT FROM
                  public.campaign_operations_tagged_fnv1a64(
                      admission.admission_identity_canonical)) THEN
        RAISE EXCEPTION 'production dispatch readiness admission evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM public.campaign_operations_dispatch_attempt attempt
        WHERE (attempt.attempt_contract_version = 2 OR
               attempt.request_production_admission_id IS NOT NULL OR
               attempt.request_production_admission_canonical IS NOT NULL OR
               attempt.request_production_admission_hash IS NOT NULL OR
               attempt.production_enablement_event_id IS NOT NULL OR
               attempt.production_enablement_event_canonical IS NOT NULL OR
               attempt.production_enablement_event_hash IS NOT NULL OR
               attempt.operation_key IS NOT NULL OR
               attempt.requesting_actor IS NOT NULL OR
               attempt.original_executing_service_principal IS NOT NULL OR
               attempt.approved_build_contract_canonical IS NOT NULL OR
               attempt.approved_build_contract_hash IS NOT NULL OR
               attempt.production_capability IS NOT NULL) AND
              (attempt.attempt_contract_version IS DISTINCT FROM 2 OR
               attempt.attempt_identity_canonical IS DISTINCT FROM
                   public.campaign_operations_dispatch_attempt_v2_canonical(
                       attempt) OR
               attempt.attempt_identity_hash IS DISTINCT FROM
                   public.campaign_operations_tagged_fnv1a64(
                       attempt.attempt_identity_canonical) OR
               NOT EXISTS (
                   SELECT 1
                   FROM public.campaign_operations_request_production_admission admission
                   WHERE admission.request_production_admission_id =
                         attempt.request_production_admission_id))) THEN
        RAISE EXCEPTION 'production dispatch readiness attempt evidence corrupt'
            USING ERRCODE = '23514';
    END IF;
END;
$$;

ALTER FUNCTION campaign_operations_production_dispatch_readiness_gate_v1(text)
    OWNER TO campaign_operations_h1_boundary_authority;

CREATE OR REPLACE FUNCTION
campaign_operations_production_dispatch_authorized_v3(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, lease_expires_at_value timestamptz,
    operation_key_value text, requesting_actor_value text,
    approved_build_contract_canonical_value text)
RETURNS public.campaign_operations_dispatch_attempt
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE attempt public.campaign_operations_dispatch_attempt%ROWTYPE;
BEGIN
    PERFORM public.campaign_operations_production_dispatch_readiness_gate_v1(
        approved_build_contract_canonical_value);
    SELECT result.* INTO STRICT attempt
    FROM public.transition_campaign_operations_request_dispatch_production_v2(
        target_request_id, expected_version_value, lease_digest_value,
        lease_expires_at_value, operation_key_value, requesting_actor_value,
        approved_build_contract_canonical_value) result;
    RETURN attempt;
END;
$$;

ALTER FUNCTION campaign_operations_production_dispatch_authorized_v3(
        bigint, integer, text, timestamptz, text, text, text)
    OWNER TO campaign_operations_h1_boundary_authority;

REVOKE ALL PRIVILEGES ON FUNCTION
    transition_campaign_operations_request_dispatch_production_v2(
        bigint, integer, text, timestamptz, text, text, text)
    FROM PUBLIC, pqxx, campaign_operations_production_dispatcher;
REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_production_dispatch_readiness_gate_v1(text)
    FROM PUBLIC, pqxx, campaign_operations_production_enabler,
         campaign_operations_production_disabler,
         campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader,
         campaign_operations_scheduler_protocol_evidence_reader;
REVOKE ALL PRIVILEGES ON FUNCTION
    campaign_operations_production_dispatch_authorized_v3(
        bigint, integer, text, timestamptz, text, text, text)
    FROM PUBLIC, pqxx, campaign_operations_production_enabler,
         campaign_operations_production_disabler,
         campaign_operations_production_dispatcher,
         campaign_operations_production_phase5_transactional,
         campaign_operations_production_reader,
         campaign_operations_scheduler_protocol_evidence_reader;
GRANT EXECUTE ON FUNCTION
    campaign_operations_production_dispatch_authorized_v3(
        bigint, integer, text, timestamptz, text, text, text)
    TO campaign_operations_h1_boundary_authority;
GRANT EXECUTE ON FUNCTION
    campaign_operations_production_dispatch_authorized_v3(
        bigint, integer, text, timestamptz, text, text, text)
    TO campaign_operations_production_dispatch_service;

DO $$
BEGIN
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
            function_row.proacl,
            pg_catalog.acldefault('f', function_row.proowner))) acl
        WHERE function_row.oid =
            'public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure
          AND (acl.grantee = 0 OR acl.is_grantable OR
               acl.grantee = 'campaign_operations_production_dispatcher'::regrole)) THEN
        RAISE EXCEPTION 'H8A003 raw production transition remains reachable'
            USING ERRCODE = '42501';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        WHERE function_row.oid =
            'public.campaign_operations_production_dispatch_authorized_v3(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure
          AND function_row.proowner =
              'campaign_operations_h1_boundary_authority'::regrole
          AND function_row.prosecdef
          AND has_function_privilege(
              'campaign_operations_production_dispatch_service', function_row.oid,
              'EXECUTE')
          AND NOT has_function_privilege(
              'campaign_operations_production_dispatcher', function_row.oid,
              'EXECUTE')
          AND NOT has_function_privilege('public', function_row.oid, 'EXECUTE')
          AND NOT has_function_privilege('pqxx', function_row.oid, 'EXECUTE')) THEN
        RAISE EXCEPTION 'H8A004 authorized production boundary ACL mismatch'
            USING ERRCODE = '42501';
    END IF;
END $$;
