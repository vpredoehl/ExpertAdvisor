-- Campaign Operations Phase H2: production handoff predecessor contract.
--
-- Migration 048's three-argument bound transition remains the immutable
-- isolated/Test Attempt V1 transition.  Migration 055 production acquisition
-- intentionally leaves the request's one-way production admission witness
-- true, so H2 needs a separate versioned transition rather than weakening the
-- V1 predicate or reverting that witness.

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM public.schema_migrations
        WHERE version = '056'
          AND filename = '056_campaign_operations_h2_privilege_deployment_contract.sql'
          AND checksum =
            '4616e269f6a670e6edbc0572a029c48531c55f48c1cf9ed951624c83fc5025cc') THEN
        RAISE EXCEPTION 'H2A007 migration 056 checksum or ledger mismatch'
            USING ERRCODE = '55000';
    END IF;
END $$;

CREATE OR REPLACE FUNCTION
transition_campaign_operations_request_bound_production_v2(
    target_request_id bigint, expected_version_value integer,
    lease_digest_value text, dispatch_attempt_id_value bigint,
    operation_key_value text, approved_build_contract_canonical_value text)
RETURNS public.campaign_operations_operational_request
LANGUAGE plpgsql
VOLATILE
SECURITY DEFINER
SET search_path = pg_catalog, public
AS $$
DECLARE request_row public.campaign_operations_operational_request%ROWTYPE;
DECLARE attempt_row public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE admission_row public.campaign_operations_request_production_admission%ROWTYPE;
DECLARE first_attempt_row public.campaign_operations_dispatch_attempt%ROWTYPE;
DECLARE enablement_row public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE current_enablement_row public.campaign_operations_production_enablement_event%ROWTYPE;
DECLARE changed public.campaign_operations_operational_request%ROWTYPE;
DECLARE outcome_count integer;
BEGIN
    SELECT candidate.* INTO STRICT request_row
    FROM public.campaign_operations_operational_request candidate
    WHERE candidate.operational_request_id = target_request_id
    FOR UPDATE;

    SELECT candidate.* INTO STRICT attempt_row
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.dispatch_attempt_id = dispatch_attempt_id_value;

    SELECT candidate.* INTO STRICT admission_row
    FROM public.campaign_operations_request_production_admission candidate
    WHERE candidate.operational_request_id = target_request_id;

    -- The immutable admission records the first acquisition key.  Recovery
    -- may produce a later Attempt V2 under a new caller key; bind ownership
    -- is therefore the exact producing attempt, while this relationship
    -- proves the admission still belongs to its original first attempt.
    SELECT candidate.* INTO STRICT first_attempt_row
    FROM public.campaign_operations_dispatch_attempt candidate
    WHERE candidate.request_production_admission_id =
          admission_row.request_production_admission_id
      AND candidate.attempt_contract_version = 2
      AND candidate.attempt_ordinal = 1;

    SELECT candidate.* INTO STRICT enablement_row
    FROM public.campaign_operations_production_enablement_event candidate
    WHERE candidate.production_enablement_event_id =
          attempt_row.production_enablement_event_id;

    SELECT candidate.* INTO STRICT current_enablement_row
    FROM public.campaign_operations_production_enablement_event candidate
    ORDER BY candidate.resulting_version DESC
    LIMIT 1;

    SELECT count(*)::integer INTO outcome_count
    FROM public.campaign_operations_dispatch_attempt_outcome outcome
    WHERE outcome.dispatch_attempt_id = dispatch_attempt_id_value;

    -- This is the exact H1-produced predecessor.  The mutable Boolean is
    -- checked as a one-way admission/Attempt V2 witness, while all durable
    -- production identities are re-read from their authoritative rows.
    IF request_row.request_state <> 'dispatching' OR
       request_row.state_version <> expected_version_value OR
       request_row.production_dispatch_enabled IS DISTINCT FROM true OR
       request_row.lease_token_hash IS DISTINCT FROM lease_digest_value OR
       request_row.lease_expires_at <= pg_catalog.transaction_timestamp() OR
       attempt_row.attempt_contract_version <> 2 OR
       attempt_row.operational_request_id <> target_request_id OR
       attempt_row.request_identity_canonical <>
           request_row.request_identity_canonical OR
       attempt_row.expected_request_version <> expected_version_value - 1 OR
       attempt_row.resulting_request_version <> expected_version_value OR
       attempt_row.lease_token_digest <> lease_digest_value OR
       attempt_row.lease_expires_at <> request_row.lease_expires_at OR
       attempt_row.dispatcher_identity <> request_row.dispatcher_identity OR
       attempt_row.operation_key IS DISTINCT FROM operation_key_value OR
       attempt_row.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value OR
       admission_row.operational_request_id <> target_request_id OR
       admission_row.request_identity_canonical <>
           request_row.request_identity_canonical OR
       admission_row.expected_request_version <>
           first_attempt_row.expected_request_version OR
       first_attempt_row.operational_request_id <> target_request_id OR
       first_attempt_row.request_production_admission_id <>
           admission_row.request_production_admission_id OR
       first_attempt_row.operation_key IS DISTINCT FROM
           admission_row.dispatch_operation_key OR
       admission_row.request_production_admission_id <>
           attempt_row.request_production_admission_id OR
       attempt_row.production_enablement_event_id <>
           admission_row.production_enablement_event_id OR
       attempt_row.request_production_admission_canonical <>
           admission_row.admission_identity_canonical OR
       attempt_row.request_production_admission_hash <>
           admission_row.admission_identity_hash OR
       attempt_row.production_enablement_event_canonical <>
           enablement_row.enablement_identity_canonical OR
       attempt_row.production_enablement_event_hash <>
           enablement_row.enablement_identity_hash OR
       enablement_row.event_kind <> 'enable' OR
       enablement_row.production_enablement_event_id <>
           current_enablement_row.production_enablement_event_id OR
       current_enablement_row.event_kind <> 'enable' OR
       current_enablement_row.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value OR
       enablement_row.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value OR
       admission_row.approved_build_contract_canonical IS DISTINCT FROM
           approved_build_contract_canonical_value OR
       outcome_count <> 0 THEN
        RAISE EXCEPTION
            'campaign operations production bind predecessor invalid'
            USING ERRCODE = '40001';
    END IF;

    UPDATE public.campaign_operations_operational_request candidate
       SET request_state = 'bound',
           state_version = candidate.state_version + 1,
           lease_token_hash = NULL,
           lease_expires_at = NULL,
           dispatcher_identity = NULL,
           updated_at = pg_catalog.transaction_timestamp()
     WHERE candidate.operational_request_id = target_request_id
       AND candidate.request_state = 'dispatching'
       AND candidate.state_version = expected_version_value
       AND candidate.lease_token_hash = lease_digest_value
       AND candidate.lease_expires_at > pg_catalog.transaction_timestamp()
       AND candidate.production_dispatch_enabled = true
     RETURNING candidate.* INTO changed;
    IF NOT FOUND THEN
        RAISE EXCEPTION
            'campaign operations production bind compare-and-set lost'
            USING ERRCODE = 'P0001';
    END IF;
    RETURN changed;
END;
$$;

ALTER FUNCTION transition_campaign_operations_request_bound_production_v2(
    bigint, integer, text, bigint, text, text)
    OWNER TO campaign_operations_owner;

REVOKE ALL PRIVILEGES ON FUNCTION
    transition_campaign_operations_request_bound_production_v2(
        bigint, integer, text, bigint, text, text)
    FROM PUBLIC, pqxx, campaign_operations_dispatcher,
         campaign_operations_phase5_transactional,
         campaign_operations_production_dispatcher;
GRANT EXECUTE ON FUNCTION
    transition_campaign_operations_request_bound_production_v2(
        bigint, integer, text, bigint, text, text)
    TO campaign_operations_production_phase5_transactional;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        WHERE function_row.oid =
            'public.transition_campaign_operations_request_bound_production_v2(bigint,integer,text,bigint,text,text)'::regprocedure
          AND function_row.proowner =
            'campaign_operations_owner'::regrole
          AND has_function_privilege(
            'campaign_operations_production_phase5_transactional',
            function_row.oid, 'EXECUTE')
          AND NOT has_function_privilege('public', function_row.oid, 'EXECUTE')
          AND NOT has_function_privilege('pqxx', function_row.oid, 'EXECUTE')
          AND NOT has_function_privilege(
            'campaign_operations_dispatcher', function_row.oid, 'EXECUTE')) THEN
        RAISE EXCEPTION 'H2A008 production bind transition ACL mismatch'
            USING ERRCODE = '42501';
    END IF;
END $$;
