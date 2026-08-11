-- Campaign Operations Phase H2 privilege/deployment contract.
--
-- This migration is deliberately additive.  Migration 055 remains the H1
-- authority and is not rewritten here.  No LOGIN is created and no role is
-- granted to a LOGIN.  The only mutation of persisted catalog authority is
-- the exact H2 EXECUTE/read/helper ACL surface below.

DO $$
DECLARE
    boundary_oid oid;
    role_name text;
    role_oid oid;
    contract record;
    actual_count integer;
    allowed_grantees oid[];
    actual_grantee boolean;
BEGIN
    SELECT oid INTO boundary_oid
    FROM pg_catalog.pg_roles
    WHERE rolname = 'campaign_operations_h1_boundary_authority';
    IF boundary_oid IS NULL THEN
        RAISE EXCEPTION 'H2A001 required H1 boundary role missing'
            USING ERRCODE = '42501';
    END IF;

    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_reader']
    LOOP
        SELECT oid INTO role_oid
        FROM pg_catalog.pg_roles
        WHERE rolname = role_name;
        IF role_oid IS NULL THEN
            RAISE EXCEPTION 'H2A001 required H2 capability role missing: %',
                role_name USING ERRCODE = '42501';
        END IF;
        IF EXISTS (
            SELECT 1 FROM pg_catalog.pg_authid role
            WHERE role.oid = role_oid
              AND (role.rolcanlogin OR role.rolsuper OR role.rolcreatedb OR
                   role.rolcreaterole OR role.rolreplication OR
                   role.rolbypassrls OR NOT role.rolinherit OR
                   role.rolconnlimit <> -1 OR role.rolpassword IS NOT NULL OR
                   role.rolvaliduntil IS NOT NULL)) THEN
            RAISE EXCEPTION 'H2A001 capability role attributes mismatch: %',
                role_name USING ERRCODE = '42501';
        END IF;
    END LOOP;

    -- H2 permits only direct capability-to-LOGIN edges.  Deployment-time
    -- membership is still outside migrations, but unsafe pre-existing graph
    -- state must fail closed rather than be normalized.
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_auth_members membership
        JOIN pg_catalog.pg_roles granted ON granted.oid = membership.roleid
        JOIN pg_catalog.pg_roles member_role ON member_role.oid = membership.member
        WHERE granted.rolname = ANY (ARRAY[
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_reader'])
          AND (NOT member_role.rolcanlogin OR membership.admin_option)) OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_auth_members membership
        JOIN pg_catalog.pg_roles granted ON granted.oid = membership.roleid
        JOIN pg_catalog.pg_roles member_role ON member_role.oid = membership.member
        WHERE member_role.rolname = ANY (ARRAY[
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_reader'])
           OR granted.rolname = 'campaign_operations_h1_boundary_authority'
           OR member_role.rolname = 'campaign_operations_h1_boundary_authority') THEN
        RAISE EXCEPTION 'H2A002 incompatible H2 role graph'
            USING ERRCODE = '42501';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM public.schema_migrations
        WHERE version = '055'
          AND filename = '055_campaign_operations_production_admission_foundation.sql'
          AND checksum =
            'dd01812b04f0f48ab8caac40a5280ff5c6831ed2fc4f53ceb9774e0dc92c3ff0') THEN
        RAISE EXCEPTION 'H2A004 migration 055 checksum or ledger mismatch'
            USING ERRCODE = '55000';
    END IF;

    -- Before first application, H1's unchanged audit must still see its
    -- accepted owner-only fixed-transition ACL.  On replay, the exact H2 ACL
    -- is already present and the H1 baseline audit is intentionally not called
    -- after H2 has extended the fixed-function ACL.
    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        JOIN pg_catalog.pg_namespace namespace
          ON namespace.oid = function_row.pronamespace
        CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
        WHERE namespace.nspname = 'public'
          AND function_row.oid IN (
            'record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
            'record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
            'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure)
          AND acl.grantee <> boundary_oid) THEN
        PERFORM public.campaign_operations_h1_deployment_audit_v1(NULL, false, false);
    END IF;

    -- Fixed H1 transitions: a capability receives exactly one transition.
    FOR contract IN
        SELECT * FROM (VALUES
          ('record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::regprocedure,
           'campaign_operations_production_enabler'::name),
          ('record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::regprocedure,
           'campaign_operations_production_disabler'::name),
          ('transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'::regprocedure,
           'campaign_operations_production_dispatcher'::name)) expected(function_id, capability)
    LOOP
        SELECT function_row.proowner,
               function_row.proacl IS NULL,
               count(*) FILTER (WHERE acl.grantee <> boundary_oid)
        INTO role_oid, actual_grantee, actual_count
        FROM pg_catalog.pg_proc function_row
        LEFT JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl ON true
        WHERE function_row.oid = contract.function_id
        GROUP BY function_row.proowner, function_row.proacl IS NULL;
        IF role_oid <> boundary_oid OR actual_grantee OR actual_count NOT IN (0, 1) THEN
            RAISE EXCEPTION 'H2A002 incompatible fixed-transition ACL: %',
                contract.function_id USING ERRCODE = '42501';
        END IF;
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc function_row
            CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
                function_row.proacl,
                pg_catalog.acldefault('f', function_row.proowner))) acl
            WHERE function_row.oid = contract.function_id
              AND (acl.grantee = 0 OR acl.is_grantable OR
                   (acl.grantee <> boundary_oid AND
                    acl.grantee <> contract.capability::regrole))) THEN
            RAISE EXCEPTION 'H2A003 unsupported fixed-transition grant: %',
                contract.function_id USING ERRCODE = '42501';
        END IF;
        EXECUTE format('GRANT EXECUTE ON FUNCTION %s TO %I',
            contract.function_id, contract.capability);
    END LOOP;

    -- Phase E production handoff executes under the production Phase 5 role.
    -- It may lock scheduler protocol evidence and hydrate production evidence,
    -- but it receives no H1 transition or direct production-table DML.
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
            function_row.proacl,
            pg_catalog.acldefault('f', function_row.proowner))) acl
        WHERE function_row.oid =
              'campaign_operations_scheduler_protocol_evidence_lock_v1()'::regprocedure
          AND (acl.grantee = 0 OR acl.is_grantable OR
               (acl.grantee <> boundary_oid AND
                acl.grantee <> 'campaign_operations_scheduler_protocol_evidence_reader'::regrole AND
                acl.grantee <> 'campaign_operations_production_phase5_transactional'::regrole))) THEN
        RAISE EXCEPTION 'H2A003 unsupported scheduler-lock grant'
            USING ERRCODE = '42501';
    END IF;
    GRANT EXECUTE ON FUNCTION
        campaign_operations_scheduler_protocol_evidence_lock_v1()
        TO campaign_operations_production_phase5_transactional;

    -- Acquisition and handoff hydrate complete immutable H1/H2 evidence under
    -- their switched capability roles.  These are SELECT-only supporting reads.
    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_phase5_transactional']
    LOOP
        EXECUTE format(
            'GRANT SELECT ON '
            'campaign_operations_production_enablement_event, '
            'campaign_operations_production_enablement_audit_reference_event, '
            'campaign_operations_request_production_admission TO %I',
            role_name);
    END LOOP;

    -- The sealed H1 readiness view invokes the sealed role-membership helper.
    -- A view runs with invoker privileges, while H2 application roles must
    -- not receive EXECUTE on that helper.  Expose only this read-only
    -- SECURITY DEFINER adapter through the unchanged reader capability.
    EXECUTE 'CREATE OR REPLACE FUNCTION '
        'public.campaign_operations_production_readiness_snapshot_v1() '
        'RETURNS SETOF public.campaign_operations_production_readiness_v1 '
        'LANGUAGE sql STABLE SECURITY DEFINER '
        'SET search_path = pg_catalog, public AS '
        '$fn$ SELECT readiness.* '
        'FROM public.campaign_operations_production_readiness_v1 readiness; $fn$';
    EXECUTE 'ALTER FUNCTION '
        'public.campaign_operations_production_readiness_snapshot_v1() '
        'OWNER TO campaign_operations_h1_boundary_authority';
    EXECUTE 'REVOKE ALL PRIVILEGES ON FUNCTION '
        'public.campaign_operations_production_readiness_snapshot_v1() '
        'FROM PUBLIC, pqxx, campaign_operations_owner, '
        'campaign_operations_production_enabler, '
        'campaign_operations_production_disabler, '
        'campaign_operations_production_dispatcher, '
        'campaign_operations_production_phase5_transactional, '
        'campaign_operations_production_reader, '
        'campaign_operations_scheduler_protocol_evidence_reader';
    EXECUTE 'GRANT EXECUTE ON FUNCTION '
        'public.campaign_operations_production_readiness_snapshot_v1() '
        'TO campaign_operations_production_reader';

    -- The production Phase 5 capability is distinct from the isolated-test
    -- role created by migration 048. Carry forward the established Phase E
    -- transactional surface explicitly so SET LOCAL ROLE can execute the
    -- common handoff. These are the existing Phase E privileges only; no H1
    -- transition or raw scheduler privilege is added.
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
    TO campaign_operations_production_phase5_transactional;
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
        TO campaign_operations_production_phase5_transactional;
    GRANT INSERT (
        request_binding_id, operational_request_id, binding_identity_canonical,
        experiment_id, control_mode, adoption_authorization_event_id,
        adoption_authorization_identity_canonical,
        adoption_authorization_identity_hash, owner_contract_version,
        owner_identity_canonical, owner_identity_hash)
        ON campaign_operations_downstream_control_owner
        TO campaign_operations_production_phase5_transactional;
    GRANT INSERT (
        reservation_id, reservation_identity_canonical, operational_request_id,
        request_identity_canonical, expected_reservation_version,
        resulting_reservation_version, amount,
        request_binding_set_identity_canonical,
        request_binding_set_identity_hash, commitment_contract_version,
        commitment_identity_canonical, commitment_identity_hash)
        ON campaign_operations_reservation_commitment
        TO campaign_operations_production_phase5_transactional;
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
        TO campaign_operations_production_phase5_transactional;
    GRANT INSERT (
        operational_campaign_id, operational_request_id, dispatch_attempt_id,
        dispatch_attempt_outcome_id, cause_kind, actor_identity, capability,
        prior_version, resulting_version, outcome, replay_disposition,
        diagnostic_code)
        ON campaign_operations_dispatch_audit_reference_event
        TO campaign_operations_production_phase5_transactional;
    GRANT EXECUTE ON FUNCTION
        campaign_operations_future_actions_allowed(bigint),
        lock_campaign_operations_authorization_head(bigint, text),
        lock_campaign_operations_budget_head(bigint),
        lock_campaign_operations_campaign(bigint),
        lock_campaign_operations_reservation(bigint),
        lock_campaign_operations_request(bigint),
        transition_campaign_operations_reservation_committed(bigint, integer),
        transition_campaign_operations_request_bound(bigint, integer, text)
        TO campaign_operations_production_phase5_transactional;
    DO $grant_phase5_sequences$
    DECLARE sequence_name text;
    BEGIN
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
                           'campaign_operations_production_phase5_transactional',
                           sequence_name);
        END LOOP;
    END $grant_phase5_sequences$;
    GRANT INSERT (
        symbol, prediction_horizon, c_next_threshold, core_lr_mult, head_lr_mult,
        target_epochs, checkpoint_interval, train_start, train_end, infer_start,
        infer_end, status, phase, resume_model_id, duplicate_nonce,
        invocation_mode, updated_at)
        ON experiment TO campaign_operations_production_phase5_transactional;
    GRANT UPDATE (status, phase, updated_at)
        ON experiment TO campaign_operations_production_phase5_transactional;
    GRANT INSERT (
        recommendation_conversion_proposal_id,
        recommendation_conversion_review_decision_id, experiment_id,
        execution_contract_version, authorization_decision,
        execution_identity_canonical, execution_identity_hash)
        ON experiment_recommendation_conversion_execution
        TO campaign_operations_production_phase5_transactional;
    GRANT INSERT (
        recommendation_conversion_execution_id,
        recommendation_conversion_proposal_id,
        recommendation_conversion_review_decision_id, experiment_id,
        activation_contract_version, previous_status, previous_phase,
        resulting_status, resulting_phase, activation_identity_canonical,
        activation_identity_hash)
        ON experiment_recommendation_conversion_activation
        TO campaign_operations_production_phase5_transactional;

    -- Exact final ACLs for H2-managed objects.  Existing Phase A-G grants are
    -- intentionally not rewritten; this catches any H2-relevant extra,
    -- PUBLIC, pqxx, or grant-option tuple before migration commit.
    FOR contract IN
        SELECT * FROM (VALUES
          ('campaign_operations_production_enablement_event'::regclass,
           ARRAY['campaign_operations_owner','campaign_operations_production_reader',
                 'campaign_operations_production_dispatcher',
                 'campaign_operations_production_phase5_transactional']::name[]),
          ('campaign_operations_production_enablement_audit_reference_event'::regclass,
           ARRAY['campaign_operations_owner','campaign_operations_production_reader',
                 'campaign_operations_production_dispatcher',
                 'campaign_operations_production_phase5_transactional']::name[]),
          ('campaign_operations_request_production_admission'::regclass,
           ARRAY['campaign_operations_owner','campaign_operations_production_reader',
                 'campaign_operations_production_dispatcher',
                 'campaign_operations_production_phase5_transactional']::name[])) expected(relation_id, allowed_roles)
    LOOP
        SELECT count(DISTINCT acl.grantee) INTO actual_count
        FROM pg_catalog.pg_class relation
        CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
            relation.relacl,
            pg_catalog.acldefault('r', relation.relowner))) acl
        WHERE relation.oid = contract.relation_id;
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class relation
            CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
                relation.relacl,
                pg_catalog.acldefault('r', relation.relowner))) acl
            WHERE relation.oid = contract.relation_id
              AND (acl.grantee = 0 OR acl.is_grantable OR
                   (acl.grantee <> boundary_oid AND
                    NOT EXISTS (SELECT 1 FROM pg_catalog.unnest(
                        contract.allowed_roles) allowed(role_name)
                    WHERE allowed.role_name::text = pg_catalog.pg_get_userbyid(acl.grantee)))) ) THEN
            RAISE EXCEPTION 'H2A003 unsupported H2 evidence-table ACL: %',
                contract.relation_id USING ERRCODE = '42501';
        END IF;
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_class relation
            CROSS JOIN LATERAL pg_catalog.aclexplode(coalesce(
                relation.relacl,
                pg_catalog.acldefault('r', relation.relowner))) acl
            WHERE relation.oid = contract.relation_id
              AND pg_catalog.pg_get_userbyid(acl.grantee) IN (
                  'campaign_operations_production_dispatcher',
                  'campaign_operations_production_phase5_transactional')
              AND (acl.privilege_type <> 'SELECT' OR acl.is_grantable)) THEN
            RAISE EXCEPTION 'H2A003 unsupported H2 evidence-table privilege: %',
                contract.relation_id USING ERRCODE = '42501';
        END IF;
        IF actual_count <> 1 + cardinality(contract.allowed_roles) THEN
            RAISE EXCEPTION 'H2A002 H2 evidence-table ACL cardinality mismatch: %',
                contract.relation_id USING ERRCODE = '42501';
        END IF;
    END LOOP;
END $$;

-- Replay is a no-op at the catalog level: GRANT is idempotent and every
-- preflight above rejects an incompatible partial or extra ACL.
