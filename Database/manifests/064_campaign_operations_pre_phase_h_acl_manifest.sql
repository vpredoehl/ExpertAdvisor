-- Compatibility ACL overlay for migration 064.
-- The 055 H1 manifest remains the immutable base contract.  This read-only
-- overlay adds exactly the two direct Phase A-G capability edges restored by
-- migration 064 and rejects every adjacent helper or authority expansion.

WITH sealed_helper(signature) AS (VALUES
    ('public.lock_campaign_operations_authorization_head(bigint,text)'),
    ('public.lock_campaign_operations_budget_head(bigint)'),
    ('public.lock_campaign_operations_campaign(bigint)'),
    ('public.lock_campaign_operations_reservation(bigint)'),
    ('public.lock_campaign_operations_request(bigint)')),
expected(grantee) AS (VALUES
    ('campaign_operations_budget_administrator'),
    ('campaign_operations_request_acceptor'))
SELECT 'P064001:missing-campaign-lock-execute:' || expected.grantee
FROM expected
WHERE NOT has_function_privilege(
    expected.grantee,
    'public.lock_campaign_operations_campaign(bigint)', 'EXECUTE')
UNION ALL
SELECT 'P064002:sealed-helper-execute:' || forbidden.grantee || ':' ||
       sealed_helper.signature
FROM sealed_helper
CROSS JOIN (VALUES
    ('PUBLIC'),
    ('pqxx'),
    ('campaign_operations_dispatcher'),
    ('campaign_operations_phase5_transactional')) forbidden(grantee)
WHERE (forbidden.grantee = 'PUBLIC' AND EXISTS (
          SELECT 1
          FROM pg_catalog.aclexplode((SELECT proacl FROM pg_catalog.pg_proc
              WHERE oid = sealed_helper.signature::regprocedure)) acl
          WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE'))
   OR (forbidden.grantee <> 'PUBLIC' AND has_function_privilege(
          forbidden.grantee, sealed_helper.signature, 'EXECUTE'))
UNION ALL
SELECT 'P064003:unnecessary-lock-execute:' || capability || ':' || helper
FROM (VALUES
    ('campaign_operations_budget_administrator', 'public.lock_campaign_operations_authorization_head(bigint,text)'),
    ('campaign_operations_budget_administrator', 'public.lock_campaign_operations_budget_head(bigint)'),
    ('campaign_operations_budget_administrator', 'public.lock_campaign_operations_reservation(bigint)'),
    ('campaign_operations_budget_administrator', 'public.lock_campaign_operations_request(bigint)'),
    ('campaign_operations_request_acceptor', 'public.lock_campaign_operations_authorization_head(bigint,text)'),
    ('campaign_operations_request_acceptor', 'public.lock_campaign_operations_budget_head(bigint)'),
    ('campaign_operations_request_acceptor', 'public.lock_campaign_operations_reservation(bigint)'),
    ('campaign_operations_request_acceptor', 'public.lock_campaign_operations_request(bigint)'))
    unnecessary(capability, helper)
WHERE has_function_privilege(capability, helper, 'EXECUTE')
UNION ALL
SELECT 'P064004:missing-direct-campaign-lock-execute:' || expected.grantee
FROM expected
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_proc function_row
    CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
    JOIN pg_catalog.pg_roles grantee_role ON grantee_role.rolname = expected.grantee
    WHERE function_row.oid =
          'public.lock_campaign_operations_campaign(bigint)'::regprocedure
      AND acl.grantee = grantee_role.oid
      AND acl.privilege_type = 'EXECUTE'
      AND NOT acl.is_grantable)
UNION ALL
SELECT 'P064004:grant-option:' || helper.signature
FROM sealed_helper helper
WHERE EXISTS (
    SELECT 1
    FROM pg_catalog.pg_proc function_row
    CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
    WHERE function_row.oid = helper.signature::regprocedure
      AND acl.grantee <> function_row.proowner
      AND acl.is_grantable)
UNION ALL
SELECT 'P064005:campaign-lock-owner-or-definition-mismatch'
WHERE NOT EXISTS (
    SELECT 1
    FROM pg_catalog.pg_proc function_row
    WHERE function_row.oid =
          'public.lock_campaign_operations_campaign(bigint)'::regprocedure
      AND pg_catalog.pg_get_userbyid(function_row.proowner) =
          'campaign_operations_h1_boundary_authority'
      AND function_row.prosecdef
      AND function_row.proconfig =
          ARRAY['search_path=pg_catalog, public']::text[])
UNION ALL
SELECT 'P064006:pre-phase-h-login-production-membership'
WHERE EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles login_role
    JOIN pg_catalog.pg_roles production_role ON production_role.rolname = ANY (ARRAY[
        'campaign_operations_production_enabler',
        'campaign_operations_production_disabler',
        'campaign_operations_production_dispatcher',
        'campaign_operations_production_dispatch_service',
        'campaign_operations_production_phase5_transactional',
        'campaign_operations_production_reader',
        'campaign_operations_scheduler_protocol_evidence_reader'])
    WHERE login_role.rolname = 'campaign_operations_pre_phase_h_login'
           AND (pg_has_role(login_role.rolname, production_role.rolname, 'MEMBER') OR
           pg_has_role(login_role.rolname,
                       'campaign_operations_owner', 'MEMBER')))
UNION ALL
SELECT 'P064007:pre-phase-h-login-superuser'
WHERE EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles login_role
    WHERE login_role.rolname = 'campaign_operations_pre_phase_h_login'
      AND login_role.rolsuper)
UNION ALL
SELECT 'P064008:pre-phase-h-login-owned-public-object'
WHERE EXISTS (
    SELECT 1
    FROM pg_catalog.pg_roles login_role
    WHERE login_role.rolname = 'campaign_operations_pre_phase_h_login'
      AND (EXISTS (SELECT 1 FROM pg_catalog.pg_class relation
                   WHERE relation.relowner = login_role.oid
                     AND relation.relnamespace = 'public'::regnamespace)
        OR EXISTS (SELECT 1 FROM pg_catalog.pg_proc function_row
                   WHERE function_row.proowner = login_role.oid
                     AND function_row.pronamespace = 'public'::regnamespace)
        OR EXISTS (SELECT 1 FROM pg_catalog.pg_type type_row
                   WHERE type_row.typowner = login_role.oid
                     AND type_row.typnamespace = 'public'::regnamespace)
        OR EXISTS (SELECT 1 FROM pg_catalog.pg_namespace namespace_row
                   WHERE namespace_row.nspowner = login_role.oid
                     AND namespace_row.nspname = 'public')))
UNION ALL
SELECT 'P064009:pre-phase-h-login-phase-h-table-mutation:' || table_name
FROM (VALUES
    ('public.campaign_operations_production_enablement_event'::text),
    ('public.campaign_operations_production_enablement_audit_reference_event'),
    ('public.campaign_operations_dispatch_attempt'),
    ('public.campaign_operations_dispatch_audit_reference_event'),
    ('public.campaign_operations_completion_event'),
    ('public.campaign_operations_completion_audit_reference_event'),
    ('public.campaign_operations_production_transition_context')) protected(table_name)
WHERE has_table_privilege(
    'campaign_operations_pre_phase_h_login', table_name,
    'INSERT,UPDATE,DELETE,TRUNCATE')
UNION ALL
SELECT 'P064010:pre-phase-h-login-transition-authority:' || signature
FROM (VALUES
    ('public.record_campaign_operations_production_enable_v1(text,integer,text,text,text,text,text,text,text,text,text)'::text),
    ('public.record_campaign_operations_production_disable_v1(text,bigint,text,integer,text,text)'::text),
    ('public.transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)'),
    ('public.campaign_operations_production_acquire_replay_v2(bigint,integer,text,timestamp with time zone,text,text,text)')) forbidden(signature)
WHERE has_function_privilege(
    'campaign_operations_pre_phase_h_login', signature, 'EXECUTE');
