DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = 'campaign_operations_pre_phase_h_login'
          AND (NOT rolcanlogin OR rolsuper OR NOT pg_has_role(
              rolname, 'campaign_operations_campaign_creator', 'MEMBER') OR
              NOT pg_has_role(
                  rolname, 'campaign_operations_budget_administrator',
                  'MEMBER') OR
              NOT pg_has_role(
                  rolname, 'campaign_operations_request_acceptor',
                  'MEMBER') OR
              (SELECT count(*)
               FROM (VALUES
                   ('campaign_operations_campaign_creator'),
                   ('campaign_operations_budget_administrator'),
                   ('campaign_operations_request_acceptor'),
                   ('campaign_operations_owner'),
                   ('campaign_operations_authorizer'),
                   ('campaign_operations_auditor'),
                   ('campaign_operations_reader'),
                   ('campaign_operations_controller'),
                   ('campaign_operations_cancellation_coordinator'),
                   ('campaign_operations_reconciler'),
                   ('campaign_operations_recovery'),
                   ('campaign_operations_completion_writer'),
                   ('campaign_operations_dispatcher'),
                   ('campaign_operations_phase5_transactional'),
                   ('campaign_operations_production_enabler'),
                   ('campaign_operations_production_disabler'),
                   ('campaign_operations_production_dispatcher'),
                   ('campaign_operations_production_dispatch_service'),
                   ('campaign_operations_production_phase5_transactional'),
                   ('campaign_operations_production_reader'),
                   ('campaign_operations_scheduler_protocol_evidence_reader'))
                   AS capability(role_name)
               WHERE EXISTS (
                   SELECT 1 FROM pg_roles known_role
                   WHERE known_role.rolname = capability.role_name
                     AND pg_has_role(pg_roles.rolname, capability.role_name,
                                     'MEMBER'))) <> 3 OR
              EXISTS (
                  SELECT 1
                  FROM pg_roles production_role
                  WHERE production_role.rolname = ANY(ARRAY[
                      'campaign_operations_production_enabler',
                      'campaign_operations_production_disabler',
                      'campaign_operations_production_dispatcher',
                      'campaign_operations_production_dispatch_service',
                      'campaign_operations_production_phase5_transactional',
                      'campaign_operations_production_reader',
                      'campaign_operations_scheduler_protocol_evidence_reader'])
                    AND pg_has_role(pg_roles.rolname, production_role.rolname,
                                    'MEMBER')))) THEN
        RAISE EXCEPTION
            'pre-Phase-H LOGIN capability boundary mismatch';
    END IF;

    IF to_regprocedure(
            'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)')
           IS NOT NULL AND
       has_function_privilege(
            'pqxx',
            'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
            'EXECUTE') THEN
        RAISE EXCEPTION
            'generic pre-Phase-H principal has Phase-H production capability';
    END IF;
    IF EXISTS (
           SELECT 1
           FROM pg_roles production_role
           WHERE production_role.rolname = ANY(ARRAY[
               'campaign_operations_production_enabler',
               'campaign_operations_production_disabler',
               'campaign_operations_production_dispatcher',
               'campaign_operations_production_dispatch_service',
               'campaign_operations_production_phase5_transactional',
               'campaign_operations_production_reader',
               'campaign_operations_scheduler_protocol_evidence_reader'])
             AND pg_has_role('pqxx', production_role.rolname, 'MEMBER')) THEN
        RAISE EXCEPTION
            'generic pre-Phase-H principal has Phase-H production capability';
    END IF;
END $$;

SELECT to_regclass('public.campaign_operations_operational_request') IS NOT NULL
    AS has_campaign_operations_schema \gset
\if :has_campaign_operations_schema
DO $$
BEGIN
    IF NOT has_table_privilege(
            'campaign_operations_owner',
            'campaign_operations_operational_request', 'SELECT') THEN
        RAISE EXCEPTION
            'Phase 2 status-view owner lacks operational-request SELECT';
    END IF;

    IF has_table_privilege(
            'pqxx', 'campaign_operations_budget_status_v1', 'SELECT') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_request_status_v1', 'SELECT') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_operational_request', 'SELECT') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_operational_request', 'INSERT') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_operational_request', 'UPDATE') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_operational_request', 'DELETE') THEN
        RAISE EXCEPTION
            'generic pre-Phase-H principal privilege boundary mismatch';
    END IF;

    IF has_table_privilege(
            'pqxx', 'campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
            'pqxx',
            'campaign_operations_completion_audit_reference_event', 'SELECT') OR
       has_table_privilege(
            'campaign_operations_pre_phase_h_login',
            'campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
            'campaign_operations_pre_phase_h_login',
            'campaign_operations_completion_audit_reference_event', 'SELECT') THEN
        RAISE EXCEPTION
            'ordinary pre-Phase-H login gained H1 completion SELECT';
    END IF;
END $$;
\endif

-- H1 sealed the lock helpers after the original 047 grants. Migration 064
-- restores only the one direct helper used by the accepted budget/request
-- repository paths.
DO $$
DECLARE helper regprocedure;
DECLARE capability text;
BEGIN
    IF NOT has_function_privilege(
           'campaign_operations_budget_administrator',
           'public.lock_campaign_operations_campaign(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
           'campaign_operations_request_acceptor',
           'public.lock_campaign_operations_campaign(bigint)', 'EXECUTE') THEN
        RAISE EXCEPTION 'pre-Phase-H campaign lock compatibility grant missing';
    END IF;
    FOREACH helper IN ARRAY ARRAY[
        'public.lock_campaign_operations_authorization_head(bigint,text)'::regprocedure,
        'public.lock_campaign_operations_budget_head(bigint)'::regprocedure,
        'public.lock_campaign_operations_campaign(bigint)'::regprocedure,
        'public.lock_campaign_operations_reservation(bigint)'::regprocedure,
        'public.lock_campaign_operations_request(bigint)'::regprocedure]
    LOOP
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.pg_proc function_row
            CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
            WHERE function_row.oid = helper
              AND acl.grantee <> function_row.proowner
              AND acl.is_grantable) THEN
            RAISE EXCEPTION 'sealed helper grant option present: %', helper::text;
        END IF;
        IF EXISTS (
            SELECT 1
            FROM pg_catalog.aclexplode((SELECT proacl FROM pg_catalog.pg_proc
                WHERE oid = helper)) acl
            WHERE acl.grantee = 0 AND acl.privilege_type = 'EXECUTE') OR
           has_function_privilege('pqxx', helper, 'EXECUTE') OR
           has_function_privilege(
               'campaign_operations_dispatcher', helper, 'EXECUTE') OR
           has_function_privilege(
               'campaign_operations_phase5_transactional', helper, 'EXECUTE') THEN
            RAISE EXCEPTION 'PUBLIC, pqxx, dispatcher, or Phase-5 helper execution present: %',
                helper::text;
        END IF;
        FOREACH capability IN ARRAY ARRAY[
            'campaign_operations_budget_administrator',
            'campaign_operations_request_acceptor']
        LOOP
            IF helper <> 'public.lock_campaign_operations_campaign(bigint)'::regprocedure AND
               has_function_privilege(capability, helper, 'EXECUTE') THEN
                RAISE EXCEPTION 'unnecessary pre-Phase-H helper grant: % -> %',
                    capability, helper::text;
            END IF;
        END LOOP;
    END LOOP;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_catalog.pg_proc function_row
        WHERE function_row.oid =
              'public.lock_campaign_operations_campaign(bigint)'::regprocedure
          AND pg_catalog.pg_get_userbyid(function_row.proowner) =
              'campaign_operations_h1_boundary_authority'
          AND function_row.prosecdef
          AND function_row.proconfig =
              ARRAY['search_path=pg_catalog, public']::text[]) THEN
        RAISE EXCEPTION 'campaign lock H1 owner/security/search_path contract mismatch';
    END IF;
    IF has_function_privilege(
           'campaign_operations_pre_phase_h_login',
           'public.lock_campaign_operations_authorization_head(bigint,text)',
           'EXECUTE') OR
       has_function_privilege(
           'campaign_operations_pre_phase_h_login',
           'public.lock_campaign_operations_budget_head(bigint)', 'EXECUTE') OR
       has_function_privilege(
           'campaign_operations_pre_phase_h_login',
           'public.lock_campaign_operations_reservation(bigint)', 'EXECUTE') OR
       has_function_privilege(
           'campaign_operations_pre_phase_h_login',
           'public.lock_campaign_operations_request(bigint)', 'EXECUTE') THEN
        RAISE EXCEPTION 'pre-Phase-H LOGIN gained an unproven sealed helper';
    END IF;
END $$;

-- These calls exercise the sealed helper through both accepted capability
-- roles. The fixture may be empty in catalog-only runs, in which case the
-- SELECT is intentionally a no-op; the real repository-path tests cover the
-- mutation with a populated campaign.
SELECT to_regclass('public.campaign_operations_campaign') IS NOT NULL
    AS has_campaign_schema \gset
\if :has_campaign_schema
SET ROLE campaign_operations_budget_administrator;
SELECT lock_campaign_operations_campaign(operational_campaign_id)
FROM campaign_operations_campaign
ORDER BY operational_campaign_id
LIMIT 1;
RESET ROLE;
SET ROLE campaign_operations_request_acceptor;
SELECT lock_campaign_operations_campaign(operational_campaign_id)
FROM campaign_operations_campaign
ORDER BY operational_campaign_id
LIMIT 1;
RESET ROLE;
\endif

-- These are executable privilege regressions, not catalog-only assertions.
-- The test runner is an administrative role against a disposable database.
SELECT to_regclass('public.campaign_operations_budget_status_v1') IS NOT NULL
    AS has_phase2_status_views \gset
\if :has_phase2_status_views
SET ROLE campaign_operations_reader;
SELECT count(*) FROM campaign_operations_budget_status_v1;
SELECT count(*) FROM campaign_operations_request_status_v1;
RESET ROLE;
SET ROLE campaign_operations_budget_administrator;
SELECT count(*) FROM campaign_operations_budget_status_v1;
RESET ROLE;

SET ROLE campaign_operations_request_acceptor;
SELECT count(*) FROM campaign_operations_request_status_v1;
RESET ROLE;
\endif

DO $$
BEGIN
    IF EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname = ANY(ARRAY[
            'campaign_operations_production_enabler',
            'campaign_operations_production_disabler',
            'campaign_operations_production_dispatcher',
            'campaign_operations_production_dispatch_service',
            'campaign_operations_production_phase5_transactional',
            'campaign_operations_production_reader',
            'campaign_operations_scheduler_protocol_evidence_reader'])
          AND rolcanlogin) THEN
        RAISE EXCEPTION 'Phase-H capability role unexpectedly has LOGIN';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_roles manager
        WHERE manager.rolname = 'campaign_operations_manager_login'
          AND (NOT manager.rolcanlogin OR NOT pg_has_role(
              manager.rolname,
              'campaign_operations_production_dispatcher', 'MEMBER'))) THEN
        RAISE EXCEPTION 'Phase-H production manager LOGIN separation mismatch';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_roles manager
        WHERE manager.rolname = 'campaign_operations_manager_login'
          AND (pg_has_role(
                   manager.rolname,
                   'campaign_operations_budget_administrator', 'MEMBER') OR
               pg_has_role(
                   manager.rolname,
                   'campaign_operations_request_acceptor', 'MEMBER'))) THEN
        RAISE EXCEPTION
            'Phase-H production manager LOGIN acquired Phase 2 capability';
    END IF;
END $$;
