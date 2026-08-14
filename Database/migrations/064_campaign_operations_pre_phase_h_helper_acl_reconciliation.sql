-- Campaign Operations pre-Phase-H helper ACL reconciliation.
--
-- H1 sealed the legacy lock helpers under the boundary authority and removed
-- predecessor execution grants from ordinary roles.  The accepted Phase A-G
-- budget and request repositories still invoke the campaign lock directly as
-- the capability role, so restore only that exact compatibility edge.  This
-- migration does not grant a Phase-H role, change ownership, or expose any
-- production transition surface.

DO $$
BEGIN
    IF NOT EXISTS (
           SELECT 1
           FROM public.schema_migrations
           WHERE version = '055'
             AND filename =
                 '055_campaign_operations_production_admission_foundation.sql') THEN
        RAISE EXCEPTION
            'pre-Phase-H helper ACL reconciliation requires migration 055'
            USING ERRCODE = '55000';
    END IF;

    IF NOT EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles
           WHERE rolname IN (
               'campaign_operations_budget_administrator',
               'campaign_operations_request_acceptor')
           GROUP BY 1
           HAVING count(*) = 2) THEN
        RAISE EXCEPTION
            'pre-Phase-H capability roles are incomplete'
            USING ERRCODE = '55000';
    END IF;

    IF NOT EXISTS (
           SELECT 1
           FROM pg_catalog.pg_proc function_row
           JOIN pg_catalog.pg_namespace namespace_row
             ON namespace_row.oid = function_row.pronamespace
           WHERE function_row.oid =
                 'public.lock_campaign_operations_campaign(bigint)'::regprocedure
             AND pg_catalog.pg_get_userbyid(function_row.proowner) =
                 'campaign_operations_h1_boundary_authority'
             AND function_row.prosecdef
             AND function_row.proconfig =
                 ARRAY['search_path=pg_catalog, public']::text[]) THEN
        RAISE EXCEPTION
            'campaign lock helper authority contract is not H1 sealed'
            USING ERRCODE = '42501';
    END IF;
END $$;

-- The accepted repository paths require this one direct helper.  The grant is
-- intentionally explicit and replay-safe.  PUBLIC, pqxx, dispatcher, and
-- Phase-5 predecessor access remain absent and are checked below.
GRANT EXECUTE ON FUNCTION public.lock_campaign_operations_campaign(bigint)
    TO campaign_operations_budget_administrator,
       campaign_operations_request_acceptor;

DO $$
DECLARE
    helper regprocedure;
    capability text;
    helper_owner oid;
BEGIN
    IF NOT has_function_privilege(
           'campaign_operations_budget_administrator',
           'public.lock_campaign_operations_campaign(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
           'campaign_operations_request_acceptor',
           'public.lock_campaign_operations_campaign(bigint)', 'EXECUTE') THEN
        RAISE EXCEPTION
            'pre-Phase-H campaign lock compatibility grant is missing'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
           SELECT 1
           FROM (VALUES
               ('campaign_operations_budget_administrator'::name),
               ('campaign_operations_request_acceptor'::name)) expected(grantee)
           WHERE NOT EXISTS (
               SELECT 1
               FROM pg_catalog.pg_proc function_row
               CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
               JOIN pg_catalog.pg_roles grantee_role
                 ON grantee_role.rolname = expected.grantee
               WHERE function_row.oid =
                   'public.lock_campaign_operations_campaign(bigint)'::regprocedure
                 AND acl.grantee = grantee_role.oid
                 AND acl.privilege_type = 'EXECUTE'
                 AND NOT acl.is_grantable)) THEN
        RAISE EXCEPTION
            'pre-Phase-H campaign lock compatibility ACL is not direct and non-grantable'
            USING ERRCODE = '42501';
    END IF;

    SELECT function_row.proowner
      INTO helper_owner
    FROM pg_catalog.pg_proc function_row
    WHERE function_row.oid =
        'public.lock_campaign_operations_campaign(bigint)'::regprocedure;

    IF EXISTS (
           SELECT 1
           FROM pg_catalog.pg_proc function_row
           CROSS JOIN LATERAL pg_catalog.aclexplode(function_row.proacl) acl
           WHERE function_row.oid =
               'public.lock_campaign_operations_campaign(bigint)'::regprocedure
             AND acl.grantee <> function_row.proowner
             AND acl.is_grantable) THEN
        RAISE EXCEPTION
            'campaign lock compatibility grant option present'
            USING ERRCODE = '42501';
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
                 AND acl.grantee <> helper_owner
                 AND acl.is_grantable) THEN
            RAISE EXCEPTION
                'sealed helper grant option present: %', helper
                USING ERRCODE = '42501';
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
            RAISE EXCEPTION
                'PUBLIC, pqxx, dispatcher, or Phase-5 helper execution present: %',
                helper
                USING ERRCODE = '42501';
        END IF;
    END LOOP;

    IF EXISTS (
           SELECT 1
           FROM pg_catalog.pg_roles login_role
           WHERE login_role.rolname = 'campaign_operations_pre_phase_h_login'
             AND login_role.rolsuper) THEN
        RAISE EXCEPTION
            'pre-Phase-H LOGIN is superuser'
            USING ERRCODE = '42501';
    END IF;

    FOREACH helper IN ARRAY ARRAY[
        'public.lock_campaign_operations_authorization_head(bigint,text)'::regprocedure,
        'public.lock_campaign_operations_budget_head(bigint)'::regprocedure,
        'public.lock_campaign_operations_reservation(bigint)'::regprocedure,
        'public.lock_campaign_operations_request(bigint)'::regprocedure]
    LOOP
        FOREACH capability IN ARRAY ARRAY[
            'campaign_operations_budget_administrator',
            'campaign_operations_request_acceptor']
        LOOP
            IF has_function_privilege(capability, helper, 'EXECUTE') THEN
                RAISE EXCEPTION
                    'unproven pre-Phase-H helper grant: % -> %',
                    capability, helper::text USING ERRCODE = '42501';
            END IF;
        END LOOP;
    END LOOP;

    IF EXISTS (
           SELECT 1 FROM pg_catalog.pg_roles login_role
           WHERE login_role.rolname = 'campaign_operations_pre_phase_h_login'
             AND (pg_has_role(login_role.rolname,
                              'campaign_operations_owner', 'MEMBER') OR
                  EXISTS (
                      SELECT 1
                      FROM pg_catalog.pg_roles production_role
                      WHERE production_role.rolname = ANY (ARRAY[
                          'campaign_operations_production_enabler',
                          'campaign_operations_production_disabler',
                          'campaign_operations_production_dispatcher',
                          'campaign_operations_production_dispatch_service',
                          'campaign_operations_production_phase5_transactional',
                          'campaign_operations_production_reader',
                          'campaign_operations_scheduler_protocol_evidence_reader'])
                        AND pg_has_role(login_role.rolname,
                                        production_role.rolname, 'MEMBER')))) THEN
        RAISE EXCEPTION
            'pre-Phase-H LOGIN crossed the Phase-H authority boundary'
            USING ERRCODE = '42501';
    END IF;

    IF pg_catalog.pg_get_userbyid((SELECT proowner FROM pg_catalog.pg_proc
        WHERE oid =
          'public.lock_campaign_operations_campaign(bigint)'::regprocedure))
           <> 'campaign_operations_h1_boundary_authority' OR
       NOT (SELECT prosecdef FROM pg_catalog.pg_proc
            WHERE oid =
              'public.lock_campaign_operations_campaign(bigint)'::regprocedure) OR
       (SELECT proconfig FROM pg_catalog.pg_proc
        WHERE oid =
          'public.lock_campaign_operations_campaign(bigint)'::regprocedure)
           IS DISTINCT FROM ARRAY['search_path=pg_catalog, public']::text[] THEN
        RAISE EXCEPTION
            'campaign lock helper H1 authority contract changed'
            USING ERRCODE = '42501';
    END IF;
END $$;
