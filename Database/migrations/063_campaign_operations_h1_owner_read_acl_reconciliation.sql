-- Campaign Operations H1 owner-read ACL reconciliation.
--
-- Migration 055 deliberately kept the former Campaign Operations owner as a
-- non-login, read-only consumer of the sealed H1 relations.  The H1 boundary
-- authority owns the relations and every H1 mutation/trigger function; the
-- former owner only supplies the established Phase 2--5 SECURITY DEFINER
-- read paths.  A deployment drift removed the four H1 SELECT tuples used by
-- those paths, causing deferred completion-boundary validation (and the
-- completion evidence readers that reach dispatch attempts) to fail under
-- campaign_operations_owner.
--
-- This is additive and restores only the explicit ACL tuples already frozen
-- by migration 055 and its H1 manifest.  It does not transfer ownership,
-- grant ordinary roles access, grant any mutation privilege, or alter the H1
-- authority-function manifest.

DO $$
BEGIN
    IF to_regclass('public.campaign_operations_completion_event') IS NULL OR
       to_regclass('public.campaign_operations_completion_audit_reference_event') IS NULL OR
       to_regclass('public.campaign_operations_dispatch_attempt') IS NULL OR
       to_regclass('public.campaign_operations_dispatch_audit_reference_event') IS NULL OR
       NOT EXISTS (
           SELECT 1 FROM public.schema_migrations
           WHERE version = '055'
             AND filename =
                 '055_campaign_operations_production_admission_foundation.sql') THEN
        RAISE EXCEPTION
            'H1 owner-read ACL reconciliation requires migration 055 and its sealed relations'
            USING ERRCODE = '55000';
    END IF;
END $$;

-- The H1 boundary authority remains the sole owner.  These are SELECT-only
-- grants to a NOLOGIN role, matching migration 055's explicit ACL contract.
GRANT SELECT ON
    public.campaign_operations_dispatch_attempt,
    public.campaign_operations_dispatch_audit_reference_event,
    public.campaign_operations_completion_event,
    public.campaign_operations_completion_audit_reference_event
TO campaign_operations_owner;

DO $$
DECLARE relation_name text;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'public.campaign_operations_dispatch_attempt',
        'public.campaign_operations_dispatch_audit_reference_event',
        'public.campaign_operations_completion_event',
        'public.campaign_operations_completion_audit_reference_event']
    LOOP
        IF NOT EXISTS (
               SELECT 1 FROM pg_class
               WHERE oid = relation_name::regclass) THEN
            RAISE EXCEPTION 'unreachable relation audit for %', relation_name;
        END IF;
        IF NOT has_table_privilege(
                'campaign_operations_owner', relation_name, 'SELECT') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'INSERT') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'UPDATE') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'DELETE') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'TRUNCATE') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'REFERENCES') OR
           has_table_privilege(
                'campaign_operations_owner', relation_name, 'TRIGGER') THEN
            RAISE EXCEPTION
                'H1 owner-read ACL is not SELECT-only for %', relation_name
                USING ERRCODE = '42501';
        END IF;
        IF pg_get_userbyid(
               (SELECT relowner FROM pg_class
                WHERE oid = relation_name::regclass)) <>
               'campaign_operations_h1_boundary_authority' THEN
            RAISE EXCEPTION
                'H1 relation ownership changed for %', relation_name
                USING ERRCODE = '42501';
        END IF;
    END LOOP;

    IF has_table_privilege(
            'pqxx', 'public.campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
            'pqxx',
            'public.campaign_operations_completion_audit_reference_event',
            'SELECT') OR
       has_table_privilege(
            'campaign_operations_campaign_creator',
            'public.campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
            'campaign_operations_campaign_creator',
            'public.campaign_operations_completion_audit_reference_event',
            'SELECT') THEN
        RAISE EXCEPTION
            'ordinary Campaign Operations role gained H1 completion SELECT'
            USING ERRCODE = '42501';
    END IF;

    IF (SELECT proowner FROM pg_proc
        WHERE oid =
            'public.enforce_campaign_operations_completion_boundary_consistent()'::regprocedure) <>
           'campaign_operations_owner'::regrole OR
       NOT (SELECT prosecdef FROM pg_proc
            WHERE oid =
                'public.enforce_campaign_operations_completion_boundary_consistent()'::regprocedure) OR
       (SELECT proconfig FROM pg_proc
        WHERE oid =
            'public.enforce_campaign_operations_completion_boundary_consistent()'::regprocedure)
           IS DISTINCT FROM ARRAY['search_path=pg_catalog, public, pg_temp']::text[] THEN
        RAISE EXCEPTION
            'completion boundary consistency function authority contract changed'
            USING ERRCODE = '42501';
    END IF;
END $$;
