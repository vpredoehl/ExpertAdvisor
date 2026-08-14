-- Campaign Operations post-064 residual ACL state remediation.
--
-- The immutable 055 H1 contract requires campaign_operations_owner to hold
-- non-grantable UPDATE on these six operational-request columns.  A
-- production-only drift removed those direct column ACLs.  The same state
-- carried twelve direct, non-grantable pqxx privileges that no 045--064
-- contract authorizes.  Restore/revoke only those eighteen catalog tuples;
-- this migration deliberately does not amend the frozen 055 contract or the
-- post-064 composition authority.

DO $$
DECLARE
    missing_predecessors text[];
BEGIN
    SELECT array_agg(expected.version ORDER BY expected.version)
      INTO missing_predecessors
    FROM (VALUES
        ('055', '055_campaign_operations_production_admission_foundation.sql',
         '1b13d3a64336d7cbd55c935396ec42c4c06320105677829f0cf405733e5715fe'),
        ('056', '056_campaign_operations_h2_privilege_deployment_contract.sql',
         '45e8a9524fdacf40dd774aecb70f9ebeaa6fc6d4b8a30ac552b4b6354cde68fe'),
        ('059', '059_campaign_operations_direct_sql_readiness_boundary.sql',
         'e975d295828084c09491d35866295e8f9eddd96d303afc2bf2e77d6b45e80232'),
        ('064', '064_campaign_operations_pre_phase_h_helper_acl_reconciliation.sql',
         'f15291b58a8f74fe9e605ef0fc3170920061c90d830c31b4881b04a094a1c884'))
        AS expected(version, filename, checksum)
    WHERE NOT EXISTS (
        SELECT 1
        FROM public.schema_migrations applied
        WHERE applied.version = expected.version
          AND applied.filename = expected.filename
          AND applied.checksum = expected.checksum);

    IF missing_predecessors IS NOT NULL THEN
        RAISE EXCEPTION
            'post-064 residual ACL remediation requires exact predecessor ledger entries: %',
            array_to_string(missing_predecessors, ',')
            USING ERRCODE = '55000';
    END IF;

    IF to_regrole('pqxx') IS NULL OR
       to_regrole('campaign_operations_owner') IS NULL OR
       to_regrole('campaign_operations_h1_boundary_authority') IS NULL OR
       to_regclass('public.campaign_operations_operational_request') IS NULL OR
       to_regclass('public.campaign_operations_dispatch_attempt') IS NULL OR
       to_regclass('public.campaign_operations_dispatch_audit_reference_event') IS NULL OR
       to_regclass('public.schema_migrations') IS NULL OR
       to_regclass('public.campaign_operations_production_readiness_v1') IS NULL OR
       to_regclass('public.campaign_operations_production_status_v1') IS NULL OR
       to_regclass('public.campaign_operations_dispatch__dispatch_audit_reference_even_seq') IS NULL OR
       to_regclass('public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq') IS NULL OR
       to_regclass('public.campaign_operations_operational_requ_operational_request_id_seq') IS NULL OR
       NOT EXISTS (SELECT 1 FROM pg_catalog.pg_namespace WHERE nspname = 'public') THEN
        RAISE EXCEPTION
            'post-064 residual ACL remediation requires the sealed H1 objects and roles'
            USING ERRCODE = '55000';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        WHERE relation.oid = ANY (ARRAY[
            'public.campaign_operations_operational_request'::regclass,
            'public.campaign_operations_dispatch_attempt'::regclass,
            'public.campaign_operations_dispatch_audit_reference_event'::regclass])
          AND pg_catalog.pg_get_userbyid(relation.relowner) <>
              'campaign_operations_h1_boundary_authority') THEN
        RAISE EXCEPTION
            'post-064 residual ACL remediation requires sealed H1 relation ownership'
            USING ERRCODE = '42501';
    END IF;
END $$;

-- Remove only the twelve proven, unauthorized pqxx tuples.  REVOKE is safe
-- on replay and does not alter any role, owner, or default ACL.
REVOKE USAGE ON SCHEMA public FROM pqxx;
REVOKE SELECT, USAGE ON SEQUENCE
    public.campaign_operations_dispatch__dispatch_audit_reference_even_seq,
    public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq,
    public.campaign_operations_operational_requ_operational_request_id_seq
FROM pqxx;
REVOKE SELECT ON TABLE
    public.campaign_operations_dispatch_attempt,
    public.campaign_operations_dispatch_audit_reference_event,
    public.schema_migrations,
    public.campaign_operations_production_readiness_v1,
    public.campaign_operations_production_status_v1
FROM pqxx;

-- Restore only the six direct 055 owner UPDATE tuples.  No GRANT OPTION is
-- requested, and column-level scope preserves the frozen contract exactly.
GRANT UPDATE (
    request_state,
    state_version,
    lease_token_hash,
    lease_expires_at,
    dispatcher_identity,
    updated_at)
ON TABLE public.campaign_operations_operational_request
TO campaign_operations_owner;

DO $$
BEGIN
    -- The postcondition checks direct catalog tuples, not effective privilege
    -- inheritance, so a membership change cannot make this remediation appear
    -- successful.  Every prohibited pqxx tuple must be absent.
    IF EXISTS (
        SELECT 1
        FROM pg_catalog.pg_namespace namespace_row
        CROSS JOIN LATERAL pg_catalog.aclexplode(namespace_row.nspacl) acl
        WHERE namespace_row.nspname = 'public'
          AND acl.grantee = 'pqxx'::regrole
          AND acl.privilege_type = 'USAGE') OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl
        WHERE relation.oid = ANY (ARRAY[
            'public.campaign_operations_dispatch__dispatch_audit_reference_even_seq'::regclass,
            'public.campaign_operations_dispatch_attempt_dispatch_attempt_id_seq'::regclass,
            'public.campaign_operations_operational_requ_operational_request_id_seq'::regclass])
          AND acl.grantee = 'pqxx'::regrole
          AND acl.privilege_type IN ('SELECT', 'USAGE')) OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_class relation
        CROSS JOIN LATERAL pg_catalog.aclexplode(relation.relacl) acl
        WHERE relation.oid = ANY (ARRAY[
            'public.campaign_operations_dispatch_attempt'::regclass,
            'public.campaign_operations_dispatch_audit_reference_event'::regclass,
            'public.schema_migrations'::regclass,
            'public.campaign_operations_production_readiness_v1'::regclass,
            'public.campaign_operations_production_status_v1'::regclass])
          AND acl.grantee = 'pqxx'::regrole
          AND acl.privilege_type = 'SELECT') THEN
        RAISE EXCEPTION
            'post-064 residual ACL remediation left an unauthorized pqxx tuple'
            USING ERRCODE = '42501';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('dispatcher_identity'),
            ('lease_expires_at'),
            ('lease_token_hash'),
            ('request_state'),
            ('state_version'),
            ('updated_at')) expected(column_name)
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_catalog.pg_attribute attribute_row
            CROSS JOIN LATERAL pg_catalog.aclexplode(attribute_row.attacl) acl
            WHERE attribute_row.attrelid =
                  'public.campaign_operations_operational_request'::regclass
              AND attribute_row.attname = expected.column_name
              AND NOT attribute_row.attisdropped
              AND acl.grantee = 'campaign_operations_owner'::regrole
              AND acl.privilege_type = 'UPDATE'
              AND NOT acl.is_grantable)) OR
       EXISTS (
        SELECT 1
        FROM pg_catalog.pg_attribute attribute_row
        CROSS JOIN LATERAL pg_catalog.aclexplode(attribute_row.attacl) acl
        WHERE attribute_row.attrelid =
              'public.campaign_operations_operational_request'::regclass
          AND attribute_row.attname IN (
              'dispatcher_identity', 'lease_expires_at', 'lease_token_hash',
              'request_state', 'state_version', 'updated_at')
          AND acl.grantee = 'campaign_operations_owner'::regrole
          AND acl.privilege_type = 'UPDATE'
          AND acl.is_grantable) THEN
        RAISE EXCEPTION
            'post-064 residual ACL remediation did not restore exact non-grantable 055 owner UPDATE tuples'
            USING ERRCODE = '42501';
    END IF;
END $$;
