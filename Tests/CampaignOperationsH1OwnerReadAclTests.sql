-- Regression assertions for migration 063.  Run as the deployment/admin
-- verifier, not as an application login.
DO $$
DECLARE relation_name text;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_audit_reference_event',
        'campaign_operations_completion_event',
        'campaign_operations_completion_audit_reference_event']
    LOOP
        IF (SELECT pg_get_userbyid(relowner) FROM pg_class
            WHERE oid = relation_name::regclass) <>
               'campaign_operations_h1_boundary_authority' OR
           NOT has_table_privilege(
               'campaign_operations_owner', relation_name, 'SELECT') OR
           has_table_privilege(
               'campaign_operations_owner', relation_name, 'INSERT') OR
           has_table_privilege(
               'campaign_operations_owner', relation_name, 'UPDATE') OR
           has_table_privilege(
               'campaign_operations_owner', relation_name, 'DELETE') OR
           has_table_privilege(
               'campaign_operations_owner', relation_name, 'TRUNCATE') THEN
            RAISE EXCEPTION 'H1 owner-read ACL regression for %', relation_name;
        END IF;
    END LOOP;

    IF has_table_privilege(
           'campaign_operations_campaign_creator',
           'campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
           'campaign_operations_campaign_creator',
           'campaign_operations_completion_audit_reference_event', 'SELECT') OR
       has_table_privilege(
           'pqxx', 'campaign_operations_completion_event', 'SELECT') OR
       has_table_privilege(
           'pqxx', 'campaign_operations_completion_audit_reference_event',
           'SELECT') THEN
        RAISE EXCEPTION 'ordinary role can read H1 completion history';
    END IF;

    IF (SELECT pg_get_userbyid(proowner) FROM pg_proc
        WHERE oid =
            'enforce_campaign_operations_completion_boundary_consistent()'::regprocedure) <>
           'campaign_operations_owner' OR
       NOT (SELECT prosecdef FROM pg_proc
            WHERE oid =
                'enforce_campaign_operations_completion_boundary_consistent()'::regprocedure) THEN
        RAISE EXCEPTION 'completion boundary function ownership changed';
    END IF;
END $$;
