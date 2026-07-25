-- Catalog-level Campaign Operations Phase 3 migration checks.  The C++ owner
-- fixture targets only a disposable database and applies migration 048 twice.
DO $$
DECLARE object_name text;
DECLARE role_name text;
DECLARE owner_name text;
DECLARE sequence_name text;
DECLARE generated_column text;
BEGIN
    FOREACH object_name IN ARRAY ARRAY[
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_attempt_outcome',
        'campaign_operations_request_binding',
        'campaign_operations_downstream_control_owner',
        'campaign_operations_reservation_commitment',
        'campaign_operations_dispatch_audit_reference_event']
    LOOP
        IF to_regclass(object_name) IS NULL THEN
            RAISE EXCEPTION 'missing Phase 3 table %', object_name;
        END IF;
        SELECT pg_get_userbyid(relowner) INTO owner_name
        FROM pg_class WHERE oid = to_regclass(object_name);
        IF owner_name <> 'campaign_operations_owner' THEN
            RAISE EXCEPTION 'unexpected owner for %: %',
                object_name, owner_name;
        END IF;
        IF EXISTS (
               SELECT 1
               FROM pg_class target,
                    LATERAL aclexplode(coalesce(
                        target.relacl,
                        acldefault('r', target.relowner))) privilege
               WHERE target.oid = to_regclass(object_name)
                 AND privilege.grantee = 0
                 AND privilege.privilege_type IS NOT NULL) OR
           has_table_privilege('pqxx', object_name, 'INSERT') OR
           has_table_privilege('pqxx', object_name, 'UPDATE') OR
           has_table_privilege('pqxx', object_name, 'DELETE') OR
           has_table_privilege(
               'campaign_operations_dispatcher', object_name, 'UPDATE') OR
           has_table_privilege(
               'campaign_operations_phase5_transactional',
               object_name, 'UPDATE') OR
           has_table_privilege(
               'campaign_operations_phase5_transactional',
               object_name, 'DELETE') OR
           has_table_privilege(
               'campaign_operations_phase5_transactional',
               object_name, 'TRUNCATE') THEN
            RAISE EXCEPTION 'forbidden Phase 3 privilege on %', object_name;
        END IF;
    END LOOP;

    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_dispatcher',
        'campaign_operations_phase5_transactional']
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_roles WHERE rolname = role_name
              AND NOT rolcanlogin AND NOT rolsuper
              AND NOT rolcreatedb AND NOT rolcreaterole
              AND NOT rolreplication AND NOT rolbypassrls) THEN
            RAISE EXCEPTION 'Phase 3 capability role not hardened: %',
                role_name;
        END IF;
        IF pg_has_role('pqxx', role_name, 'MEMBER') THEN
            RAISE EXCEPTION 'pqxx inherits Phase 3 capability %', role_name;
        END IF;
        IF EXISTS (
            SELECT 1
            FROM pg_auth_members membership
            JOIN pg_roles member_role
              ON member_role.oid = membership.member
            JOIN pg_roles capability_role
              ON capability_role.oid = membership.roleid
            WHERE capability_role.rolname = role_name
              AND member_role.rolcanlogin) THEN
            RAISE EXCEPTION
                'login principal inherits Phase 3 capability %', role_name;
        END IF;
    END LOOP;

    IF has_table_privilege(
           'campaign_operations_dispatcher',
           'campaign_operations_request_binding', 'INSERT') OR
       has_table_privilege(
           'campaign_operations_dispatcher',
           'campaign_operations_dispatch_attempt_outcome', 'INSERT') OR
       has_table_privilege(
           'campaign_operations_phase5_transactional',
           'campaign_operations_dispatch_attempt', 'INSERT') THEN
        RAISE EXCEPTION 'Phase 3 capabilities overlap';
    END IF;

    IF NOT has_column_privilege(
           'campaign_operations_dispatcher',
           'campaign_operations_dispatch_attempt',
           'operational_request_id', 'INSERT') OR
       NOT has_column_privilege(
           'campaign_operations_phase5_transactional',
           'campaign_operations_request_binding',
           'operational_request_id', 'INSERT') OR
       NOT has_column_privilege(
           'campaign_operations_phase5_transactional',
           'campaign_operations_downstream_control_owner',
           'request_binding_id', 'INSERT') OR
       NOT has_column_privilege(
           'campaign_operations_phase5_transactional',
           'campaign_operations_reservation_commitment',
           'reservation_id', 'INSERT') OR
       NOT has_column_privilege(
           'campaign_operations_phase5_transactional',
           'campaign_operations_dispatch_attempt_outcome',
           'dispatch_attempt_id', 'INSERT') THEN
        RAISE EXCEPTION 'required Phase 3 allowlist privilege missing';
    END IF;

    FOR object_name, generated_column IN
        SELECT * FROM (VALUES
            ('campaign_operations_dispatch_attempt',
             'dispatch_attempt_id'),
            ('campaign_operations_dispatch_attempt', 'acquired_at'),
            ('campaign_operations_request_binding', 'request_binding_id'),
            ('campaign_operations_request_binding', 'created_at'),
            ('campaign_operations_downstream_control_owner',
             'downstream_control_owner_id'),
            ('campaign_operations_downstream_control_owner', 'created_at'),
            ('campaign_operations_reservation_commitment',
             'reservation_commitment_id'),
            ('campaign_operations_reservation_commitment', 'committed_at'),
            ('campaign_operations_dispatch_attempt_outcome',
             'dispatch_attempt_outcome_id'),
            ('campaign_operations_dispatch_attempt_outcome', 'created_at'),
            ('campaign_operations_dispatch_audit_reference_event',
             'dispatch_audit_reference_event_id'),
            ('campaign_operations_dispatch_audit_reference_event',
             'created_at'))
            AS generated(object_name, generated_column)
    LOOP
        IF has_column_privilege(
               'campaign_operations_dispatcher', object_name,
               generated_column, 'INSERT') OR
           has_column_privilege(
               'campaign_operations_phase5_transactional', object_name,
               generated_column, 'INSERT') THEN
            RAISE EXCEPTION
                'generated column INSERT permitted on %.%',
                object_name, generated_column;
        END IF;
    END LOOP;

    FOR sequence_name IN
        SELECT pg_get_serial_sequence(table_name, column_name)
        FROM (VALUES
            ('campaign_operations_dispatch_attempt',
             'dispatch_attempt_id'),
            ('campaign_operations_request_binding', 'request_binding_id'),
            ('campaign_operations_downstream_control_owner',
             'downstream_control_owner_id'),
            ('campaign_operations_reservation_commitment',
             'reservation_commitment_id'),
            ('campaign_operations_dispatch_attempt_outcome',
             'dispatch_attempt_outcome_id'),
            ('campaign_operations_dispatch_audit_reference_event',
             'dispatch_audit_reference_event_id'))
             AS sequences(table_name, column_name)
    LOOP
        SELECT pg_get_userbyid(relowner) INTO owner_name
        FROM pg_class WHERE oid = sequence_name::regclass;
        IF owner_name <> 'campaign_operations_owner' OR
           has_sequence_privilege(
               'campaign_operations_dispatcher', sequence_name, 'UPDATE') OR
           has_sequence_privilege(
               'campaign_operations_phase5_transactional',
               sequence_name, 'UPDATE') OR
           has_sequence_privilege(
               'campaign_operations_dispatcher', sequence_name, 'SELECT') OR
           has_sequence_privilege(
               'campaign_operations_phase5_transactional',
               sequence_name, 'SELECT') THEN
            RAISE EXCEPTION 'Phase 3 sequence ACL mismatch on %',
                sequence_name;
        END IF;
    END LOOP;

    IF NOT has_sequence_privilege(
           'campaign_operations_dispatcher',
           pg_get_serial_sequence(
               'campaign_operations_dispatch_attempt',
               'dispatch_attempt_id'), 'USAGE') OR
       NOT has_sequence_privilege(
           'campaign_operations_dispatcher',
           pg_get_serial_sequence(
               'campaign_operations_dispatch_audit_reference_event',
               'dispatch_audit_reference_event_id'), 'USAGE') OR
       EXISTS (
           SELECT 1
           FROM (VALUES
               ('campaign_operations_request_binding',
                'request_binding_id'),
               ('campaign_operations_downstream_control_owner',
                'downstream_control_owner_id'),
               ('campaign_operations_reservation_commitment',
                'reservation_commitment_id'),
               ('campaign_operations_dispatch_attempt_outcome',
                'dispatch_attempt_outcome_id'),
               ('campaign_operations_dispatch_audit_reference_event',
                'dispatch_audit_reference_event_id'))
                AS required(table_name, column_name)
           WHERE NOT has_sequence_privilege(
               'campaign_operations_phase5_transactional',
               pg_get_serial_sequence(table_name, column_name), 'USAGE')) THEN
        RAISE EXCEPTION 'required Phase 3 sequence USAGE missing';
    END IF;

    IF has_table_privilege(
           'campaign_operations_dispatcher', 'experiment', 'INSERT') OR
       has_table_privilege(
           'campaign_operations_dispatcher', 'experiment', 'UPDATE') OR
       has_table_privilege(
           'campaign_operations_phase5_transactional',
           'experiment', 'DELETE') OR
       has_table_privilege(
           'campaign_operations_phase5_transactional',
           'experiment', 'TRUNCATE') THEN
        RAISE EXCEPTION 'unrelated experiment privilege granted';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_request_binding_member_uidx') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_request_binding_ordinal_uidx') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_control_owner_adoption_shape_check') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname =
            'campaign_operations_request_binding_disposition_check') OR
       NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgname = 'campaign_operations_complete_request_trigger'
          AND tgdeferrable AND tginitdeferred) THEN
        RAISE EXCEPTION 'Phase 3 constraint set incomplete';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_proc procedure
        JOIN pg_namespace namespace
          ON namespace.oid = procedure.pronamespace
        WHERE procedure.proname =
            'transition_campaign_operations_request_bound'
          AND procedure.prosecdef
          AND procedure.proconfig @>
              ARRAY['search_path=pg_catalog, ' ||
                    current_schema() || ', pg_temp']) THEN
        RAISE EXCEPTION 'Phase 3 function security configuration missing';
    END IF;

    IF EXISTS (
           SELECT 1
           FROM pg_proc procedure,
                LATERAL aclexplode(coalesce(
                    procedure.proacl,
                    acldefault('f', procedure.proowner))) privilege
           WHERE procedure.oid =
               'transition_campaign_operations_request_bound(bigint,integer,text)'::regprocedure
             AND privilege.grantee = 0
             AND privilege.privilege_type = 'EXECUTE') OR
       has_function_privilege(
           'pqxx',
           'transition_campaign_operations_request_bound(bigint,integer,text)',
           'EXECUTE') OR
       has_function_privilege(
           'campaign_operations_dispatcher',
           'transition_campaign_operations_request_bound(bigint,integer,text)',
           'EXECUTE') OR
       NOT has_function_privilege(
           'campaign_operations_phase5_transactional',
           'transition_campaign_operations_request_bound(bigint,integer,text)',
           'EXECUTE') THEN
        RAISE EXCEPTION 'Phase 3 transition EXECUTE ACL mismatch';
    END IF;

    IF (SELECT count(*)
        FROM pg_trigger
        WHERE tgname IN (
            'campaign_operations_phase5_experiment_bound_trigger',
            'campaign_operations_phase5_execution_bound_trigger',
            'campaign_operations_phase5_activation_bound_trigger')
          AND tgdeferrable AND tginitdeferred) <> 3 THEN
        RAISE EXCEPTION
            'Phase 5 transactional mutation binding guards missing';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_default_acl default_acl
        JOIN pg_roles owner_role
          ON owner_role.oid = default_acl.defaclrole,
        LATERAL aclexplode(default_acl.defaclacl) privilege
        WHERE owner_role.rolname = 'campaign_operations_owner'
          AND default_acl.defaclnamespace = current_schema()::regnamespace
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION
            'Campaign Operations owner default ACL grants PUBLIC';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid =
            'campaign_operations_operational_request'::regclass
          AND pg_get_constraintdef(oid) LIKE
              '%production_dispatch_enabled = false%') THEN
        RAISE EXCEPTION 'production dispatch gate weakened';
    END IF;
END $$;
