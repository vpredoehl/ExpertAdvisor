DO $$
DECLARE
    object_name text;
    owner_name text;
    function_configuration text[];
    mismatch_detail text;
BEGIN
    FOREACH object_name IN ARRAY ARRAY[
        'campaign_operations_budget_ledger_entry',
        'campaign_operations_reservation',
        'campaign_operations_operational_request',
        'campaign_operations_reservation_event',
        'campaign_operations_budget_status_v1',
        'campaign_operations_request_status_v1']
    LOOP
        IF to_regclass(object_name) IS NULL THEN
            RAISE EXCEPTION 'Campaign Operations Phase 2 object missing: %',
                object_name;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM pg_roles
        WHERE rolname IN (
            'campaign_operations_budget_administrator',
            'campaign_operations_request_acceptor')
          AND (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR
               rolreplication OR rolbypassrls)) THEN
        RAISE EXCEPTION
            'Campaign Operations Phase 2 capability role is not hardened';
    END IF;

    IF pg_has_role(
            'pqxx', 'campaign_operations_budget_administrator', 'MEMBER') OR
       pg_has_role(
            'pqxx', 'campaign_operations_request_acceptor', 'MEMBER') THEN
        RAISE EXCEPTION
            'runtime login unexpectedly holds Campaign Operations capability';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_auth_members membership
        WHERE membership.member IN (
            'campaign_operations_budget_administrator'::regrole,
            'campaign_operations_request_acceptor'::regrole)) THEN
        RAISE EXCEPTION
            'Phase 2 capability role unexpectedly inherits another role';
    END IF;

    FOREACH object_name IN ARRAY ARRAY[
        'campaign_operations_budget_ledger_entry',
        'campaign_operations_reservation',
        'campaign_operations_operational_request',
        'campaign_operations_reservation_event']
    LOOP
        SELECT tableowner INTO owner_name
        FROM pg_tables
        WHERE schemaname = current_schema()
          AND tablename = object_name;
        IF owner_name <> 'campaign_operations_owner' THEN
            RAISE EXCEPTION 'Phase 2 owner mismatch for %: %',
                object_name, owner_name;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE oid IN (
            'enforce_campaign_operations_budget_entry()'::regprocedure,
            'enforce_campaign_operations_reservation()'::regprocedure,
            'enforce_campaign_operations_request()'::regprocedure,
            'enforce_campaign_operations_reservation_event()'::regprocedure,
            'enforce_campaign_operations_request_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_reservation_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_budget_audit_complete()'::
                regprocedure,
            'lock_campaign_operations_campaign(bigint)'::regprocedure)
          AND NOT prosecdef) THEN
        RAISE EXCEPTION
            'Phase 2 invariant or lock function is not owner-executed';
    END IF;

    FOREACH object_name IN ARRAY ARRAY[
        'enforce_campaign_operations_budget_entry()',
        'enforce_campaign_operations_reservation()',
        'enforce_campaign_operations_request()',
        'enforce_campaign_operations_reservation_event()',
        'enforce_campaign_operations_request_acquisition_complete()',
        'enforce_campaign_operations_reservation_acquisition_complete()',
        'enforce_campaign_operations_budget_audit_complete()',
        'lock_campaign_operations_campaign(bigint)']
    LOOP
        SELECT proconfig INTO function_configuration
        FROM pg_proc
        WHERE oid = object_name::regprocedure;
        IF function_configuration IS NULL OR NOT EXISTS (
            SELECT 1
            FROM unnest(function_configuration) setting
            WHERE setting =
                'search_path=pg_catalog, ' || current_schema() || ', pg_temp')
        THEN
            RAISE EXCEPTION 'Phase 2 function search path not pinned: %',
                object_name;
        END IF;
    END LOOP;

    IF EXISTS (
        SELECT 1
        FROM pg_proc function_record,
             aclexplode(coalesce(function_record.proacl,
                 acldefault('f', function_record.proowner))) access
        WHERE function_record.oid =
                'lock_campaign_operations_campaign(bigint)'::regprocedure
          AND access.grantee = 0
          AND access.privilege_type = 'EXECUTE') OR
       has_function_privilege(
            'pqxx', 'lock_campaign_operations_campaign(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_budget_administrator',
            'lock_campaign_operations_campaign(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_request_acceptor',
            'lock_campaign_operations_campaign(bigint)', 'EXECUTE') THEN
        RAISE EXCEPTION
            'Campaign Operations campaign-lock capability grant mismatch';
    END IF;

    IF has_table_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_operational_request', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_operational_request', 'DELETE') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_reservation', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_operational_request', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_operational_request', 'DELETE') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_operational_request', 'SELECT') OR
       has_table_privilege(
            'pqxx', 'campaign_operations_operational_request', 'SELECT') THEN
        RAISE EXCEPTION
            'Campaign Operations Phase 2 table privilege is overbroad';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_budget_administrator'),
            ('campaign_operations_request_acceptor')) role_name(name)
        CROSS JOIN (VALUES
            ('campaign_operations_budget_ledger_entry'),
            ('campaign_operations_reservation'),
            ('campaign_operations_operational_request'),
            ('campaign_operations_reservation_event'),
            ('campaign_operations_audit_reference_event')) relation_name(name)
        CROSS JOIN (VALUES ('UPDATE'),('DELETE'),('TRUNCATE'))
            privilege_name(name)
        WHERE has_table_privilege(
            role_name.name, relation_name.name, privilege_name.name)) THEN
        RAISE EXCEPTION
            'Phase 2 capability has mutable or destructive table privilege';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_budget_administrator'),
            ('campaign_operations_request_acceptor')) role_name(name)
        CROSS JOIN pg_class relation
        JOIN pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        CROSS JOIN (VALUES ('UPDATE'),('DELETE'),('TRUNCATE'))
            privilege_name(name)
        WHERE namespace.nspname = current_schema()
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
          AND has_table_privilege(
                role_name.name, relation.oid, privilege_name.name)) THEN
        RAISE EXCEPTION
            'Phase 2 capability can mutate a non-Phase-2 relation';
    END IF;

    IF has_table_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_reservation', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_operational_request', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_reservation_event', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_budget_ledger_entry', 'INSERT') THEN
        RAISE EXCEPTION
            'Phase 2 capability can cross the budget/request boundary';
    END IF;

    IF NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_budget_status_v1', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_request_status_v1', 'SELECT') THEN
        RAISE EXCEPTION
            'Campaign Operations reader cannot access approved status views';
    END IF;
    WITH expected(role_name, relation_name) AS (
        VALUES
        ('campaign_operations_budget_administrator',
         'campaign_operations_campaign'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_budget_ledger_entry'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_reservation'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_operational_request'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_reservation_event'),
        ('campaign_operations_budget_administrator',
         'experiment_recommendation_campaign_materialization'),
        ('campaign_operations_budget_administrator',
         'experiment_recommendation_campaign_materialization_member'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_budget_status_v1'),
        ('campaign_operations_budget_administrator',
         'campaign_operations_request_status_v1'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_campaign'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_governance_provenance_event'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_authorization_event'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_budget_ledger_entry'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_reservation'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_operational_request'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_reservation_event'),
        ('campaign_operations_request_acceptor',
         'experiment_recommendation_campaign_materialization'),
        ('campaign_operations_request_acceptor',
         'experiment_recommendation_campaign_materialization_member'),
        ('campaign_operations_request_acceptor',
         'experiment_recommendation_campaign_follow_up_proposal'),
        ('campaign_operations_request_acceptor',
         to_regclass(
             'experiment_recommendation_campaign_follow_up_proposal_review_event'
         )::regclass::text),
        ('campaign_operations_request_acceptor',
         'experiment_recommendation_campaign_follow_up_ratification_event'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_budget_status_v1'),
        ('campaign_operations_request_acceptor',
         'campaign_operations_request_status_v1')),
    actual(role_name, relation_name) AS (
        SELECT role_name.name, relation.relname
        FROM (VALUES
            ('campaign_operations_budget_administrator'),
            ('campaign_operations_request_acceptor')) role_name(name)
        CROSS JOIN pg_class relation
        JOIN pg_namespace namespace
          ON namespace.oid = relation.relnamespace
        WHERE namespace.nspname = current_schema()
          AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
          AND has_table_privilege(
                role_name.name, relation.oid, 'SELECT')),
    mismatch AS (
        (SELECT 'missing' AS kind, * FROM expected
         EXCEPT SELECT 'missing', * FROM actual)
        UNION ALL
        (SELECT 'extra' AS kind, * FROM actual
         EXCEPT SELECT 'extra', * FROM expected))
    SELECT string_agg(
        kind || ':' || role_name || ':' || relation_name, ','
        ORDER BY kind, role_name, relation_name)
    INTO mismatch_detail
    FROM mismatch;
    IF mismatch_detail IS NOT NULL THEN
        RAISE EXCEPTION
            'Phase 2 table SELECT privilege allowlist mismatch: %',
            mismatch_detail;
    END IF;

    IF NOT has_column_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_budget_ledger_entry',
            'operational_campaign_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_budget_ledger_entry',
            'budget_ledger_entry_id', 'INSERT') OR
       NOT has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_operational_request',
            'operational_campaign_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_operational_request',
            'operational_request_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_operational_request',
            'production_dispatch_enabled', 'INSERT') THEN
        RAISE EXCEPTION
            'Campaign Operations Phase 2 column privilege mismatch';
    END IF;
    IF has_column_privilege(
            'campaign_operations_budget_administrator',
            'campaign_operations_audit_reference_event',
            'audit_reference_event_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_reservation',
            'reservation_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_reservation_event',
            'reservation_event_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_request_acceptor',
            'campaign_operations_audit_reference_event',
            'audit_reference_event_id', 'INSERT') THEN
        RAISE EXCEPTION
            'Phase 2 capability can insert a generated identifier';
    END IF;
    IF EXISTS (
        WITH expected(role_name, relation_name, column_names) AS (
            VALUES
            ('campaign_operations_budget_administrator',
             'campaign_operations_budget_ledger_entry',
             'operational_campaign_id,campaign_identity_canonical,previous_entry_id,previous_entry_identity_canonical,previous_entry_identity_hash,ledger_version,entry_kind,ledger_status,budget_unit,delta,prior_total,resulting_total,administrator_identity,reason,budget_contract_version,budget_identity_canonical,budget_identity_hash'),
            ('campaign_operations_budget_administrator',
             'campaign_operations_audit_reference_event',
             'operational_campaign_id,governance_provenance_event_id,authorization_event_id,budget_ledger_entry_id,reservation_id,reservation_event_id,operational_request_id,cause_kind,actor_identity,capability,reason,prior_version,resulting_version,outcome,replay_disposition'),
            ('campaign_operations_request_acceptor',
             'campaign_operations_reservation',
             'operational_campaign_id,campaign_identity_canonical,logical_operation_contract_version,logical_operation_canonical,logical_operation_hash,authorization_event_id,authorization_identity_canonical,authorization_identity_hash,budget_ledger_entry_id,budget_ledger_version,budget_identity_canonical,budget_identity_hash,action_kind,action_contract_version,recommendation_campaign_materialization_id,materialization_contract_version,materialization_identity_canonical,materialization_identity_hash,scope_kind,scope_contract_version,materialization_member_count,amount,budget_unit,expires_at,reservation_contract_version,reservation_identity_canonical,reservation_identity_hash,reservation_state,state_version'),
            ('campaign_operations_request_acceptor',
             'campaign_operations_operational_request',
             'operational_campaign_id,campaign_identity_canonical,logical_operation_contract_version,logical_operation_canonical,logical_operation_hash,authorization_event_id,authorization_identity_canonical,authorization_identity_hash,reservation_id,reservation_identity_canonical,reservation_identity_hash,action_kind,action_contract_version,recommendation_campaign_materialization_id,materialization_contract_version,materialization_identity_canonical,materialization_identity_hash,ordered_scope_digest,materialization_member_count,accepting_actor_identity,reason,prerequisite_policy,provenance_identity_canonical,provenance_identity_hash,request_contract_version,request_identity_canonical,request_identity_hash,request_state,state_version'),
            ('campaign_operations_request_acceptor',
             'campaign_operations_reservation_event',
             'reservation_id,reservation_identity_canonical,transition_kind,expected_state,resulting_state,expected_version,resulting_version,operational_request_id,request_identity_canonical,amount,reservation_event_contract_version,reservation_event_identity_canonical,reservation_event_identity_hash'),
            ('campaign_operations_request_acceptor',
             'campaign_operations_audit_reference_event',
             'operational_campaign_id,governance_provenance_event_id,authorization_event_id,budget_ledger_entry_id,reservation_id,reservation_event_id,operational_request_id,cause_kind,actor_identity,capability,reason,prior_version,resulting_version,outcome,replay_disposition')),
        expected_columns AS (
            SELECT role_name, relation_name,
                   unnest(string_to_array(column_names, ',')) AS column_name
            FROM expected),
        actual_columns AS (
            SELECT role_name.name, relation.relname, attribute.attname
            FROM (VALUES
                ('campaign_operations_budget_administrator'),
                ('campaign_operations_request_acceptor')) role_name(name)
            CROSS JOIN pg_class relation
            JOIN pg_namespace namespace
              ON namespace.oid = relation.relnamespace
            JOIN pg_attribute attribute
              ON attribute.attrelid = relation.oid
             AND attribute.attnum > 0
             AND NOT attribute.attisdropped
            WHERE namespace.nspname = current_schema()
              AND relation.relkind IN ('r', 'p', 'v', 'm', 'f')
              AND has_column_privilege(
                    role_name.name, relation.oid, attribute.attnum, 'INSERT')),
        mismatch AS (
            (SELECT * FROM expected_columns
             EXCEPT SELECT * FROM actual_columns)
            UNION ALL
            (SELECT * FROM actual_columns
             EXCEPT SELECT * FROM expected_columns))
        SELECT 1 FROM mismatch) THEN
        RAISE EXCEPTION
            'Phase 2 column INSERT privilege allowlist mismatch';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_budget_administrator'),
            ('campaign_operations_request_acceptor')) role_name(name)
        CROSS JOIN LATERAL (
            SELECT sequence_record.oid::regclass AS sequence_name
            FROM pg_class sequence_record
            JOIN pg_namespace namespace_record
              ON namespace_record.oid = sequence_record.relnamespace
            WHERE namespace_record.nspname = current_schema()
              AND sequence_record.relkind = 'S'
              AND sequence_record.relname LIKE
                    'campaign_operations_%_id_seq') sequences
        WHERE has_sequence_privilege(
            role_name.name, sequences.sequence_name, 'UPDATE')) THEN
        RAISE EXCEPTION
            'Phase 2 capability has sequence UPDATE privilege';
    END IF;

    IF NOT has_sequence_privilege(
            'campaign_operations_budget_administrator',
            pg_get_serial_sequence(
                'campaign_operations_budget_ledger_entry',
                'budget_ledger_entry_id'), 'USAGE') OR
       has_sequence_privilege(
            'campaign_operations_budget_administrator',
            pg_get_serial_sequence(
                'campaign_operations_reservation', 'reservation_id'),
            'USAGE') OR
       NOT has_sequence_privilege(
            'campaign_operations_request_acceptor',
            pg_get_serial_sequence(
                'campaign_operations_reservation', 'reservation_id'),
            'USAGE') OR
       has_sequence_privilege(
            'campaign_operations_request_acceptor',
            pg_get_serial_sequence(
                'campaign_operations_budget_ledger_entry',
                'budget_ledger_entry_id'), 'USAGE') THEN
        RAISE EXCEPTION
            'Phase 2 sequence USAGE allowlist mismatch';
    END IF;
    IF EXISTS (
        WITH phase_sequences(sequence_name) AS (
            VALUES
            (pg_get_serial_sequence(
                'campaign_operations_budget_ledger_entry',
                'budget_ledger_entry_id')),
            (pg_get_serial_sequence(
                'campaign_operations_reservation', 'reservation_id')),
            (pg_get_serial_sequence(
                'campaign_operations_operational_request',
                'operational_request_id')),
            (pg_get_serial_sequence(
                'campaign_operations_reservation_event',
                'reservation_event_id')),
            (pg_get_serial_sequence(
                'campaign_operations_audit_reference_event',
                'audit_reference_event_id'))),
        expected(role_name, sequence_name) AS (
            VALUES
            ('campaign_operations_budget_administrator',
             pg_get_serial_sequence(
                'campaign_operations_budget_ledger_entry',
                'budget_ledger_entry_id')),
            ('campaign_operations_budget_administrator',
             pg_get_serial_sequence(
                'campaign_operations_audit_reference_event',
                'audit_reference_event_id')),
            ('campaign_operations_request_acceptor',
             pg_get_serial_sequence(
                'campaign_operations_reservation', 'reservation_id')),
            ('campaign_operations_request_acceptor',
             pg_get_serial_sequence(
                'campaign_operations_operational_request',
                'operational_request_id')),
            ('campaign_operations_request_acceptor',
             pg_get_serial_sequence(
                'campaign_operations_reservation_event',
                'reservation_event_id')),
            ('campaign_operations_request_acceptor',
             pg_get_serial_sequence(
                'campaign_operations_audit_reference_event',
                'audit_reference_event_id'))),
        actual AS (
            SELECT role_name.name, phase_sequences.sequence_name
            FROM (VALUES
                ('campaign_operations_budget_administrator'),
                ('campaign_operations_request_acceptor')) role_name(name)
            CROSS JOIN phase_sequences
            WHERE has_sequence_privilege(
                role_name.name, phase_sequences.sequence_name, 'USAGE')),
        mismatch AS (
            (SELECT * FROM expected EXCEPT SELECT * FROM actual)
            UNION ALL
            (SELECT * FROM actual EXCEPT SELECT * FROM expected))
        SELECT 1 FROM mismatch) THEN
        RAISE EXCEPTION
            'Phase 2 exact sequence USAGE allowlist mismatch';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_proc function_record
        CROSS JOIN (VALUES
            ('campaign_operations_budget_administrator'),
            ('campaign_operations_request_acceptor'),
            ('pqxx')) role_name(name)
        WHERE function_record.oid IN (
            'enforce_campaign_operations_budget_entry()'::regprocedure,
            'enforce_campaign_operations_reservation()'::regprocedure,
            'enforce_campaign_operations_request()'::regprocedure,
            'enforce_campaign_operations_reservation_event()'::regprocedure,
            'enforce_campaign_operations_request_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_reservation_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_budget_audit_complete()'::
                regprocedure)
          AND has_function_privilege(
                role_name.name, function_record.oid, 'EXECUTE')) THEN
        RAISE EXCEPTION
            'Phase 2 invariant function EXECUTE allowlist is overbroad';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_proc function_record,
             aclexplode(coalesce(function_record.proacl,
                 acldefault('f', function_record.proowner))) access
        WHERE function_record.oid IN (
            'enforce_campaign_operations_budget_entry()'::regprocedure,
            'enforce_campaign_operations_reservation()'::regprocedure,
            'enforce_campaign_operations_request()'::regprocedure,
            'enforce_campaign_operations_reservation_event()'::regprocedure,
            'enforce_campaign_operations_request_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_reservation_acquisition_complete()'::
                regprocedure,
            'enforce_campaign_operations_budget_audit_complete()'::
                regprocedure)
          AND access.grantee = 0
          AND access.privilege_type = 'EXECUTE') THEN
        RAISE EXCEPTION
            'PUBLIC can execute a Phase 2 invariant function';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.table_privileges privilege
        WHERE privilege.table_schema = current_schema()
          AND privilege.table_name IN (
                'campaign_operations_budget_ledger_entry',
                'campaign_operations_reservation',
                'campaign_operations_operational_request',
                'campaign_operations_reservation_event',
                'campaign_operations_audit_reference_event',
                'campaign_operations_budget_status_v1',
                'campaign_operations_request_status_v1')
          AND privilege.grantee IN ('PUBLIC', 'pqxx')
          AND privilege.privilege_type IN (
                'INSERT', 'UPDATE', 'DELETE', 'TRUNCATE')) THEN
        RAISE EXCEPTION
            'PUBLIC or pqxx has Phase 2 mutation privilege';
    END IF;

    IF has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_budget_status_v1', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_budget_status_v1', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_request_status_v1', 'DELETE') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'experiment_recommendation_campaign_materialization', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_request_acceptor',
            'experiment_recommendation_campaign_materialization', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_budget_administrator',
            'experiment_recommendation_campaign_materialization', 'DELETE')
    THEN
        RAISE EXCEPTION
            'Phase 2 view or upstream experiment privilege is mutable';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname IN (
            'campaign_operations_reservation_initial_state_check',
            'campaign_operations_request_initial_state_check')) THEN
        RAISE EXCEPTION
            'insert-only initial state constraint blocks future lifecycle';
    END IF;

    IF position(
            'NEW.reservation_state <> ''held''' IN pg_get_functiondef(
                'enforce_campaign_operations_reservation()'::regprocedure)) =
            0 OR
       position(
            '1179402835030003' IN pg_get_functiondef(
                'enforce_campaign_operations_authorization_chain()'::
                    regprocedure)) = 0 OR
       position(
            'NEW.request_state <> ''ready''' IN pg_get_functiondef(
                'enforce_campaign_operations_request()'::regprocedure)) = 0 OR
       position(
            'NEW.production_dispatch_enabled' IN pg_get_functiondef(
                'enforce_campaign_operations_request()'::regprocedure)) = 0
    THEN
        RAISE EXCEPTION
            'Phase 2 authorization lock or held/ready/undispatched guard missing';
    END IF;

    IF position(
            'authz.prerequisite_policy = request.prerequisite_policy'
            IN pg_get_viewdef(
                'campaign_operations_request_status_v1'::regclass, true)) =
            0 OR
       position(
            'budget.budget_identity_canonical = reservation.budget_identity_canonical'
            IN pg_get_viewdef(
                'campaign_operations_request_status_v1'::regclass, true)) =
            0 OR
       position(
            'audit.actor_identity = request.accepting_actor_identity'
            IN pg_get_viewdef(
                'campaign_operations_request_status_v1'::regclass, true)) =
            0 THEN
        RAISE EXCEPTION
            'Phase 2 request status exact evidence validation missing';
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgrelid =
                'campaign_operations_operational_request'::regclass
          AND tgname =
                'campaign_operations_request_acquisition_complete_trigger'
          AND tgdeferrable AND tginitdeferred AND NOT tgisinternal) THEN
        RAISE EXCEPTION
            'atomic request/reservation acquisition constraint missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgrelid = 'campaign_operations_reservation'::regclass
          AND tgname =
              'campaign_operations_reservation_acquisition_complete_trigger'
          AND tgdeferrable AND tginitdeferred AND NOT tgisinternal) OR
       NOT EXISTS (
        SELECT 1
        FROM pg_trigger
        WHERE tgrelid =
                'campaign_operations_budget_ledger_entry'::regclass
          AND tgname =
                'campaign_operations_budget_audit_complete_trigger'
          AND tgdeferrable AND tginitdeferred AND NOT tgisinternal) THEN
        RAISE EXCEPTION
            'atomic reservation or budget audit constraint missing';
    END IF;

    IF to_regclass('campaign_operations_dispatch_attempt') IS NOT NULL OR
       to_regclass('campaign_operations_downstream_binding') IS NOT NULL OR
       to_regclass('campaign_operations_campaign_control') IS NOT NULL OR
       to_regclass('campaign_operations_completion') IS NOT NULL THEN
        RAISE EXCEPTION
            'later Campaign Operations authority leaked into Phase 2';
    END IF;
END
$$;
