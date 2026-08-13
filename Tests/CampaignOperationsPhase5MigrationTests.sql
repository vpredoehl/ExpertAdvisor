-- Catalog-level Campaign Operations Phase 5 / architectural Phase G checks.
DO $$
DECLARE table_name text;
DECLARE mutation_table text;
BEGIN
    FOREACH table_name IN ARRAY ARRAY[
        'campaign_operations_completion_event',
        'campaign_operations_completion_audit_reference_event']
    LOOP
        IF to_regclass(table_name) IS NULL THEN
            RAISE EXCEPTION 'missing Phase G table %', table_name;
        END IF;
    END LOOP;
    IF to_regclass('campaign_operations_completion_status_v1') IS NULL THEN
        RAISE EXCEPTION 'missing Phase G read-only status view';
    END IF;

    IF NOT EXISTS (
            SELECT 1 FROM pg_roles
            WHERE rolname = 'campaign_operations_completion_writer'
              AND NOT rolcanlogin AND NOT rolsuper AND NOT rolcreatedb
              AND NOT rolcreaterole AND NOT rolreplication
              AND NOT rolbypassrls) OR
       pg_has_role(
            'pqxx', 'campaign_operations_completion_writer', 'MEMBER') THEN
        RAISE EXCEPTION 'Phase G completion role hardening mismatch';
    END IF;

    IF to_regprocedure(
            'campaign_operations_tagged_fnv1a64(text)') IS NULL OR
       to_regprocedure(
            'campaign_operations_completion_identity_valid(campaign_operations_completion_event)') IS NULL OR
       to_regprocedure(
            'guard_campaign_operations_completion_boundary_mutation()') IS NULL OR
       to_regprocedure(
            'enforce_campaign_operations_completion_boundary_consistent()') IS NULL OR
       (SELECT pg_get_userbyid(proowner) FROM pg_proc
        WHERE oid = 'campaign_operations_tagged_fnv1a64(text)'::
            regprocedure) <> 'campaign_operations_owner' OR
       (SELECT pg_get_userbyid(proowner) FROM pg_proc
        WHERE oid =
          'campaign_operations_completion_identity_valid(campaign_operations_completion_event)'::regprocedure) <>
            'campaign_operations_owner' OR
       (SELECT pg_get_userbyid(proowner) FROM pg_proc
        WHERE oid =
          'guard_campaign_operations_completion_boundary_mutation()'::regprocedure) <>
            'campaign_operations_owner' OR
       (SELECT pg_get_userbyid(proowner) FROM pg_proc
        WHERE oid =
          'enforce_campaign_operations_completion_boundary_consistent()'::regprocedure) <>
            'campaign_operations_owner' THEN
        RAISE EXCEPTION 'Phase G completion identity validator mismatch';
    END IF;

    IF (SELECT pg_get_userbyid(relowner)
        FROM pg_class
        WHERE oid = 'campaign_operations_completion_event'::regclass) <>
            'campaign_operations_owner' OR
       (SELECT pg_get_userbyid(relowner)
        FROM pg_class
        WHERE oid =
          'campaign_operations_completion_audit_reference_event'::regclass) <>
            'campaign_operations_owner' THEN
        RAISE EXCEPTION 'Phase G owner mismatch';
    END IF;

    IF has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event', 'INSERT') OR
       NOT has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event',
            'operational_campaign_id', 'INSERT') OR
       NOT has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event',
            'completion_identity_hash', 'INSERT') OR
       NOT has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event',
            'authorization_evidence_canonical', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event',
            'completion_event_id', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event', 'DELETE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_event', 'TRUNCATE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_audit_reference_event',
            'INSERT') OR
       NOT has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_audit_reference_event',
            'completion_event_id', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_audit_reference_event',
            'completion_audit_reference_event_id', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_audit_reference_event',
            'UPDATE') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_completion_status_v1', 'SELECT') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_completion_event', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_completion_event', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_completion_event', 'DELETE') THEN
        RAISE EXCEPTION 'Phase G completion/read ACL mismatch';
    END IF;

    IF has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_operational_request', 'INSERT') OR
       has_column_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_campaign',
            'completion_boundary_closed', 'UPDATE') OR
       has_column_privilege(
            'experiment_lifecycle_cancellation',
            'campaign_operations_campaign',
            'completion_boundary_closed', 'UPDATE') OR
       has_column_privilege(
            'pqxx', 'campaign_operations_campaign',
            'completion_boundary_closed', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_reservation', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_cancellation_settlement', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_reconciliation_resolution', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'experiment', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'experiment', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_completion_writer',
            'experiment', 'DELETE') THEN
        RAISE EXCEPTION 'Phase G writer crosses an owning boundary';
    END IF;
    IF has_table_privilege(
            'campaign_operations_owner', 'experiment', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_owner', 'experiment', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_owner', 'experiment', 'DELETE') OR
       NOT has_column_privilege(
            'campaign_operations_owner', 'experiment',
            'status', 'SELECT') OR
       has_column_privilege(
            'campaign_operations_owner', 'experiment',
            'symbol', 'SELECT') THEN
        RAISE EXCEPTION 'Phase G lifecycle evidence privilege is not narrow';
    END IF;

    IF NOT has_function_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_blockers(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_classification(bigint)',
            'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_reader',
            'campaign_operations_completion_blockers(bigint)', 'EXECUTE') OR
       EXISTS (
            SELECT 1
            FROM pg_proc procedure,
                 LATERAL aclexplode(coalesce(
                     procedure.proacl,
                     acldefault('f', procedure.proowner))) privilege
            WHERE procedure.oid =
                  'campaign_operations_completion_classification(bigint)'::
                      regprocedure
              AND privilege.grantee = 0
              AND privilege.privilege_type = 'EXECUTE') THEN
        RAISE EXCEPTION 'Phase G function ACL mismatch';
    END IF;
    IF has_function_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_tagged_fnv1a64(text)', 'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_completion_writer',
            'campaign_operations_completion_identity_valid(campaign_operations_completion_event)',
            'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_completion_writer',
            'guard_campaign_operations_completion_boundary_mutation()',
            'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_completion_writer',
            'enforce_campaign_operations_completion_boundary_consistent()',
            'EXECUTE') OR
       EXISTS (
            SELECT 1 FROM pg_proc procedure,
            LATERAL aclexplode(coalesce(
                procedure.proacl,
                acldefault('f', procedure.proowner))) privilege
            WHERE procedure.oid IN (
                'campaign_operations_tagged_fnv1a64(text)'::regprocedure,
                'campaign_operations_completion_identity_valid(campaign_operations_completion_event)'::regprocedure,
                'guard_campaign_operations_completion_boundary_mutation()'::regprocedure,
                'enforce_campaign_operations_completion_boundary_consistent()'::regprocedure)
              AND privilege.grantee = 0
              AND privilege.privilege_type = 'EXECUTE') THEN
        RAISE EXCEPTION
            'Phase G internal identity validation authority leaked';
    END IF;
    IF has_function_privilege(
            'experiment_lifecycle_cancellation',
            'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)',
            'EXECUTE') OR
       NOT has_function_privilege(
            'experiment_lifecycle_cancellation_owner',
            'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)',
            'EXECUTE') OR
       (SELECT pg_get_userbyid(proowner) FROM pg_proc WHERE oid =
            'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)'::
                regprocedure) <> 'campaign_operations_owner' THEN
        RAISE EXCEPTION
            'Phase G lifecycle boundary helper authority mismatch';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM unnest(ARRAY[
            'campaign_operations_tagged_fnv1a64(text)'::regprocedure,
            'campaign_operations_completion_identity_valid(campaign_operations_completion_event)'::regprocedure,
            'guard_campaign_operations_completion_boundary_mutation()'::regprocedure,
            'enforce_campaign_operations_completion_boundary_consistent()'::regprocedure,
            'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)'::regprocedure,
            'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)'::regprocedure
        ]) AS functions(function_oid)
        JOIN pg_proc procedure ON procedure.oid = functions.function_oid
        WHERE procedure.proconfig IS NULL OR NOT EXISTS (
            SELECT 1 FROM unnest(procedure.proconfig) setting
            WHERE setting = 'search_path=pg_catalog, ' ||
                current_schema() || ', pg_temp')) THEN
        RAISE EXCEPTION 'Phase G corrected function search path is not pinned';
    END IF;

    IF NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid =
                  'campaign_operations_completion_event'::regclass
              AND contype = 'u'
              AND pg_get_constraintdef(oid) =
                  'UNIQUE (operational_campaign_id)') OR
       NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid =
                  'campaign_operations_completion_event'::regclass
              AND conname =
                  'campaign_operations_completion_member_equation') OR
       NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid =
                  'campaign_operations_completion_event'::regclass
              AND conname =
                  'campaign_operations_completion_budget_equation') OR
       NOT EXISTS (
            SELECT 1 FROM pg_constraint
            WHERE conrelid =
                  'campaign_operations_completion_event'::regclass
              AND conname =
                  'campaign_operations_completion_state_classification') THEN
        RAISE EXCEPTION 'Phase G invariant constraint set incomplete';
    END IF;

    IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_validate_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_audit_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_operational_request_completion_gate_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_authorization_event_completion_gate_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_immutable_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_audit_immutable_trigger') THEN
        RAISE EXCEPTION 'Phase G enforcement trigger set incomplete';
    END IF;
    IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_boundary_update_guard') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_boundary_truncate_guard') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_boundary_consistency_trigger'
              AND tgdeferrable AND tginitdeferred) THEN
        RAISE EXCEPTION
            'Phase G completion boundary invariant triggers missing';
    END IF;
    IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_truncate_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                  'campaign_operations_completion_audit_truncate_trigger') THEN
        RAISE EXCEPTION 'Phase G truncate guards missing';
    END IF;

    FOREACH mutation_table IN ARRAY ARRAY[
        'campaign_operations_governance_provenance_event',
        'campaign_operations_authorization_event',
        'campaign_operations_budget_ledger_entry',
        'campaign_operations_reservation',
        'campaign_operations_operational_request',
        'campaign_operations_reservation_event',
        'campaign_operations_dispatch_attempt',
        'campaign_operations_dispatch_attempt_outcome',
        'campaign_operations_request_binding',
        'campaign_operations_downstream_control_owner',
        'campaign_operations_reservation_commitment',
        'campaign_operations_control_event',
        'campaign_operations_cancellation_request',
        'campaign_operations_cancellation_settlement',
        'campaign_operations_reconciliation_observation',
        'campaign_operations_reconciliation_resolution',
        'campaign_operations_audit_reference_event',
        'campaign_operations_dispatch_audit_reference_event',
        'campaign_operations_control_audit_reference_event',
        'experiment_lifecycle_cancellation_event']
    LOOP
        IF NOT EXISTS (
            SELECT 1
            FROM pg_trigger trigger_catalog
            JOIN pg_class relation_catalog
              ON relation_catalog.oid = trigger_catalog.tgrelid
            WHERE relation_catalog.relname = mutation_table
              AND relation_catalog.relnamespace =
                  to_regnamespace(current_schema())
              AND trigger_catalog.tgfoid =
                  'guard_campaign_operations_completed_campaign()'::
                      regprocedure
              AND (trigger_catalog.tgtype & 4) = 4
              AND NOT trigger_catalog.tgisinternal) THEN
            RAISE EXCEPTION
                'Phase G completion INSERT gate missing for %',
                mutation_table;
        END IF;
    END LOOP;
    IF (SELECT count(*)
        FROM pg_trigger trigger_catalog
        JOIN pg_class relation_catalog
          ON relation_catalog.oid = trigger_catalog.tgrelid
        WHERE relation_catalog.relname IN (
                  'campaign_operations_reservation',
                  'campaign_operations_operational_request')
          AND relation_catalog.relnamespace =
              to_regnamespace(current_schema())
          AND trigger_catalog.tgfoid =
              'guard_campaign_operations_completed_campaign()'::regprocedure
          AND (trigger_catalog.tgtype & 16) = 16
          AND NOT trigger_catalog.tgisinternal) <> 2 THEN
        RAISE EXCEPTION 'Phase G guarded UPDATE gates incomplete';
    END IF;
    IF position('campaign_operations_completion_event' in
            pg_get_functiondef(
                'campaign_operations_future_actions_allowed(bigint)'::
                    regprocedure)) = 0 THEN
        RAISE EXCEPTION
            'Phase G future-action service gate omits completion';
    END IF;
    IF position(
            'campaign_operations_lifecycle_cancellation_boundary_completed' in
            pg_get_functiondef(
                'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)'::
                    regprocedure)) = 0 OR
       position('event_identity_canonical <> event_canonical_value' in
            pg_get_functiondef(
                'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)'::
                    regprocedure)) = 0 THEN
        RAISE EXCEPTION
            'Phase G lifecycle-cancellation completion/replay gate missing';
    END IF;
    IF position('completion_boundary_closed' in
            pg_get_functiondef(
                'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)'::
                    regprocedure)) = 0 OR
       position('FOR UPDATE' in upper(pg_get_functiondef(
                'campaign_operations_lifecycle_cancellation_boundary_completed(bigint)'::
                    regprocedure))) = 0 THEN
        RAISE EXCEPTION
            'Phase G lifecycle-cancellation campaign serialization missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.columns columns_catalog
        WHERE columns_catalog.table_schema = current_schema()
          AND columns_catalog.table_name = 'campaign_operations_campaign'
          AND columns_catalog.column_name = 'completion_boundary_closed'
          AND columns_catalog.data_type = 'boolean'
          AND columns_catalog.is_nullable = 'NO'
          AND columns_catalog.column_default = 'false') THEN
        RAISE EXCEPTION
            'Phase G durable campaign completion boundary missing';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM information_schema.columns columns_catalog
        WHERE columns_catalog.table_schema = current_schema()
          AND columns_catalog.table_name =
              'campaign_operations_completion_event'
          AND columns_catalog.column_name IN (
              'superseded_by', 'reopened_at', 'deleted_at',
              'override_actor', 'force_completed')) THEN
        RAISE EXCEPTION 'Phase G contains a prohibited override/reopen path';
    END IF;
END $$;
