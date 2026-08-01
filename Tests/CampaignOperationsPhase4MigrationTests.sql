-- Catalog-level Campaign Operations Phase 4 / architectural Phase F checks.
-- The owner fixture applies migration 053 twice before executing this file.
DO $$
DECLARE object_name text;
DECLARE role_name text;
DECLARE function_name text;
DECLARE owner_name text;
DECLARE function_configuration text[];
BEGIN
    FOREACH object_name IN ARRAY ARRAY[
        'campaign_operations_control_event',
        'campaign_operations_cancellation_request',
        'campaign_operations_cancellation_settlement',
        'experiment_lifecycle_cancellation_event',
        'campaign_operations_reconciliation_observation',
        'campaign_operations_reconciliation_resolution',
        'campaign_operations_reconciliation_cursor_event',
        'campaign_operations_control_audit_reference_event']
    LOOP
        IF to_regclass(object_name) IS NULL THEN
            RAISE EXCEPTION 'missing Campaign Operations Phase 4 table %',
                object_name;
        END IF;
        SELECT pg_get_userbyid(relowner) INTO owner_name
        FROM pg_class WHERE oid = to_regclass(object_name);
        IF owner_name <> 'campaign_operations_owner' THEN
            RAISE EXCEPTION 'unexpected Phase 4 owner for %: %',
                object_name, owner_name;
        END IF;
        IF has_table_privilege('pqxx', object_name, 'SELECT') OR
           has_table_privilege('pqxx', object_name, 'INSERT') OR
           has_table_privilege('pqxx', object_name, 'UPDATE') OR
           has_table_privilege('pqxx', object_name, 'DELETE') THEN
            RAISE EXCEPTION 'pqxx has forbidden Phase 4 privilege on %',
                object_name;
        END IF;
    END LOOP;

    FOREACH role_name IN ARRAY ARRAY[
        'campaign_operations_controller',
        'campaign_operations_cancellation_coordinator',
        'campaign_operations_reconciler',
        'campaign_operations_recovery',
        'experiment_lifecycle_cancellation',
        'experiment_lifecycle_cancellation_owner']
    LOOP
        IF NOT EXISTS (
            SELECT 1 FROM pg_roles
            WHERE rolname = role_name
              AND NOT rolcanlogin AND NOT rolsuper
              AND NOT rolcreatedb AND NOT rolcreaterole
              AND NOT rolreplication AND NOT rolbypassrls) THEN
            RAISE EXCEPTION 'Phase 4 capability role not hardened: %',
                role_name;
        END IF;
        IF pg_has_role('pqxx', role_name, 'MEMBER') OR EXISTS (
            SELECT 1
            FROM pg_auth_members membership
            JOIN pg_roles member_role
              ON member_role.oid = membership.member
            WHERE membership.roleid = role_name::regrole
              AND member_role.rolcanlogin) THEN
            RAISE EXCEPTION
                'login principal inherits Phase 4 capability %', role_name;
        END IF;
    END LOOP;

    FOREACH function_name IN ARRAY ARRAY[
        'enforce_campaign_operations_control_chain()',
        'enforce_campaign_operations_future_action_gate()',
        'enforce_campaign_operations_cancellation_target()',
        'campaign_operations_future_actions_allowed(bigint)',
        'guard_campaign_operations_dispatch_control()',
        'enforce_campaign_operations_control_audit_complete()',
        'enforce_campaign_operations_phase4_request_transition_complete()',
        'append_campaign_operations_cancellation_resolution(bigint,text,text,text,text)',
        'append_campaign_operations_recovery_resolution(bigint,text,text,text,text,text)',
        'enforce_campaign_operations_reconciliation_cursor_complete()',
        'transition_campaign_operations_request_cancelled(bigint,integer)',
        'transition_campaign_operations_reservation_released(bigint,integer)',
        'transition_campaign_operations_request_ready_recovered(bigint,integer)',
        'apply_experiment_lifecycle_cancellation(bigint,bigint,bigint,text,text,text,text,text)']
    LOOP
        SELECT proconfig INTO function_configuration
        FROM pg_proc WHERE oid = function_name::regprocedure;
        IF function_configuration IS NULL OR NOT EXISTS (
            SELECT 1 FROM unnest(function_configuration) setting
            WHERE setting =
                'search_path=pg_catalog, ' || current_schema() || ', pg_temp')
        THEN
            RAISE EXCEPTION 'Phase 4 function search path not pinned: %',
                function_name;
        END IF;
    END LOOP;

    IF NOT has_column_privilege(
            'campaign_operations_controller',
            'campaign_operations_control_event',
            'control_identity_canonical', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_controller',
            'campaign_operations_control_event', 'UPDATE') OR
       NOT has_column_privilege(
            'campaign_operations_cancellation_coordinator',
            'campaign_operations_cancellation_request',
            'cancellation_identity_canonical', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_cancellation_coordinator',
            'campaign_operations_cancellation_request', 'UPDATE') OR
       NOT has_column_privilege(
            'campaign_operations_reconciler',
            'campaign_operations_reconciliation_observation',
            'observation_identity_canonical', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_reconciler',
            'campaign_operations_reconciliation_resolution', 'INSERT') OR
       has_table_privilege(
            'campaign_operations_recovery',
            'campaign_operations_reconciliation_resolution',
            'INSERT') OR
       has_table_privilege(
            'campaign_operations_cancellation_coordinator',
            'campaign_operations_reconciliation_resolution',
            'INSERT') OR
       NOT has_function_privilege(
            'campaign_operations_recovery',
            'append_campaign_operations_recovery_resolution(bigint,text,text,text,text,text)',
            'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_recovery',
            'append_campaign_operations_cancellation_resolution(bigint,text,text,text,text)',
            'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_cancellation_coordinator',
            'append_campaign_operations_cancellation_resolution(bigint,text,text,text,text)',
            'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_cancellation_coordinator',
            'append_campaign_operations_recovery_resolution(bigint,text,text,text,text,text)',
            'EXECUTE') OR
       NOT has_column_privilege(
            'campaign_operations_cancellation_coordinator',
            'campaign_operations_control_audit_reference_event',
            'reconciliation_resolution_id', 'INSERT') OR
       has_sequence_privilege(
            'campaign_operations_cancellation_coordinator',
            pg_get_serial_sequence(
                'campaign_operations_reconciliation_resolution',
                'reconciliation_resolution_id'), 'USAGE') OR
       has_sequence_privilege(
            'campaign_operations_recovery',
            pg_get_serial_sequence(
                'campaign_operations_reconciliation_resolution',
                'reconciliation_resolution_id'), 'USAGE') OR
       NOT has_table_privilege(
            'campaign_operations_recovery',
            'campaign_operations_cancellation_settlement', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_recovery',
            'campaign_operations_dispatch_audit_reference_event', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_control_event', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_cancellation_request', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_cancellation_settlement', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_reconciliation_observation', 'SELECT') OR
       NOT has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_reconciliation_resolution', 'SELECT') OR
       has_table_privilege(
            'campaign_operations_reader',
            'campaign_operations_reconciliation_resolution', 'INSERT') OR
       has_sequence_privilege(
            'campaign_operations_reader',
            pg_get_serial_sequence(
                'campaign_operations_reconciliation_resolution',
                'reconciliation_resolution_id'), 'USAGE') OR
       NOT has_function_privilege(
            'campaign_operations_reconciler',
            'lock_campaign_operations_request(bigint)', 'EXECUTE') OR
       NOT has_function_privilege(
            'campaign_operations_reconciler',
            'lock_campaign_operations_campaign(bigint)', 'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_reconciler',
            'lock_campaign_operations_reservation(bigint)', 'EXECUTE') OR
       has_function_privilege(
            'campaign_operations_reconciler',
            'lock_campaign_operations_budget_head(bigint)', 'EXECUTE') THEN
        RAISE EXCEPTION 'Phase 4 capability boundary mismatch';
    END IF;

    IF NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_dispatch_control_gate_trigger') OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_control_audit_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_cancellation_request_audit_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_reconciliation_resolution_audit_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_reconciliation_cursor_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_reconciliation_membership_complete_trigger'
              AND tgdeferrable AND tginitdeferred) OR
       NOT EXISTS (
            SELECT 1 FROM pg_trigger
            WHERE tgname =
                'campaign_operations_phase4_request_transition_complete_trigger'
              AND tgdeferrable AND tginitdeferred) THEN
        RAISE EXCEPTION 'Phase 4 enforcement trigger set incomplete';
    END IF;

    IF NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_row
            WHERE constraint_row.conrelid =
                'campaign_operations_reconciliation_cursor_event'::regclass
              AND constraint_row.contype = 'u'
              AND pg_get_constraintdef(constraint_row.oid) =
                  'UNIQUE (run_key, prior_target_id)') THEN
        RAISE EXCEPTION
            'Phase 4 reconciliation cursor replay key mismatch';
    END IF;

    IF NOT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname =
                  'campaign_operations_cancellation_request_target_uidx'
              AND indexdef LIKE
                  '%UNIQUE%operational_request_id%WHERE (operational_request_id IS NOT NULL)%') OR
       NOT EXISTS (
            SELECT 1
            FROM pg_indexes
            WHERE schemaname = current_schema()
              AND indexname =
                  'campaign_operations_cancellation_campaign_target_uidx'
              AND indexdef LIKE
                  '%UNIQUE%operational_campaign_id%WHERE (operational_request_id IS NULL)%') OR
       NOT EXISTS (
            SELECT 1
            FROM pg_constraint
            WHERE conrelid =
                'campaign_operations_reconciliation_resolution'::regclass
              AND conname =
                  'campaign_operations_resolution_causal_shape_check') THEN
        RAISE EXCEPTION
            'Phase 4 causal ownership constraint set incomplete';
    END IF;

    IF EXISTS (
            SELECT 1
            FROM information_schema.columns
            WHERE table_schema = current_schema()
              AND table_name =
                  'campaign_operations_reconciliation_observation'
              AND column_name = 'reconciliation_cursor_event_id'
              AND is_nullable <> 'NO') OR
       NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_row
            WHERE constraint_row.conrelid =
                'campaign_operations_reconciliation_observation'::regclass
              AND constraint_row.conname =
                  'campaign_operations_observation_cursor_fk'
              AND constraint_row.contype = 'f') OR
       NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_row
            WHERE constraint_row.conrelid =
                'campaign_operations_reconciliation_observation'::regclass
              AND constraint_row.conname =
                  'campaign_operations_observation_cursor_request_uidx'
              AND constraint_row.contype = 'u') THEN
        RAISE EXCEPTION
            'Phase 4 exact reconciliation batch membership mismatch';
    END IF;

    IF has_table_privilege(
            'campaign_operations_controller', 'experiment', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_cancellation_coordinator',
            'experiment', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_reconciler', 'experiment', 'UPDATE') OR
       has_table_privilege(
            'campaign_operations_recovery', 'experiment', 'UPDATE') OR
       has_table_privilege(
            'experiment_lifecycle_cancellation',
            'campaign_operations_operational_request', 'UPDATE') THEN
        RAISE EXCEPTION
            'Phase 4 capability bypasses its owning service boundary';
    END IF;
END $$;
