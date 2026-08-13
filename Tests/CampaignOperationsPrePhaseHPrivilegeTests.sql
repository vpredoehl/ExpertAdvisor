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

    IF has_function_privilege(
            'pqxx',
            'transition_campaign_operations_request_dispatch_production_v2(bigint,integer,text,timestamp with time zone,text,text,text)',
            'EXECUTE') OR
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
             AND pg_has_role('pqxx', production_role.rolname, 'MEMBER')) THEN
        RAISE EXCEPTION
            'generic pre-Phase-H principal has Phase-H production capability';
    END IF;
END $$;

-- These are executable privilege regressions, not catalog-only assertions.
-- The test runner is an administrative role against a disposable database.
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
