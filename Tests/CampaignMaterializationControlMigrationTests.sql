DO $migration_test$
BEGIN
    IF to_regclass(
           'experiment_campaign_materialization_control_operation') IS NULL OR
       to_regclass(
           'experiment_campaign_materialization_control_member') IS NULL OR
       to_regclass(
           'experiment_campaign_materialization_control_outcome') IS NULL OR
       to_regclass(
           'experiment_campaign_materialization_pause_ownership') IS NULL THEN
        RAISE EXCEPTION 'campaign materialization control tables missing';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid =
            'experiment_campaign_materialization_control_operation'::regclass
          AND contype = 'f'
          AND confrelid =
            'experiment_recommendation_campaign_materialization'::regclass
    ) OR NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid =
            'experiment_campaign_materialization_control_member'::regclass
          AND contype = 'f'
          AND confrelid =
            'experiment_recommendation_campaign_materialization_member'::regclass
    ) THEN
        RAISE EXCEPTION 'materialization provenance foreign keys missing';
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND indexname =
              'campaign_materialization_pause_active_experiment_uidx'
          AND indexdef LIKE '%WHERE (ownership_state = ''active''::text)%'
    ) THEN
        RAISE EXCEPTION 'single active ownership index missing';
    END IF;

    IF (SELECT count(*) FROM pg_trigger
        WHERE tgrelid IN (
            'experiment_campaign_materialization_control_operation'::regclass,
            'experiment_campaign_materialization_control_member'::regclass,
            'experiment_campaign_materialization_control_outcome'::regclass)
          AND NOT tgisinternal
          AND tgname LIKE '%immutable_trigger') <> 3 THEN
        RAISE EXCEPTION 'immutable evidence triggers incomplete';
    END IF;

    IF NOT has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_operation',
           'SELECT') OR
       NOT has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_member',
           'INSERT') OR
       NOT has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_outcome',
           'INSERT') OR
       NOT has_column_privilege(
           'pqxx',
           'experiment_campaign_materialization_pause_ownership',
           'ownership_state',
           'UPDATE') THEN
        RAISE EXCEPTION 'runtime campaign control privileges incomplete';
    END IF;

    IF has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_operation',
           'UPDATE') OR
       has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_member',
           'DELETE') OR
       has_table_privilege(
           'pqxx',
           'experiment_campaign_materialization_control_outcome',
           'UPDATE') OR
       has_column_privilege(
           'pqxx',
           'experiment_campaign_materialization_pause_ownership',
           'experiment_id',
           'UPDATE') THEN
        RAISE EXCEPTION 'immutable evidence or ownership identity is mutable';
    END IF;
END
$migration_test$;
