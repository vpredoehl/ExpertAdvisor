DO $$
BEGIN
    IF to_regclass('experiment_recommendation_campaign_materialization') IS NULL
       OR to_regclass(
           'experiment_recommendation_campaign_materialization_member') IS NULL
    THEN
        RAISE EXCEPTION 'campaign materialization tables missing';
    END IF;
    IF (SELECT count(*) FROM information_schema.columns
        WHERE table_schema=current_schema()
          AND table_name='experiment_recommendation_campaign_materialization'
          AND column_name IN (
              'recommendation_campaign_materialization_id',
              'recommendation_campaign_approval_id',
              'materialization_identity_canonical',
              'materialization_identity_hash','created_at')) <> 5 THEN
        RAISE EXCEPTION 'campaign materialization columns missing';
    END IF;
    IF (SELECT count(*) FROM pg_constraint
        WHERE conrelid='experiment_recommendation_campaign_materialization'::regclass
          AND contype='f') <> 2 THEN
        RAISE EXCEPTION 'manifest restrictive foreign keys missing';
    END IF;
    IF (SELECT count(*) FROM pg_constraint
        WHERE conrelid=
          'experiment_recommendation_campaign_materialization_member'::regclass
          AND contype='f') <> 5 THEN
        RAISE EXCEPTION 'member restrictive foreign keys missing';
    END IF;
    IF (SELECT count(*) FROM pg_trigger
        WHERE tgrelid IN (
            'experiment_recommendation_campaign_materialization'::regclass,
            'experiment_recommendation_campaign_materialization_member'::regclass)
          AND NOT tgisinternal
          AND tgname LIKE 'enforce_recommendation_campaign_materialization_%')
       <> 3 THEN
        RAISE EXCEPTION 'materialization enforcement triggers missing';
    END IF;
    IF has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization','UPDATE')
       OR has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization','DELETE')
       OR has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization','TRUNCATE')
       OR has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization_member','UPDATE')
       OR has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization_member','DELETE')
       OR has_table_privilege(
        'pqxx','experiment_recommendation_campaign_materialization_member','TRUNCATE')
    THEN
        RAISE EXCEPTION 'append-only runtime privilege violated';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.role_table_grants
        WHERE table_schema=current_schema() AND grantee='PUBLIC'
          AND table_name IN (
            'experiment_recommendation_campaign_materialization',
            'experiment_recommendation_campaign_materialization_member'))
    THEN
        RAISE EXCEPTION 'PUBLIC materialization privilege present';
    END IF;
    IF has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_materialization_manifest_insert()',
           'EXECUTE')
       OR has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_materialization_member_insert()',
           'EXECUTE')
       OR has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_materialization_complete()',
           'EXECUTE')
    THEN
        RAISE EXCEPTION 'runtime trigger-function execution privilege present';
    END IF;
END $$;
