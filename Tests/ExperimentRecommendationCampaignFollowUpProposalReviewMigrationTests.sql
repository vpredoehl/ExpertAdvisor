DO $migration_test$
DECLARE
    review_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal_review_event',
        current_schema());
    proposal_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal',
        current_schema());
    review_sequence text;
    review_function regprocedure;
BEGIN
    IF to_regclass(review_table) IS NULL THEN
        RAISE EXCEPTION 'follow-up proposal review-event table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(review_table)
          AND conname =
              'recommendation_campaign_follow_up_proposal_review_proposal_fk'
          AND contype = 'f'
          AND confrelid = to_regclass(proposal_table)
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'follow-up proposal review restrictive FK missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(review_table)
          AND conname =
              'recommendation_campaign_follow_up_proposal_review_once_uidx'
          AND contype = 'u') THEN
        RAISE EXCEPTION 'one-review-per-proposal uniqueness missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_campaign_follow_up_proposal_review_event'
          AND indexname =
              'recommendation_campaign_follow_up_proposal_review_identity_hash_idx')
    THEN
        RAISE EXCEPTION 'follow-up proposal review hash index missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(review_table)
          AND NOT tgisinternal
          AND tgname =
              'enforce_recommendation_campaign_follow_up_proposal_review_provenance_trigger')
    THEN
        RAISE EXCEPTION 'follow-up proposal review provenance trigger missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_attribute attribute
        JOIN pg_collation coll ON coll.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(review_table)
          AND attribute.attname IN (
              'proposal_identity_canonical',
              'review_identity_canonical')
          AND coll.collname = 'C') <> 2 THEN
        RAISE EXCEPTION 'review authoritative canonicals are not byte-exact';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name =
              'experiment_recommendation_campaign_follow_up_proposal_review_event'
          AND column_name IN (
              'activated', 'execution_authorized', 'follow_up_authorized',
              'queued', 'scheduled', 'scheduler_started',
              'scheduler_signaled', 'workers_started',
              'experiments_created', 'experiments_modified',
              'campaign_success_declared', 'superseded', 'expired')) THEN
        RAISE EXCEPTION 'review-event table contains forbidden authority state';
    END IF;

    IF NOT has_table_privilege('pqxx', review_table, 'SELECT') THEN
        RAISE EXCEPTION 'follow-up proposal review runtime SELECT missing';
    END IF;
    IF has_table_privilege('pqxx', review_table, 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal review table INSERT too broad';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', review_table, 'review_contract_version', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', review_table, 'review_decision', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', review_table, 'review_identity_canonical', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal review payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', review_table,
            'recommendation_campaign_follow_up_proposal_review_event_id',
            'INSERT') OR
       has_column_privilege(
            'pqxx', review_table, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal review generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', review_table, 'UPDATE') OR
       has_table_privilege('pqxx', review_table, 'DELETE') OR
       has_table_privilege('pqxx', review_table, 'TRUNCATE') THEN
        RAISE EXCEPTION 'follow-up proposal review mutation privilege present';
    END IF;
    IF has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_follow_up_proposal_review_provenance()',
           'EXECUTE') THEN
        RAISE EXCEPTION 'follow-up proposal review trigger privilege present';
    END IF;

    SELECT trigger.tgfoid::regprocedure
    INTO STRICT review_function
    FROM pg_trigger trigger
    WHERE trigger.tgrelid = to_regclass(review_table)
      AND NOT trigger.tgisinternal
      AND trigger.tgname =
          'enforce_recommendation_campaign_follow_up_proposal_review_provenance_trigger';
    IF review_function IS DISTINCT FROM to_regprocedure(format(
            '%I.enforce_recommendation_campaign_follow_up_proposal_review_provenance()',
            current_schema())) THEN
        RAISE EXCEPTION 'follow-up proposal review trigger function incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_proc function
        CROSS JOIN LATERAL aclexplode(coalesce(
            function.proacl, acldefault('f', function.proowner))) privilege
        WHERE function.oid = review_function
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal review trigger PUBLIC privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_proc function
        WHERE function.oid = review_function
          AND (function.prosecdef OR function.proconfig IS DISTINCT FROM
              ARRAY[format('search_path=pg_catalog, %I, pg_temp',
                  current_schema())])) THEN
        RAISE EXCEPTION 'follow-up proposal review trigger execution context unsafe';
    END IF;

    review_sequence := pg_get_serial_sequence(
        review_table,
        'recommendation_campaign_follow_up_proposal_review_event_id');
    IF NOT has_sequence_privilege('pqxx', review_sequence, 'USAGE') OR
       has_sequence_privilege('pqxx', review_sequence, 'SELECT') OR
       has_sequence_privilege('pqxx', review_sequence, 'UPDATE') THEN
        RAISE EXCEPTION 'follow-up proposal review sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('s', relation.relowner))) privilege
        WHERE relation.oid = review_sequence::regclass
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal review sequence PUBLIC privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(review_table)
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal review PUBLIC privilege present';
    END IF;
END
$migration_test$;
