DO $migration_test$
DECLARE
    ratification_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_ratification_event',
        current_schema());
    review_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal_review_event',
        current_schema());
    proposal_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal',
        current_schema());
    ratification_sequence text;
    ratification_function regprocedure;
BEGIN
    IF to_regclass(ratification_table) IS NULL THEN
        RAISE EXCEPTION 'follow-up proposal ratification-event table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND conname =
              'follow_up_ratification_review_fk'
          AND contype = 'f'
          AND confrelid = to_regclass(review_table)
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'follow-up proposal ratification review FK missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND conname =
              'follow_up_ratification_proposal_fk'
          AND contype = 'f'
          AND confrelid = to_regclass(proposal_table)
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'follow-up proposal ratification proposal FK missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND conname IN (
              'follow_up_ratification_review_uidx',
              'follow_up_ratification_proposal_uidx')
          AND contype = 'u') <> 2 THEN
        RAISE EXCEPTION 'one-ratification-per-review/proposal uniqueness missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_campaign_follow_up_ratification_event'
          AND indexname =
              'follow_up_ratification_identity_hash_idx')
    THEN
        RAISE EXCEPTION 'follow-up proposal ratification hash index missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(ratification_table)
          AND NOT tgisinternal
          AND tgname =
              'follow_up_ratification_provenance_trigger')
    THEN
        RAISE EXCEPTION 'follow-up proposal ratification provenance trigger missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_attribute attribute
        JOIN pg_collation coll ON coll.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(ratification_table)
          AND attribute.attname IN (
              'review_identity_canonical',
              'reviewer_identity',
              'proposal_identity_canonical',
              'ratification_authority_role',
              'ratifier_identity',
              'ratification_basis',
              'ratification_identity_canonical')
          AND coll.collname = 'C') <> 7 THEN
        RAISE EXCEPTION 'ratification identity-bearing text is not byte-exact';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name =
              'experiment_recommendation_campaign_follow_up_ratification_event'
          AND column_name IN (
              'activated', 'execution_authorized', 'follow_up_authorized',
              'queued', 'scheduled', 'scheduler_started',
              'scheduler_signaled', 'workers_started',
              'experiments_created', 'experiments_modified',
              'campaign_success_declared', 'superseded', 'expired')) THEN
        RAISE EXCEPTION 'ratification-event table contains forbidden authority state';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE
              '%review_decision = ''approved''%') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE
              '%ratification_decision = ''ratified''%') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND conname =
              'follow_up_ratification_separation_of_duties_check'
          AND contype = 'c') OR
       NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(ratification_table)
          AND contype = 'c'
          AND pg_get_constraintdef(oid) LIKE
              '%ratification_authority_role = ''follow_up_governance_ratifier''%') THEN
        RAISE EXCEPTION 'ratification eligibility/decision constraint missing';
    END IF;

    IF NOT has_table_privilege('pqxx', ratification_table, 'SELECT') THEN
        RAISE EXCEPTION 'follow-up proposal ratification runtime SELECT missing';
    END IF;
    IF has_table_privilege('pqxx', ratification_table, 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal ratification table INSERT too broad';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', ratification_table, 'ratification_contract_version', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', ratification_table, 'review_identity_canonical', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', ratification_table, 'reviewer_identity', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', ratification_table, 'ratification_authority_role', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', ratification_table, 'ratification_identity_canonical', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal ratification payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', ratification_table,
            'recommendation_campaign_follow_up_ratification_event_id',
            'INSERT') OR
       has_column_privilege(
            'pqxx', ratification_table, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal ratification generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', ratification_table, 'UPDATE') OR
       has_table_privilege('pqxx', ratification_table, 'DELETE') OR
       has_table_privilege('pqxx', ratification_table, 'TRUNCATE') THEN
        RAISE EXCEPTION 'follow-up proposal ratification mutation privilege present';
    END IF;
    IF has_function_privilege(
           'pqxx',
           'enforce_follow_up_ratification_provenance()',
           'EXECUTE') THEN
        RAISE EXCEPTION 'follow-up proposal ratification trigger privilege present';
    END IF;

    SELECT trigger.tgfoid::regprocedure
    INTO STRICT ratification_function
    FROM pg_trigger trigger
    WHERE trigger.tgrelid = to_regclass(ratification_table)
      AND NOT trigger.tgisinternal
      AND trigger.tgname =
          'follow_up_ratification_provenance_trigger';
    IF ratification_function IS DISTINCT FROM to_regprocedure(format(
            '%I.enforce_follow_up_ratification_provenance()',
            current_schema())) THEN
        RAISE EXCEPTION 'follow-up proposal ratification trigger function incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_proc function
        CROSS JOIN LATERAL aclexplode(coalesce(
            function.proacl, acldefault('f', function.proowner))) privilege
        WHERE function.oid = ratification_function
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal ratification trigger PUBLIC privilege present';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_proc function
        CROSS JOIN LATERAL aclexplode(coalesce(
            NULL::aclitem[], acldefault('f', function.proowner))) privilege
        WHERE function.oid = ratification_function
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'function NULL-ACL fallback fixture invalid';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_proc function
        WHERE function.oid = ratification_function
          AND (function.prosecdef OR function.proconfig IS DISTINCT FROM
              ARRAY[format('search_path=pg_catalog, %I, pg_temp',
                  current_schema())])) THEN
        RAISE EXCEPTION 'follow-up proposal ratification trigger execution context unsafe';
    END IF;

    ratification_sequence := pg_get_serial_sequence(
        ratification_table,
        'recommendation_campaign_follow_up_ratification_event_id');
    IF NOT has_sequence_privilege('pqxx', ratification_sequence, 'USAGE') OR
       has_sequence_privilege('pqxx', ratification_sequence, 'SELECT') OR
       has_sequence_privilege('pqxx', ratification_sequence, 'UPDATE') THEN
        RAISE EXCEPTION 'follow-up proposal ratification sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('s', relation.relowner))) privilege
        WHERE relation.oid = ratification_sequence::regclass
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal ratification sequence PUBLIC privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            NULL::aclitem[], acldefault('s', relation.relowner))) privilege
        WHERE relation.oid = ratification_sequence::regclass
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'sequence NULL-ACL fallback grants PUBLIC';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(ratification_table)
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal ratification PUBLIC privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            NULL::aclitem[], acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(ratification_table)
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'table NULL-ACL fallback grants PUBLIC';
    END IF;
END
$migration_test$;
