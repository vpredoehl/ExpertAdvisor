DO $migration_test$
DECLARE
    table_name text := format(
        '%I.experiment_recommendation_campaign_approval', current_schema());
BEGIN
    IF to_regclass(table_name) IS NULL THEN
        RAISE EXCEPTION 'campaign approval table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND contype = 'f'
          AND confrelid =
              'experiment_recommendation_ranking_snapshot'::regclass
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'campaign approval snapshot FK is not restrictive';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname =
              'recommendation_campaign_approval_review_hash_ordinal_uidx') THEN
        RAISE EXCEPTION 'campaign approval uniqueness missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname =
              'recommendation_campaign_approval_hash_format_check'
          AND contype = 'c') THEN
        RAISE EXCEPTION 'campaign approval hash format constraint missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = 'experiment_recommendation_campaign_approval'
          AND indexname IN (
              'recommendation_campaign_approval_review_hash_idx',
              'recommendation_campaign_approval_review_canonical_idx',
              'recommendation_campaign_approval_snapshot_idx',
              'recommendation_campaign_approval_decision_idx')) <> 4 THEN
        RAISE EXCEPTION 'campaign approval indexes missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = 'experiment_recommendation_campaign_approval'
          AND indexname =
              'recommendation_campaign_approval_review_canonical_idx'
          AND indexdef LIKE '%USING hash (campaign_review_identity_canonical)%'
    ) THEN
        RAISE EXCEPTION 'campaign approval canonical hash index invalid';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_attribute attribute
        JOIN pg_collation coll ON coll.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(table_name)
          AND attribute.attname = 'campaign_review_identity_canonical'
          AND coll.collname = 'C') THEN
        RAISE EXCEPTION 'campaign review canonical is not byte-exact';
    END IF;
    IF NOT has_table_privilege('pqxx', table_name, 'SELECT') THEN
        RAISE EXCEPTION 'campaign approval runtime SELECT missing';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'INSERT') THEN
        RAISE EXCEPTION 'campaign approval runtime table INSERT is too broad';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', table_name, 'decision', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name,
            'campaign_review_identity_canonical', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'approval_identity_canonical', 'INSERT') THEN
        RAISE EXCEPTION 'campaign approval payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', table_name,
            'recommendation_campaign_approval_id', 'INSERT') OR
       has_column_privilege('pqxx', table_name, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'campaign approval generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'UPDATE') OR
       has_table_privilege('pqxx', table_name, 'DELETE') OR
       has_table_privilege('pqxx', table_name, 'TRUNCATE') THEN
        RAISE EXCEPTION 'campaign approval runtime mutation privilege present';
    END IF;
    IF NOT has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_campaign_approval_id'), 'USAGE') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_campaign_approval_id'), 'SELECT') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_campaign_approval_id'), 'UPDATE') THEN
        RAISE EXCEPTION 'campaign approval sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('S', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(pg_get_serial_sequence(
                  table_name, 'recommendation_campaign_approval_id'))
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'campaign approval PUBLIC sequence privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(table_name)
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'campaign approval PUBLIC table privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(table_name) AND NOT tgisinternal) THEN
        RAISE EXCEPTION 'campaign approval must not have mutation triggers';
    END IF;
END
$migration_test$;
