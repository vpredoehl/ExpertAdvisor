DO $migration_test$
DECLARE
    table_name text := format(
        '%I.experiment_recommendation_conversion_review_decision',
        current_schema());
BEGIN
    IF to_regclass(table_name) IS NULL THEN
        RAISE EXCEPTION 'conversion review decision table missing';
    END IF;
    IF (
        SELECT count(*)
        FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname IN (
              'conversion_review_decision_value_check',
              'conversion_review_request_id_check',
              'conversion_review_operator_check',
              'conversion_review_reason_check',
              'conversion_review_proposal_request_uidx')
    ) <> 5 THEN
        RAISE EXCEPTION 'conversion review constraints missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND contype = 'f'
          AND confrelid =
              'experiment_recommendation_conversion_proposal'::regclass
          AND confdeltype = 'r'
    ) THEN
        RAISE EXCEPTION 'conversion review proposal foreign key not restrictive';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute attribute
        JOIN pg_collation c
          ON c.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(table_name)
          AND attribute.attname = 'decision_request_id'
          AND NOT attribute.attisdropped
          AND c.collname = 'C'
    ) THEN
        RAISE EXCEPTION 'conversion review request identity is not byte-exact';
    END IF;
    IF (
        SELECT count(*)
        FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_conversion_review_decision'
          AND indexname IN (
              'conversion_review_proposal_history_idx',
              'conversion_review_decision_idx')
    ) <> 2 THEN
        RAISE EXCEPTION 'conversion review indexes missing';
    END IF;
    IF NOT has_table_privilege('pqxx', table_name, 'SELECT') THEN
        RAISE EXCEPTION 'conversion review runtime select privilege missing';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_proposal_id', 'INSERT') OR
       NOT has_column_privilege('pqxx', table_name, 'decision', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'decision_request_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'operator_identity', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'reason_text', 'INSERT') THEN
        RAISE EXCEPTION 'conversion review runtime insert columns missing';
    END IF;
    IF has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_review_decision_id', 'INSERT') OR
       has_column_privilege('pqxx', table_name, 'decided_at', 'INSERT') OR
       has_column_privilege('pqxx', table_name, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'conversion review generated columns are writable';
    END IF;
    IF NOT has_sequence_privilege(
        'pqxx', pg_get_serial_sequence(
            table_name, 'recommendation_conversion_review_decision_id'),
        'USAGE') THEN
        RAISE EXCEPTION 'conversion review sequence usage privilege missing';
    END IF;
    IF has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_review_decision_id'),
           'SELECT') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_review_decision_id'),
           'UPDATE') THEN
        RAISE EXCEPTION 'conversion review sequence privilege too broad';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(table_name)
          AND privilege.grantee = 0
    ) THEN
        RAISE EXCEPTION 'conversion review PUBLIC table privilege present';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('S', relation.relowner))) privilege
        WHERE relation.oid = pg_get_serial_sequence(
            table_name,
            'recommendation_conversion_review_decision_id')::regclass
          AND privilege.grantee = 0
    ) THEN
        RAISE EXCEPTION 'conversion review PUBLIC sequence privilege present';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'UPDATE') OR
       has_table_privilege('pqxx', table_name, 'DELETE') OR
       has_table_privilege('pqxx', table_name, 'TRUNCATE') THEN
        RAISE EXCEPTION 'conversion review runtime mutation privilege present';
    END IF;
    IF has_table_privilege(
        'pqxx',
        format('%I.experiment_recommendation_conversion_proposal',
               current_schema()),
        'UPDATE') OR
       has_table_privilege(
        'pqxx',
        format('%I.experiment_recommendation_conversion_proposal',
               current_schema()),
        'DELETE') OR
       has_table_privilege(
        'pqxx',
        format('%I.experiment_recommendation_conversion_proposal',
               current_schema()),
        'TRUNCATE') THEN
        RAISE EXCEPTION 'conversion proposal runtime mutation privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(table_name) AND NOT tgisinternal
    ) THEN
        RAISE EXCEPTION 'conversion review must not trigger source mutation';
    END IF;
END
$migration_test$;
