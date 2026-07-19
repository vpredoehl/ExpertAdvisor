DO $migration_test$
DECLARE
    table_name text := format(
        '%I.experiment_recommendation_conversion_execution', current_schema());
BEGIN
    IF to_regclass(table_name) IS NULL THEN
        RAISE EXCEPTION 'conversion execution table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname = 'conversion_execution_proposal_fkey'
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'conversion execution proposal FK not restrictive';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname = 'conversion_execution_review_proposal_fkey'
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'conversion execution review FK not restrictive';
    END IF;
    IF (
        SELECT count(*) FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND contype = 'u') <> 2 THEN
        RAISE EXCEPTION 'conversion execution uniqueness constraints missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_conversion_execution'
          AND indexname IN (
              'conversion_execution_review_idx',
              'conversion_execution_hash_idx')) <> 2 THEN
        RAISE EXCEPTION 'conversion execution indexes missing';
    END IF;
    IF NOT has_table_privilege('pqxx', table_name, 'SELECT') THEN
        RAISE EXCEPTION 'conversion execution runtime SELECT missing';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(table_name)
          AND privilege.grantee = 0
    ) THEN
        RAISE EXCEPTION 'conversion execution PUBLIC privilege present';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_proposal_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_review_decision_id', 'INSERT') OR
       NOT has_column_privilege('pqxx', table_name, 'experiment_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'execution_contract_version', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'authorization_decision', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'execution_identity_canonical', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'execution_identity_hash', 'INSERT') THEN
        RAISE EXCEPTION 'conversion execution payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_execution_id', 'INSERT') OR
       has_column_privilege('pqxx', table_name, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'conversion execution generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'UPDATE') OR
       has_table_privilege('pqxx', table_name, 'DELETE') OR
       has_table_privilege('pqxx', table_name, 'TRUNCATE') THEN
        RAISE EXCEPTION 'conversion execution runtime mutation present';
    END IF;
    IF NOT has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_execution_id'),
           'USAGE') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_execution_id'),
           'SELECT') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_execution_id'),
           'UPDATE') THEN
        RAISE EXCEPTION 'conversion execution sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('S', relation.relowner))) privilege
        WHERE relation.oid = pg_get_serial_sequence(
            table_name,
            'recommendation_conversion_execution_id')::regclass
          AND privilege.grantee = 0
    ) THEN
        RAISE EXCEPTION 'conversion execution PUBLIC sequence privilege present';
    END IF;
    IF has_table_privilege(
            'pqxx',
            format('%I.experiment_recommendation_conversion_proposal',
                   current_schema()),
            'UPDATE') THEN
        RAISE EXCEPTION 'conversion proposal UPDATE unexpectedly granted';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(table_name) AND NOT tgisinternal) THEN
        RAISE EXCEPTION 'conversion execution must not have mutation triggers';
    END IF;
END
$migration_test$;
