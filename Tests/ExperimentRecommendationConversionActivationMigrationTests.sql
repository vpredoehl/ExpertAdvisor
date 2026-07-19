DO $migration_test$
DECLARE
    table_name text := format(
        '%I.experiment_recommendation_conversion_activation',
        current_schema());
BEGIN
    IF to_regclass(table_name) IS NULL THEN
        RAISE EXCEPTION 'conversion activation table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname = 'conversion_activation_execution_provenance_fkey'
          AND confdeltype = 'r'
          AND array_length(conkey, 1) = 4) THEN
        RAISE EXCEPTION 'conversion activation provenance FK invalid';
    END IF;
    IF (
        SELECT count(*) FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND contype = 'u') <> 2 THEN
        RAISE EXCEPTION 'conversion activation uniqueness missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_conversion_activation'
          AND indexname IN (
              'conversion_activation_proposal_idx',
              'conversion_activation_review_idx',
              'conversion_activation_hash_idx')) <> 3 THEN
        RAISE EXCEPTION 'conversion activation indexes missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_conversion_execution'
          AND indexname = 'conversion_execution_provenance_uidx') THEN
        RAISE EXCEPTION 'conversion execution provenance index missing';
    END IF;
    IF NOT has_table_privilege('pqxx', table_name, 'SELECT') THEN
        RAISE EXCEPTION 'conversion activation runtime SELECT missing';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_execution_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_proposal_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_review_decision_id', 'INSERT') OR
       NOT has_column_privilege('pqxx', table_name, 'experiment_id', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'activation_contract_version', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'previous_status', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'previous_phase', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'resulting_status', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'resulting_phase', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'activation_identity_canonical', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', table_name, 'activation_identity_hash', 'INSERT') THEN
        RAISE EXCEPTION 'conversion activation payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', table_name,
            'recommendation_conversion_activation_id', 'INSERT') OR
       has_column_privilege('pqxx', table_name, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'conversion activation generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'UPDATE') OR
       has_table_privilege('pqxx', table_name, 'DELETE') OR
       has_table_privilege('pqxx', table_name, 'TRUNCATE') THEN
        RAISE EXCEPTION 'conversion activation runtime mutation present';
    END IF;
    IF NOT has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_activation_id'),
           'USAGE') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_activation_id'),
           'SELECT') OR
       has_sequence_privilege(
           'pqxx', pg_get_serial_sequence(
               table_name, 'recommendation_conversion_activation_id'),
           'UPDATE') THEN
        RAISE EXCEPTION 'conversion activation sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid = to_regclass(table_name)
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'conversion activation PUBLIC table privilege present';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('S', relation.relowner))) privilege
        WHERE relation.oid = pg_get_serial_sequence(
            table_name,
            'recommendation_conversion_activation_id')::regclass
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'conversion activation PUBLIC sequence privilege present';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(table_name) AND NOT tgisinternal) THEN
        RAISE EXCEPTION 'conversion activation must not have mutation triggers';
    END IF;
END
$migration_test$;
