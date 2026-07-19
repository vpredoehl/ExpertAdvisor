DO $migration_test$
DECLARE
    table_name text := format(
        '%I.experiment_recommendation_conversion_proposal', current_schema());
BEGIN
    IF to_regclass(table_name) IS NULL THEN
        RAISE EXCEPTION 'conversion proposal table missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND conname = 'recommendation_conversion_proposal_hash_ordinal_uidx'
          AND contype = 'u'
    ) THEN
        RAISE EXCEPTION 'hash collision ordinal uniqueness missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename = 'experiment_recommendation_conversion_proposal'
          AND indexname = 'recommendation_conversion_proposal_canonical_idx'
    ) THEN
        RAISE EXCEPTION 'exact canonical lookup index missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_attribute attribute
        JOIN pg_collation c
          ON c.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(table_name)
          AND attribute.attname = 'conversion_identity_canonical'
          AND NOT attribute.attisdropped
          AND c.collname = 'C'
    ) THEN
        RAISE EXCEPTION 'canonical identity equality is not byte-exact';
    END IF;
    IF NOT has_table_privilege('pqxx', table_name, 'SELECT') OR
       NOT has_table_privilege('pqxx', table_name, 'INSERT') THEN
        RAISE EXCEPTION 'runtime conversion privileges missing';
    END IF;
    IF has_table_privilege('pqxx', table_name, 'UPDATE') OR
       has_table_privilege('pqxx', table_name, 'DELETE') THEN
        RAISE EXCEPTION 'runtime conversion history is mutable';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = to_regclass(table_name) AND NOT tgisinternal
    ) THEN
        RAISE EXCEPTION 'conversion proposal must not trigger source mutation';
    END IF;
    IF (
        SELECT count(*)
        FROM pg_constraint
        WHERE conrelid = to_regclass(table_name)
          AND contype = 'f'
          AND confdeltype = 'r'
    ) <> 3 THEN
        RAISE EXCEPTION 'conversion proposal foreign keys are not restrictive';
    END IF;
END
$migration_test$;
