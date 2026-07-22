DO $migration_test$
DECLARE
    proposal_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal',
        current_schema());
    member_table text := format(
        '%I.experiment_recommendation_campaign_follow_up_proposal_member',
        current_schema());
    proposal_sequence text;
    member_sequence text;
BEGIN
    IF to_regclass(proposal_table) IS NULL OR
       to_regclass(member_table) IS NULL THEN
        RAISE EXCEPTION 'follow-up proposal tables missing';
    END IF;

    IF (
        SELECT count(*) FROM pg_constraint
        WHERE conrelid = to_regclass(proposal_table)
          AND contype = 'f'
          AND confdeltype = 'r') <> 2 THEN
        RAISE EXCEPTION 'follow-up proposal restrictive provenance FKs missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(member_table)
          AND contype = 'f'
          AND confrelid = to_regclass(proposal_table)
          AND confdeltype = 'r') THEN
        RAISE EXCEPTION 'follow-up proposal member restrictive FK missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = to_regclass(proposal_table)
          AND conname =
              'recommendation_campaign_follow_up_proposal_hash_ordinal_uidx')
    THEN
        RAISE EXCEPTION 'follow-up proposal collision uniqueness missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_campaign_follow_up_proposal'
          AND indexname IN (
              'recommendation_campaign_follow_up_proposal_identity_hash_idx',
              'recommendation_campaign_follow_up_proposal_identity_canonical_idx',
              'recommendation_campaign_follow_up_proposal_campaign_idx',
              'recommendation_campaign_follow_up_proposal_materialization_idx'))
       <> 4 THEN
        RAISE EXCEPTION 'follow-up proposal indexes missing';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname = current_schema()
          AND tablename =
              'experiment_recommendation_campaign_follow_up_proposal'
          AND indexname =
              'recommendation_campaign_follow_up_proposal_identity_canonical_idx'
          AND indexdef LIKE
              '%USING hash (proposal_identity_canonical)%') THEN
        RAISE EXCEPTION 'follow-up proposal canonical hash index invalid';
    END IF;
    IF (
        SELECT count(*) FROM pg_trigger
        WHERE tgrelid IN (
                  to_regclass(proposal_table), to_regclass(member_table))
          AND NOT tgisinternal
          AND tgname IN (
              'enforce_recommendation_campaign_follow_up_proposal_provenance_trigger',
              'enforce_recommendation_campaign_follow_up_proposal_member_provenance_trigger',
              'enforce_recommendation_campaign_follow_up_proposal_complete_trigger'))
       <> 3 THEN
        RAISE EXCEPTION 'follow-up proposal provenance/completeness triggers missing';
    END IF;
    IF (
        SELECT count(*) FROM pg_attribute attribute
        JOIN pg_collation coll ON coll.oid = attribute.attcollation
        WHERE attribute.attrelid = to_regclass(proposal_table)
          AND attribute.attname IN (
              'proposal_identity_canonical',
              'assessment_identity_canonical',
              'policy_identity_canonical',
              'policy_decision_identity_canonical',
              'campaign_identity_canonical',
              'materialization_identity_canonical')
          AND coll.collname = 'C') <> 6 THEN
        RAISE EXCEPTION 'follow-up proposal canonicals are not byte-exact';
    END IF;

    IF NOT has_table_privilege('pqxx', proposal_table, 'SELECT') OR
       NOT has_table_privilege('pqxx', member_table, 'SELECT') THEN
        RAISE EXCEPTION 'follow-up proposal runtime SELECT missing';
    END IF;
    IF has_table_privilege('pqxx', proposal_table, 'INSERT') OR
       has_table_privilege('pqxx', member_table, 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal runtime table INSERT too broad';
    END IF;
    IF NOT has_column_privilege(
            'pqxx', proposal_table, 'proposal_identity_canonical', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', proposal_table, 'follow_up_authorized', 'INSERT') OR
       NOT has_column_privilege(
            'pqxx', member_table, 'member_ordinal', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal payload INSERT missing';
    END IF;
    IF has_column_privilege(
            'pqxx', proposal_table,
            'recommendation_campaign_follow_up_proposal_id', 'INSERT') OR
       has_column_privilege(
            'pqxx', proposal_table, 'created_at', 'INSERT') OR
       has_column_privilege(
            'pqxx', member_table,
            'recommendation_campaign_follow_up_proposal_member_id',
            'INSERT') OR
       has_column_privilege(
            'pqxx', member_table, 'created_at', 'INSERT') THEN
        RAISE EXCEPTION 'follow-up proposal generated columns writable';
    END IF;
    IF has_table_privilege('pqxx', proposal_table, 'UPDATE') OR
       has_table_privilege('pqxx', proposal_table, 'DELETE') OR
       has_table_privilege('pqxx', proposal_table, 'TRUNCATE') OR
       has_table_privilege('pqxx', member_table, 'UPDATE') OR
       has_table_privilege('pqxx', member_table, 'DELETE') OR
       has_table_privilege('pqxx', member_table, 'TRUNCATE') THEN
        RAISE EXCEPTION 'follow-up proposal runtime mutation privilege present';
    END IF;
    IF has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_follow_up_proposal_provenance()',
           'EXECUTE') OR
       has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_follow_up_proposal_member_provenance()',
           'EXECUTE') OR
       has_function_privilege(
           'pqxx',
           'enforce_recommendation_campaign_follow_up_proposal_complete()',
           'EXECUTE') THEN
        RAISE EXCEPTION 'follow-up proposal trigger function privilege present';
    END IF;

    proposal_sequence := pg_get_serial_sequence(
        proposal_table, 'recommendation_campaign_follow_up_proposal_id');
    member_sequence := pg_get_serial_sequence(
        member_table,
        'recommendation_campaign_follow_up_proposal_member_id');
    IF NOT has_sequence_privilege('pqxx', proposal_sequence, 'USAGE') OR
       has_sequence_privilege('pqxx', proposal_sequence, 'SELECT') OR
       has_sequence_privilege('pqxx', proposal_sequence, 'UPDATE') OR
       NOT has_sequence_privilege('pqxx', member_sequence, 'USAGE') OR
       has_sequence_privilege('pqxx', member_sequence, 'SELECT') OR
       has_sequence_privilege('pqxx', member_sequence, 'UPDATE') THEN
        RAISE EXCEPTION 'follow-up proposal sequence privilege incorrect';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_class relation
        CROSS JOIN LATERAL aclexplode(coalesce(
            relation.relacl, acldefault('r', relation.relowner))) privilege
        WHERE relation.oid IN (
                  to_regclass(proposal_table), to_regclass(member_table))
          AND privilege.grantee = 0) THEN
        RAISE EXCEPTION 'follow-up proposal PUBLIC table privilege present';
    END IF;
END
$migration_test$;
