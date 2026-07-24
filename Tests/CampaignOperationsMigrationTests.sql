DO $$
DECLARE
    actual_owner text;
    actual_collation text;
    function_configuration text[];
BEGIN
    IF to_regclass('campaign_operations_campaign') IS NULL OR
       to_regclass('campaign_operations_governance_provenance_event') IS NULL OR
       to_regclass('campaign_operations_authorization_event') IS NULL OR
       to_regclass('campaign_operations_audit_reference_event') IS NULL THEN
        RAISE EXCEPTION 'campaign operations foundation tables missing';
    END IF;

    IF EXISTS (
        SELECT 1 FROM pg_roles
        WHERE rolname IN (
            'campaign_operations_owner',
            'campaign_operations_campaign_creator',
            'campaign_operations_authorizer',
            'campaign_operations_auditor',
            'campaign_operations_reader')
          AND (rolcanlogin OR rolsuper OR rolcreatedb OR rolcreaterole OR
               rolreplication OR rolbypassrls)) THEN
        RAISE EXCEPTION 'campaign operations capability role is not hardened';
    END IF;

    IF to_regclass('campaign_operations_current_projection') IS NOT NULL OR
       to_regclass('campaign_operations_campaign_readiness') IS NOT NULL THEN
        RAISE EXCEPTION 'prohibited campaign operations projection/readiness authority present';
    END IF;

    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = current_schema()
          AND table_name = 'campaign_operations_campaign'
          AND column_name IN ('status', 'state', 'drafted')) THEN
        RAISE EXCEPTION 'mutable or drafted campaign state present';
    END IF;

    SELECT tableowner INTO actual_owner
    FROM pg_tables
    WHERE schemaname = current_schema()
      AND tablename = 'campaign_operations_campaign';
    IF actual_owner <> 'campaign_operations_owner' THEN
        RAISE EXCEPTION 'campaign operations owner mismatch: %', actual_owner;
    END IF;

    SELECT collation_name INTO actual_collation
    FROM information_schema.columns
    WHERE table_schema = current_schema()
      AND table_name = 'campaign_operations_campaign'
      AND column_name = 'campaign_identity_canonical';
    IF actual_collation <> 'C' THEN
        RAISE EXCEPTION 'campaign canonical collation mismatch: %',
            actual_collation;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = 'campaign_operations_campaign'::regclass
          AND tgname = 'campaign_operations_campaign_binding_trigger'
          AND NOT tgisinternal) OR
       NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid =
            'campaign_operations_governance_provenance_event'::regclass
          AND tgname =
            'campaign_operations_governance_provenance_trigger'
          AND NOT tgisinternal) OR
       NOT EXISTS (
        SELECT 1 FROM pg_trigger
        WHERE tgrelid = 'campaign_operations_authorization_event'::regclass
          AND tgname = 'campaign_operations_authorization_chain_trigger'
          AND NOT tgisinternal) THEN
        RAISE EXCEPTION 'campaign operations provenance trigger missing';
    END IF;

    SELECT proconfig INTO function_configuration
    FROM pg_proc
    WHERE oid =
        'enforce_campaign_operations_authorization_chain()'::regprocedure;
    IF function_configuration IS NULL OR NOT EXISTS (
        SELECT 1 FROM unnest(function_configuration) setting
        WHERE setting = 'search_path=pg_catalog, ' || current_schema() ||
            ', pg_temp') THEN
        RAISE EXCEPTION 'authorization trigger search path not pinned: %',
            function_configuration;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid =
                'campaign_operations_authorization_event'::regclass
          AND conname =
                'campaign_operations_authorization_initial_kind_check'
          AND pg_get_constraintdef(oid) LIKE
                '%chain_version > 1%event_kind%granted%') THEN
        RAISE EXCEPTION 'initial authorization grant constraint missing';
    END IF;

    IF to_regclass(
            'campaign_operations_campaign_identity_uidx') IS NOT NULL OR
       to_regclass(
            'campaign_operations_provenance_identity_uidx') IS NOT NULL OR
       to_regclass(
            'campaign_operations_authorization_identity_uidx') IS NOT NULL OR
       EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid IN (
                'campaign_operations_campaign'::regclass,
                'campaign_operations_governance_provenance_event'::regclass,
                'campaign_operations_authorization_event'::regclass)
          AND conname IN (
                'campaign_operations_campaign_identity_uidx',
                'campaign_operations_provenance_identity_uidx',
                'campaign_operations_authorization_identity_uidx')) THEN
        RAISE EXCEPTION
            'removed direct canonical uniqueness object remains';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_campaign'::regclass,
                'campaign_identity_canonical'),
            ('campaign_operations_governance_provenance_event'::regclass,
                'provenance_identity_canonical'),
            ('campaign_operations_authorization_event'::regclass,
                'authorization_identity_canonical'))
            AS unsafe_key(table_oid, column_name)
        JOIN pg_index index_record
          ON index_record.indrelid = unsafe_key.table_oid
        JOIN pg_class index_class
          ON index_class.oid = index_record.indexrelid
        JOIN pg_am access_method
          ON access_method.oid = index_class.relam
        WHERE index_record.indisunique
          AND access_method.amname = 'btree'
          AND EXISTS (
              SELECT 1
              FROM unnest(index_record.indkey)
                  WITH ORDINALITY AS indexed_attribute(attnum, key_ordinal)
              JOIN pg_attribute attribute_record
                ON attribute_record.attrelid = index_record.indrelid
               AND attribute_record.attnum = indexed_attribute.attnum
              WHERE indexed_attribute.key_ordinal <=
                        index_record.indnkeyatts
                AND attribute_record.attname::text =
                        unsafe_key.column_name)) THEN
        RAISE EXCEPTION
            'authoritative canonical text remains a direct B-tree uniqueness key';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_campaign'::regclass,
                'campaign_operations_campaign_materialization_uidx',
                ARRAY['recommendation_campaign_materialization_id']),
            ('campaign_operations_governance_provenance_event'::regclass,
                'campaign_operations_provenance_ratification_uidx',
                ARRAY['operational_campaign_id',
                    'recommendation_campaign_follow_up_ratification_event_id']),
            ('campaign_operations_authorization_event'::regclass,
                'campaign_operations_authorization_chain_uidx',
                ARRAY['operational_campaign_id', 'action_kind',
                    'action_contract_version', 'scope_kind',
                    'scope_contract_version', 'chain_version']),
            ('campaign_operations_authorization_event'::regclass,
                'campaign_operations_authorization_previous_uidx',
                ARRAY['previous_event_id']))
            AS expected_constraint(
                table_oid, constraint_name, key_columns)
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_constraint constraint_record
            JOIN pg_index index_record
              ON index_record.indexrelid = constraint_record.conindid
            WHERE constraint_record.conrelid =
                      expected_constraint.table_oid
              AND constraint_record.conname =
                      expected_constraint.constraint_name
              AND constraint_record.contype = 'u'
              AND index_record.indisunique
              AND index_record.indisvalid
              AND index_record.indisready
              AND index_record.indpred IS NULL
              AND index_record.indexprs IS NULL
              AND ARRAY(
                  SELECT attribute_record.attname::text
                  FROM unnest(index_record.indkey)
                      WITH ORDINALITY AS indexed_attribute(
                          attnum, key_ordinal)
                  JOIN pg_attribute attribute_record
                    ON attribute_record.attrelid = index_record.indrelid
                   AND attribute_record.attnum =
                          indexed_attribute.attnum
                  WHERE indexed_attribute.key_ordinal <=
                            index_record.indnkeyatts
                  ORDER BY indexed_attribute.key_ordinal) =
                      expected_constraint.key_columns)) THEN
        RAISE EXCEPTION
            'required exact Campaign Operations natural uniqueness missing';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM (VALUES
            ('campaign_operations_campaign'::regclass,
                'campaign_operations_campaign_identity_hash_idx',
                'campaign_identity_hash'),
            ('campaign_operations_governance_provenance_event'::regclass,
                'campaign_operations_governance_provenance_identity_hash_idx',
                'provenance_identity_hash'),
            ('campaign_operations_authorization_event'::regclass,
                'campaign_operations_authorization_identity_hash_idx',
                'authorization_identity_hash'))
            AS expected_index(table_oid, index_name, key_column)
        WHERE NOT EXISTS (
            SELECT 1
            FROM pg_class index_class
            JOIN pg_index index_record
              ON index_record.indexrelid = index_class.oid
            JOIN pg_am access_method
              ON access_method.oid = index_class.relam
            WHERE index_record.indrelid = expected_index.table_oid
              AND index_class.relname = expected_index.index_name
              AND index_class.relnamespace =
                      current_schema()::regnamespace
              AND access_method.amname = 'btree'
              AND NOT index_record.indisunique
              AND index_record.indisvalid
              AND index_record.indisready
              AND index_record.indpred IS NULL
              AND index_record.indexprs IS NULL
              AND ARRAY(
                  SELECT attribute_record.attname::text
                  FROM unnest(index_record.indkey)
                      WITH ORDINALITY AS indexed_attribute(
                          attnum, key_ordinal)
                  JOIN pg_attribute attribute_record
                    ON attribute_record.attrelid = index_record.indrelid
                   AND attribute_record.attnum =
                          indexed_attribute.attnum
                  WHERE indexed_attribute.key_ordinal <=
                            index_record.indnkeyatts
                  ORDER BY indexed_attribute.key_ordinal) =
                      ARRAY[expected_index.key_column])) THEN
        RAISE EXCEPTION
            'canonical hash index definition mismatch';
    END IF;

    IF position('proposal.materialization_id' IN pg_get_functiondef(
            'enforce_campaign_operations_governance_provenance()'::
                regprocedure)) = 0 THEN
        RAISE EXCEPTION 'transitive Phase 6D materialization check missing';
    END IF;

    IF pg_get_functiondef(
            'enforce_campaign_operations_authorization_chain()'::
                regprocedure) !~*
            'provenance[.]prerequisite_policy[[:space:]]*=[[:space:]]*NEW[.]prerequisite_policy'
    THEN
        RAISE EXCEPTION
            'authorization provenance prerequisite-policy equality missing';
    END IF;

    IF has_table_privilege('pqxx', 'campaign_operations_campaign', 'SELECT') OR
       has_table_privilege('pqxx', 'campaign_operations_campaign', 'INSERT') OR
       has_table_privilege('pqxx',
           'campaign_operations_authorization_event', 'SELECT') OR
       has_table_privilege('pqxx',
           'campaign_operations_authorization_event', 'INSERT') THEN
        RAISE EXCEPTION 'foundation improperly enabled for pqxx';
    END IF;

    IF NOT has_table_privilege('campaign_operations_campaign_creator',
           'campaign_operations_campaign', 'SELECT') OR
       NOT has_column_privilege('campaign_operations_campaign_creator',
           'campaign_operations_campaign',
           'campaign_identity_canonical', 'INSERT') OR
       has_table_privilege('campaign_operations_campaign_creator',
           'campaign_operations_campaign', 'UPDATE') OR
       has_table_privilege('campaign_operations_campaign_creator',
           'campaign_operations_campaign', 'DELETE') OR
       has_table_privilege('campaign_operations_campaign_creator',
           'campaign_operations_campaign', 'TRUNCATE') THEN
        RAISE EXCEPTION 'campaign creator privileges invalid';
    END IF;

    IF NOT has_column_privilege('campaign_operations_authorizer',
           'campaign_operations_authorization_event',
           'authorization_identity_canonical', 'INSERT') OR
       has_table_privilege('campaign_operations_authorizer',
           'campaign_operations_authorization_event', 'UPDATE') OR
       has_table_privilege('campaign_operations_authorizer',
           'campaign_operations_authorization_event', 'DELETE') OR
       has_table_privilege('campaign_operations_authorizer',
           'campaign_operations_authorization_event', 'TRUNCATE') THEN
        RAISE EXCEPTION 'authorizer privileges invalid';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM pg_proc function_record
        CROSS JOIN LATERAL aclexplode(coalesce(function_record.proacl,
            acldefault('f', function_record.proowner))) acl
        WHERE function_record.oid =
            'enforce_campaign_operations_authorization_chain()'::regprocedure
          AND acl.grantee = 0
          AND acl.privilege_type = 'EXECUTE') OR
       has_function_privilege('pqxx',
           'enforce_campaign_operations_authorization_chain()', 'EXECUTE') THEN
        RAISE EXCEPTION 'trigger function execution not revoked';
    END IF;
END $$;

DO $$
BEGIN
    BEGIN
        INSERT INTO campaign_operations_authorization_event (
            operational_campaign_id,campaign_identity_canonical,
            chain_version,event_kind,action_kind,action_contract_version,
            scope_kind,scope_contract_version,prerequisite_policy,
            authorization_role,actor_identity,reason,not_before,
            authorization_contract_version,authorization_identity_canonical,
            authorization_identity_hash)
        VALUES (1,'invalid',1,'superseded','dispatch_full_materialization',1,
            'complete_materialization',1,
            'phase4d_materialization_only_v1',
            'campaign_operations_authorizer','actor','reason',
            '2026-07-22 00:00:00+00',1,'invalid',
            'fnv1a64:0000000000000000');
        RAISE EXCEPTION 'superseded authorization event unexpectedly accepted';
    EXCEPTION WHEN check_violation OR foreign_key_violation THEN
        NULL;
    END;
END $$;
