\set ON_ERROR_STOP on

DO $$
DECLARE
    identity_index text;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='experiment'
          AND column_name='feature_ablation_mask'
          AND data_type='text' AND is_nullable='NO'
          AND column_default=$default$''::text$default$
    ) THEN
        RAISE EXCEPTION 'feature_ablation_mask must be text NOT NULL DEFAULT empty';
    END IF;
    IF EXISTS (
        SELECT 1 FROM experiment WHERE feature_ablation_mask <> ''
    ) AND NOT EXISTS (
        SELECT 1 FROM experiment WHERE feature_ablation_mask = ''
    ) THEN
        RAISE EXCEPTION 'migration did not retain a no-ablation compatibility default';
    END IF;
    SELECT pg_get_indexdef(indexrelid) INTO identity_index
    FROM pg_index
    WHERE indexrelid='experiment_unique_identity_uidx'::regclass;
    IF identity_index IS NULL OR
       identity_index NOT LIKE '%donchian20_mode%' OR
       identity_index NOT LIKE '%feature_warmup_scope%' OR
       identity_index NOT LIKE '%donchian_lookback%' OR
       identity_index NOT LIKE '%feature_ablation_mask%' THEN
        RAISE EXCEPTION 'experiment identity index is missing persisted configuration fields: %', identity_index;
    END IF;
    IF NOT has_table_privilege('pqxx', 'experiment', 'SELECT') OR
       NOT has_table_privilege('pqxx', 'experiment', 'INSERT') OR
       NOT has_table_privilege('pqxx', 'experiment', 'UPDATE') THEN
        RAISE EXCEPTION 'pqxx experiment role grants are insufficient';
    END IF;
END $$;
