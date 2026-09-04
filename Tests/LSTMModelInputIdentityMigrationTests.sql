DO $$
DECLARE
    fixture_id bigint;
    index_columns text[];
    index_predicate text;
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='experiment'
          AND column_name='model_input_width') OR
       NOT EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema='public' AND table_name='experiment'
          AND column_name='model_input_semantic_layout_version')
    THEN
        RAISE EXCEPTION 'model input identity columns missing';
    END IF;

    INSERT INTO experiment(
        symbol,prediction_horizon,c_next_threshold,target_epochs,
        checkpoint_interval,train_start,train_end,status,phase,
        duplicate_nonce,donchian20_mode,feature_warmup_scope,
        donchian_lookback,feature_ablation_mask,
        model_input_width,model_input_semantic_layout_version)
    VALUES(
        'phase20migration',4,0.0008,1,0,'2020-01-01','2020-01-02',
        'pending','train',2089001,'enabled','legacy_cold_boundary',20,'',
        75,5)
    RETURNING experiment_id INTO fixture_id;

    -- Phase 2 uses the same migration-089 durable identity columns for the
    -- append-only width-77/layout-6 generation. Historical width 75/layout 5
    -- remains independently identifiable.
    INSERT INTO experiment(
        symbol,prediction_horizon,c_next_threshold,target_epochs,
        checkpoint_interval,train_start,train_end,status,phase,
        duplicate_nonce,donchian20_mode,feature_warmup_scope,
        donchian_lookback,feature_ablation_mask,
        model_input_width,model_input_semantic_layout_version)
    VALUES(
        'phase2causalsurprise',4,0.0008,1,0,'2020-01-01','2020-01-02',
        'pending','train',2091001,'enabled','legacy_cold_boundary',20,'',
        77,6);
    IF NOT EXISTS (
        SELECT 1 FROM experiment
        WHERE symbol='phase2causalsurprise'
          AND model_input_width=77
          AND model_input_semantic_layout_version=6
    ) THEN
        RAISE EXCEPTION 'width-77 layout-6 identity was not persisted';
    END IF;

    BEGIN
        UPDATE experiment SET model_input_width=71
        WHERE experiment_id=fixture_id;
        RAISE EXCEPTION 'model input identity mutation unexpectedly succeeded';
    EXCEPTION
        WHEN raise_exception THEN
            IF SQLERRM <> 'experiment model input identity is immutable' THEN
                RAISE;
            END IF;
    END;

    BEGIN
        INSERT INTO experiment(
            symbol,prediction_horizon,c_next_threshold,target_epochs,
            checkpoint_interval,train_start,train_end,status,phase,
            duplicate_nonce,donchian20_mode,feature_warmup_scope,
            donchian_lookback,feature_ablation_mask,
            model_input_width,model_input_semantic_layout_version)
        VALUES(
            'phase20malformed',4,0.0008,1,0,'2020-01-01','2020-01-02',
            'pending','train',2089002,'enabled','legacy_cold_boundary',20,'',
            75,NULL);
        RAISE EXCEPTION 'partial model input identity unexpectedly succeeded';
    EXCEPTION
        WHEN raise_exception THEN
            IF SQLERRM <> 'new experiment requires complete model input identity' THEN
                RAISE;
            END IF;
    END;

    BEGIN
        INSERT INTO experiment(
            symbol,prediction_horizon,c_next_threshold,target_epochs,
            checkpoint_interval,train_start,train_end,status,phase,
            duplicate_nonce,donchian20_mode,feature_warmup_scope,
            donchian_lookback,feature_ablation_mask)
        VALUES(
            'phase20missing',4,0.0008,1,0,'2020-01-01','2020-01-02',
            'pending','train',2089003,'enabled','legacy_cold_boundary',20,'');
        RAISE EXCEPTION 'missing model input identity unexpectedly succeeded';
    EXCEPTION
        WHEN raise_exception THEN
            IF SQLERRM <> 'new experiment requires complete model input identity' THEN
                RAISE;
            END IF;
    END;

    -- Width and semantic layout are both duplicate-identity dimensions. The
    -- otherwise identical rows below must coexist; an exact replay must not.
    INSERT INTO experiment(
        symbol,prediction_horizon,c_next_threshold,target_epochs,
        checkpoint_interval,train_start,train_end,status,phase,
        duplicate_nonce,donchian20_mode,feature_warmup_scope,
        donchian_lookback,feature_ablation_mask,
        model_input_width,model_input_semantic_layout_version)
    VALUES
        ('phase20identity',4,0.0008,1,0,'2020-01-01','2020-01-02',
         'pending','train',2089004,'enabled','legacy_cold_boundary',20,'',71,5),
        ('phase20identity',4,0.0008,1,0,'2020-01-01','2020-01-02',
         'pending','train',2089004,'enabled','legacy_cold_boundary',20,'',75,5),
        ('phase20identity',4,0.0008,1,0,'2020-01-01','2020-01-02',
         'pending','train',2089004,'enabled','legacy_cold_boundary',20,'',75,4);

    BEGIN
        INSERT INTO experiment(
            symbol,prediction_horizon,c_next_threshold,target_epochs,
            checkpoint_interval,train_start,train_end,status,phase,
            duplicate_nonce,donchian20_mode,feature_warmup_scope,
            donchian_lookback,feature_ablation_mask,
            model_input_width,model_input_semantic_layout_version)
        VALUES(
            'phase20identity',4,0.0008,1,0,'2020-01-01','2020-01-02',
            'pending','train',2089004,'enabled','legacy_cold_boundary',20,'',
            75,5);
        RAISE EXCEPTION 'exact duplicate identity unexpectedly succeeded';
    EXCEPTION
        WHEN unique_violation THEN NULL;
    END;

    SELECT array_agg(pg_get_indexdef(i.indexrelid, key_position, true)
                     ORDER BY key_position),
           pg_get_expr(i.indpred, i.indrelid, true)
      INTO index_columns, index_predicate
      FROM pg_index i
      JOIN pg_class c ON c.oid=i.indexrelid
      CROSS JOIN LATERAL generate_series(1, i.indnkeyatts) key_position
     WHERE c.relname='experiment_unique_identity_uidx'
     GROUP BY i.indpred,i.indrelid;
    IF index_columns IS DISTINCT FROM ARRAY[
        'symbol','prediction_horizon','c_next_threshold',
        'COALESCE(core_lr_mult, ''-Infinity''::double precision)',
        'COALESCE(head_lr_mult, ''-Infinity''::double precision)',
        'target_epochs','checkpoint_interval','train_start','train_end',
        'COALESCE(infer_start, ''-infinity''::timestamp with time zone)',
        'COALESCE(infer_end, ''-infinity''::timestamp with time zone)',
        'COALESCE(resume_model_id, ''-1''::integer::bigint)','donchian20_mode',
        'donchian_lookback','feature_warmup_scope','feature_ablation_mask',
        'resume_expand_input_width','training_objective_hash',
        'COALESCE(model_input_width, ''-1''::integer)',
        'COALESCE(model_input_semantic_layout_version, ''-1''::integer)',
        'duplicate_nonce']::text[] OR
       index_predicate IS DISTINCT FROM 'status <> ''cancelled''::text'
    THEN
        RAISE EXCEPTION 'experiment unique identity definition mismatch: % / %',
            index_columns,index_predicate;
    END IF;
END;
$$;
