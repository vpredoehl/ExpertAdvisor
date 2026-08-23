\set ON_ERROR_STOP on

CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    core_lr_mult double precision,
    head_lr_mult double precision,
    target_epochs integer NOT NULL,
    checkpoint_interval integer NOT NULL,
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    resume_model_id bigint,
    donchian20_mode text NOT NULL DEFAULT 'enabled',
    donchian_lookback integer NOT NULL DEFAULT 20,
    feature_warmup_scope text NOT NULL DEFAULT 'legacy_cold_boundary',
    feature_ablation_mask text NOT NULL DEFAULT '',
    resume_expand_input_width boolean NOT NULL DEFAULT false,
    duplicate_nonce bigint NOT NULL DEFAULT 0,
    status text NOT NULL DEFAULT 'pending'
);

CREATE UNIQUE INDEX experiment_unique_identity_uidx
    ON experiment (experiment_id);

INSERT INTO experiment (
    symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,
    target_epochs,checkpoint_interval,train_start,train_end,infer_start,
    infer_end,resume_model_id)
VALUES (
    'eurusdrmp',6,0.0008,120,25,60,20,
    '2010-01-01','2025-01-01','2025-01-01','2026-01-01',NULL);

\ir ../Database/migrations/079_training_objective_provenance.sql
\ir ../Database/migrations/079_training_objective_provenance.sql

DO $$
DECLARE
    historical experiment%ROWTYPE;
BEGIN
    SELECT * INTO STRICT historical FROM experiment WHERE experiment_id=1;
    IF historical.training_objective_id <>
           'legacy_first_hit_weighted_ce_v1' OR
       historical.training_objective_version <> 1 OR
       historical.loss_definition_version <> 1 OR
       historical.training_objective_hash <>
           'fnv1a64:65818f2e1fa1a324' OR
       historical.training_objective_hash <>
           training_objective_tagged_fnv1a64(
               historical.training_objective_canonical) OR
       historical.auxiliary_loss_mode <> 'disabled' OR
       historical.auxiliary_loss_coefficient <> 0.0 OR
       historical.regression_target_definition IS NOT NULL OR
       historical.robust_loss_definition IS NOT NULL OR
       historical.robust_loss_delta IS NOT NULL
    THEN
        RAISE EXCEPTION 'historical objective did not resolve to legacy';
    END IF;
END;
$$;

-- A materially different objective identity must not collide with the
-- historical/default row even when every pre-Phase-4A identity field matches.
INSERT INTO experiment (
    symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,
    target_epochs,checkpoint_interval,train_start,train_end,infer_start,
    infer_end,resume_model_id,training_objective_id,
    training_objective_version,loss_definition_version,
    training_objective_canonical,training_objective_hash)
SELECT
    symbol,prediction_horizon,c_next_threshold,core_lr_mult,head_lr_mult,
    target_epochs,checkpoint_interval,train_start,train_end,infer_start,
    infer_end,resume_model_id,'fixture_changed_objective_v1',1,1,
    'fixture_changed_objective_canonical_v1',
    training_objective_tagged_fnv1a64(
        'fixture_changed_objective_canonical_v1')
FROM experiment WHERE experiment_id=1;

DO $$
BEGIN
    IF (SELECT count(*) FROM experiment) <> 2 THEN
        RAISE EXCEPTION 'objective-aware experiment identity collision';
    END IF;
END;
$$;
