\set ON_ERROR_STOP on

BEGIN;
CREATE SCHEMA recommendation_phase3a_profitability_test;
SET LOCAL search_path TO recommendation_phase3a_profitability_test, public;

CREATE TABLE experiment (
    experiment_id bigint PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    last_model_id bigint
);
CREATE TABLE model (
    model_id bigint PRIMARY KEY,
    experiment_id bigint REFERENCES experiment(experiment_id)
);
CREATE TABLE matrix (
    model_id bigint NOT NULL REFERENCES model(model_id),
    param_name text NOT NULL,
    n_rows integer NOT NULL,
    n_cols integer NOT NULL,
    row_idx integer NOT NULL,
    col_idx integer NOT NULL,
    value double precision NOT NULL,
    PRIMARY KEY(model_id,param_name,row_idx,col_idx)
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigint PRIMARY KEY,
    parent_experiment_id bigint REFERENCES experiment(experiment_id),
    checkpoint_model_id bigint REFERENCES model(model_id)
);
CREATE TABLE inference_eval_result (
    id bigint PRIMARY KEY,
    model_id bigint NOT NULL REFERENCES model(model_id),
    status text NOT NULL,
    inference_scope text NOT NULL,
    checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval,
    parent_experiment_id bigint REFERENCES experiment,
    checkpoint_epoch integer,
    symbol text NOT NULL,
    prediction_horizon bigint NOT NULL,
    threshold_logret double precision NOT NULL,
    window_size bigint NOT NULL,
    label_rule_id integer NOT NULL,
    target_type integer NOT NULL,
    from_date text NOT NULL,
    to_date text NOT NULL,
    completed_epochs bigint
);
CREATE UNIQUE INDEX inference_eval_result_final_completed_uidx
    ON inference_eval_result(
        model_id,symbol,prediction_horizon,threshold_logret,window_size,
        label_rule_id,target_type,from_date,to_date)
    WHERE status='completed' AND inference_scope='final';
CREATE TABLE experiment_recommendation (
    recommendation_id bigint PRIMARY KEY,
    source_experiment_id bigint NOT NULL REFERENCES experiment,
    source_model_id bigint REFERENCES model
);
CREATE TABLE experiment_recommendation_evaluation_result (
    recommendation_evaluation_result_id bigint PRIMARY KEY,
    source_experiment_id bigint NOT NULL REFERENCES experiment,
    source_model_id bigint REFERENCES model
);

INSERT INTO experiment VALUES
    (1,'USDJPYRMP',5,0.001,'2025-01-01','2026-01-01',10),
    (2,'EURUSDRMP',5,0.002,'2025-02-01','2026-02-01',20);
INSERT INTO model VALUES (10,1),(11,1),(20,2);
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT fixture.model_id,'train_config_meta',1,14,0,col_idx,
       CASE col_idx
           WHEN 1 THEN 5
           WHEN 2 THEN fixture.threshold_logret
           WHEN 3 THEN 64
           WHEN 4 THEN 1
           WHEN 10 THEN 100
           ELSE 0
       END
FROM (VALUES (10,0.001::double precision),
             (20,0.002::double precision)) fixture(model_id,threshold_logret)
CROSS JOIN generate_series(0,13) col_idx;
INSERT INTO matrix VALUES
    (10,'target_meta',1,1,0,0,1),
    (20,'target_meta',1,1,0,0,1);
INSERT INTO experiment_checkpoint_eval VALUES (200,1,10);
INSERT INTO inference_eval_result(
    id,model_id,status,inference_scope,checkpoint_eval_id,
    parent_experiment_id,checkpoint_epoch,symbol,prediction_horizon,
    threshold_logret,window_size,label_rule_id,target_type,from_date,to_date,
    completed_epochs)
VALUES
    (100,10,'completed','final',NULL,NULL,NULL,'USDJPYRMP',5,0.001,
     64,1,1,'2025-01-01','2026-01-01',100),
    (101,10,'completed','checkpoint',200,1,100,'USDJPYRMP',5,0.001,
     64,1,1,'2025-01-01','2026-01-01',100),
    (102,10,'completed','final',NULL,NULL,NULL,'USDJPYRMP',5,0.001,
     64,1,1,'2024-01-01','2024-12-31',100),
    (200,20,'completed','final',NULL,NULL,NULL,'EURUSDRMP',5,0.002,
     64,1,1,'2025-02-01','2026-02-01',100);

-- This row predates Phase 3A and must remain honestly legacy.
INSERT INTO experiment_recommendation VALUES (1,1,10);

\ir ../Database/migrations/073_inference_profitability_observation.sql
\ir ../Database/migrations/076_campaign_manager_final_profitability_provenance.sql
\ir ../Database/migrations/076_campaign_manager_final_profitability_provenance.sql

INSERT INTO inference_profitability_observation(
    profitability_observation_id,experiment_id,model_id,
    inference_eval_result_id,inference_scope,checkpoint_eval_id,
    inference_start,inference_end,prediction_count,actionable_count,
    winning_actionable_count,losing_actionable_count,
    gross_positive_terminal_horizon_log_return_sum,
    gross_negative_terminal_horizon_log_return_sum,
    aggregate_terminal_horizon_log_return_sum,
    average_terminal_horizon_log_return_per_actionable_prediction,
    metric_definition_canonical,metric_definition_hash,source_content_hash,
    observation_identity_canonical,observation_identity_hash)
VALUES
    (1000,1,10,100,'final',NULL,'2025-01-01','2026-01-01',100,100,
     60,40,12,-2,10,0.1,'metric-v1','fnv1a64:1111111111111111',
     'fnv1a64:2222222222222222','positive-final',
     'fnv1a64:3333333333333333'),
    (1001,1,10,101,'checkpoint',200,'2025-01-01','2026-01-01',100,100,
     40,60,2,-12,-10,-0.1,'metric-v1','fnv1a64:1111111111111111',
     'fnv1a64:4444444444444444','checkpoint',
     'fnv1a64:5555555555555555'),
    (1002,1,10,100,'final',NULL,'2025-01-01','2026-01-01',100,0,
     0,0,0,0,0,NULL,'metric-v1','fnv1a64:1111111111111111',
     'fnv1a64:6666666666666666','zero-actionable-final',
     'fnv1a64:7777777777777777'),
    (1004,1,10,102,'final',NULL,'2024-01-01','2024-12-31',100,100,
     50,50,8,-3,5,0.05,'metric-v1','fnv1a64:1111111111111111',
     'fnv1a64:bbbbbbbbbbbbbbbb','wrong-range-final',
     'fnv1a64:cccccccccccccccc');

INSERT INTO experiment_recommendation(
    recommendation_id,source_experiment_id,source_model_id,
    final_profitability_provenance_version,
    source_final_inference_eval_result_id,
    source_final_profitability_observation_id,
    source_final_profitability_inference_scope,
    source_final_profitability_inference_start,
    source_final_profitability_inference_end,
    source_final_profitability_actionable_count,
    source_final_profitability_aggregate_return,
    source_final_profitability_average_return,
    source_final_profitability_metric_definition_hash,
    source_final_profitability_source_content_hash,
    source_final_profitability_observation_identity_hash)
VALUES
    (2,1,10,1,100,1000,'final','2025-01-01','2026-01-01',100,
     10,0.1,'fnv1a64:1111111111111111','fnv1a64:2222222222222222',
     'fnv1a64:3333333333333333'),
    (4,1,10,1,100,1002,'final','2025-01-01','2026-01-01',0,
     0,NULL,'fnv1a64:1111111111111111','fnv1a64:6666666666666666',
     'fnv1a64:7777777777777777');

INSERT INTO experiment_recommendation(
    recommendation_id,source_experiment_id,source_model_id,
    final_profitability_provenance_version,
    source_final_inference_eval_result_id,
    source_final_profitability_unavailable_reason,
    source_final_profitability_inference_scope)
VALUES (3,1,10,1,100,'no_profitability_observation','final');

INSERT INTO experiment_recommendation_evaluation_result(
    recommendation_evaluation_result_id,source_experiment_id,source_model_id,
    final_profitability_provenance_version,
    source_final_inference_eval_result_id,
    source_final_profitability_observation_id,
    source_final_profitability_inference_scope,
    source_final_profitability_inference_start,
    source_final_profitability_inference_end,
    source_final_profitability_actionable_count,
    source_final_profitability_aggregate_return,
    source_final_profitability_average_return,
    source_final_profitability_metric_definition_hash,
    source_final_profitability_source_content_hash,
    source_final_profitability_observation_identity_hash,
    profitability_evidence_canonical,profitability_evidence_hash)
VALUES (1,1,10,1,100,1000,'final','2025-01-01','2026-01-01',100,
        10,0.1,'fnv1a64:1111111111111111','fnv1a64:2222222222222222',
        'fnv1a64:3333333333333333','observed-positive-final',
        'fnv1a64:8888888888888888');

DO $$
DECLARE observed text;
BEGIN
    IF (SELECT final_profitability_provenance_version
        FROM experiment_recommendation WHERE recommendation_id=1) IS NOT NULL
    THEN RAISE EXCEPTION 'legacy_recommendation_was_backfilled'; END IF;
    IF (SELECT source_final_inference_eval_result_id
        FROM experiment_recommendation WHERE recommendation_id=2) <> 100
       OR (SELECT source_final_profitability_observation_id
           FROM experiment_recommendation WHERE recommendation_id=2) <> 1000
    THEN RAISE EXCEPTION 'available_observation_not_frozen'; END IF;
    IF (SELECT source_final_inference_eval_result_id
        FROM experiment_recommendation WHERE recommendation_id=3) <> 100
       OR (SELECT source_final_profitability_observation_id
           FROM experiment_recommendation WHERE recommendation_id=3) IS NOT NULL
       OR (SELECT source_final_profitability_unavailable_reason
           FROM experiment_recommendation WHERE recommendation_id=3) <>
              'no_profitability_observation'
    THEN RAISE EXCEPTION 'missing_profitability_not_explicit'; END IF;
    IF (SELECT source_final_profitability_actionable_count
        FROM experiment_recommendation WHERE recommendation_id=4) <> 0
       OR (SELECT source_final_profitability_observation_id
           FROM experiment_recommendation WHERE recommendation_id=4) IS NULL
       OR (SELECT source_final_profitability_average_return
           FROM experiment_recommendation WHERE recommendation_id=4) IS NOT NULL
    THEN RAISE EXCEPTION 'zero_actionable_not_distinct'; END IF;

    -- A checkpoint result cannot be frozen even when the unavailable shape
    -- does not attach a profitability observation.
    BEGIN
        INSERT INTO experiment_recommendation(
            recommendation_id,source_experiment_id,source_model_id,
            final_profitability_provenance_version,
            source_final_inference_eval_result_id,
            source_final_profitability_unavailable_reason,
            source_final_profitability_inference_scope)
        VALUES (5,1,10,1,101,'checkpoint_result_is_not_final','final');
        RAISE EXCEPTION 'checkpoint_final_result_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    -- Model 20 and result 200 are a valid completed FINAL pair for experiment
    -- B, but cannot be frozen for recommendation source experiment A. This is
    -- also the critical no-observation unavailable-profitability regression.
    BEGIN
        INSERT INTO experiment_recommendation(
            recommendation_id,source_experiment_id,source_model_id,
            final_profitability_provenance_version,
            source_final_inference_eval_result_id,
            source_final_profitability_unavailable_reason,
            source_final_profitability_inference_scope)
        VALUES (6,1,20,1,200,'no_profitability_observation','final');
        RAISE EXCEPTION 'wrong_experiment_final_result_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    -- Result 102 is completed FINAL evidence for the correct experiment/model,
    -- but its date range is not experiment A's configured FINAL range. It must
    -- fail independently of profitability-observation availability.
    BEGIN
        INSERT INTO experiment_recommendation(
            recommendation_id,source_experiment_id,source_model_id,
            final_profitability_provenance_version,
            source_final_inference_eval_result_id,
            source_final_profitability_unavailable_reason,
            source_final_profitability_inference_scope)
        VALUES (7,1,10,1,102,'no_profitability_observation','final');
        RAISE EXCEPTION 'wrong_range_unavailable_final_result_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    -- Attaching an internally consistent observation to the wrong-range FINAL
    -- row cannot turn that row into the exact source inference result.
    BEGIN
        INSERT INTO experiment_recommendation(
            recommendation_id,source_experiment_id,source_model_id,
            final_profitability_provenance_version,
            source_final_inference_eval_result_id,
            source_final_profitability_observation_id,
            source_final_profitability_inference_scope,
            source_final_profitability_inference_start,
            source_final_profitability_inference_end,
            source_final_profitability_actionable_count,
            source_final_profitability_aggregate_return,
            source_final_profitability_average_return,
            source_final_profitability_metric_definition_hash,
            source_final_profitability_source_content_hash,
            source_final_profitability_observation_identity_hash)
        VALUES (8,1,10,1,102,1004,'final','2024-01-01','2024-12-31',100,
                5,0.05,'fnv1a64:1111111111111111',
                'fnv1a64:bbbbbbbbbbbbbbbb',
                'fnv1a64:cccccccccccccccc');
        RAISE EXCEPTION 'wrong_range_available_final_result_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    BEGIN
        UPDATE experiment_recommendation
        SET source_final_profitability_aggregate_return=11
        WHERE recommendation_id=2;
        RAISE EXCEPTION 'profitability_snapshot_was_mutable';
    EXCEPTION WHEN object_not_in_prerequisite_state THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '55000' THEN RAISE; END IF;
    END;
END $$;

-- A later immutable observation for the same FINAL result cannot reinterpret
-- the already-frozen recommendation or evaluation provenance.
INSERT INTO inference_profitability_observation(
    profitability_observation_id,experiment_id,model_id,
    inference_eval_result_id,inference_scope,checkpoint_eval_id,
    inference_start,inference_end,prediction_count,actionable_count,
    winning_actionable_count,losing_actionable_count,
    gross_positive_terminal_horizon_log_return_sum,
    gross_negative_terminal_horizon_log_return_sum,
    aggregate_terminal_horizon_log_return_sum,
    average_terminal_horizon_log_return_per_actionable_prediction,
    metric_definition_canonical,metric_definition_hash,source_content_hash,
    observation_identity_canonical,observation_identity_hash)
VALUES (1003,1,10,100,'final',NULL,'2025-01-01','2026-01-01',100,100,
        100,0,20,0,20,0.2,'metric-v1','fnv1a64:1111111111111111',
        'fnv1a64:9999999999999999','later-final',
        'fnv1a64:aaaaaaaaaaaaaaaa');

DO $$
BEGIN
    IF (SELECT source_final_profitability_observation_id
        FROM experiment_recommendation WHERE recommendation_id=2) <> 1000
       OR (SELECT source_final_profitability_aggregate_return
           FROM experiment_recommendation WHERE recommendation_id=2) <> 10
       OR (SELECT source_final_profitability_observation_id
           FROM experiment_recommendation_evaluation_result
           WHERE recommendation_evaluation_result_id=1) <> 1000
    THEN RAISE EXCEPTION 'historical_profitability_was_reinterpreted'; END IF;
END $$;

ROLLBACK;
