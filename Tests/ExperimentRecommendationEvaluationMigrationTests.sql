\set ON_ERROR_STOP on

BEGIN;
SET TRANSACTION READ WRITE;
CREATE SCHEMA phase4b_evaluation_migration_test;
SET LOCAL search_path TO phase4b_evaluation_migration_test, public;

CREATE TABLE experiment (experiment_id bigserial PRIMARY KEY);
CREATE TABLE model (model_id bigserial PRIMARY KEY);
CREATE TABLE experiment_analysis_result (analysis_id bigserial PRIMARY KEY);
CREATE TABLE experiment_recommendation_scan (
    recommendation_scan_id bigserial PRIMARY KEY);
CREATE TABLE experiment_recommendation (
    recommendation_id bigserial PRIMARY KEY,
    recommendation_scan_id bigint REFERENCES experiment_recommendation_scan,
    source_experiment_id bigint REFERENCES experiment,
    source_analysis_id bigint REFERENCES experiment_analysis_result);

\ir ../Database/migrations/034_experiment_recommendation_evaluation.sql
\ir ../Database/migrations/034_experiment_recommendation_evaluation.sql
\ir ../Database/migrations/066_phase4b_canonical_identity_btree_scale.sql
\ir ../Database/migrations/066_phase4b_canonical_identity_btree_scale.sql

DO $$
DECLARE
    owner_name text := current_user;
BEGIN
    IF to_regclass('experiment_recommendation_evaluation_run') IS NULL
       OR to_regclass('experiment_recommendation_evaluation_result') IS NULL
       OR to_regclass('experiment_recommendation_evaluation_component') IS NULL THEN
        RAISE EXCEPTION 'evaluation_tables_missing';
    END IF;
    IF has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_result','UPDATE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_result','DELETE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_component','UPDATE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_component','DELETE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_run','DELETE') THEN
        RAISE EXCEPTION 'runtime_evaluation_history_is_not_append_only';
    END IF;
    IF NOT has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_result','SELECT,INSERT')
       OR NOT has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_component','SELECT,INSERT')
       OR NOT has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_run','SELECT,INSERT')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_evaluation_run','UPDATE')
       OR NOT has_column_privilege('pqxx',
           'experiment_recommendation_evaluation_run','status','UPDATE')
       OR has_column_privilege('pqxx',
           'experiment_recommendation_evaluation_run',
           'evaluation_policy_canonical','UPDATE') THEN
        RAISE EXCEPTION 'runtime_evaluation_privileges_incomplete';
    END IF;
    IF (SELECT count(*) FROM information_schema.table_constraints
        WHERE table_schema=current_schema()
          AND table_name LIKE 'experiment_recommendation_evaluation_%'
          AND constraint_type IN ('PRIMARY KEY','FOREIGN KEY','UNIQUE','CHECK')) < 20 THEN
        RAISE EXCEPTION 'evaluation_constraints_incomplete';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conrelid =
              'experiment_recommendation_evaluation_result'::regclass
          AND confrelid = 'model'::regclass
          AND contype = 'f') THEN
        RAISE EXCEPTION 'evaluation_source_model_foreign_key_missing';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid IN (
            'experiment_recommendation_evaluation_run'::regclass,
            'experiment_recommendation_evaluation_result'::regclass)
          AND conname IN (
            'experiment_recommendation_evaluation_run_identity_uidx',
            'experiment_recommendation_evaluation_result_identity_uidx')) THEN
        RAISE EXCEPTION 'unsafe_evaluation_canonical_btree_constraint_present';
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_indexes
        WHERE schemaname=current_schema()
          AND indexname='experiment_recommendation_evaluation_result_hash_idx') THEN
        RAISE EXCEPTION 'evaluation_hash_lookup_index_missing';
    END IF;
END $$;

INSERT INTO experiment DEFAULT VALUES;
INSERT INTO model DEFAULT VALUES;
INSERT INTO experiment_analysis_result DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation(
    recommendation_scan_id,source_experiment_id,source_analysis_id)
VALUES (1,1,1);
INSERT INTO experiment_recommendation(
    recommendation_scan_id,source_experiment_id,source_analysis_id)
VALUES (1,1,1);

INSERT INTO experiment_recommendation_evaluation_run(
    status,evaluation_run_identity_canonical,evaluation_run_identity_hash,
    evaluation_policy_canonical,evaluation_policy_hash,evaluation_version,
    evaluator_version,scoring_policy_canonical,scoring_policy_hash,
    scoring_version,evidence_snapshot_canonical,evidence_snapshot_hash)
VALUES ('running','run_identity','run_hash','evaluation_policy','evaluation_hash',
        1,1,'scoring_policy','scoring_hash',1,'snapshot','snapshot_hash');

INSERT INTO experiment_recommendation_evaluation_result(
    recommendation_evaluation_run_id,recommendation_id,
    evaluation_identity_canonical,evaluation_identity_hash,
    recommendation_semantic_canonical,recommendation_semantic_hash,
    recommendation_policy_canonical,recommendation_policy_hash,
    recommendation_scan_id,source_experiment_id,source_model_id,source_analysis_id,
    evidence_canonical,evidence_hash,eligibility,disposition,reason_code,
    explanation,final_score,raw_positive_score,raw_penalty_score,
    raw_total_score,component_count,missing_evidence_count,ranking_ordinal)
VALUES (1,1,'evaluation_identity','evaluation_hash','semantic','semantic_hash',
        'policy','policy_hash',1,1,1,1,'evidence','evidence_hash','ineligible',
        'insufficient_evidence','missing','Missing evidence.',NULL,NULL,NULL,NULL,
        0,1,1);

DO $$
DECLARE
    observed text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,evidence_canonical,
            evidence_hash,eligibility,disposition,reason_code,explanation,
            component_count,missing_evidence_count,ranking_ordinal)
        VALUES (1,2,'bad','bad','semantic','hash','policy','hash',1,1,
                'evidence','hash','eligible','advisory_ready','bad','bad',0,0,2);
        RAISE EXCEPTION 'eligible_result_without_score_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE
    observed text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,source_model_id,
            source_analysis_id,evidence_canonical,evidence_hash,eligibility,
            disposition,reason_code,explanation,component_count,
            missing_evidence_count,ranking_ordinal)
        VALUES (1,2,'bad_model','bad_model_hash','semantic_2','hash_2',
                'policy','hash',1,1,999999,1,'evidence','hash','ineligible',
                'insufficient_evidence','missing','Missing evidence.',0,1,2);
        RAISE EXCEPTION 'missing_source_model_was_accepted';
    EXCEPTION WHEN foreign_key_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23503' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE
    observed text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,source_model_id,
            source_analysis_id,evidence_canonical,evidence_hash,eligibility,
            disposition,reason_code,explanation,final_score,
            raw_positive_score,raw_penalty_score,raw_total_score,
            component_count,missing_evidence_count,ranking_ordinal)
        VALUES (1,2,'nonfinite','nonfinite_hash','semantic_2','hash_2',
                'policy','hash',1,1,1,1,'evidence','hash','eligible',
                'advisory_ready','ready','Ready.','NaN'::double precision,
                1,0,1,1,0,2);
        RAISE EXCEPTION 'nonfinite_score_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed = RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
BEGIN
    IF (SELECT count(*) FROM experiment) <> 1
       OR (SELECT count(*) FROM model) <> 1
       OR (SELECT count(*) FROM experiment_recommendation) <> 2
       OR (SELECT count(*) FROM experiment_analysis_result) <> 1 THEN
        RAISE EXCEPTION 'source_tables_modified';
    END IF;
END $$;

ROLLBACK;
