\set ON_ERROR_STOP on

BEGIN;
SET TRANSACTION READ WRITE;
CREATE SCHEMA step5_review_migration_test;
SET LOCAL search_path TO step5_review_migration_test, public;

CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY
);
CREATE TABLE experiment_recommendation_scan (
    recommendation_scan_id bigserial PRIMARY KEY
);
CREATE TABLE experiment_recommendation (
    recommendation_id bigserial PRIMARY KEY,
    recommendation_scan_id bigint REFERENCES experiment_recommendation_scan,
    status text NOT NULL,
    source_experiment_id bigint REFERENCES experiment,
    semantic_configuration_canonical text,
    semantic_hash text,
    policy_canonical text,
    policy_hash text,
    approved_experiment_id bigint REFERENCES experiment,
    rejected_at timestamptz,
    rejected_reason text,
    expired_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT experiment_recommendation_step3_status_shape_check CHECK (
        status <> 'approved' OR approved_experiment_id IS NOT NULL),
    CONSTRAINT experiment_recommendation_status_metadata_symmetric_check CHECK (
        status <> 'approved' OR approved_experiment_id IS NOT NULL)
);
CREATE TABLE experiment_recommendation_score_run (
    recommendation_score_run_id bigserial PRIMARY KEY,
    status text NOT NULL
);
CREATE TABLE experiment_recommendation_score (
    recommendation_score_id bigserial PRIMARY KEY,
    recommendation_score_run_id bigint NOT NULL
        REFERENCES experiment_recommendation_score_run,
    recommendation_id bigint NOT NULL REFERENCES experiment_recommendation
);

\ir ../Database/migrations/032_experiment_recommendation_review.sql
\ir ../Database/migrations/032_experiment_recommendation_review.sql
\ir ../Database/migrations/033_experiment_recommendation_review_permissions.sql
\ir ../Database/migrations/033_experiment_recommendation_review_permissions.sql

INSERT INTO experiment DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation (
    recommendation_scan_id,status,source_experiment_id,
    semantic_configuration_canonical,semantic_hash,
    policy_canonical,policy_hash)
VALUES
    (1,'proposed',1,'semantic_one','hash_one','policy_one','policy_hash_one'),
    (1,'proposed',1,'semantic_two','hash_two','policy_two','policy_hash_two');
INSERT INTO experiment_recommendation_score_run (status) VALUES ('completed');
INSERT INTO experiment_recommendation_score (
    recommendation_score_run_id,recommendation_id)
VALUES (1,1);

DO $$
DECLARE
    observed_sqlstate text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_review_event (
            recommendation_id,action,previous_status,resulting_status,
            reason_code,recommendation_semantic_canonical,
            recommendation_semantic_hash,recommendation_policy_canonical,
            recommendation_policy_hash,recommendation_scan_id,
            source_experiment_id)
        VALUES (1,'approve','proposed','rejected','operator_approved',
            'semantic_one','hash_one','policy_one','policy_hash_one',1,1);
        RAISE EXCEPTION 'action_result_mismatch_accepted';
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS observed_sqlstate = RETURNED_SQLSTATE;
        IF observed_sqlstate <> '23514' THEN
            RAISE EXCEPTION 'action_result_mismatch_sqlstate_%', observed_sqlstate;
        END IF;
    END;
    BEGIN
        INSERT INTO experiment_recommendation_review_event (
            recommendation_id,action,previous_status,resulting_status,
            reason_code,recommendation_semantic_canonical,
            recommendation_semantic_hash,recommendation_policy_canonical,
            recommendation_policy_hash,recommendation_scan_id,
            source_experiment_id)
        VALUES (1,'reject','proposed','rejected','operator_rejected',
            'semantic_one','hash_one','policy_one','policy_hash_one',1,1);
        RAISE EXCEPTION 'rejection_without_reason_accepted';
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS observed_sqlstate = RETURNED_SQLSTATE;
        IF observed_sqlstate <> '23514' THEN
            RAISE EXCEPTION 'rejection_reason_sqlstate_%', observed_sqlstate;
        END IF;
    END;
    BEGIN
        INSERT INTO experiment_recommendation_review_event (
            recommendation_id,action,previous_status,resulting_status,
            reason_code,reason_text,recommendation_semantic_canonical,
            recommendation_semantic_hash,recommendation_policy_canonical,
            recommendation_policy_hash,recommendation_scan_id,
            source_experiment_id)
        VALUES (1,'expire','proposed','expired','operator_expired','  ',
            'semantic_one','hash_one','policy_one','policy_hash_one',1,1);
        RAISE EXCEPTION 'expiration_without_reason_accepted';
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS observed_sqlstate = RETURNED_SQLSTATE;
        IF observed_sqlstate <> '23514' THEN
            RAISE EXCEPTION 'expiration_reason_sqlstate_%', observed_sqlstate;
        END IF;
    END;
    BEGIN
        INSERT INTO experiment_recommendation_review_event (
            recommendation_id,recommendation_score_id,action,previous_status,
            resulting_status,reason_code,recommendation_semantic_canonical,
            recommendation_semantic_hash,recommendation_policy_canonical,
            recommendation_policy_hash,recommendation_scan_id,
            source_experiment_id)
        VALUES (2,1,'approve','proposed','approved','operator_approved',
            'semantic_two','hash_two','policy_two','policy_hash_two',1,1);
        RAISE EXCEPTION 'score_ownership_mismatch_accepted';
    EXCEPTION WHEN OTHERS THEN
        GET STACKED DIAGNOSTICS observed_sqlstate = RETURNED_SQLSTATE;
        IF observed_sqlstate <> '23503' THEN
            RAISE EXCEPTION 'score_ownership_sqlstate_%', observed_sqlstate;
        END IF;
    END;
END $$;

INSERT INTO experiment_recommendation_review_event (
    recommendation_id,recommendation_score_id,action,previous_status,
    resulting_status,reason_code,recommendation_semantic_canonical,
    recommendation_semantic_hash,recommendation_policy_canonical,
    recommendation_policy_hash,recommendation_scan_id,source_experiment_id)
VALUES (1,1,'approve','proposed','approved','operator_approved',
    'semantic_one','hash_one','policy_one','policy_hash_one',1,1);

DO $$
BEGIN
    IF (SELECT count(*) FROM information_schema.tables
        WHERE table_schema = 'step5_review_migration_test'
          AND table_name = 'experiment_recommendation_review_event') <> 1 THEN
        RAISE EXCEPTION 'review_event_table_missing';
    END IF;
    IF (SELECT count(*) FROM pg_constraint
        WHERE conrelid = 'experiment_recommendation_review_event'::regclass
          AND conname IN (
              'experiment_recommendation_review_event_result_check',
              'experiment_recommendation_review_event_reason_check',
              'experiment_recommendation_review_event_score_owner_fkey',
              'experiment_recommendation_review_event_one_per_recommendation')) <> 4 THEN
        RAISE EXCEPTION 'review_event_constraints_missing';
    END IF;
    IF EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_recommendation_review_event'::regclass
          AND contype = 'f' AND confdeltype <> 'a'
    ) THEN
        RAISE EXCEPTION 'review_event_foreign_key_not_restrictive';
    END IF;
    IF EXISTS (
        SELECT 1 FROM information_schema.columns
        WHERE table_schema = 'step5_review_migration_test'
          AND table_name = 'experiment'
          AND column_name = 'approved_at'
    ) THEN
        RAISE EXCEPTION 'experiment_schema_changed';
    END IF;
    IF NOT has_table_privilege(
        'pqxx', 'experiment_recommendation_review_event', 'SELECT')
       OR NOT has_table_privilege(
        'pqxx', 'experiment_recommendation_review_event', 'INSERT') THEN
        RAISE EXCEPTION 'review_event_runtime_read_insert_privileges_missing';
    END IF;
    IF has_table_privilege(
        'pqxx', 'experiment_recommendation_review_event', 'UPDATE')
       OR has_table_privilege(
        'pqxx', 'experiment_recommendation_review_event', 'DELETE') THEN
        RAISE EXCEPTION 'review_event_runtime_mutation_privilege_present';
    END IF;
END $$;

ROLLBACK;
