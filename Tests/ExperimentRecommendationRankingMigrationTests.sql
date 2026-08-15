\set ON_ERROR_STOP on

BEGIN;
SET TRANSACTION READ WRITE;
CREATE SCHEMA phase4b_ranking_migration_test;
SET LOCAL search_path TO phase4b_ranking_migration_test, public;

CREATE TABLE experiment (experiment_id bigserial PRIMARY KEY);
CREATE TABLE model (model_id bigserial PRIMARY KEY);
CREATE TABLE experiment_analysis_result (analysis_id bigserial PRIMARY KEY);
CREATE TABLE experiment_recommendation_scan (
    recommendation_scan_id bigserial PRIMARY KEY);
CREATE TABLE experiment_recommendation (
    recommendation_id bigserial PRIMARY KEY,
    recommendation_scan_id bigint REFERENCES experiment_recommendation_scan,
    source_experiment_id bigint REFERENCES experiment,
    source_analysis_id bigint REFERENCES experiment_analysis_result,
    source_symbol text NOT NULL DEFAULT 'EURUSD',
    source_prediction_horizon integer NOT NULL DEFAULT 12,
    changed_parameter text NOT NULL DEFAULT 'core_lr_mult',
    source_value_canonical text NOT NULL DEFAULT '1',
    proposed_value_canonical text NOT NULL DEFAULT '1.25');

\ir ../Database/migrations/034_experiment_recommendation_evaluation.sql
\ir ../Database/migrations/035_experiment_recommendation_ranking.sql
\ir ../Database/migrations/035_experiment_recommendation_ranking.sql
\ir ../Database/migrations/066_phase4b_canonical_identity_btree_scale.sql
\ir ../Database/migrations/066_phase4b_canonical_identity_btree_scale.sql
GRANT USAGE ON SCHEMA phase4b_ranking_migration_test TO pqxx;
GRANT SELECT ON experiment_recommendation,
    experiment_recommendation_evaluation_result TO pqxx;

DO $$
BEGIN
    IF to_regclass('experiment_recommendation_ranking_snapshot') IS NULL
       OR to_regclass('experiment_recommendation_ranking_member') IS NULL THEN
        RAISE EXCEPTION 'ranking_tables_missing';
    END IF;
    IF has_table_privilege('pqxx',
           'experiment_recommendation_ranking_snapshot','UPDATE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_ranking_snapshot','DELETE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_ranking_member','UPDATE')
       OR has_table_privilege('pqxx',
           'experiment_recommendation_ranking_member','DELETE') THEN
        RAISE EXCEPTION 'runtime_ranking_history_is_not_append_only';
    END IF;
    IF NOT has_table_privilege('pqxx',
           'experiment_recommendation_ranking_snapshot','SELECT,INSERT')
       OR NOT has_table_privilege('pqxx',
           'experiment_recommendation_ranking_member','SELECT,INSERT')
       OR NOT has_column_privilege('pqxx',
           'experiment_recommendation_ranking_snapshot','status','UPDATE')
       OR has_column_privilege('pqxx',
           'experiment_recommendation_ranking_snapshot',
           'ranking_policy_canonical','UPDATE') THEN
        RAISE EXCEPTION 'runtime_ranking_privileges_incomplete';
    END IF;
    IF (SELECT count(*) FROM information_schema.table_constraints
        WHERE table_schema=current_schema()
          AND table_name LIKE 'experiment_recommendation_ranking_%'
          AND constraint_type IN ('PRIMARY KEY','FOREIGN KEY','UNIQUE','CHECK'))
       < 18 THEN
        RAISE EXCEPTION 'ranking_constraints_incomplete';
    END IF;
    IF (SELECT count(*) FROM pg_trigger
        WHERE tgrelid='experiment_recommendation_ranking_member'::regclass
          AND tgname='enforce_recommendation_ranking_member_insert_trigger'
          AND NOT tgisinternal)<>1 THEN
        RAISE EXCEPTION 'ranking_member_enforcement_trigger_count_invalid';
    END IF;
    IF (SELECT count(*) FROM pg_trigger
        WHERE tgrelid='experiment_recommendation_ranking_snapshot'::regclass
          AND tgname='enforce_recommendation_ranking_snapshot_transition_trigger'
          AND NOT tgisinternal)<>1 THEN
        RAISE EXCEPTION 'ranking_snapshot_transition_trigger_count_invalid';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE oid='enforce_recommendation_ranking_member_insert()'::regprocedure
          AND array_to_string(proconfig, ',') LIKE
              '%search_path=pg_catalog, phase4b_ranking_migration_test, pg_temp%'
    ) THEN
        RAISE EXCEPTION 'ranking_trigger_search_path_not_pinned';
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE oid=
            'enforce_recommendation_ranking_snapshot_transition()'::regprocedure
          AND array_to_string(proconfig, ',') LIKE
              '%search_path=pg_catalog, phase4b_ranking_migration_test, pg_temp%'
    ) THEN
        RAISE EXCEPTION 'ranking_transition_search_path_not_pinned';
    END IF;
    IF EXISTS (
        SELECT 1
        FROM pg_proc
        WHERE oid IN (
            'enforce_recommendation_ranking_member_insert()'::regprocedure,
            'enforce_recommendation_ranking_snapshot_transition()'::regprocedure,
            'recommendation_ranking_membership_contains(text,text,bigint)'::regprocedure,
            'recommendation_ranking_tie_break_matches(text,text,double precision)'::regprocedure)
          AND prosecdef
    ) THEN
        RAISE EXCEPTION 'ranking_trigger_function_is_security_definer';
    END IF;
END $$;

DO $$
DECLARE
    injected_identity text :=
        'prefix;member[0].evaluation_identity=12:evaluation_2;member[0].evaluation_result_id=2;suffix';
    injected_membership text;
    unicode_identity text := 'évaluation';
    unicode_membership text;
BEGIN
    IF recommendation_ranking_membership_contains(
        'experiment_recommendation_ranking_membership_v1;count=1;member[0].evaluation_identity=10:evaluation;member[0].evaluation_result_id=10',
        'evaluation', 1) THEN
        RAISE EXCEPTION 'membership_result_id_prefix_was_accepted';
    END IF;
    injected_membership :=
        'experiment_recommendation_ranking_membership_v1;count=1;' ||
        'member[0].evaluation_identity=' || octet_length(injected_identity) ||
        ':' || injected_identity || ';member[0].evaluation_result_id=11';
    IF recommendation_ranking_membership_contains(
        injected_membership, 'evaluation_2', 2) THEN
        RAISE EXCEPTION 'membership_identity_delimiter_injection_was_accepted';
    END IF;
    unicode_membership :=
        'experiment_recommendation_ranking_membership_v1;count=1;' ||
        'member[0].evaluation_identity=' || octet_length(unicode_identity) ||
        ':' || unicode_identity || ';member[0].evaluation_result_id=42';
    IF NOT recommendation_ranking_membership_contains(
        unicode_membership, unicode_identity, 42) THEN
        RAISE EXCEPTION 'valid_byte_length_membership_was_rejected';
    END IF;
    IF recommendation_ranking_membership_contains(
        unicode_membership || ';trailing', unicode_identity, 42) THEN
        RAISE EXCEPTION 'malformed_membership_suffix_was_accepted';
    END IF;
    IF NOT recommendation_ranking_tie_break_matches(
        'score:0.80000000000000004', 'advisory_ready', 0.8)
       OR recommendation_ranking_tie_break_matches(
        'score:0.7', 'advisory_ready', 0.8)
       OR NOT recommendation_ranking_tie_break_matches(
        'priority:1', 'stale_source_evidence', NULL)
       OR recommendation_ranking_tie_break_matches(
        'priority:0', 'stale_source_evidence', NULL) THEN
        RAISE EXCEPTION 'ranking_tie_break_validation_invalid';
    END IF;
END $$;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_snapshot(
            recommendation_ranking_snapshot_id,status,
            ranking_snapshot_identity_canonical,ranking_snapshot_identity_hash,
            ranking_policy_canonical,ranking_policy_hash,ranking_version,
            scope_type,scope_canonical,scope_hash,requested_limit,
            source_membership_canonical,source_membership_hash,completed_at)
        VALUES (9000,'completed','terminal_insert','terminal_insert_hash',
                'ranking_policy','ranking_policy_hash',1,'global',
                'global_scope','global_scope_hash',100,
                'experiment_recommendation_ranking_membership_v1;count=0',
                'empty_membership_hash',now());
        RAISE EXCEPTION 'terminal_ranking_snapshot_insert_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

INSERT INTO experiment DEFAULT VALUES;
INSERT INTO model DEFAULT VALUES;
INSERT INTO experiment_analysis_result DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation(
    recommendation_scan_id,source_experiment_id,source_analysis_id)
VALUES (1,1,1),(1,1,1);
INSERT INTO experiment_recommendation_evaluation_run(
    status,evaluation_run_identity_canonical,evaluation_run_identity_hash,
    evaluation_policy_canonical,evaluation_policy_hash,evaluation_version,
    evaluator_version,scoring_policy_canonical,scoring_policy_hash,
    scoring_version,evidence_snapshot_canonical,evidence_snapshot_hash,
    completed_at)
VALUES ('completed','run','run_hash','ep','ep_hash',1,1,'sp','sp_hash',1,
        'evidence','evidence_hash',now());
INSERT INTO experiment_recommendation_evaluation_result(
    recommendation_evaluation_run_id,recommendation_id,
    evaluation_identity_canonical,evaluation_identity_hash,
    recommendation_semantic_canonical,recommendation_semantic_hash,
    recommendation_policy_canonical,recommendation_policy_hash,
    recommendation_scan_id,source_experiment_id,source_model_id,
    source_analysis_id,evidence_canonical,evidence_hash,eligibility,
    disposition,reason_code,explanation,final_score,raw_positive_score,
    raw_penalty_score,raw_total_score,component_count,missing_evidence_count,
    ranking_ordinal)
VALUES (1,1,'evaluation','evaluation_hash','semantic','semantic_hash',
        'policy','policy_hash',1,1,1,1,'evidence','evidence_hash','eligible',
        'advisory_ready','ready','Ready.',0.8,1.0,0.2,0.8,1,0,1);
INSERT INTO experiment_recommendation_evaluation_component(
    recommendation_evaluation_result_id,component_ordinal,component_name,
    reason_code,input_canonical,normalized_value,weight,weighted_contribution,
    is_penalty,is_missing,explanation)
VALUES (1,1,'leader_quality','quality','input',0.8,0.2,0.16,false,false,
        'Quality.');
INSERT INTO experiment_recommendation_evaluation_result(
    recommendation_evaluation_run_id,recommendation_id,
    evaluation_identity_canonical,evaluation_identity_hash,
    recommendation_semantic_canonical,recommendation_semantic_hash,
    recommendation_policy_canonical,recommendation_policy_hash,
    recommendation_scan_id,source_experiment_id,evidence_canonical,
    evidence_hash,eligibility,disposition,reason_code,explanation,
    component_count,missing_evidence_count,ranking_ordinal)
VALUES (1,2,'evaluation_2','evaluation_hash_2','semantic_2','semantic_hash_2',
        'policy','policy_hash',1,1,'evidence_2','evidence_hash_2','ineligible',
        'stale_source_evidence','stale','Stale.',0,0,2);

INSERT INTO experiment_recommendation_ranking_snapshot(
    status,ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,ranking_policy_canonical,
    ranking_policy_hash,ranking_version,scope_type,scope_canonical,scope_hash,
    evaluation_run_filter,requested_limit,source_membership_canonical,
    source_membership_hash)
VALUES ('running','snapshot','snapshot_hash','ranking_policy','ranking_hash',1,
        'evaluation_run','scope','scope_hash',1,100,
        'experiment_recommendation_ranking_membership_v1;count=2;member[0].evaluation_identity=10:evaluation;member[0].evaluation_result_id=1;member[1].evaluation_identity=12:evaluation_2;member[1].evaluation_result_id=2',
        'membership_hash');
INSERT INTO experiment_recommendation_ranking_member(
    recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
    recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
    bucket,bucket_rank,global_ordinal,final_score,disposition,
    tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
    inclusion_reason,top_positive_component,symbol,horizon,family,source_value_canonical,
    proposed_value_canonical,source_experiment_id,source_model_id,
    source_analysis_id,recommendation_scan_id)
VALUES (1,1,1,'semantic_hash','evaluation_hash','advisory_ready',1,1,0.8,
        'advisory_ready','score:0.8','semantic_hash','evaluation_hash',
        'included_advisory_ready','leader_quality','EURUSD',12,
        'core_lr_mult','1','1.25',
        1,1,1,1);
ALTER TABLE experiment_recommendation_ranking_member
    DISABLE TRIGGER enforce_recommendation_ranking_member_insert_trigger;
DO $$
DECLARE observed text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,block_reason,symbol,horizon,family,source_value_canonical,
            proposed_value_canonical,source_experiment_id,recommendation_scan_id)
        VALUES (1,2,2,'hash','hash','blocked',2,2,0.5,
                'blocked_active_duplicate','priority:1','hash','hash','included',
                'blocked_reason','EURUSD',12,'core_lr_mult','1','2',1,1);
        RAISE EXCEPTION 'invalid_bucket_shape_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE observed text;
BEGIN
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,symbol,horizon,family,source_value_canonical,
            proposed_value_canonical,source_experiment_id,recommendation_scan_id)
        VALUES (1,2,2,'hash','hash','advisory_ready',2,2,
                'NaN'::double precision,'advisory_ready','score:NaN','hash',
                'hash','included','EURUSD',12,'core_lr_mult','1','2',1,1);
        RAISE EXCEPTION 'nonfinite_ranking_score_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE
    observed text;
    invalid_score double precision;
BEGIN
    FOREACH invalid_score IN ARRAY ARRAY[
        'Infinity'::double precision,
        '-Infinity'::double precision
    ]
    LOOP
        BEGIN
            INSERT INTO experiment_recommendation_ranking_member(
                recommendation_ranking_snapshot_id,
                recommendation_evaluation_result_id,recommendation_id,
                recommendation_semantic_hash,evaluation_identity_hash,bucket,
                bucket_rank,global_ordinal,final_score,disposition,
                tie_break_primary,tie_break_semantic_hash,
                tie_break_evaluation_hash,inclusion_reason,symbol,horizon,
                family,source_value_canonical,proposed_value_canonical,
                source_experiment_id,recommendation_scan_id)
            VALUES (1,2,2,'hash','hash','advisory_ready',2,2,invalid_score,
                    'advisory_ready','score:nonfinite','hash','hash','included',
                    'EURUSD',12,'core_lr_mult','1','2',1,1);
            RAISE EXCEPTION 'infinite_ranking_score_was_accepted';
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
            IF observed<>'23514' THEN RAISE; END IF;
        END;
    END LOOP;
END $$;
ALTER TABLE experiment_recommendation_ranking_member
    ENABLE TRIGGER enforce_recommendation_ranking_member_insert_trigger;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,disposition,tie_break_primary,
            tie_break_semantic_hash,tie_break_evaluation_hash,inclusion_reason,
            block_reason,symbol,horizon,family,source_value_canonical,
            proposed_value_canonical,source_experiment_id,recommendation_scan_id)
        VALUES (1,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',
                1,2,'stale_source_evidence','priority:1','semantic_hash_2',
                'evaluation_hash_2','included_for_advisory_inspection',
                'wrong_block_reason','EURUSD',12,'core_lr_mult','1','1.25',1,1);
        RAISE EXCEPTION 'mismatched_ranking_block_reason_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,disposition,tie_break_primary,
            tie_break_semantic_hash,tie_break_evaluation_hash,inclusion_reason,
            block_reason,symbol,horizon,family,source_value_canonical,
            proposed_value_canonical,source_experiment_id,recommendation_scan_id)
        VALUES (1,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',
                1,2,'stale_source_evidence','priority:0','semantic_hash_2',
                'evaluation_hash_2','included_for_advisory_inspection','stale',
                'EURUSD',12,'core_lr_mult','1','1.25',1,1);
        RAISE EXCEPTION 'mismatched_ranking_priority_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

UPDATE experiment_recommendation_ranking_snapshot SET status='completed',
       member_count=1,advisory_ready_count=1,completed_at=now()
WHERE recommendation_ranking_snapshot_id=1;

DO $$
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    INSERT INTO experiment_recommendation_ranking_member(
        recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
        recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
        bucket,bucket_rank,global_ordinal,final_score,disposition,
        tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
        inclusion_reason,top_positive_component,symbol,horizon,family,
        source_value_canonical,proposed_value_canonical,source_experiment_id,
        source_model_id,source_analysis_id,recommendation_scan_id)
    VALUES (1,1,1,'semantic_hash','evaluation_hash','advisory_ready',1,1,0.8,
            'advisory_ready','score:0.8','semantic_hash','evaluation_hash',
            'included_advisory_ready','leader_quality','EURUSD',12,
            'core_lr_mult','1','1.25',1,1,1,1)
    ON CONFLICT (recommendation_ranking_snapshot_id,
                 recommendation_evaluation_result_id) DO NOTHING;
END $$;
RESET ROLE;

DO $$
BEGIN
    IF (SELECT count(*) FROM experiment_recommendation_ranking_member
        WHERE recommendation_ranking_snapshot_id=1)<>1 THEN
        RAISE EXCEPTION 'completed_snapshot_exact_retry_changed_membership';
    END IF;
END $$;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        UPDATE experiment_recommendation_ranking_snapshot
        SET status='running',member_count=0,advisory_ready_count=0,
            completed_at=NULL,updated_at=now()
        WHERE recommendation_ranking_snapshot_id=1;
        RAISE EXCEPTION 'terminal_ranking_snapshot_was_reopened';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        UPDATE experiment_recommendation_ranking_member
        SET final_score=0.1 WHERE recommendation_ranking_member_id=1;
        RAISE EXCEPTION 'member_update_was_accepted';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'42501' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

INSERT INTO experiment_recommendation_ranking_snapshot(
    status,ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,ranking_policy_canonical,
    ranking_policy_hash,ranking_version,scope_type,scope_canonical,scope_hash,
    symbol_filter,requested_limit,source_membership_canonical,
    source_membership_hash)
VALUES ('running','scope_mismatch_snapshot','scope_mismatch_hash',
        'ranking_policy','ranking_hash',1,'symbol','symbol_scope','scope_hash',
        'GBPUSD',100,
        'experiment_recommendation_ranking_membership_v1;count=1;member[0].evaluation_identity=12:evaluation_2;member[0].evaluation_result_id=2',
        'membership_hash_2');

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,block_reason,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            recommendation_scan_id)
        VALUES (2,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',
                1,1,NULL,'stale_source_evidence','priority:1',
                'semantic_hash_2','evaluation_hash_2',
                'included_for_advisory_inspection','stale','EURUSD',12,
                'core_lr_mult','1','1.25',1,1);
        RAISE EXCEPTION 'out_of_scope_ranking_member_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

INSERT INTO experiment_recommendation_ranking_snapshot(
    status,ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,ranking_policy_canonical,
    ranking_policy_hash,ranking_version,scope_type,scope_canonical,scope_hash,
    evaluation_run_filter,requested_limit,source_membership_canonical,
    source_membership_hash)
VALUES ('running','membership_mismatch_snapshot','membership_mismatch_hash',
        'ranking_policy','ranking_hash',1,'evaluation_run','run_scope','scope_hash',
        1,100,
        'experiment_recommendation_ranking_membership_v1;count=1;member[0].evaluation_identity=10:evaluation;member[0].evaluation_result_id=1',
        'membership_hash_3');

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        UPDATE experiment_recommendation_ranking_snapshot
        SET status='completed',member_count=1,advisory_ready_count=1,
            completed_at=now(),updated_at=now()
        WHERE recommendation_ranking_snapshot_id=3;
        RAISE EXCEPTION 'ranking_snapshot_false_counts_were_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,top_positive_component,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            source_model_id,source_analysis_id,recommendation_scan_id)
        VALUES (3,1,1,'semantic_hash','evaluation_hash','advisory_ready',
                101,101,0.8,'advisory_ready','score:0.8','semantic_hash',
                'evaluation_hash','included_advisory_ready','leader_quality',
                'EURUSD',12,'core_lr_mult','1','1.25',1,1,1,1);
        RAISE EXCEPTION 'out_of_limit_ranking_ordinal_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,block_reason,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            recommendation_scan_id)
        VALUES (3,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',
                1,1,NULL,'stale_source_evidence','priority:1',
                'semantic_hash_2','evaluation_hash_2',
                'included_for_advisory_inspection','stale','EURUSD',12,
                'core_lr_mult','1','1.25',1,1);
        RAISE EXCEPTION 'uncaptured_ranking_member_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,top_positive_component,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            source_model_id,source_analysis_id,recommendation_scan_id)
        VALUES (3,1,1,'semantic_hash','evaluation_hash','advisory_ready',
                1,1,0.8,'advisory_ready','score:0.7','semantic_hash',
                'evaluation_hash','included_advisory_ready','leader_quality',
                'EURUSD',12,'core_lr_mult','1','1.25',1,1,1,1);
        RAISE EXCEPTION 'mismatched_primary_tie_break_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,top_positive_component,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            source_model_id,source_analysis_id,recommendation_scan_id)
        VALUES (3,1,1,'wrong_semantic_hash','evaluation_hash','advisory_ready',
                1,1,0.8,'advisory_ready','score:0.8','wrong_semantic_hash',
                'evaluation_hash','included_advisory_ready','leader_quality',
                'EURUSD',12,'core_lr_mult','1','1.25',1,1,1,1);
        RAISE EXCEPTION 'mismatched_ranking_identity_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,top_positive_component,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,source_experiment_id,
            source_model_id,source_analysis_id,recommendation_scan_id)
        VALUES (3,1,1,'semantic_hash','evaluation_hash','advisory_ready',
                1,1,0.8,'advisory_ready','score:0.8','semantic_hash',
                'evaluation_hash','included_advisory_ready','wrong_component',
                'EURUSD',12,'core_lr_mult','1','1.25',1,1,1,1);
        RAISE EXCEPTION 'mismatched_top_component_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

INSERT INTO experiment_recommendation_ranking_snapshot(
    status,ranking_snapshot_identity_canonical,
    ranking_snapshot_identity_hash,ranking_policy_canonical,
    ranking_policy_hash,ranking_version,scope_type,scope_canonical,scope_hash,
    requested_limit,source_membership_canonical,source_membership_hash)
VALUES ('running','multirow_limit_snapshot','multirow_limit_hash',
        'ranking_policy','ranking_hash',1,'global','global_scope','scope_hash',1,
        'experiment_recommendation_ranking_membership_v1;count=2;member[0].evaluation_identity=10:evaluation;member[0].evaluation_result_id=1;member[1].evaluation_identity=12:evaluation_2;member[1].evaluation_result_id=2',
        'multirow_membership_hash');

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,block_reason,top_positive_component,symbol,horizon,
            family,source_value_canonical,proposed_value_canonical,
            source_experiment_id,source_model_id,source_analysis_id,
            recommendation_scan_id)
        VALUES
            (4,1,1,'semantic_hash','evaluation_hash','advisory_ready',1,1,0.8,
             'advisory_ready','score:0.8','semantic_hash','evaluation_hash',
             'included_advisory_ready',NULL,'leader_quality','EURUSD',12,
             'core_lr_mult','1','1.25',1,1,1,1),
            (4,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',1,2,NULL,
             'stale_source_evidence','priority:1','semantic_hash_2',
             'evaluation_hash_2','included_for_advisory_inspection','stale',NULL,
             'EURUSD',12,'core_lr_mult','1','1.25',1,NULL,NULL,1);
        RAISE EXCEPTION 'multirow_ranking_member_limit_was_bypassed';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
BEGIN
    IF EXISTS (SELECT 1 FROM experiment_recommendation_ranking_member
               WHERE recommendation_ranking_snapshot_id=4) THEN
        RAISE EXCEPTION 'multirow_limit_failure_left_partial_membership';
    END IF;
END $$;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        UPDATE experiment_recommendation_ranking_snapshot
        SET ranking_policy_canonical='changed'
        WHERE recommendation_ranking_snapshot_id=1;
        RAISE EXCEPTION 'immutable_snapshot_update_was_accepted';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'42501' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,recommendation_evaluation_result_id,
            recommendation_id,recommendation_semantic_hash,evaluation_identity_hash,
            bucket,bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,tie_break_evaluation_hash,
            inclusion_reason,block_reason,symbol,horizon,family,source_value_canonical,
            proposed_value_canonical,source_experiment_id,recommendation_scan_id)
        VALUES (1,2,2,'semantic_hash_2','evaluation_hash_2','non_actionable',1,2,
                NULL,'stale_source_evidence','priority:1','semantic_hash_2',
                'evaluation_hash_2','included_for_advisory_inspection','stale',
                'EURUSD',12,
                'core_lr_mult','1','1.25',1,1);
        RAISE EXCEPTION 'terminal_snapshot_member_insert_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed<>'23514' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

DO $$
BEGIN
    IF (SELECT count(*) FROM experiment)<>1
       OR (SELECT count(*) FROM model)<>1
       OR (SELECT count(*) FROM experiment_recommendation)<>2
       OR (SELECT count(*) FROM experiment_recommendation_evaluation_result)<>2
       OR (SELECT count(*) FROM experiment_recommendation_evaluation_component)<>1
       OR (SELECT count(*) FROM experiment_analysis_result)<>1 THEN
        RAISE EXCEPTION 'source_tables_modified';
    END IF;
END $$;

ROLLBACK;
