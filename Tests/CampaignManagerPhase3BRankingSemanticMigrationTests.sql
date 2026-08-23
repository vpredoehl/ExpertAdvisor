\set ON_ERROR_STOP on

BEGIN;
SET TRANSACTION READ WRITE;
CREATE SCHEMA campaign_manager_phase3b_migration_test;
SET LOCAL search_path TO campaign_manager_phase3b_migration_test, public;

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

CREATE FUNCTION phase3b_test_hash(canonical_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE STRICT
AS $$
DECLARE bytes bytea := convert_to(canonical_value, 'UTF8');
DECLARE value numeric := 14695981039346656037;
DECLARE index integer;
DECLARE low_byte integer;
DECLARE high_word bigint;
DECLARE low_word bigint;
BEGIN
    IF octet_length(bytes) > 0 THEN
        FOR index IN 0..octet_length(bytes) - 1 LOOP
            low_byte := mod(value, 256)::integer # get_byte(bytes, index);
            value := value - mod(value, 256) + low_byte;
            value := mod(value * 1099511628211, 18446744073709551616);
        END LOOP;
    END IF;
    high_word := trunc(value / 4294967296)::bigint;
    low_word := mod(value, 4294967296)::bigint;
    RETURN 'fnv1a64:' || lpad(to_hex(high_word), 8, '0') ||
        lpad(to_hex(low_word), 8, '0');
END;
$$;

INSERT INTO experiment DEFAULT VALUES;
INSERT INTO model DEFAULT VALUES;
INSERT INTO experiment_analysis_result DEFAULT VALUES;
INSERT INTO experiment_recommendation_scan DEFAULT VALUES;
INSERT INTO experiment_recommendation(
    recommendation_scan_id,source_experiment_id,source_analysis_id)
VALUES (1,1,1),(1,1,1),(1,1,1),(1,1,1);

DO $$
DECLARE
    scoring_a text :=
        'experiment_recommendation_scoring_policy_v1;scoring_version=1;' ||
        'leader_score_weight=0.25;inference_accuracy_weight=0.25;' ||
        'evidence_strength_weight=0.14999999999999999;' ||
        'neutral_balance_weight=0.10000000000000001;' ||
        'structural_distance_weight=0.14999999999999999;' ||
        'parameter_preference_weight=0.050000000000000003;' ||
        'source_rank_weight=0.050000000000000003;' ||
        'horizon_change_penalty_weight=0.050000000000000003;' ||
        'relative_mutation_penalty_weight=0.10000000000000001;' ||
        'minimum_evidence_count=1;evidence_saturation_count=5000;' ||
        'preferred_neutral_proportion=0.33333333333333331;' ||
        'maximum_neutral_proportion=0.80000000000000004;' ||
        'maximum_relative_mutation=0.5;' ||
        'maximum_absolute_structural_distance=1;' ||
        'allow_missing_neutral_proportion=1;score_floor=0;score_ceiling=1;' ||
        'core_lr_preference=1;head_lr_preference=1;' ||
        'label_threshold_preference=1;prediction_horizon_preference=1';
    scoring_b text;
    evaluation_a text;
    evaluation_b text;
BEGIN
    scoring_b := replace(scoring_a, 'leader_score_weight=0.25',
                         'leader_score_weight=0.29999999999999999');
    evaluation_a :=
        'experiment_recommendation_evaluation_policy_v1;evaluation_version=1;' ||
        'evaluator_version=1;scoring_policy=' || octet_length(scoring_a) || ':' ||
        scoring_a;
    evaluation_b :=
        'experiment_recommendation_evaluation_policy_v1;evaluation_version=1;' ||
        'evaluator_version=1;scoring_policy=' || octet_length(scoring_b) || ':' ||
        scoring_b;
    INSERT INTO experiment_recommendation_evaluation_run(
        status,evaluation_run_identity_canonical,evaluation_run_identity_hash,
        evaluation_policy_canonical,evaluation_policy_hash,evaluation_version,
        evaluator_version,scoring_policy_canonical,scoring_policy_hash,
        scoring_version,evidence_snapshot_canonical,evidence_snapshot_hash,
        completed_at)
    VALUES
      ('completed','run_a1',phase3b_test_hash('run_a1'),evaluation_a,
       phase3b_test_hash(evaluation_a),1,1,scoring_a,
       phase3b_test_hash(scoring_a),1,'snapshot_a1',
       phase3b_test_hash('snapshot_a1'),now()),
      ('completed','run_a2',phase3b_test_hash('run_a2'),evaluation_a,
       phase3b_test_hash(evaluation_a),1,1,scoring_a,
       phase3b_test_hash(scoring_a),1,'snapshot_a2',
       phase3b_test_hash('snapshot_a2'),now()),
      ('completed','run_b',phase3b_test_hash('run_b'),evaluation_b,
       phase3b_test_hash(evaluation_b),1,1,scoring_b,
       phase3b_test_hash(scoring_b),1,'snapshot_b',
       phase3b_test_hash('snapshot_b'),now());
END $$;

DO $$
DECLARE
    item record;
    evaluation_policy text;
    semantic_canonical text;
    recommendation_policy text := 'recommendation_policy';
    evidence_canonical text;
    identity_canonical text;
BEGIN
    FOR item IN
        SELECT run.recommendation_evaluation_run_id AS run_id,
               run.evaluation_policy_canonical,
               run.recommendation_evaluation_run_id AS recommendation_id
        FROM experiment_recommendation_evaluation_run run
        ORDER BY run.recommendation_evaluation_run_id
    LOOP
        evaluation_policy := item.evaluation_policy_canonical;
        semantic_canonical := 'semantic_' || item.run_id;
        evidence_canonical := 'evidence_' || item.run_id;
        identity_canonical :=
            'experiment_recommendation_evaluation_identity_v1;' ||
            'evaluation_policy=' || octet_length(evaluation_policy) || ':' ||
            evaluation_policy || ';recommendation_semantic=' ||
            octet_length(semantic_canonical) || ':' || semantic_canonical ||
            ';recommendation_policy=' || octet_length(recommendation_policy) ||
            ':' || recommendation_policy || ';evidence=' ||
            octet_length(evidence_canonical) || ':' || evidence_canonical;
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,source_model_id,
            source_analysis_id,evidence_canonical,evidence_hash,eligibility,
            disposition,reason_code,explanation,final_score,raw_positive_score,
            raw_penalty_score,raw_total_score,component_count,
            missing_evidence_count,ranking_ordinal)
        VALUES (item.run_id,item.recommendation_id,identity_canonical,
            phase3b_test_hash(identity_canonical),semantic_canonical,
            phase3b_test_hash(semantic_canonical),recommendation_policy,
            phase3b_test_hash(recommendation_policy),1,1,1,1,
            evidence_canonical,phase3b_test_hash(evidence_canonical),'eligible',
            'advisory_ready','advisory_ready','Ready.',0.8,1.0,0.2,0.8,
            1,0,1);
    END LOOP;
END $$;

ALTER TABLE experiment_recommendation_ranking_snapshot
    DISABLE TRIGGER enforce_recommendation_ranking_snapshot_transition_trigger;
DO $$
DECLARE
    identity_1 text;
    identity_2 text;
    identity_3 text;
    membership_verified text;
    membership_heterogeneous text;
BEGIN
    SELECT evaluation_identity_canonical INTO identity_1
    FROM experiment_recommendation_evaluation_result
    WHERE recommendation_evaluation_result_id=1;
    SELECT evaluation_identity_canonical INTO identity_2
    FROM experiment_recommendation_evaluation_result
    WHERE recommendation_evaluation_result_id=2;
    SELECT evaluation_identity_canonical INTO identity_3
    FROM experiment_recommendation_evaluation_result
    WHERE recommendation_evaluation_result_id=3;
    membership_verified :=
        'experiment_recommendation_ranking_membership_v1;count=2;' ||
        'member[0].evaluation_identity=' || octet_length(least(identity_1,identity_2)) ||
        ':' || least(identity_1,identity_2) || ';member[0].evaluation_result_id=' ||
        CASE WHEN identity_1 < identity_2 THEN '1' ELSE '2' END ||
        ';member[1].evaluation_identity=' || octet_length(greatest(identity_1,identity_2)) ||
        ':' || greatest(identity_1,identity_2) || ';member[1].evaluation_result_id=' ||
        CASE WHEN identity_1 < identity_2 THEN '2' ELSE '1' END;
    membership_heterogeneous :=
        'experiment_recommendation_ranking_membership_v1;count=2;' ||
        'member[0].evaluation_identity=' || octet_length(least(identity_1,identity_3)) ||
        ':' || least(identity_1,identity_3) || ';member[0].evaluation_result_id=' ||
        CASE WHEN identity_1 < identity_3 THEN '1' ELSE '3' END ||
        ';member[1].evaluation_identity=' || octet_length(greatest(identity_1,identity_3)) ||
        ':' || greatest(identity_1,identity_3) || ';member[1].evaluation_result_id=' ||
        CASE WHEN identity_1 < identity_3 THEN '3' ELSE '1' END;
    INSERT INTO experiment_recommendation_ranking_snapshot(
        status,ranking_snapshot_identity_canonical,
        ranking_snapshot_identity_hash,ranking_policy_canonical,
        ranking_policy_hash,ranking_version,scope_type,scope_canonical,
        scope_hash,requested_limit,source_membership_canonical,
        source_membership_hash,completed_at)
    VALUES
      ('completed','legacy_verified','legacy_verified_hash','ranking','hash',1,
       'global','scope','hash',1,membership_verified,'hash',now()),
      ('completed','legacy_heterogeneous','legacy_heterogeneous_hash','ranking',
       'hash',1,'global','scope','hash',1,membership_heterogeneous,'hash',now()),
      ('completed','legacy_unverified','legacy_unverified_hash','ranking','hash',1,
       'global','scope','hash',1,
       'experiment_recommendation_ranking_membership_v1;count=1;malformed',
       'hash',now()),
      ('completed','legacy_empty','legacy_empty_hash','ranking','hash',1,
       'global','scope','hash',1,
       'experiment_recommendation_ranking_membership_v1;count=0','hash',now());
END $$;
ALTER TABLE experiment_recommendation_ranking_snapshot
    ENABLE TRIGGER enforce_recommendation_ranking_snapshot_transition_trigger;

\ir ../Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql
\ir ../Database/migrations/077_campaign_manager_ranking_semantic_homogeneity.sql


-- Adversarial SQL-side scoring-policy parity tests. Each malformed policy gets
-- its hash recomputed, proving hash correctness alone cannot bless invalid
-- semantic provenance.
DO $$
DECLARE
    valid_scoring text :=
        'experiment_recommendation_scoring_policy_v1;scoring_version=1;' ||
        'leader_score_weight=0.25;inference_accuracy_weight=0.25;' ||
        'evidence_strength_weight=0.14999999999999999;' ||
        'neutral_balance_weight=0.10000000000000001;' ||
        'structural_distance_weight=0.14999999999999999;' ||
        'parameter_preference_weight=0.050000000000000003;' ||
        'source_rank_weight=0.050000000000000003;' ||
        'horizon_change_penalty_weight=0.050000000000000003;' ||
        'relative_mutation_penalty_weight=0.10000000000000001;' ||
        'minimum_evidence_count=1;evidence_saturation_count=5000;' ||
        'preferred_neutral_proportion=0.33333333333333331;' ||
        'maximum_neutral_proportion=0.80000000000000004;' ||
        'maximum_relative_mutation=0.5;' ||
        'maximum_absolute_structural_distance=1;' ||
        'allow_missing_neutral_proportion=1;score_floor=0;score_ceiling=1;' ||
        'core_lr_preference=1;head_lr_preference=1;' ||
        'label_threshold_preference=1;prediction_horizon_preference=1';
    invalid_policies text[];
    invalid_scoring text;
    evaluation_policy text;
    observed text;
    ordinal integer := 0;
BEGIN
    IF NOT recommendation_scoring_policy_provenance_valid_v1(
        valid_scoring, phase3b_test_hash(valid_scoring), 1)
    THEN
        RAISE EXCEPTION 'valid_cpp_canonical_scoring_policy_rejected';
    END IF;

    invalid_policies := ARRAY[
        replace(valid_scoring, 'leader_score_weight=0.25',
                 'leader_score_weight=-0.25'),
        replace(valid_scoring, 'minimum_evidence_count=1',
                 'minimum_evidence_count=0'),
        replace(valid_scoring,
                'minimum_evidence_count=1;evidence_saturation_count=5000',
                'minimum_evidence_count=10;evidence_saturation_count=9'),
        replace(valid_scoring, 'score_floor=0;score_ceiling=1',
                 'score_floor=1;score_ceiling=1'),
        replace(valid_scoring, 'core_lr_preference=1',
                 'core_lr_preference=1.5'),
        replace(valid_scoring, 'leader_score_weight=0.25',
                 'leader_score_weight=0.250')
    ];

    FOREACH invalid_scoring IN ARRAY invalid_policies LOOP
        ordinal := ordinal + 1;

        IF recommendation_scoring_policy_provenance_valid_v1(
            invalid_scoring, phase3b_test_hash(invalid_scoring), 1)
        THEN
            RAISE EXCEPTION
                'invalid_scoring_policy_validator_accepted_case_%', ordinal;
        END IF;

        evaluation_policy :=
            'experiment_recommendation_evaluation_policy_v1;evaluation_version=1;' ||
            'evaluator_version=1;scoring_policy=' ||
            octet_length(invalid_scoring) || ':' || invalid_scoring;

        BEGIN
            INSERT INTO experiment_recommendation_evaluation_run(
                status,evaluation_run_identity_canonical,
                evaluation_run_identity_hash,evaluation_policy_canonical,
                evaluation_policy_hash,evaluation_version,evaluator_version,
                scoring_policy_canonical,scoring_policy_hash,scoring_version,
                evidence_snapshot_canonical,evidence_snapshot_hash,completed_at)
            VALUES (
                'completed','invalid_policy_run_' || ordinal,
                phase3b_test_hash('invalid_policy_run_' || ordinal),
                evaluation_policy,phase3b_test_hash(evaluation_policy),1,1,
                invalid_scoring,phase3b_test_hash(invalid_scoring),1,
                'invalid_policy_snapshot_' || ordinal,
                phase3b_test_hash('invalid_policy_snapshot_' || ordinal),now());

            RAISE EXCEPTION
                'invalid_scoring_policy_run_was_accepted_case_%', ordinal;
        EXCEPTION WHEN check_violation THEN
            GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
            IF observed <> '23514' THEN RAISE; END IF;
        END;
    END LOOP;
END $$;

-- Migration 034 requires subordinate hashes to be nonempty but does not
-- recompute them. Phase 3B must independently bind every canonical/hash pair.
DO $$
DECLARE
    run_policy text;
    recommendation_semantic text := 'semantic_4';
    recommendation_policy text := 'recommendation_policy';
    evidence_canonical text := 'evidence_4';
    identity_canonical text;
    observed text;
BEGIN
    SELECT evaluation_policy_canonical INTO run_policy
    FROM experiment_recommendation_evaluation_run
    WHERE recommendation_evaluation_run_id=1;

    identity_canonical :=
        'experiment_recommendation_evaluation_identity_v1;' ||
        'evaluation_policy=' || octet_length(run_policy) || ':' || run_policy ||
        ';recommendation_semantic=' || octet_length(recommendation_semantic) ||
        ':' || recommendation_semantic || ';recommendation_policy=' ||
        octet_length(recommendation_policy) || ':' || recommendation_policy ||
        ';evidence=' || octet_length(evidence_canonical) || ':' ||
        evidence_canonical;

    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,evidence_canonical,
            evidence_hash,eligibility,disposition,reason_code,explanation,
            component_count,missing_evidence_count,ranking_ordinal)
        VALUES (1,4,identity_canonical,phase3b_test_hash(identity_canonical),
            recommendation_semantic,phase3b_test_hash(recommendation_semantic),
            recommendation_policy,phase3b_test_hash(recommendation_policy),
            1,1,evidence_canonical,'fnv1a64:0000000000000000','ineligible',
            'insufficient_evidence','missing','Missing.',0,1,2);

        RAISE EXCEPTION 'wrong_evidence_hash_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,evidence_canonical,
            evidence_hash,eligibility,disposition,reason_code,explanation,
            component_count,missing_evidence_count,ranking_ordinal)
        VALUES (1,4,identity_canonical,phase3b_test_hash(identity_canonical),
            recommendation_semantic,'fnv1a64:0000000000000000',
            recommendation_policy,phase3b_test_hash(recommendation_policy),
            1,1,evidence_canonical,phase3b_test_hash(evidence_canonical),
            'ineligible','insufficient_evidence','missing','Missing.',0,1,2);

        RAISE EXCEPTION 'wrong_recommendation_semantic_hash_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;

    BEGIN
        INSERT INTO experiment_recommendation_evaluation_result(
            recommendation_evaluation_run_id,recommendation_id,
            evaluation_identity_canonical,evaluation_identity_hash,
            recommendation_semantic_canonical,recommendation_semantic_hash,
            recommendation_policy_canonical,recommendation_policy_hash,
            recommendation_scan_id,source_experiment_id,evidence_canonical,
            evidence_hash,eligibility,disposition,reason_code,explanation,
            component_count,missing_evidence_count,ranking_ordinal)
        VALUES (1,4,identity_canonical,phase3b_test_hash(identity_canonical),
            recommendation_semantic,phase3b_test_hash(recommendation_semantic),
            recommendation_policy,'fnv1a64:0000000000000000',
            1,1,evidence_canonical,phase3b_test_hash(evidence_canonical),
            'ineligible','insufficient_evidence','missing','Missing.',0,1,2);

        RAISE EXCEPTION 'wrong_recommendation_policy_hash_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
BEGIN
    IF (SELECT population_semantic_state
        FROM experiment_recommendation_ranking_snapshot
        WHERE ranking_snapshot_identity_canonical='legacy_verified') <>
       'verified_homogeneous' THEN
        RAISE EXCEPTION 'historical_homogeneous_population_not_verified';
    END IF;
    IF (SELECT population_semantic_state
        FROM experiment_recommendation_ranking_snapshot
        WHERE ranking_snapshot_identity_canonical='legacy_heterogeneous') <>
       'legacy_heterogeneous' THEN
        RAISE EXCEPTION 'historical_heterogeneity_not_classified';
    END IF;
    IF (SELECT population_semantic_state
        FROM experiment_recommendation_ranking_snapshot
        WHERE ranking_snapshot_identity_canonical='legacy_unverified') <>
       'legacy_unverified' THEN
        RAISE EXCEPTION 'malformed_historical_membership_not_unverified';
    END IF;
    IF (SELECT population_semantic_state
        FROM experiment_recommendation_ranking_snapshot
        WHERE ranking_snapshot_identity_canonical='legacy_empty') <> 'empty' THEN
        RAISE EXCEPTION 'historical_empty_population_not_classified';
    END IF;
END $$;

GRANT USAGE ON SCHEMA campaign_manager_phase3b_migration_test TO pqxx;
DO $$
DECLARE observed text;
DECLARE snapshot_id bigint;
DECLARE evaluation experiment_recommendation_evaluation_result%ROWTYPE;
BEGIN
    SELECT recommendation_ranking_snapshot_id INTO snapshot_id
    FROM experiment_recommendation_ranking_snapshot
    WHERE ranking_snapshot_identity_canonical='legacy_verified';
    SELECT * INTO evaluation
    FROM experiment_recommendation_evaluation_result
    WHERE recommendation_evaluation_result_id=3;
    BEGIN
        INSERT INTO experiment_recommendation_ranking_member(
            recommendation_ranking_snapshot_id,
            recommendation_evaluation_result_id,recommendation_id,
            recommendation_semantic_hash,evaluation_identity_hash,bucket,
            bucket_rank,global_ordinal,final_score,disposition,
            tie_break_primary,tie_break_semantic_hash,
            tie_break_evaluation_hash,inclusion_reason,symbol,horizon,family,
            source_value_canonical,proposed_value_canonical,
            source_experiment_id,source_model_id,source_analysis_id,
            recommendation_scan_id)
        VALUES (snapshot_id,evaluation.recommendation_evaluation_result_id,
            evaluation.recommendation_id,evaluation.recommendation_semantic_hash,
            evaluation.evaluation_identity_hash,'advisory_ready',1,1,
            evaluation.final_score,evaluation.disposition,'score:0.8',
            evaluation.recommendation_semantic_hash,
            evaluation.evaluation_identity_hash,'included_advisory_ready',
            'EURUSD',12,'core_lr_mult','1','1.25',1,1,1,1);
        RAISE EXCEPTION 'semantic_mismatch_ranking_member_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE
    policy_canonical text := 'ranking';
    scope_canonical_value text := 'scope';
    membership text :=
        'experiment_recommendation_ranking_membership_v1;count=0';
    identity_canonical text;
BEGIN
    identity_canonical :=
        'experiment_recommendation_ranking_snapshot_identity_v2;policy=' ||
        octet_length(policy_canonical) || ':' || policy_canonical ||
        ';scope=' || octet_length(scope_canonical_value) || ':' ||
        scope_canonical_value || ';limit=10;population_state=empty;' ||
        'scoring_semantic=NULL;evaluation_semantic=NULL;membership=' ||
        octet_length(membership) || ':' || membership;
    INSERT INTO experiment_recommendation_ranking_snapshot(
        status,ranking_snapshot_identity_canonical,
        ranking_snapshot_identity_hash,ranking_policy_canonical,
        ranking_policy_hash,ranking_version,scope_type,scope_canonical,
        scope_hash,requested_limit,source_membership_canonical,
        source_membership_hash,ranking_snapshot_identity_version,
        population_semantic_state,distinct_scoring_semantic_count,
        distinct_evaluation_semantic_count,homogeneity_validation_result)
    VALUES ('running',identity_canonical,
        recommendation_semantic_tagged_fnv1a64(identity_canonical),
        policy_canonical,recommendation_semantic_tagged_fnv1a64(policy_canonical),
        1,'global',scope_canonical_value,
        recommendation_semantic_tagged_fnv1a64(scope_canonical_value),10,
        membership,recommendation_semantic_tagged_fnv1a64(membership),2,'empty',
        0,0,'empty_population');
END $$;

DO $$
DECLARE observed text;
DECLARE membership text;
DECLARE scoring_semantic text;
DECLARE evaluation_semantic text;
DECLARE identity_canonical text;
DECLARE policy_canonical text := 'ranking';
DECLARE scope_canonical_value text := 'global';
BEGIN
    SELECT source_membership_canonical INTO membership
    FROM experiment_recommendation_ranking_snapshot
    WHERE ranking_snapshot_identity_canonical='legacy_heterogeneous';
    SELECT recommendation_scoring_semantic_canonical_v1(
               scoring_policy_canonical,scoring_version),
           recommendation_evaluation_semantic_canonical_v1(
               evaluation_version,evaluator_version,
               scoring_policy_canonical,scoring_version)
      INTO scoring_semantic,evaluation_semantic
    FROM experiment_recommendation_evaluation_run
    WHERE recommendation_evaluation_run_id=1;
    identity_canonical :=
        'experiment_recommendation_ranking_snapshot_identity_v2;policy=' ||
        octet_length(policy_canonical) || ':' || policy_canonical ||
        ';scope=' || octet_length(scope_canonical_value) || ':' ||
        scope_canonical_value ||
        ';limit=1;population_state=verified_homogeneous;' ||
        'scoring_semantic=' || octet_length(scoring_semantic) || ':' ||
        scoring_semantic || ';evaluation_semantic=' ||
        octet_length(evaluation_semantic) || ':' || evaluation_semantic ||
        ';membership=' || octet_length(membership) || ':' || membership;
    BEGIN
        INSERT INTO experiment_recommendation_ranking_snapshot(
            status,ranking_snapshot_identity_canonical,
            ranking_snapshot_identity_hash,ranking_policy_canonical,
            ranking_policy_hash,ranking_version,scope_type,scope_canonical,
            scope_hash,requested_limit,source_membership_canonical,
            source_membership_hash,ranking_snapshot_identity_version,
            population_semantic_state,scoring_semantic_canonical,
            scoring_semantic_hash,scoring_semantic_version,
            evaluation_semantic_canonical,evaluation_semantic_hash,
            evaluation_semantic_version,distinct_scoring_semantic_count,
            distinct_evaluation_semantic_count,homogeneity_validation_result)
        VALUES ('running',identity_canonical,
            recommendation_semantic_tagged_fnv1a64(identity_canonical),
            policy_canonical,recommendation_semantic_tagged_fnv1a64(
                policy_canonical),1,'global',scope_canonical_value,
            recommendation_semantic_tagged_fnv1a64(scope_canonical_value),1,
            membership,recommendation_semantic_tagged_fnv1a64(membership),2,
            'verified_homogeneous',scoring_semantic,
            recommendation_semantic_tagged_fnv1a64(scoring_semantic),1,
            evaluation_semantic,
            recommendation_semantic_tagged_fnv1a64(evaluation_semantic),1,
            1,1,'verified_homogeneous');
        RAISE EXCEPTION 'heterogeneous_new_snapshot_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE observed text;
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
        VALUES (1,4,'wrong_identity',
            recommendation_semantic_tagged_fnv1a64('wrong_identity'),
            'semantic_4',recommendation_semantic_tagged_fnv1a64('semantic_4'),
            'recommendation_policy',
            recommendation_semantic_tagged_fnv1a64('recommendation_policy'),
            1,1,'evidence_4',recommendation_semantic_tagged_fnv1a64('evidence_4'),
            'ineligible','insufficient_evidence','missing','Missing.',0,1,2);
        RAISE EXCEPTION 'incoherent_evaluation_result_was_accepted';
    EXCEPTION WHEN check_violation THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '23514' THEN RAISE; END IF;
    END;
END $$;

DO $$
DECLARE observed text;
BEGIN
    EXECUTE 'SET LOCAL ROLE pqxx';
    EXECUTE 'SET LOCAL search_path TO campaign_manager_phase3b_migration_test, public';
    BEGIN
        UPDATE experiment_recommendation_ranking_snapshot
        SET scoring_semantic_hash='fnv1a64:0000000000000000'
        WHERE ranking_snapshot_identity_canonical='legacy_verified';
        RAISE EXCEPTION 'snapshot_semantic_identity_mutation_was_accepted';
    EXCEPTION WHEN insufficient_privilege THEN
        GET STACKED DIAGNOSTICS observed=RETURNED_SQLSTATE;
        IF observed <> '42501' THEN RAISE; END IF;
    END;
END $$;
RESET ROLE;

ROLLBACK;
