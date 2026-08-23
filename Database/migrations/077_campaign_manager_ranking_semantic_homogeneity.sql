-- Campaign Manager Phase 3B: fail-closed scoring/evaluation semantic
-- homogeneity for recommendation ranking populations.  Profitability remains
-- observational and is deliberately absent from every identity below.

CREATE OR REPLACE FUNCTION recommendation_semantic_tagged_fnv1a64(
    canonical_value text)
RETURNS text
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
DECLARE
    canonical_bytes bytea := convert_to(canonical_value, 'UTF8');
    hash_value numeric := 14695981039346656037;
    byte_index integer;
    low_byte integer;
    high_word bigint;
    low_word bigint;
BEGIN
    IF octet_length(canonical_bytes) > 0 THEN
        FOR byte_index IN 0..octet_length(canonical_bytes) - 1 LOOP
            low_byte := mod(hash_value, 256)::integer #
                get_byte(canonical_bytes, byte_index);
            hash_value := hash_value - mod(hash_value, 256) + low_byte;
            hash_value := mod(
                hash_value * 1099511628211, 18446744073709551616);
        END LOOP;
    END IF;
    high_word := trunc(hash_value / 4294967296)::bigint;
    low_word := mod(hash_value, 4294967296)::bigint;
    RETURN 'fnv1a64:' || lpad(to_hex(high_word), 8, '0') ||
        lpad(to_hex(low_word), 8, '0');
END;
$$;

CREATE OR REPLACE FUNCTION recommendation_scoring_policy_provenance_valid_v1(
    policy_canonical text,
    policy_hash text,
    policy_version integer)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
DECLARE
    parts text[];
    expected_keys constant text[] := ARRAY[
        'scoring_version',
        'leader_score_weight',
        'inference_accuracy_weight',
        'evidence_strength_weight',
        'neutral_balance_weight',
        'structural_distance_weight',
        'parameter_preference_weight',
        'source_rank_weight',
        'horizon_change_penalty_weight',
        'relative_mutation_penalty_weight',
        'minimum_evidence_count',
        'evidence_saturation_count',
        'preferred_neutral_proportion',
        'maximum_neutral_proportion',
        'maximum_relative_mutation',
        'maximum_absolute_structural_distance',
        'allow_missing_neutral_proportion',
        'score_floor',
        'score_ceiling',
        'core_lr_preference',
        'head_lr_preference',
        'label_threshold_preference',
        'prediction_horizon_preference'
    ];
    values_text text[] := ARRAY[]::text[];
    index_value integer;
    separator integer;
    key_text text;
    value_text text;
    canonical_double_pattern constant text :=
        '^(0|[1-9][0-9]*)([.][0-9]*[1-9])?([e][+-][0-9]{2,3})?$';
    weights double precision[];
    minimum_evidence bigint;
    evidence_saturation bigint;
    preferred_neutral double precision;
    maximum_neutral double precision;
    maximum_relative double precision;
    maximum_absolute double precision;
    score_floor_value double precision;
    score_ceiling_value double precision;
    preferences double precision[];
BEGIN
    IF policy_version <> 1 OR
       policy_hash <> recommendation_semantic_tagged_fnv1a64(policy_canonical)
    THEN
        RETURN false;
    END IF;

    parts := string_to_array(policy_canonical, ';');
    IF cardinality(parts) <> cardinality(expected_keys) + 1 OR
       parts[1] <> 'experiment_recommendation_scoring_policy_v1'
    THEN
        RETURN false;
    END IF;

    FOR index_value IN 1..cardinality(expected_keys) LOOP
        separator := strpos(parts[index_value + 1], '=');
        IF separator <= 1 OR
           strpos(substr(parts[index_value + 1], separator + 1), '=') <> 0
        THEN
            RETURN false;
        END IF;

        key_text := substr(parts[index_value + 1], 1, separator - 1);
        value_text := substr(parts[index_value + 1], separator + 1);

        IF key_text <> expected_keys[index_value] OR value_text = '' THEN
            RETURN false;
        END IF;

        values_text := array_append(values_text, value_text);
    END LOOP;

    IF values_text[1] <> '1' THEN
        RETURN false;
    END IF;

    -- C++ v1 emits non-negative finite policy values in a deterministic
    -- decimal/scientific spelling. Reject alternate spellings such as
    -- trailing fractional zeros, uppercase E, leading '+', and negative zero.
    FOREACH index_value IN ARRAY ARRAY[
        2,3,4,5,6,7,8,9,10,13,14,15,16,18,19,20,21,22,23
    ]
    LOOP
        IF values_text[index_value] !~ canonical_double_pattern THEN
            RETURN false;
        END IF;
    END LOOP;

    IF values_text[11] !~ '^[1-9][0-9]*$' OR
       values_text[12] !~ '^(0|[1-9][0-9]*)$' OR
       values_text[17] !~ '^[01]$'
    THEN
        RETURN false;
    END IF;

    BEGIN
        weights := ARRAY[
            values_text[2]::double precision,
            values_text[3]::double precision,
            values_text[4]::double precision,
            values_text[5]::double precision,
            values_text[6]::double precision,
            values_text[7]::double precision,
            values_text[8]::double precision,
            values_text[9]::double precision,
            values_text[10]::double precision
        ];
        minimum_evidence := values_text[11]::bigint;
        evidence_saturation := values_text[12]::bigint;
        preferred_neutral := values_text[13]::double precision;
        maximum_neutral := values_text[14]::double precision;
        maximum_relative := values_text[15]::double precision;
        maximum_absolute := values_text[16]::double precision;
        score_floor_value := values_text[18]::double precision;
        score_ceiling_value := values_text[19]::double precision;
        preferences := ARRAY[
            values_text[20]::double precision,
            values_text[21]::double precision,
            values_text[22]::double precision,
            values_text[23]::double precision
        ];
    EXCEPTION
        WHEN numeric_value_out_of_range OR invalid_text_representation THEN
            RETURN false;
    END;

    -- Mirror ValidateRecommendationScoringPolicy().
    IF EXISTS (
        SELECT 1
        FROM unnest(weights) value
        WHERE value < 0.0)
    THEN
        RETURN false;
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM unnest(weights[1:7]) value
        WHERE value > 0.0)
    THEN
        RETURN false;
    END IF;

    IF minimum_evidence <= 0 OR evidence_saturation < minimum_evidence THEN
        RETURN false;
    END IF;

    IF preferred_neutral < 0.0 OR preferred_neutral > 1.0 THEN
        RETURN false;
    END IF;

    IF maximum_neutral < preferred_neutral OR maximum_neutral > 1.0 THEN
        RETURN false;
    END IF;

    IF maximum_relative <= 0.0 OR maximum_absolute <= 0.0 THEN
        RETURN false;
    END IF;

    IF score_floor_value < 0.0 OR score_ceiling_value > 1.0 OR
       score_floor_value >= score_ceiling_value
    THEN
        RETURN false;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM unnest(preferences) value
        WHERE value < 0.0 OR value > 1.0)
    THEN
        RETURN false;
    END IF;

    RETURN true;
END;
$$;

CREATE OR REPLACE FUNCTION recommendation_evaluation_policy_provenance_valid_v1(
    evaluation_policy_canonical text,
    evaluation_policy_hash text,
    evaluation_version integer,
    evaluator_version integer,
    scoring_policy_canonical text,
    scoring_policy_hash text,
    scoring_version integer)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
PARALLEL SAFE
SECURITY INVOKER
AS $$
    SELECT recommendation_scoring_policy_provenance_valid_v1(
               scoring_policy_canonical, scoring_policy_hash, scoring_version)
       AND evaluation_version = 1
       AND evaluator_version = 1
       AND evaluation_policy_canonical =
           'experiment_recommendation_evaluation_policy_v1;evaluation_version=' ||
           evaluation_version || ';evaluator_version=' || evaluator_version ||
           ';scoring_policy=' || octet_length(scoring_policy_canonical) || ':' ||
           scoring_policy_canonical
       AND evaluation_policy_hash =
           recommendation_semantic_tagged_fnv1a64(evaluation_policy_canonical)
$$;

CREATE OR REPLACE FUNCTION recommendation_scoring_semantic_canonical_v1(
    scoring_policy_canonical text,
    scoring_version integer)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
SET search_path TO pg_catalog
AS $$
    SELECT 'experiment_recommendation_scoring_semantic_v1;semantic_version=1;' ||
        'algorithm=weighted_component_score_v1;' ||
        'components=leader_quality,inference_accuracy,evidence_strength,neutral_balance,' ||
        'structural_proximity,parameter_preference,source_rank,horizon_change_penalty,' ||
        'relative_mutation_penalty;' ||
        'normalization=component_clamp_unit_and_policy_thresholds_v1;' ||
        'aggregation=positive_weight_normalized_minus_penalty_weight_normalized_then_policy_clamp_v1;' ||
        'missing=neutral_policy_or_ineligible_v1;' ||
        'structural_distance=relative_delta_else_absolute_delta_v1;' ||
        'source_metrics=persisted_leader_accuracy_neutral_evidence_v1;policy=' ||
        octet_length(scoring_policy_canonical) || ':' || scoring_policy_canonical
    WHERE scoring_version = 1
$$;

CREATE OR REPLACE FUNCTION recommendation_evaluation_semantic_canonical_v1(
    evaluation_version integer,
    evaluator_version integer,
    scoring_policy_canonical text,
    scoring_version integer)
RETURNS text
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
    SELECT 'experiment_recommendation_evaluation_semantic_v1;semantic_version=1;' ||
        'algorithm=advisory_evaluator_v1;' ||
        'classification=invalid,unsupported,insufficient,stale,pending_duplicate,' ||
        'active_duplicate,completed_duplicate,advisory_ready_v1;' ||
        'eligibility=only_advisory_ready_is_eligible_v1;' ||
        'disposition_precedence=persisted_evidence_then_family_then_missing_then_stale_then_duplicate_then_score_v1;' ||
        'evaluation_version=' || evaluation_version ||
        ';evaluator_version=' || evaluator_version ||
        ';scoring_semantic=' ||
        octet_length(recommendation_scoring_semantic_canonical_v1(
            scoring_policy_canonical, scoring_version)) || ':' ||
        recommendation_scoring_semantic_canonical_v1(
            scoring_policy_canonical, scoring_version)
    WHERE evaluation_version = 1 AND evaluator_version = 1
      AND scoring_version = 1
$$;

CREATE OR REPLACE FUNCTION recommendation_evaluation_run_semantics_valid_v1(
    run experiment_recommendation_evaluation_run)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
    SELECT recommendation_evaluation_policy_provenance_valid_v1(
        run.evaluation_policy_canonical, run.evaluation_policy_hash,
        run.evaluation_version, run.evaluator_version,
        run.scoring_policy_canonical, run.scoring_policy_hash,
        run.scoring_version)
$$;

CREATE OR REPLACE FUNCTION recommendation_evaluation_result_semantics_valid_v1(
    evaluation experiment_recommendation_evaluation_result,
    run experiment_recommendation_evaluation_run)
RETURNS boolean
LANGUAGE sql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
AS $$
    SELECT recommendation_evaluation_run_semantics_valid_v1(run)
       AND evaluation.recommendation_evaluation_run_id =
           run.recommendation_evaluation_run_id
       AND evaluation.recommendation_semantic_hash =
           recommendation_semantic_tagged_fnv1a64(
               evaluation.recommendation_semantic_canonical)
       AND evaluation.recommendation_policy_hash =
           recommendation_semantic_tagged_fnv1a64(
               evaluation.recommendation_policy_canonical)
       AND evaluation.evidence_hash =
           recommendation_semantic_tagged_fnv1a64(
               evaluation.evidence_canonical)
       AND evaluation.evaluation_identity_canonical =
           'experiment_recommendation_evaluation_identity_v1;evaluation_policy=' ||
           octet_length(run.evaluation_policy_canonical) || ':' ||
           run.evaluation_policy_canonical || ';recommendation_semantic=' ||
           octet_length(evaluation.recommendation_semantic_canonical) || ':' ||
           evaluation.recommendation_semantic_canonical ||
           ';recommendation_policy=' ||
           octet_length(evaluation.recommendation_policy_canonical) || ':' ||
           evaluation.recommendation_policy_canonical || ';evidence=' ||
           octet_length(evaluation.evidence_canonical) || ':' ||
           evaluation.evidence_canonical
       AND evaluation.evaluation_identity_hash =
           recommendation_semantic_tagged_fnv1a64(
               evaluation.evaluation_identity_canonical)
$$;

CREATE OR REPLACE FUNCTION enforce_recommendation_evaluation_run_semantics_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    IF TG_OP = 'UPDATE' AND
       (OLD.evaluation_policy_canonical, OLD.evaluation_policy_hash,
        OLD.evaluation_version, OLD.evaluator_version,
        OLD.scoring_policy_canonical, OLD.scoring_policy_hash,
        OLD.scoring_version, OLD.evaluation_run_identity_canonical,
        OLD.evaluation_run_identity_hash, OLD.evidence_snapshot_canonical,
        OLD.evidence_snapshot_hash) IS NOT DISTINCT FROM
       (NEW.evaluation_policy_canonical, NEW.evaluation_policy_hash,
        NEW.evaluation_version, NEW.evaluator_version,
        NEW.scoring_policy_canonical, NEW.scoring_policy_hash,
        NEW.scoring_version, NEW.evaluation_run_identity_canonical,
        NEW.evaluation_run_identity_hash, NEW.evidence_snapshot_canonical,
        NEW.evidence_snapshot_hash) THEN
        RETURN NEW;
    END IF;
    IF NOT recommendation_evaluation_run_semantics_valid_v1(NEW)
       OR NEW.evaluation_run_identity_hash <>
          recommendation_semantic_tagged_fnv1a64(
              NEW.evaluation_run_identity_canonical)
       OR NEW.evidence_snapshot_hash <>
          recommendation_semantic_tagged_fnv1a64(
              NEW.evidence_snapshot_canonical) THEN
        RAISE EXCEPTION 'invalid evaluation run semantic provenance'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_evaluation_run_semantics_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_recommendation_evaluation_result_semantics_v1()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    parent_run experiment_recommendation_evaluation_run%ROWTYPE;
BEGIN
    SELECT * INTO parent_run
    FROM experiment_recommendation_evaluation_run
    WHERE recommendation_evaluation_run_id =
          NEW.recommendation_evaluation_run_id
    FOR SHARE;
    IF NOT FOUND OR NOT recommendation_evaluation_result_semantics_valid_v1(
            NEW, parent_run) THEN
        RAISE EXCEPTION 'evaluation result semantic provenance differs from run'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_evaluation_result_run_semantics_check';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recommendation_evaluation_run_semantics_v1_trigger
    ON experiment_recommendation_evaluation_run;
CREATE TRIGGER recommendation_evaluation_run_semantics_v1_trigger
BEFORE INSERT OR UPDATE ON experiment_recommendation_evaluation_run
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_evaluation_run_semantics_v1();

DROP TRIGGER IF EXISTS recommendation_evaluation_result_semantics_v1_trigger
    ON experiment_recommendation_evaluation_result;
CREATE TRIGGER recommendation_evaluation_result_semantics_v1_trigger
BEFORE INSERT OR UPDATE ON experiment_recommendation_evaluation_result
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_evaluation_result_semantics_v1();

ALTER TABLE experiment_recommendation_ranking_snapshot
    ADD COLUMN IF NOT EXISTS ranking_snapshot_identity_version integer,
    ADD COLUMN IF NOT EXISTS population_semantic_state text,
    ADD COLUMN IF NOT EXISTS scoring_semantic_canonical text,
    ADD COLUMN IF NOT EXISTS scoring_semantic_hash text,
    ADD COLUMN IF NOT EXISTS scoring_semantic_version integer,
    ADD COLUMN IF NOT EXISTS evaluation_semantic_canonical text,
    ADD COLUMN IF NOT EXISTS evaluation_semantic_hash text,
    ADD COLUMN IF NOT EXISTS evaluation_semantic_version integer,
    ADD COLUMN IF NOT EXISTS distinct_scoring_semantic_count integer,
    ADD COLUMN IF NOT EXISTS distinct_evaluation_semantic_count integer,
    ADD COLUMN IF NOT EXISTS homogeneity_validation_result text;

CREATE OR REPLACE FUNCTION recommendation_ranking_membership_count_v1(
    membership_canonical text)
RETURNS integer
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
SET search_path TO pg_catalog
AS $$
DECLARE
    matched text[];
    parsed numeric;
BEGIN
    matched := regexp_match(membership_canonical,
        '^experiment_recommendation_ranking_membership_v1;count=(0|[1-9][0-9]*)(;|$)');
    IF matched IS NULL OR length(matched[1]) > 4 THEN RETURN NULL; END IF;
    parsed := matched[1]::numeric;
    IF parsed > 1000 THEN RETURN NULL; END IF;
    IF parsed = 0 AND membership_canonical <>
       'experiment_recommendation_ranking_membership_v1;count=0' THEN
        RETURN NULL;
    END IF;
    RETURN parsed::integer;
EXCEPTION WHEN invalid_text_representation OR numeric_value_out_of_range THEN
    RETURN NULL;
END;
$$;

-- Migration-time classification updates completed historical snapshots.  The
-- Phase 3A transition trigger correctly rejects such application updates, so
-- suspend it only for this authoritative, identity-preserving classification.
ALTER TABLE experiment_recommendation_ranking_snapshot
    DISABLE TRIGGER enforce_recommendation_ranking_snapshot_transition_trigger;

WITH population AS (
    SELECT snapshot.recommendation_ranking_snapshot_id AS snapshot_id,
           recommendation_ranking_membership_count_v1(
               snapshot.source_membership_canonical) AS declared_count,
           count(evaluation.recommendation_evaluation_result_id)::integer AS
               resolved_count,
           count(evaluation.recommendation_evaluation_result_id) FILTER (
               WHERE NOT recommendation_evaluation_result_semantics_valid_v1(
                   evaluation, run))::integer AS invalid_count,
           count(DISTINCT recommendation_scoring_semantic_canonical_v1(
               run.scoring_policy_canonical, run.scoring_version)) FILTER (
               WHERE recommendation_evaluation_result_semantics_valid_v1(
                   evaluation, run))::integer AS scoring_count,
           count(DISTINCT recommendation_evaluation_semantic_canonical_v1(
               run.evaluation_version, run.evaluator_version,
               run.scoring_policy_canonical, run.scoring_version)) FILTER (
               WHERE recommendation_evaluation_result_semantics_valid_v1(
                   evaluation, run))::integer AS evaluation_count,
           min(recommendation_scoring_semantic_canonical_v1(
               run.scoring_policy_canonical, run.scoring_version)) FILTER (
               WHERE recommendation_evaluation_result_semantics_valid_v1(
                   evaluation, run)) AS scoring_canonical,
           min(recommendation_evaluation_semantic_canonical_v1(
               run.evaluation_version, run.evaluator_version,
               run.scoring_policy_canonical, run.scoring_version)) FILTER (
               WHERE recommendation_evaluation_result_semantics_valid_v1(
                   evaluation, run)) AS evaluation_canonical
    FROM experiment_recommendation_ranking_snapshot snapshot
    LEFT JOIN experiment_recommendation_evaluation_result evaluation
      ON recommendation_ranking_membership_contains(
           snapshot.source_membership_canonical,
           evaluation.evaluation_identity_canonical,
           evaluation.recommendation_evaluation_result_id)
    LEFT JOIN experiment_recommendation_evaluation_run run
      ON run.recommendation_evaluation_run_id =
         evaluation.recommendation_evaluation_run_id
    WHERE snapshot.ranking_snapshot_identity_version IS NULL
    GROUP BY snapshot.recommendation_ranking_snapshot_id,
             snapshot.source_membership_canonical
), classified AS (
    SELECT population.*,
        CASE
          WHEN declared_count = 0 THEN 'empty'
          WHEN declared_count IS NULL OR resolved_count <> declared_count OR
               invalid_count <> 0 THEN 'legacy_unverified'
          WHEN scoring_count = 1 AND evaluation_count = 1 THEN
               'verified_homogeneous'
          ELSE 'legacy_heterogeneous'
        END AS semantic_state
    FROM population
)
UPDATE experiment_recommendation_ranking_snapshot snapshot
SET ranking_snapshot_identity_version = 1,
    population_semantic_state = classified.semantic_state,
    scoring_semantic_canonical = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN classified.scoring_canonical END,
    scoring_semantic_hash = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN recommendation_semantic_tagged_fnv1a64(
            classified.scoring_canonical) END,
    scoring_semantic_version = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN 1 END,
    evaluation_semantic_canonical = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN classified.evaluation_canonical END,
    evaluation_semantic_hash = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN recommendation_semantic_tagged_fnv1a64(
            classified.evaluation_canonical) END,
    evaluation_semantic_version = CASE WHEN classified.semantic_state =
        'verified_homogeneous' THEN 1 END,
    distinct_scoring_semantic_count = classified.scoring_count,
    distinct_evaluation_semantic_count = classified.evaluation_count,
    homogeneity_validation_result = CASE classified.semantic_state
        WHEN 'verified_homogeneous' THEN 'verified_homogeneous'
        WHEN 'empty' THEN 'empty_population'
        WHEN 'legacy_heterogeneous' THEN CASE
            WHEN classified.scoring_count > 1 THEN
                'heterogeneous_scoring_semantics'
            ELSE 'heterogeneous_evaluation_semantics' END
        ELSE 'legacy_unverified' END
FROM classified
WHERE snapshot.recommendation_ranking_snapshot_id = classified.snapshot_id;

ALTER TABLE experiment_recommendation_ranking_snapshot
    ENABLE TRIGGER enforce_recommendation_ranking_snapshot_transition_trigger;

ALTER TABLE experiment_recommendation_ranking_snapshot
    ALTER COLUMN ranking_snapshot_identity_version SET NOT NULL,
    ALTER COLUMN population_semantic_state SET NOT NULL,
    ALTER COLUMN distinct_scoring_semantic_count SET NOT NULL,
    ALTER COLUMN distinct_evaluation_semantic_count SET NOT NULL,
    ALTER COLUMN homogeneity_validation_result SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_recommendation_ranking_snapshot'::regclass
          AND conname = 'recommendation_ranking_population_semantics_shape_check'
    ) THEN
        ALTER TABLE experiment_recommendation_ranking_snapshot
        ADD CONSTRAINT recommendation_ranking_population_semantics_shape_check
        CHECK (
          ranking_snapshot_identity_version IN (1,2) AND
          population_semantic_state IN (
            'verified_homogeneous','empty','legacy_heterogeneous',
            'legacy_unverified') AND
          distinct_scoring_semantic_count >= 0 AND
          distinct_evaluation_semantic_count >= 0 AND
          ((population_semantic_state = 'verified_homogeneous' AND
            scoring_semantic_canonical IS NOT NULL AND
            scoring_semantic_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
            scoring_semantic_version = 1 AND
            evaluation_semantic_canonical IS NOT NULL AND
            evaluation_semantic_hash ~ '^fnv1a64:[0-9a-f]{16}$' AND
            evaluation_semantic_version = 1 AND
            distinct_scoring_semantic_count = 1 AND
            distinct_evaluation_semantic_count = 1 AND
            homogeneity_validation_result = 'verified_homogeneous') OR
           (population_semantic_state = 'empty' AND
            scoring_semantic_canonical IS NULL AND scoring_semantic_hash IS NULL AND
            scoring_semantic_version IS NULL AND
            evaluation_semantic_canonical IS NULL AND
            evaluation_semantic_hash IS NULL AND
            evaluation_semantic_version IS NULL AND
            distinct_scoring_semantic_count = 0 AND
            distinct_evaluation_semantic_count = 0 AND
            homogeneity_validation_result = 'empty_population') OR
           (population_semantic_state IN (
                'legacy_heterogeneous','legacy_unverified') AND
            scoring_semantic_canonical IS NULL AND scoring_semantic_hash IS NULL AND
            scoring_semantic_version IS NULL AND
            evaluation_semantic_canonical IS NULL AND
            evaluation_semantic_hash IS NULL AND
            evaluation_semantic_version IS NULL))
        );
    END IF;
END $$;

CREATE OR REPLACE FUNCTION enforce_recommendation_ranking_snapshot_semantics_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    declared_count integer;
    resolved_count integer;
    invalid_count integer;
    scoring_count integer;
    evaluation_count integer;
    expected_identity text;
BEGIN
    IF TG_OP = 'UPDATE' AND
       (OLD.ranking_snapshot_identity_version, OLD.population_semantic_state,
        OLD.scoring_semantic_canonical, OLD.scoring_semantic_hash,
        OLD.scoring_semantic_version, OLD.evaluation_semantic_canonical,
        OLD.evaluation_semantic_hash, OLD.evaluation_semantic_version,
        OLD.distinct_scoring_semantic_count,
        OLD.distinct_evaluation_semantic_count,
        OLD.homogeneity_validation_result) IS NOT DISTINCT FROM
       (NEW.ranking_snapshot_identity_version, NEW.population_semantic_state,
        NEW.scoring_semantic_canonical, NEW.scoring_semantic_hash,
        NEW.scoring_semantic_version, NEW.evaluation_semantic_canonical,
        NEW.evaluation_semantic_hash, NEW.evaluation_semantic_version,
        NEW.distinct_scoring_semantic_count,
        NEW.distinct_evaluation_semantic_count,
        NEW.homogeneity_validation_result) THEN
        RETURN NEW;
    END IF;
    IF TG_OP = 'UPDATE' THEN
        RAISE EXCEPTION 'ranking snapshot semantic identity is immutable'
            USING ERRCODE = '55000';
    END IF;
    IF NEW.ranking_snapshot_identity_version <> 2 OR
       NEW.ranking_policy_hash <>
           recommendation_semantic_tagged_fnv1a64(NEW.ranking_policy_canonical) OR
       NEW.scope_hash <>
           recommendation_semantic_tagged_fnv1a64(NEW.scope_canonical) OR
       NEW.source_membership_hash <>
           recommendation_semantic_tagged_fnv1a64(
               NEW.source_membership_canonical) THEN
        RAISE EXCEPTION 'invalid Phase 3B ranking snapshot provenance'
            USING ERRCODE = '23514';
    END IF;
    declared_count := recommendation_ranking_membership_count_v1(
        NEW.source_membership_canonical);
    IF NEW.population_semantic_state = 'empty' THEN
        IF declared_count IS DISTINCT FROM 0 OR
           NEW.homogeneity_validation_result <> 'empty_population' THEN
            RAISE EXCEPTION 'invalid empty ranking semantic state'
                USING ERRCODE = '23514';
        END IF;
    ELSIF NEW.population_semantic_state = 'verified_homogeneous' THEN
        IF declared_count IS NULL OR declared_count <= 0 OR
           NEW.scoring_semantic_hash <>
               recommendation_semantic_tagged_fnv1a64(
                   NEW.scoring_semantic_canonical) OR
           NEW.evaluation_semantic_hash <>
               recommendation_semantic_tagged_fnv1a64(
                   NEW.evaluation_semantic_canonical) THEN
            RAISE EXCEPTION 'invalid homogeneous ranking semantic identity'
                USING ERRCODE = '23514';
        END IF;
        SELECT count(*)::integer,
               count(*) FILTER (WHERE NOT
                   recommendation_evaluation_result_semantics_valid_v1(
                       evaluation, run))::integer,
               count(DISTINCT recommendation_scoring_semantic_canonical_v1(
                   run.scoring_policy_canonical, run.scoring_version))::integer,
               count(DISTINCT recommendation_evaluation_semantic_canonical_v1(
                   run.evaluation_version, run.evaluator_version,
                   run.scoring_policy_canonical, run.scoring_version))::integer
          INTO resolved_count, invalid_count, scoring_count, evaluation_count
        FROM experiment_recommendation_evaluation_result evaluation
        JOIN experiment_recommendation_evaluation_run run
          ON run.recommendation_evaluation_run_id =
             evaluation.recommendation_evaluation_run_id
        WHERE recommendation_ranking_membership_contains(
            NEW.source_membership_canonical,
            evaluation.evaluation_identity_canonical,
            evaluation.recommendation_evaluation_result_id)
          AND recommendation_scoring_semantic_canonical_v1(
              run.scoring_policy_canonical, run.scoring_version) =
              NEW.scoring_semantic_canonical
          AND recommendation_evaluation_semantic_canonical_v1(
              run.evaluation_version, run.evaluator_version,
              run.scoring_policy_canonical, run.scoring_version) =
              NEW.evaluation_semantic_canonical;
        IF resolved_count <> declared_count OR invalid_count <> 0 OR
           scoring_count <> 1 OR evaluation_count <> 1 THEN
            RAISE EXCEPTION 'ranking population is not verified homogeneous'
                USING ERRCODE = '23514';
        END IF;
    ELSE
        RAISE EXCEPTION 'new ranking snapshot semantic state is not admissible'
            USING ERRCODE = '23514';
    END IF;
    expected_identity :=
        'experiment_recommendation_ranking_snapshot_identity_v2;policy=' ||
        octet_length(NEW.ranking_policy_canonical) || ':' ||
        NEW.ranking_policy_canonical || ';scope=' ||
        octet_length(NEW.scope_canonical) || ':' || NEW.scope_canonical ||
        ';limit=' || NEW.requested_limit || ';population_state=' ||
        NEW.population_semantic_state || ';scoring_semantic=' ||
        CASE WHEN NEW.population_semantic_state = 'empty' THEN 'NULL'
             ELSE octet_length(NEW.scoring_semantic_canonical) || ':' ||
                  NEW.scoring_semantic_canonical END ||
        ';evaluation_semantic=' ||
        CASE WHEN NEW.population_semantic_state = 'empty' THEN 'NULL'
             ELSE octet_length(NEW.evaluation_semantic_canonical) || ':' ||
                  NEW.evaluation_semantic_canonical END || ';membership=' ||
        octet_length(NEW.source_membership_canonical) || ':' ||
        NEW.source_membership_canonical;
    IF NEW.ranking_snapshot_identity_canonical <> expected_identity OR
       NEW.ranking_snapshot_identity_hash <>
           recommendation_semantic_tagged_fnv1a64(expected_identity) THEN
        RAISE EXCEPTION 'invalid Phase 3B ranking snapshot identity'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_recommendation_ranking_member_semantics_v2()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
BEGIN
    PERFORM 1
    FROM experiment_recommendation_ranking_snapshot snapshot
    JOIN experiment_recommendation_evaluation_result evaluation
      ON evaluation.recommendation_evaluation_result_id =
         NEW.recommendation_evaluation_result_id
    JOIN experiment_recommendation_evaluation_run run
      ON run.recommendation_evaluation_run_id =
         evaluation.recommendation_evaluation_run_id
    WHERE snapshot.recommendation_ranking_snapshot_id =
          NEW.recommendation_ranking_snapshot_id
      AND snapshot.population_semantic_state = 'verified_homogeneous'
      AND recommendation_evaluation_result_semantics_valid_v1(evaluation, run)
      AND snapshot.scoring_semantic_canonical =
          recommendation_scoring_semantic_canonical_v1(
              run.scoring_policy_canonical, run.scoring_version)
      AND snapshot.scoring_semantic_hash =
          recommendation_semantic_tagged_fnv1a64(
              snapshot.scoring_semantic_canonical)
      AND snapshot.evaluation_semantic_canonical =
          recommendation_evaluation_semantic_canonical_v1(
              run.evaluation_version, run.evaluator_version,
              run.scoring_policy_canonical, run.scoring_version)
      AND snapshot.evaluation_semantic_hash =
          recommendation_semantic_tagged_fnv1a64(
              snapshot.evaluation_semantic_canonical);
    IF NOT FOUND THEN
        RAISE EXCEPTION 'ranking member semantic identity differs from snapshot'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_ranking_member_semantics_check';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS recommendation_00_ranking_snapshot_semantics_v2_trigger
    ON experiment_recommendation_ranking_snapshot;
CREATE TRIGGER recommendation_00_ranking_snapshot_semantics_v2_trigger
BEFORE INSERT OR UPDATE ON experiment_recommendation_ranking_snapshot
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_ranking_snapshot_semantics_v2();

DROP TRIGGER IF EXISTS recommendation_00_ranking_member_semantics_v2_trigger
    ON experiment_recommendation_ranking_member;
CREATE TRIGGER recommendation_00_ranking_member_semantics_v2_trigger
BEFORE INSERT ON experiment_recommendation_ranking_member
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_ranking_member_semantics_v2();

CREATE INDEX IF NOT EXISTS recommendation_ranking_snapshot_semantic_state_idx
    ON experiment_recommendation_ranking_snapshot(
        population_semantic_state, recommendation_ranking_snapshot_id DESC);
CREATE INDEX IF NOT EXISTS recommendation_ranking_snapshot_scoring_semantic_idx
    ON experiment_recommendation_ranking_snapshot(
        scoring_semantic_hash, recommendation_ranking_snapshot_id DESC)
    WHERE scoring_semantic_hash IS NOT NULL;

REVOKE UPDATE, DELETE ON experiment_recommendation_ranking_snapshot FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_ranking_snapshot TO pqxx;
GRANT UPDATE (status,member_count,advisory_ready_count,blocked_count,
    non_actionable_count,completed_at,error_message,updated_at)
    ON experiment_recommendation_ranking_snapshot TO pqxx;
REVOKE UPDATE, DELETE ON experiment_recommendation_evaluation_result FROM pqxx;
REVOKE UPDATE, DELETE ON experiment_recommendation_evaluation_component FROM pqxx;

COMMENT ON COLUMN experiment_recommendation_ranking_snapshot.population_semantic_state IS
    'Phase 3B full source-population classification; only verified_homogeneous or empty is admissible for new snapshots.';
COMMENT ON COLUMN experiment_recommendation_ranking_snapshot.scoring_semantic_canonical IS
    'Authoritative common scoring semantic identity for the full pre-limit population; excludes profitability.';
COMMENT ON COLUMN experiment_recommendation_ranking_snapshot.evaluation_semantic_canonical IS
    'Authoritative common evaluation semantic identity for the full pre-limit population; excludes profitability.';
