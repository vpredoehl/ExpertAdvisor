-- Phase 4B Step 2: immutable advisory ranking snapshots over persisted Step 1
-- evaluation evidence. Ranking owns no experiment, review, or scheduler state.

CREATE TABLE IF NOT EXISTS experiment_recommendation_ranking_snapshot (
    recommendation_ranking_snapshot_id bigserial PRIMARY KEY,
    status text NOT NULL CHECK (status IN ('running','completed','failed')),
    ranking_snapshot_identity_canonical text NOT NULL
        CHECK (btrim(ranking_snapshot_identity_canonical) <> ''),
    ranking_snapshot_identity_hash text NOT NULL
        CHECK (btrim(ranking_snapshot_identity_hash) <> ''),
    ranking_policy_canonical text NOT NULL
        CHECK (btrim(ranking_policy_canonical) <> ''),
    ranking_policy_hash text NOT NULL CHECK (btrim(ranking_policy_hash) <> ''),
    ranking_version integer NOT NULL CHECK (ranking_version > 0),
    scope_type text NOT NULL CHECK (scope_type IN (
        'evaluation_run','recommendation_scan','symbol','horizon','family',
        'symbol_horizon','global')),
    scope_canonical text NOT NULL CHECK (btrim(scope_canonical) <> ''),
    scope_hash text NOT NULL CHECK (btrim(scope_hash) <> ''),
    evaluation_run_filter bigint REFERENCES
        experiment_recommendation_evaluation_run(recommendation_evaluation_run_id),
    recommendation_scan_filter bigint REFERENCES
        experiment_recommendation_scan(recommendation_scan_id),
    symbol_filter text,
    horizon_filter integer CHECK (horizon_filter IS NULL OR horizon_filter > 0),
    family_filter text,
    requested_limit integer NOT NULL CHECK (
        requested_limit > 0 AND requested_limit <= 1000),
    source_membership_canonical text NOT NULL
        CHECK (btrim(source_membership_canonical) <> ''),
    source_membership_hash text NOT NULL
        CHECK (btrim(source_membership_hash) <> ''),
    member_count integer NOT NULL DEFAULT 0 CHECK (member_count >= 0),
    advisory_ready_count integer NOT NULL DEFAULT 0
        CHECK (advisory_ready_count >= 0),
    blocked_count integer NOT NULL DEFAULT 0 CHECK (blocked_count >= 0),
    non_actionable_count integer NOT NULL DEFAULT 0
        CHECK (non_actionable_count >= 0),
    started_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    error_message text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_ranking_snapshot_identity_uidx
        UNIQUE (ranking_snapshot_identity_canonical),
    CONSTRAINT recommendation_ranking_snapshot_scope_shape_check CHECK (
        (scope_type='evaluation_run' AND evaluation_run_filter IS NOT NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NULL
            AND horizon_filter IS NULL AND family_filter IS NULL)
        OR (scope_type='recommendation_scan' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NOT NULL AND symbol_filter IS NULL
            AND horizon_filter IS NULL AND family_filter IS NULL)
        OR (scope_type='symbol' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NOT NULL
            AND btrim(symbol_filter)<>'' AND horizon_filter IS NULL
            AND family_filter IS NULL)
        OR (scope_type='horizon' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NULL
            AND horizon_filter IS NOT NULL AND family_filter IS NULL)
        OR (scope_type='family' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NULL
            AND horizon_filter IS NULL AND family_filter IS NOT NULL
            AND btrim(family_filter)<>'')
        OR (scope_type='symbol_horizon' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NOT NULL
            AND btrim(symbol_filter)<>'' AND horizon_filter IS NOT NULL
            AND family_filter IS NULL)
        OR (scope_type='global' AND evaluation_run_filter IS NULL
            AND recommendation_scan_filter IS NULL AND symbol_filter IS NULL
            AND horizon_filter IS NULL AND family_filter IS NULL)
    ),
    CONSTRAINT recommendation_ranking_snapshot_lifecycle_check CHECK (
        (status='running' AND completed_at IS NULL AND error_message IS NULL
            AND member_count=0 AND advisory_ready_count=0
            AND blocked_count=0 AND non_actionable_count=0)
        OR (status='completed' AND completed_at IS NOT NULL
            AND error_message IS NULL
            AND member_count=advisory_ready_count+blocked_count+
                non_actionable_count)
        OR (status='failed' AND completed_at IS NOT NULL
            AND error_message IS NOT NULL AND btrim(error_message)<>''
            AND member_count=0 AND advisory_ready_count=0
            AND blocked_count=0 AND non_actionable_count=0)
    )
);

CREATE TABLE IF NOT EXISTS experiment_recommendation_ranking_member (
    recommendation_ranking_member_id bigserial PRIMARY KEY,
    recommendation_ranking_snapshot_id bigint NOT NULL REFERENCES
        experiment_recommendation_ranking_snapshot(
            recommendation_ranking_snapshot_id),
    recommendation_evaluation_result_id bigint NOT NULL REFERENCES
        experiment_recommendation_evaluation_result(
            recommendation_evaluation_result_id),
    recommendation_id bigint NOT NULL REFERENCES
        experiment_recommendation(recommendation_id),
    recommendation_semantic_hash text NOT NULL
        CHECK (btrim(recommendation_semantic_hash)<>''),
    evaluation_identity_hash text NOT NULL
        CHECK (btrim(evaluation_identity_hash)<>''),
    bucket text NOT NULL CHECK (bucket IN (
        'advisory_ready','blocked','non_actionable')),
    bucket_rank integer NOT NULL CHECK (bucket_rank > 0),
    global_ordinal integer NOT NULL CHECK (global_ordinal > 0),
    final_score double precision,
    disposition text NOT NULL CHECK (disposition IN (
        'advisory_ready','insufficient_evidence',
        'blocked_pending_duplicate','blocked_active_duplicate',
        'completed_duplicate','stale_source_evidence',
        'unsupported_recommendation_family','invalid_persisted_evidence')),
    tie_break_primary text NOT NULL CHECK (btrim(tie_break_primary)<>''),
    tie_break_semantic_hash text NOT NULL
        CHECK (btrim(tie_break_semantic_hash)<>''),
    tie_break_evaluation_hash text NOT NULL
        CHECK (btrim(tie_break_evaluation_hash)<>''),
    inclusion_reason text NOT NULL CHECK (btrim(inclusion_reason)<>''),
    block_reason text,
    top_positive_component text,
    top_penalty_component text,
    symbol text NOT NULL CHECK (btrim(symbol)<>''),
    horizon integer NOT NULL CHECK (horizon > 0),
    family text NOT NULL CHECK (btrim(family)<>''),
    source_value_canonical text NOT NULL,
    proposed_value_canonical text NOT NULL,
    source_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    source_model_id bigint REFERENCES model(model_id),
    source_analysis_id bigint REFERENCES experiment_analysis_result(analysis_id),
    recommendation_scan_id bigint NOT NULL REFERENCES
        experiment_recommendation_scan(recommendation_scan_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT recommendation_ranking_member_result_uidx UNIQUE
        (recommendation_ranking_snapshot_id,recommendation_evaluation_result_id),
    CONSTRAINT recommendation_ranking_member_bucket_rank_uidx UNIQUE
        (recommendation_ranking_snapshot_id,bucket,bucket_rank),
    CONSTRAINT recommendation_ranking_member_ordinal_uidx UNIQUE
        (recommendation_ranking_snapshot_id,global_ordinal),
    CONSTRAINT recommendation_ranking_member_shape_check CHECK (
        (bucket='advisory_ready' AND disposition='advisory_ready'
            AND final_score IS NOT NULL AND final_score>=0.0 AND final_score<=1.0
            AND block_reason IS NULL)
        OR (bucket='blocked' AND disposition IN (
                'blocked_pending_duplicate','blocked_active_duplicate',
                'completed_duplicate') AND final_score IS NULL
            AND block_reason IS NOT NULL AND btrim(block_reason)<>'')
        OR (bucket='non_actionable' AND disposition IN (
                'insufficient_evidence','stale_source_evidence',
                'unsupported_recommendation_family','invalid_persisted_evidence')
            AND final_score IS NULL AND block_reason IS NOT NULL
            AND btrim(block_reason)<>'')
    ),
    CONSTRAINT recommendation_ranking_member_finite_check CHECK (
        final_score IS NULL OR final_score NOT IN (
            'NaN'::double precision,'Infinity'::double precision,
            '-Infinity'::double precision))
);

CREATE OR REPLACE FUNCTION recommendation_ranking_membership_contains(
    membership_canonical text,
    target_evaluation_identity text,
    target_evaluation_result_id bigint)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
STRICT
PARALLEL SAFE
SECURITY INVOKER
SET search_path TO pg_catalog
AS $$
DECLARE
    membership_bytes bytea := convert_to(membership_canonical, 'UTF8');
    target_identity_bytes bytea := convert_to(target_evaluation_identity, 'UTF8');
    header_bytes bytea := convert_to(
        'experiment_recommendation_ranking_membership_v1;count=', 'UTF8');
    delimiter_bytes bytea := convert_to(';', 'UTF8');
    colon_bytes bytea := convert_to(':', 'UTF8');
    cursor_position integer;
    delimiter_offset integer;
    member_count_text text;
    member_count_numeric numeric;
    member_count integer;
    member_ordinal integer;
    member_prefix_bytes bytea;
    identity_length_text text;
    identity_length_numeric numeric;
    identity_length integer;
    member_identity_bytes bytea;
    result_prefix_bytes bytea;
    member_result_id_text text;
    member_result_id_numeric numeric;
    target_found boolean := false;
    total_length integer := octet_length(membership_bytes);
BEGIN
    IF substring(membership_bytes FROM 1 FOR octet_length(header_bytes)) <>
       header_bytes THEN
        RETURN false;
    END IF;
    cursor_position := octet_length(header_bytes) + 1;
    delimiter_offset := position(
        delimiter_bytes IN substring(membership_bytes FROM cursor_position));
    IF delimiter_offset = 0 THEN
        member_count_text := convert_from(
            substring(membership_bytes FROM cursor_position), 'UTF8');
        cursor_position := total_length + 1;
    ELSE
        member_count_text := convert_from(substring(
            membership_bytes FROM cursor_position FOR delimiter_offset - 1),
            'UTF8');
        cursor_position := cursor_position + delimiter_offset - 1;
    END IF;
    IF member_count_text !~ '^(0|[1-9][0-9]*)$'
       OR length(member_count_text) > 4 THEN
        RETURN false;
    END IF;
    member_count_numeric := member_count_text::numeric;
    IF member_count_numeric > 1000 THEN
        RETURN false;
    END IF;
    member_count := member_count_numeric::integer;
    IF member_count = 0 THEN
        RETURN false;
    END IF;
    IF cursor_position > total_length THEN
        RETURN false;
    END IF;

    FOR member_ordinal IN 0..member_count - 1 LOOP
        member_prefix_bytes := convert_to(
            ';member[' || member_ordinal::text || '].evaluation_identity=',
            'UTF8');
        IF substring(membership_bytes FROM cursor_position
                     FOR octet_length(member_prefix_bytes)) <>
           member_prefix_bytes THEN
            RETURN false;
        END IF;
        cursor_position := cursor_position + octet_length(member_prefix_bytes);
        delimiter_offset := position(
            colon_bytes IN substring(membership_bytes FROM cursor_position));
        IF delimiter_offset = 0 THEN
            RETURN false;
        END IF;
        identity_length_text := convert_from(substring(
            membership_bytes FROM cursor_position FOR delimiter_offset - 1),
            'UTF8');
        IF identity_length_text !~ '^(0|[1-9][0-9]*)$'
           OR length(identity_length_text) > 10 THEN
            RETURN false;
        END IF;
        identity_length_numeric := identity_length_text::numeric;
        IF identity_length_numeric > total_length THEN
            RETURN false;
        END IF;
        identity_length := identity_length_numeric::integer;
        cursor_position := cursor_position + delimiter_offset;
        member_identity_bytes := substring(
            membership_bytes FROM cursor_position FOR identity_length);
        IF octet_length(member_identity_bytes) <> identity_length THEN
            RETURN false;
        END IF;
        cursor_position := cursor_position + identity_length;

        result_prefix_bytes := convert_to(
            ';member[' || member_ordinal::text || '].evaluation_result_id=',
            'UTF8');
        IF substring(membership_bytes FROM cursor_position
                     FOR octet_length(result_prefix_bytes)) <>
           result_prefix_bytes THEN
            RETURN false;
        END IF;
        cursor_position := cursor_position + octet_length(result_prefix_bytes);
        IF member_ordinal < member_count - 1 THEN
            delimiter_offset := position(delimiter_bytes IN substring(
                membership_bytes FROM cursor_position));
            IF delimiter_offset = 0 THEN
                RETURN false;
            END IF;
            member_result_id_text := convert_from(substring(
                membership_bytes FROM cursor_position FOR delimiter_offset - 1),
                'UTF8');
            cursor_position := cursor_position + delimiter_offset - 1;
        ELSE
            member_result_id_text := convert_from(
                substring(membership_bytes FROM cursor_position), 'UTF8');
            cursor_position := total_length + 1;
        END IF;
        IF member_result_id_text !~ '^[1-9][0-9]*$'
           OR length(member_result_id_text) > 19 THEN
            RETURN false;
        END IF;
        member_result_id_numeric := member_result_id_text::numeric;
        IF member_result_id_numeric > 9223372036854775807::numeric THEN
            RETURN false;
        END IF;
        IF member_identity_bytes = target_identity_bytes
           AND member_result_id_numeric = target_evaluation_result_id THEN
            target_found := true;
        END IF;
    END LOOP;
    RETURN target_found AND cursor_position = total_length + 1;
EXCEPTION
    WHEN invalid_text_representation OR numeric_value_out_of_range OR
         character_not_in_repertoire THEN
        RETURN false;
END;
$$;

CREATE OR REPLACE FUNCTION recommendation_ranking_tie_break_matches(
    tie_break_primary text,
    evaluation_disposition text,
    evaluation_final_score double precision)
RETURNS boolean
LANGUAGE plpgsql
IMMUTABLE
PARALLEL SAFE
SECURITY INVOKER
SET search_path TO pg_catalog
AS $$
DECLARE
    parsed_score double precision;
BEGIN
    IF evaluation_final_score IS NOT NULL THEN
        IF left(tie_break_primary, 6) <> 'score:' THEN
            RETURN false;
        END IF;
        parsed_score := substring(tie_break_primary FROM 7)::double precision;
        RETURN parsed_score NOT IN (
                   'NaN'::double precision,
                   'Infinity'::double precision,
                   '-Infinity'::double precision)
               AND parsed_score = evaluation_final_score;
    END IF;
    RETURN tie_break_primary = CASE evaluation_disposition
        WHEN 'blocked_pending_duplicate' THEN 'priority:0'
        WHEN 'blocked_active_duplicate' THEN 'priority:1'
        WHEN 'completed_duplicate' THEN 'priority:2'
        WHEN 'insufficient_evidence' THEN 'priority:0'
        WHEN 'stale_source_evidence' THEN 'priority:1'
        WHEN 'unsupported_recommendation_family' THEN 'priority:2'
        WHEN 'invalid_persisted_evidence' THEN 'priority:3'
        ELSE NULL
    END;
EXCEPTION
    WHEN invalid_text_representation OR numeric_value_out_of_range THEN
        RETURN false;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_recommendation_ranking_member_insert()
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
    JOIN experiment_recommendation recommendation
      ON recommendation.recommendation_id = evaluation.recommendation_id
    WHERE snapshot.recommendation_ranking_snapshot_id =
          NEW.recommendation_ranking_snapshot_id
      AND (
          snapshot.status = 'running'
          OR (snapshot.status = 'completed' AND EXISTS (
              SELECT 1
              FROM experiment_recommendation_ranking_member existing
              WHERE existing.recommendation_ranking_snapshot_id =
                    NEW.recommendation_ranking_snapshot_id
                AND existing.recommendation_evaluation_result_id =
                    NEW.recommendation_evaluation_result_id
          ))
      )
      AND evaluation.recommendation_id = NEW.recommendation_id
      AND evaluation.recommendation_semantic_hash =
          NEW.recommendation_semantic_hash
      AND evaluation.evaluation_identity_hash = NEW.evaluation_identity_hash
      AND evaluation.final_score IS NOT DISTINCT FROM NEW.final_score
      AND evaluation.disposition = NEW.disposition
      AND evaluation.recommendation_scan_id = NEW.recommendation_scan_id
      AND evaluation.source_experiment_id = NEW.source_experiment_id
      AND evaluation.source_model_id IS NOT DISTINCT FROM NEW.source_model_id
      AND evaluation.source_analysis_id IS NOT DISTINCT FROM NEW.source_analysis_id
      AND (
          (snapshot.scope_type = 'evaluation_run'
              AND evaluation.recommendation_evaluation_run_id =
                  snapshot.evaluation_run_filter)
          OR (snapshot.scope_type = 'recommendation_scan'
              AND evaluation.recommendation_scan_id =
                  snapshot.recommendation_scan_filter)
          OR (snapshot.scope_type = 'symbol'
              AND recommendation.source_symbol = snapshot.symbol_filter)
          OR (snapshot.scope_type = 'horizon'
              AND recommendation.source_prediction_horizon =
                  snapshot.horizon_filter)
          OR (snapshot.scope_type = 'family'
              AND recommendation.changed_parameter = snapshot.family_filter)
          OR (snapshot.scope_type = 'symbol_horizon'
              AND recommendation.source_symbol = snapshot.symbol_filter
              AND recommendation.source_prediction_horizon =
                  snapshot.horizon_filter)
          OR snapshot.scope_type = 'global'
      )
      AND recommendation_ranking_membership_contains(
          snapshot.source_membership_canonical,
          evaluation.evaluation_identity_canonical,
          evaluation.recommendation_evaluation_result_id)
      AND (
          EXISTS (
              SELECT 1
              FROM experiment_recommendation_ranking_member existing
              WHERE existing.recommendation_ranking_snapshot_id =
                    NEW.recommendation_ranking_snapshot_id
                AND existing.recommendation_evaluation_result_id =
                    NEW.recommendation_evaluation_result_id
          )
          OR (
              SELECT count(*)
              FROM experiment_recommendation_ranking_member existing
              WHERE existing.recommendation_ranking_snapshot_id =
                    NEW.recommendation_ranking_snapshot_id
          ) < snapshot.requested_limit
      )
      AND NEW.bucket_rank <= snapshot.requested_limit
      AND NEW.global_ordinal <= snapshot.requested_limit
      AND recommendation.source_symbol = NEW.symbol
      AND recommendation.source_prediction_horizon = NEW.horizon
      AND recommendation.changed_parameter = NEW.family
      AND recommendation.source_value_canonical = NEW.source_value_canonical
      AND recommendation.proposed_value_canonical = NEW.proposed_value_canonical
      AND NEW.tie_break_semantic_hash = NEW.recommendation_semantic_hash
      AND NEW.tie_break_evaluation_hash = NEW.evaluation_identity_hash
      AND recommendation_ranking_tie_break_matches(
          NEW.tie_break_primary,
          evaluation.disposition,
          evaluation.final_score)
      AND NEW.inclusion_reason = CASE NEW.bucket
          WHEN 'advisory_ready' THEN 'included_advisory_ready'
          ELSE 'included_for_advisory_inspection'
      END
      AND NEW.block_reason IS NOT DISTINCT FROM CASE NEW.bucket
          WHEN 'advisory_ready' THEN NULL::text
          ELSE evaluation.reason_code
      END
      AND NEW.top_positive_component IS NOT DISTINCT FROM (
          SELECT component.component_name
          FROM experiment_recommendation_evaluation_component component
          WHERE component.recommendation_evaluation_result_id =
                evaluation.recommendation_evaluation_result_id
            AND NOT component.is_penalty
          ORDER BY component.weighted_contribution DESC,
                   component.component_name ASC
          LIMIT 1
      )
      AND NEW.top_penalty_component IS NOT DISTINCT FROM (
          SELECT component.component_name
          FROM experiment_recommendation_evaluation_component component
          WHERE component.recommendation_evaluation_result_id =
                evaluation.recommendation_evaluation_result_id
            AND component.is_penalty
          ORDER BY component.weighted_contribution DESC,
                   component.component_name ASC
          LIMIT 1
      )
    FOR UPDATE OF snapshot;
    IF NOT FOUND THEN
        RAISE EXCEPTION 'invalid or terminal recommendation ranking membership'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_ranking_member_authority_check';
    END IF;
    RETURN NEW;
END;
$$;

CREATE OR REPLACE FUNCTION enforce_recommendation_ranking_snapshot_transition()
RETURNS trigger
LANGUAGE plpgsql
SECURITY INVOKER
AS $$
DECLARE
    persisted_member_count integer;
    persisted_advisory_ready_count integer;
    persisted_blocked_count integer;
    persisted_non_actionable_count integer;
BEGIN
    IF TG_OP = 'INSERT' THEN
        IF NEW.status <> 'running' THEN
            RAISE EXCEPTION 'recommendation ranking snapshot must start running'
                USING ERRCODE = '23514',
                      CONSTRAINT =
                          'recommendation_ranking_snapshot_initial_status_check';
        END IF;
        RETURN NEW;
    END IF;
    IF OLD.status <> 'running' OR NEW.status NOT IN ('completed', 'failed') THEN
        RAISE EXCEPTION 'invalid recommendation ranking snapshot transition'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_ranking_snapshot_transition_check';
    END IF;

    SELECT count(*)::integer,
           count(*) FILTER (WHERE bucket = 'advisory_ready')::integer,
           count(*) FILTER (WHERE bucket = 'blocked')::integer,
           count(*) FILTER (WHERE bucket = 'non_actionable')::integer
    INTO persisted_member_count,
         persisted_advisory_ready_count,
         persisted_blocked_count,
         persisted_non_actionable_count
    FROM experiment_recommendation_ranking_member
    WHERE recommendation_ranking_snapshot_id =
          OLD.recommendation_ranking_snapshot_id;

    IF (NEW.status = 'completed' AND (
            NEW.member_count <> persisted_member_count
            OR NEW.advisory_ready_count <> persisted_advisory_ready_count
            OR NEW.blocked_count <> persisted_blocked_count
            OR NEW.non_actionable_count <> persisted_non_actionable_count))
       OR (NEW.status = 'failed' AND persisted_member_count <> 0) THEN
        RAISE EXCEPTION 'recommendation ranking snapshot counts do not match members'
            USING ERRCODE = '23514',
                  CONSTRAINT = 'recommendation_ranking_snapshot_member_count_check';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    ranking_schema text := current_schema();
BEGIN
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_ranking_member_insert() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        ranking_schema,
        ranking_schema);
    EXECUTE format(
        'ALTER FUNCTION %I.enforce_recommendation_ranking_snapshot_transition() '
        'SET search_path TO pg_catalog, %I, pg_temp',
        ranking_schema,
        ranking_schema);
END $$;

DROP TRIGGER IF EXISTS enforce_recommendation_ranking_member_insert_trigger
    ON experiment_recommendation_ranking_member;
CREATE TRIGGER enforce_recommendation_ranking_member_insert_trigger
BEFORE INSERT ON experiment_recommendation_ranking_member
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_ranking_member_insert();

DROP TRIGGER IF EXISTS enforce_recommendation_ranking_snapshot_transition_trigger
    ON experiment_recommendation_ranking_snapshot;
CREATE TRIGGER enforce_recommendation_ranking_snapshot_transition_trigger
BEFORE INSERT OR UPDATE ON experiment_recommendation_ranking_snapshot
FOR EACH ROW EXECUTE FUNCTION enforce_recommendation_ranking_snapshot_transition();

CREATE INDEX IF NOT EXISTS recommendation_ranking_snapshot_status_idx
    ON experiment_recommendation_ranking_snapshot(
        status,recommendation_ranking_snapshot_id DESC);
CREATE INDEX IF NOT EXISTS recommendation_ranking_snapshot_hash_idx
    ON experiment_recommendation_ranking_snapshot(
        ranking_snapshot_identity_hash,recommendation_ranking_snapshot_id DESC);
CREATE INDEX IF NOT EXISTS recommendation_ranking_member_result_idx
    ON experiment_recommendation_ranking_member(
        recommendation_evaluation_result_id,recommendation_ranking_snapshot_id);

REVOKE UPDATE, DELETE ON experiment_recommendation_ranking_snapshot FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_ranking_snapshot TO pqxx;
GRANT UPDATE (status,member_count,advisory_ready_count,blocked_count,
    non_actionable_count,completed_at,error_message,updated_at)
    ON experiment_recommendation_ranking_snapshot TO pqxx;
REVOKE UPDATE, DELETE ON experiment_recommendation_ranking_member FROM pqxx;
GRANT SELECT, INSERT ON experiment_recommendation_ranking_member TO pqxx;

DO $$
DECLARE
    sequence_name text;
BEGIN
    FOREACH sequence_name IN ARRAY ARRAY[
        pg_get_serial_sequence('experiment_recommendation_ranking_snapshot',
                               'recommendation_ranking_snapshot_id'),
        pg_get_serial_sequence('experiment_recommendation_ranking_member',
                               'recommendation_ranking_member_id')
    ]
    LOOP
        EXECUTE format('GRANT USAGE, SELECT ON SEQUENCE %s TO pqxx',
                       sequence_name);
    END LOOP;
END $$;
