-- Freeze the complete model-facing economic-calendar evidence universe once
-- and bind experiments/models to its deterministic identity. Historical
-- experiments and models remain NULL-bound and retain legacy live-corpus
-- behavior; this migration never invents a retrospective binding.

CREATE TABLE IF NOT EXISTS economic_calendar_snapshot (
    economic_calendar_snapshot_id bigserial PRIMARY KEY,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finalized_at timestamptz,
    snapshot_state text NOT NULL DEFAULT 'creating'
        CHECK (snapshot_state IN ('creating', 'finalized')),
    hash_contract_version integer NOT NULL DEFAULT 1
        CHECK (hash_contract_version = 1),
    content_hash text NOT NULL
        CHECK (content_hash ~ '^fnv1a64:[0-9a-f]{16}$'),
    created_by text NOT NULL CHECK (created_by <> ''),
    creation_note text,
    canonical_event_count bigint,
    selected_consensus_count bigint,
    release_actual_count bigint,
    proven_first_release_actual_count bigint,
    provenance_unavailable_count bigint,
    ambiguous_first_release_count bigint,
    source_family_counts jsonb,
    CONSTRAINT economic_calendar_snapshot_finalization_shape_ck CHECK (
        (snapshot_state = 'creating' AND finalized_at IS NULL AND
         canonical_event_count IS NULL AND selected_consensus_count IS NULL AND
         release_actual_count IS NULL AND
         proven_first_release_actual_count IS NULL AND
         provenance_unavailable_count IS NULL AND
         ambiguous_first_release_count IS NULL AND
         source_family_counts IS NULL)
        OR
        (snapshot_state = 'finalized' AND finalized_at IS NOT NULL AND
         canonical_event_count >= 0 AND selected_consensus_count >= 0 AND
         release_actual_count >= 0 AND
         proven_first_release_actual_count >= 0 AND
         provenance_unavailable_count >= 0 AND
         ambiguous_first_release_count >= 0 AND
         jsonb_typeof(source_family_counts) = 'object'))
);

CREATE UNIQUE INDEX IF NOT EXISTS economic_calendar_snapshot_content_hash_uq
    ON economic_calendar_snapshot(content_hash);

CREATE TABLE IF NOT EXISTS economic_calendar_snapshot_event (
    economic_calendar_snapshot_id bigint NOT NULL
        REFERENCES economic_calendar_snapshot(economic_calendar_snapshot_id),
    economic_event_id bigint NOT NULL,
    canonical_event_order bigint NOT NULL CHECK (canonical_event_order > 0),
    currency text NOT NULL,
    event_family text NOT NULL,
    event_timestamp_utc timestamptz NOT NULL,
    source_agency text NOT NULL,
    source_event_id text,
    source_url text NOT NULL,
    reference_period text,
    event_importance smallint NOT NULL,
    historical_time_confidence text NOT NULL,
    source_release_date date,
    source_release_time time without time zone,
    source_timezone text,
    PRIMARY KEY (economic_calendar_snapshot_id, economic_event_id)
);

CREATE UNIQUE INDEX IF NOT EXISTS
    economic_calendar_snapshot_event_canonical_order_uq
    ON economic_calendar_snapshot_event(
        economic_calendar_snapshot_id, canonical_event_order);

CREATE INDEX IF NOT EXISTS economic_calendar_snapshot_event_range_idx
    ON economic_calendar_snapshot_event(
        economic_calendar_snapshot_id, currency, event_timestamp_utc,
        economic_event_id);

CREATE TABLE IF NOT EXISTS economic_calendar_snapshot_consensus (
    economic_calendar_snapshot_id bigint NOT NULL,
    economic_event_id bigint NOT NULL,
    economic_event_consensus_id bigint NOT NULL,
    consensus_value_low numeric NOT NULL,
    consensus_value_high numeric,
    consensus_value_kind text NOT NULL,
    consensus_unit text NOT NULL,
    consensus_scale numeric NOT NULL,
    consensus_qualifier text,
    consensus_source text NOT NULL,
    source_report_id bigint,
    source_event_id bigint NOT NULL,
    source_observation_id text NOT NULL,
    source_release_date date NOT NULL,
    source_artifact_path text NOT NULL,
    source_artifact_sha256 text,
    candidate_classification text NOT NULL,
    match_rule text NOT NULL,
    semantic_contract text NOT NULL,
    provider_provenance jsonb NOT NULL,
    provider_observed_at timestamptz,
    forecast_available_at timestamptz,
    source_retrieved_at timestamptz,
    forecast_availability_proof text,
    PRIMARY KEY (economic_calendar_snapshot_id, economic_event_id),
    UNIQUE (economic_calendar_snapshot_id, economic_event_consensus_id),
    FOREIGN KEY (economic_calendar_snapshot_id, economic_event_id)
        REFERENCES economic_calendar_snapshot_event(
            economic_calendar_snapshot_id, economic_event_id)
);

CREATE TABLE IF NOT EXISTS economic_calendar_snapshot_release_actual (
    economic_calendar_snapshot_id bigint NOT NULL,
    economic_event_id bigint NOT NULL,
    economic_event_release_actual_id bigint NOT NULL,
    available_at timestamptz NOT NULL,
    actual_value_kind text NOT NULL,
    actual_canonical_value_low numeric NOT NULL,
    actual_canonical_value_high numeric,
    actual_unit text NOT NULL,
    actual_scale numeric NOT NULL,
    actual_qualifier text,
    source_agency text NOT NULL,
    source_observation_id text NOT NULL,
    source_artifact_path text NOT NULL,
    source_artifact_sha256 text NOT NULL,
    semantic_contract text NOT NULL,
    source_provenance jsonb NOT NULL,
    PRIMARY KEY (economic_calendar_snapshot_id, economic_event_id),
    UNIQUE (economic_calendar_snapshot_id, economic_event_release_actual_id),
    FOREIGN KEY (economic_calendar_snapshot_id, economic_event_id)
        REFERENCES economic_calendar_snapshot_event(
            economic_calendar_snapshot_id, economic_event_id)
);

CREATE TABLE IF NOT EXISTS economic_calendar_snapshot_first_release_actual (
    economic_calendar_snapshot_id bigint NOT NULL,
    economic_event_id bigint NOT NULL,
    provenance_state text NOT NULL CHECK (provenance_state IN (
        'proven_first_release', 'provenance_unavailable', 'ambiguous')),
    selection_reason text NOT NULL,
    observation_count bigint NOT NULL CHECK (observation_count >= 0),
    observed_semantic_value_count bigint NOT NULL
        CHECK (observed_semantic_value_count >= 0),
    earliest_candidate_count bigint NOT NULL
        CHECK (earliest_candidate_count >= 0),
    possible_first_semantic_value_count bigint NOT NULL
        CHECK (possible_first_semantic_value_count >= 0),
    first_release_observation_id bigint,
    first_release_value_kind text,
    first_release_value_low numeric,
    first_release_value_high numeric,
    first_release_unit text,
    first_release_scale numeric,
    first_release_qualifier text,
    proven_available_at timestamptz,
    first_release_source text,
    first_release_source_native_event_id text,
    first_release_source_observation_id text,
    first_release_evidence_key text,
    PRIMARY KEY (economic_calendar_snapshot_id, economic_event_id),
    FOREIGN KEY (economic_calendar_snapshot_id, economic_event_id)
        REFERENCES economic_calendar_snapshot_event(
            economic_calendar_snapshot_id, economic_event_id),
    CONSTRAINT economic_calendar_snapshot_first_release_shape_ck CHECK (
        (provenance_state = 'proven_first_release' AND
         first_release_observation_id IS NOT NULL AND
         first_release_value_kind IS NOT NULL AND
         first_release_value_low IS NOT NULL AND
         first_release_unit IS NOT NULL AND
         first_release_scale IS NOT NULL AND
         proven_available_at IS NOT NULL AND
         first_release_source IS NOT NULL AND
         first_release_source_observation_id IS NOT NULL AND
         first_release_evidence_key IS NOT NULL)
        OR
        (provenance_state <> 'proven_first_release' AND
         first_release_observation_id IS NULL AND
         first_release_value_kind IS NULL AND
         first_release_value_low IS NULL AND
         first_release_value_high IS NULL AND
         first_release_unit IS NULL AND
         first_release_scale IS NULL AND
         first_release_qualifier IS NULL AND
         proven_available_at IS NULL AND
         first_release_source IS NULL AND
         first_release_source_native_event_id IS NULL AND
         first_release_source_observation_id IS NULL AND
         first_release_evidence_key IS NULL))
);

CREATE OR REPLACE FUNCTION enforce_economic_calendar_snapshot_content_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    parent_state text;
BEGIN
    IF TG_OP <> 'INSERT' THEN
        RAISE EXCEPTION 'economic calendar snapshot content is immutable';
    END IF;
    SELECT snapshot_state INTO parent_state
      FROM economic_calendar_snapshot
     WHERE economic_calendar_snapshot_id =
           NEW.economic_calendar_snapshot_id
     FOR KEY SHARE;
    IF parent_state IS DISTINCT FROM 'creating' THEN
        RAISE EXCEPTION
            'economic calendar snapshot content requires creating snapshot';
    END IF;
    RETURN NEW;
END;
$$;

DO $$
DECLARE
    relation_name text;
BEGIN
    FOREACH relation_name IN ARRAY ARRAY[
        'economic_calendar_snapshot_event',
        'economic_calendar_snapshot_consensus',
        'economic_calendar_snapshot_release_actual',
        'economic_calendar_snapshot_first_release_actual'
    ]
    LOOP
        EXECUTE format('DROP TRIGGER IF EXISTS %I ON %I',
            'calendar_snapshot_content_immutable', relation_name);
        EXECUTE format(
            'CREATE TRIGGER %I BEFORE INSERT OR UPDATE OR DELETE ON %I '
            'FOR EACH ROW EXECUTE FUNCTION '
            'enforce_economic_calendar_snapshot_content_immutable()',
            'calendar_snapshot_content_immutable', relation_name);
    END LOOP;
END $$;

CREATE OR REPLACE FUNCTION enforce_economic_calendar_snapshot_immutable()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    IF TG_OP = 'DELETE' THEN
        RAISE EXCEPTION 'economic calendar snapshots are immutable';
    END IF;
    IF OLD.snapshot_state = 'creating' AND NEW.snapshot_state = 'finalized' AND
       NEW.economic_calendar_snapshot_id = OLD.economic_calendar_snapshot_id AND
       NEW.created_at = OLD.created_at AND
       NEW.hash_contract_version = OLD.hash_contract_version AND
       NEW.content_hash = OLD.content_hash AND
       NEW.created_by = OLD.created_by AND
       NEW.creation_note IS NOT DISTINCT FROM OLD.creation_note
    THEN
        RETURN NEW;
    END IF;
    RAISE EXCEPTION 'economic calendar snapshots are immutable after creation';
END;
$$;

DROP TRIGGER IF EXISTS economic_calendar_snapshot_immutable_trigger
    ON economic_calendar_snapshot;
CREATE TRIGGER economic_calendar_snapshot_immutable_trigger
BEFORE UPDATE OR DELETE ON economic_calendar_snapshot
FOR EACH ROW EXECUTE FUNCTION enforce_economic_calendar_snapshot_immutable();

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS economic_calendar_snapshot_id bigint,
    ADD COLUMN IF NOT EXISTS economic_calendar_snapshot_hash text;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_economic_calendar_snapshot_fkey,
    DROP CONSTRAINT IF EXISTS experiment_economic_calendar_snapshot_shape_ck;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_economic_calendar_snapshot_fkey
        FOREIGN KEY (economic_calendar_snapshot_id)
        REFERENCES economic_calendar_snapshot(economic_calendar_snapshot_id),
    ADD CONSTRAINT experiment_economic_calendar_snapshot_shape_ck CHECK (
        (economic_calendar_snapshot_id IS NULL AND
         economic_calendar_snapshot_hash IS NULL)
        OR
        (economic_calendar_snapshot_id IS NOT NULL AND
         economic_calendar_snapshot_hash ~ '^fnv1a64:[0-9a-f]{16}$'));

CREATE OR REPLACE FUNCTION enforce_experiment_economic_calendar_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    persisted_hash text;
    persisted_state text;
BEGIN
    IF TG_OP = 'UPDATE' AND
       (OLD.economic_calendar_snapshot_id IS NOT NULL OR
        OLD.economic_calendar_snapshot_hash IS NOT NULL) AND
       (NEW.economic_calendar_snapshot_id IS DISTINCT FROM
            OLD.economic_calendar_snapshot_id OR
        NEW.economic_calendar_snapshot_hash IS DISTINCT FROM
            OLD.economic_calendar_snapshot_hash)
    THEN
        RAISE EXCEPTION
            'experiment economic calendar snapshot binding is immutable';
    END IF;
    IF TG_OP = 'UPDATE' AND
       OLD.economic_calendar_snapshot_id IS NULL AND
       NEW.economic_calendar_snapshot_id IS NOT NULL AND
       (OLD.status <> 'pending' OR OLD.phase <> 'train' OR
        OLD.started_at IS NOT NULL OR OLD.last_model_id IS NOT NULL)
    THEN
        RAISE EXCEPTION
            'scientifically active experiment cannot acquire a snapshot';
    END IF;
    IF NEW.economic_calendar_snapshot_id IS NOT NULL THEN
        SELECT content_hash, snapshot_state
          INTO persisted_hash, persisted_state
          FROM economic_calendar_snapshot
         WHERE economic_calendar_snapshot_id =
               NEW.economic_calendar_snapshot_id;
        IF NOT FOUND OR persisted_state <> 'finalized' OR
           persisted_hash IS DISTINCT FROM
               NEW.economic_calendar_snapshot_hash
        THEN
            RAISE EXCEPTION
                'experiment economic calendar snapshot identity invalid';
        END IF;
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS experiment_economic_calendar_snapshot_trigger
    ON experiment;
CREATE TRIGGER experiment_economic_calendar_snapshot_trigger
BEFORE INSERT OR UPDATE OF economic_calendar_snapshot_id,
    economic_calendar_snapshot_hash ON experiment
FOR EACH ROW EXECUTE FUNCTION enforce_experiment_economic_calendar_snapshot();

DROP INDEX IF EXISTS experiment_unique_identity_uidx;
CREATE UNIQUE INDEX experiment_unique_identity_uidx
    ON experiment (
        symbol, prediction_horizon, c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs, checkpoint_interval, train_start, train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1), donchian20_mode,
        donchian_lookback, feature_warmup_scope, feature_ablation_mask,
        resume_expand_input_width, training_objective_hash,
        COALESCE(model_input_width, -1),
        COALESCE(model_input_semantic_layout_version, -1),
        COALESCE(economic_calendar_snapshot_id, -1),
        COALESCE(economic_calendar_snapshot_hash, ''),
        duplicate_nonce
    ) WHERE status <> 'cancelled';

ALTER TABLE model
    ADD COLUMN IF NOT EXISTS economic_calendar_snapshot_id bigint,
    ADD COLUMN IF NOT EXISTS economic_calendar_snapshot_hash text;

ALTER TABLE model
    DROP CONSTRAINT IF EXISTS model_economic_calendar_snapshot_fkey,
    DROP CONSTRAINT IF EXISTS model_economic_calendar_snapshot_shape_ck;
ALTER TABLE model
    ADD CONSTRAINT model_economic_calendar_snapshot_fkey
        FOREIGN KEY (economic_calendar_snapshot_id)
        REFERENCES economic_calendar_snapshot(economic_calendar_snapshot_id),
    ADD CONSTRAINT model_economic_calendar_snapshot_shape_ck CHECK (
        (economic_calendar_snapshot_id IS NULL AND
         economic_calendar_snapshot_hash IS NULL)
        OR
        (economic_calendar_snapshot_id IS NOT NULL AND
         economic_calendar_snapshot_hash ~ '^fnv1a64:[0-9a-f]{16}$'));

CREATE OR REPLACE FUNCTION enforce_model_economic_calendar_snapshot()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    experiment_snapshot_id bigint;
    experiment_snapshot_hash text;
BEGIN
    IF TG_OP = 'UPDATE' AND
       (OLD.economic_calendar_snapshot_id IS NOT NULL OR
        OLD.economic_calendar_snapshot_hash IS NOT NULL) AND
       (NEW.economic_calendar_snapshot_id IS DISTINCT FROM
            OLD.economic_calendar_snapshot_id OR
        NEW.economic_calendar_snapshot_hash IS DISTINCT FROM
            OLD.economic_calendar_snapshot_hash)
    THEN
        RAISE EXCEPTION 'model economic calendar snapshot identity is immutable';
    END IF;
    IF NEW.experiment_id IS NOT NULL THEN
        SELECT economic_calendar_snapshot_id,
               economic_calendar_snapshot_hash
          INTO experiment_snapshot_id, experiment_snapshot_hash
          FROM experiment WHERE experiment_id = NEW.experiment_id;
        IF NOT FOUND THEN
            RAISE EXCEPTION 'model references missing experiment';
        END IF;
        IF NEW.economic_calendar_snapshot_id IS NULL AND
           NEW.economic_calendar_snapshot_hash IS NULL
        THEN
            NEW.economic_calendar_snapshot_id := experiment_snapshot_id;
            NEW.economic_calendar_snapshot_hash := experiment_snapshot_hash;
        ELSIF NEW.economic_calendar_snapshot_id IS DISTINCT FROM
                  experiment_snapshot_id OR
              NEW.economic_calendar_snapshot_hash IS DISTINCT FROM
                  experiment_snapshot_hash
        THEN
            RAISE EXCEPTION
                'model and experiment economic calendar snapshot mismatch';
        END IF;
    ELSIF NEW.economic_calendar_snapshot_id IS NOT NULL THEN
        RAISE EXCEPTION
            'unlinked model cannot carry economic calendar snapshot identity';
    END IF;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS model_economic_calendar_snapshot_trigger ON model;
CREATE TRIGGER model_economic_calendar_snapshot_trigger
BEFORE INSERT OR UPDATE OF experiment_id, economic_calendar_snapshot_id,
    economic_calendar_snapshot_hash ON model
FOR EACH ROW EXECUTE FUNCTION enforce_model_economic_calendar_snapshot();

COMMENT ON TABLE economic_calendar_snapshot IS
    'Reusable immutable model-facing economic-calendar evidence universes. Content identity is FNV-1a-64 over versioned canonical rows, never creation time.';
COMMENT ON TABLE economic_calendar_snapshot_event IS
    'Materialized canonical event content; copying is required because legacy economic_event rows lack database immutability enforcement.';
COMMENT ON TABLE economic_calendar_snapshot_consensus IS
    'Materialized selected consensus identity, value, provenance, and historical availability.';
COMMENT ON TABLE economic_calendar_snapshot_release_actual IS
    'Materialized migration-088 initial release-actual projection retained for existing feature channels.';
COMMENT ON TABLE economic_calendar_snapshot_first_release_actual IS
    'Materialized deterministic migration-090 first-release assessment and selected provenance; PIT gating still applies to proven_available_at.';
COMMENT ON COLUMN experiment.economic_calendar_snapshot_id IS
    'Immutable frozen calendar binding. NULL denotes explicit pre-092 live-corpus compatibility behavior.';
COMMENT ON COLUMN model.economic_calendar_snapshot_id IS
    'Model-local copy of its experiment calendar snapshot identity for resume and inference fail-closed validation.';

GRANT SELECT, INSERT, UPDATE ON economic_calendar_snapshot TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE
    economic_calendar_snapshot_economic_calendar_snapshot_id_seq TO pqxx;
GRANT SELECT, INSERT ON economic_calendar_snapshot_event TO pqxx;
GRANT SELECT, INSERT ON economic_calendar_snapshot_consensus TO pqxx;
GRANT SELECT, INSERT ON economic_calendar_snapshot_release_actual TO pqxx;
GRANT SELECT, INSERT ON economic_calendar_snapshot_first_release_actual TO pqxx;
GRANT EXECUTE ON FUNCTION
    enforce_economic_calendar_snapshot_content_immutable() TO pqxx;
GRANT EXECUTE ON FUNCTION
    enforce_economic_calendar_snapshot_immutable() TO pqxx;
GRANT EXECUTE ON FUNCTION
    enforce_experiment_economic_calendar_snapshot() TO pqxx;
GRANT EXECUTE ON FUNCTION
    enforce_model_economic_calendar_snapshot() TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON model TO pqxx;
