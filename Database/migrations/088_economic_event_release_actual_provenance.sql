-- Persist authoritative release-actual observations without reusing the
-- secondary-provider actual fields stored with consensus evidence.  Every
-- observation has an explicit causal availability instant, immutable source
-- evidence, and revision identity.  Runtime features select only revision 0;
-- later revisions remain auditable evidence and can never replace the
-- first-release surprise.
CREATE TABLE economic_event_release_actual (
    economic_event_release_actual_id bigserial PRIMARY KEY,
    economic_event_id bigint NOT NULL
        REFERENCES economic_event(economic_event_id),

    source_agency text NOT NULL
        CHECK (source_agency <> ''),
    source_observation_id text NOT NULL
        CHECK (source_observation_id <> ''),

    publication_state text NOT NULL
        CHECK (publication_state IN ('initial', 'revision')),
    revision_sequence integer NOT NULL
        CHECK (revision_sequence >= 0),
    available_at timestamptz NOT NULL,
    retrieved_at timestamptz NOT NULL,

    source_url text NOT NULL
        CHECK (source_url ~ '^https://'),
    source_artifact_path text NOT NULL
        CHECK (source_artifact_path <> ''),
    source_artifact_sha256 text NOT NULL
        CHECK (source_artifact_sha256 ~ '^[0-9a-f]{64}$'),
    semantic_contract text NOT NULL
        CHECK (semantic_contract <> ''),
    source_provenance jsonb NOT NULL
        CHECK (jsonb_typeof(source_provenance) = 'object' AND
               source_provenance <> '{}'::jsonb),

    actual_raw text NOT NULL
        CHECK (actual_raw <> ''),
    actual_value_kind text NOT NULL
        CHECK (actual_value_kind IN ('scalar', 'range')),
    actual_value_low numeric NOT NULL,
    actual_value_high numeric,
    actual_canonical_value_low numeric NOT NULL,
    actual_canonical_value_high numeric,
    actual_unit text NOT NULL
        CHECK (actual_unit <> ''),
    actual_scale numeric NOT NULL
        CHECK (actual_scale > 0),
    actual_qualifier text,

    imported_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT economic_event_release_actual_source_observation_uq
        UNIQUE (source_agency, source_observation_id),
    CONSTRAINT economic_event_release_actual_revision_uq
        UNIQUE (economic_event_id, revision_sequence),
    CONSTRAINT economic_event_release_actual_publication_state_ck CHECK (
        (publication_state = 'initial' AND revision_sequence = 0) OR
        (publication_state = 'revision' AND revision_sequence > 0)
    ),
    CONSTRAINT economic_event_release_actual_retrieval_ck CHECK (
        retrieved_at >= available_at
    ),
    CONSTRAINT economic_event_release_actual_value_shape_ck CHECK (
        (actual_value_kind = 'scalar' AND
         actual_value_high IS NULL AND
         actual_canonical_value_high IS NULL AND
         actual_canonical_value_low = actual_value_low * actual_scale) OR
        (actual_value_kind = 'range' AND
         actual_value_high IS NOT NULL AND
         actual_canonical_value_high IS NOT NULL AND
         actual_value_low <= actual_value_high AND
         actual_canonical_value_low <= actual_canonical_value_high AND
         actual_canonical_value_low = actual_value_low * actual_scale AND
         actual_canonical_value_high = actual_value_high * actual_scale)
    )
);

CREATE INDEX economic_event_release_actual_available_idx
    ON economic_event_release_actual (economic_event_id, available_at);

CREATE OR REPLACE FUNCTION validate_economic_event_release_actual()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    official_source_agency text;
    official_event_timestamp timestamptz;
BEGIN
    -- Serialize observations for one authoritative event so concurrent
    -- inserts cannot evade revision-order validation.
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'economic_event_release_actual:' || NEW.economic_event_id::text,
            0));

    SELECT source_agency, event_timestamp_utc
    INTO official_source_agency, official_event_timestamp
    FROM economic_event
    WHERE economic_event_id = NEW.economic_event_id
    FOR KEY SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'economic event release actual references missing event'
            USING ERRCODE = 'foreign_key_violation';
    END IF;

    IF NEW.source_agency <> official_source_agency THEN
        RAISE EXCEPTION 'economic event release actual source agency mismatch'
            USING ERRCODE = 'check_violation';
    END IF;

    IF NEW.available_at < official_event_timestamp THEN
        RAISE EXCEPTION 'economic event release actual predates release event'
            USING ERRCODE = 'check_violation';
    END IF;

    IF EXISTS (
        SELECT 1
        FROM economic_event_release_actual existing
        WHERE existing.economic_event_id = NEW.economic_event_id
          AND (
              (existing.revision_sequence < NEW.revision_sequence AND
               existing.available_at > NEW.available_at) OR
              (existing.revision_sequence > NEW.revision_sequence AND
               existing.available_at < NEW.available_at)
          )
    ) THEN
        RAISE EXCEPTION 'economic event release actual revision order conflict'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER economic_event_release_actual_validation_trigger
BEFORE INSERT ON economic_event_release_actual
FOR EACH ROW EXECUTE FUNCTION validate_economic_event_release_actual();

CREATE OR REPLACE FUNCTION reject_economic_event_release_actual_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'economic event release actual observations are immutable';
END;
$$;

CREATE TRIGGER economic_event_release_actual_immutable_trigger
BEFORE UPDATE OR DELETE ON economic_event_release_actual
FOR EACH ROW EXECUTE FUNCTION reject_economic_event_release_actual_mutation();

-- Exactly one row can satisfy this view for an event because revision 0 is
-- unique.  No provider precedence or latest-row selection is performed.
CREATE VIEW economic_event_feature_release_actual AS
SELECT
    a.economic_event_release_actual_id,
    a.economic_event_id,
    a.available_at,
    a.actual_value_kind,
    a.actual_canonical_value_low,
    a.actual_canonical_value_high,
    a.actual_unit,
    a.actual_scale,
    a.actual_qualifier,
    a.source_agency,
    a.source_observation_id,
    a.source_artifact_path,
    a.source_artifact_sha256,
    a.semantic_contract,
    a.source_provenance
FROM economic_event_release_actual a
WHERE a.publication_state = 'initial'
  AND a.revision_sequence = 0;

COMMENT ON TABLE economic_event_release_actual IS
    'Immutable authoritative initial and revised actual observations with explicit point-in-time availability and source provenance.';
COMMENT ON COLUMN economic_event_release_actual.available_at IS
    'First instant at which this exact observation was published and knowable; never ingestion or retrieval time.';
COMMENT ON COLUMN economic_event_release_actual.retrieved_at IS
    'Time the immutable source artifact was retrieved, distinct from causal availability and database import time.';
COMMENT ON COLUMN economic_event_release_actual.revision_sequence IS
    'Source-backed sequence: zero is the initial release and positive values are later revisions.';
COMMENT ON VIEW economic_event_feature_release_actual IS
    'The unique provenance-certified initial actual eligible for release-surprise features; later revisions are deliberately excluded.';

-- Runtime is read-only for this evidence.  A future authoritative importer
-- must receive separately reviewed, column-limited INSERT authority.
GRANT SELECT ON economic_event_release_actual TO pqxx;
GRANT SELECT ON economic_event_feature_release_actual TO pqxx;
GRANT EXECUTE ON FUNCTION validate_economic_event_release_actual() TO pqxx;
GRANT EXECUTE ON FUNCTION reject_economic_event_release_actual_mutation() TO pqxx;
