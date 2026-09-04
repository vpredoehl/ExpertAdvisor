-- Generalize migration-088 release-actual evidence into an append-only,
-- provider-neutral observation ledger.  The legacy table and feature view
-- remain unchanged so width-75 behavior is not reinterpreted.  Only evidence
-- already carrying migration-088's explicit publication contract is eligible
-- for deterministic first-release selection; provider snapshots are retained
-- conservatively with system-observation availability only.

CREATE TABLE economic_event_actual_observation (
    economic_event_actual_observation_id bigserial PRIMARY KEY,
    economic_event_id bigint NOT NULL
        REFERENCES economic_event(economic_event_id),

    source_name text NOT NULL CHECK (source_name <> ''),
    source_role text NOT NULL
        CHECK (source_role IN ('authoritative', 'secondary')),
    source_native_event_id text,
    source_observation_id text NOT NULL
        CHECK (source_observation_id <> ''),
    evidence_key text NOT NULL CHECK (evidence_key <> ''),

    observation_kind text NOT NULL CHECK (
        observation_kind IN (
            'initial',
            'revision',
            'correction',
            'alternate_observation',
            'unclassified'
        )
    ),
    revision_sequence integer CHECK (revision_sequence >= 0),

    source_publication_at timestamptz,
    source_publication_time_status text NOT NULL CHECK (
        source_publication_time_status IN ('exact', 'unavailable')
    ),
    observed_at timestamptz NOT NULL,
    ingested_at timestamptz NOT NULL DEFAULT now(),
    availability_proof text NOT NULL CHECK (
        availability_proof IN ('source_publication', 'system_observation')
    ),
    proven_available_at timestamptz GENERATED ALWAYS AS (
        CASE availability_proof
            WHEN 'source_publication' THEN source_publication_at
            WHEN 'system_observation' THEN observed_at
        END
    ) STORED,

    source_url text CHECK (source_url IS NULL OR source_url ~ '^https://'),
    source_artifact_path text NOT NULL
        CHECK (source_artifact_path <> ''),
    source_artifact_sha256 text CHECK (
        source_artifact_sha256 IS NULL OR
        source_artifact_sha256 ~ '^[0-9a-f]{64}$'
    ),
    semantic_contract text NOT NULL CHECK (semantic_contract <> ''),
    source_provenance jsonb NOT NULL CHECK (
        jsonb_typeof(source_provenance) = 'object' AND
        source_provenance <> '{}'::jsonb
    ),

    actual_raw text NOT NULL CHECK (actual_raw <> ''),
    actual_value_kind text NOT NULL
        CHECK (actual_value_kind IN ('scalar', 'range')),
    actual_value_low numeric NOT NULL,
    actual_value_high numeric,
    actual_canonical_value_low numeric NOT NULL,
    actual_canonical_value_high numeric,
    actual_unit text NOT NULL CHECK (actual_unit <> ''),
    actual_scale numeric NOT NULL CHECK (actual_scale > 0),
    actual_qualifier text,

    legacy_release_actual_id bigint UNIQUE
        REFERENCES economic_event_release_actual(
            economic_event_release_actual_id),
    legacy_consensus_id bigint UNIQUE
        REFERENCES economic_event_consensus(economic_event_consensus_id),

    CONSTRAINT economic_event_actual_observation_evidence_uq
        UNIQUE (economic_event_id, source_name, evidence_key),
    CONSTRAINT economic_event_actual_observation_source_time_ck CHECK (
        (source_publication_time_status = 'exact' AND
         source_publication_at IS NOT NULL AND
         availability_proof = 'source_publication')
        OR
        (source_publication_time_status = 'unavailable' AND
         source_publication_at IS NULL AND
         availability_proof = 'system_observation')
    ),
    CONSTRAINT economic_event_actual_observation_sequence_ck CHECK (
        (observation_kind = 'initial' AND revision_sequence = 0)
        OR
        (observation_kind = 'revision' AND revision_sequence > 0)
        OR
        (observation_kind IN (
            'correction', 'alternate_observation', 'unclassified') AND
         revision_sequence IS NULL)
    ),
    CONSTRAINT economic_event_actual_observation_chronology_ck CHECK (
        observed_at <= ingested_at AND
        (source_publication_at IS NULL OR
         source_publication_at <= observed_at)
    ),
    CONSTRAINT economic_event_actual_observation_value_shape_ck CHECK (
        (actual_value_kind = 'scalar' AND
         actual_value_high IS NULL AND
         actual_canonical_value_high IS NULL AND
         actual_canonical_value_low = actual_value_low * actual_scale)
        OR
        (actual_value_kind = 'range' AND
         actual_value_high IS NOT NULL AND
         actual_canonical_value_high IS NOT NULL AND
         actual_value_low <= actual_value_high AND
         actual_canonical_value_low <= actual_canonical_value_high AND
         actual_canonical_value_low = actual_value_low * actual_scale AND
         actual_canonical_value_high = actual_value_high * actual_scale)
    )
);

CREATE INDEX economic_event_actual_observation_event_idx
    ON economic_event_actual_observation (
        economic_event_id,
        economic_event_actual_observation_id
    );
CREATE INDEX economic_event_actual_observation_chronology_idx
    ON economic_event_actual_observation (
        economic_event_id,
        proven_available_at,
        economic_event_actual_observation_id
    );
CREATE INDEX economic_event_actual_observation_first_release_idx
    ON economic_event_actual_observation (
        economic_event_id,
        source_publication_at,
        economic_event_actual_observation_id
    )
    WHERE source_role = 'authoritative'
      AND observation_kind = 'initial'
      AND source_publication_time_status = 'exact';
CREATE INDEX economic_event_actual_observation_cutoff_idx
    ON economic_event_actual_observation (
        proven_available_at,
        economic_event_id
    );
CREATE INDEX economic_event_actual_observation_source_idx
    ON economic_event_actual_observation (
        source_name,
        source_native_event_id,
        source_observation_id
    );

CREATE OR REPLACE FUNCTION validate_economic_event_actual_observation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    event_source_name text;
    event_release_at timestamptz;
    observation_available_at timestamptz;
BEGIN
    PERFORM pg_advisory_xact_lock(
        hashtextextended(
            'economic_event_actual_observation:' ||
            NEW.economic_event_id::text,
            0));

    SELECT source_agency, event_timestamp_utc
    INTO event_source_name, event_release_at
    FROM public.economic_event
    WHERE economic_event_id = NEW.economic_event_id
    FOR KEY SHARE;

    IF NOT FOUND THEN
        RAISE EXCEPTION 'economic event actual observation references missing event'
            USING ERRCODE = 'foreign_key_violation';
    END IF;

    IF NEW.source_role = 'authoritative' AND
       NEW.source_name <> event_source_name
    THEN
        RAISE EXCEPTION 'authoritative actual observation source mismatch'
            USING ERRCODE = 'check_violation';
    END IF;

    -- Generated columns are populated after BEFORE triggers.  Derive the
    -- candidate boundary from the submitted evidence instead of reading
    -- NEW.proven_available_at before PostgreSQL has computed it.
    observation_available_at := CASE NEW.availability_proof
        WHEN 'source_publication' THEN NEW.source_publication_at
        WHEN 'system_observation' THEN NEW.observed_at
    END;

    IF observation_available_at < event_release_at THEN
        RAISE EXCEPTION 'economic event actual observation predates release event'
            USING ERRCODE = 'check_violation';
    END IF;

    RETURN NEW;
END;
$$;

CREATE TRIGGER economic_event_actual_observation_validation_trigger
BEFORE INSERT ON economic_event_actual_observation
FOR EACH ROW EXECUTE FUNCTION
    validate_economic_event_actual_observation();

CREATE OR REPLACE FUNCTION reject_economic_event_actual_observation_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'economic event actual observations are immutable';
END;
$$;

CREATE TRIGGER economic_event_actual_observation_immutable_trigger
BEFORE UPDATE OR DELETE ON economic_event_actual_observation
FOR EACH ROW EXECUTE FUNCTION
    reject_economic_event_actual_observation_mutation();

-- Migration-088 rows already carry direct authoritative initial/revision
-- classification and an explicit publication-availability contract.  Copy
-- that evidence without changing the legacy rows or their feature view.
INSERT INTO economic_event_actual_observation (
    economic_event_id,
    source_name,
    source_role,
    source_native_event_id,
    source_observation_id,
    evidence_key,
    observation_kind,
    revision_sequence,
    source_publication_at,
    source_publication_time_status,
    observed_at,
    ingested_at,
    availability_proof,
    source_url,
    source_artifact_path,
    source_artifact_sha256,
    semantic_contract,
    source_provenance,
    actual_raw,
    actual_value_kind,
    actual_value_low,
    actual_value_high,
    actual_canonical_value_low,
    actual_canonical_value_high,
    actual_unit,
    actual_scale,
    actual_qualifier,
    legacy_release_actual_id
)
SELECT
    a.economic_event_id,
    a.source_agency,
    'authoritative',
    e.source_event_id,
    a.source_observation_id,
    'migration088:' || a.economic_event_release_actual_id::text,
    a.publication_state,
    a.revision_sequence,
    a.available_at,
    'exact',
    a.retrieved_at,
    a.imported_at,
    'source_publication',
    a.source_url,
    a.source_artifact_path,
    a.source_artifact_sha256,
    a.semantic_contract,
    a.source_provenance || jsonb_build_object(
        'provenance_migration', '090',
        'legacy_release_actual_id', a.economic_event_release_actual_id),
    a.actual_raw,
    a.actual_value_kind,
    a.actual_value_low,
    a.actual_value_high,
    a.actual_canonical_value_low,
    a.actual_canonical_value_high,
    a.actual_unit,
    a.actual_scale,
    a.actual_qualifier,
    a.economic_event_release_actual_id
FROM economic_event_release_actual a
JOIN economic_event e USING (economic_event_id)
ORDER BY a.economic_event_release_actual_id;

-- Provider actual snapshots are useful competing evidence, but their source
-- publication instant and initial/revision status were never established.
-- Preserve them as secondary alternate observations and make them knowable
-- only from their historical database observation/import instant.
INSERT INTO economic_event_actual_observation (
    economic_event_id,
    source_name,
    source_role,
    source_native_event_id,
    source_observation_id,
    evidence_key,
    observation_kind,
    revision_sequence,
    source_publication_at,
    source_publication_time_status,
    observed_at,
    ingested_at,
    availability_proof,
    source_url,
    source_artifact_path,
    source_artifact_sha256,
    semantic_contract,
    source_provenance,
    actual_raw,
    actual_value_kind,
    actual_value_low,
    actual_value_high,
    actual_canonical_value_low,
    actual_canonical_value_high,
    actual_unit,
    actual_scale,
    actual_qualifier,
    legacy_consensus_id
)
SELECT
    c.economic_event_id,
    c.consensus_source,
    'secondary',
    c.source_event_id::text,
    c.source_observation_id,
    'consensus-observation:' || c.economic_event_consensus_id::text,
    'alternate_observation',
    NULL,
    NULL,
    'unavailable',
    c.imported_at,
    c.imported_at,
    'system_observation',
    NULL,
    c.source_artifact_path,
    c.source_artifact_sha256,
    c.semantic_contract,
    c.provider_provenance || jsonb_build_object(
        'provenance_migration', '090',
        'legacy_consensus_id', c.economic_event_consensus_id,
        'publication_time_status', 'unavailable'),
    c.actual_raw,
    c.actual_value_kind,
    c.actual_value_low,
    c.actual_value_high,
    c.actual_canonical_value_low,
    c.actual_canonical_value_high,
    c.actual_unit,
    c.actual_scale,
    c.actual_qualifier,
    c.economic_event_consensus_id
FROM economic_event_consensus c
WHERE c.actual_parse_status = 'parsed'
ORDER BY c.economic_event_consensus_id;

-- The current canonical value is the latest proven authoritative value.  It
-- is an audit abstraction only; no existing consumer is redirected here.
CREATE VIEW economic_event_canonical_actual AS
SELECT DISTINCT ON (o.economic_event_id)
    o.economic_event_actual_observation_id,
    o.economic_event_id,
    o.actual_value_kind,
    o.actual_canonical_value_low,
    o.actual_canonical_value_high,
    o.actual_unit,
    o.actual_scale,
    o.actual_qualifier,
    o.proven_available_at,
    o.source_name,
    o.source_native_event_id,
    o.source_observation_id,
    o.observation_kind,
    o.revision_sequence
FROM economic_event_actual_observation o
WHERE o.source_role = 'authoritative'
  AND o.observation_kind IN ('initial', 'revision', 'correction')
ORDER BY
    o.economic_event_id,
    o.proven_available_at DESC,
    o.revision_sequence DESC NULLS LAST,
    o.economic_event_actual_observation_id DESC;

-- Selection is derived rather than designated.  Only an authoritative
-- observation explicitly classified as initial with an exact source
-- publication timestamp can qualify.  The earliest publication wins.  Equal
-- earliest publications with different semantic values fail closed.
CREATE VIEW economic_event_first_release_actual AS
WITH eligible AS (
    SELECT o.*
    FROM economic_event_actual_observation o
    JOIN economic_event e USING (economic_event_id)
    WHERE o.source_role = 'authoritative'
      AND o.source_name = e.source_agency
      AND o.observation_kind = 'initial'
      AND o.revision_sequence = 0
      AND o.source_publication_time_status = 'exact'
      AND o.availability_proof = 'source_publication'
),
earliest AS (
    SELECT economic_event_id, min(source_publication_at) AS first_at
    FROM eligible
    GROUP BY economic_event_id
),
earliest_assessment AS (
    SELECT
        q.economic_event_id,
        q.first_at,
        count(*) AS earliest_candidate_count,
        count(DISTINCT jsonb_build_array(
            q.actual_value_kind,
            q.actual_canonical_value_low,
            q.actual_canonical_value_high,
            q.actual_unit,
            q.actual_scale,
            q.actual_qualifier
        )) AS earliest_semantic_value_count
    FROM (
        SELECT e.*, x.first_at
        FROM eligible e
        JOIN earliest x USING (economic_event_id)
        WHERE e.source_publication_at = x.first_at
    ) q
    GROUP BY q.economic_event_id, q.first_at
),
selected AS (
    SELECT DISTINCT ON (o.economic_event_id) o.*
    FROM eligible o
    JOIN earliest e USING (economic_event_id)
    WHERE o.source_publication_at = e.first_at
    ORDER BY
        o.economic_event_id,
        o.source_name,
        o.source_observation_id,
        o.evidence_key,
        o.economic_event_actual_observation_id
),
possible_first_claims AS (
    SELECT
        s.economic_event_id,
        s.actual_value_kind,
        s.actual_canonical_value_low,
        s.actual_canonical_value_high,
        s.actual_unit,
        s.actual_scale,
        s.actual_qualifier
    FROM selected s
    UNION ALL
    SELECT
        o.economic_event_id,
        o.actual_value_kind,
        o.actual_canonical_value_low,
        o.actual_canonical_value_high,
        o.actual_unit,
        o.actual_scale,
        o.actual_qualifier
    FROM economic_event_actual_observation o
    JOIN economic_event e USING (economic_event_id)
    WHERE o.source_role = 'authoritative'
      AND o.source_name = e.source_agency
      AND (
          o.observation_kind = 'unclassified' OR
          (o.observation_kind = 'initial' AND
           o.source_publication_time_status = 'unavailable')
      )
),
possible_first_assessment AS (
    SELECT
        economic_event_id,
        count(DISTINCT jsonb_build_array(
            actual_value_kind,
            actual_canonical_value_low,
            actual_canonical_value_high,
            actual_unit,
            actual_scale,
            actual_qualifier
        )) AS possible_semantic_value_count
    FROM possible_first_claims
    GROUP BY economic_event_id
),
observation_counts AS (
    SELECT
        economic_event_id,
        count(*) AS observation_count,
        count(DISTINCT jsonb_build_array(
            actual_value_kind,
            actual_canonical_value_low,
            actual_canonical_value_high,
            actual_unit,
            actual_scale,
            actual_qualifier
        )) AS observed_semantic_value_count
    FROM economic_event_actual_observation
    GROUP BY economic_event_id
)
SELECT
    e.economic_event_id,
    e.currency,
    e.event_family,
    e.event_timestamp_utc AS event_release_at,
    e.source_agency AS canonical_source_agency,
    e.source_event_id AS canonical_source_event_id,
    CASE
        WHEN a.earliest_semantic_value_count = 1 AND
             coalesce(p.possible_semantic_value_count, 1) = 1
            THEN 'proven_first_release'
        WHEN coalesce(a.earliest_semantic_value_count, 0) > 1 OR
             coalesce(p.possible_semantic_value_count, 0) > 1
            THEN 'ambiguous'
        ELSE 'provenance_unavailable'
    END AS provenance_state,
    CASE
        WHEN a.earliest_semantic_value_count = 1 AND
             coalesce(p.possible_semantic_value_count, 1) = 1 AND
             a.earliest_candidate_count = 1
            THEN 'unique_authoritative_initial_at_earliest_source_publication'
        WHEN a.earliest_semantic_value_count = 1 AND
             coalesce(p.possible_semantic_value_count, 1) = 1 AND
             a.earliest_candidate_count > 1
            THEN 'corroborated_authoritative_initial_at_earliest_source_publication'
        WHEN a.earliest_semantic_value_count > 1
            THEN 'conflicting_authoritative_initials_at_same_earliest_publication'
        WHEN coalesce(p.possible_semantic_value_count, 0) > 1
            THEN 'unresolved_authoritative_possible_first_value_conflict'
        WHEN coalesce(c.observation_count, 0) = 0
            THEN 'no_actual_observation'
        ELSE 'no_authoritative_initial_with_exact_source_publication'
    END AS selection_reason,
    coalesce(c.observation_count, 0) AS observation_count,
    coalesce(c.observed_semantic_value_count, 0)
        AS observed_semantic_value_count,
    coalesce(a.earliest_candidate_count, 0) AS earliest_candidate_count,
    coalesce(p.possible_semantic_value_count, 0)
        AS possible_first_semantic_value_count,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.economic_event_actual_observation_id END
        AS first_release_observation_id,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_value_kind END AS first_release_value_kind,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_canonical_value_low END AS first_release_value_low,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_canonical_value_high END AS first_release_value_high,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_unit END AS first_release_unit,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_scale END AS first_release_scale,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.actual_qualifier END AS first_release_qualifier,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.source_publication_at END AS proven_available_at,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.source_name END AS first_release_source,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.source_native_event_id END AS first_release_source_native_event_id,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.source_observation_id END AS first_release_source_observation_id,
    CASE WHEN a.earliest_semantic_value_count = 1 AND
                   coalesce(p.possible_semantic_value_count, 1) = 1
        THEN s.evidence_key END AS first_release_evidence_key,
    ca.economic_event_actual_observation_id AS canonical_observation_id,
    ca.actual_value_kind AS canonical_value_kind,
    ca.actual_canonical_value_low AS canonical_value_low,
    ca.actual_canonical_value_high AS canonical_value_high,
    ca.actual_unit AS canonical_unit,
    ca.actual_qualifier AS canonical_qualifier,
    ca.proven_available_at AS canonical_available_at,
    CASE
        WHEN a.earliest_semantic_value_count IS DISTINCT FROM 1 OR
             coalesce(p.possible_semantic_value_count, 1) <> 1 OR
             ca.economic_event_actual_observation_id IS NULL
            THEN NULL
        ELSE jsonb_build_array(
            s.actual_value_kind,
            s.actual_canonical_value_low,
            s.actual_canonical_value_high,
            s.actual_unit,
            s.actual_scale,
            s.actual_qualifier
        ) IS DISTINCT FROM jsonb_build_array(
            ca.actual_value_kind,
            ca.actual_canonical_value_low,
            ca.actual_canonical_value_high,
            ca.actual_unit,
            ca.actual_scale,
            ca.actual_qualifier
        )
    END AS canonical_differs_from_first_release
FROM economic_event e
LEFT JOIN observation_counts c USING (economic_event_id)
LEFT JOIN earliest_assessment a USING (economic_event_id)
LEFT JOIN possible_first_assessment p USING (economic_event_id)
LEFT JOIN selected s USING (economic_event_id)
LEFT JOIN economic_event_canonical_actual ca USING (economic_event_id);

CREATE FUNCTION economic_event_first_release_actual_at(
    information_cutoff timestamptz)
RETURNS TABLE (
    economic_event_id bigint,
    event_release_at timestamptz,
    first_release_actual_value_kind text,
    first_release_actual_value_low numeric,
    first_release_actual_value_high numeric,
    first_release_actual_unit text,
    first_release_actual_scale numeric,
    first_release_actual_qualifier text,
    proven_available_at timestamptz,
    provenance_state text,
    selection_reason text,
    source_name text,
    source_native_event_id text,
    source_observation_id text,
    evidence_key text
)
LANGUAGE sql
STABLE
AS $$
    SELECT
        f.economic_event_id,
        f.event_release_at,
        f.first_release_value_kind,
        f.first_release_value_low,
        f.first_release_value_high,
        f.first_release_unit,
        f.first_release_scale,
        f.first_release_qualifier,
        f.proven_available_at,
        f.provenance_state,
        f.selection_reason,
        f.first_release_source,
        f.first_release_source_native_event_id,
        f.first_release_source_observation_id,
        f.first_release_evidence_key
    FROM public.economic_event_first_release_actual f
    WHERE f.provenance_state = 'proven_first_release'
      AND f.proven_available_at <= information_cutoff
    ORDER BY f.event_release_at, f.economic_event_id
$$;

COMMENT ON TABLE economic_event_actual_observation IS
    'Immutable provider-neutral actual observations; source publication, system observation, and database ingestion times remain distinct.';
COMMENT ON COLUMN economic_event_actual_observation.evidence_key IS
    'Deterministic observation-level evidence identity. Exact retries reuse it; a revision or changed artifact must use distinct evidence.';
COMMENT ON COLUMN economic_event_actual_observation.source_publication_at IS
    'Source publication instant when directly evidenced; NULL means it was not proved.';
COMMENT ON COLUMN economic_event_actual_observation.observed_at IS
    'First proved system retrieval/observation instant, never inferred from filesystem mtime.';
COMMENT ON COLUMN economic_event_actual_observation.ingested_at IS
    'Database ingestion instant, distinct from source publication and system observation.';
COMMENT ON COLUMN economic_event_actual_observation.proven_available_at IS
    'Conservative PIT boundary: exact source publication when proved, otherwise system observation time.';
COMMENT ON VIEW economic_event_first_release_actual IS
    'One deterministic provenance assessment per logical event. Conflicting earliest authoritative initial values fail closed as ambiguous.';
COMMENT ON VIEW economic_event_canonical_actual IS
    'Latest proven authoritative value for read-only audit; existing feature consumers are not redirected to this view.';
COMMENT ON FUNCTION economic_event_first_release_actual_at(timestamptz) IS
    'PIT-safe Phase-2 API: returns only proven first-release actuals whose proven availability is at or before the caller cutoff.';

GRANT SELECT ON economic_event_actual_observation TO pqxx;
GRANT SELECT ON economic_event_canonical_actual TO pqxx;
GRANT SELECT ON economic_event_first_release_actual TO pqxx;
GRANT EXECUTE ON FUNCTION
    economic_event_first_release_actual_at(timestamptz) TO pqxx;
GRANT EXECUTE ON FUNCTION
    validate_economic_event_actual_observation() TO pqxx;
GRANT EXECUTE ON FUNCTION
    reject_economic_event_actual_observation_mutation() TO pqxx;
