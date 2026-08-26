-- Phase 1 persists audited secondary-source consensus enrichment without
-- changing the authoritative economic_event release identity or any model
-- feature contract.  Rows are immutable after insertion; exact import retries
-- are recognized by the repository before it attempts an INSERT.
CREATE TABLE economic_event_consensus (
    economic_event_id bigint PRIMARY KEY
        REFERENCES economic_event(economic_event_id),

    consensus_source text NOT NULL
        CHECK (consensus_source <> ''),
    source_report_id bigint NOT NULL
        CHECK (source_report_id > 0),
    source_event_id bigint NOT NULL
        CHECK (source_event_id > 0),
    source_event_name text NOT NULL
        CHECK (source_event_name <> ''),
    source_period text NOT NULL
        CHECK (source_period <> ''),
    source_priority smallint NOT NULL
        CHECK (source_priority BETWEEN 1 AND 3),
    source_timestamp_epoch bigint NOT NULL
        CHECK (source_timestamp_epoch > 0),
    source_date timestamp without time zone NOT NULL,
    source_artifact_path text NOT NULL
        CHECK (source_artifact_path <> ''),
    match_rule text NOT NULL
        CHECK (match_rule <> ''),
    semantic_contract text NOT NULL
        CHECK (semantic_contract <> ''),

    forecast_raw text,
    forecast_parse_status text NOT NULL
        CHECK (forecast_parse_status IN ('missing', 'parsed')),
    forecast_value_kind text
        CHECK (forecast_value_kind IN ('scalar', 'range')),
    forecast_value_low numeric,
    forecast_value_high numeric,
    forecast_canonical_value_low numeric,
    forecast_canonical_value_high numeric,
    forecast_unit text,
    forecast_scale numeric,
    forecast_qualifier text,

    previous_raw text,
    previous_parse_status text NOT NULL
        CHECK (previous_parse_status IN ('missing', 'parsed')),
    previous_value_kind text
        CHECK (previous_value_kind IN ('scalar', 'range')),
    previous_value_low numeric,
    previous_value_high numeric,
    previous_canonical_value_low numeric,
    previous_canonical_value_high numeric,
    previous_unit text,
    previous_scale numeric,
    previous_qualifier text,

    actual_raw text,
    actual_parse_status text NOT NULL
        CHECK (actual_parse_status IN ('missing', 'parsed')),
    actual_value_kind text
        CHECK (actual_value_kind IN ('scalar', 'range')),
    actual_value_low numeric,
    actual_value_high numeric,
    actual_canonical_value_low numeric,
    actual_canonical_value_high numeric,
    actual_unit text,
    actual_scale numeric,
    actual_qualifier text,

    imported_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT economic_event_consensus_source_event_uq
        UNIQUE (consensus_source, source_event_id),

    CONSTRAINT economic_event_consensus_forecast_semantics_ck CHECK (
        (forecast_parse_status = 'missing' AND
         forecast_raw IS NULL AND forecast_value_kind IS NULL AND
         forecast_value_low IS NULL AND forecast_value_high IS NULL AND
         forecast_canonical_value_low IS NULL AND
         forecast_canonical_value_high IS NULL AND forecast_unit IS NULL AND
         forecast_scale IS NULL AND forecast_qualifier IS NULL)
        OR
        (forecast_parse_status = 'parsed' AND forecast_raw IS NOT NULL AND
         forecast_raw <> '' AND forecast_value_kind IS NOT NULL AND
         forecast_value_low IS NOT NULL AND
         forecast_canonical_value_low IS NOT NULL AND
         forecast_unit IS NOT NULL AND forecast_unit <> '' AND
         forecast_scale IS NOT NULL AND forecast_scale > 0 AND
         ((forecast_value_kind = 'scalar' AND
           forecast_value_high IS NULL AND
           forecast_canonical_value_high IS NULL)
          OR
          (forecast_value_kind = 'range' AND
           forecast_value_high IS NOT NULL AND
           forecast_canonical_value_high IS NOT NULL AND
           forecast_value_low <= forecast_value_high AND
           forecast_canonical_value_low <=
               forecast_canonical_value_high)))
    ),

    CONSTRAINT economic_event_consensus_previous_semantics_ck CHECK (
        (previous_parse_status = 'missing' AND
         previous_raw IS NULL AND previous_value_kind IS NULL AND
         previous_value_low IS NULL AND previous_value_high IS NULL AND
         previous_canonical_value_low IS NULL AND
         previous_canonical_value_high IS NULL AND previous_unit IS NULL AND
         previous_scale IS NULL AND previous_qualifier IS NULL)
        OR
        (previous_parse_status = 'parsed' AND previous_raw IS NOT NULL AND
         previous_raw <> '' AND previous_value_kind IS NOT NULL AND
         previous_value_low IS NOT NULL AND
         previous_canonical_value_low IS NOT NULL AND
         previous_unit IS NOT NULL AND previous_unit <> '' AND
         previous_scale IS NOT NULL AND previous_scale > 0 AND
         ((previous_value_kind = 'scalar' AND
           previous_value_high IS NULL AND
           previous_canonical_value_high IS NULL)
          OR
          (previous_value_kind = 'range' AND
           previous_value_high IS NOT NULL AND
           previous_canonical_value_high IS NOT NULL AND
           previous_value_low <= previous_value_high AND
           previous_canonical_value_low <=
               previous_canonical_value_high)))
    ),

    CONSTRAINT economic_event_consensus_actual_semantics_ck CHECK (
        (actual_parse_status = 'missing' AND
         actual_raw IS NULL AND actual_value_kind IS NULL AND
         actual_value_low IS NULL AND actual_value_high IS NULL AND
         actual_canonical_value_low IS NULL AND
         actual_canonical_value_high IS NULL AND actual_unit IS NULL AND
         actual_scale IS NULL AND actual_qualifier IS NULL)
        OR
        (actual_parse_status = 'parsed' AND actual_raw IS NOT NULL AND
         actual_raw <> '' AND actual_value_kind IS NOT NULL AND
         actual_value_low IS NOT NULL AND
         actual_canonical_value_low IS NOT NULL AND
         actual_unit IS NOT NULL AND actual_unit <> '' AND
         actual_scale IS NOT NULL AND actual_scale > 0 AND
         ((actual_value_kind = 'scalar' AND
           actual_value_high IS NULL AND
           actual_canonical_value_high IS NULL)
          OR
          (actual_value_kind = 'range' AND
           actual_value_high IS NOT NULL AND
           actual_canonical_value_high IS NOT NULL AND
           actual_value_low <= actual_value_high AND
           actual_canonical_value_low <= actual_canonical_value_high)))
    )
);

CREATE INDEX economic_event_consensus_source_report_idx
    ON economic_event_consensus (consensus_source, source_report_id);

CREATE OR REPLACE FUNCTION reject_economic_event_consensus_mutation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    RAISE EXCEPTION 'economic event consensus enrichments are immutable';
END;
$$;

CREATE TRIGGER economic_event_consensus_immutable_trigger
BEFORE UPDATE OR DELETE ON economic_event_consensus
FOR EACH ROW EXECUTE FUNCTION reject_economic_event_consensus_mutation();

COMMENT ON TABLE economic_event_consensus IS
    'Immutable audited secondary-source consensus enrichment joined to the authoritative economic_event calendar.';
COMMENT ON COLUMN economic_event_consensus.economic_event_id IS
    'One selected consensus enrichment for one authoritative release; release identity remains in economic_event.';
COMMENT ON COLUMN economic_event_consensus.source_timestamp_epoch IS
    'Secondary-source timestamp preserved exactly as Unix epoch seconds.';
COMMENT ON COLUMN economic_event_consensus.source_date IS
    'Secondary-source display date/time preserved without inventing a timezone.';
COMMENT ON COLUMN economic_event_consensus.semantic_contract IS
    'Versioned identity of the already-audited raw-to-semantic parsing contract preserved by the importer.';
COMMENT ON COLUMN economic_event_consensus.forecast_canonical_value_low IS
    'Exact canonical low/scalar value; percentage values remain percentage points and count scales are expanded.';

GRANT SELECT, INSERT ON economic_event_consensus TO pqxx;
GRANT EXECUTE ON FUNCTION reject_economic_event_consensus_mutation() TO pqxx;
