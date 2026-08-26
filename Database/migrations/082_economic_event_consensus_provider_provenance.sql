-- Extend the phase-one OANDA enrichment table into a provider-neutral,
-- provenance-preserving consensus-observation store.  Existing rows are
-- retained exactly; no consensus value is rewritten or manufactured.

ALTER TABLE economic_event_consensus
    DROP CONSTRAINT economic_event_consensus_pkey,
    DROP CONSTRAINT economic_event_consensus_source_event_uq,
    DROP CONSTRAINT economic_event_consensus_source_report_id_check,
    DROP CONSTRAINT economic_event_consensus_source_event_id_check,
    DROP CONSTRAINT economic_event_consensus_source_period_check,
    DROP CONSTRAINT economic_event_consensus_source_priority_check,
    DROP CONSTRAINT economic_event_consensus_source_timestamp_epoch_check,
    ALTER COLUMN source_report_id DROP NOT NULL,
    ALTER COLUMN source_period DROP NOT NULL,
    ALTER COLUMN source_priority DROP NOT NULL,
    ALTER COLUMN source_timestamp_epoch DROP NOT NULL,
    ALTER COLUMN source_date DROP NOT NULL,
    ADD COLUMN economic_event_consensus_id bigserial,
    ADD COLUMN source_observation_id text,
    ADD COLUMN source_release_date date,
    ADD COLUMN source_artifact_sha256 text,
    ADD COLUMN candidate_classification text,
    ADD COLUMN provider_provenance jsonb;

-- Deterministically describe all already-persisted phase-one OANDA rows.
-- source_event_id is an observation-level OANDA identity, so it is sufficient
-- to construct the new source-scoped observation identity without consulting
-- mutable external state.
ALTER TABLE economic_event_consensus
    DISABLE TRIGGER economic_event_consensus_immutable_trigger;

UPDATE economic_event_consensus
SET source_observation_id = 'oanda:event:' || source_event_id::text,
    source_release_date = source_date::date,
    candidate_classification = CASE forecast_parse_status
        WHEN 'parsed' THEN 'oanda_populated_initial'
        ELSE 'oanda_matched_blank'
    END,
    provider_provenance = jsonb_build_object(
        'provider', 'OANDA',
        'oanda_report_id', source_report_id,
        'oanda_event_id', source_event_id,
        'source_artifact', source_artifact_path,
        'phase', 'phase_1_initial_population'
    );

ALTER TABLE economic_event_consensus
    ENABLE TRIGGER economic_event_consensus_immutable_trigger;

ALTER TABLE economic_event_consensus
    ALTER COLUMN economic_event_consensus_id SET NOT NULL,
    ALTER COLUMN source_observation_id SET NOT NULL,
    ALTER COLUMN source_release_date SET NOT NULL,
    ALTER COLUMN candidate_classification SET NOT NULL,
    ALTER COLUMN provider_provenance SET NOT NULL,
    ADD CONSTRAINT economic_event_consensus_pkey
        PRIMARY KEY (economic_event_consensus_id),
    ADD CONSTRAINT economic_event_consensus_provider_observation_uq
        UNIQUE (consensus_source, source_observation_id),
    ADD CONSTRAINT economic_event_consensus_source_report_id_ck
        CHECK (source_report_id IS NULL OR source_report_id > 0),
    ADD CONSTRAINT economic_event_consensus_source_event_id_ck
        CHECK (source_event_id <> 0),
    ADD CONSTRAINT economic_event_consensus_source_period_ck
        CHECK (source_period IS NULL OR source_period <> ''),
    ADD CONSTRAINT economic_event_consensus_source_priority_ck
        CHECK (
            source_priority IS NULL OR source_priority BETWEEN 1 AND 3
        ),
    ADD CONSTRAINT economic_event_consensus_source_timestamp_epoch_ck
        CHECK (
            source_timestamp_epoch IS NULL OR source_timestamp_epoch > 0
        ),
    ADD CONSTRAINT economic_event_consensus_source_observation_id_ck
        CHECK (source_observation_id <> ''),
    ADD CONSTRAINT economic_event_consensus_source_artifact_sha256_ck
        CHECK (
            source_artifact_sha256 IS NULL OR
            source_artifact_sha256 ~ '^[0-9a-f]{64}$'
        ),
    ADD CONSTRAINT economic_event_consensus_candidate_classification_ck
        CHECK (
            candidate_classification IN (
                'oanda_populated_initial',
                'oanda_matched_blank',
                'myfxbook_oanda_blank_fill',
                'myfxbook_jolts_gap_fill'
            )
        ),
    ADD CONSTRAINT economic_event_consensus_provider_contract_ck
        CHECK (
            (
                consensus_source = 'OANDA' AND
                source_report_id IS NOT NULL AND
                source_event_id > 0 AND
                source_period IS NOT NULL AND
                source_priority IS NOT NULL AND
                source_timestamp_epoch IS NOT NULL AND
                source_date IS NOT NULL AND
                candidate_classification IN (
                    'oanda_populated_initial',
                    'oanda_matched_blank'
                )
            ) OR (
                consensus_source = 'MYFXBOOK' AND
                source_report_id IS NULL AND
                source_event_id <> 0 AND
                source_period IS NULL AND
                source_priority IS NULL AND
                source_timestamp_epoch IS NULL AND
                source_date IS NULL AND
                forecast_parse_status = 'parsed' AND
                previous_parse_status = 'missing' AND
                actual_parse_status = 'missing' AND
                source_artifact_sha256 IS NOT NULL AND
                candidate_classification IN (
                    'myfxbook_oanda_blank_fill',
                    'myfxbook_jolts_gap_fill'
                )
            )
        );

CREATE INDEX economic_event_consensus_event_idx
    ON economic_event_consensus (economic_event_id);

-- A blank OANDA evidence row may coexist with a populated Myfxbook fill, but
-- two populated providers can never silently compete for the selected value.
CREATE UNIQUE INDEX economic_event_consensus_one_populated_per_event_uq
    ON economic_event_consensus (economic_event_id)
    WHERE forecast_parse_status = 'parsed';

CREATE VIEW economic_event_selected_consensus AS
SELECT
    c.economic_event_consensus_id,
    c.economic_event_id,
    c.forecast_canonical_value_low AS consensus_value_low,
    c.forecast_canonical_value_high AS consensus_value_high,
    c.forecast_value_kind AS consensus_value_kind,
    c.forecast_unit AS consensus_unit,
    c.forecast_scale AS consensus_scale,
    c.forecast_qualifier AS consensus_qualifier,
    c.consensus_source,
    c.source_report_id,
    c.source_event_id,
    c.source_observation_id,
    c.source_release_date,
    c.source_artifact_path,
    c.source_artifact_sha256,
    c.candidate_classification,
    c.match_rule,
    c.semantic_contract,
    c.provider_provenance,
    c.imported_at
FROM economic_event_consensus c
WHERE c.forecast_parse_status = 'parsed';

COMMENT ON COLUMN economic_event_consensus.economic_event_consensus_id IS
    'Immutable provider-observation row identity; economic_event_id remains the authoritative government release identity.';
COMMENT ON COLUMN economic_event_consensus.source_observation_id IS
    'Deterministic provider-scoped observation identity, distinct from a provider series/event ID.';
COMMENT ON COLUMN economic_event_consensus.source_release_date IS
    'Provider observation release date retained without inventing a provider time or timezone.';
COMMENT ON COLUMN economic_event_consensus.source_artifact_sha256 IS
    'Optional lowercase SHA-256 of the retained source artifact; mandatory for Myfxbook observations.';
COMMENT ON COLUMN economic_event_consensus.candidate_classification IS
    'Audited initial-population or later source-backed fill classification.';
COMMENT ON COLUMN economic_event_consensus.provider_provenance IS
    'Exact provider-specific audit metadata not used as canonical event identity.';
COMMENT ON VIEW economic_event_selected_consensus IS
    'The sole populated consensus observation for each authoritative economic event; provider identity and provenance remain explicit.';

GRANT SELECT, INSERT ON economic_event_consensus TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE economic_event_consensus_economic_event_consensus_id_seq TO pqxx;
GRANT SELECT ON economic_event_selected_consensus TO pqxx;
