-- Record the point-in-time evidence required for a historically causal
-- Myfxbook Weekly Claims forecast.  Existing 081/082 observations remain
-- valid and unchanged; their historical-availability contract is not
-- retroactively inferred.

ALTER TABLE economic_event_consensus
    DROP CONSTRAINT economic_event_consensus_candidate_classification_ck,
    DROP CONSTRAINT economic_event_consensus_provider_contract_ck,
    ADD COLUMN provider_observed_at timestamptz,
    ADD COLUMN forecast_available_at timestamptz,
    ADD COLUMN source_retrieved_at timestamptz,
    ADD COLUMN forecast_availability_proof text,
    ADD CONSTRAINT economic_event_consensus_forecast_availability_shape_ck
        CHECK (
            (provider_observed_at IS NULL AND
             forecast_available_at IS NULL AND
             source_retrieved_at IS NULL AND
             forecast_availability_proof IS NULL)
            OR
            (provider_observed_at IS NOT NULL AND
             forecast_available_at IS NOT NULL AND
             source_retrieved_at IS NOT NULL AND
             forecast_availability_proof <> '' AND
             provider_observed_at <= forecast_available_at AND
             forecast_available_at <= source_retrieved_at)
        ),
    ADD CONSTRAINT economic_event_consensus_candidate_classification_ck
        CHECK (
            candidate_classification IN (
                'oanda_populated_initial',
                'oanda_matched_blank',
                'myfxbook_oanda_blank_fill',
                'myfxbook_jolts_gap_fill',
                'myfxbook_weekly_claims_pre_release_snapshot'
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
                ) AND
                provider_observed_at IS NULL AND
                forecast_available_at IS NULL AND
                source_retrieved_at IS NULL AND
                forecast_availability_proof IS NULL
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
                ) AND
                provider_observed_at IS NULL AND
                forecast_available_at IS NULL AND
                source_retrieved_at IS NULL AND
                forecast_availability_proof IS NULL
            ) OR (
                consensus_source = 'MYFXBOOK' AND
                source_report_id IS NULL AND
                source_event_id > 0 AND
                source_period IS NULL AND
                source_priority IS NULL AND
                source_timestamp_epoch IS NULL AND
                source_date IS NULL AND
                forecast_parse_status = 'parsed' AND
                previous_parse_status = 'missing' AND
                actual_parse_status = 'missing' AND
                source_artifact_sha256 IS NOT NULL AND
                candidate_classification =
                    'myfxbook_weekly_claims_pre_release_snapshot' AND
                semantic_contract =
                    'myfxbook_weekly_claims_pre_release_snapshot_v1' AND
                provider_observed_at IS NOT NULL AND
                forecast_available_at IS NOT NULL AND
                source_retrieved_at IS NOT NULL AND
                forecast_availability_proof =
                    'internet_archive_pre_release_capture'
            )
        );

CREATE OR REPLACE FUNCTION enforce_historical_consensus_availability()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    canonical_family text;
    canonical_release timestamptz;
BEGIN
    IF NEW.candidate_classification <>
        'myfxbook_weekly_claims_pre_release_snapshot' THEN
        RETURN NEW;
    END IF;

    SELECT event_family, event_timestamp_utc
      INTO canonical_family, canonical_release
      FROM economic_event
     WHERE economic_event_id = NEW.economic_event_id;

    IF canonical_family IS DISTINCT FROM 'WEEKLY_CLAIMS' THEN
        RAISE EXCEPTION
            'historical Weekly Claims consensus mapped to non-Weekly-Claims event';
    END IF;
    IF NEW.forecast_available_at >= canonical_release THEN
        RAISE EXCEPTION
            'historical consensus was not archived before canonical release';
    END IF;
    RETURN NEW;
END;
$$;

CREATE TRIGGER economic_event_consensus_historical_availability_trigger
BEFORE INSERT ON economic_event_consensus
FOR EACH ROW EXECUTE FUNCTION enforce_historical_consensus_availability();

CREATE OR REPLACE VIEW economic_event_selected_consensus AS
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
    c.imported_at,
    c.provider_observed_at,
    c.forecast_available_at,
    c.source_retrieved_at,
    c.forecast_availability_proof
FROM economic_event_consensus c
WHERE c.forecast_parse_status = 'parsed';

COMMENT ON COLUMN economic_event_consensus.provider_observed_at IS
    'Provider page server timestamp embedded in the retained historical artifact.';
COMMENT ON COLUMN economic_event_consensus.forecast_available_at IS
    'Latest timestamp by which the retained artifact proves the forecast was public; for Phase 8 this is the Internet Archive capture timestamp.';
COMMENT ON COLUMN economic_event_consensus.source_retrieved_at IS
    'Current acquisition timestamp, retained separately and never used as historical availability evidence.';
COMMENT ON COLUMN economic_event_consensus.forecast_availability_proof IS
    'Versioned proof basis for point-in-time forecast availability.';
COMMENT ON FUNCTION enforce_historical_consensus_availability() IS
    'Fails closed unless a Phase-8 Weekly Claims snapshot was archived strictly before the canonical release.';

GRANT EXECUTE ON FUNCTION enforce_historical_consensus_availability() TO pqxx;
