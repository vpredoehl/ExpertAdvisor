-- Distinct authoritative occurrences may share a conservative causal boundary.
-- Source identity remains unique; timestamp coincidence is audited by the
-- authoritative ingestion workflow and is no longer a database identity key.
ALTER TABLE economic_event
    DROP CONSTRAINT economic_event_source_timestamp_uq;

ALTER TABLE economic_event
    ADD CONSTRAINT economic_event_fomc_20140917_identity_ck
    CHECK (
        source_agency <> 'FEDERAL_RESERVE'
        OR event_family <> 'FOMC_STATEMENT'
        OR event_timestamp_utc <> TIMESTAMPTZ '2014-09-18 04:00:00+00'
        OR source_event_id IN (
            'federal_reserve:monetary20140917a',
            'federal_reserve:monetary20140917c'
        )
    );

CREATE UNIQUE INDEX economic_event_source_timestamp_guard_uq
    ON economic_event (
        source_agency,
        event_family,
        event_timestamp_utc,
        (
            CASE
                WHEN source_agency = 'FEDERAL_RESERVE'
                    AND event_family = 'FOMC_STATEMENT'
                    AND event_timestamp_utc =
                        TIMESTAMPTZ '2014-09-18 04:00:00+00'
                THEN source_event_id
                ELSE ''
            END
        )
    );

COMMENT ON COLUMN economic_event.event_timestamp_utc IS
    'Authoritative release timestamp or conservative causal availability boundary normalized to UTC; distinct source identities may share a boundary.';
