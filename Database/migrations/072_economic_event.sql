CREATE TABLE economic_event (
    economic_event_id bigserial PRIMARY KEY,

    currency text NOT NULL
        CHECK (currency ~ '^[A-Z]{3}$'),

    event_family text NOT NULL,

    event_timestamp_utc timestamptz NOT NULL,

    source_agency text NOT NULL,
    source_event_id text,
    source_url text NOT NULL,

    reference_period text,

    event_importance smallint NOT NULL
        CHECK (event_importance BETWEEN 1 AND 3),

    historical_time_confidence text NOT NULL
        CHECK (
            historical_time_confidence IN (
                'exact',
                'reconstructed',
                'date_only'
            )
        ),

    source_release_date date,
    source_release_time time without time zone,
    source_timezone text,

    imported_at timestamptz NOT NULL DEFAULT now(),

    CONSTRAINT economic_event_source_timestamp_uq
        UNIQUE (
            source_agency,
            event_family,
            event_timestamp_utc
        )
);

CREATE UNIQUE INDEX economic_event_source_event_id_uq
    ON economic_event (
        source_agency,
        source_event_id
    )
    WHERE source_event_id IS NOT NULL;

CREATE INDEX economic_event_currency_timestamp_idx
    ON economic_event (
        currency,
        event_timestamp_utc
    );

CREATE INDEX economic_event_family_timestamp_idx
    ON economic_event (
        event_family,
        event_timestamp_utc
    );

COMMENT ON TABLE economic_event IS
    'Normalized causal economic release-calendar events from authoritative sources.';

COMMENT ON COLUMN economic_event.event_timestamp_utc IS
    'Authoritative release timestamp normalized to UTC.';

COMMENT ON COLUMN economic_event.source_release_date IS
    'Release date published by the authoritative source in source-local time.';

COMMENT ON COLUMN economic_event.source_release_time IS
    'Release time published by the authoritative source in source-local time.';

COMMENT ON COLUMN economic_event.source_timezone IS
    'IANA timezone used to interpret the source-local release timestamp.';
