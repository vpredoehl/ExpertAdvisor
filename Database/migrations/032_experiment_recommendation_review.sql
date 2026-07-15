-- Phase 4A Step 5: explicit advisory review with immutable event provenance.
-- Approval records operator review only; it does not create or queue an
-- experiment. Existing approved_experiment_id history remains supported.

ALTER TABLE experiment_recommendation
    ADD COLUMN IF NOT EXISTS approved_at timestamptz;

-- Migration 026 reserved approval for later conversion and therefore required
-- approved_experiment_id. Step 5 approval is deliberately non-converting, so
-- approved_at is the authoritative new-workflow marker while historical rows
-- with approved_experiment_id remain valid.
ALTER TABLE experiment_recommendation
    DROP CONSTRAINT IF EXISTS experiment_recommendation_step3_status_shape_check,
    DROP CONSTRAINT IF EXISTS experiment_recommendation_status_metadata_symmetric_check;

ALTER TABLE experiment_recommendation
    ADD CONSTRAINT experiment_recommendation_step3_status_shape_check
    CHECK (
        (status <> 'rejected' OR
            (rejected_at IS NOT NULL AND rejected_reason IS NOT NULL))
        AND (status <> 'expired' OR expired_at IS NOT NULL)
        AND (status <> 'approved' OR
            (approved_at IS NOT NULL OR approved_experiment_id IS NOT NULL))
    ) NOT VALID,
    ADD CONSTRAINT experiment_recommendation_status_metadata_symmetric_check
    CHECK (
        (
            (status = 'rejected'
             AND rejected_at IS NOT NULL
             AND rejected_reason IS NOT NULL
             AND btrim(rejected_reason) <> '')
            OR
            (status <> 'rejected'
             AND rejected_at IS NULL
             AND rejected_reason IS NULL)
        )
        AND (
            (status = 'expired' AND expired_at IS NOT NULL)
            OR (status <> 'expired' AND expired_at IS NULL)
        )
        AND (
            (status = 'approved'
             AND (approved_at IS NOT NULL OR approved_experiment_id IS NOT NULL))
            OR
            (status <> 'approved'
             AND approved_at IS NULL
             AND approved_experiment_id IS NULL)
        )
    ) NOT VALID;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conname = 'experiment_recommendation_score_id_recommendation_uidx'
          AND conrelid = 'experiment_recommendation_score'::regclass
    ) THEN
        ALTER TABLE experiment_recommendation_score
            ADD CONSTRAINT experiment_recommendation_score_id_recommendation_uidx
            UNIQUE (recommendation_score_id, recommendation_id);
    END IF;
END $$;

CREATE SEQUENCE IF NOT EXISTS experiment_recommendation_review_event_id_seq;

CREATE TABLE IF NOT EXISTS experiment_recommendation_review_event (
    recommendation_review_event_id bigint PRIMARY KEY DEFAULT
        nextval('experiment_recommendation_review_event_id_seq'),
    recommendation_id bigint NOT NULL
        REFERENCES experiment_recommendation(recommendation_id),
    recommendation_score_id bigint,
    action text NOT NULL CHECK (action IN ('approve','reject','expire')),
    previous_status text NOT NULL CHECK (previous_status = 'proposed'),
    resulting_status text NOT NULL
        CHECK (resulting_status IN ('approved','rejected','expired')),
    reason_code text NOT NULL CHECK (
        btrim(reason_code) <> '' AND octet_length(reason_code) <= 64),
    reason_text text,
    reviewer text,
    note text,
    recommendation_semantic_canonical text NOT NULL
        CHECK (btrim(recommendation_semantic_canonical) <> ''),
    recommendation_semantic_hash text NOT NULL
        CHECK (btrim(recommendation_semantic_hash) <> ''),
    recommendation_policy_canonical text NOT NULL
        CHECK (btrim(recommendation_policy_canonical) <> ''),
    recommendation_policy_hash text NOT NULL
        CHECK (btrim(recommendation_policy_hash) <> ''),
    recommendation_scan_id bigint NOT NULL
        REFERENCES experiment_recommendation_scan(recommendation_scan_id),
    source_experiment_id bigint NOT NULL
        REFERENCES experiment(experiment_id),
    created_at timestamptz NOT NULL DEFAULT now(),
    CONSTRAINT experiment_recommendation_review_event_one_per_recommendation
        UNIQUE (recommendation_id),
    CONSTRAINT experiment_recommendation_review_event_score_owner_fkey
        FOREIGN KEY (recommendation_score_id, recommendation_id)
        REFERENCES experiment_recommendation_score(
            recommendation_score_id, recommendation_id),
    CONSTRAINT experiment_recommendation_review_event_result_check CHECK (
        (action = 'approve' AND resulting_status = 'approved')
        OR (action = 'reject' AND resulting_status = 'rejected')
        OR (action = 'expire' AND resulting_status = 'expired')
    ),
    CONSTRAINT experiment_recommendation_review_event_reason_check CHECK (
        (
            action = 'approve'
            AND (reason_text IS NULL OR
                 (btrim(reason_text) <> '' AND octet_length(reason_text) <= 2000))
        )
        OR
        (
            action IN ('reject','expire')
            AND reason_text IS NOT NULL
            AND btrim(reason_text) <> ''
            AND octet_length(reason_text) <= 2000
        )
    ),
    CONSTRAINT experiment_recommendation_review_event_reviewer_check CHECK (
        reviewer IS NULL OR
        (btrim(reviewer) <> '' AND octet_length(reviewer) <= 200)),
    CONSTRAINT experiment_recommendation_review_event_note_check CHECK (
        note IS NULL OR (btrim(note) <> '' AND octet_length(note) <= 2000))
);

ALTER SEQUENCE experiment_recommendation_review_event_id_seq OWNED BY
    experiment_recommendation_review_event.recommendation_review_event_id;

CREATE INDEX IF NOT EXISTS experiment_recommendation_review_history_idx
    ON experiment_recommendation_review_event(
        recommendation_id, recommendation_review_event_id);
CREATE INDEX IF NOT EXISTS experiment_recommendation_review_action_idx
    ON experiment_recommendation_review_event(
        action, recommendation_review_event_id DESC);

GRANT SELECT, INSERT, DELETE ON experiment_recommendation_review_event TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_recommendation_review_event_id_seq
    TO pqxx;
