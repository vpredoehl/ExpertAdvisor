-- Durable resume origin and persistent-priority-first scheduler admission.

BEGIN;

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS scheduler_resume_origin text NOT NULL
        DEFAULT 'none';

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_scheduler_resume_origin_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_scheduler_resume_origin_check
        CHECK (scheduler_resume_origin IN ('none', 'operator', 'preemption'));

-- Migration 086 had only a boolean urgency marker. Existing explicit resume
-- requests were all operator/admin initiated, so their origin is deterministic.
UPDATE experiment
SET scheduler_resume_origin = CASE
        WHEN resume_requested AND scheduler_resume_origin = 'none'
            THEN 'operator'
        WHEN resume_requested THEN scheduler_resume_origin
        ELSE 'none'
    END;

-- The experiment table has deferred lifecycle constraint triggers. Drain the
-- events produced by legacy normalization before the next ALTER TABLE while
-- keeping the entire migration atomic.
SET CONSTRAINTS ALL IMMEDIATE;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_scheduler_resume_state_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_scheduler_resume_state_check
        CHECK (
            (resume_requested AND
             scheduler_resume_origin IN ('operator', 'preemption'))
            OR
            (NOT resume_requested AND scheduler_resume_origin = 'none')
        );

DROP INDEX IF EXISTS experiment_scheduler_pending_priority_idx;
CREATE INDEX experiment_scheduler_pending_priority_idx
    ON experiment (
        phase,
        (CASE scheduler_priority
            WHEN 'high' THEN 0
            WHEN 'normal' THEN 1
            ELSE 2
         END),
        (CASE scheduler_resume_origin
            WHEN 'operator' THEN 0
            WHEN 'preemption' THEN 1
            ELSE 2
         END),
        updated_at,
        experiment_id
    )
    WHERE status = 'pending';

COMMENT ON COLUMN experiment.scheduler_resume_origin IS
    'Pending admission origin: none for ordinary work, operator for explicit '
    'operator/admin resume intent, preemption for scheduler-stopped work. '
    'Persistent scheduler_priority always orders before this origin.';

COMMIT;
