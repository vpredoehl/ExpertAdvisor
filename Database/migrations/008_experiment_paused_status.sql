ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_status_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_status_check
    CHECK (status IN ('pending', 'paused', 'running', 'completed', 'failed', 'cancelled'));
