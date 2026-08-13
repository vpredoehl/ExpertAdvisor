ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS current_epoch integer,
    ADD COLUMN IF NOT EXISTS worker_pid integer,
    ADD COLUMN IF NOT EXISTS current_operation text;

CREATE INDEX IF NOT EXISTS experiment_worker_pid_idx
    ON experiment (worker_pid);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
