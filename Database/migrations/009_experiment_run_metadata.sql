ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS git_commit text,
    ADD COLUMN IF NOT EXISTS git_branch text,
    ADD COLUMN IF NOT EXISTS git_dirty boolean,
    ADD COLUMN IF NOT EXISTS build_config text,
    ADD COLUMN IF NOT EXISTS compiler_version text,
    ADD COLUMN IF NOT EXISTS schema_version text,
    ADD COLUMN IF NOT EXISTS scheduler_version text,
    ADD COLUMN IF NOT EXISTS binary_name text,
    ADD COLUMN IF NOT EXISTS invocation_mode text,
    ADD COLUMN IF NOT EXISTS run_metadata_captured_at timestamptz;

CREATE INDEX IF NOT EXISTS experiment_git_commit_idx
    ON experiment (git_commit);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
