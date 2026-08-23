-- Durable operator intent for an explicit scheduler-managed FINAL inference
-- rerun. Historical experiments remain false and no inference result rows are
-- modified or backfilled by this migration.
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS
        operator_forced_final_inference_rerun_requested boolean NOT NULL
        DEFAULT false;

COMMENT ON COLUMN
    experiment.operator_forced_final_inference_rerun_requested IS
    'True only while an operator-requested FINAL inference rerun has not yet produced attempt-relative completed FINAL inference evidence.';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
