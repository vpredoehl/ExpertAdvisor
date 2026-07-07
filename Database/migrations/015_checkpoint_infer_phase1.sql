ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS checkpoint_infer_enabled boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS checkpoint_infer_min_epoch integer,
    ADD COLUMN IF NOT EXISTS checkpoint_infer_interval integer;

UPDATE experiment
SET checkpoint_infer_enabled = true
WHERE EXISTS (
    SELECT 1
    FROM information_schema.columns
    WHERE table_schema = 'public'
      AND table_name = 'experiment'
      AND column_name = 'opportunistic_checkpoint_infer'
)
AND opportunistic_checkpoint_infer = true
AND checkpoint_infer_enabled = false;

ALTER TABLE experiment_checkpoint_eval
    ADD COLUMN IF NOT EXISTS parent_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS symbol text,
    ADD COLUMN IF NOT EXISTS prediction_horizon integer,
    ADD COLUMN IF NOT EXISTS infer_started_at timestamptz,
    ADD COLUMN IF NOT EXISTS infer_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS analyze_started_at timestamptz,
    ADD COLUMN IF NOT EXISTS analyze_completed_at timestamptz,
    ADD COLUMN IF NOT EXISTS analysis_id bigint;

UPDATE experiment_checkpoint_eval ce
SET parent_experiment_id = ce.experiment_id
WHERE ce.parent_experiment_id IS NULL;

UPDATE experiment_checkpoint_eval ce
SET symbol = e.symbol,
    prediction_horizon = e.prediction_horizon
FROM experiment e
WHERE e.experiment_id = ce.experiment_id
  AND (ce.symbol IS NULL OR ce.prediction_horizon IS NULL);

ALTER TABLE experiment_checkpoint_eval
    ALTER COLUMN parent_experiment_id SET NOT NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_eval_parent_experiment_id_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_eval
            ADD CONSTRAINT experiment_checkpoint_eval_parent_experiment_id_fkey
            FOREIGN KEY (parent_experiment_id)
            REFERENCES experiment(experiment_id);
    END IF;
END $$;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_checkpoint_eval_parent_uidx
    ON experiment_checkpoint_eval(parent_experiment_id, checkpoint_model_id, checkpoint_epoch);

CREATE INDEX IF NOT EXISTS experiment_checkpoint_eval_parent_status_phase_idx
    ON experiment_checkpoint_eval(parent_experiment_id, status, phase);

CREATE INDEX IF NOT EXISTS experiment_checkpoint_eval_analysis_id_idx
    ON experiment_checkpoint_eval(analysis_id);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_eval TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_checkpoint_eval_checkpoint_eval_id_seq TO pqxx;
