ALTER TABLE model
    ADD COLUMN IF NOT EXISTS experiment_id BIGINT NULL;

CREATE INDEX IF NOT EXISTS model_experiment_id_idx
    ON model (experiment_id);

UPDATE model m
SET experiment_id = parsed.experiment_id
FROM (
    SELECT
        model_id,
        substring(COALESCE(name, '') FROM 'experiment([0-9]+)')::bigint AS experiment_id
    FROM model
    WHERE experiment_id IS NULL
      AND substring(COALESCE(name, '') FROM 'experiment([0-9]+)') IS NOT NULL
) parsed
JOIN experiment e ON e.experiment_id = parsed.experiment_id
WHERE m.model_id = parsed.model_id
  AND m.experiment_id IS NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'model_experiment_id_fkey'
    ) THEN
        ALTER TABLE model
            ADD CONSTRAINT model_experiment_id_fkey
            FOREIGN KEY (experiment_id)
            REFERENCES experiment(experiment_id)
            NOT VALID;
    END IF;
END $$;

GRANT SELECT, INSERT, UPDATE, DELETE ON model TO pqxx;
