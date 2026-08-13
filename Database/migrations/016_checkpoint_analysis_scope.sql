ALTER TABLE experiment_analysis_result
    ADD COLUMN IF NOT EXISTS analysis_scope text NOT NULL DEFAULT 'final',
    ADD COLUMN IF NOT EXISTS checkpoint_eval_id bigint,
    ADD COLUMN IF NOT EXISTS checkpoint_epoch integer,
    ADD COLUMN IF NOT EXISTS parent_experiment_id bigint;

UPDATE experiment_analysis_result
SET analysis_scope = 'final'
WHERE analysis_scope IS NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_analysis_result_checkpoint_eval_id_fkey'
    ) THEN
        ALTER TABLE experiment_analysis_result
            ADD CONSTRAINT experiment_analysis_result_checkpoint_eval_id_fkey
            FOREIGN KEY (checkpoint_eval_id)
            REFERENCES experiment_checkpoint_eval(checkpoint_eval_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_analysis_result_parent_experiment_id_fkey'
    ) THEN
        ALTER TABLE experiment_analysis_result
            ADD CONSTRAINT experiment_analysis_result_parent_experiment_id_fkey
            FOREIGN KEY (parent_experiment_id)
            REFERENCES experiment(experiment_id);
    END IF;
END $$;

DROP INDEX IF EXISTS experiment_analysis_result_experiment_model_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS experiment_analysis_result_final_uidx
    ON experiment_analysis_result (experiment_id, model_id)
    WHERE analysis_scope = 'final';

CREATE UNIQUE INDEX IF NOT EXISTS experiment_analysis_result_checkpoint_eval_uidx
    ON experiment_analysis_result (checkpoint_eval_id)
    WHERE analysis_scope = 'checkpoint' AND checkpoint_eval_id IS NOT NULL;

CREATE INDEX IF NOT EXISTS experiment_analysis_result_scope_idx
    ON experiment_analysis_result (analysis_scope, experiment_id, model_id);

CREATE INDEX IF NOT EXISTS experiment_analysis_result_parent_scope_idx
    ON experiment_analysis_result (parent_experiment_id, analysis_scope);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_analysis_result TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_analysis_result_analysis_id_seq TO pqxx;
