ALTER TABLE inference_eval_result
    ADD COLUMN IF NOT EXISTS inference_scope text NOT NULL DEFAULT 'final',
    ADD COLUMN IF NOT EXISTS checkpoint_eval_id bigint,
    ADD COLUMN IF NOT EXISTS parent_experiment_id bigint,
    ADD COLUMN IF NOT EXISTS checkpoint_epoch integer;

UPDATE inference_eval_result
SET inference_scope = 'final'
WHERE inference_scope IS NULL;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'inference_eval_result_scope_check'
    ) THEN
        ALTER TABLE inference_eval_result
            ADD CONSTRAINT inference_eval_result_scope_check
            CHECK (inference_scope IN ('final', 'checkpoint'));
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'inference_eval_result_scope_identity_check'
    ) THEN
        ALTER TABLE inference_eval_result
            ADD CONSTRAINT inference_eval_result_scope_identity_check
            CHECK (
                (inference_scope = 'final'
                 AND checkpoint_eval_id IS NULL
                 AND parent_experiment_id IS NULL
                 AND checkpoint_epoch IS NULL)
                OR
                (inference_scope = 'checkpoint'
                 AND checkpoint_eval_id IS NOT NULL
                 AND parent_experiment_id IS NOT NULL
                 AND checkpoint_epoch IS NOT NULL)
            );
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'inference_eval_result_checkpoint_eval_id_fkey'
    ) THEN
        ALTER TABLE inference_eval_result
            ADD CONSTRAINT inference_eval_result_checkpoint_eval_id_fkey
            FOREIGN KEY (checkpoint_eval_id)
            REFERENCES experiment_checkpoint_eval(checkpoint_eval_id);
    END IF;

    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'inference_eval_result_parent_experiment_id_fkey'
    ) THEN
        ALTER TABLE inference_eval_result
            ADD CONSTRAINT inference_eval_result_parent_experiment_id_fkey
            FOREIGN KEY (parent_experiment_id)
            REFERENCES experiment(experiment_id);
    END IF;
END $$;

DROP INDEX IF EXISTS inference_eval_result_completed_uidx;

CREATE UNIQUE INDEX IF NOT EXISTS inference_eval_result_final_completed_uidx
    ON inference_eval_result (
        model_id,
        symbol,
        prediction_horizon,
        threshold_logret,
        window_size,
        label_rule_id,
        target_type,
        from_date,
        to_date
    )
    WHERE status = 'completed' AND inference_scope = 'final';

CREATE UNIQUE INDEX IF NOT EXISTS inference_eval_result_checkpoint_eval_uidx
    ON inference_eval_result (checkpoint_eval_id)
    WHERE inference_scope = 'checkpoint';

CREATE INDEX IF NOT EXISTS inference_eval_result_scope_model_status_idx
    ON inference_eval_result (inference_scope, model_id, status);

CREATE INDEX IF NOT EXISTS inference_eval_result_parent_scope_idx
    ON inference_eval_result (parent_experiment_id, inference_scope);

GRANT SELECT, INSERT, UPDATE, DELETE ON inference_eval_result TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE inference_eval_result_id_seq TO pqxx;
