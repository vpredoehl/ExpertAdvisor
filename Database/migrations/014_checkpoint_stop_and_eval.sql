ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS stop_after_checkpoint_epoch integer,
    ADD COLUMN IF NOT EXISTS stopped_at_checkpoint_epoch integer,
    ADD COLUMN IF NOT EXISTS stopped_at_checkpoint_model_id bigint,
    ADD COLUMN IF NOT EXISTS opportunistic_checkpoint_infer boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS checkpoint_infer_min_epoch integer,
    ADD COLUMN IF NOT EXISTS checkpoint_infer_interval integer;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_stopped_at_checkpoint_model_id_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_stopped_at_checkpoint_model_id_fkey
            FOREIGN KEY (stopped_at_checkpoint_model_id)
            REFERENCES model(model_id)
            NOT VALID;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS experiment_checkpoint_eval (
    checkpoint_eval_id BIGSERIAL PRIMARY KEY,
    experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch INTEGER NOT NULL,
    checkpoint_model_id BIGINT NOT NULL REFERENCES model(model_id),
    status TEXT NOT NULL DEFAULT 'pending',
    phase TEXT NOT NULL DEFAULT 'infer',
    worker_pid INTEGER,
    infer_log_path TEXT,
    analysis_log_path TEXT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    started_at TIMESTAMPTZ,
    completed_at TIMESTAMPTZ,
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    error_message TEXT,
    UNIQUE(experiment_id, checkpoint_epoch, checkpoint_model_id)
);

CREATE INDEX IF NOT EXISTS experiment_checkpoint_eval_status_phase_idx
    ON experiment_checkpoint_eval(status, phase);

CREATE INDEX IF NOT EXISTS experiment_checkpoint_eval_experiment_idx
    ON experiment_checkpoint_eval(experiment_id, checkpoint_epoch);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_eval TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_checkpoint_eval_checkpoint_eval_id_seq TO pqxx;
