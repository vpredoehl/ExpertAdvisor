ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS checkpoint_policy_enabled boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_min_leader_score double precision,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_min_infer_accuracy double precision,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_top_n integer,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_scope text NOT NULL DEFAULT 'symbol_horizon',
    ADD COLUMN IF NOT EXISTS checkpoint_policy_stop_mode text NOT NULL DEFAULT 'next_checkpoint',
    ADD COLUMN IF NOT EXISTS checkpoint_policy_grace_evals integer NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_last_decision text,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_last_decision_at timestamptz,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_last_checkpoint_eval_id bigint,
    ADD COLUMN IF NOT EXISTS checkpoint_policy_last_reason text;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM pg_constraint
        WHERE conname = 'experiment_checkpoint_policy_last_eval_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_checkpoint_policy_last_eval_fkey
            FOREIGN KEY (checkpoint_policy_last_checkpoint_eval_id)
            REFERENCES experiment_checkpoint_eval(checkpoint_eval_id);
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS experiment_checkpoint_decision (
    checkpoint_decision_id BIGSERIAL PRIMARY KEY,
    checkpoint_eval_id BIGINT NOT NULL REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    parent_experiment_id BIGINT NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch INTEGER NOT NULL,
    checkpoint_model_id BIGINT NOT NULL REFERENCES model(model_id),
    decision TEXT NOT NULL,
    reason TEXT NOT NULL,
    leader_score DOUBLE PRECISION,
    infer_accuracy DOUBLE PRECISION,
    rank_value INTEGER,
    rank_scope TEXT,
    requested_stop_epoch INTEGER,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_checkpoint_decision_eval_uidx
    ON experiment_checkpoint_decision(checkpoint_eval_id);

CREATE INDEX IF NOT EXISTS experiment_checkpoint_decision_parent_idx
    ON experiment_checkpoint_decision(parent_experiment_id, checkpoint_epoch);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_decision TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_checkpoint_decision_checkpoint_decision_id_seq TO pqxx;
