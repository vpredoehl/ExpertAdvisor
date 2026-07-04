CREATE TABLE IF NOT EXISTS experiment (
    experiment_id bigserial PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    core_lr_mult double precision,
    head_lr_mult double precision,
    target_epochs integer NOT NULL,
    checkpoint_interval integer NOT NULL DEFAULT 20,
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    status text NOT NULL DEFAULT 'pending'
        CHECK (status IN ('pending', 'running', 'completed', 'failed', 'cancelled')),
    phase text NOT NULL DEFAULT 'train'
        CHECK (phase IN ('train', 'infer', 'analyze', 'done')),
    last_model_id bigint,
    resume_model_id bigint,
    train_log_path text,
    infer_log_path text,
    analysis_log_path text,
    exit_code integer,
    error_message text,
    duplicate_nonce bigint NOT NULL DEFAULT 0,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS experiment_status_phase_idx
    ON experiment (status, phase);

CREATE INDEX IF NOT EXISTS experiment_symbol_horizon_idx
    ON experiment (symbol, prediction_horizon);

CREATE INDEX IF NOT EXISTS experiment_last_model_id_idx
    ON experiment (last_model_id);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_unique_identity_uidx
    ON experiment (
        symbol,
        prediction_horizon,
        c_next_threshold,
        COALESCE(core_lr_mult, '-infinity'::double precision),
        COALESCE(head_lr_mult, '-infinity'::double precision),
        target_epochs,
        checkpoint_interval,
        train_start,
        train_end,
        COALESCE(infer_start, '-infinity'::timestamptz),
        COALESCE(infer_end, '-infinity'::timestamptz),
        COALESCE(resume_model_id, -1),
        duplicate_nonce
    )
    WHERE status <> 'cancelled';

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_experiment_id_seq TO pqxx;
