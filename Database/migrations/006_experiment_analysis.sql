CREATE TABLE IF NOT EXISTS experiment_analysis_result (
    analysis_id bigserial PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    model_id bigint,
    symbol text,
    prediction_horizon integer,
    target_epochs integer,
    completed_epochs integer,
    train_accuracy double precision,
    validation_accuracy double precision,
    infer_accuracy double precision,
    actual_down_count bigint,
    actual_neutral_count bigint,
    actual_up_count bigint,
    pred_down_count bigint,
    pred_neutral_count bigint,
    pred_up_count bigint,
    confusion_down_down bigint,
    confusion_down_neutral bigint,
    confusion_down_up bigint,
    confusion_neutral_down bigint,
    confusion_neutral_neutral bigint,
    confusion_neutral_up bigint,
    confusion_up_down bigint,
    confusion_up_neutral bigint,
    confusion_up_up bigint,
    accept_count bigint,
    accept_rate double precision,
    accept_accuracy double precision,
    reject_count bigint,
    loss_last double precision,
    best_metric_name text,
    best_metric_value double precision,
    leader_score double precision,
    analysis_status text NOT NULL DEFAULT 'pending',
    analysis_notes text,
    source_train_log_path text,
    source_infer_log_path text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS experiment_analysis_result_experiment_model_uidx
    ON experiment_analysis_result (experiment_id, model_id);

CREATE INDEX IF NOT EXISTS experiment_analysis_result_leader_idx
    ON experiment_analysis_result (leader_score DESC NULLS LAST);

CREATE INDEX IF NOT EXISTS experiment_analysis_result_symbol_horizon_idx
    ON experiment_analysis_result (symbol, prediction_horizon);

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_analysis_result TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_analysis_result_analysis_id_seq TO pqxx;
