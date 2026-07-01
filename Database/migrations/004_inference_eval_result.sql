CREATE TABLE IF NOT EXISTS inference_eval_result (
    id bigserial PRIMARY KEY,
    model_id bigint NOT NULL,
    symbol text NOT NULL,
    prediction_horizon bigint NOT NULL,
    threshold_logret double precision NOT NULL,
    window_size bigint NOT NULL,
    label_rule_id integer NOT NULL,
    target_type integer NOT NULL,
    from_date text NOT NULL,
    to_date text NOT NULL,
    completed_epochs bigint,
    accuracy double precision,
    accept_model boolean,
    reject_reason text,
    pred_down double precision,
    pred_neutral double precision,
    pred_up double precision,
    status text NOT NULL CHECK (status IN ('completed', 'failed')),
    completed_at timestamptz NOT NULL DEFAULT now()
);

CREATE UNIQUE INDEX IF NOT EXISTS inference_eval_result_completed_uidx
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
    WHERE status = 'completed';

GRANT SELECT, INSERT, UPDATE, DELETE ON inference_eval_result TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE inference_eval_result_id_seq TO pqxx;
