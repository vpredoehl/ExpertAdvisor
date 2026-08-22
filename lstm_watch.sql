SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    worker_pid AS pid
FROM experiment
WHERE status = 'running'
ORDER BY experiment_id;

SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    worker_pid AS pid
FROM experiment
WHERE status = 'pending'
ORDER BY experiment_id
LIMIT 7;

SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    worker_pid AS pid
FROM experiment
WHERE status = 'paused'
ORDER BY experiment_id
LIMIT 7;

SELECT
    checkpoint_eval_id AS eval,
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    checkpoint_epoch AS epoch,
    checkpoint_model_id AS model,
    status,
    phase,
    worker_pid AS pid
FROM experiment_checkpoint_eval
WHERE status IN ('running','pending')
  AND phase = 'infer'
ORDER BY
    CASE status WHEN 'running' THEN 1 ELSE 2 END,
    checkpoint_eval_id;

SELECT
    count(*) FILTER (
        WHERE status = 'running' AND phase = 'train'
    ) AS train_running,
    count(*) FILTER (
        WHERE status = 'pending' AND phase = 'train'
    ) AS train_pending
FROM experiment;

SELECT
    count(*) FILTER (
        WHERE status = 'running' AND phase = 'infer'
    ) AS cp_infer_running,
    count(*) FILTER (
        WHERE status = 'pending' AND phase = 'infer'
    ) AS cp_infer_pending
FROM experiment_checkpoint_eval;
