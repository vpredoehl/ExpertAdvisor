SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    scheduler_priority AS priority,
    worker_pid AS pid
FROM experiment
WHERE status = 'running'
ORDER BY
    CASE scheduler_priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
    experiment_id;

SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    scheduler_priority AS priority,
    worker_pid AS pid
FROM experiment
WHERE status = 'pending'
ORDER BY 
    CASE scheduler_priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
    experiment_id
LIMIT 7;

SELECT
    experiment_id AS exp,
    symbol,
    prediction_horizon AS h,
    current_epoch AS epoch,
    target_epochs AS target,
    status,
    phase,
    scheduler_priority AS priority,
    worker_pid AS pid
FROM experiment
WHERE status = 'paused'
ORDER BY 
    CASE scheduler_priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
    experiment_id
LIMIT 7;

SELECT
    ce.checkpoint_eval_id AS eval,
    ce.experiment_id AS exp,
    ce.symbol,
    ce.prediction_horizon AS h,
    ce.checkpoint_epoch AS epoch,
    ce.checkpoint_model_id AS model,
    ce.status,
    ce.phase,
    e.scheduler_priority AS priority,
    ce.worker_pid AS pid
FROM experiment_checkpoint_eval ce
JOIN experiment e
  ON e.experiment_id = ce.experiment_id
WHERE ce.status IN ('running','pending')
  AND ce.phase = 'infer'
ORDER BY
    CASE e.scheduler_priority WHEN 'high' THEN 1 WHEN 'normal' THEN 2 WHEN 'low' THEN 3 ELSE 4 END,
    CASE ce.status WHEN 'running' THEN 1 ELSE 2 END,
    ce.checkpoint_eval_id;

SELECT
    count(*) FILTER (
        WHERE status = 'running' AND phase = 'train'
    ) AS train_running,
    count(*) FILTER (
        WHERE status = 'pending' AND phase = 'train'
    ) AS train_pending,
    count(*) FILTER ( where status = 'paused' AND phase = 'train' ) AS train_paused
FROM experiment;

SELECT
    (SELECT count(*) FROM experiment
     WHERE status = 'running' AND phase = 'infer') AS final_infer_running,
    (SELECT count(*) FROM experiment
     WHERE status = 'pending' AND phase = 'infer') AS final_infer_pending,
    (SELECT count(*) FROM experiment
     WHERE status = 'paused' AND phase = 'infer') AS final_infer_paused,
    (SELECT count(*) FROM experiment_checkpoint_eval
     WHERE status = 'running' AND phase = 'infer') AS cp_infer_running,
    (SELECT count(*) FROM experiment_checkpoint_eval
     WHERE status = 'pending' AND phase = 'infer') AS cp_infer_pending,
    (SELECT count(*) FROM experiment
     WHERE status = 'paused' AND phase = 'infer') AS cp_infer_paused;

SELECT 
	count(*) FILTER ( where phase = 'analyze' and status = 'running' ) as analyze_running,
	count(*) FILTER ( where phase = 'analyze' and status = 'pending' ) as analyze_pending,
	count(*) FILTER ( where phase = 'analyze' and status = 'paused' ) as analyze_paused
FROM experiment;
