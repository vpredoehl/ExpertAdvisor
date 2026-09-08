SELECT EXISTS (
    SELECT 1
    FROM experiment
    WHERE status = 'running'
) AS have_running
\gset

\if :have_running

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

\endif

SELECT EXISTS (
    SELECT 1
    FROM experiment
    WHERE status = 'pending'
) AS have_pending
\gset

\if :have_pending

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
LIMIT 15;

\endif

SELECT EXISTS (
    SELECT 1
    FROM experiment
    WHERE status = 'paused'
) AS have_paused
\gset

\if :have_paused

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
LIMIT 15;

\endif

SELECT EXISTS (
    SELECT 1
    FROM experiment_checkpoint_eval
    WHERE status IN ('running','pending')
      AND phase = 'infer'
) AS have_checkpoint_infer
\gset

\if :have_checkpoint_infer

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
    CASE e.scheduler_priority
        WHEN 'high' THEN 1
        WHEN 'normal' THEN 2
        WHEN 'low' THEN 3
        ELSE 4
    END,
    CASE ce.status
        WHEN 'running' THEN 1
        ELSE 2
    END,
    ce.checkpoint_eval_id;

\endif

WITH counts AS (
    SELECT *
    FROM (
        VALUES
            ('train_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'train')),
            ('train_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'train')),
            ('train_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'train')),

            ('final_infer_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'infer')),
            ('final_infer_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'infer')),
            ('final_infer_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'infer')),

            ('cp_infer_running',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'running' AND phase = 'infer')),
            ('cp_infer_pending',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'pending' AND phase = 'infer')),
            ('cp_infer_paused',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'paused' AND phase = 'infer')),

            ('analyze_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'analyze')),
            ('analyze_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'analyze')),
            ('analyze_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'analyze'))
    ) AS v(category, count)
)
SELECT EXISTS (
    SELECT 1 FROM counts WHERE count > 0
) AS have_counts
\gset

\if :have_counts

WITH counts AS (
    SELECT *
    FROM (
        VALUES
            ('train_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'train')),
            ('train_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'train')),
            ('train_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'train')),

            ('final_infer_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'infer')),
            ('final_infer_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'infer')),
            ('final_infer_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'infer')),

            ('cp_infer_running',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'running' AND phase = 'infer')),
            ('cp_infer_pending',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'pending' AND phase = 'infer')),
            ('cp_infer_paused',
                (SELECT count(*) FROM experiment_checkpoint_eval
                 WHERE status = 'paused' AND phase = 'infer')),

            ('analyze_running',
                (SELECT count(*) FROM experiment
                 WHERE status = 'running' AND phase = 'analyze')),
            ('analyze_pending',
                (SELECT count(*) FROM experiment
                 WHERE status = 'pending' AND phase = 'analyze')),
            ('analyze_paused',
                (SELECT count(*) FROM experiment
                 WHERE status = 'paused' AND phase = 'analyze'))
    ) AS v(category, count)
)
SELECT category, count
FROM counts
WHERE count > 0
ORDER BY
    CASE
        WHEN category LIKE 'train_%' THEN 1
        WHEN category LIKE 'final_infer_%' THEN 2
        WHEN category LIKE 'cp_infer_%' THEN 3
        WHEN category LIKE 'analyze_%' THEN 4
        ELSE 5
    END,
    CASE
        WHEN category LIKE '%_running' THEN 1
        WHEN category LIKE '%_pending' THEN 2
        WHEN category LIKE '%_paused' THEN 3
        ELSE 4
    END;

\endif
