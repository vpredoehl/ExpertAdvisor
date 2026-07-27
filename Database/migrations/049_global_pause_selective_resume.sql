-- Durable, audited selective release of workers suspended by global pause.

ALTER TABLE experiment_admin_request
    DROP CONSTRAINT IF EXISTS experiment_admin_request_action_check;

ALTER TABLE experiment_admin_request
    ADD CONSTRAINT experiment_admin_request_action_check
    CHECK (action IN (
        'pause_all', 'resume_all', 'cancel_all', 'resume_experiment'
    )),
    ADD COLUMN IF NOT EXISTS target_experiment_id bigint
        REFERENCES experiment(experiment_id);

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_admin_request'::regclass
          AND conname = 'experiment_admin_request_target_shape_check'
    ) THEN
        ALTER TABLE experiment_admin_request
            ADD CONSTRAINT experiment_admin_request_target_shape_check
            CHECK (
                (action = 'resume_experiment'
                 AND target_experiment_id IS NOT NULL
                 AND cancellation_mode IS NULL
                 AND NOT infer_before_cancel)
                OR
                (action <> 'resume_experiment'
                 AND target_experiment_id IS NULL)
            );
    END IF;
END $$;

ALTER TABLE experiment_global_control
    ADD COLUMN IF NOT EXISTS current_pause_request_id bigint
        REFERENCES experiment_admin_request(request_id)
        DEFERRABLE INITIALLY IMMEDIATE;

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS worker_global_pause_request_id bigint
        REFERENCES experiment_admin_request(request_id);

ALTER TABLE experiment_checkpoint_eval
    ADD COLUMN IF NOT EXISTS worker_global_pause_request_id bigint
        REFERENCES experiment_admin_request(request_id);

ALTER TABLE experiment_admin_worker_outcome
    ADD COLUMN IF NOT EXISTS worker_executable text,
    ADD COLUMN IF NOT EXISTS worker_command_line text,
    ADD COLUMN IF NOT EXISTS source_pause_request_id bigint
        REFERENCES experiment_admin_request(request_id);

-- Preserve the currently effective pause generation when upgrading an already
-- paused installation. The current persisted worker identity becomes the
-- frozen executable/command evidence for historical migration-046 outcomes.
WITH applicable_pause AS (
    SELECT r.request_id
    FROM experiment_global_control c
    JOIN LATERAL (
        SELECT request_id
        FROM experiment_admin_request
        WHERE action = 'pause_all'
          AND status IN ('completed', 'partial')
        ORDER BY request_id DESC
        LIMIT 1
    ) r ON c.desired_state = 'paused'
    WHERE c.singleton
)
UPDATE experiment_global_control c
SET current_pause_request_id = p.request_id
FROM applicable_pause p
WHERE c.singleton
  AND c.desired_state = 'paused'
  AND c.current_pause_request_id IS NULL;

UPDATE experiment e
SET worker_global_pause_request_id = c.current_pause_request_id
FROM experiment_global_control c
WHERE c.singleton
  AND c.desired_state = 'paused'
  AND c.current_pause_request_id IS NOT NULL
  AND e.status = 'running'
  AND e.worker_control_state = 'paused'
  AND e.worker_global_pause_request_id IS NULL;

UPDATE experiment_checkpoint_eval ce
SET worker_global_pause_request_id = c.current_pause_request_id
FROM experiment_global_control c
WHERE c.singleton
  AND c.desired_state = 'paused'
  AND c.current_pause_request_id IS NOT NULL
  AND ce.status = 'running'
  AND ce.phase = 'infer'
  AND ce.worker_control_state = 'paused'
  AND ce.worker_global_pause_request_id IS NULL;

UPDATE experiment_admin_worker_outcome o
SET worker_executable = COALESCE(o.worker_executable, e.worker_executable),
    worker_command_line = COALESCE(
        o.worker_command_line, e.worker_command_line
    )
FROM experiment e
WHERE o.worker_kind = 'experiment'
  AND o.experiment_id = e.experiment_id
  AND (o.worker_executable IS NULL OR o.worker_command_line IS NULL);

UPDATE experiment_admin_worker_outcome o
SET worker_executable = COALESCE(o.worker_executable, ce.worker_executable),
    worker_command_line = COALESCE(
        o.worker_command_line, ce.worker_command_line
    )
FROM experiment_checkpoint_eval ce
WHERE o.worker_kind = 'checkpoint_infer'
  AND o.checkpoint_eval_id = ce.checkpoint_eval_id
  AND (o.worker_executable IS NULL OR o.worker_command_line IS NULL);

CREATE INDEX IF NOT EXISTS experiment_worker_global_pause_idx
    ON experiment(worker_global_pause_request_id, experiment_id)
    WHERE worker_global_pause_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS checkpoint_eval_worker_global_pause_idx
    ON experiment_checkpoint_eval(
        worker_global_pause_request_id, checkpoint_eval_id
    )
    WHERE worker_global_pause_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS admin_worker_outcome_source_pause_idx
    ON experiment_admin_worker_outcome(
        source_pause_request_id, experiment_id
    )
    WHERE source_pause_request_id IS NOT NULL;

REVOKE UPDATE (
    worker_executable, worker_command_line, source_pause_request_id
) ON experiment_admin_worker_outcome FROM pqxx;
