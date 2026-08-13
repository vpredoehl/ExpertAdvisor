-- Database-authoritative global scheduler execution control and administrative audit.

CREATE TABLE IF NOT EXISTS experiment_global_control (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    desired_state text NOT NULL DEFAULT 'running'
        CHECK (desired_state IN ('running', 'paused')),
    active_request_id bigint,
    revision bigint NOT NULL DEFAULT 1 CHECK (revision > 0),
    updated_at timestamptz NOT NULL DEFAULT now()
);

INSERT INTO experiment_global_control (singleton, desired_state)
VALUES (true, 'running')
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS experiment_admin_request (
    request_id bigserial PRIMARY KEY,
    action text NOT NULL CHECK (action IN ('pause_all', 'resume_all', 'cancel_all')),
    cancellation_mode text
        CHECK (cancellation_mode IS NULL OR
               cancellation_mode IN ('immediate', 'after_next_checkpoint')),
    infer_before_cancel boolean NOT NULL DEFAULT false,
    invocation_identity text NOT NULL,
    requester_identity text,
    application_owner text,
    application_lease_until timestamptz,
    requested_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    status text NOT NULL DEFAULT 'applying'
        CHECK (status IN ('applying', 'pending', 'completed', 'partial', 'failed')),
    previous_global_state text NOT NULL
        CHECK (previous_global_state IN ('running', 'paused')),
    resulting_global_state text NOT NULL
        CHECK (resulting_global_state IN ('running', 'paused')),
    scheduler_running_observed boolean,
    target_count integer NOT NULL DEFAULT 0 CHECK (target_count >= 0),
    successful_count integer NOT NULL DEFAULT 0 CHECK (successful_count >= 0),
    already_satisfied_count integer NOT NULL DEFAULT 0
        CHECK (already_satisfied_count >= 0),
    missing_count integer NOT NULL DEFAULT 0 CHECK (missing_count >= 0),
    rejected_count integer NOT NULL DEFAULT 0 CHECK (rejected_count >= 0),
    failed_count integer NOT NULL DEFAULT 0 CHECK (failed_count >= 0),
    result_summary jsonb NOT NULL DEFAULT '{}'::jsonb,
    error_details text,
    CHECK (
        (action = 'cancel_all' AND cancellation_mode IS NOT NULL)
        OR
        (action <> 'cancel_all' AND cancellation_mode IS NULL
         AND NOT infer_before_cancel)
    )
);

ALTER TABLE experiment_admin_request
    ADD COLUMN IF NOT EXISTS application_owner text,
    ADD COLUMN IF NOT EXISTS application_lease_until timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_global_control'::regclass
          AND conname = 'experiment_global_control_active_request_fkey'
    ) THEN
        ALTER TABLE experiment_global_control
            ADD CONSTRAINT experiment_global_control_active_request_fkey
            FOREIGN KEY (active_request_id)
            REFERENCES experiment_admin_request(request_id)
            DEFERRABLE INITIALLY IMMEDIATE;
    END IF;
END $$;

CREATE TABLE IF NOT EXISTS experiment_admin_worker_outcome (
    request_id bigint NOT NULL
        REFERENCES experiment_admin_request(request_id) ON DELETE RESTRICT,
    worker_identity text NOT NULL,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id) ON DELETE RESTRICT,
    checkpoint_eval_id bigint REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    worker_kind text NOT NULL DEFAULT 'experiment'
        CHECK (worker_kind IN ('experiment', 'checkpoint_infer')),
    phase text NOT NULL,
    lifecycle_status text NOT NULL,
    worker_pid integer,
    worker_process_group_id integer,
    worker_process_start_identity text,
    identity_result text NOT NULL DEFAULT 'not_checked'
        CHECK (identity_result IN (
            'not_checked', 'validated', 'process_missing', 'stale_pid',
            'identity_validation_failed', 'unsafe_process_group',
            'permission_denied', 'inspection_failed'
        )),
    requested_signal text,
    signal_result text NOT NULL DEFAULT 'not_attempted'
        CHECK (signal_result IN (
            'not_attempted', 'signaled', 'already_requested_state',
            'process_missing', 'stale_pid', 'identity_validation_failed',
            'permission_failure', 'signaling_failure', 'escalated'
        )),
    cancellation_checkpoint_epoch integer,
    cancellation_checkpoint_model_id bigint REFERENCES model(model_id),
    inference_action text NOT NULL DEFAULT 'none'
        CHECK (inference_action IN (
            'none', 'already_completed', 'queued', 'running', 'completed',
            'failed', 'no_checkpoint', 'identity_ambiguous'
        )),
    outcome_status text NOT NULL DEFAULT 'planned'
        CHECK (outcome_status IN (
            'planned', 'pending_checkpoint', 'awaiting_inference',
            'completed', 'partial', 'failed'
        )),
    detail text,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (request_id, worker_identity)
);

ALTER TABLE experiment_admin_worker_outcome
    ADD COLUMN IF NOT EXISTS worker_process_start_identity text;

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS worker_process_group_id integer,
    ADD COLUMN IF NOT EXISTS worker_executable text,
    ADD COLUMN IF NOT EXISTS worker_command_line text,
    ADD COLUMN IF NOT EXISTS worker_process_start_identity text,
    ADD COLUMN IF NOT EXISTS worker_control_state text NOT NULL DEFAULT 'running',
    ADD COLUMN IF NOT EXISTS cancellation_request_id bigint,
    ADD COLUMN IF NOT EXISTS cancel_after_checkpoint_epoch integer,
    ADD COLUMN IF NOT EXISTS last_checkpoint_stop_decision_epoch integer,
    ADD COLUMN IF NOT EXISTS cancel_infer_before boolean NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS cancellation_completed_at timestamptz;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment'::regclass
          AND conname = 'experiment_worker_control_state_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_worker_control_state_check
            CHECK (worker_control_state IN ('running', 'paused'));
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment'::regclass
          AND conname = 'experiment_cancellation_request_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_cancellation_request_fkey
            FOREIGN KEY (cancellation_request_id)
            REFERENCES experiment_admin_request(request_id);
    END IF;
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment'::regclass
          AND conname = 'experiment_cancel_checkpoint_shape_check'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_cancel_checkpoint_shape_check
            CHECK (
                cancel_after_checkpoint_epoch IS NULL
                OR (cancel_after_checkpoint_epoch > 0
                    AND cancellation_request_id IS NOT NULL)
            );
    END IF;
END $$;

ALTER TABLE experiment_checkpoint_eval
    ADD COLUMN IF NOT EXISTS cancellation_request_id bigint
        REFERENCES experiment_admin_request(request_id),
    ADD COLUMN IF NOT EXISTS worker_process_group_id integer,
    ADD COLUMN IF NOT EXISTS worker_executable text,
    ADD COLUMN IF NOT EXISTS worker_command_line text,
    ADD COLUMN IF NOT EXISTS worker_process_start_identity text,
    ADD COLUMN IF NOT EXISTS worker_control_state text NOT NULL DEFAULT 'running';

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_checkpoint_eval'::regclass
          AND conname = 'experiment_checkpoint_eval_worker_control_state_check'
    ) THEN
        ALTER TABLE experiment_checkpoint_eval
            ADD CONSTRAINT experiment_checkpoint_eval_worker_control_state_check
            CHECK (worker_control_state IN ('running', 'paused'));
    END IF;
END $$;

CREATE INDEX IF NOT EXISTS experiment_admin_request_status_idx
    ON experiment_admin_request(status, requested_at, request_id);
CREATE INDEX IF NOT EXISTS experiment_admin_worker_outcome_status_idx
    ON experiment_admin_worker_outcome(request_id, outcome_status, experiment_id);
CREATE INDEX IF NOT EXISTS experiment_cancellation_request_idx
    ON experiment(cancellation_request_id)
    WHERE cancellation_request_id IS NOT NULL;
CREATE INDEX IF NOT EXISTS experiment_checkpoint_eval_cancellation_idx
    ON experiment_checkpoint_eval(cancellation_request_id, status, phase)
    WHERE cancellation_request_id IS NOT NULL;

GRANT SELECT, INSERT ON experiment_admin_request TO pqxx;
REVOKE UPDATE ON experiment_admin_request FROM pqxx;
GRANT UPDATE (
    completed_at, status, target_count, successful_count,
    already_satisfied_count, missing_count, rejected_count, failed_count,
    result_summary, error_details, application_owner, application_lease_until
) ON experiment_admin_request TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE experiment_admin_request_request_id_seq TO pqxx;
GRANT SELECT, INSERT ON experiment_admin_worker_outcome TO pqxx;
REVOKE UPDATE ON experiment_admin_worker_outcome FROM pqxx;
GRANT UPDATE (
    worker_pid, worker_process_group_id, worker_process_start_identity,
    identity_result, requested_signal,
    signal_result, cancellation_checkpoint_epoch,
    cancellation_checkpoint_model_id, inference_action, outcome_status,
    detail, updated_at
) ON experiment_admin_worker_outcome TO pqxx;
GRANT SELECT, INSERT, UPDATE ON experiment_global_control TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_eval TO pqxx;
