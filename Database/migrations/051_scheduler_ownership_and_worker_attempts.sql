-- Database-authoritative scheduler ownership, fenced launch attempts, and
-- conservative migration of pre-existing active workers.

BEGIN;

CREATE TABLE IF NOT EXISTS experiment_scheduler_invocation (
    scheduler_invocation_id text PRIMARY KEY,
    process_pid integer NOT NULL CHECK (process_pid > 0),
    process_group_id integer NOT NULL CHECK (process_group_id > 0),
    process_start_identity text NOT NULL CHECK (length(process_start_identity) > 0),
    canonical_executable_path text NOT NULL
        CHECK (canonical_executable_path LIKE '/%'),
    command_line text NOT NULL CHECK (length(command_line) > 0),
    invocation_nonce text NOT NULL UNIQUE CHECK (length(invocation_nonce) >= 16),
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    last_heartbeat_at timestamptz,
    ownership_acquired_at timestamptz,
    ownership_released_at timestamptz,
    ended_at timestamptz,
    status text NOT NULL DEFAULT 'starting'
        CHECK (status IN (
            'starting', 'owner', 'rejected', 'released', 'lost', 'crashed'
        )),
    terminal_reason text
);

CREATE TABLE IF NOT EXISTS experiment_scheduler_lease (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    owner_scheduler_invocation_id text
        REFERENCES experiment_scheduler_invocation(scheduler_invocation_id),
    fencing_token bigint NOT NULL DEFAULT 0 CHECK (fencing_token >= 0),
    authority_state text NOT NULL DEFAULT 'vacant'
        CHECK (authority_state IN ('vacant', 'active', 'released', 'lost')),
    acquired_at timestamptz,
    heartbeat_at timestamptz,
    expires_at timestamptz,
    released_at timestamptz,
    transition_reason text NOT NULL DEFAULT 'migration_initialized',
    CHECK (
        (authority_state = 'vacant'
         AND owner_scheduler_invocation_id IS NULL
         AND acquired_at IS NULL
         AND heartbeat_at IS NULL
         AND expires_at IS NULL)
        OR
        (authority_state IN ('active', 'released', 'lost')
         AND owner_scheduler_invocation_id IS NOT NULL
         AND acquired_at IS NOT NULL
         AND heartbeat_at IS NOT NULL
         AND expires_at IS NOT NULL)
    )
);

INSERT INTO experiment_scheduler_lease (singleton)
VALUES (true)
ON CONFLICT (singleton) DO NOTHING;

CREATE TABLE IF NOT EXISTS experiment_scheduler_worker_attempt (
    worker_attempt_id bigserial PRIMARY KEY,
    launch_attempt_identity text NOT NULL UNIQUE
        CHECK (length(launch_attempt_identity) >= 16),
    scheduler_invocation_id text
        REFERENCES experiment_scheduler_invocation(scheduler_invocation_id),
    scheduler_fencing_token bigint CHECK (scheduler_fencing_token IS NULL
                                           OR scheduler_fencing_token > 0),
    experiment_id bigint
        REFERENCES experiment(experiment_id) ON DELETE RESTRICT,
    checkpoint_eval_id bigint
        REFERENCES experiment_checkpoint_eval(checkpoint_eval_id)
        ON DELETE RESTRICT,
    worker_kind text NOT NULL
        CHECK (worker_kind IN ('experiment', 'checkpoint_infer')),
    lifecycle_phase text NOT NULL
        CHECK (lifecycle_phase IN ('train', 'infer', 'analyze')),
    capacity_class text NOT NULL
        CHECK (capacity_class IN ('train', 'infer', 'analyze')),
    ownership_origin text NOT NULL DEFAULT 'scheduler_launch'
        CHECK (ownership_origin IN (
            'scheduler_launch', 'prior_scheduler_observed',
            'legacy_unverified'
        )),
    lifecycle_state text NOT NULL DEFAULT 'reserved'
        CHECK (lifecycle_state IN (
            'reserved', 'spawned', 'running', 'observed',
            'identity_ambiguous', 'completed', 'failed',
            'launch_failed', 'abandoned'
        )),
    worker_pid integer CHECK (worker_pid IS NULL OR worker_pid > 0),
    worker_process_group_id integer
        CHECK (worker_process_group_id IS NULL
               OR worker_process_group_id > 0),
    worker_process_start_identity text,
    canonical_executable_path text,
    command_line text,
    command_identity text NOT NULL CHECK (length(command_identity) > 0),
    log_path text,
    reserved_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    spawned_at timestamptz,
    registered_at timestamptz,
    last_observed_at timestamptz,
    completed_at timestamptz,
    observed_by_scheduler_invocation_id text
        REFERENCES experiment_scheduler_invocation(scheduler_invocation_id),
    exit_code integer,
    signal_number integer,
    reconciliation_result text,
    diagnostic text,
    CHECK (
        (worker_kind = 'experiment'
         AND experiment_id IS NOT NULL
         AND checkpoint_eval_id IS NULL
         AND lifecycle_phase IN ('train', 'infer', 'analyze'))
        OR
        (worker_kind = 'checkpoint_infer'
         AND experiment_id IS NOT NULL
         AND checkpoint_eval_id IS NOT NULL
         AND lifecycle_phase = 'infer'
         AND capacity_class = 'infer')
    ),
    CHECK (
        lifecycle_state IN ('reserved', 'identity_ambiguous')
        OR worker_pid IS NOT NULL
        OR completed_at IS NOT NULL
    ),
    CHECK (
        ownership_origin = 'legacy_unverified'
        OR (scheduler_invocation_id IS NOT NULL
            AND scheduler_fencing_token IS NOT NULL)
    )
);

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS active_scheduler_worker_attempt_id bigint;

ALTER TABLE experiment_checkpoint_eval
    ADD COLUMN IF NOT EXISTS active_scheduler_worker_attempt_id bigint;

DO $$
BEGIN
    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment'::regclass
          AND conname = 'experiment_active_scheduler_worker_attempt_fkey'
    ) THEN
        ALTER TABLE experiment
            ADD CONSTRAINT experiment_active_scheduler_worker_attempt_fkey
            FOREIGN KEY (active_scheduler_worker_attempt_id)
            REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id)
            DEFERRABLE INITIALLY IMMEDIATE;
    END IF;

    IF NOT EXISTS (
        SELECT 1 FROM pg_constraint
        WHERE conrelid = 'experiment_checkpoint_eval'::regclass
          AND conname =
              'checkpoint_eval_active_scheduler_worker_attempt_fkey'
    ) THEN
        ALTER TABLE experiment_checkpoint_eval
            ADD CONSTRAINT
                checkpoint_eval_active_scheduler_worker_attempt_fkey
            FOREIGN KEY (active_scheduler_worker_attempt_id)
            REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id)
            DEFERRABLE INITIALLY IMMEDIATE;
    END IF;
END $$;

-- An active attempt is the capacity authority even if its lifecycle row has
-- become inconsistent.  Ambiguous identity deliberately continues consuming
-- capacity until exact process evidence permits deterministic reconciliation.
CREATE UNIQUE INDEX IF NOT EXISTS
    experiment_scheduler_worker_attempt_active_experiment_uidx
    ON experiment_scheduler_worker_attempt(experiment_id, lifecycle_phase)
    WHERE worker_kind = 'experiment'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE UNIQUE INDEX IF NOT EXISTS
    experiment_scheduler_worker_attempt_active_checkpoint_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_infer'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE INDEX IF NOT EXISTS
    experiment_scheduler_worker_attempt_capacity_idx
    ON experiment_scheduler_worker_attempt(capacity_class, lifecycle_state);

CREATE INDEX IF NOT EXISTS
    experiment_scheduler_worker_attempt_scheduler_idx
    ON experiment_scheduler_worker_attempt(
        scheduler_invocation_id, lifecycle_state, worker_attempt_id
    );

-- Existing active lifecycle rows are not silently asserted to be valid
-- workers.  They become conservative legacy reservations, carrying only the
-- evidence already persisted.  They consume capacity until a new authoritative
-- scheduler validates the exact OS process or proves it absent.
INSERT INTO experiment_scheduler_worker_attempt (
    launch_attempt_identity,
    experiment_id,
    worker_kind,
    lifecycle_phase,
    capacity_class,
    ownership_origin,
    lifecycle_state,
    worker_pid,
    worker_process_group_id,
    worker_process_start_identity,
    canonical_executable_path,
    command_line,
    command_identity,
    log_path,
    reserved_at,
    spawned_at,
    registered_at,
    diagnostic
)
SELECT
    'legacy:experiment:' || e.experiment_id::text || ':' || e.phase,
    e.experiment_id,
    'experiment',
    e.phase,
    e.phase,
    'legacy_unverified',
    'identity_ambiguous',
    e.worker_pid,
    e.worker_process_group_id,
    e.worker_process_start_identity,
    e.worker_executable,
    e.worker_command_line,
    'experiment:' || e.experiment_id::text || ':' || e.phase,
    CASE e.phase
        WHEN 'train' THEN e.train_log_path
        WHEN 'infer' THEN e.infer_log_path
        ELSE e.analysis_log_path
    END,
    COALESCE(e.worker_started_at, e.updated_at, clock_timestamp()),
    CASE WHEN e.worker_pid IS NULL THEN NULL
         ELSE COALESCE(e.worker_started_at, e.updated_at, clock_timestamp())
    END,
    CASE WHEN e.worker_process_start_identity IS NULL THEN NULL
         ELSE COALESCE(e.worker_started_at, e.updated_at, clock_timestamp())
    END,
    'migration_051_conservative_legacy_evidence'
FROM experiment e
WHERE e.status = 'running'
  AND e.phase IN ('train', 'infer', 'analyze')
  AND e.active_scheduler_worker_attempt_id IS NULL
ON CONFLICT (launch_attempt_identity) DO NOTHING;

UPDATE experiment e
SET active_scheduler_worker_attempt_id = a.worker_attempt_id
FROM experiment_scheduler_worker_attempt a
WHERE e.active_scheduler_worker_attempt_id IS NULL
  AND a.ownership_origin = 'legacy_unverified'
  AND a.worker_kind = 'experiment'
  AND a.experiment_id = e.experiment_id
  AND a.lifecycle_phase = e.phase
  AND a.lifecycle_state = 'identity_ambiguous';

INSERT INTO experiment_scheduler_worker_attempt (
    launch_attempt_identity,
    experiment_id,
    checkpoint_eval_id,
    worker_kind,
    lifecycle_phase,
    capacity_class,
    ownership_origin,
    lifecycle_state,
    worker_pid,
    worker_process_group_id,
    worker_process_start_identity,
    canonical_executable_path,
    command_line,
    command_identity,
    log_path,
    reserved_at,
    spawned_at,
    registered_at,
    diagnostic
)
SELECT
    'legacy:checkpoint:' || ce.checkpoint_eval_id::text || ':infer',
    ce.parent_experiment_id,
    ce.checkpoint_eval_id,
    'checkpoint_infer',
    'infer',
    'infer',
    'legacy_unverified',
    'identity_ambiguous',
    ce.worker_pid,
    ce.worker_process_group_id,
    ce.worker_process_start_identity,
    ce.worker_executable,
    ce.worker_command_line,
    'checkpoint_infer:' || ce.checkpoint_eval_id::text,
    ce.infer_log_path,
    COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at,
             clock_timestamp()),
    CASE WHEN ce.worker_pid IS NULL THEN NULL
         ELSE COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at,
                       clock_timestamp())
    END,
    CASE WHEN ce.worker_process_start_identity IS NULL THEN NULL
         ELSE COALESCE(ce.infer_started_at, ce.started_at, ce.updated_at,
                       clock_timestamp())
    END,
    'migration_051_conservative_legacy_evidence'
FROM experiment_checkpoint_eval ce
WHERE ce.status = 'running'
  AND ce.phase = 'infer'
  AND ce.active_scheduler_worker_attempt_id IS NULL
ON CONFLICT (launch_attempt_identity) DO NOTHING;

UPDATE experiment_checkpoint_eval ce
SET active_scheduler_worker_attempt_id = a.worker_attempt_id
FROM experiment_scheduler_worker_attempt a
WHERE ce.active_scheduler_worker_attempt_id IS NULL
  AND a.ownership_origin = 'legacy_unverified'
  AND a.worker_kind = 'checkpoint_infer'
  AND a.checkpoint_eval_id = ce.checkpoint_eval_id
  AND a.lifecycle_state = 'identity_ambiguous';

GRANT SELECT, INSERT ON experiment_scheduler_invocation TO pqxx;
GRANT UPDATE (
    last_heartbeat_at, ownership_acquired_at, ownership_released_at,
    ended_at, status, terminal_reason
) ON experiment_scheduler_invocation TO pqxx;

GRANT SELECT, UPDATE ON experiment_scheduler_lease TO pqxx;

GRANT SELECT, INSERT ON experiment_scheduler_worker_attempt TO pqxx;
GRANT UPDATE (
    lifecycle_state, worker_pid, worker_process_group_id,
    worker_process_start_identity, canonical_executable_path, command_line,
    spawned_at, registered_at, last_observed_at, completed_at,
    observed_by_scheduler_invocation_id, exit_code, signal_number,
    reconciliation_result, diagnostic
) ON experiment_scheduler_worker_attempt TO pqxx;
GRANT USAGE, SELECT ON SEQUENCE
    experiment_scheduler_worker_attempt_worker_attempt_id_seq TO pqxx;

GRANT SELECT, INSERT, UPDATE, DELETE ON experiment TO pqxx;
GRANT SELECT, INSERT, UPDATE, DELETE ON experiment_checkpoint_eval TO pqxx;

COMMIT;
