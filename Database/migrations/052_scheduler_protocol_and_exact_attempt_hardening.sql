-- Scheduler protocol generation 52:
-- mixed-version cutover barrier, exact-attempt administrative identity,
-- durable in-process checkpoint analysis, and guarded dispatch claims.

BEGIN;

CREATE TABLE IF NOT EXISTS experiment_scheduler_protocol (
    singleton boolean PRIMARY KEY DEFAULT true CHECK (singleton),
    required_generation integer NOT NULL CHECK (required_generation > 0),
    cutover_state text NOT NULL
        CHECK (cutover_state IN ('pending', 'complete', 'failed')),
    migration_installed_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    cutover_completed_at timestamptz,
    cutover_completed_by text,
    cutover_executable_path text,
    cutover_process_evidence text,
    legacy_no_pid_grace_seconds integer NOT NULL DEFAULT 120
        CHECK (legacy_no_pid_grace_seconds BETWEEN 30 AND 86400),
    failure_diagnostic text,
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    CHECK (
        (cutover_state = 'complete'
         AND cutover_completed_at IS NOT NULL
         AND cutover_completed_by IS NOT NULL
         AND cutover_executable_path LIKE '/%'
         AND cutover_process_evidence IS NOT NULL)
        OR
        (cutover_state IN ('pending', 'failed')
         AND cutover_completed_at IS NULL)
    )
);

INSERT INTO experiment_scheduler_protocol (
    singleton, required_generation, cutover_state
)
VALUES (true, 52, 'pending')
ON CONFLICT (singleton) DO UPDATE
SET required_generation = GREATEST(
        experiment_scheduler_protocol.required_generation,
        EXCLUDED.required_generation
    ),
    updated_at = clock_timestamp();

ALTER TABLE experiment_scheduler_invocation
    ADD COLUMN IF NOT EXISTS protocol_generation integer;

ALTER TABLE experiment_scheduler_worker_attempt
    ADD COLUMN IF NOT EXISTS reconciled_at timestamptz,
    ADD COLUMN IF NOT EXISTS reconciled_by_scheduler_invocation_id text
        REFERENCES experiment_scheduler_invocation(scheduler_invocation_id);

ALTER TABLE experiment_admin_worker_outcome
    ADD COLUMN IF NOT EXISTS worker_attempt_id bigint
        REFERENCES experiment_scheduler_worker_attempt(worker_attempt_id)
        ON DELETE RESTRICT;

-- Freeze any pre-052 administrative plan to the exact still-active attempt.
-- Rows without one unambiguous lifecycle binding remain NULL and therefore
-- non-signalable.
UPDATE experiment_admin_worker_outcome o
SET worker_attempt_id = a.worker_attempt_id
FROM experiment e
JOIN experiment_scheduler_worker_attempt a
  ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id
WHERE o.worker_attempt_id IS NULL
  AND o.worker_kind='experiment'
  AND o.checkpoint_eval_id IS NULL
  AND e.experiment_id=o.experiment_id
  AND a.worker_kind='experiment'
  AND a.experiment_id=o.experiment_id
  AND a.checkpoint_eval_id IS NULL
  AND a.lifecycle_phase=o.phase
  AND a.lifecycle_state IN (
      'reserved', 'spawned', 'running', 'observed',
      'identity_ambiguous'
  );

UPDATE experiment_admin_worker_outcome o
SET worker_attempt_id = a.worker_attempt_id
FROM experiment_checkpoint_eval ce
JOIN experiment_scheduler_worker_attempt a
  ON a.worker_attempt_id=ce.active_scheduler_worker_attempt_id
WHERE o.worker_attempt_id IS NULL
  AND o.worker_kind='checkpoint_infer'
  AND o.checkpoint_eval_id IS NOT NULL
  AND ce.checkpoint_eval_id=o.checkpoint_eval_id
  AND a.worker_kind='checkpoint_infer'
  AND a.checkpoint_eval_id=o.checkpoint_eval_id
  AND a.lifecycle_phase='infer'
  AND a.lifecycle_state IN (
      'reserved', 'spawned', 'running', 'observed',
      'identity_ambiguous'
  );

-- Replace migration-051 generated checks with named generation-52 checks.
DO $$
DECLARE
    item record;
BEGIN
    FOR item IN
        SELECT conname
        FROM pg_constraint
        WHERE conrelid =
              'experiment_scheduler_worker_attempt'::regclass
          AND contype = 'c'
          AND (
              pg_get_constraintdef(oid) LIKE
                  '%worker_kind%checkpoint_infer%'
              OR pg_get_constraintdef(oid) LIKE
                  '%ownership_origin%legacy_unverified%'
              OR pg_get_constraintdef(oid) LIKE
                  '%worker_kind = ''experiment''%'
              OR pg_get_constraintdef(oid) LIKE
                  '%lifecycle_state = ANY%worker_pid%'
          )
    LOOP
        EXECUTE format(
            'ALTER TABLE experiment_scheduler_worker_attempt '
            'DROP CONSTRAINT %I',
            item.conname
        );
    END LOOP;
END $$;

ALTER TABLE experiment_scheduler_worker_attempt
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_generation_52_kind_check,
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_generation_52_origin_check,
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_generation_52_shape_check,
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_generation_52_pid_check;

ALTER TABLE experiment_scheduler_worker_attempt
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_generation_52_kind_check
        CHECK (worker_kind IN (
            'experiment', 'checkpoint_infer', 'checkpoint_analyze'
        )),
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_generation_52_origin_check
        CHECK (ownership_origin IN (
            'scheduler_launch', 'scheduler_in_process',
            'prior_scheduler_observed', 'legacy_unverified'
        )),
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_generation_52_shape_check
        CHECK (
            (worker_kind = 'experiment'
             AND experiment_id IS NOT NULL
             AND checkpoint_eval_id IS NULL
             AND lifecycle_phase IN ('train', 'infer', 'analyze')
             AND capacity_class = lifecycle_phase)
            OR
            (worker_kind = 'checkpoint_infer'
             AND experiment_id IS NOT NULL
             AND checkpoint_eval_id IS NOT NULL
             AND lifecycle_phase = 'infer'
             AND capacity_class = 'infer')
            OR
            (worker_kind = 'checkpoint_analyze'
             AND experiment_id IS NOT NULL
             AND checkpoint_eval_id IS NOT NULL
             AND lifecycle_phase = 'analyze'
             AND capacity_class = 'analyze')
        ),
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_generation_52_pid_check
        CHECK (
            worker_kind = 'checkpoint_analyze'
            OR lifecycle_state IN (
                'reserved', 'identity_ambiguous',
                'completed', 'failed', 'launch_failed', 'abandoned'
            )
            OR worker_pid IS NOT NULL
        );

CREATE UNIQUE INDEX IF NOT EXISTS
    scheduler_worker_attempt_active_checkpoint_analyze_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_analyze'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

CREATE INDEX IF NOT EXISTS
    experiment_admin_worker_outcome_attempt_idx
    ON experiment_admin_worker_outcome(worker_attempt_id)
    WHERE worker_attempt_id IS NOT NULL;

CREATE OR REPLACE FUNCTION
    expertadvisor_require_scheduler_protocol_generation_52()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    configured_generation text;
    protocol_ready boolean;
    dispatch_claim boolean;
BEGIN
    dispatch_claim :=
        NEW.status = 'running'
        AND (
            OLD.status IS DISTINCT FROM NEW.status
            OR OLD.phase IS DISTINCT FROM NEW.phase
            OR OLD.active_scheduler_worker_attempt_id
               IS DISTINCT FROM NEW.active_scheduler_worker_attempt_id
        );
    IF NOT dispatch_claim THEN
        RETURN NEW;
    END IF;

    configured_generation :=
        current_setting(
            'expertadvisor.scheduler_protocol_generation',
            true
        );
    SELECT EXISTS (
        SELECT 1
        FROM experiment_scheduler_protocol
        WHERE singleton
          AND required_generation = 52
          AND cutover_state = 'complete'
    )
    INTO protocol_ready;

    IF configured_generation IS DISTINCT FROM '52'
       OR NOT protocol_ready
       OR NEW.active_scheduler_worker_attempt_id IS NULL
    THEN
        RAISE EXCEPTION
            'scheduler protocol generation 52 dispatch barrier rejected claim: configured_generation=%, protocol_ready=%, active_attempt_present=%, old_status=%, new_status=%, old_phase=%, new_phase=%',
            COALESCE(configured_generation, 'NULL'),
            protocol_ready,
            NEW.active_scheduler_worker_attempt_id IS NOT NULL,
            OLD.status,
            NEW.status,
            OLD.phase,
            NEW.phase
            USING ERRCODE = '55000';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS
    experiment_scheduler_protocol_generation_52_guard
    ON experiment;
CREATE TRIGGER experiment_scheduler_protocol_generation_52_guard
BEFORE UPDATE OF status, phase, active_scheduler_worker_attempt_id
ON experiment
FOR EACH ROW
EXECUTE FUNCTION
    expertadvisor_require_scheduler_protocol_generation_52();

DROP TRIGGER IF EXISTS
    checkpoint_eval_scheduler_protocol_generation_52_guard
    ON experiment_checkpoint_eval;
CREATE TRIGGER checkpoint_eval_scheduler_protocol_generation_52_guard
BEFORE UPDATE OF status, phase, active_scheduler_worker_attempt_id
ON experiment_checkpoint_eval
FOR EACH ROW
EXECUTE FUNCTION
    expertadvisor_require_scheduler_protocol_generation_52();

CREATE OR REPLACE FUNCTION
    expertadvisor_validate_experiment_active_attempt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    bound_phase text := NEW.phase;
BEGIN
    IF NEW.active_scheduler_worker_attempt_id IS NULL THEN
        RETURN NEW;
    END IF;
    IF TG_OP = 'UPDATE'
       AND OLD.active_scheduler_worker_attempt_id =
           NEW.active_scheduler_worker_attempt_id
    THEN
        bound_phase := OLD.phase;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM experiment_scheduler_worker_attempt a
        WHERE a.worker_attempt_id =
              NEW.active_scheduler_worker_attempt_id
          AND a.worker_kind = 'experiment'
          AND a.experiment_id = NEW.experiment_id
          AND a.checkpoint_eval_id IS NULL
          AND a.lifecycle_phase = bound_phase
          AND a.capacity_class = bound_phase
          AND a.lifecycle_state IN (
              'reserved', 'spawned', 'running', 'observed',
              'identity_ambiguous'
          )
    ) THEN
        RAISE EXCEPTION
            'experiment active worker attempt identity mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;

CREATE OR REPLACE FUNCTION
    expertadvisor_validate_checkpoint_active_attempt()
RETURNS trigger
LANGUAGE plpgsql
AS $$
DECLARE
    bound_phase text := NEW.phase;
BEGIN
    IF NEW.active_scheduler_worker_attempt_id IS NULL THEN
        RETURN NEW;
    END IF;
    IF TG_OP = 'UPDATE'
       AND OLD.active_scheduler_worker_attempt_id =
           NEW.active_scheduler_worker_attempt_id
    THEN
        bound_phase := OLD.phase;
    END IF;
    IF NOT EXISTS (
        SELECT 1
        FROM experiment_scheduler_worker_attempt a
        WHERE a.worker_attempt_id =
              NEW.active_scheduler_worker_attempt_id
          AND a.experiment_id =
              COALESCE(NEW.parent_experiment_id, NEW.experiment_id)
          AND a.checkpoint_eval_id = NEW.checkpoint_eval_id
          AND (
              (bound_phase = 'infer'
               AND a.worker_kind = 'checkpoint_infer'
               AND a.lifecycle_phase = 'infer'
               AND a.capacity_class = 'infer')
              OR
              (bound_phase = 'analyze'
               AND a.worker_kind = 'checkpoint_analyze'
               AND a.lifecycle_phase = 'analyze'
               AND a.capacity_class = 'analyze')
          )
          AND a.lifecycle_state IN (
              'reserved', 'spawned', 'running', 'observed',
              'identity_ambiguous'
          )
    ) THEN
        RAISE EXCEPTION
            'checkpoint active worker attempt identity mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;

DROP TRIGGER IF EXISTS
    experiment_exact_active_attempt_guard ON experiment;
CREATE TRIGGER experiment_exact_active_attempt_guard
BEFORE INSERT OR UPDATE OF
    phase, active_scheduler_worker_attempt_id
ON experiment
FOR EACH ROW
EXECUTE FUNCTION
    expertadvisor_validate_experiment_active_attempt();

DROP TRIGGER IF EXISTS
    checkpoint_exact_active_attempt_guard
    ON experiment_checkpoint_eval;
CREATE TRIGGER checkpoint_exact_active_attempt_guard
BEFORE INSERT OR UPDATE OF
    phase, active_scheduler_worker_attempt_id
ON experiment_checkpoint_eval
FOR EACH ROW
EXECUTE FUNCTION
    expertadvisor_validate_checkpoint_active_attempt();

GRANT SELECT ON experiment_scheduler_protocol TO pqxx;
GRANT UPDATE (
    cutover_state, cutover_completed_at, cutover_completed_by,
    cutover_executable_path, cutover_process_evidence,
    failure_diagnostic, updated_at
) ON experiment_scheduler_protocol TO pqxx;

GRANT UPDATE (
    reconciled_at, reconciled_by_scheduler_invocation_id
) ON experiment_scheduler_worker_attempt TO pqxx;

GRANT UPDATE (protocol_generation)
ON experiment_scheduler_invocation TO pqxx;

GRANT UPDATE (worker_attempt_id)
ON experiment_admin_worker_outcome TO pqxx;

COMMIT;
