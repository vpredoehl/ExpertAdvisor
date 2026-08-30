-- Persistent scheduler priority and admission-safe stopped-worker state.

BEGIN;

ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS scheduler_priority text NOT NULL DEFAULT 'normal',
    ADD COLUMN IF NOT EXISTS resume_requested boolean NOT NULL DEFAULT false;

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_scheduler_priority_check;

ALTER TABLE experiment
    ADD CONSTRAINT experiment_scheduler_priority_check
        CHECK (scheduler_priority IN ('high', 'normal', 'low'));

-- Migration 051 created this inline check with PostgreSQL's generated name.
-- Replace it so a live but scheduler-suspended process remains an active,
-- uniquely bound attempt without consuming execution capacity.
ALTER TABLE experiment_scheduler_worker_attempt
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_lifecycle_state_check;

ALTER TABLE experiment_scheduler_worker_attempt
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_lifecycle_state_check
        CHECK (lifecycle_state IN (
            'reserved', 'spawned', 'running', 'observed',
            'stopped', 'identity_ambiguous', 'completed', 'failed',
            'launch_failed', 'abandoned'
        ));

ALTER TABLE experiment_scheduler_worker_attempt
    DROP CONSTRAINT IF EXISTS
        experiment_scheduler_worker_attempt_stopped_shape_check;

ALTER TABLE experiment_scheduler_worker_attempt
    ADD CONSTRAINT
        experiment_scheduler_worker_attempt_stopped_shape_check
        CHECK (
            lifecycle_state <> 'stopped'
            OR (
                worker_kind = 'experiment'
                AND checkpoint_eval_id IS NULL
                AND lifecycle_phase IN ('train', 'infer')
                AND capacity_class = lifecycle_phase
            )
        );

DROP INDEX IF EXISTS
    experiment_scheduler_worker_attempt_active_experiment_uidx;
CREATE UNIQUE INDEX
    experiment_scheduler_worker_attempt_active_experiment_uidx
    ON experiment_scheduler_worker_attempt(experiment_id, lifecycle_phase)
    WHERE worker_kind = 'experiment'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed', 'stopped',
          'identity_ambiguous'
      );

DROP INDEX IF EXISTS
    experiment_scheduler_worker_attempt_active_checkpoint_uidx;
CREATE UNIQUE INDEX
    experiment_scheduler_worker_attempt_active_checkpoint_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_infer'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

DROP INDEX IF EXISTS scheduler_worker_attempt_active_checkpoint_analyze_uidx;
CREATE UNIQUE INDEX scheduler_worker_attempt_active_checkpoint_analyze_uidx
    ON experiment_scheduler_worker_attempt(checkpoint_eval_id)
    WHERE worker_kind = 'checkpoint_analyze'
      AND lifecycle_state IN (
          'reserved', 'spawned', 'running', 'observed',
          'identity_ambiguous'
      );

-- Migration 052's exact-binding trigger predates the stopped lifecycle.
-- A stopped train/infer process remains the one authoritative active attempt
-- even while its experiment row is paused or pending admission.
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
              'reserved', 'spawned', 'running', 'observed', 'stopped',
              'identity_ambiguous'
          )
    ) THEN
        RAISE EXCEPTION
            'experiment active worker attempt identity mismatch'
            USING ERRCODE = '23514';
    END IF;
    RETURN NEW;
END $$;

DROP INDEX IF EXISTS experiment_scheduler_pending_priority_idx;
CREATE INDEX IF NOT EXISTS experiment_scheduler_pending_priority_idx
    ON experiment (
        phase,
        resume_requested DESC,
        (CASE scheduler_priority
            WHEN 'high' THEN 0
            WHEN 'normal' THEN 1
            ELSE 2
         END),
        updated_at,
        experiment_id
    )
    WHERE status = 'pending';

COMMIT;
