-- Restore one persisted scheduler operation vocabulary while allowing a
-- rolling deployment to safely absorb writes from an already-running legacy
-- worker. Lifecycle control detail remains in status, phase,
-- worker_control_state, administrative outcomes, and error_message.

-- Serialize reconciliation with legacy worker writes. The repository migration
-- runner wraps this file and its schema_migrations insert in one transaction.
LOCK TABLE experiment IN ACCESS EXCLUSIVE MODE;

UPDATE experiment
SET current_operation =
    CASE current_operation
        WHEN 'training' THEN 'train'
        WHEN 'inference' THEN 'infer'
        WHEN 'analysis' THEN 'analyze'
        ELSE current_operation
    END
WHERE current_operation IN ('training', 'inference', 'analysis');

-- Reconcile historical control labels only when the authoritative persisted
-- fields prove the exact shape written by the legacy producer. Any mismatch or
-- unsupported value remains visible and aborts the migration below.
UPDATE experiment
SET current_operation =
    CASE current_operation
        WHEN 'checkpoint_stopped' THEN phase
        WHEN 'cancel_checkpoint_reached' THEN 'train'
        WHEN 'cancel_checkpoint_restart_pending' THEN 'train'
        WHEN 'cancelled_by_global_request' THEN phase
        ELSE current_operation
    END
WHERE
    (
        current_operation = 'checkpoint_stopped'
        AND status = 'pending'
        AND phase IN ('infer', 'analyze')
        AND stopped_at_checkpoint_epoch IS NOT NULL
        AND stopped_at_checkpoint_model_id IS NOT NULL
        AND last_model_id = stopped_at_checkpoint_model_id
    )
    OR
    (
        current_operation = 'cancel_checkpoint_reached'
        AND status = 'cancelled'
        AND phase = 'train'
        AND cancellation_request_id IS NOT NULL
        AND cancellation_completed_at IS NOT NULL
        AND stopped_at_checkpoint_epoch IS NOT NULL
        AND stopped_at_checkpoint_model_id IS NOT NULL
        AND last_model_id = stopped_at_checkpoint_model_id
    )
    OR
    (
        current_operation = 'cancel_checkpoint_restart_pending'
        AND status = 'pending'
        AND phase = 'train'
        AND cancellation_request_id IS NOT NULL
        AND last_model_id IS NOT NULL
    )
    OR
    (
        current_operation = 'cancelled_by_global_request'
        AND status = 'cancelled'
        AND phase IN ('train', 'infer', 'analyze')
        AND cancellation_request_id IS NOT NULL
        AND cancellation_completed_at IS NOT NULL
    );

DO $$
DECLARE
    unsupported_values text;
BEGIN
    SELECT string_agg(
        format(
            'experiment_id=%s,value=%L,status=%L,phase=%L',
            experiment_id,
            current_operation,
            status,
            phase
        ),
        '; ' ORDER BY experiment_id
    )
    INTO unsupported_values
    FROM experiment
    WHERE current_operation IS NOT NULL
      AND current_operation NOT IN ('train', 'infer', 'analyze');

    IF unsupported_values IS NOT NULL THEN
        RAISE EXCEPTION
            'unsupported experiment.current_operation rows: %',
            unsupported_values;
    END IF;
END
$$;

CREATE OR REPLACE FUNCTION canonicalize_experiment_current_operation()
RETURNS trigger
LANGUAGE plpgsql
AS $$
BEGIN
    NEW.current_operation :=
        CASE NEW.current_operation
            WHEN 'training' THEN 'train'
            WHEN 'inference' THEN 'infer'
            WHEN 'analysis' THEN 'analyze'
            WHEN 'checkpoint_stopped' THEN
                CASE
                    WHEN NEW.status = 'pending'
                     AND NEW.phase IN ('infer', 'analyze')
                     AND NEW.stopped_at_checkpoint_epoch IS NOT NULL
                     AND NEW.stopped_at_checkpoint_model_id IS NOT NULL
                     AND NEW.last_model_id = NEW.stopped_at_checkpoint_model_id
                    THEN NEW.phase
                    ELSE NEW.current_operation
                END
            WHEN 'cancel_checkpoint_reached' THEN
                CASE
                    WHEN NEW.status = 'cancelled'
                     AND NEW.phase = 'train'
                     AND NEW.cancellation_request_id IS NOT NULL
                     AND NEW.cancellation_completed_at IS NOT NULL
                     AND NEW.stopped_at_checkpoint_epoch IS NOT NULL
                     AND NEW.stopped_at_checkpoint_model_id IS NOT NULL
                     AND NEW.last_model_id = NEW.stopped_at_checkpoint_model_id
                    THEN 'train'
                    ELSE NEW.current_operation
                END
            WHEN 'cancel_checkpoint_restart_pending' THEN
                CASE
                    WHEN NEW.status = 'pending'
                     AND NEW.phase = 'train'
                     AND NEW.cancellation_request_id IS NOT NULL
                     AND NEW.last_model_id IS NOT NULL
                    THEN 'train'
                    ELSE NEW.current_operation
                END
            WHEN 'cancelled_by_global_request' THEN
                CASE
                    WHEN NEW.status = 'cancelled'
                     AND NEW.phase IN ('train', 'infer', 'analyze')
                     AND NEW.cancellation_request_id IS NOT NULL
                     AND NEW.cancellation_completed_at IS NOT NULL
                    THEN NEW.phase
                    ELSE NEW.current_operation
                END
            ELSE NEW.current_operation
        END;
    RETURN NEW;
END;
$$;

DROP TRIGGER IF EXISTS experiment_current_operation_canonicalize
    ON experiment;
CREATE TRIGGER experiment_current_operation_canonicalize
BEFORE INSERT OR UPDATE OF current_operation, phase
ON experiment
FOR EACH ROW
EXECUTE FUNCTION canonicalize_experiment_current_operation();

-- This trigger is intentionally temporary rolling-deployment compatibility.
-- Remove it in a later reviewed migration only after all pre-050 schedulers,
-- workers, and administrative executables have exited. Keep the constraint.
ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_current_operation_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_current_operation_check
    CHECK (
        current_operation IS NULL OR
        current_operation IN ('train', 'infer', 'analyze')
    );
