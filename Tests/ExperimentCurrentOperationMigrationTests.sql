BEGIN;

CREATE SCHEMA experiment_current_operation_migration_test;
SET LOCAL search_path TO experiment_current_operation_migration_test, public;

CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY,
    status text NOT NULL,
    phase text NOT NULL,
    current_operation text,
    last_model_id bigint,
    cancellation_request_id bigint,
    cancellation_completed_at timestamptz,
    stopped_at_checkpoint_epoch integer,
    stopped_at_checkpoint_model_id bigint
);

INSERT INTO experiment (
    status, phase, current_operation, last_model_id,
    cancellation_request_id, cancellation_completed_at,
    stopped_at_checkpoint_epoch, stopped_at_checkpoint_model_id
) VALUES
    ('running', 'train', 'train', NULL, NULL, NULL, NULL, NULL),
    ('running', 'infer', 'infer', NULL, NULL, NULL, NULL, NULL),
    ('running', 'analyze', 'analyze', NULL, NULL, NULL, NULL, NULL),
    ('running', 'train', 'training', NULL, NULL, NULL, NULL, NULL),
    ('running', 'infer', 'inference', NULL, NULL, NULL, NULL, NULL),
    ('running', 'analyze', 'analysis', NULL, NULL, NULL, NULL, NULL),
    ('pending', 'infer', 'checkpoint_stopped', 101, NULL, NULL, 20, 101),
    ('cancelled', 'train', 'cancel_checkpoint_reached', 102, 1, now(), 40, 102),
    ('pending', 'train', 'cancel_checkpoint_restart_pending', 103, 1, NULL, NULL, NULL),
    ('cancelled', 'analyze', 'cancelled_by_global_request', NULL, 1, now(), NULL, NULL),
    ('pending', 'train', NULL, NULL, NULL, NULL, NULL, NULL);

\ir ../Database/migrations/050_experiment_current_operation_canonicalization.sql
\ir ../Database/migrations/050_experiment_current_operation_canonicalization.sql

DO $test$
DECLARE
    values_seen text[];
BEGIN
    SELECT array_agg(current_operation ORDER BY current_operation)
    INTO values_seen
    FROM (
        SELECT DISTINCT current_operation
        FROM experiment
        WHERE current_operation IS NOT NULL
    ) canonical;
    IF values_seen <> ARRAY['analyze', 'infer', 'train'] THEN
        RAISE EXCEPTION 'unexpected canonical values: %', values_seen;
    END IF;

    IF EXISTS (
        SELECT 1
        FROM experiment
        WHERE current_operation IS NOT NULL
          AND current_operation NOT IN ('train', 'infer', 'analyze')
    ) THEN
        RAISE EXCEPTION 'migration retained a noncanonical operation';
    END IF;

    IF (
        SELECT array_agg(current_operation ORDER BY experiment_id)
        FROM experiment
        WHERE experiment_id BETWEEN 4 AND 10
    ) <> ARRAY[
        'train', 'infer', 'analyze', 'infer', 'train', 'train', 'analyze'
    ] THEN
        RAISE EXCEPTION 'legacy rows were not reconciled exactly';
    END IF;

    IF (
        SELECT current_operation IS NULL
        FROM experiment
        WHERE experiment_id = 11
    ) IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'NULL current_operation was not preserved';
    END IF;

    INSERT INTO experiment (status, phase, current_operation)
    VALUES ('running', 'train', 'training');
    IF (
        SELECT current_operation
        FROM experiment
        ORDER BY experiment_id DESC
        LIMIT 1
    ) <> 'train' THEN
        RAISE EXCEPTION 'legacy training write was not canonicalized';
    END IF;

    INSERT INTO experiment (
        status, phase, current_operation, cancellation_request_id,
        cancellation_completed_at
    )
    VALUES ('cancelled', 'infer', 'cancelled_by_global_request', 2, now());
    IF (
        SELECT current_operation
        FROM experiment
        ORDER BY experiment_id DESC
        LIMIT 1
    ) <> 'infer' THEN
        RAISE EXCEPTION 'legacy cancellation write was not canonicalized';
    END IF;

    INSERT INTO experiment (
        status, phase, current_operation, last_model_id,
        cancellation_request_id, cancellation_completed_at,
        stopped_at_checkpoint_epoch, stopped_at_checkpoint_model_id
    ) VALUES
        ('pending', 'analyze', 'checkpoint_stopped', 201, NULL, NULL, 20, 201),
        ('cancelled', 'train', 'cancel_checkpoint_reached', 202, 2, now(), 40, 202),
        ('pending', 'train', 'cancel_checkpoint_restart_pending', 203, 2, NULL, NULL, NULL);
    IF (
        SELECT array_agg(current_operation ORDER BY experiment_id)
        FROM (
            SELECT experiment_id, current_operation
            FROM experiment
            ORDER BY experiment_id DESC
            LIMIT 3
        ) legacy_trigger_writes
    ) <> ARRAY['analyze', 'train', 'train'] THEN
        RAISE EXCEPTION 'legacy control INSERTs were not canonicalized';
    END IF;

    INSERT INTO experiment (status, phase, current_operation)
    VALUES ('pending', 'train', NULL);
    IF (
        SELECT current_operation IS NULL
        FROM experiment
        ORDER BY experiment_id DESC
        LIMIT 1
    ) IS DISTINCT FROM true THEN
        RAISE EXCEPTION 'NULL trigger write was not preserved';
    END IF;

    UPDATE experiment
    SET status = 'running', phase = 'analyze', current_operation = 'analysis'
    WHERE experiment_id = 1;
    IF (
        SELECT current_operation
        FROM experiment
        WHERE experiment_id = 1
    ) <> 'analyze' THEN
        RAISE EXCEPTION 'legacy UPDATE was not canonicalized';
    END IF;

    BEGIN
        INSERT INTO experiment (status, phase, current_operation)
        VALUES ('running', 'train', 'unexpected');
        RAISE EXCEPTION 'noncanonical current_operation was accepted';
    EXCEPTION
        WHEN check_violation THEN NULL;
    END;

    BEGIN
        INSERT INTO experiment (
            status, phase, current_operation, cancellation_request_id,
            cancellation_completed_at
        )
        VALUES (
            'cancelled', 'done', 'cancelled_by_global_request', 3, now()
        );
        RAISE EXCEPTION 'ambiguous control label was accepted';
    EXCEPTION
        WHEN check_violation THEN NULL;
    END;
END;
$test$;

ROLLBACK;
