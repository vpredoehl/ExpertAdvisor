BEGIN;

CREATE SCHEMA experiment_current_operation_migration_failure_test;
SET LOCAL search_path TO
    experiment_current_operation_migration_failure_test, public;

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

INSERT INTO experiment (status, phase, current_operation)
VALUES ('completed', 'done', 'support');

\ir ../Database/migrations/050_experiment_current_operation_canonicalization.sql

ROLLBACK;
