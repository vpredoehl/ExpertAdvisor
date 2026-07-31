#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
test_db="ea_scheduler_ownership_test_${$}"

case "${test_db}" in
    ea_scheduler_ownership_test_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
CREATE TABLE model (
    model_id bigserial PRIMARY KEY
);
CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY,
    symbol text NOT NULL,
    prediction_horizon integer NOT NULL,
    c_next_threshold double precision NOT NULL,
    core_lr_mult double precision,
    head_lr_mult double precision,
    target_epochs integer NOT NULL,
    checkpoint_interval integer NOT NULL DEFAULT 20,
    train_start timestamptz NOT NULL,
    train_end timestamptz NOT NULL,
    infer_start timestamptz,
    infer_end timestamptz,
    status text NOT NULL,
    phase text NOT NULL,
    last_model_id bigint,
    resume_model_id bigint,
    train_log_path text,
    infer_log_path text,
    analysis_log_path text,
    worker_pid integer,
    worker_process_group_id integer,
    worker_process_start_identity text,
    worker_executable text,
    worker_command_line text,
    worker_started_at timestamptz,
    current_operation text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    duplicate_nonce bigint NOT NULL DEFAULT 0,
    completed_at timestamptz,
    exit_code integer,
    error_message text
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigserial PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch integer NOT NULL,
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id),
    status text NOT NULL,
    phase text NOT NULL,
    worker_pid integer,
    worker_process_group_id integer,
    worker_process_start_identity text,
    worker_executable text,
    worker_command_line text,
    infer_log_path text,
    started_at timestamptz,
    infer_started_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    completed_at timestamptz,
    error_message text
);
CREATE TABLE experiment_admin_request (
    request_id bigserial PRIMARY KEY
);
CREATE TABLE experiment_admin_worker_outcome (
    request_id bigint NOT NULL
        REFERENCES experiment_admin_request(request_id),
    worker_identity text NOT NULL,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_eval_id bigint
        REFERENCES experiment_checkpoint_eval(checkpoint_eval_id),
    worker_kind text NOT NULL DEFAULT 'experiment',
    phase text NOT NULL DEFAULT 'train',
    PRIMARY KEY(request_id, worker_identity)
);

INSERT INTO model(model_id) VALUES (910050);
INSERT INTO experiment (
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    status,phase,worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,
    worker_command_line,worker_started_at,current_operation,
    duplicate_nonce
) VALUES (
    910050,'legacyfixture',4,0.0008,20,20,
    '2020-01-01','2021-01-01','running','train',
    910050,910050,'legacy-start','/tmp/LSTM_Release',
    '/tmp/LSTM_Release --train --scheduler-experiment-id=910050',
    clock_timestamp(),'train',910050
);
SQL

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
DO $$
DECLARE
    attempts integer;
    linked boolean;
BEGIN
    SELECT count(*) INTO attempts
    FROM experiment_scheduler_worker_attempt
    WHERE experiment_id=910050
      AND ownership_origin='legacy_unverified'
      AND lifecycle_state='identity_ambiguous';
    SELECT active_scheduler_worker_attempt_id IS NOT NULL INTO linked
    FROM experiment WHERE experiment_id=910050;
    IF attempts <> 1 OR NOT linked THEN
        RAISE EXCEPTION
            'legacy active row was not conservatively capacity-accounted';
    END IF;
END $$;

DO $$
DECLARE
    legacy_attempt_id bigint;
BEGIN
    SELECT active_scheduler_worker_attempt_id
    INTO legacy_attempt_id
    FROM experiment
    WHERE experiment_id=910050;
    UPDATE experiment
    SET status='pending'
    WHERE experiment_id=910050;
    BEGIN
        PERFORM set_config(
            'expertadvisor.scheduler_protocol_generation',
            '52',
            true
        );
        UPDATE experiment
        SET status='running'
        WHERE experiment_id=910050;
        RAISE EXCEPTION
            'pending protocol barrier accepted a dispatch claim';
    EXCEPTION
        WHEN sqlstate '55000' THEN
            NULL;
    END;
    IF (SELECT cutover_state
        FROM experiment_scheduler_protocol
        WHERE singleton) <> 'pending'
    THEN
        RAISE EXCEPTION 'barrier rejection mutated cutover state';
    END IF;
    IF (SELECT active_scheduler_worker_attempt_id
        FROM experiment
        WHERE experiment_id=910050) <> legacy_attempt_id
    THEN
        RAISE EXCEPTION 'barrier rejection mutated lifecycle binding';
    END IF;
END $$;

UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='isolated_migration_test',
    cutover_executable_path='/tmp/LSTM_Release',
    cutover_process_evidence=
        'isolated_disposable_database;no_dispatch_process_can_connect',
    updated_at=clock_timestamp()
WHERE singleton
  AND required_generation=52
  AND cutover_state='pending';

UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='failed',completed_at=clock_timestamp()
WHERE experiment_id=910050;
UPDATE experiment
SET status='failed',active_scheduler_worker_attempt_id=NULL
WHERE experiment_id=910050;
SQL

# Direct replay proves the DDL/data migration itself is conservative and
# idempotent in addition to checksum-based repository migration bookkeeping.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/SchedulerOwnershipMigrationTests.sql"

printf '%s\n' "SchedulerOwnershipIntegrationTests passed"
