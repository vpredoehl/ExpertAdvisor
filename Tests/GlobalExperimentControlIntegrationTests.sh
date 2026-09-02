#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
process_test_binary="${2:-}"
test_db="ea_global_control_test_${$}_$(date +%s)"
test_tmp="$(mktemp -d /tmp/ea_global_control_test.XXXXXX)"
scheduler_fixture_pid=""
scheduler_fixture_pgid=""
scheduler_fixture_start_identity=""
scheduler_fixture_executable=""
migration_fixture_pid=""
migration_fixture_pgid=""
migration_fixture_start_identity=""
migration_fixture_executable=""

duplicate_migration_versions="$(
    for migration in "${repo_root}"/Database/migrations/*.sql; do
        filename="$(basename "${migration}")"
        printf '%s\n' "${filename%%_*}"
    done | sort | uniq -d
)"
if [[ -n "${duplicate_migration_versions}" ]]; then
    echo "duplicate migration versions: ${duplicate_migration_versions}" >&2
    exit 1
fi

read_fixture_identity() {
    local target_pid="$1"
    local identity_line=""
    local observed_pid=""
    local observed_pgid=""
    local observed_start_identity=""
    local observed_executable=""
    local observed_command=""
    identity_line="$(
        "${process_test_binary}" \
            "--inspect-managed-test-process=${target_pid}" 2>/dev/null ||
            true
    )"
    IFS='|' read -r observed_pid observed_pgid observed_start_identity \
        observed_executable observed_command \
        <<<"${identity_line}"
    [[ "${observed_pid}" = "${target_pid}" ]] ||
        return 1
    [[ "${observed_pgid}" = "${target_pid}" ]] ||
        return 1
    [[ -n "${observed_executable}" ]] ||
        return 1
    [[ "${observed_command}" == *"${test_tmp}/LSTM_Release"* ]] ||
        return 1
    [[ "${observed_command}" == *"--managed-test-worker"* ]] ||
        return 1
    [[ "${observed_command}" == *"--scheduler-experiment-id=800002"* ]] ||
        return 1
    [[ "${observed_command}" != *"--schedule-experiments"* ]] ||
        return 1
    [[ "${observed_command}" != *"--scheduler-status"* ]] ||
        return 1
    fixture_observed_pgid="${observed_pgid}"
    fixture_observed_start_identity="${observed_start_identity}"
    fixture_observed_executable="${observed_executable}"
}

capture_fixture_identity() {
    read_fixture_identity "${scheduler_fixture_pid}" ||
        return 1
    scheduler_fixture_pgid="${fixture_observed_pgid}"
    scheduler_fixture_start_identity="${fixture_observed_start_identity}"
    scheduler_fixture_executable="${fixture_observed_executable}"
}

fixture_identity_matches() {
    [[ -n "${scheduler_fixture_pid}" ]] &&
        [[ -n "${scheduler_fixture_pgid}" ]] &&
        [[ -n "${scheduler_fixture_start_identity}" ]] &&
        [[ -n "${scheduler_fixture_executable}" ]] ||
        return 1
    read_fixture_identity "${scheduler_fixture_pid}" ||
        return 1
    [[ "${fixture_observed_pgid}" = "${scheduler_fixture_pgid}" ]] &&
        [[ "${fixture_observed_start_identity}" = \
            "${scheduler_fixture_start_identity}" ]] &&
        [[ "${fixture_observed_executable}" = \
            "${scheduler_fixture_executable}" ]]
}

cleanup() {
    if [[ -n "${migration_fixture_pid}" ]] &&
        kill -0 "${migration_fixture_pid}" >/dev/null 2>&1; then
        local migration_identity=""
        local migration_pid=""
        local migration_pgid=""
        local migration_start=""
        local migration_executable=""
        local migration_command=""
        migration_identity="$(
            "${process_test_binary}" \
                "--inspect-managed-test-process=${migration_fixture_pid}" \
                2>/dev/null || true
        )"
        IFS='|' read -r migration_pid migration_pgid migration_start \
            migration_executable migration_command <<<"${migration_identity}"
        if [[ "${migration_pid}" = "${migration_fixture_pid}" ]] &&
            [[ "${migration_pgid}" = "${migration_fixture_pgid}" ]] &&
            [[ "${migration_start}" = "${migration_fixture_start_identity}" ]] &&
            [[ "${migration_executable}" = \
                "${migration_fixture_executable}" ]]; then
            kill -CONT -- "-${migration_fixture_pgid}" \
                >/dev/null 2>&1 || true
            kill -TERM -- "-${migration_fixture_pgid}" \
                >/dev/null 2>&1 || true
        else
            echo "migration fixture identity could not be proven during cleanup" >&2
        fi
    fi
    if [[ -n "${migration_fixture_pid}" ]]; then
        wait "${migration_fixture_pid}" 2>/dev/null || true
    fi
    if [[ -n "${scheduler_fixture_pid}" ]] &&
        kill -0 "${scheduler_fixture_pid}" >/dev/null 2>&1; then
        if fixture_identity_matches; then
            kill -TERM -- "-${scheduler_fixture_pgid}" \
                >/dev/null 2>&1 || true
        else
            echo "disposable scheduler fixture identity could not be proven during cleanup" >&2
        fi
    fi
    if [[ -n "${scheduler_fixture_pid}" ]]; then
        wait "${scheduler_fixture_pid}" 2>/dev/null || true
    fi
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_tmp}"
}
trap cleanup EXIT

createdb "${test_db}"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY,
    status text NOT NULL DEFAULT 'pending',
    phase text NOT NULL DEFAULT 'train',
    worker_pid integer,
    c_next_threshold double precision NOT NULL DEFAULT 0.001,
    core_lr_mult double precision NOT NULL DEFAULT 1,
    head_lr_mult double precision NOT NULL DEFAULT 1,
    current_epoch integer,
    checkpoint_interval integer NOT NULL DEFAULT 20,
    target_epochs integer NOT NULL DEFAULT 100,
    train_start date NOT NULL DEFAULT '2020-01-01',
    train_end date NOT NULL DEFAULT '2020-02-01',
    infer_start date,
    infer_end date,
    stopped_at_checkpoint_epoch integer,
    stopped_at_checkpoint_model_id bigint,
    last_model_id bigint,
    resume_model_id bigint,
    train_log_path text,
    infer_log_path text,
    analysis_log_path text,
    symbol text NOT NULL DEFAULT 'TEST',
    prediction_horizon integer NOT NULL DEFAULT 1,
    exit_code integer,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    worker_started_at timestamptz,
    error_message text,
    current_operation text,
    stop_after_checkpoint_epoch integer
);
CREATE TABLE experiment_analysis_result (
    experiment_id bigint
);
CREATE TABLE model (
    model_id bigserial PRIMARY KEY,
    experiment_id bigint,
    comment text
);
CREATE TABLE matrix (
    model_id bigint,
    param_name text,
    row_idx integer,
    col_idx integer,
    value double precision
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigserial PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    parent_experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    checkpoint_epoch integer NOT NULL,
    checkpoint_model_id bigint NOT NULL REFERENCES model(model_id),
    symbol text,
    prediction_horizon integer,
    status text NOT NULL DEFAULT 'pending',
    phase text NOT NULL DEFAULT 'infer',
    worker_pid integer,
    infer_log_path text,
    analysis_log_path text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    infer_started_at timestamptz,
    completed_at timestamptz,
    error_message text,
    updated_at timestamptz NOT NULL DEFAULT now(),
    UNIQUE(parent_experiment_id, checkpoint_model_id, checkpoint_epoch)
);
SQL
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/046_global_experiment_control.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment (
    experiment_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES (
    990001,'running','train',990001,990001,'/tmp/LSTM_Release',
    '/tmp/LSTM_Release --train --scheduler-experiment-id=990001',
    '1700000000:1','paused'
),(
    990002,'running','train',990002,990002,'/tmp/LSTM_Release',
    '/tmp/LSTM_Release --train --scheduler-experiment-id=990002',
    '1700000000:2','running'
),(
    990003,'running','train',NULL,NULL,'/tmp/LSTM_Release',
    '/tmp/LSTM_Release --train --scheduler-experiment-id=990003',
    '1700000000:3','running'
),(
    990004,'running','train',990004,990004,'/tmp/replacement',
    '/tmp/replacement --train --scheduler-experiment-id=990004',
    '1700000000:4','running'
);
INSERT INTO model(model_id,experiment_id,comment) VALUES (
    999002,990002,'selective resume migration checkpoint fixture'
);
INSERT INTO experiment_checkpoint_eval (
    checkpoint_eval_id,experiment_id,parent_experiment_id,checkpoint_epoch,
    checkpoint_model_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES (
    999002,990002,990002,20,999002,'running','infer',999002,999002,
    '/tmp/LSTM_Release',
    '/tmp/LSTM_Release --infer --scheduler-experiment-id=990002 --scheduler-checkpoint-eval-id=999002',
    '1700000000:22','paused'
);
WITH request AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count,missing_count,rejected_count,failed_count
    ) VALUES (
        'pause_all','selective-resume-migration-fixture','partial',now(),
        'running','paused',5,2,1,1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,checkpoint_eval_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'experiment:990001',990001,NULL,'experiment','train',
       'running',990001,990001,'1700000000:1','validated','17',
       'signaled','completed','validated'
FROM request
UNION ALL
SELECT request_id,'checkpoint_eval:999002',990002,999002,
       'checkpoint_infer','checkpoint_infer','running',999002,999002,
       '1700000000:22','validated','17','signaled','completed','validated'
FROM request
UNION ALL
SELECT request_id,'experiment:990002',990002,NULL,'experiment','train',
       'running',990002,990002,'1700000000:2','validated',NULL,
       'already_requested_state','completed','already stopped'
FROM request
UNION ALL
SELECT request_id,'experiment:990003',990003,NULL,'experiment','train',
       'running',NULL,NULL,'1700000000:3','process_missing',NULL,
       'process_missing','completed','missing'
FROM request
UNION ALL
SELECT request_id,'experiment:990004',990004,NULL,'experiment','train',
       'running',990004,990004,'1700000000:4','identity_validation_failed',
       NULL,'identity_validation_failed','failed','replacement process'
FROM request;
WITH active AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,application_owner,
        application_lease_until,status,previous_global_state,
        resulting_global_state,target_count
    ) VALUES (
        'pause_all','selective-resume-migration-active',
        'selective-resume-migration-active',now()+interval '5 minutes',
        'applying','paused','paused',0
    ) RETURNING request_id
)
UPDATE experiment_global_control SET desired_state='paused',
    active_request_id=active.request_id
FROM active WHERE singleton;
SQL
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/049_global_pause_selective_resume.sql"
test "$(
    psql -Atq -d "${test_db}" -c \
        "SELECT c.current_pause_request_id::text||':'||
                e.worker_global_pause_request_id::text||':'||
                o.worker_executable||':'||o.worker_command_line
         FROM experiment_global_control c
         JOIN experiment e ON e.experiment_id=990001
         JOIN experiment_admin_worker_outcome o
           ON o.experiment_id=e.experiment_id
         WHERE c.singleton"
)" = "$(
    psql -Atq -d "${test_db}" -c \
        "SELECT r.request_id::text||':'||r.request_id::text||
                ':/tmp/LSTM_Release:/tmp/LSTM_Release --train --scheduler-experiment-id=990001'
         FROM experiment_admin_request r
         WHERE r.invocation_identity='selective-resume-migration-fixture'"
)"
test "$(
    psql -Atq -d "${test_db}" -c \
        "SELECT ce.worker_global_pause_request_id::text||':'||
                o.worker_executable||':'||o.worker_command_line
         FROM experiment_checkpoint_eval ce
         JOIN experiment_admin_worker_outcome o
           ON o.checkpoint_eval_id=ce.checkpoint_eval_id
         WHERE ce.checkpoint_eval_id=999002"
)" = "$(
    psql -Atq -d "${test_db}" -c \
        "SELECT r.request_id::text||
                ':/tmp/LSTM_Release:/tmp/LSTM_Release --infer --scheduler-experiment-id=990002 --scheduler-checkpoint-eval-id=999002'
         FROM experiment_admin_request r
         WHERE r.invocation_identity='selective-resume-migration-fixture'"
)"
test "$(psql -Atq -d "${test_db}" -c \
    "SELECT string_agg(experiment_id::text||':'||
        (worker_global_pause_request_id IS NULL)::text,',' ORDER BY experiment_id)
     FROM experiment WHERE experiment_id BETWEEN 990001 AND 990004")" = \
    "990001:false,990002:true,990003:true,990004:true"
test "$(psql -Atq -d "${test_db}" -c \
    "SELECT r.invocation_identity FROM experiment_global_control c
     JOIN experiment_admin_request r ON r.request_id=c.active_request_id
     WHERE c.singleton")" = selective-resume-migration-active
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE experiment_global_control SET desired_state='running',
    active_request_id=NULL,current_pause_request_id=NULL WHERE singleton;
DELETE FROM experiment_admin_worker_outcome
WHERE experiment_id BETWEEN 990001 AND 990004;
DELETE FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=999002;
DELETE FROM experiment WHERE experiment_id BETWEEN 990001 AND 990004;
DELETE FROM experiment_admin_request
WHERE invocation_identity IN (
    'selective-resume-migration-fixture',
    'selective-resume-migration-active'
);
SQL

psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/GlobalExperimentControlMigrationTests.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/046_global_experiment_control.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/049_global_pause_selective_resume.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/ExperimentCurrentOperationMigrationTests.sql"
current_operation_failure_output="${test_tmp}/current_operation_migration_failure.out"
if psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/ExperimentCurrentOperationMigrationFailureTests.sql" \
    >"${current_operation_failure_output}" 2>&1; then
    echo "current_operation migration unexpectedly accepted unsupported data" >&2
    exit 1
fi
grep -q "unsupported experiment.current_operation rows" \
    "${current_operation_failure_output}"
test "$(psql -v ON_ERROR_STOP=1 -Atq -d "${test_db}" -c \
    "SELECT to_regnamespace(
        'experiment_current_operation_migration_failure_test') IS NULL")" = "t"
psql --single-transaction -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/050_experiment_current_operation_canonicalization.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='global_control_integration_test',
    cutover_executable_path='/tmp/LSTM_Release',
    cutover_process_evidence='isolated_disposable_database',
    updated_at=clock_timestamp()
WHERE singleton AND cutover_state='pending';
UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='observed',
    last_observed_at=clock_timestamp(),
    reconciliation_result='mock_process_fixture_exact_identity_verified'
WHERE ownership_origin='legacy_unverified'
  AND lifecycle_state='identity_ambiguous'
  AND worker_pid IS NOT NULL
  AND worker_process_group_id IS NOT NULL
  AND worker_process_start_identity IS NOT NULL
  AND canonical_executable_path IS NOT NULL
  AND command_line IS NOT NULL;
SQL
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/GlobalExperimentControlMigrationTests.sql"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -c "GRANT SELECT ON model,matrix,experiment_analysis_result TO pqxx;"

schema_psql() {
    local schema_name="$1"
    shift
    PGOPTIONS="-c search_path=${schema_name},public" \
        psql -v ON_ERROR_STOP=1 -q -d "${test_db}" "$@"
}

schema_scalar() {
    local schema_name="$1"
    local query="$2"
    PGOPTIONS="-c search_path=${schema_name},public" \
        psql -v ON_ERROR_STOP=1 -Atq -d "${test_db}" -c "${query}"
}

create_pre047_schema() {
    local schema_name="$1"
    psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
        "CREATE SCHEMA ${schema_name};"
    schema_psql "${schema_name}" <<'SQL'
CREATE TABLE experiment (
    experiment_id bigserial PRIMARY KEY,
    status text NOT NULL DEFAULT 'pending',
    phase text NOT NULL DEFAULT 'train',
    worker_pid integer,
    c_next_threshold double precision NOT NULL DEFAULT 0.001,
    core_lr_mult double precision NOT NULL DEFAULT 1,
    head_lr_mult double precision NOT NULL DEFAULT 1,
    current_epoch integer,
    checkpoint_interval integer NOT NULL DEFAULT 20,
    target_epochs integer NOT NULL DEFAULT 100,
    train_start date NOT NULL DEFAULT '2020-01-01',
    train_end date NOT NULL DEFAULT '2020-02-01',
    infer_start date,
    infer_end date,
    stopped_at_checkpoint_epoch integer,
    stopped_at_checkpoint_model_id bigint,
    last_model_id bigint,
    resume_model_id bigint,
    train_log_path text,
    infer_log_path text,
    analysis_log_path text,
    symbol text NOT NULL DEFAULT 'TEST',
    prediction_horizon integer NOT NULL DEFAULT 1,
    exit_code integer,
    completed_at timestamptz,
    updated_at timestamptz NOT NULL DEFAULT now(),
    worker_started_at timestamptz,
    error_message text,
    current_operation text,
    stop_after_checkpoint_epoch integer
);
CREATE TABLE experiment_analysis_result (experiment_id bigint);
CREATE TABLE model (
    model_id bigserial PRIMARY KEY,
    experiment_id bigint,
    comment text
);
CREATE TABLE matrix (
    model_id bigint,
    param_name text,
    row_idx integer,
    col_idx integer,
    value double precision
);
CREATE TABLE experiment_checkpoint_eval (
    checkpoint_eval_id bigserial PRIMARY KEY,
    experiment_id bigint NOT NULL REFERENCES experiment(experiment_id),
    parent_experiment_id bigint REFERENCES experiment(experiment_id),
    checkpoint_epoch integer,
    checkpoint_model_id bigint REFERENCES model(model_id),
    symbol text,
    prediction_horizon integer,
    status text NOT NULL DEFAULT 'pending',
    phase text NOT NULL DEFAULT 'infer',
    worker_pid integer,
    infer_log_path text,
    analysis_log_path text,
    created_at timestamptz NOT NULL DEFAULT now(),
    started_at timestamptz,
    infer_started_at timestamptz,
    completed_at timestamptz,
    error_message text,
    updated_at timestamptz NOT NULL DEFAULT now()
);
SQL
    schema_psql "${schema_name}" \
        -f "${repo_root}/Database/migrations/046_global_experiment_control.sql"

    test "$(schema_scalar "${schema_name}" \
        "SELECT count(*) FROM information_schema.columns
         WHERE table_schema=current_schema()
           AND ((table_name='experiment_global_control'
                 AND column_name='current_pause_request_id')
             OR (table_name='experiment_admin_request'
                 AND column_name='target_experiment_id')
             OR (table_name IN (
                    'experiment','experiment_checkpoint_eval')
                 AND column_name='worker_global_pause_request_id')
             OR (table_name='experiment_admin_worker_outcome'
                 AND column_name IN (
                    'worker_executable','worker_command_line',
                    'source_pause_request_id')))")" = 0
    test "$(schema_scalar "${schema_name}" \
        "SELECT to_regclass(
            current_schema()||'.experiment_worker_global_pause_idx')
                IS NULL")" = t
    test "$(schema_scalar "${schema_name}" \
        "SELECT position(
            'resume_experiment' in pg_get_constraintdef(oid))=0
         FROM pg_constraint
         WHERE conrelid='experiment_admin_request'::regclass
           AND conname='experiment_admin_request_action_check'")" = t
}

apply_047_schema() {
    local schema_name="$1"
    schema_psql "${schema_name}" \
        -f "${repo_root}/Database/migrations/049_global_pause_selective_resume.sql"
}

assert_047_schema() {
    local schema_name="$1"
    schema_psql "${schema_name}" \
        -f "${repo_root}/Tests/GlobalExperimentControlMigrationTests.sql"
}

drop_fixture_schema() {
    local schema_name="$1"
    psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
        "DROP SCHEMA ${schema_name} CASCADE;" >/dev/null
}

# Each migration scenario below owns an uncontaminated migration-046 schema.
# No scenario can observe columns, constraints, indexes, or rows from a prior
# application of migration 049.
create_pre047_schema gp_mig_primary
schema_psql gp_mig_primary <<'SQL'
INSERT INTO experiment (
    experiment_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES (
    1,'running','train',101,101,'/fixture/primary',
    '/fixture/primary --train --scheduler-experiment-id=1',
    'primary-start','paused'
);
WITH pause AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count
    ) VALUES (
        'pause_all','pre047-primary','completed',now(),
        'running','paused',1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'experiment:1',1,'experiment','train','running',
       101,101,'primary-start','validated','17','signaled','completed',
       'pre047_primary'
FROM pause;
UPDATE experiment_global_control SET desired_state='paused' WHERE singleton;
SQL
apply_047_schema gp_mig_primary
test "$(schema_scalar gp_mig_primary \
    "SELECT (c.current_pause_request_id=r.request_id
             AND e.worker_global_pause_request_id=r.request_id
             AND o.worker_executable='/fixture/primary'
             AND o.worker_command_line=
                 '/fixture/primary --train --scheduler-experiment-id=1')
     FROM experiment_global_control c
     JOIN experiment_admin_request r
       ON r.invocation_identity='pre047-primary'
     JOIN experiment e ON e.experiment_id=1
     JOIN experiment_admin_worker_outcome o
       ON o.request_id=r.request_id AND o.experiment_id=1
     WHERE c.singleton")" = t
assert_047_schema gp_mig_primary
drop_fixture_schema gp_mig_primary

create_pre047_schema gp_mig_child
schema_psql gp_mig_child <<'SQL'
INSERT INTO experiment (experiment_id,status,phase)
VALUES (2,'running','train');
INSERT INTO experiment_checkpoint_eval (
    checkpoint_eval_id,experiment_id,parent_experiment_id,status,phase,
    worker_pid,worker_process_group_id,worker_executable,worker_command_line,
    worker_process_start_identity,worker_control_state
) VALUES (
    20,2,2,'running','infer',202,202,'/fixture/child',
    '/fixture/child --infer --scheduler-experiment-id=2 --scheduler-checkpoint-eval-id=20',
    'child-start','paused'
);
WITH pause AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count
    ) VALUES (
        'pause_all','pre047-child','completed',now(),
        'running','paused',1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,checkpoint_eval_id,worker_kind,
    phase,lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'checkpoint_eval:20',2,20,'checkpoint_infer',
       'checkpoint_infer','running',202,202,'child-start','validated','17',
       'signaled','completed','pre047_child'
FROM pause;
UPDATE experiment_global_control SET desired_state='paused' WHERE singleton;
SQL
apply_047_schema gp_mig_child
test "$(schema_scalar gp_mig_child \
    "SELECT (c.current_pause_request_id=r.request_id
             AND ce.worker_global_pause_request_id=r.request_id
             AND o.worker_executable='/fixture/child'
             AND o.worker_command_line LIKE
                 '/fixture/child --infer%checkpoint-eval-id=20')
     FROM experiment_global_control c
     JOIN experiment_admin_request r
       ON r.invocation_identity='pre047-child'
     JOIN experiment_checkpoint_eval ce ON ce.checkpoint_eval_id=20
     JOIN experiment_admin_worker_outcome o
       ON o.request_id=r.request_id AND o.checkpoint_eval_id=20
     WHERE c.singleton")" = t
assert_047_schema gp_mig_child
drop_fixture_schema gp_mig_child

create_pre047_schema gp_mig_mixed
schema_psql gp_mig_mixed <<'SQL'
INSERT INTO experiment (
    experiment_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES
    (31,'running','train',301,301,'/fixture/a','/fixture/a --train',
     'a-start','paused'),
    (32,'running','train',302,302,'/fixture/b','/fixture/b --train',
     'b-start','running'),
    (33,'running','train',NULL,NULL,'/fixture/c','/fixture/c --train',
     'c-start','running'),
    (34,'running','train',304,304,'/fixture/replacement',
     '/fixture/replacement --train','replacement-start','running');
WITH pause AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count,already_satisfied_count,missing_count,
        rejected_count,failed_count
    ) VALUES (
        'pause_all','pre047-mixed','partial',now(),'running','paused',
        4,1,1,1,1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'experiment:31',31,'experiment','train','running',
       301,301,'a-start','validated','17','signaled','completed',
       'signaled'
FROM pause
UNION ALL
SELECT request_id,'experiment:32',32,'experiment','train','running',
       302,302,'b-start','validated',NULL,'already_requested_state',
       'completed','already_satisfied'
FROM pause
UNION ALL
SELECT request_id,'experiment:33',33,'experiment','train','running',
       NULL,NULL,'c-start','process_missing',NULL,'process_missing',
       'completed','process_missing'
FROM pause
UNION ALL
SELECT request_id,'experiment:34',34,'experiment','train','running',
       304,304,'original-start','identity_validation_failed',NULL,
       'identity_validation_failed','failed','replacement_process'
FROM pause;
UPDATE experiment_global_control SET desired_state='paused' WHERE singleton;
SQL
apply_047_schema gp_mig_mixed
test "$(schema_scalar gp_mig_mixed \
    "SELECT string_agg(
         experiment_id::text||':'||
         (worker_global_pause_request_id IS NOT NULL)::text,
         ',' ORDER BY experiment_id)
     FROM experiment")" = \
    "31:true,32:false,33:false,34:false"
test "$(schema_scalar gp_mig_mixed \
    "SELECT target_count||':'||successful_count||':'||
            already_satisfied_count||':'||missing_count||':'||
            rejected_count||':'||failed_count
     FROM experiment_admin_request
     WHERE invocation_identity='pre047-mixed'")" = "4:1:1:1:1:1"
assert_047_schema gp_mig_mixed
drop_fixture_schema gp_mig_mixed

create_pre047_schema gp_mig_active
schema_psql gp_mig_active <<'SQL'
WITH completed AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state
    ) VALUES (
        'pause_all','pre047-completed-generation','completed',now(),
        'running','paused'
    ) RETURNING request_id
), active AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,application_owner,
        application_lease_until,status,previous_global_state,
        resulting_global_state
    ) VALUES (
        'pause_all','pre047-active','foreign-owner',
        now()+interval '5 minutes','applying','paused','paused'
    ) RETURNING request_id
)
UPDATE experiment_global_control c
SET desired_state='paused',active_request_id=active.request_id
FROM active
WHERE c.singleton;
SQL
apply_047_schema gp_mig_active
test "$(schema_scalar gp_mig_active \
    "SELECT active.invocation_identity||':'||pause.invocation_identity
     FROM experiment_global_control c
     JOIN experiment_admin_request active
       ON active.request_id=c.active_request_id
     JOIN experiment_admin_request pause
       ON pause.request_id=c.current_pause_request_id
     WHERE c.singleton")" = \
    "pre047-active:pre047-completed-generation"
assert_047_schema gp_mig_active
drop_fixture_schema gp_mig_active

create_pre047_schema gp_mig_no_usable
schema_psql gp_mig_no_usable <<'SQL'
WITH active AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,application_owner,
        application_lease_until,status,previous_global_state,
        resulting_global_state
    ) VALUES (
        'pause_all','pre047-no-usable','foreign-owner',
        now()+interval '5 minutes','applying','running','paused'
    ) RETURNING request_id
)
UPDATE experiment_global_control c
SET desired_state='paused',active_request_id=active.request_id
FROM active
WHERE c.singleton;
SQL
apply_047_schema gp_mig_no_usable
test "$(schema_scalar gp_mig_no_usable \
    "SELECT current_pause_request_id IS NULL
     FROM experiment_global_control WHERE singleton")" = t
test "$(schema_scalar gp_mig_no_usable \
    "SELECT r.invocation_identity
     FROM experiment_global_control c
     JOIN experiment_admin_request r ON r.request_id=c.active_request_id
     WHERE c.singleton")" = pre047-no-usable
assert_047_schema gp_mig_no_usable
drop_fixture_schema gp_mig_no_usable

create_pre047_schema gp_mig_idempotent
schema_psql gp_mig_idempotent <<'SQL'
INSERT INTO experiment (
    experiment_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES (
    50,'running','train',501,501,'/fixture/idempotent',
    '/fixture/idempotent --train','idempotent-start','paused'
);
WITH pause AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count
    ) VALUES (
        'pause_all','pre047-idempotent','completed',now(),
        'running','paused',1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'experiment:50',50,'experiment','train','running',
       501,501,'idempotent-start','validated','17','signaled','completed',
       'idempotent'
FROM pause;
UPDATE experiment_global_control SET desired_state='paused' WHERE singleton;
SQL
apply_047_schema gp_mig_idempotent
idempotent_before="$(
    schema_scalar gp_mig_idempotent \
        "SELECT c.current_pause_request_id::text||':'||
                e.worker_global_pause_request_id::text||':'||
                o.worker_executable||':'||o.worker_command_line
         FROM experiment_global_control c
         JOIN experiment e ON e.experiment_id=50
         JOIN experiment_admin_worker_outcome o
           ON o.request_id=c.current_pause_request_id
          AND o.experiment_id=e.experiment_id
         WHERE c.singleton"
)"
apply_047_schema gp_mig_idempotent
test "$(schema_scalar gp_mig_idempotent \
    "SELECT c.current_pause_request_id::text||':'||
            e.worker_global_pause_request_id::text||':'||
            o.worker_executable||':'||o.worker_command_line
     FROM experiment_global_control c
     JOIN experiment e ON e.experiment_id=50
     JOIN experiment_admin_worker_outcome o
       ON o.request_id=c.current_pause_request_id
      AND o.experiment_id=e.experiment_id
     WHERE c.singleton")" = "${idempotent_before}"
assert_047_schema gp_mig_idempotent
drop_fixture_schema gp_mig_idempotent

if [[ -z "${process_test_binary}" ]]; then
    process_test_binary="${test_tmp}/GlobalExperimentControlProcessTests"
    read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
    read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"
    "${CXX:-clang++}" -std=c++20 -O0 -g \
        -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
        -I"${repo_root}/Headers" \
        "${pqxx_compile_flags[@]}" \
        "${repo_root}/Tests/GlobalExperimentControlProcessTests.cpp" \
        "${repo_root}/Sources/GlobalExperimentControl.cpp" \
        "${pqxx_link_flags[@]}" \
        -o "${process_test_binary}"
fi

# End-to-end upgrade path in a final clean migration-046 schema: a real
# disposable stopped worker has only migration-046-era persisted evidence.
# Migration 049 freezes identity and generation evidence, then the production
# selective-resume CLI consumes it.
create_pre047_schema gp_mig_selective
ln -s "${process_test_binary}" "${test_tmp}/Migration_LSTM_Release"
"${test_tmp}/Migration_LSTM_Release" --managed-test-worker --self-session \
    --train --scheduler-experiment-id=990010 --ready-fd=9 \
    9>"${test_tmp}/migration_fixture.ready" &
migration_fixture_pid=$!
migration_identity=""
for _ in {1..100}; do
    migration_identity="$(
        "${process_test_binary}" \
            "--inspect-managed-test-process=${migration_fixture_pid}" \
            2>/dev/null || true
    )"
    if [[ -s "${test_tmp}/migration_fixture.ready" ]] &&
        [[ -n "${migration_identity}" ]]; then
        break
    fi
    sleep 0.02
done
IFS='|' read -r observed_migration_pid migration_fixture_pgid \
    migration_fixture_start_identity migration_fixture_executable \
    migration_fixture_command <<<"${migration_identity}"
test "${observed_migration_pid}" = "${migration_fixture_pid}"
test "${migration_fixture_pgid}" = "${migration_fixture_pid}"
test -n "${migration_fixture_start_identity}"
test -n "${migration_fixture_executable}"
test "${migration_fixture_command}" != ""
# Executable identity is the canonical proc_pidpath/realpath result.  The
# argv[0] command token may be a symlink and is diagnostic only.
migration_fixture_persisted_executable="${migration_fixture_executable}"
test -n "${migration_fixture_persisted_executable}"
kill -STOP -- "-${migration_fixture_pgid}"
for _ in {1..100}; do
    migration_state="$(ps -o state= -p "${migration_fixture_pid}" | tr -d ' ')"
    [[ "${migration_state}" == T* ]] && break
    sleep 0.02
done
[[ "${migration_state}" == T* ]]

schema_psql gp_mig_selective \
    -v fixture_pid="${migration_fixture_pid}" \
    -v fixture_pgid="${migration_fixture_pgid}" \
    -v fixture_start="${migration_fixture_start_identity}" \
    -v fixture_executable="${migration_fixture_persisted_executable}" \
    -v fixture_command="${migration_fixture_command}" <<'SQL'
INSERT INTO experiment (
    experiment_id,status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    worker_control_state
) VALUES (
    990010,'running','train',:fixture_pid,:fixture_pgid,
    :'fixture_executable',:'fixture_command',:'fixture_start','paused'
);
WITH request AS (
    INSERT INTO experiment_admin_request (
        action,invocation_identity,status,completed_at,
        previous_global_state,resulting_global_state,target_count,
        successful_count
    ) VALUES (
        'pause_all','migration-046-selective-e2e','completed',now(),
        'running','paused',1,1
    ) RETURNING request_id
)
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,outcome_status,detail
)
SELECT request_id,'experiment:990010',990010,'experiment','train',
       'running',:fixture_pid,:fixture_pgid,:'fixture_start','validated','17',
       'signaled','completed','migration_046_pause_evidence'
FROM request;
UPDATE experiment_global_control SET desired_state='paused',
    active_request_id=NULL WHERE singleton;
SQL
apply_047_schema gp_mig_selective
assert_047_schema gp_mig_selective
schema_psql gp_mig_selective \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
schema_psql gp_mig_selective \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
schema_psql gp_mig_selective \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"
schema_psql gp_mig_selective <<'SQL'
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='global_control_migration_test',
    cutover_executable_path='/tmp/LSTM_Release',
    cutover_process_evidence=
        'isolated_schema;fixture_process_exactly_inspected',
    updated_at=clock_timestamp()
WHERE singleton AND cutover_state='pending';
UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='stopped',
    last_observed_at=clock_timestamp(),
    reconciliation_result='test_fixture_exact_identity_verified_and_stopped'
WHERE experiment_id=990010
  AND ownership_origin='legacy_unverified'
  AND lifecycle_state='identity_ambiguous';
UPDATE experiment
SET status='paused',resume_requested=false,updated_at=clock_timestamp()
WHERE experiment_id=990010 AND status='running';
SQL
migration_pause_request="$(
    schema_scalar gp_mig_selective \
        "SELECT request_id FROM experiment_admin_request
         WHERE invocation_identity='migration-046-selective-e2e'"
)"
test "$(schema_scalar gp_mig_selective \
    "SELECT c.current_pause_request_id::text||':'||
            e.worker_global_pause_request_id::text||':'||
            o.worker_executable||':'||o.worker_command_line
     FROM experiment_global_control c
     JOIN experiment e ON e.experiment_id=990010
     JOIN experiment_admin_worker_outcome o
       ON o.request_id=c.current_pause_request_id
      AND o.experiment_id=e.experiment_id
     WHERE c.singleton")" = \
    "${migration_pause_request}:${migration_pause_request}:${migration_fixture_persisted_executable}:${migration_fixture_command}"
# LSTM_Release supplies an explicit libpq connection string, so select this
# isolated fixture schema through a database-scoped runtime-role setting.
schema_psql gp_mig_selective -c \
    "GRANT USAGE ON SCHEMA gp_mig_selective TO pqxx;
     GRANT SELECT ON model,matrix,experiment_analysis_result TO pqxx;"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "ALTER ROLE pqxx IN DATABASE ${test_db}
     SET search_path TO gp_mig_selective,public;"
migration_selective_output="$(
    PGOPTIONS="-c search_path=gp_mig_selective,public" \
        LSTM_DB_NAME="${test_db}" "${binary}" \
        --resume-experiment=990010 --yes
)"
grep -q 'new_status=pending.*resume_requested=true.*worker_state=stopped.*signal=none.*result=queued_for_admission' \
    <<<"${migration_selective_output}"
test "$(schema_scalar gp_mig_selective \
    "SELECT status||':'||resume_requested::text FROM experiment
     WHERE experiment_id=990010")" = "pending:true"
migration_replay_output="$(
    PGOPTIONS="-c search_path=gp_mig_selective,public" \
        LSTM_DB_NAME="${test_db}" "${binary}" \
        --resume-experiment=990010 --yes
)"
grep -q 'new_status=pending.*resume_requested=true.*worker_state=stopped.*signal=none.*result=already_satisfied' \
    <<<"${migration_replay_output}"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "ALTER ROLE pqxx IN DATABASE ${test_db} RESET search_path;"
kill -CONT -- "-${migration_fixture_pgid}"
kill -TERM -- "-${migration_fixture_pgid}"
wait "${migration_fixture_pid}"
migration_fixture_pid=""
migration_fixture_pgid=""
migration_fixture_start_identity=""
migration_fixture_executable=""
migration_fixture_persisted_executable=""
drop_fixture_schema gp_mig_selective

"${process_test_binary}" --database-legacy-control-tests \
    "dbname=${test_db}"

# The tests above intentionally exercise legacy cancellation/reconciliation
# behavior on the migration-052 schema. The current executable reads later
# experiment-configuration columns, but replaying their historical migrations
# is inappropriate here: those migrations also rebuild unrelated production
# identity indexes and assume the complete production schema lineage.
#
# Reproduce only the cumulative experiment-row schema contract required by the
# current executable, using the exact historical compatibility defaults. Then
# execute migration 086 itself because its priority/stopped-attempt semantics
# are the behavior under test.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
ALTER TABLE experiment
    ADD COLUMN IF NOT EXISTS donchian20_mode text
        NOT NULL DEFAULT 'enabled',
    ADD COLUMN IF NOT EXISTS feature_warmup_scope text
        NOT NULL DEFAULT 'legacy_cold_boundary',
    ADD COLUMN IF NOT EXISTS donchian_lookback integer
        NOT NULL DEFAULT 20,
    ADD COLUMN IF NOT EXISTS feature_ablation_mask text
        NOT NULL DEFAULT '',
    ADD COLUMN IF NOT EXISTS resume_expand_input_width boolean
        NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS
        operator_forced_final_inference_rerun_requested boolean
        NOT NULL DEFAULT false,
    ADD COLUMN IF NOT EXISTS training_objective_id text
        NOT NULL DEFAULT 'legacy_first_hit_weighted_ce_v1',
    ADD COLUMN IF NOT EXISTS training_objective_version integer
        NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS loss_definition_version integer
        NOT NULL DEFAULT 1,
    ADD COLUMN IF NOT EXISTS training_objective_canonical text
        NOT NULL DEFAULT 'training_objective_configuration_v1;schema_version=1;objective_id=legacy_first_hit_weighted_ce_v1;objective_family=up_neutral_down_first_hit_classification;objective_version=1;loss_definition_version=1;mode=legacy_first_hit_classification;classification_loss=true_class_weighted_softmax_cross_entropy_v1;classification_target=up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1;class_index_order=down_0_neutral_1_up_2;class_weight_semantics=true_class_weight_multiplies_loss_and_all_logit_components_v1;class_weight_down=1;class_weight_neutral=1;class_weight_up=1;softmax_loss_probability_floor=1e-12;classification_logit_gradient_scale=0.1;shared_core_classification_gradient_scale=4;internal_loss_normalization=weighted_loss_sum_divided_by_true_class_weight_sum_v1;calculate_batch_return_normalization=weighted_loss_sum_divided_by_example_count_v1;gradient_normalization=all_calculate_batch_gradients_divided_by_true_class_weight_sum_v1;batch_window_boundary=overlapping_windows_do_not_cross_outer_tensor_batch_v1;optimizer_family=sgd;optimizer_update=parameter_minus_learning_rate_times_gradient_v1;learning_rate_contract=base_rate_and_parameter_group_multipliers_persisted_in_training_config_v1;gradient_clipping_mode=componentwise_after_normalization_before_update;gradient_clip_threshold=10;nonfinite_gradient_policy=skip_parameter_update_v1;weight_decay=none;gradient_accumulation_precision=core_gradient_accumulation_double_head_gradient_accumulation_float_loss_accumulation_double_v1;auxiliary_loss_mode=disabled;auxiliary_loss_coefficient=0;regression_target_definition=NULL;regression_normalization_identity=NULL;robust_loss_definition=NULL;robust_loss_delta=NULL;target_clipping_definition=none;shared_gradient_combination=classification_only_v1;',
    ADD COLUMN IF NOT EXISTS training_objective_hash text
        NOT NULL DEFAULT 'fnv1a64:65818f2e1fa1a324',
    ADD COLUMN IF NOT EXISTS auxiliary_loss_mode text
        NOT NULL DEFAULT 'disabled',
    ADD COLUMN IF NOT EXISTS auxiliary_loss_coefficient double precision
        NOT NULL DEFAULT 0.0,
    ADD COLUMN IF NOT EXISTS regression_target_definition text,
    ADD COLUMN IF NOT EXISTS regression_normalization_identity text,
    ADD COLUMN IF NOT EXISTS robust_loss_definition text,
    ADD COLUMN IF NOT EXISTS robust_loss_delta double precision,
    ADD COLUMN IF NOT EXISTS target_clipping_definition text
        NOT NULL DEFAULT 'none',
    ADD COLUMN IF NOT EXISTS objective_normalization_identity text
        NOT NULL DEFAULT
        'weighted_loss_sum_by_weight_sum_gradients__calculate_batch_return_by_example_count_v1';

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_donchian20_mode_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_donchian20_mode_check
        CHECK (donchian20_mode IN ('enabled', 'zero_ablation'));

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_feature_warmup_scope_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_feature_warmup_scope_check
        CHECK (feature_warmup_scope IN (
            'legacy_cold_boundary',
            'full_history_warmup'
        ));

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_donchian_lookback_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_donchian_lookback_check
        CHECK (donchian_lookback BETWEEN 1 AND 10000);

ALTER TABLE experiment
    DROP CONSTRAINT IF EXISTS experiment_resume_expand_input_width_source_check;
ALTER TABLE experiment
    ADD CONSTRAINT experiment_resume_expand_input_width_source_check
        CHECK (
            NOT resume_expand_input_width
            OR resume_model_id IS NOT NULL
        );
SQL

psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"

"${process_test_binary}" --database-priority-control-tests \
    "dbname=${test_db}"

run_control() {
    LSTM_DB_NAME="${test_db}" "${binary}" "$@"
}
scalar() {
    psql -Atq -d "${test_db}" -c "$1"
}

run_control --pause-all-experiments --yes |
    grep -q 'status=completed.*target_count=0'
run_control --pause-all-experiments --yes |
    grep -q 'status=completed.*target_count=0'
test "$(scalar "SELECT desired_state FROM experiment_global_control")" = paused

# Restart recovery is exercised with two distinct scheduler processes. Both
# must reload the durable paused gate and exit without launching the pending
# train row, independent of any unrelated scheduler state.
psql -q -d "${test_db}" -c \
    "INSERT INTO experiment(experiment_id,status,phase)
     VALUES (900001,'pending','train');"
for restart_attempt in 1 2; do
    run_control --schedule-experiments --scheduler-once --dry-run \
        >"${test_tmp}/scheduler_restart_${restart_attempt}.out"
    grep -q 'SCHEDULER_START.*global_desired_state=paused' \
        "${test_tmp}/scheduler_restart_${restart_attempt}.out"
    grep -q 'SCHEDULER_STOP,exit_code=0' \
        "${test_tmp}/scheduler_restart_${restart_attempt}.out"
    test "$(scalar "SELECT status FROM experiment
        WHERE experiment_id=900001")" = pending
done
psql -q -d "${test_db}" -c \
    "DELETE FROM experiment WHERE experiment_id=900001;"

run_control --resume-all-experiments --yes |
    grep -q 'status=completed.*target_count=0'
run_control --resume-all-experiments --yes |
    grep -q 'status=completed.*target_count=0'
test "$(scalar "SELECT desired_state FROM experiment_global_control")" = running

# Priority-aware pause/resume-all rejects any active administrative request;
# lease expiry does not transfer ownership into this control path.
lease_request_id="$(
    scalar "INSERT INTO experiment_admin_request (
        action,invocation_identity,application_owner,application_lease_until,
        status,previous_global_state,resulting_global_state
    ) VALUES (
        'pause_all','lease-seed','lease-owner',now()+interval '5 minutes',
        'applying','running','paused'
    ) RETURNING request_id"
)"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_global_control SET desired_state='paused',
     active_request_id=${lease_request_id} WHERE singleton;"
set +e
run_control --pause-all-experiments --yes \
    >"${test_tmp}/lease_rejected.out" 2>&1
lease_rejected=$?
set -e
test "${lease_rejected}" -ne 0
grep -q "conflicting_administrative_request_active,request_id=${lease_request_id}" \
    "${test_tmp}/lease_rejected.out"
test "$(scalar "SELECT application_owner FROM experiment_admin_request
    WHERE request_id=${lease_request_id}")" = lease-owner

psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_admin_request
     SET application_lease_until=now()-interval '1 second'
     WHERE request_id=${lease_request_id};"
set +e
run_control --pause-all-experiments --yes \
    >"${test_tmp}/expired_lease_rejected.out" 2>&1
expired_lease_rejected=$?
set -e
test "${expired_lease_rejected}" -ne 0
grep -q "conflicting_administrative_request_active,request_id=${lease_request_id}" \
    "${test_tmp}/expired_lease_rejected.out"
test "$(scalar "SELECT application_owner FROM experiment_admin_request
    WHERE request_id=${lease_request_id}")" = lease-owner
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_global_control SET active_request_id=NULL
     WHERE singleton AND active_request_id=${lease_request_id};
     DELETE FROM experiment_admin_request
     WHERE request_id=${lease_request_id};"

run_control --resume-all-experiments --yes |
    grep -q 'status=completed.*target_count=0'
test "$(scalar "SELECT desired_state FROM experiment_global_control")" = running

# A selective individual resume may queue one member of an active global-pause
# generation, but the durable global gate still prevents scheduler admission.
# Resume-all later retires the generation, releases the remaining owned member,
# and leaves an independently paused experiment untouched. The selectively
# resumed member retains resume priority and is the first dry-run train
# admission once the global gate returns to running.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO experiment(
         experiment_id,status,phase,scheduler_priority,resume_requested,updated_at
     ) VALUES
         (909010,'pending','train','normal',false,'2026-01-01 00:00:10+00'),
         (909011,'pending','train','normal',false,'2026-01-01 00:00:11+00'),
         (909012,'paused','train','normal',false,'2026-01-01 00:00:00+00');"

run_control --pause-all-experiments --yes \
    >"${test_tmp}/selective_global_pause.out"
grep -q 'status=completed.*target_count=2' \
    "${test_tmp}/selective_global_pause.out"
selective_pause_request="$(
    scalar "SELECT current_pause_request_id FROM experiment_global_control"
)"
test -n "${selective_pause_request}"
test "$(scalar "SELECT desired_state FROM experiment_global_control")" = paused
test "$(scalar "SELECT string_agg(
    experiment_id||':'||status||':'||resume_requested::text||':'||
    COALESCE(worker_global_pause_request_id::text,'NULL'),
    ',' ORDER BY experiment_id) FROM experiment
    WHERE experiment_id BETWEEN 909010 AND 909012")" = \
    "909010:paused:false:${selective_pause_request},909011:paused:false:${selective_pause_request},909012:paused:false:NULL"

run_control --resume-experiment=909010 --yes \
    >"${test_tmp}/selective_global_individual_resume.out"
grep -q 'new_status=pending.*resume_requested=true.*result=queued_for_admission' \
    "${test_tmp}/selective_global_individual_resume.out"
test "$(scalar "SELECT desired_state FROM experiment_global_control")" = paused
test "$(scalar "SELECT status||':'||resume_requested::text||':'||
    worker_global_pause_request_id::text FROM experiment
    WHERE experiment_id=909010")" = \
    "pending:true:${selective_pause_request}"

run_control --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    >"${test_tmp}/selective_global_blocked_scheduler.out"
grep -q 'SCHEDULER_START.*global_desired_state=paused' \
    "${test_tmp}/selective_global_blocked_scheduler.out"
! grep -q 'EXPERIMENT_CHILD_COMMAND.*experiment_id=909010' \
    "${test_tmp}/selective_global_blocked_scheduler.out"
test "$(scalar "SELECT status||':'||resume_requested::text FROM experiment
    WHERE experiment_id=909010")" = pending:true

run_control --resume-all-experiments --yes \
    >"${test_tmp}/selective_global_resume_all.out"
grep -q 'status=completed.*target_count=1.*resume_requested_count=1' \
    "${test_tmp}/selective_global_resume_all.out"
test "$(scalar "SELECT desired_state||':'||
    COALESCE(current_pause_request_id::text,'NULL')
    FROM experiment_global_control")" = running:NULL
test "$(scalar "SELECT string_agg(
    experiment_id||':'||status||':'||resume_requested::text||':'||
    COALESCE(worker_global_pause_request_id::text,'NULL'),
    ',' ORDER BY experiment_id) FROM experiment
    WHERE experiment_id BETWEEN 909010 AND 909012")" = \
    '909010:pending:true:NULL,909011:pending:true:NULL,909012:paused:false:NULL'

# Add older high-priority ordinary pending work. resume_requested must still
# dominate scheduler_priority and updated_at for resumed admission ordering.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO experiment(
         experiment_id,status,phase,scheduler_priority,resume_requested,updated_at
     ) VALUES
         (909013,'pending','train','high',false,'2025-01-01 00:00:00+00');"

run_control --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    >"${test_tmp}/selective_global_released_scheduler.out"
grep -q 'SCHEDULER_START.*global_desired_state=running' \
    "${test_tmp}/selective_global_released_scheduler.out"
grep -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=909010,phase=train,dry_run=1' \
    "${test_tmp}/selective_global_released_scheduler.out"
! grep -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=909013' \
    "${test_tmp}/selective_global_released_scheduler.out"
test "$(scalar "SELECT status||':'||resume_requested::text FROM experiment
    WHERE experiment_id=909012")" = paused:false

psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
DELETE FROM experiment_admin_worker_outcome
WHERE experiment_id BETWEEN 909010 AND 909013;

DELETE FROM experiment_scheduler_worker_attempt
WHERE experiment_id BETWEEN 909010 AND 909013;

DELETE FROM experiment_admin_request
WHERE target_experiment_id BETWEEN 909010 AND 909013;

DELETE FROM experiment
WHERE experiment_id BETWEEN 909010 AND 909013;
SQL

# Selective resume queues paused lifecycle work for scheduler admission, while
# an ordinary running row is already satisfied without a process signal.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "INSERT INTO experiment(experiment_id,status,phase)
     VALUES (910000,'paused','train'),(910001,'running','train');"
run_control --resume-experiment=910000 --yes |
    grep -q 'new_status=pending.*resume_requested=true.*worker_state=none.*signal=none.*result=queued_for_admission'
test "$(scalar "SELECT status||':'||resume_requested::text
    FROM experiment WHERE experiment_id=910000")" = pending:true
run_control --resume-experiment=910001 --yes |
    grep -q 'new_status=running.*resume_requested=false.*worker_state=none.*signal=none.*result=already_satisfied'
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM experiment WHERE experiment_id IN (910000,910001);"

request_count="$(scalar "SELECT count(*) FROM experiment_admin_request")"
run_control --cancel-all-experiments --immediate \
    --infer-before-cancel --dry-run |
    grep -q 'targets=0'
test "${request_count}" = \
    "$(scalar "SELECT count(*) FROM experiment_admin_request")"

psql -q -d "${test_db}" -c \
    "INSERT INTO experiment(status,phase) VALUES ('pending','train');"
run_control --cancel-all-experiments --immediate --yes |
    grep -q 'status=completed.*target_count=1'
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=1")" = cancelled

psql -q -d "${test_db}" <<'SQL'
INSERT INTO experiment (
    status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    infer_start,infer_end,
    current_operation,worker_started_at
) VALUES (
    'running','train',999999,999999,'/missing/LSTM_Release',
    'LSTM_Release --train --scheduler-experiment-id=2',
    'missing-process-start-2',
    '2020-01-01','2020-02-01','train',now()
);
BEGIN;
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation','52',true);
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,worker_pid,
        worker_process_group_id,worker_process_start_identity,
        canonical_executable_path,command_line,command_identity,
        reserved_at,spawned_at,registered_at
    ) VALUES (
        'test:missing-process:2',2,'experiment','train','train',
        'legacy_unverified','observed',999999,999999,
        'missing-process-start-2','/missing/LSTM_Release',
        'LSTM_Release --train --scheduler-experiment-id=2',
        'experiment:2:train',now(),now(),now()
    ) RETURNING worker_attempt_id
)
UPDATE experiment SET active_scheduler_worker_attempt_id=
    (SELECT worker_attempt_id FROM attempt)
WHERE experiment_id=2;
COMMIT;
SQL
set +e
no_checkpoint="$(
    run_control --cancel-all-experiments --immediate \
        --infer-before-cancel --yes
)"
no_checkpoint_exit=$?
set -e
test "${no_checkpoint_exit}" -eq 1
grep -q 'status=partial' <<<"${no_checkpoint}"
grep -q 'inference_action=no_checkpoint' <<<"${no_checkpoint}"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=2")" = cancelled

psql -q -d "${test_db}" <<'SQL'
INSERT INTO experiment (
    status,phase,worker_pid,worker_process_group_id,
    worker_executable,worker_command_line,worker_process_start_identity,
    current_epoch,
    infer_start,infer_end,current_operation,worker_started_at
) VALUES (
    'running','train',999998,999998,'/missing/LSTM_Release',
    'LSTM_Release --train --scheduler-experiment-id=3',
    'missing-process-start-3',21,
    '2020-01-01','2020-02-01','train',now()
);
BEGIN;
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation','52',true);
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,worker_pid,
        worker_process_group_id,worker_process_start_identity,
        canonical_executable_path,command_line,command_identity,
        reserved_at,spawned_at,registered_at
    ) VALUES (
        'test:missing-process:3',3,'experiment','train','train',
        'legacy_unverified','observed',999998,999998,
        'missing-process-start-3','/missing/LSTM_Release',
        'LSTM_Release --train --scheduler-experiment-id=3',
        'experiment:3:train',now(),now(),now()
    ) RETURNING worker_attempt_id
)
UPDATE experiment SET active_scheduler_worker_attempt_id=
    (SELECT worker_attempt_id FROM attempt)
WHERE experiment_id=3;
COMMIT;
INSERT INTO model(experiment_id,comment)
VALUES (3,'periodic training checkpoint');
UPDATE experiment SET last_model_id=currval('model_model_id_seq')
WHERE experiment_id=3;
INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value)
VALUES (
    currval('model_model_id_seq'),'train_config_meta',0,10,20
);
SQL
checkpoint_output="$(
    run_control --cancel-all-experiments --after-next-checkpoint --yes
)"
grep -q 'checkpoint_target=40' <<<"${checkpoint_output}"
grep -q 'status=pending' <<<"${checkpoint_output}"
test "$(scalar \
    "SELECT cancel_after_checkpoint_epoch FROM experiment WHERE experiment_id=3")" = 40
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=3")" = pending

# A second control process reconstructs the same durable checkpoint request.
# This is the crash/restart recovery path: it neither creates a new request nor
# loses the exact checkpoint target.
checkpoint_request_id="$(
    scalar "SELECT active_request_id FROM experiment_global_control
            WHERE singleton"
)"
restart_output="$(
    run_control --cancel-all-experiments --after-next-checkpoint --yes
)"
grep -q "request_id=${checkpoint_request_id}.*status=pending" \
    <<<"${restart_output}"
grep -q 'checkpoint_target=40' <<<"${restart_output}"
test "$(scalar "SELECT count(*) FROM experiment_admin_request
    WHERE request_id=${checkpoint_request_id}")" = 1
test "$(scalar "SELECT active_request_id FROM experiment_global_control
    WHERE singleton")" = "${checkpoint_request_id}"

# Simulate durable completion so the isolated request does not remain active.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE experiment SET status='cancelled',completed_at=now(),
    cancellation_completed_at=now() WHERE experiment_id=3;
UPDATE experiment_admin_worker_outcome SET outcome_status='completed'
WHERE request_id=(SELECT active_request_id FROM experiment_global_control);
UPDATE experiment_admin_request SET status='completed',completed_at=now(),
    application_lease_until=NULL
WHERE request_id=(SELECT active_request_id FROM experiment_global_control);
UPDATE experiment_global_control SET active_request_id=NULL WHERE singleton;
SQL

# A fresh scheduler reconstructs an active, partially applied checkpoint
# cancellation. The fixture is the exact post-replay recovery state: a missing
# worker was requeued from a real durable checkpoint, with its cancellation
# identity and target retained. A disposable matching test worker makes the
# dry-run scheduler take the already-running discovery path, so the authorized
# row is examined without emitting a child command. Ordinary pending work stays
# excluded by the database-authoritative cancellation gate.
restart_cancel_request_id="$(
    scalar "INSERT INTO experiment_admin_request (
        action,cancellation_mode,infer_before_cancel,invocation_identity,
        requester_identity,application_owner,application_lease_until,status,
        previous_global_state,resulting_global_state,
        scheduler_running_observed,target_count,missing_count,result_summary
    ) VALUES (
        'cancel_all','after_next_checkpoint',false,
        'scheduler-restart-seed','integration-fixture-requester',
        'scheduler-restart-replay',NULL,'pending',
        'running','running',false,1,1,
        '{\"target_count\":1,\"successful_count\":0,
          \"already_satisfied_count\":0,\"missing_count\":1,
          \"rejected_count\":0,\"failed_count\":0,\"pending_count\":1}'::jsonb
    ) RETURNING request_id"
)"
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO experiment (
    experiment_id,status,phase,current_epoch
) VALUES (
    800001,'pending','train',0
);
INSERT INTO experiment (
    experiment_id,status,phase,current_epoch,checkpoint_interval,target_epochs,
    worker_pid,worker_process_group_id,worker_executable,worker_command_line,
    worker_process_start_identity,worker_control_state,
    cancellation_request_id,cancel_after_checkpoint_epoch,
    stop_after_checkpoint_epoch,current_operation,error_message,
    worker_started_at
) VALUES (
    800002,'pending','train',10,20,100,NULL,NULL,'LSTM_Release',
    'LSTM_Release --train --scheduler-experiment-id=800002',
    '1700008002:2','running',${restart_cancel_request_id},20,20,
    'train',
    'cancellation_worker_restart_required',now()-interval '1 minute'
);
INSERT INTO experiment (
    experiment_id,status,phase,current_epoch
) VALUES (
    800003,'pending','train',0
);
INSERT INTO model(experiment_id,comment)
VALUES (800002,'periodic training checkpoint');
INSERT INTO matrix(model_id,param_name,row_idx,col_idx,value)
VALUES (currval('model_model_id_seq'),'train_config_meta',0,10,10);
UPDATE experiment SET last_model_id=currval('model_model_id_seq')
WHERE experiment_id=800002;
INSERT INTO experiment_admin_worker_outcome (
    request_id,worker_identity,experiment_id,worker_kind,phase,
    lifecycle_status,worker_pid,worker_process_group_id,
    worker_process_start_identity,identity_result,requested_signal,
    signal_result,cancellation_checkpoint_epoch,inference_action,
    outcome_status,detail
) VALUES (
    ${restart_cancel_request_id},'experiment:800002',800002,'experiment',
    'train','running',910002,910002,'1700008002:2','process_missing',NULL,
    'process_missing',20,'none','pending_checkpoint',
    'missing_worker_requeued_from_durable_checkpoint'
);
UPDATE experiment_global_control SET desired_state='running',
    active_request_id=${restart_cancel_request_id},
    revision=revision+1,updated_at=now() WHERE singleton;
SQL
test "$(scalar "SELECT action||':'||cancellation_mode||':'||
    infer_before_cancel::text||':'||requester_identity||':'||
    application_owner||':'||status||':'||previous_global_state||':'||
    resulting_global_state||':'||scheduler_running_observed::text||':'||
    target_count::text||':'||successful_count::text||':'||
    already_satisfied_count::text||':'||missing_count::text||':'||
    rejected_count::text||':'||failed_count::text||':'||
    (result_summary->>'pending_count')||':'||
    (result_summary->>'missing_count')||':'||
    (application_lease_until IS NULL)::text
    FROM experiment_admin_request
    WHERE request_id=${restart_cancel_request_id}")" = \
    "cancel_all:after_next_checkpoint:false:integration-fixture-requester:scheduler-restart-replay:pending:running:running:false:1:0:0:1:0:0:1:1:true"
test "$(scalar "SELECT
    count(*) FILTER (WHERE outcome_status IN
        ('planned','pending_checkpoint','awaiting_inference'))::text||':'||
    count(*) FILTER (WHERE outcome_status='completed')::text||':'||
    count(*) FILTER
        (WHERE signal_result='already_requested_state')::text||':'||
    count(*) FILTER (WHERE identity_result='process_missing')::text
    FROM experiment_admin_worker_outcome
    WHERE request_id=${restart_cancel_request_id}")" = "1:0:0:1"
test "$(scalar "SELECT status||':'||phase||':'||
    (worker_pid IS NULL)::text||':'||
    (worker_process_group_id IS NULL)::text||':'||
    worker_executable||':'||worker_command_line||':'||
    worker_process_start_identity||':'||worker_control_state||':'||
    (worker_started_at IS NOT NULL)::text||':'||
    last_model_id::text||':'||(resume_model_id IS NULL)::text||':'||
    current_operation||':'||error_message||':'||
    cancellation_request_id::text||':'||
    cancel_after_checkpoint_epoch::text||':'||
    stop_after_checkpoint_epoch::text
    FROM experiment WHERE experiment_id=800002")" = \
    "pending:train:true:true:LSTM_Release:LSTM_Release --train --scheduler-experiment-id=800002:1700008002:2:running:true:$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=800002"):true:train:cancellation_worker_restart_required:${restart_cancel_request_id}:20:20"
restart_model_id="$(
    scalar "WITH cfg AS (
        SELECT model_id,max(value) FILTER (WHERE col_idx=10) completed_epochs
        FROM matrix WHERE param_name='train_config_meta' AND row_idx=0
        GROUP BY model_id
    ) SELECT m.model_id FROM model m
    LEFT JOIN cfg ON cfg.model_id=m.model_id
    WHERE m.experiment_id=800002
    ORDER BY cfg.completed_epochs DESC NULLS LAST,m.model_id DESC LIMIT 1"
)"
test "${restart_model_id}" = \
    "$(scalar "SELECT last_model_id FROM experiment WHERE experiment_id=800002")"
test "$(scalar "SELECT comment||':'||
    (SELECT round(value)::int FROM matrix
     WHERE model_id=m.model_id AND param_name='train_config_meta'
       AND row_idx=0 AND col_idx=10)
    FROM model m WHERE model_id=${restart_model_id}")" = \
    "periodic training checkpoint:10"
test "$(scalar "SELECT bool_and(cancellation_request_id IS NULL)
    FROM experiment WHERE experiment_id IN (800001,800003)")" = t
test "$(scalar "SELECT worker_pid::text||':'||
    worker_process_group_id::text||':'||worker_process_start_identity||':'||
    phase||':'||lifecycle_status||':'||
    COALESCE(requested_signal,'NULL')||':'||identity_result||':'||
    signal_result||':'||cancellation_checkpoint_epoch::text||':'||
    inference_action||':'||outcome_status||':'||detail
    FROM experiment_admin_worker_outcome
    WHERE request_id=${restart_cancel_request_id}")" = \
    "910002:910002:1700008002:2:train:running:NULL:process_missing:process_missing:20:none:pending_checkpoint:missing_worker_requeued_from_durable_checkpoint"
restart_request_identity="$(
    scalar "SELECT invocation_identity||':'||requester_identity||':'||
        application_owner||':'||action||':'||cancellation_mode||':'||
        infer_before_cancel::text
        FROM experiment_admin_request
        WHERE request_id=${restart_cancel_request_id}"
)"
restart_outcome_identity="$(
    scalar "SELECT worker_identity||':'||worker_pid::text||':'||
        worker_process_group_id::text||':'||worker_process_start_identity||':'||
        phase||':'||lifecycle_status||':'||
        COALESCE(requested_signal,'NULL')||':'||identity_result||':'||
        signal_result||':'||cancellation_checkpoint_epoch::text||':'||
        inference_action||':'||outcome_status||':'||detail
        FROM experiment_admin_worker_outcome
        WHERE request_id=${restart_cancel_request_id}"
)"

ln -s "${process_test_binary}" "${test_tmp}/LSTM_Release"
"${test_tmp}/LSTM_Release" --managed-test-worker --self-session --train \
    --scheduler-experiment-id=800002 --ready-fd=9 \
    9>"${test_tmp}/scheduler_fixture.ready" &
scheduler_fixture_pid=$!
scheduler_fixture_identity_captured=false
for _ in {1..100}; do
    if kill -0 "${scheduler_fixture_pid}" >/dev/null 2>&1 &&
        capture_fixture_identity; then
        scheduler_fixture_identity_captured=true
    fi
    if [[ "${scheduler_fixture_identity_captured}" = true ]] &&
        [[ -s "${test_tmp}/scheduler_fixture.ready" ]]; then
        break
    fi
    sleep 0.02
done
test "${scheduler_fixture_identity_captured}" = true
test -s "${test_tmp}/scheduler_fixture.ready"
kill -0 "${scheduler_fixture_pid}"

# The restart fixture represents a pre-existing scheduler worker.  Under the
# durable accounting contract it must have an exact active attempt; local
# process discovery is intentionally not an ownership source.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v fixture_pid="${scheduler_fixture_pid}" \
    -v fixture_pgid="${scheduler_fixture_pgid}" \
    -v fixture_start="${scheduler_fixture_start_identity}" \
    -v fixture_executable="${scheduler_fixture_executable}" \
    -v fixture_command="${test_tmp}/LSTM_Release --managed-test-worker --self-session --train --scheduler-experiment-id=800002 --ready-fd=9" <<'SQL'
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt (
        launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,worker_pid,
        worker_process_group_id,worker_process_start_identity,
        canonical_executable_path,command_line,command_identity,
        reserved_at,spawned_at,registered_at
    ) VALUES (
        'legacy:global-control-restart:800002',800002,'experiment','train',
        'train','legacy_unverified','identity_ambiguous',:fixture_pid,
        :fixture_pgid,:'fixture_start',:'fixture_executable',
        :'fixture_command','experiment:800002:train',
        clock_timestamp(),clock_timestamp(),clock_timestamp()
    ) RETURNING worker_attempt_id
)
UPDATE experiment SET active_scheduler_worker_attempt_id=
    (SELECT worker_attempt_id FROM attempt)
WHERE experiment_id=800002;
SQL

restart_request_count="$(
    scalar "SELECT count(*) FROM experiment_admin_request"
)"
restart_outcome_count="$(
    scalar "SELECT count(*) FROM experiment_admin_worker_outcome
            WHERE request_id=${restart_cancel_request_id}"
)"
run_control --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=1 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_tmp}/scheduler_restart_logs" \
    >"${test_tmp}/scheduler_active_cancellation_restart.out"
grep -q 'SCHEDULER_START.*global_desired_state=running' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
grep -q 'SCHEDULER_QUEUE_PHASE,phase=train,examined=1,skipped=1,launched=0,free_slots=0' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
! grep -q 'experiment_id=800001' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
! grep -q 'experiment_id=800003' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
! grep -q 'EXPERIMENT_CHILD_COMMAND' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
! grep -q 'GLOBAL_EXPERIMENT_CONTROL_WORKER' \
    "${test_tmp}/scheduler_active_cancellation_restart.out"
kill -0 "${scheduler_fixture_pid}"
test "$(scalar "SELECT count(*) FROM experiment_admin_request")" = \
    "${restart_request_count}"
test "$(scalar "SELECT count(*) FROM experiment_admin_worker_outcome
    WHERE request_id=${restart_cancel_request_id}")" = \
    "${restart_outcome_count}"
test "$(scalar "SELECT worker_identity||':'||outcome_status||':'||
    COALESCE(requested_signal,'NULL')
    FROM experiment_admin_worker_outcome
    WHERE request_id=${restart_cancel_request_id}")" = \
    "experiment:800002:pending_checkpoint:NULL"
test "$(scalar "SELECT invocation_identity||':'||requester_identity||':'||
    application_owner||':'||action||':'||cancellation_mode||':'||
    infer_before_cancel::text
    FROM experiment_admin_request
    WHERE request_id=${restart_cancel_request_id}")" = \
    "${restart_request_identity}"
test "$(scalar "SELECT worker_identity||':'||worker_pid::text||':'||
    worker_process_group_id::text||':'||worker_process_start_identity||':'||
    phase||':'||lifecycle_status||':'||
    COALESCE(requested_signal,'NULL')||':'||identity_result||':'||
    signal_result||':'||cancellation_checkpoint_epoch::text||':'||
    inference_action||':'||outcome_status||':'||detail
    FROM experiment_admin_worker_outcome
    WHERE request_id=${restart_cancel_request_id}")" = \
    "${restart_outcome_identity}"
test "$(scalar "SELECT active_request_id FROM experiment_global_control
    WHERE singleton")" = "${restart_cancel_request_id}"
test "$(scalar "SELECT string_agg(experiment_id::text||':'||status,','
    ORDER BY experiment_id) FROM experiment
    WHERE experiment_id IN (800001,800002,800003)")" = \
    "800001:pending,800002:pending,800003:pending"
test "$(scalar "SELECT last_model_id FROM experiment
    WHERE experiment_id=800002")" = "${restart_model_id}"
if fixture_identity_matches; then
    kill -TERM -- "-${scheduler_fixture_pgid}"
else
    echo "disposable scheduler fixture identity could not be revalidated" >&2
    exit 1
fi
wait "${scheduler_fixture_pid}"
scheduler_fixture_pid=""
scheduler_fixture_pgid=""
scheduler_fixture_start_identity=""
scheduler_fixture_executable=""
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE experiment_global_control SET active_request_id=NULL WHERE singleton;
DELETE FROM experiment_admin_worker_outcome
WHERE request_id=${restart_cancel_request_id};
DELETE FROM matrix WHERE model_id=${restart_model_id};
UPDATE experiment SET active_scheduler_worker_attempt_id=NULL
WHERE experiment_id=800002;
DELETE FROM experiment_scheduler_worker_attempt
WHERE experiment_id=800002;
DELETE FROM experiment WHERE experiment_id IN (800001,800002,800003);
DELETE FROM model WHERE model_id=${restart_model_id};
DELETE FROM experiment_admin_request
WHERE request_id=${restart_cancel_request_id};
SQL

# Regression for the checkpoint-inference/cancellation race: both paths use
# the same advisory lock. The inference transaction must remain blocked until
# cancellation commits, then observe cancellation_request_id and insert no row.
psql -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(status,phase) VALUES ('running','train');
INSERT INTO model(experiment_id,comment)
VALUES (4,'periodic training checkpoint');
SQL
race_request_id="$(
    scalar "INSERT INTO experiment_admin_request (
        action,cancellation_mode,invocation_identity,status,
        previous_global_state,resulting_global_state
    ) VALUES (
        'cancel_all','immediate','checkpoint-race','applying',
        'running','running'
    ) RETURNING request_id"
)"
(
    psql -v ON_ERROR_STOP=1 -Atq -d "${test_db}" \
        >"${test_tmp}/checkpoint_lock_holder.out" <<SQL
BEGIN;
SELECT pg_advisory_xact_lock(
    hashtextextended('expertadvisor.global_experiment_execution_control.v1',0));
SELECT 'checkpoint_lock_held';
UPDATE experiment SET cancellation_request_id=${race_request_id}
WHERE experiment_id=4;
SELECT pg_sleep(1);
COMMIT;
SQL
) &
lock_holder_pid=$!
for _ in {1..100}; do
    if grep -q checkpoint_lock_held \
        "${test_tmp}/checkpoint_lock_holder.out" 2>/dev/null; then
        break
    fi
    sleep 0.02
done
grep -q checkpoint_lock_held "${test_tmp}/checkpoint_lock_holder.out"
(
    psql -v ON_ERROR_STOP=1 -Atq -d "${test_db}" \
        >"${test_tmp}/checkpoint_contender.out" <<'SQL'
BEGIN;
SELECT pg_advisory_xact_lock(
    hashtextextended('expertadvisor.global_experiment_execution_control.v1',0));
INSERT INTO experiment_checkpoint_eval (
    experiment_id,parent_experiment_id,checkpoint_epoch,checkpoint_model_id,
    status,phase
)
SELECT e.experiment_id,e.experiment_id,20,m.model_id,'pending','infer'
FROM experiment e
JOIN model m ON m.experiment_id=e.experiment_id
WHERE e.experiment_id=4 AND e.cancellation_request_id IS NULL;
COMMIT;
SQL
) &
checkpoint_contender_pid=$!
sleep 0.2
kill -0 "${checkpoint_contender_pid}"
wait "${lock_holder_pid}"
wait "${checkpoint_contender_pid}"
test "$(scalar "SELECT count(*) FROM experiment_checkpoint_eval
    WHERE experiment_id=4")" = 0

set +e
run_control --pause-all-experiments --resume-all-experiments --yes \
    >"${test_tmp}/invalid_pause_resume.out" 2>&1
invalid_pause_resume=$?
run_control --cancel-all-experiments --yes \
    >"${test_tmp}/invalid_cancel_mode.out" 2>&1
invalid_cancel_mode=$?
run_control --infer-before-cancel --yes \
    >"${test_tmp}/invalid_infer.out" 2>&1
invalid_infer=$?
set -e
test "${invalid_pause_resume}" -ne 0
test "${invalid_cancel_mode}" -ne 0
test "${invalid_infer}" -ne 0
grep -q 'mutually exclusive' "${test_tmp}/invalid_pause_resume.out"
grep -q 'requires exactly one' "${test_tmp}/invalid_cancel_mode.out"
grep -q 'requires --cancel-all-experiments' "${test_tmp}/invalid_infer.out"

test "$(scalar "SELECT count(*) FROM experiment
    WHERE current_operation IS NOT NULL
      AND current_operation NOT IN ('train','infer','analyze')")" = 0

echo "GlobalExperimentControlIntegrationTests passed"
