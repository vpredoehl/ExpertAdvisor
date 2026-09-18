#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"
semantic_worker_registry="${2:-${repo_root}/Builds/SemanticWorkers/registry.json}"
semantic_worker_registry="$(cd "$(dirname "${semantic_worker_registry}")" && pwd)/$(basename "${semantic_worker_registry}")"
test -f "${semantic_worker_registry}"
worker_path_for_layout() {
    /usr/bin/python3 - "${semantic_worker_registry}" "$1" <<'PY'
import json
from pathlib import Path
import sys
path = Path(sys.argv[1])
layout = int(sys.argv[2])
registry = json.loads(path.read_text(encoding="utf-8"))
by_layout = {entry["semantic_layout"]: entry for entry in registry["workers"]}
print((path.parent / by_layout[layout]["executable"]).resolve(strict=True))
PY
}
current_worker_binary="$(worker_path_for_layout 7)"
legacy_layout6_binary="$(worker_path_for_layout 6)"
test -x "${current_worker_binary}"
test -x "${legacy_layout6_binary}"
test_db="ea_scheduler_semantic_admission_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_semantic_admission.XXXXXX)"
current_only_root="${test_dir}/current-only-semantic-workers"
current_only_registry="${current_only_root}/registry.json"
current_only_worker_binary=""

cleanup() {
    local status=$?
    local pid=""
    while IFS= read -r pid; do
        [[ -n "${pid}" ]] || continue
        if kill -0 "${pid}" >/dev/null 2>&1; then
            local command=""
            command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
            if [[ ( "${command}" == *"${scheduler_binary}"* ||
                    "${command}" == *"${current_worker_binary}"* ||
                    ( -n "${current_only_worker_binary}" &&
                      "${command}" == *"${current_only_worker_binary}"* ) ||
                    "${command}" == *"${legacy_layout6_binary}"* ) &&
                  ( "${command}" == *"--scheduler-experiment-id=917001"* ||
                    "${command}" == *"--scheduler-experiment-id=917002"* ||
                    "${command}" == *"--scheduler-experiment-id=917003"* ||
                    "${command}" == *"--scheduler-checkpoint-eval-id=917101"* ) ]]; then
                kill -TERM -- "-${pid}" >/dev/null 2>&1 ||
                    kill -TERM "${pid}" >/dev/null 2>&1 || true
            fi
        fi
    done < <(psql -X -At -d "${test_db}" -c \
        "SELECT worker_pid FROM experiment_scheduler_worker_attempt
         WHERE experiment_id IN (917001,917002,917003,917004,917005)
           AND worker_pid IS NOT NULL" \
        2>/dev/null || true)
    if [[ "${status}" -ne 0 ]]; then
        for output in "${test_dir}"/*.out; do
            [[ -f "${output}" ]] || continue
            sed -n '1,240p' "${output}" >&2
        done
    fi
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
    return "${status}"
}
trap cleanup EXIT

current_only_worker_binary="$(/usr/bin/python3 - \
    "${semantic_worker_registry}" "${current_only_root}" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

source, destination = map(Path, sys.argv[1:])
registry = json.loads(source.read_text(encoding="utf-8"))
current = [entry for entry in registry["workers"]
           if entry["semantic_layout"] == registry["current_layout"]]
assert len(current) == 1
entry = current[0]
for field in ("executable", "manifest"):
    source_artifact = (source.parent / entry[field]).resolve(strict=True)
    destination_artifact = destination / entry[field]
    destination_artifact.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source_artifact, destination_artifact)
    except OSError:
        shutil.copy2(source_artifact, destination_artifact)
runtime = next(item for item in registry["runtimes"]
               if item["identity"] == entry["runtime_identity"])
source_runtime = (source.parent / runtime["directory"]).resolve(strict=True)
destination_runtime = destination / runtime["directory"]
destination_runtime.mkdir(parents=True, exist_ok=True)
for source_resource in source_runtime.iterdir():
    if not source_resource.is_file():
        continue
    destination_resource = destination_runtime / source_resource.name
    try:
        os.link(source_resource, destination_resource)
    except OSError:
        shutil.copy2(source_resource, destination_resource)
worker_directory = (destination / entry["executable"]).parent
for runtime_name in ("default.metallib", "MetaNN.metallib"):
    link = worker_directory / runtime_name
    link.symlink_to(os.path.relpath(destination_runtime / runtime_name,
                                    worker_directory))
registry["workers"] = [entry]
registry["runtimes"] = [runtime]
(destination / "registry.json").write_text(
    json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
print((destination / entry["executable"]).resolve(strict=True))
PY
)"
test -x "${current_only_worker_binary}"

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql \
    086_scheduler_pause_resume_priority.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running')
ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='phase17d-semantic-admission-test',
    cutover_executable_path='/isolated/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;

INSERT INTO model(model_id,name) VALUES
    (917011,'phase17d-layout6-model'),
    (917012,'phase17d-layout7-model'),
    (917013,'legacy-layout2-resume-model'),
    (917014,'legacy-layout6-last-model'),
    (917015,'legacy-layout6-infer-model'),
    (917016,'legacy-layout6-checkpoint-model');

WITH v AS (SELECT ARRAY[1.0,53.0,1.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 917013,'model_meta',1,3,0,i-1,a[i]
FROM v,generate_series(1,3)i;

WITH v AS (SELECT ARRAY[1.0,2.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 917013,'model_input_semantics_meta',1,2,0,i-1,a[i]
FROM v,generate_series(1,2)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT 917013,'param',54,4,(i-1)/4,(i-1)%4,0.0
FROM generate_series(1,216)i;

WITH v AS (SELECT ARRAY[1.0,77.0,1.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT model_id,'model_meta',1,3,0,i-1,a[i]
FROM v,
     (VALUES(917014),(917015),(917016)) models(model_id),
     generate_series(1,3)i;

WITH v AS (SELECT ARRAY[1.0,6.0]::double precision[] a)
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT model_id,'model_input_semantics_meta',1,2,0,i-1,a[i]
FROM v,
     (VALUES(917014),(917015),(917016)) models(model_id),
     generate_series(1,2)i;

INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT model_id,'param',78,4,(i-1)/4,(i-1)%4,0.0
FROM (VALUES(917014),(917015),(917016)) models(model_id),
     generate_series(1,312)i;
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,current_operation,last_model_id,
    scheduler_priority,duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,updated_at
) VALUES
    (917001,'phase17dincompatible',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'pending','infer','infer',917011,'high',917001,77,6,
     clock_timestamp()-interval '2 minutes'),
    (917002,'phase17dcompatible',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'pending','infer','infer',917012,'normal',917002,77,7,
     clock_timestamp()-interval '1 minute');
UPDATE model SET experiment_id=917001 WHERE model_id=917011;
UPDATE model SET experiment_id=917002 WHERE model_id=917012;

ALTER TABLE experiment DISABLE TRIGGER USER;

INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,checkpoint_interval,train_start,train_end,
    infer_start,infer_end,status,phase,current_operation,
    last_model_id,resume_model_id,current_epoch,
    scheduler_priority,resume_requested,scheduler_resume_origin,
    duplicate_nonce,model_input_width,
    model_input_semantic_layout_version,updated_at
) VALUES
    (917003,'legacyresume',4,0.0008,80,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'pending','train','train',
     917014,917013,60,
     'high',true,'operator',
     917003,NULL,NULL,
     clock_timestamp()-interval '4 minutes'),

    (917004,'legacyinferbad',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'pending','infer','infer',
     917015,NULL,20,
     'high',false,'none',
     917004,NULL,NULL,
     clock_timestamp()-interval '3 minutes'),

    (917005,'legacycheckpoint',4,0.0008,20,20,
     '2020-01-01','2020-02-01','2020-02-01','2020-03-01',
     'completed','done',NULL,
     917016,NULL,20,
     'normal',false,'none',
     917005,77,6,
     clock_timestamp()-interval '3 minutes');

ALTER TABLE experiment ENABLE TRIGGER USER;

UPDATE model SET experiment_id=917003
WHERE model_id IN (917013,917014);

UPDATE model SET experiment_id=917004
WHERE model_id=917015;

UPDATE model SET experiment_id=917005
WHERE model_id=917016;

INSERT INTO experiment_checkpoint_eval(
    checkpoint_eval_id,experiment_id,parent_experiment_id,
    checkpoint_epoch,checkpoint_model_id,status,phase,created_at,updated_at
) VALUES(
    917101,917005,917005,20,917016,'pending','infer',
    clock_timestamp()-interval '2 minutes',clock_timestamp()
);
SQL

current_worker_directory="$(dirname "${current_only_worker_binary}")"
mv "${current_worker_directory}/MetaNN.metallib" \
   "${current_worker_directory}/MetaNN.metallib.missing"
invocations_before_missing_runtime="$(psql -X -At -d "${test_db}" -c \
    'SELECT count(*) FROM experiment_scheduler_invocation')"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=0 --max-infer-procs=1 --max-analyze-procs=0 \
    --semantic-worker-registry="${current_only_registry}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/missing-runtime.out" 2>&1
missing_runtime_rc=$?
set -e
mv "${current_worker_directory}/MetaNN.metallib.missing" \
   "${current_worker_directory}/MetaNN.metallib"
test "${missing_runtime_rc}" = 1
grep -Fq 'SCHEDULER_START_REJECTED,diagnostic=semantic_worker_runtime_dependency_missing:resource=MetaNN.metallib' \
    "${test_dir}/missing-runtime.out"
grep -Fq 'authority_acquired=0,workers_launched=0' \
    "${test_dir}/missing-runtime.out"
test "$(psql -X -At -d "${test_db}" -c \
    'SELECT count(*) FROM experiment_scheduler_invocation')" = \
    "${invocations_before_missing_runtime}"
test "$(psql -X -At -d "${test_db}" -c \
    'SELECT status||'\'':'\''||phase FROM experiment WHERE experiment_id=917002')" = \
    'pending:infer'
test "$(psql -X -At -d "${test_db}" -c \
    'SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=917002')" = 0

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=0 --max-infer-procs=1 --max-analyze-procs=0 \
    --semantic-worker-registry="${current_only_registry}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/admission.out" 2>&1

grep -q 'SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE,experiment_id=917001,phase=infer,.*diagnostic=semantic_worker_layout_unsupported:layout=6,capacity_consumed=0,child_launched=0,experiment_status_changed=false' \
    "${test_dir}/admission.out"
! grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917001' \
    "${test_dir}/admission.out"

grep -q 'SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE,experiment_id=917004,phase=infer,.*model_input_width=77,.*model_input_semantic_layout_version=6,.*diagnostic=semantic_worker_layout_unsupported:layout=6,capacity_consumed=0,child_launched=0,experiment_status_changed=false' \
    "${test_dir}/admission.out"
! grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917004' \
    "${test_dir}/admission.out"

grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917002,.*phase=infer' \
    "${test_dir}/admission.out"
grep -Fq "SCHEDULER_INFER_WORKER_SELECTED,experiment_id=917002,model_id=917012,model_input_width=77,model_input_semantic_layout_version=7,worker_semantic_layout_version=7,worker_executable=${current_only_worker_binary},reason=current_published_semantic_worker" \
    "${test_dir}/admission.out"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917001")" = "0"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase FROM experiment WHERE experiment_id=917001")" = \
    "pending:infer"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917002")" = "1"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT canonical_executable_path
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917002 ORDER BY worker_attempt_id DESC LIMIT 1")" = \
    "${current_only_worker_binary}"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT position('--semantic-worker-registry' IN command_line)
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917002 ORDER BY worker_attempt_id DESC LIMIT 1")" = 0

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917004")" = "0"

layout7_pid="$(psql -X -At -d "${test_db}" -c \
        "SELECT worker_pid FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917002 ORDER BY worker_attempt_id DESC LIMIT 1")"
for _ in {1..200}; do
    if [[ -z "${layout7_pid}" ]] ||
       ! kill -0 "${layout7_pid}" >/dev/null 2>&1; then
        break
    fi
    sleep 0.02
done
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --max-train-procs=0 --max-infer-procs=0 --max-analyze-procs=0 \
    --semantic-worker-registry="${current_only_registry}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/layout7-recovery.out" 2>&1
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET scheduler_priority='low'
     WHERE experiment_id=917004"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --max-train-procs=0 --max-infer-procs=0 --max-analyze-procs=0 \
    --semantic-worker-registry="${semantic_worker_registry}" \
    --legacy-layout6-infer-worker="${legacy_layout6_binary}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/matching-legacy-assertion.out" 2>&1
grep -Fq "legacy_layout6_identity_assertion=${legacy_layout6_binary}" \
    "${test_dir}/matching-legacy-assertion.out"

invocations_before_conflict="$(psql -X -At -d "${test_db}" -c \
    'SELECT count(*) FROM experiment_scheduler_invocation')"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --max-train-procs=0 --max-infer-procs=0 --max-analyze-procs=0 \
    --semantic-worker-registry="${semantic_worker_registry}" \
    --legacy-layout6-infer-worker="${current_worker_binary}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/conflicting-legacy-assertion.out" 2>&1
conflict_rc=$?
set -e
test "${conflict_rc}" = 1
grep -q 'SCHEDULER_START_REJECTED,diagnostic=legacy_layout6_worker_registry_conflict:.*authority_acquired=0,workers_launched=0' \
    "${test_dir}/conflicting-legacy-assertion.out"
test "$(psql -X -At -d "${test_db}" -c \
    'SELECT count(*) FROM experiment_scheduler_invocation')" = \
    "${invocations_before_conflict}"

    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-train-procs=0 --max-infer-procs=1 --max-analyze-procs=0 \
        --semantic-worker-registry="${semantic_worker_registry}" \
        --scheduler-log-dir="${test_dir}/logs" \
        >"${test_dir}/layout6-routing.out" 2>&1

    grep -Fq "SCHEDULER_INFER_WORKER_SELECTED,experiment_id=917001,model_id=917011,model_input_width=77,model_input_semantic_layout_version=6,worker_semantic_layout_version=6,worker_executable=${legacy_layout6_binary},reason=immutable_historical_semantic_worker" \
        "${test_dir}/layout6-routing.out"
    grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917001,.*phase=infer' \
        "${test_dir}/layout6-routing.out"

    legacy_registered=false
    for _ in {1..200}; do
        legacy_registered="$(psql -X -At -d "${test_db}" -c \
            "SELECT COALESCE(registered_at IS NOT NULL,false)::text
             FROM experiment_scheduler_worker_attempt
             WHERE experiment_id=917001
             ORDER BY worker_attempt_id DESC LIMIT 1")"
        [[ "${legacy_registered}" = true ]] && break
        sleep 0.02
    done
    test "${legacy_registered}" = true
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT canonical_executable_path
         FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917001 ORDER BY worker_attempt_id DESC LIMIT 1")" = \
        "${legacy_layout6_binary}"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT position('--legacy-layout6-infer-worker' IN command_line)
         FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917001 ORDER BY worker_attempt_id DESC LIMIT 1")" = 0
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT position('--semantic-worker-registry' IN command_line)
         FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917001 ORDER BY worker_attempt_id DESC LIMIT 1")" = 0
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT worker_executable FROM experiment WHERE experiment_id=917001")" = \
        "${legacy_layout6_binary}"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT canonical_executable_path
         FROM experiment_scheduler_invocation
         WHERE command_line LIKE '%--semantic-worker-registry=%'
           AND command_line NOT LIKE '%--legacy-layout6-infer-worker=%'
         ORDER BY ownership_acquired_at DESC LIMIT 1")" = \
        "${scheduler_binary}"

    legacy_final_pid="$(psql -X -At -d "${test_db}" -c \
        "SELECT worker_pid FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917001
         ORDER BY worker_attempt_id DESC LIMIT 1")"
    if [[ -n "${legacy_final_pid}" ]] &&
       kill -0 "${legacy_final_pid}" >/dev/null 2>&1; then
        legacy_final_command="$(ps -p "${legacy_final_pid}" -o command= 2>/dev/null || true)"
        [[ "${legacy_final_command}" == *"${legacy_layout6_binary}"* ]]
        [[ "${legacy_final_command}" == *"--scheduler-experiment-id=917001"* ]]
        kill -TERM -- "-${legacy_final_pid}" >/dev/null 2>&1 ||
            kill -TERM "${legacy_final_pid}" >/dev/null 2>&1 || true
    fi
    for _ in {1..200}; do
        if [[ -z "${legacy_final_pid}" ]] ||
           ! kill -0 "${legacy_final_pid}" >/dev/null 2>&1; then
            break
        fi
        sleep 0.02
    done
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once --recover-orphans-only \
        --max-train-procs=0 --max-infer-procs=0 --max-analyze-procs=0 \
        --semantic-worker-registry="${semantic_worker_registry}" \
        --scheduler-log-dir="${test_dir}/logs" \
        >"${test_dir}/layout6-final-recovery.out" 2>&1
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=917001
           AND canonical_executable_path='${legacy_layout6_binary}'
           AND lifecycle_state NOT IN
               ('reserved','spawned','running','observed','stopped',
                'identity_ambiguous')")" = 1
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
        "UPDATE experiment SET status='failed',completed_at=clock_timestamp(),
             error_message='fixture_final_queue_retired',updated_at=clock_timestamp()
         WHERE experiment_id=917004 AND status='pending' AND phase='infer'"

    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-train-procs=0 --max-infer-procs=1 --max-analyze-procs=0 \
        --semantic-worker-registry="${semantic_worker_registry}" \
        --scheduler-log-dir="${test_dir}/logs" \
        >"${test_dir}/layout6-checkpoint-routing.out" 2>&1

    grep -Fq "SCHEDULER_INFER_WORKER_SELECTED,experiment_id=917005,model_id=917016,model_input_width=77,model_input_semantic_layout_version=6,worker_semantic_layout_version=6,worker_executable=${legacy_layout6_binary},reason=immutable_historical_semantic_worker" \
        "${test_dir}/layout6-checkpoint-routing.out"
    grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917005,.*phase=infer,worker_kind=checkpoint_infer' \
        "${test_dir}/layout6-checkpoint-routing.out"

    checkpoint_registered=false
    for _ in {1..200}; do
        checkpoint_registered="$(psql -X -At -d "${test_db}" -c \
            "SELECT COALESCE(registered_at IS NOT NULL,false)::text
             FROM experiment_scheduler_worker_attempt
             WHERE checkpoint_eval_id=917101
             ORDER BY worker_attempt_id DESC LIMIT 1")"
        [[ "${checkpoint_registered}" = true ]] && break
        sleep 0.02
    done
    test "${checkpoint_registered}" = true
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT worker_kind||':'||capacity_class||':'||canonical_executable_path
         FROM experiment_scheduler_worker_attempt
         WHERE checkpoint_eval_id=917101
         ORDER BY worker_attempt_id DESC LIMIT 1")" = \
        "checkpoint_infer:infer:${legacy_layout6_binary}"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT position('--legacy-layout6-infer-worker' IN command_line)
         FROM experiment_scheduler_worker_attempt
         WHERE checkpoint_eval_id=917101
         ORDER BY worker_attempt_id DESC LIMIT 1")" = 0
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT position('--semantic-worker-registry' IN command_line)
         FROM experiment_scheduler_worker_attempt
         WHERE checkpoint_eval_id=917101
         ORDER BY worker_attempt_id DESC LIMIT 1")" = 0
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT worker_executable
         FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=917101")" = \
        "${legacy_layout6_binary}"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --semantic-worker-registry="${semantic_worker_registry}" \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/legacy-train-admission.out" 2>&1

grep -q 'SCHEDULER_CHILD_LAUNCHED,.*experiment_id=917003,.*phase=train' \
    "${test_dir}/legacy-train-admission.out"

! grep -q 'SCHEDULER_SEMANTIC_WORKER_INCOMPATIBLE,experiment_id=917003' \
    "${test_dir}/legacy-train-admission.out"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917003")" = "1"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT canonical_executable_path
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917003 ORDER BY worker_attempt_id DESC LIMIT 1")" = \
    "${current_worker_binary}"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT position('--semantic-worker-registry' IN command_line)
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=917003 ORDER BY worker_attempt_id DESC LIMIT 1")" = 0

printf '%s\n' "SchedulerSemanticAdmissionIntegrationTests passed"
