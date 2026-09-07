#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/Development/PauseResumePriorityContinuation/Build/Products/Debug/LSTM_Release}"
test_db="ea_scheduler_pause_priority_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-scheduler-pause-priority.XXXXXX")"
process_binary="${test_dir}/GlobalExperimentControlProcessTests"
worker_pids=()
worker_pgids=()
worker_starts=()
worker_executables=()
worker_commands=()
export PGOPTIONS=

cleanup() {
    local index
    for index in "${!worker_pids[@]}"; do
        if kill -0 "${worker_pids[${index}]}" >/dev/null 2>&1; then
            local observed="" pid="" pgid="" start="" executable="" command=""
            observed="$("${process_binary}" \
                --inspect-managed-test-process="${worker_pids[${index}]}" \
                2>/dev/null || true)"
            IFS='|' read -r pid pgid start executable command <<<"${observed}"
            if [[ "${pid}" = "${worker_pids[${index}]}" &&
                  "${pgid}" = "${worker_pgids[${index}]}" &&
                  "${start}" = "${worker_starts[${index}]}" &&
                  "${executable}" = "${worker_executables[${index}]}" ]]; then
                kill -CONT -- "-${pgid}" >/dev/null 2>&1 || true
                kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
            fi
        fi
        wait "${worker_pids[${index}]}" 2>/dev/null || true
    done
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf "${test_dir}"
}
trap cleanup EXIT

diagnose_error() {
    local result=$?
    for diagnostic in "${test_dir}"/*.out; do
        [[ -f "${diagnostic}" ]] || continue
        printf '%s\n' "--- ${diagnostic} ---" >&2
        tail -80 "${diagnostic}" >&2 || true
    done
    psql -X -Atq -d "${test_db}" -c \
        "SELECT e.experiment_id,e.status,e.phase,e.resume_requested,
                e.scheduler_priority,e.active_scheduler_worker_attempt_id,
                a.lifecycle_state,a.reconciliation_result
         FROM experiment e LEFT JOIN experiment_scheduler_worker_attempt a
           ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id
         ORDER BY e.experiment_id" >&2 || true
    return "${result}"
}
trap diagnose_error ERR

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/093_scheduler_priority_preemption.sql"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE
SET desired_state='running',active_request_id=NULL,current_pause_request_id=NULL;
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',cutover_completed_at=clock_timestamp(),
    cutover_completed_by='pause_resume_priority_integration',
    cutover_executable_path='/tmp/isolated-test',
    cutover_process_evidence='isolated_disposable_database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;
SQL

read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"
"${CXX:-clang++}" -std=c++20 -O0 -g \
    -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" "${pqxx_compile_flags[@]}" \
    "${repo_root}/Tests/GlobalExperimentControlProcessTests.cpp" \
    "${repo_root}/Sources/GlobalExperimentControl.cpp" \
    "${pqxx_link_flags[@]}" -o "${process_binary}"

scalar() {
    psql -X -Atq -d "${test_db}" -c "$1"
}

run_cli() {
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" "$@"
}

insert_pending() {
    local experiment_id="$1" phase="$2" priority="$3" updated="$4"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v phase="${phase}" \
        -v priority="${priority}" -v updated="${updated}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,status,phase,current_operation,duplicate_nonce,
    scheduler_priority,updated_at,model_input_width,
    model_input_semantic_layout_version
) VALUES(
    :experiment_id,'priorityfixture',1,0.0008,1.0,1.0,2,1,
    '2020-01-01','2020-02-01','pending',:'phase',:'phase',
    :experiment_id,:'priority',:'updated'::timestamptz,75,5
);
SQL
}

# Persistent priority defaults, validation, ordering, phase independence, and
# deterministic tie-breaking are all exercised in the disposable database.
insert_pending 860001 train normal '2026-01-01 00:00:01+00'
test "$(scalar "SELECT scheduler_priority||':'||resume_requested::text
    FROM experiment WHERE experiment_id=860001")" = "normal:false"
run_cli --set-experiment-priority=860001:high |
    grep -q 'SCHEDULER_PRIORITY_SET,experiment_id=860001,priority=high'
test "$(scalar "SELECT scheduler_priority FROM experiment
    WHERE experiment_id=860001")" = high
run_cli --set-experiment-priority=860001:low >/dev/null
run_cli --set-experiment-priority=860001:normal >/dev/null
test "$(scalar "SELECT scheduler_priority FROM experiment
    WHERE experiment_id=860001")" = normal
set +e
run_cli --set-experiment-priority=860001:urgent \
    >"${test_dir}/invalid-priority.out" 2>&1
invalid_priority_result=$?
set -e
test "${invalid_priority_result}" -ne 0
grep -q 'requires high, normal, or low' "${test_dir}/invalid-priority.out"

insert_pending 860002 train high '2026-01-01 00:00:02+00'
insert_pending 860003 train normal '2026-01-01 00:00:03+00'
insert_pending 860004 train low '2026-01-01 00:00:04+00'
insert_pending 860005 infer normal '2026-01-01 00:00:03+00'
insert_pending 860006 train normal '2026-01-01 00:00:03+00'
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET resume_requested=true,
     scheduler_resume_origin='preemption',scheduler_priority='low',
     updated_at='2026-01-01 00:00:05+00' WHERE experiment_id=860004"
test "$(scalar "SELECT string_agg(experiment_id::text,',' ORDER BY
        CASE scheduler_priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,
        CASE scheduler_resume_origin WHEN 'operator' THEN 0
             WHEN 'preemption' THEN 1 ELSE 2 END,
        updated_at,experiment_id)
    FROM experiment WHERE experiment_id BETWEEN 860002 AND 860006")" = \
    "860002,860003,860005,860006,860004"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "DELETE FROM experiment WHERE experiment_id BETWEEN 860001 AND 860006"

launch_worker() {
    local experiment_id="$1" attempt_id="$2"
    local worker_link="${test_dir}/LSTM_Release_${experiment_id}"
    local ready="${test_dir}/${experiment_id}.ready"
    local identity="" pid="" pgid="" start="" executable="" command=""
    ln -sf "${process_binary}" "${worker_link}"
    "${worker_link}" --managed-test-worker --self-session --train \
        --scheduler-experiment-id="${experiment_id}" \
        --scheduler-worker-attempt-id="${attempt_id}" --ready-fd=9 \
        9>"${ready}" &
    local launched_pid=$!
    for _ in {1..100}; do
        identity="$("${process_binary}" \
            --inspect-managed-test-process="${launched_pid}" 2>/dev/null || true)"
        [[ -s "${ready}" && -n "${identity}" ]] && break
        sleep 0.02
    done
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    test "${pid}" = "${launched_pid}"
    test "${pgid}" = "${launched_pid}"
    test -n "${start}"
    test -n "${executable}"
    worker_pids+=("${pid}")
    worker_pgids+=("${pgid}")
    worker_starts+=("${start}")
    worker_executables+=("${executable}")
    worker_commands+=("${command}")
}

persist_worker() {
    local index="$1" experiment_id="$2" attempt_id="$3"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v attempt_id="${attempt_id}" \
        -v pid="${worker_pids[${index}]}" -v pgid="${worker_pgids[${index}]}" \
        -v start="${worker_starts[${index}]}" \
        -v executable="${worker_executables[${index}]}" \
        -v command="${worker_commands[${index}]}" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,status,phase,current_operation,duplicate_nonce,
    scheduler_priority,worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    worker_control_state,model_input_width,
    model_input_semantic_layout_version
) VALUES(
    :experiment_id,'workerfixture',1,0.0008,1.0,1.0,2,1,
    '2020-01-01','2020-02-01','running','train','train',
    :experiment_id,'low',:pid,:pgid,:'start',:'executable',:'command','running',
    75,5
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,
    spawned_at,registered_at
) VALUES(
    :attempt_id,'pause-priority-'||:attempt_id,:experiment_id,'experiment',
    'train','train','prior_scheduler_observed','running',
    :pid,:pgid,:'start',:'executable',:'command',
    'experiment:'||:experiment_id||':train',clock_timestamp(),clock_timestamp()
);
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=:experiment_id;
SQL
}

launch_worker 861001 9861001
persist_worker 0 861001 9861001
run_cli --scheduler-status >"${test_dir}/managed-running-status.out"
grep -q "SCHEDULER_STATUS_WORKER,pid=${worker_pids[0]},kind=train,managed=1,authoritative=1,detected=1,execution_state=running,lifecycle_status=running,attempt_state=running,identity_result=validated,executable_identity_match=1" \
    "${test_dir}/managed-running-status.out"
grep -q 'managed_train_workers=1.*managed_running_train_workers=1.*managed_paused_train_workers=0' \
    "${test_dir}/managed-running-status.out"
run_cli --pause-experiment=861001 --yes |
    grep -q 'new_status=paused,resume_requested=false,worker_state=stopped'
for _ in {1..100}; do
    state="$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')"
    [[ "${state}" == T* ]] && break
    sleep 0.02
done
[[ "${state}" == T* ]]
test "$(scalar "SELECT e.status||':'||e.resume_requested::text||':'||
        a.lifecycle_state FROM experiment e JOIN
        experiment_scheduler_worker_attempt a ON a.worker_attempt_id=
        e.active_scheduler_worker_attempt_id WHERE e.experiment_id=861001")" = \
    "paused:false:stopped"
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt
    WHERE capacity_class='train' AND lifecycle_state IN
    ('reserved','spawned','running','observed','identity_ambiguous')")" = 0
kill -0 "${worker_pids[0]}"
run_cli --scheduler-status >"${test_dir}/managed-paused-status.out"
grep -q "SCHEDULER_STATUS_WORKER,pid=${worker_pids[0]},kind=train,managed=1,authoritative=1,detected=1,execution_state=stopped,lifecycle_status=paused,attempt_state=stopped,identity_result=validated,executable_identity_match=1" \
    "${test_dir}/managed-paused-status.out"
grep -q 'managed_train_workers=1.*managed_running_train_workers=0.*managed_paused_train_workers=1' \
    "${test_dir}/managed-paused-status.out"
! grep -q "SCHEDULER_STATUS_UNMANAGED_WORKER,pid=${worker_pids[0]}" \
    "${test_dir}/managed-paused-status.out"

launch_worker 861002 9861002
persist_worker 1 861002 9861002
run_cli --resume-experiment=861001 --yes |
    grep -q 'queued_for_admission'
run_cli --resume-experiment=861001 --yes |
    grep -q 'already_satisfied'
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]

run_cli --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/full-capacity-logs" \
    >"${test_dir}/full-capacity.out"
test "$(scalar "SELECT status||':'||resume_requested::text FROM experiment
    WHERE experiment_id=861001")" = "pending:true"
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]

# Re-pausing an exact stopped worker awaiting scheduler admission retains the
# attempt and process identity, clears resume priority, and sends no SIGSTOP.
repause_attempt="$(scalar "SELECT active_scheduler_worker_attempt_id FROM
    experiment WHERE experiment_id=861001")"
repause_signal="$(scalar "SELECT signal_number FROM
    experiment_scheduler_worker_attempt WHERE worker_attempt_id=9861001")"
repause_dry_run_state="$(scalar "SELECT e.status||':'||
    e.resume_requested::text||':'||e.scheduler_priority||':'||
    e.active_scheduler_worker_attempt_id::text||':'||a.lifecycle_state||':'||
    COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN
    experiment_scheduler_worker_attempt a ON a.worker_attempt_id=
    e.active_scheduler_worker_attempt_id WHERE e.experiment_id=861001")"
run_cli --pause-experiment=861001 --dry-run |
    grep -q 'worker_action=validate_and_retain_stopped'
test "${repause_dry_run_state}" = "$(scalar "SELECT e.status||':'||
    e.resume_requested::text||':'||e.scheduler_priority||':'||
    e.active_scheduler_worker_attempt_id::text||':'||a.lifecycle_state||':'||
    COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN
    experiment_scheduler_worker_attempt a ON a.worker_attempt_id=
    e.active_scheduler_worker_attempt_id WHERE e.experiment_id=861001")"
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]
run_cli --pause-experiment=861001 --yes |
    grep -q 'new_status=paused,resume_requested=false,worker_state=stopped'
test "$(scalar "SELECT e.status||':'||e.resume_requested::text||':'||
    e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id::text||':'||
    a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL')
    FROM experiment e JOIN experiment_scheduler_worker_attempt a ON
    a.worker_attempt_id=e.active_scheduler_worker_attempt_id
    WHERE e.experiment_id=861001")" = \
    "paused:false:low:${repause_attempt}:stopped:${repause_signal}"
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt
    WHERE experiment_id=861001")" = 1
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]
run_cli --resume-experiment=861001 --yes |
    grep -q 'queued_for_admission'
test "$(scalar "SELECT status||':'||resume_requested::text FROM experiment
    WHERE experiment_id=861001")" = "pending:true"

# Release only the disposable capacity fixture, then prove the exact stopped
# low-priority worker gets the next available slot without a duplicate launch.
kill -TERM -- "-${worker_pgids[1]}"
wait "${worker_pids[1]}" || true
run_cli --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/resume-admission-logs" \
    >"${test_dir}/resume-admission.out"
grep -q 'SCHEDULER_STOPPED_WORKER_ADMITTED,experiment_id=861001' \
    "${test_dir}/resume-admission.out"
test "$(scalar "SELECT status||':'||resume_requested::text||':'||
        scheduler_priority FROM experiment WHERE experiment_id=861001")" = \
    "running:false:low"
test "$(scalar "SELECT worker_pid FROM experiment
    WHERE experiment_id=861001")" = "${worker_pids[0]}"
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" != T* ]]
run_cli --scheduler-status >"${test_dir}/managed-resumed-status.out"
grep -q "SCHEDULER_STATUS_WORKER,pid=${worker_pids[0]},kind=train,managed=1,authoritative=1,detected=1,execution_state=running,lifecycle_status=running,attempt_state=running,identity_result=validated,executable_identity_match=1" \
    "${test_dir}/managed-resumed-status.out"

# Repeated pause is reconciliatory, and a scheduler restart observes one
# stopped exact attempt without consuming capacity or launching a duplicate.
run_cli --pause-experiment=861001 --yes >/dev/null
run_cli --pause-experiment=861001 --yes |
    grep -q 'worker_state=stopped'
run_cli --schedule-experiments --scheduler-once --recover-orphans-only \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/stopped-restart-logs" \
    >"${test_dir}/stopped-restart.out"
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt
    WHERE experiment_id=861001")" = 1
test "$(scalar "SELECT lifecycle_state FROM
    experiment_scheduler_worker_attempt WHERE experiment_id=861001")" = stopped

run_cli --resume-experiment=861001 --yes >/dev/null
run_cli --schedule-experiments --scheduler-once \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/pre-global-admission-logs" \
    >"${test_dir}/pre-global-admission.out"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=861001")" = \
    running

# Global generation membership excludes a row paused before pause-all. Resume
# all queues its own members and sends no eager SIGCONT.
insert_pending 861004 train normal '2027-01-01 00:00:00+00'
insert_pending 861005 train normal '2027-01-01 00:00:01+00'
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET status='paused' WHERE experiment_id=861005"
run_cli --pause-all-experiments --yes |
    grep -q 'global_state=paused'
pause_generation="$(scalar "SELECT current_pause_request_id FROM
    experiment_global_control WHERE singleton")"
test "$(scalar "SELECT worker_global_pause_request_id::text FROM experiment
    WHERE experiment_id=861004")" = "${pause_generation}"
test "$(scalar "SELECT worker_global_pause_request_id IS NULL FROM experiment
    WHERE experiment_id=861005")" = t
run_cli --resume-all-experiments --yes |
    grep -q 'signal_attempted=0,global_state=running'
test "$(scalar "SELECT status||':'||resume_requested::text FROM experiment
    WHERE experiment_id=861001")" = "pending:true"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=861005")" = paused
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]

# A vanished stopped worker is detached without clearing resume priority. The
# following dry-run proves it takes the ordinary restart command path.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET status='paused',resume_requested=false,
     scheduler_resume_origin='none'
     WHERE experiment_id IN (861001,861004)"
launch_worker 861006 9861006
persist_worker 2 861006 9861006
run_cli --pause-experiment=861006 --yes >/dev/null
kill -CONT -- "-${worker_pgids[2]}"
kill -TERM -- "-${worker_pgids[2]}"
wait "${worker_pids[2]}" || true
run_cli --scheduler-status >"${test_dir}/paused-missing-status.out"
grep -q "SCHEDULER_STATUS_WORKER,pid=${worker_pids[2]},kind=train,managed=0,authoritative=1,detected=0,execution_state=missing,lifecycle_status=paused,attempt_state=stopped,identity_result=process_missing" \
    "${test_dir}/paused-missing-status.out"
grep -q 'expected_missing_train_workers=1' \
    "${test_dir}/paused-missing-status.out"
! grep -q "SCHEDULER_STATUS_UNMANAGED_WORKER,pid=${worker_pids[2]}" \
    "${test_dir}/paused-missing-status.out"
run_cli --resume-experiment=861006 --yes >/dev/null
run_cli --schedule-experiments --scheduler-once --recover-orphans-only \
    --max-train-procs=0 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/missing-reconcile-logs" \
    >"${test_dir}/missing-reconcile.out"
test "$(scalar "SELECT status||':'||resume_requested::text||':'||
        (active_scheduler_worker_attempt_id IS NULL)::text
    FROM experiment WHERE experiment_id=861006")" = "pending:true:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM
    experiment_scheduler_worker_attempt WHERE experiment_id=861006")" = \
    "abandoned:stopped_process_missing"
run_cli --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=1 --max-infer-procs=0 --max-analyze-procs=0 \
    --scheduler-log-dir="${test_dir}/missing-fallback-logs" \
    >"${test_dir}/missing-fallback.out"
grep -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=861006,phase=train,dry_run=1' \
    "${test_dir}/missing-fallback.out"

printf '%s\n' \
    'SchedulerPauseResumePriorityIntegrationTests passed' \
    'priority_order=high,normal,low,operator,preemption,ordinary,updated_at,experiment_id' \
    'phase_in_priority_order=false' \
    'production_processes_signaled=0'
