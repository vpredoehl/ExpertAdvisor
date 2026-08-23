#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/SchedulerOwnership/Build/Products/Release/LSTM_Release}"
process_binary="${2:-${repo_root}/DerivedData/SchedulerOwnership/Tests/GlobalExperimentControlProcessTests}"
test_db="ea_scheduler_process_test_${$}"
test_dir="$(mktemp -d /tmp/ea_scheduler_process_test.XXXXXX)"
worker_pids=()
worker_pgids=()
worker_starts=()
worker_executables=()
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

case "${test_db}" in
    ea_scheduler_process_test_[0-9]*) ;;
    *) exit 90 ;;
esac

inspect_process() {
    "${process_binary}" --inspect-managed-test-process="$1" 2>/dev/null
}

process_identity_matches() {
    local index="$1"
    local identity=""
    local pid="" pgid="" start="" executable="" command=""
    identity="$(inspect_process "${worker_pids[${index}]}" || true)"
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    [[ "${pid}" = "${worker_pids[${index}]}" ]] &&
        [[ "${pgid}" = "${worker_pgids[${index}]}" ]] &&
        [[ "${start}" = "${worker_starts[${index}]}" ]] &&
        [[ "${executable}" = "${worker_executables[${index}]}" ]] &&
        [[ "${command}" == *"--managed-test-worker"* ||
           "${command}" == *"--scheduler-worker-attempt-id"* ]]
}

cleanup() {
    local index=""
    if [[ -n "${scheduler_pid}" ]] &&
       kill -0 "${scheduler_pid}" >/dev/null 2>&1; then
        local scheduler_identity=""
        local observed_pid="" observed_pgid="" observed_start=""
        local observed_executable="" observed_command=""
        scheduler_identity="$(inspect_process "${scheduler_pid}" || true)"
        IFS='|' read -r observed_pid observed_pgid observed_start \
            observed_executable observed_command <<<"${scheduler_identity}"
        if [[ "${observed_pid}" = "${scheduler_pid}" ]] &&
           [[ "${observed_start}" = "${scheduler_start}" ]] &&
           [[ "${observed_executable}" = "${scheduler_executable}" ]] &&
           [[ "${observed_command}" == *"--schedule-experiments"* ]]; then
            kill -TERM "${scheduler_pid}" >/dev/null 2>&1 || true
        fi
        wait "${scheduler_pid}" 2>/dev/null || true
    fi
    for index in "${!worker_pids[@]}"; do
        if kill -0 "${worker_pids[${index}]}" >/dev/null 2>&1 &&
           process_identity_matches "${index}"; then
            kill -TERM -- "-${worker_pgids[${index}]}" \
                >/dev/null 2>&1 || true
        fi
        wait "${worker_pids[${index}]}" 2>/dev/null || true
    done
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/071_resume_input_width_expansion.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/078_operator_forced_final_inference_rerun.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING;
SQL

# A corrected scheduler cannot even create an invocation-history row while
# the protocol cutover is pending.
protocol_snapshot="$(psql -X -At -d "${test_db}" -c \
    "SELECT cutover_state||':'||required_generation
     FROM experiment_scheduler_protocol WHERE singleton=true")"
invocations_before="$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_invocation")"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/pending-cutover.out" 2>&1
pending_cutover_result=$?
set -e
test "${pending_cutover_result}" = "3"
grep -q 'SCHEDULER_PROTOCOL_BARRIER_REJECTED' \
    "${test_dir}/pending-cutover.out"
grep -q 'mutations=0' "${test_dir}/pending-cutover.out"
test "${protocol_snapshot}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT cutover_state||':'||required_generation
     FROM experiment_scheduler_protocol WHERE singleton=true")"
test "${invocations_before}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_invocation")"

# The cutover command is fail-closed for an old scheduler observation and for
# process-inspection failure. Both rejections leave the protocol row exact.
cutover_path="${repo_root}/Tests/fixtures/scheduler_protocol_cutover_bin"
for cutover_mode in old_scheduler corrected_scheduler inspection_failure; do
    set +e
    PATH="${cutover_path}:${PATH}" \
    EA_SCHEDULER_PROTOCOL_TEST_PS_MODE="${cutover_mode}" \
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --complete-scheduler-protocol-cutover --yes \
        >"${test_dir}/cutover-${cutover_mode}.out" 2>&1
    cutover_result=$?
    set -e
    test "${cutover_result}" = "1"
    grep -q 'SCHEDULER_PROTOCOL_CUTOVER_REJECTED' \
        "${test_dir}/cutover-${cutover_mode}.out"
    grep -q 'mutations=0' \
        "${test_dir}/cutover-${cutover_mode}.out"
    test "${protocol_snapshot}" = "$(psql -X -At -d "${test_db}" -c \
        "SELECT cutover_state||':'||required_generation
         FROM experiment_scheduler_protocol WHERE singleton=true")"
done

# A failed/partial cutover remains a barrier, but the explicit command can
# replay it after positive absence evidence. Repeating a completed cutover is
# idempotent.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_protocol
     SET cutover_state='failed',
         failure_diagnostic='test_partial_cutover',
         updated_at=clock_timestamp()
     WHERE singleton=true AND cutover_state='pending'"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/failed-cutover-startup.out" 2>&1
failed_cutover_startup_result=$?
set -e
test "${failed_cutover_startup_result}" = "3"
grep -q 'reason=test_partial_cutover' \
    "${test_dir}/failed-cutover-startup.out"

for expected_result in completed already_complete; do
    PATH="${cutover_path}:${PATH}" \
    EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --complete-scheduler-protocol-cutover --yes \
        >"${test_dir}/cutover-${expected_result}.out" 2>&1
    grep -q "result=${expected_result}" \
        "${test_dir}/cutover-${expected_result}.out"
done
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT cutover_state||':'||required_generation||':'||
            (cutover_completed_at IS NOT NULL)::text
     FROM experiment_scheduler_protocol WHERE singleton=true")" = \
    "complete:52:true"

# Managed direct CLI targets cannot execute without the exact durable attempt.
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --train --scheduler-experiment-id=920051 \
    --symbol=fixturetrain 2020-01-01 2020-02-01 \
    >"${test_dir}/direct-train.out" 2>&1
direct_train_result=$?
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --infer --scheduler-experiment-id=920051 \
    --model=1 --symbol=fixtureinfer 2020-01-01 2020-02-01 \
    >"${test_dir}/direct-infer.out" 2>&1
direct_infer_result=$?
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --infer --scheduler-checkpoint-eval-id=925051 \
    --model=1 --symbol=fixturecheckpoint 2020-01-01 2020-02-01 \
    >"${test_dir}/direct-checkpoint-infer.out" 2>&1
direct_checkpoint_infer_result=$?
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --analyze-experiment=920051 \
    >"${test_dir}/direct-analyze.out" 2>&1
direct_analyze_result=$?
set -e
test "${direct_train_result}" = "1"
test "${direct_infer_result}" = "1"
test "${direct_checkpoint_infer_result}" = "1"
test "${direct_analyze_result}" = "1"
grep -q 'direct CLI execution of scheduler-managed work is prohibited' \
    "${test_dir}/direct-train.out"
grep -q 'direct CLI execution of scheduler-managed work is prohibited' \
    "${test_dir}/direct-infer.out"
grep -q 'direct CLI execution of scheduler-managed work is prohibited' \
    "${test_dir}/direct-checkpoint-infer.out"
grep -q 'DIRECT_CLI_MANAGED_WORK_REJECTED' \
    "${test_dir}/direct-analyze.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt")" = "0"

# Legacy no-PID rows fail closed during the bounded grace period, remain
# visible as capacity consumers, and then reconcile exactly once after positive
# cutover/process-absence evidence. The checkpoint-analyze fixture proves that
# a stale in-process owner is abandoned and requeued without retaining capacity.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
UPDATE experiment_scheduler_protocol
SET legacy_no_pid_grace_seconds=30,
    cutover_completed_at=clock_timestamp(),
    updated_at=clock_timestamp()
WHERE singleton=true AND cutover_state='complete';
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920064,'legacynopidfixture',4,0.0008,20,
    '2020-01-01','2021-01-01','running','train','train',920064
);
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt(
        launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,command_identity,
        reserved_at,diagnostic
    ) VALUES(
        'legacy:no-pid:process-test:920064',920064,'experiment','train',
        'train','legacy_unverified','identity_ambiguous',
        'experiment:920064:train',clock_timestamp(),
        'legacy_no_pid_fixture'
    ) RETURNING worker_attempt_id
)
UPDATE experiment
SET active_scheduler_worker_attempt_id=
    (SELECT worker_attempt_id FROM attempt)
WHERE experiment_id=920064;
SQL

PATH="${cutover_path}:${PATH}" \
EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/legacy-no-pid-grace.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result||':'||diagnostic
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920064")" = \
    "identity_ambiguous:legacy_no_pid_unresolved:legacy_no_pid_grace_not_elapsed"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --scheduler-status >"${test_dir}/legacy-no-pid-status.out" 2>&1
grep -q \
    'SCHEDULER_STATUS_LEGACY_NO_PID,capacity_class=train,unresolved=1,capacity_consumed=1' \
    "${test_dir}/legacy-no-pid-status.out"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_protocol
     SET cutover_completed_at=clock_timestamp()-interval '31 seconds',
         updated_at=clock_timestamp()
     WHERE singleton=true AND cutover_state='complete'"
PATH="${cutover_path}:${PATH}" \
EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/legacy-no-pid-reconciled.out" 2>&1
grep -q 'SCHEDULER_LEGACY_NO_PID_RECONCILED.*worker_attempt_id=' \
    "${test_dir}/legacy-no-pid-reconciled.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920064")" = \
    "abandoned:legacy_no_pid_proven_absent_after_cutover"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||(active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment WHERE experiment_id=920064")" = "failed:true"
legacy_terminal_snapshot="$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||completed_at::text
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920064")"
PATH="${cutover_path}:${PATH}" \
EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/legacy-no-pid-replay.out" 2>&1
test "${legacy_terminal_snapshot}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||completed_at::text
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920064")"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920065,'staleanalyzefixture',4,0.0008,20,
    '2020-01-01','2021-01-01','completed','done','analyze',920065
);
INSERT INTO model(model_id,experiment_id,name)
VALUES(925065,920065,'stale-checkpoint-analyze-fixture');
INSERT INTO experiment_checkpoint_eval(
    checkpoint_eval_id,experiment_id,parent_experiment_id,
    checkpoint_epoch,checkpoint_model_id,status,phase
) VALUES(
    925065,920065,920065,20,925065,'running','analyze'
);
INSERT INTO experiment_scheduler_invocation(
    scheduler_invocation_id,process_pid,process_group_id,
    process_start_identity,canonical_executable_path,command_line,
    invocation_nonce,status,protocol_generation
) VALUES(
    'stale-analyze-owner',999991,999991,'stale-start',
    '/missing/LSTM_Release','LSTM_Release --schedule-experiments',
    'stale-analyze-owner-nonce-0001','lost',52
);
WITH attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt(
        launch_attempt_identity,scheduler_invocation_id,
        scheduler_fencing_token,experiment_id,checkpoint_eval_id,
        worker_kind,lifecycle_phase,capacity_class,ownership_origin,
        lifecycle_state,canonical_executable_path,command_line,
        command_identity,reserved_at
    ) VALUES(
        'stale-checkpoint-analyze-attempt-925065',
        'stale-analyze-owner',1,920065,925065,'checkpoint_analyze',
        'analyze','analyze','scheduler_in_process','running',
        '/missing/LSTM_Release','LSTM_Release --schedule-experiments',
        'checkpoint_analyze:925065',clock_timestamp()
    ) RETURNING worker_attempt_id
)
UPDATE experiment_checkpoint_eval
SET active_scheduler_worker_attempt_id=
        (SELECT worker_attempt_id FROM attempt),
    worker_executable='/missing/LSTM_Release',
    worker_command_line='LSTM_Release --schedule-experiments'
WHERE checkpoint_eval_id=925065;
SQL
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/stale-checkpoint-analyze.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result
     FROM experiment_scheduler_worker_attempt
     WHERE checkpoint_eval_id=925065")" = \
    "abandoned:checkpoint_analysis_owner_lost"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||
            (active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment_checkpoint_eval
     WHERE checkpoint_eval_id=925065")" = "pending:analyze:true"

# Production claim/work/finalize orchestration is forced through three
# deterministic interruption boundaries. Each failure leaves one consuming
# exact attempt; a later proven takeover abandons only that attempt and requeues
# the checkpoint analysis for retry.
set +e
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=checkpoint_analysis_crash_after_claim \
EA_SCHEDULER_OWNERSHIP_TEST_FOREIGN_INVOCATION_ID=stale-analyze-owner \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/checkpoint-analyze-crash-after-claim.out" 2>&1
analyze_crash_result=$?
set -e
test "${analyze_crash_result}" = "86"
grep -q 'CHECKPOINT_ANALYSIS_TEST_CRASH_AFTER_CLAIM' \
    "${test_dir}/checkpoint-analyze-crash-after-claim.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE checkpoint_eval_id=925065
       AND worker_kind='checkpoint_analyze'
       AND lifecycle_state='running'")" = "1"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||
            (active_scheduler_worker_attempt_id IS NOT NULL)::text
     FROM experiment_checkpoint_eval
     WHERE checkpoint_eval_id=925065")" = "running:analyze:true"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_lease
     SET expires_at=clock_timestamp()-interval '1 second'
     WHERE singleton=true"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/checkpoint-analyze-crash-recovery.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE checkpoint_eval_id=925065
       AND worker_kind='checkpoint_analyze'
       AND lifecycle_state='abandoned'
       AND reconciliation_result='checkpoint_analysis_owner_lost'")" = "2"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||
            (active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment_checkpoint_eval
     WHERE checkpoint_eval_id=925065")" = "pending:analyze:true"

for analysis_boundary in \
    checkpoint_analysis_lease_loss_during_work \
    checkpoint_analysis_lease_loss_before_finalize
do
    set +e
    EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
    EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY="${analysis_boundary}" \
    EA_SCHEDULER_OWNERSHIP_TEST_FOREIGN_INVOCATION_ID=stale-analyze-owner \
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-analyze-procs=1 \
        --scheduler-log-dir="${test_dir}/logs" \
        >"${test_dir}/${analysis_boundary}.out" 2>&1
    analysis_lease_loss_result=$?
    set -e
    test "${analysis_lease_loss_result}" = "4"
    grep -q 'SCHEDULER_OWNERSHIP_LOST' \
        "${test_dir}/${analysis_boundary}.out"
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT count(*) FROM experiment_scheduler_worker_attempt
         WHERE checkpoint_eval_id=925065
           AND worker_kind='checkpoint_analyze'
           AND lifecycle_state='running'")" = "1"
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --scheduler-status \
        >"${test_dir}/${analysis_boundary}-status.out" 2>&1
    grep -q \
        'SCHEDULER_STATUS_GLOBAL_CAPACITY,capacity_class=analyze,consuming=1' \
        "${test_dir}/${analysis_boundary}-status.out"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
        "UPDATE experiment_scheduler_lease
         SET expires_at=clock_timestamp()-interval '1 second'
         WHERE singleton=true"
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once --recover-orphans-only \
        --scheduler-log-dir="${test_dir}/logs" \
        >"${test_dir}/${analysis_boundary}-recovery.out" 2>&1
    test "$(psql -X -At -d "${test_db}" -c \
        "SELECT status||':'||phase||':'||
                (active_scheduler_worker_attempt_id IS NULL)::text
         FROM experiment_checkpoint_eval
         WHERE checkpoint_eval_id=925065")" = "pending:analyze:true"
done
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE checkpoint_eval_id=925065
       AND worker_kind='checkpoint_analyze'
       AND lifecycle_state='abandoned'
       AND reconciliation_result='checkpoint_analysis_owner_lost'")" = "4"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
UPDATE experiment_checkpoint_eval
SET status='failed',
    phase='done',
    error_message='analysis_recovery_fixture_retired',
    completed_at=clock_timestamp(),
    updated_at=clock_timestamp()
WHERE checkpoint_eval_id=925065
  AND status='pending' AND phase='analyze'
  AND active_scheduler_worker_attempt_id IS NULL;
SQL

launch_worker() {
    local kind="$1"
    local experiment_id="$2"
    local checkpoint_eval_id="${3:-}"
    local worker_attempt_id="${4:-}"
    local worker_link="${test_dir}/LSTM_Release_${kind}"
    local ready="${test_dir}/${kind}.ready"
    local identity="" pid="" pgid="" start="" executable="" command=""
    ln -sf "${process_binary}" "${worker_link}"
    local args=(
        --managed-test-worker --self-session
        --scheduler-experiment-id="${experiment_id}"
        --ready-fd=9
    )
    case "${kind}" in
        train) args+=(--train) ;;
        infer) args+=(--infer) ;;
        analyze) args+=(--analyze-experiment="${experiment_id}") ;;
        checkpoint)
            args+=(--infer --scheduler-checkpoint-eval-id="${checkpoint_eval_id}")
            ;;
    esac
    if [[ -n "${worker_attempt_id}" ]]; then
        args+=(--scheduler-worker-attempt-id="${worker_attempt_id}")
    fi
    "${worker_link}" "${args[@]}" 9>"${ready}" &
    pid=$!
    for _ in {1..100}; do
        identity="$(inspect_process "${pid}" || true)"
        [[ -s "${ready}" && -n "${identity}" ]] && break
        sleep 0.02
    done
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    [[ -n "${pid}" && "${pid}" = "${pgid}" && -n "${start}" &&
       -n "${executable}" && "${command}" == *"--managed-test-worker"* ]]
    worker_pids+=("${pid}")
    worker_pgids+=("${pgid}")
    worker_starts+=("${start}")
    worker_executables+=("${executable}")
}

track_disposable_worker() {
    local target_pid="$1"
    local identity="" pid="" pgid="" start="" executable="" command=""
    identity="$(inspect_process "${target_pid}")"
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    [[ "${pid}" = "${target_pid}" && "${pgid}" = "${target_pid}" &&
       -n "${start}" && -n "${executable}" &&
       "${command}" == *"--scheduler-worker-attempt-id"* ]]
    worker_pids+=("${pid}")
    worker_pgids+=("${pgid}")
    worker_starts+=("${start}")
    worker_executables+=("${executable}")
}

launch_worker train 920051
launch_worker infer 920052
launch_worker analyze 920053
launch_worker checkpoint 920054 925051

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v train_pid="${worker_pids[0]}" \
    -v train_pgid="${worker_pgids[0]}" \
    -v train_start="${worker_starts[0]}" \
    -v train_executable="${worker_executables[0]}" \
    -v infer_pid="${worker_pids[1]}" \
    -v infer_pgid="${worker_pgids[1]}" \
    -v infer_start="${worker_starts[1]}" \
    -v infer_executable="${worker_executables[1]}" \
    -v analyze_pid="${worker_pids[2]}" \
    -v analyze_pgid="${worker_pgids[2]}" \
    -v analyze_start="${worker_starts[2]}" \
    -v analyze_executable="${worker_executables[2]}" \
    -v checkpoint_pid="${worker_pids[3]}" \
    -v checkpoint_pgid="${worker_pgids[3]}" \
    -v checkpoint_start="${worker_starts[3]}" \
    -v checkpoint_executable="${worker_executables[3]}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO model(model_id,name) VALUES(925051,'scheduler-process-fixture');
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    worker_executable,worker_command_line,worker_started_at,duplicate_nonce
) VALUES
    (920051,'fixturetrain',4,0.0008,20,'2020-01-01','2021-01-01',
     'running','train','train',:train_pid,:train_pgid,:'train_start',
     :'train_executable',
     :'train_executable'||' --managed-test-worker --self-session --scheduler-experiment-id=920051 --train',
     clock_timestamp(),920051),
    (920052,'fixtureinfer',4,0.0008,20,'2020-01-01','2021-01-01',
     'running','infer','infer',:infer_pid,:infer_pgid,:'infer_start',
     :'infer_executable',
     :'infer_executable'||' --managed-test-worker --self-session --scheduler-experiment-id=920052 --infer',
     clock_timestamp(),920052),
    (920053,'fixtureanalyze',4,0.0008,20,'2020-01-01','2021-01-01',
     'running','analyze','analyze',:analyze_pid,:analyze_pgid,:'analyze_start',
     :'analyze_executable',
     :'analyze_executable'||' --managed-test-worker --self-session --scheduler-experiment-id=920053 --analyze-experiment=920053',
     clock_timestamp(),920053),
    (920054,'fixturecheckpoint',4,0.0008,20,'2020-01-01','2021-01-01',
     'completed','done',NULL,NULL,NULL,NULL,NULL,NULL,
     clock_timestamp(),920054);
INSERT INTO experiment_checkpoint_eval(
    checkpoint_eval_id,experiment_id,parent_experiment_id,
    checkpoint_epoch,checkpoint_model_id,symbol,prediction_horizon,
    status,phase,worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    started_at,infer_started_at
) VALUES(
    925051,920054,920054,20,925051,'fixturecheckpoint',4,
    'running','infer',:checkpoint_pid,:checkpoint_pgid,
    :'checkpoint_start',:'checkpoint_executable',
    :'checkpoint_executable'||' --managed-test-worker --self-session --scheduler-experiment-id=920054 --infer --scheduler-checkpoint-eval-id=925051',
    clock_timestamp(),clock_timestamp()
);

INSERT INTO experiment_scheduler_worker_attempt(
    launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
    capacity_class,ownership_origin,lifecycle_state,worker_pid,
    worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
)
SELECT
    'legacy:process-test:'||experiment_id::text||':'||phase,
    experiment_id,'experiment',phase,phase,'legacy_unverified',
    'identity_ambiguous',worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    'experiment:'||experiment_id::text||':'||phase,
    worker_started_at,worker_started_at,worker_started_at
FROM experiment WHERE experiment_id BETWEEN 920051 AND 920053;

UPDATE experiment e
SET active_scheduler_worker_attempt_id=a.worker_attempt_id
FROM experiment_scheduler_worker_attempt a
WHERE a.experiment_id=e.experiment_id
  AND a.worker_kind='experiment'
  AND e.experiment_id BETWEEN 920051 AND 920053;

INSERT INTO experiment_scheduler_worker_attempt(
    launch_attempt_identity,experiment_id,checkpoint_eval_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
)
SELECT
    'legacy:process-test:checkpoint:925051',parent_experiment_id,
    checkpoint_eval_id,'checkpoint_infer','infer','infer',
    'legacy_unverified','identity_ambiguous',worker_pid,
    worker_process_group_id,worker_process_start_identity,
    worker_executable,worker_command_line,'checkpoint_infer:925051',
    infer_started_at,infer_started_at,infer_started_at
FROM experiment_checkpoint_eval WHERE checkpoint_eval_id=925051;

UPDATE experiment_checkpoint_eval ce
SET active_scheduler_worker_attempt_id=a.worker_attempt_id
FROM experiment_scheduler_worker_attempt a
WHERE a.checkpoint_eval_id=ce.checkpoint_eval_id
  AND ce.checkpoint_eval_id=925051;
SQL

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=7 --max-infer-procs=5 --max-analyze-procs=1 \
    --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/restart.out" 2>&1
grep -q 'SCHEDULER_PRIOR_WORKER_OBSERVED' "${test_dir}/restart.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE lifecycle_state='observed'")" = "4"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment
     WHERE experiment_id BETWEEN 920051 AND 920053
       AND status='running'")" = "3"

# Real live processes with reused-start, executable, and command/work
# mismatches remain ambiguous, signal-free, lifecycle-bound, and
# capacity-consuming. After the test harness terminates each process by its
# independently captured exact identity, scheduler recovery may safely
# terminalize only that exact attempt.
launch_worker train 920061
launch_worker train 920062
launch_worker train 999999
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v reused_pid="${worker_pids[4]}" \
    -v reused_pgid="${worker_pgids[4]}" \
    -v reused_executable="${worker_executables[4]}" \
    -v executable_pid="${worker_pids[5]}" \
    -v executable_pgid="${worker_pgids[5]}" \
    -v executable_start="${worker_starts[5]}" \
    -v command_pid="${worker_pids[6]}" \
    -v command_pgid="${worker_pgids[6]}" \
    -v command_start="${worker_starts[6]}" \
    -v command_executable="${worker_executables[6]}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    worker_executable,worker_command_line,worker_started_at,duplicate_nonce
) VALUES
    (920061,'pidreusefixture',4,0.0008,20,
     '2020-01-01','2021-01-01','running','train','train',
     :reused_pid,:reused_pgid,'persisted-reused-start',
     :'reused_executable',
     :'reused_executable'||' --train --scheduler-experiment-id=920061',
     clock_timestamp(),920061),
    (920062,'executablemismatchfixture',4,0.0008,20,
     '2020-01-01','2021-01-01','running','train','train',
     :executable_pid,:executable_pgid,:'executable_start',
     '/missing/foreign-executable',
     '/missing/foreign-executable --train --scheduler-experiment-id=920062',
     clock_timestamp(),920062),
    (920063,'commandmismatchfixture',4,0.0008,20,
     '2020-01-01','2021-01-01','running','train','train',
     :command_pid,:command_pgid,:'command_start',
     :'command_executable',
     :'command_executable'||' --train --scheduler-experiment-id=999999',
     clock_timestamp(),920063);
INSERT INTO experiment_scheduler_worker_attempt(
    launch_attempt_identity,experiment_id,worker_kind,lifecycle_phase,
    capacity_class,ownership_origin,lifecycle_state,worker_pid,
    worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,
    reserved_at,spawned_at,registered_at
)
SELECT
    'legacy:mismatched-process-test:'||experiment_id::text,
    experiment_id,'experiment','train','train','legacy_unverified',
    'identity_ambiguous',worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    'experiment:'||experiment_id::text||':train',
    worker_started_at,worker_started_at,worker_started_at
FROM experiment
WHERE experiment_id BETWEEN 920061 AND 920063;
UPDATE experiment e
SET active_scheduler_worker_attempt_id=a.worker_attempt_id
FROM experiment_scheduler_worker_attempt a
WHERE a.experiment_id=e.experiment_id
  AND a.worker_kind='experiment'
  AND e.experiment_id BETWEEN 920061 AND 920063;
SQL
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/mismatched-live-workers.out" 2>&1
test "$(grep -c 'SCHEDULER_WORKER_IDENTITY_AMBIGUOUS' \
    "${test_dir}/mismatched-live-workers.out")" = "3"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id BETWEEN 920061 AND 920063
       AND lifecycle_state='identity_ambiguous'")" = "3"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment
     WHERE experiment_id BETWEEN 920061 AND 920063
       AND status='running'
       AND active_scheduler_worker_attempt_id IS NOT NULL")" = "3"
for mismatch_index in 4 5 6; do
    kill -0 "${worker_pids[${mismatch_index}]}"
    process_identity_matches "${mismatch_index}"
done
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE capacity_class='train'
       AND lifecycle_state IN
       ('reserved','spawned','running','observed','identity_ambiguous')")" = "4"

for mismatch_index in 4 5 6; do
    process_identity_matches "${mismatch_index}"
    kill -TERM -- "-${worker_pgids[${mismatch_index}]}"
    wait "${worker_pids[${mismatch_index}]}"
done
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --recover-orphans-only \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/mismatched-departed-workers.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id BETWEEN 920061 AND 920063
       AND lifecycle_state='failed'
       AND reconciliation_result='process_missing_no_result'")" = "3"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment
     WHERE experiment_id BETWEEN 920061 AND 920063
       AND status='failed'
       AND active_scheduler_worker_attempt_id IS NULL")" = "3"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --dry-run \
    --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/owner.out" 2>&1 &
scheduler_pid=$!
for _ in {1..100}; do
    grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' "${test_dir}/owner.out" &&
        break
    sleep 0.02
done
grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' "${test_dir}/owner.out"
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
test "${observed_pid}" = "${scheduler_pid}"
test "${scheduler_executable}" = "${scheduler_binary}"
[[ "${observed_command}" == *"--schedule-experiments"* ]]

lifecycle_before="$(psql -X -At -d "${test_db}" -c \
    "SELECT string_agg(experiment_id::text||':'||status||':'||phase,',' ORDER BY experiment_id)
     FROM experiment WHERE experiment_id BETWEEN 920051 AND 920054")"
active_attempts_before="$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE lifecycle_state IN
       ('reserved','spawned','running','observed','identity_ambiguous')")"

set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/duplicate.out" 2>&1
duplicate_result=$?
set -e
test "${duplicate_result}" = "3"
grep -q 'SCHEDULER_OWNERSHIP_REJECTED' "${test_dir}/duplicate.out"
grep -q 'mutations=0' "${test_dir}/duplicate.out"
test "${lifecycle_before}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT string_agg(experiment_id::text||':'||status||':'||phase,',' ORDER BY experiment_id)
     FROM experiment WHERE experiment_id BETWEEN 920051 AND 920054")"
test "${active_attempts_before}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE lifecycle_state IN
       ('reserved','spawned','running','observed','identity_ambiguous')")"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --scheduler-status \
    >"${test_dir}/status.out" 2>&1
grep -q 'SCHEDULER_STATUS_OWNERSHIP,authority_state=active' \
    "${test_dir}/status.out"
grep -Fq "canonical_executable_path=${scheduler_binary}" \
    "${test_dir}/status.out"
grep -q 'SCHEDULER_STATUS_GLOBAL_CAPACITY,capacity_class=train,consuming=1' \
    "${test_dir}/status.out"
grep -q 'SCHEDULER_STATUS_GLOBAL_CAPACITY,capacity_class=infer,consuming=2' \
    "${test_dir}/status.out"
grep -q 'SCHEDULER_STATUS_GLOBAL_CAPACITY,capacity_class=analyze,consuming=1' \
    "${test_dir}/status.out"
test "$(grep -c '^SCHEDULER_STATUS_WORKER_ATTEMPT' \
    "${test_dir}/status.out")" = "4"

kill -TERM "${scheduler_pid}"
wait "${scheduler_pid}"
scheduler_pid=""
scheduler_start=""
scheduler_executable=""
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT authority_state FROM experiment_scheduler_lease")" = "released"

# A scheduler whose fencing token is displaced must stop before another poll
# can claim or launch, and its release guard must not release the foreign row.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --dry-run \
    --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/lease-loss-owner.out" 2>&1 &
scheduler_pid=$!
for _ in {1..100}; do
    grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
        "${test_dir}/lease-loss-owner.out" && break
    sleep 0.02
done
grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
    "${test_dir}/lease-loss-owner.out"
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
foreign_invocation="$(psql -X -At -d "${test_db}" -c \
    "SELECT scheduler_invocation_id
     FROM experiment_scheduler_invocation
     WHERE status='rejected'
     ORDER BY started_at DESC LIMIT 1")"
test -n "${foreign_invocation}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v foreign_invocation="${foreign_invocation}" <<'SQL'
UPDATE experiment_scheduler_lease
SET owner_scheduler_invocation_id=:'foreign_invocation',
    fencing_token=fencing_token+1,
    authority_state='active',
    heartbeat_at=clock_timestamp(),
    expires_at=clock_timestamp()+interval '90 seconds',
    transition_reason='process_test_foreign_fence'
WHERE singleton=true;
SQL
set +e
wait "${scheduler_pid}"
lease_loss_result=$?
set -e
scheduler_pid=""
scheduler_start=""
scheduler_executable=""
test "${lease_loss_result}" = "4"
grep -q 'ownership_lost=1' "${test_dir}/lease-loss-owner.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT owner_scheduler_invocation_id||':'||authority_state
     FROM experiment_scheduler_lease")" = \
    "${foreign_invocation}:active"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_lease
     SET authority_state='released',released_at=clock_timestamp(),
         transition_reason='process_test_admin_release'
     WHERE singleton=true"

# Crash recovery is deliberately split into two proofs. A fresh lease rejects
# takeover even after exact owner death, then an expired lease plus the same
# immutable identity mismatch permits a fenced takeover.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --dry-run \
    --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/crash-owner.out" 2>&1 &
scheduler_pid=$!
for _ in {1..100}; do
    grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
        "${test_dir}/crash-owner.out" && break
    sleep 0.02
done
grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' "${test_dir}/crash-owner.out"
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
test "${observed_pid}" = "${scheduler_pid}"
test "${scheduler_executable}" = "${scheduler_binary}"
[[ "${observed_command}" == *"--schedule-experiments"* ]]
crashed_fence="$(psql -X -At -d "${test_db}" -c \
    "SELECT fencing_token FROM experiment_scheduler_lease")"

kill -KILL "${scheduler_pid}"
wait "${scheduler_pid}" 2>/dev/null || true
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/fresh-dead-owner.out" 2>&1
fresh_dead_result=$?
set -e
test "${fresh_dead_result}" = "3"
grep -q 'reason=owner_lease_valid' "${test_dir}/fresh-dead-owner.out"
test "${crashed_fence}" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT fencing_token FROM experiment_scheduler_lease")"

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_lease
     SET expires_at=clock_timestamp()-interval '1 second'
     WHERE singleton=true"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/takeover.out" 2>&1
grep -q 'reason=expired_and_owner_identity_invalid' \
    "${test_dir}/takeover.out"
test "$((crashed_fence + 1))" = "$(psql -X -At -d "${test_db}" -c \
    "SELECT fencing_token FROM experiment_scheduler_lease")"

# Launch one real, disposable worker from a different working directory through
# a basename symlink.  The fixture symbol has no price table, so the child
# registers and exits without doing training work; the assertion is that exec
# used the canonical scheduler executable and never produced the basename/127
# failure seen in production.
analysis_fixture_log="${test_dir}/final-analysis-source.log"
printf '%s\n' \
    'Overall 3-class accuracy: 75.0%' \
    'Overall 3-class confusion matrix [[3, 0, 0], [0, 3, 0], [0, 0, 4]]' \
    'MODEL_ACCEPTANCE ACCEPT_MODEL=true REJECT_REASON=none' \
    >"${analysis_fixture_log}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v analysis_fixture_log="${analysis_fixture_log}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920055,'schedulerlaunchfixture',4,0.0008,1,
    '2020-01-01','2020-02-01','pending','train','train',920055
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920056,'neverlaunchedfixture',4,0.0008,1,
    '2020-01-01','2020-02-01','running','train','train',920056
);
WITH owner AS (
    SELECT owner_scheduler_invocation_id AS scheduler_invocation_id,
           fencing_token
    FROM experiment_scheduler_lease WHERE singleton=true
), attempt AS (
    INSERT INTO experiment_scheduler_worker_attempt(
        launch_attempt_identity,scheduler_invocation_id,
        scheduler_fencing_token,experiment_id,worker_kind,lifecycle_phase,
        capacity_class,ownership_origin,lifecycle_state,
        canonical_executable_path,command_identity,reserved_at
    )
    SELECT
        'process-test-never-spawned-920056',
        scheduler_invocation_id,fencing_token,920056,'experiment','train',
        'train','scheduler_launch','reserved',
        '/tmp/never-spawned','experiment:920056:train',
        clock_timestamp()-interval '1 minute'
    FROM owner
    RETURNING worker_attempt_id
)
UPDATE experiment SET active_scheduler_worker_attempt_id=
    (SELECT worker_attempt_id FROM attempt)
WHERE experiment_id=920056;
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,infer_start,infer_end,
    status,phase,current_operation,duplicate_nonce,last_model_id,
    infer_log_path
) VALUES(
    920058,'finalanalysisfixture',4,0.0008,20,
    '2020-01-01','2020-02-01','2020-03-01','2020-04-01',
    'pending','analyze','analyze',920058,925058,
    :'analysis_fixture_log'
);
INSERT INTO model(model_id,name,experiment_id)
VALUES(925058,'exact-final-analysis-fixture',920058);
SQL
ln -s "${scheduler_binary}" "${test_dir}/SchedulerLaunchAlias"
(
    cd "${test_dir}"
    exec env PATH="${test_dir}:${PATH}" LSTM_DB_NAME="${test_db}" \
        SchedulerLaunchAlias --schedule-experiments \
        --max-train-procs=2 --max-infer-procs=2 \
        --max-analyze-procs=2 --scheduler-poll-seconds=1 \
        --scheduler-log-dir="${test_dir}/launch-logs"
) >"${test_dir}/canonical-launch.out" 2>&1 &
scheduler_pid=$!
for _ in {1..200}; do
    grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
        "${test_dir}/canonical-launch.out" 2>/dev/null && break
    sleep 0.02
done
grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
    "${test_dir}/canonical-launch.out"
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
test "${observed_pid}" = "${scheduler_pid}"
test "${scheduler_executable}" = "${scheduler_binary}"

launch_terminal=false
for _ in {1..300}; do
    launch_state="$(psql -X -At -d "${test_db}" -c \
        "SELECT lifecycle_state
         FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=920055
         ORDER BY worker_attempt_id DESC LIMIT 1")"
    case "${launch_state}" in
        completed|failed|launch_failed|abandoned)
            launch_terminal=true
            break
            ;;
    esac
    sleep 0.05
done
test "${launch_terminal}" = true
analysis_terminal=false
for _ in {1..300}; do
    analysis_state="$(psql -X -At -d "${test_db}" -c \
        "SELECT lifecycle_state
         FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=920058
         ORDER BY worker_attempt_id DESC LIMIT 1")"
    case "${analysis_state}" in
        completed|failed|launch_failed|abandoned)
            analysis_terminal=true
            break
            ;;
    esac
    sleep 0.05
done
if [[ "${analysis_terminal}" != true ]]; then
    sed -n '1,260p' "${test_dir}/canonical-launch.out" >&2
    sed -n '1,260p' \
        "${test_dir}/launch-logs/"*920058* 2>/dev/null >&2 || true
    exit 1
fi
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||
            (active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment WHERE experiment_id=920058")" = \
    "completed:done:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920058")" = \
    "completed:parent_observed_exit"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_analysis_result
     WHERE experiment_id=920058 AND model_id=925058
       AND analysis_status='completed'")" = "1"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||COALESCE(reconciliation_result,'NULL')
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920056")" = \
    "launch_failed:never_spawned"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||(active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment WHERE experiment_id=920056")" = "failed:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920055
       AND ownership_origin='scheduler_launch'
       AND canonical_executable_path='${scheduler_binary}'
       AND command_line LIKE '${scheduler_binary} --train %'
       AND command_line LIKE '%--scheduler-worker-attempt-id=%'
       AND COALESCE(exit_code,-999)<>127")" = "1"
grep -q 'SCHEDULER_CHILD_LAUNCHED.*experiment_id=920055' \
    "${test_dir}/canonical-launch.out"

kill -TERM "${scheduler_pid}"
set +e
wait "${scheduler_pid}"
canonical_scheduler_result=$?
set -e
if [[ "${canonical_scheduler_result}" -ne 0 ]]; then
    sed -n '1,320p' "${test_dir}/canonical-launch.out" >&2
    exit "${canonical_scheduler_result}"
fi
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

# The gated child commits immutable launch identity before it can exec. Force
# the parent to crash only after that exact durable state is observable. The
# gate then closes, the child exits 126 without executing work, and a new
# scheduler recovers only the exact bound attempt and releases its capacity.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920059,'parentcrashfixture',4,0.0008,1.0,1.0,1,1,
    '2020-01-01','2021-01-01','pending','train','train',
    920059
);
SQL
set +e
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=\
scheduler_parent_crash_after_child_registration \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=2 --max-infer-procs=2 \
    --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/parent-crash-logs" \
    >"${test_dir}/parent-crash-after-registration.out" 2>&1
parent_crash_result=$?
set -e
test "${parent_crash_result}" = "87"
grep -q 'SCHEDULER_TEST_PARENT_CRASH_AFTER_CHILD_REGISTRATION' \
    "${test_dir}/parent-crash-after-registration.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state || ':' ||
            (worker_pid IS NOT NULL)::text || ':' ||
            (worker_process_group_id=worker_pid)::text || ':' ||
            (worker_process_start_identity IS NOT NULL)::text
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920059")" = "spawned:true:true:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status || ':' ||
            (active_scheduler_worker_attempt_id IS NOT NULL)::text
     FROM experiment WHERE experiment_id=920059")" = "running:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920059
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "1"
sleep 1
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment_scheduler_lease
     SET expires_at=clock_timestamp()-interval '1 second'
     WHERE singleton=true" >/dev/null
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=2 --max-infer-procs=2 \
    --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/parent-crash-recovery-logs" \
    >"${test_dir}/parent-crash-recovery.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state || ':' || reconciliation_result
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920059")" = \
    "failed:process_missing_no_result"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status || ':' ||
            (active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment WHERE experiment_id=920059")" = "failed:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920059
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "0"

# Replace the lifecycle binding after a real child has exited but immediately
# before the parent reaper performs its exact-attempt verification.  The stale
# reaper must reject its observation and leave the replacement attempt active.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    target_epochs,train_start,train_end,status,phase,current_operation,
    duplicate_nonce
) VALUES(
    920057,'stalereaperfixture',4,0.0008,1,
    '2020-01-01','2020-02-01','pending','train','train',920057
);
SQL
(
    cd "${test_dir}"
    exec env PATH="${test_dir}:${PATH}" \
        LSTM_DB_NAME="${test_db}" \
        EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
        EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=\
stale_reaper_before_exact_verification \
        EA_SCHEDULER_OWNERSHIP_TEST_REPLACEMENT_PID="${worker_pids[0]}" \
        SchedulerLaunchAlias --schedule-experiments \
        --max-train-procs=20 --max-infer-procs=20 \
        --max-analyze-procs=1 --scheduler-poll-seconds=1 \
        --scheduler-log-dir="${test_dir}/stale-reaper-logs"
) >"${test_dir}/stale-reaper.out" 2>&1 &
scheduler_pid=$!
for _ in {1..200}; do
    grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
        "${test_dir}/stale-reaper.out" 2>/dev/null && break
    sleep 0.02
done
grep -q 'SCHEDULER_OWNERSHIP_ACQUIRED' \
    "${test_dir}/stale-reaper.out"
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
test "${observed_pid}" = "${scheduler_pid}"
test "${scheduler_executable}" = "${scheduler_binary}"
for _ in {1..500}; do
    if grep -q \
        'SCHEDULER_STALE_CHILD_REAP_REJECTED.*experiment_id=920057' \
        "${test_dir}/stale-reaper.out" 2>/dev/null; then
        break
    fi
    sleep 0.02
done
grep -q 'SCHEDULER_STALE_CHILD_REAP_REJECTED.*experiment_id=920057' \
    "${test_dir}/stale-reaper.out"
wait "${scheduler_pid}"
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920057
     ORDER BY worker_attempt_id LIMIT 1")" = \
    "abandoned:test_stale_reaper_replaced"
replacement_attempt_id="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_attempt_id
     FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920057
       AND launch_attempt_identity LIKE
           'test-stale-reaper-replacement-%'")"
test -n "${replacement_attempt_id}"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${replacement_attempt_id}")" = \
    "identity_ambiguous"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||
            active_scheduler_worker_attempt_id::text
     FROM experiment WHERE experiment_id=920057")" = \
    "running:train:${replacement_attempt_id}"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920057
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "1"

# Scenario 37: a real generation-52 scheduler launches a train worker whose
# production checkpoint path atomically completes the exact durable attempt.
# The test-only boundary stops both old and next-phase children at explicit OS
# state barriers, so claim and delayed-reap ordering never depends on timing.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_operation,stop_after_checkpoint_epoch,duplicate_nonce
) VALUES(
    920060,'checkpointstopfixture',4,0.0008,1.0,1.0,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'pending','train','train',20,920060
);
INSERT INTO model(experiment_id,name,comment)
VALUES(920060,'checkpoint-stop-fixture-model',
       'periodic training checkpoint');
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT model_id,'train_config_meta',1,14,0,10,20
FROM model WHERE experiment_id=920060;
SQL
scenario37_model_id="$(psql -X -At -d "${test_db}" -c \
    "SELECT model_id FROM model WHERE experiment_id=920060")"
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=\
checkpoint_stop_after_transition_before_exit \
EA_SCHEDULER_OWNERSHIP_TEST_CHECKPOINT_MODEL_ID="${scenario37_model_id}" \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments \
    --max-train-procs=20 --max-infer-procs=20 \
    --max-analyze-procs=2 --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/checkpoint-stop-logs" \
    >"${test_dir}/checkpoint-stop.out" 2>&1 &
scheduler_pid=$!
for _ in {1..500}; do
    scenario37_state="$(psql -X -At -d "${test_db}" -c \
        "SELECT e.status||':'||e.phase||':'||
                COALESCE(e.active_scheduler_worker_attempt_id::text,'NULL')||
                ':'||(SELECT count(*) FROM
                    experiment_scheduler_worker_attempt a
                    WHERE a.experiment_id=e.experiment_id
                      AND a.lifecycle_phase='train'
                      AND a.lifecycle_state='completed')::text
         FROM experiment e WHERE e.experiment_id=920060" 2>/dev/null || true)"
    [[ "${scenario37_state}" == running:infer:*:1 ]] && break
    sleep 0.02
done
[[ "${scenario37_state}" == running:infer:*:1 ]]
scheduler_identity="$(inspect_process "${scheduler_pid}")"
IFS='|' read -r observed_pid observed_pgid scheduler_start \
    scheduler_executable observed_command <<<"${scheduler_identity}"
test "${observed_pid}" = "${scheduler_pid}"
test "${scheduler_executable}" = "${scheduler_binary}"

scenario37_old_attempt="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_attempt_id FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920060 AND lifecycle_phase='train'")"
scenario37_old_pid="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_pid FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_old_attempt}")"
scenario37_new_attempt="$(psql -X -At -d "${test_db}" -c \
    "SELECT active_scheduler_worker_attempt_id FROM experiment
     WHERE experiment_id=920060")"
scenario37_new_pid="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_pid FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_new_attempt}")"
track_disposable_worker "${scenario37_old_pid}"
track_disposable_worker "${scenario37_new_pid}"

test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||reconciliation_result||':'||
            (diagnostic LIKE
             '%checkpoint_epoch=20;checkpoint_model_id=${scenario37_model_id};next_phase=infer;worker_attempt_id=${scenario37_old_attempt}%')::text
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_old_attempt}")" = \
    "completed:checkpoint_stop_completed:true"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT status||':'||phase||':'||current_operation||':'||
            active_scheduler_worker_attempt_id::text||':'||
            (worker_pid=${scenario37_new_pid})::text||':'||
            stopped_at_checkpoint_epoch::text||':'||
            stopped_at_checkpoint_model_id::text
     FROM experiment WHERE experiment_id=920060")" = \
    "running:infer:infer:${scenario37_new_attempt}:true:20:${scenario37_model_id}"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920060 AND capacity_class='train'
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "0"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||capacity_class
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_new_attempt}")" = "running:infer"

kill -CONT -- "-${scenario37_old_pid}"
for _ in {1..500}; do
    grep -q \
        'SCHEDULER_CHECKPOINT_STOP_EXIT_OBSERVED.*experiment_id=920060.*lifecycle_rows_affected=0' \
        "${test_dir}/checkpoint-stop.out" 2>/dev/null && break
    sleep 0.02
done
grep -q \
    'SCHEDULER_CHECKPOINT_STOP_EXIT_OBSERVED.*experiment_id=920060.*lifecycle_rows_affected=0' \
    "${test_dir}/checkpoint-stop.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT active_scheduler_worker_attempt_id::text||':'||status||':'||phase
     FROM experiment WHERE experiment_id=920060")" = \
    "${scenario37_new_attempt}:running:infer"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state||':'||capacity_class
     FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_new_attempt}")" = "running:infer"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920060 AND capacity_class='train'
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "0"

kill -TERM "${scheduler_pid}"
wait "${scheduler_pid}"
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

# Repeat the production checkpoint boundary, then restart the scheduler while
# the old stopped child has not exited. Recovery must retain only the new infer
# binding and must never resurrect or reclassify the completed train attempt.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
SELECT set_config(
    'expertadvisor.scheduler_protocol_generation',
    '52',
    false
);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_operation,stop_after_checkpoint_epoch,duplicate_nonce
) VALUES(
    920066,'checkpointrestartfixture',4,0.0008,1.0,1.0,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'pending','train','train',20,920066
);
INSERT INTO model(experiment_id,name,comment)
VALUES(920066,'checkpoint-restart-fixture-model',
       'periodic training checkpoint');
INSERT INTO matrix(
    model_id,param_name,n_rows,n_cols,row_idx,col_idx,value
)
SELECT model_id,'train_config_meta',1,14,0,10,20
FROM model WHERE experiment_id=920066;
SQL
scenario37_restart_model="$(psql -X -At -d "${test_db}" -c \
    "SELECT model_id FROM model WHERE experiment_id=920066")"
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=\
checkpoint_stop_after_transition_before_exit \
EA_SCHEDULER_OWNERSHIP_TEST_CHECKPOINT_MODEL_ID="${scenario37_restart_model}" \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments \
    --max-train-procs=20 --max-infer-procs=20 \
    --max-analyze-procs=2 --scheduler-poll-seconds=1 \
    --scheduler-log-dir="${test_dir}/checkpoint-restart-logs" \
    >"${test_dir}/checkpoint-restart.out" 2>&1 &
scheduler_pid=$!
for _ in {1..500}; do
    scenario37_restart_state="$(psql -X -At -d "${test_db}" -c \
        "SELECT status||':'||phase||':'||
                COALESCE(active_scheduler_worker_attempt_id::text,'NULL')
         FROM experiment WHERE experiment_id=920066" 2>/dev/null || true)"
    [[ "${scenario37_restart_state}" == running:infer:* ]] && break
    sleep 0.02
done
[[ "${scenario37_restart_state}" == running:infer:* ]]
scenario37_restart_old_attempt="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_attempt_id FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920066 AND lifecycle_phase='train'")"
scenario37_restart_old_pid="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_pid FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_restart_old_attempt}")"
scenario37_restart_new_attempt="$(psql -X -At -d "${test_db}" -c \
    "SELECT active_scheduler_worker_attempt_id FROM experiment
     WHERE experiment_id=920066")"
scenario37_restart_new_pid="$(psql -X -At -d "${test_db}" -c \
    "SELECT worker_pid FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${scenario37_restart_new_attempt}")"
track_disposable_worker "${scenario37_restart_old_pid}"
track_disposable_worker "${scenario37_restart_new_pid}"
kill -TERM "${scheduler_pid}"
wait "${scheduler_pid}"
scheduler_pid=""
scheduler_start=""
scheduler_executable=""

restart_snapshot="$(psql -X -At -d "${test_db}" -c \
    "SELECT e.status||':'||e.phase||':'||
            e.active_scheduler_worker_attempt_id::text||':'||
            old.lifecycle_state||':'||old.reconciliation_result||':'||
            replacement.lifecycle_state
     FROM experiment e
     JOIN experiment_scheduler_worker_attempt old
       ON old.worker_attempt_id=${scenario37_restart_old_attempt}
     JOIN experiment_scheduler_worker_attempt replacement
       ON replacement.worker_attempt_id=
          e.active_scheduler_worker_attempt_id
     WHERE e.experiment_id=920066")"
test "${restart_snapshot}" = \
    "running:infer:${scenario37_restart_new_attempt}:completed:checkpoint_stop_completed:running"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once \
    --max-train-procs=20 --max-infer-procs=20 \
    --max-analyze-procs=2 \
    --scheduler-log-dir="${test_dir}/checkpoint-recovery-logs" \
    >"${test_dir}/checkpoint-recovery.out" 2>&1
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT e.status||':'||e.phase||':'||
            e.active_scheduler_worker_attempt_id::text||':'||
            old.lifecycle_state||':'||old.reconciliation_result||':'||
            replacement.lifecycle_state
     FROM experiment e
     JOIN experiment_scheduler_worker_attempt old
       ON old.worker_attempt_id=${scenario37_restart_old_attempt}
     JOIN experiment_scheduler_worker_attempt replacement
       ON replacement.worker_attempt_id=
          e.active_scheduler_worker_attempt_id
     WHERE e.experiment_id=920066")" = \
    "running:infer:${scenario37_restart_new_attempt}:completed:checkpoint_stop_completed:observed"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920066 AND capacity_class='train'
       AND lifecycle_state IN
           ('reserved','spawned','running','observed',
            'identity_ambiguous')")" = "0"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT count(*) FROM experiment_scheduler_worker_attempt
     WHERE experiment_id=920066")" = "2"

# Exact-attempt administrative reconciliation is intentionally narrower than
# scheduler recovery: it accepts only the one identity_ambiguous attempt whose
# locked experiment binding and live process evidence all still agree. It does
# not signal the disposable worker.
reconcile_attempt_id=990051
launch_worker train 930001 "" "${reconcile_attempt_id}"
reconcile_index=$((${#worker_pids[@]} - 1))
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v attempt_id="${reconcile_attempt_id}" \
    -v pid="${worker_pids[${reconcile_index}]}" \
    -v pgid="${worker_pgids[${reconcile_index}]}" \
    -v start="${worker_starts[${reconcile_index}]}" \
    -v executable="${worker_executables[${reconcile_index}]}" \
    -v worker_link="${test_dir}/LSTM_Release_train" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,target_epochs,
    train_start,train_end,status,phase,current_operation,worker_pid,
    worker_process_group_id,worker_process_start_identity,worker_executable,
    worker_command_line,worker_started_at,duplicate_nonce
) VALUES(
    930001,'reconcilefixture',4,0.0008,20,'2020-01-01','2021-01-01',
    'running','train','train',:pid,:pgid,:'start',:'executable',
    :'worker_link'||' --managed-test-worker --self-session '
      ||'--scheduler-experiment-id=930001 --ready-fd=9 --train '
      ||'--scheduler-worker-attempt-id=990051',clock_timestamp(),930001
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
) SELECT :attempt_id,'test:reconcile:990051',930001,'experiment','train',
    'train','scheduler_launch','identity_ambiguous',worker_pid,
    worker_process_group_id,worker_process_start_identity,worker_executable,
    worker_command_line,'experiment:930001:train',worker_started_at,
    worker_started_at,worker_started_at
  FROM experiment WHERE experiment_id=930001;
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=930001;
SQL
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${reconcile_attempt_id}" --dry-run \
    >"${test_dir}/reconcile-dry-run.out" 2>&1
grep -q 'WORKER_ATTEMPT_RECONCILIATION,outcome=eligible' \
    "${test_dir}/reconcile-dry-run.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${reconcile_attempt_id}")" = "identity_ambiguous"
process_identity_matches "${reconcile_index}"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${reconcile_attempt_id}" --yes \
    >"${test_dir}/reconcile-apply.out" 2>&1
grep -q 'WORKER_ATTEMPT_RECONCILIATION,outcome=applied' \
    "${test_dir}/reconcile-apply.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt
     WHERE worker_attempt_id=${reconcile_attempt_id}")" = "observed"
process_identity_matches "${reconcile_index}"

# Exact-attempt reconciliation has a separate, positive-absence branch.
absent_reconcile_attempt_id=990052
launch_worker train 930002 "" "${absent_reconcile_attempt_id}"
absent_reconcile_index=$((${#worker_pids[@]} - 1))
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v attempt_id="${absent_reconcile_attempt_id}" \
    -v pid="${worker_pids[${absent_reconcile_index}]}" \
    -v pgid="${worker_pgids[${absent_reconcile_index}]}" \
    -v start="${worker_starts[${absent_reconcile_index}]}" \
    -v executable="${worker_executables[${absent_reconcile_index}]}" \
    -v worker_link="${test_dir}/LSTM_Release_train" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,target_epochs,
    train_start,train_end,status,phase,current_operation,worker_pid,
    worker_process_group_id,worker_process_start_identity,worker_executable,
    worker_command_line,worker_started_at,duplicate_nonce
) VALUES(
    930002,'absentreconcilefixture',4,0.0008,20,'2020-01-01','2021-01-01',
    'running','train','train',:pid,:pgid,:'start',:'executable',
    :'worker_link'||' --managed-test-worker --self-session '
      ||'--scheduler-experiment-id=930002 --ready-fd=9 --train '
      ||'--scheduler-worker-attempt-id=990052',clock_timestamp(),930002
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,reserved_at,
    spawned_at,registered_at
) SELECT :attempt_id,'test:reconcile:990052',930002,'experiment','train',
    'train','scheduler_launch','identity_ambiguous',worker_pid,
    worker_process_group_id,worker_process_start_identity,worker_executable,
    worker_command_line,'experiment:930002:train',worker_started_at,
    worker_started_at,worker_started_at
  FROM experiment WHERE experiment_id=930002;
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=930002;
SQL
process_identity_matches "${absent_reconcile_index}"
kill -TERM -- "-${worker_pgids[${absent_reconcile_index}]}"
wait "${worker_pids[${absent_reconcile_index}]}"
test -z "$(ps -p "${worker_pids[${absent_reconcile_index}]}" -o pid=)"
absent_before="$(psql -X -At -d "${test_db}" -c "SELECT a.lifecycle_state||':'||COALESCE(a.reconciliation_result,'NULL')||':'||e.status||':'||(e.active_scheduler_worker_attempt_id IS NOT NULL)::text FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id WHERE a.worker_attempt_id=${absent_reconcile_attempt_id}")"
test "${absent_before}" = "identity_ambiguous:NULL:running:true"
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --reconcile-worker-attempt="${absent_reconcile_attempt_id}" --dry-run >"${test_dir}/reconcile-absent-dry-run.out" 2>&1
grep -q 'WORKER_ATTEMPT_RECONCILIATION,outcome=eligible,worker_attempt_id=990052' "${test_dir}/reconcile-absent-dry-run.out"
grep -q 'process_presence=absent' "${test_dir}/reconcile-absent-dry-run.out"
grep -q 'proposed_lifecycle_state=failed' "${test_dir}/reconcile-absent-dry-run.out"
grep -q 'reason=exact_process_absent_no_result' "${test_dir}/reconcile-absent-dry-run.out"
test "${absent_before}" = "$(psql -X -At -d "${test_db}" -c "SELECT a.lifecycle_state||':'||COALESCE(a.reconciliation_result,'NULL')||':'||e.status||':'||(e.active_scheduler_worker_attempt_id IS NOT NULL)::text FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id WHERE a.worker_attempt_id=${absent_reconcile_attempt_id}")"

# A stale lifecycle identity is rejected before either guarded mutation; it
# must not produce a false applied result or a partial terminalization.
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET worker_command_line=worker_command_line || ' --stale'
     WHERE experiment_id=930002"
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${absent_reconcile_attempt_id}" --yes \
    >"${test_dir}/reconcile-absent-stale.out" 2>&1
absent_stale_result=$?
set -e
test "${absent_stale_result}" = "1"
grep -q 'outcome=rejected' "${test_dir}/reconcile-absent-stale.out"
grep -q 'exact_ambiguous_attempt_verification_failed' \
    "${test_dir}/reconcile-absent-stale.out"
! grep -q 'outcome=applied' "${test_dir}/reconcile-absent-stale.out"
test "$(psql -X -At -d "${test_db}" -c \
    "SELECT a.lifecycle_state||':'||COALESCE(a.reconciliation_result,'NULL')||':'||e.status||':'||(e.active_scheduler_worker_attempt_id IS NULL)::text
     FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id
     WHERE a.worker_attempt_id=${absent_reconcile_attempt_id}")" = \
    "identity_ambiguous:NULL:running:false"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET worker_command_line=replace(worker_command_line, ' --stale', '')
     WHERE experiment_id=930002"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" --reconcile-worker-attempt="${absent_reconcile_attempt_id}" --yes >"${test_dir}/reconcile-absent-apply.out" 2>&1
test "$(awk '/outcome=applying/{print NR; exit}' "${test_dir}/reconcile-absent-apply.out")" -lt \
    "$(awk '/outcome=applied/{print NR; exit}' "${test_dir}/reconcile-absent-apply.out")"
grep -q 'WORKER_ATTEMPT_RECONCILIATION,outcome=applied,worker_attempt_id=990052' "${test_dir}/reconcile-absent-apply.out"
test "$(psql -X -At -d "${test_db}" -c "SELECT a.lifecycle_state||':'||(a.completed_at IS NOT NULL)::text||':'||a.exit_code||':'||a.reconciliation_result||':'||a.diagnostic||':'||e.status||':'||e.phase||':'||(e.worker_pid IS NULL)::text||':'||(e.worker_process_group_id IS NULL)::text||':'||(e.active_scheduler_worker_attempt_id IS NULL)::text||':'||(e.completed_at IS NOT NULL)::text||':'||e.exit_code||':'||e.error_message FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id WHERE a.worker_attempt_id=${absent_reconcile_attempt_id}")" = "failed:true:-1:process_missing_no_result:exact_process_identity_absent_no_result:failed:train:true:true:true:true:-1:worker_process_missing_no_result"
test "$(psql -X -At -d "${test_db}" -c "SELECT a.lifecycle_state||':'||COALESCE(a.reconciliation_result,'NULL')||':'||e.status||':'||(e.active_scheduler_worker_attempt_id IS NOT NULL)::text FROM experiment_scheduler_worker_attempt a JOIN experiment e ON e.experiment_id=a.experiment_id WHERE a.worker_attempt_id=${reconcile_attempt_id}")" = "observed:valid_process_observed:running:true"

# A state other than the one explicitly authorized source state is rejected;
# it is not treated as an idempotent administrative rewrite.
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --reconcile-worker-attempt="${reconcile_attempt_id}" --dry-run \
    >"${test_dir}/reconcile-nonambiguous.out" 2>&1
reconcile_nonambiguous_status=$?
set -e
test "${reconcile_nonambiguous_status}" = "1"
grep -q 'exact_ambiguous_attempt_verification_failed' \
    "${test_dir}/reconcile-nonambiguous.out"
process_identity_matches "${reconcile_index}"

printf '%s\n' "SchedulerOwnershipProcessIntegrationTests passed"
