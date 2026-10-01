#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/LSTM_Release [semantic-worker-registry]}"
scheduler_binary="$(cd "$(dirname "${scheduler_binary}")" && pwd)/$(basename "${scheduler_binary}")"
semantic_worker_registry="${2:-${repo_root}/Builds/SemanticWorkers/registry.json}"
semantic_worker_registry="$(cd "$(dirname "${semantic_worker_registry}")" && pwd)/$(basename "${semantic_worker_registry}")"
test -x "${scheduler_binary}"
test -f "${semantic_worker_registry}"

layout10_worker="$(/usr/bin/python3 - "${semantic_worker_registry}" <<'PY'
import json
from pathlib import Path
import sys

path = Path(sys.argv[1])
registry = json.loads(path.read_text(encoding="utf-8"))
assert registry["current_layout"] == 10
workers = registry["workers"]
layout9 = [worker for worker in workers
           if worker["semantic_layout"] == 9
           and worker["model_input_width"] == 103
           and "train" in worker["capabilities"]]
layout10 = [worker for worker in workers
            if worker["semantic_layout"] == 10
            and worker["model_input_width"] == 114
            and "train" in worker["capabilities"]
            and "train_feature_ablation_v1" in worker["capabilities"]]
assert layout9
assert len(layout10) == 1
assert not any(worker["semantic_layout"] == 9
               and worker["model_input_width"] == 114
               for worker in workers)
print((path.parent / layout10[0]["executable"]).resolve(strict=True))
PY
)"

test_db="ea_scheduler_semantic_rollover_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-scheduler-semantic-rollover.XXXXXX")"
process_binary="${test_dir}/GlobalExperimentControlProcessTests"
worker_pids=()
worker_pgids=()

cleanup() {
    local status=$? index pid pgid command
    for index in "${!worker_pids[@]}"; do
        pid="${worker_pids[${index}]}"
        pgid="${worker_pgids[${index}]}"
        if kill -0 "${pid}" >/dev/null 2>&1; then
            command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
            if [[ ( "${command}" == *"${process_binary}"* ||
                    "${command}" == *"${test_dir}/LSTM_Release_"* ) &&
                  "${command}" == *'--scheduler-observed-worker-fixture'* &&
                  "${command}" == *'--scheduler-experiment-id=9951'* ]]; then
                kill -CONT -- "-${pgid}" >/dev/null 2>&1 || true
                kill -TERM -- "-${pgid}" >/dev/null 2>&1 || true
            else
                printf '%s\n' \
                    "refusing to signal unverified fixture pid ${pid}: ${command}" \
                    >&2
                continue
            fi
        fi
        wait "${pid}" 2>/dev/null || true
    done
    if [[ "${status}" -ne 0 ]]; then
        for output in "${test_dir}"/*.out; do
            [[ -f "${output}" ]] || continue
            tail -160 "${output}" >&2 || true
        done
    fi
    case "${test_db}" in
        ea_scheduler_semantic_rollover_[0-9]*)
            dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
            ;;
    esac
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
    return "${status}"
}
trap cleanup EXIT

createdb "${test_db}"
# LSTM is used only as a read-only schema source. All rows and scheduler
# authority below live in the disposable database named above.
PGOPTIONS='-c default_transaction_read_only=on' \
    pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
for migration in \
    094_fresh_initialization_seed.sql \
    095_fresh_initialization_seed_identity.sql \
    097_resume_seed_conditional_identity.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done

psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE
SET desired_state='running',active_request_id=NULL,current_pause_request_id=NULL;
INSERT INTO experiment_scheduler_lease(singleton)
VALUES(true) ON CONFLICT(singleton) DO NOTHING;
INSERT INTO experiment_scheduler_protocol(
    singleton,required_generation,cutover_state,cutover_completed_at,
    cutover_completed_by,cutover_executable_path,cutover_process_evidence
) VALUES(
    true,52,'complete',clock_timestamp(),
    'scheduler-semantic-rollover-integration','/test/LSTM_Release',
    'disposable-test-database'
) ON CONFLICT(singleton) DO NOTHING;
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='scheduler-semantic-rollover-integration',
    cutover_executable_path='/test/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;
SQL

"${CXX:-clang++}" -std=c++20 -O0 -g -Wall -Wextra -Werror \
    "${repo_root}/Tests/SchedulerObservedWorkerFixture.cpp" \
    -o "${process_binary}"

scalar() { psql -X -Atq -d "${test_db}" -c "$1"; }

launch_observed_layout9_worker() {
    local experiment_id="$1" attempt_id="$2"
    local worker_link="${test_dir}/LSTM_Release_${experiment_id}"
    local ready="${test_dir}/${experiment_id}.ready"
    local identity='' pid='' pgid='' start='' executable='' command=''
    ln -sf "${process_binary}" "${worker_link}"
    "${worker_link}" --scheduler-observed-worker-fixture \
        --train --scheduler-experiment-id="${experiment_id}" \
        --scheduler-worker-attempt-id="${attempt_id}" \
        --ready-file="${ready}" &
    local launched_pid=$!
    for _ in {1..150}; do
        if [[ -s "${ready}" ]]; then
            identity="$(sed -n '1p' "${ready}")"
            [[ -n "${identity}" ]] && break
        fi
        sleep 0.02
    done
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    test "${pid}" = "${launched_pid}"
    test "${pgid}" = "${launched_pid}"
    worker_pids+=("${pid}")
    worker_pgids+=("${pgid}")

    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v attempt_id="${attempt_id}" \
        -v pid="${pid}" -v pgid="${pgid}" -v start="${start}" \
        -v executable="${executable}" -v command="${command}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_operation,duplicate_nonce,scheduler_priority,resume_requested,
    scheduler_resume_origin,worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    worker_control_state,worker_started_at,feature_ablation_mask,
    model_input_width,model_input_semantic_layout_version
) VALUES(
    :experiment_id,'eurusdrmp',1,0.0008,1,1,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'running','train','train',:experiment_id,'normal',false,'none',
    :pid,:pgid,:'start',:'executable',:'command','running',clock_timestamp(),
    '',103,9
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,
    semantic_layout_version,model_input_width,semantic_worker_role,
    reserved_at,spawned_at,registered_at
) VALUES(
    :attempt_id,'semantic-rollover-'||:attempt_id,:experiment_id,'experiment',
    'train','train','prior_scheduler_observed','observed',
    :pid,:pgid,:'start',:'executable',:'command',
    'experiment:'||:experiment_id||':train',9,103,'train',
    clock_timestamp(),clock_timestamp(),clock_timestamp()
);
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=:experiment_id;
SQL
}

retire_observed_worker() {
    local index="$1" experiment_id="$2" attempt_id="$3"
    local pid="${worker_pids[${index}]}" pgid="${worker_pgids[${index}]}"
    if kill -0 "${pid}" >/dev/null 2>&1; then
        kill -TERM -- "-${pgid}"
        wait "${pid}" 2>/dev/null || true
    fi
    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
UPDATE experiment
SET status='failed',active_scheduler_worker_attempt_id=NULL,worker_pid=NULL,
    worker_process_group_id=NULL,worker_process_start_identity=NULL,
    worker_executable=NULL,worker_command_line=NULL,
    completed_at=clock_timestamp(),exit_code=17,
    error_message='disposable rollover fixture retired'
WHERE experiment_id=${experiment_id};
UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='completed',completed_at=clock_timestamp(),exit_code=17
WHERE worker_attempt_id=${attempt_id};
SQL
}

run_scheduler() {
    local output="$1"
    shift
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-train-procs=2 --max-infer-procs=0 --max-analyze-procs=0 \
        --semantic-worker-registry="${semantic_worker_registry}" \
        --scheduler-log-dir="${test_dir}/logs" "$@" >"${output}" 2>&1
}

canonical_pocket_mask='pocket_recent_price_scale_valid,pocket_bull_recent_count_log,pocket_bull_youngest_age20,pocket_bull_median_touch_distance,pocket_bull_median_close_distance,pocket_bull_median_width,pocket_bear_recent_count_log,pocket_bear_youngest_age20,pocket_bear_median_touch_distance,pocket_bear_median_close_distance,pocket_bear_median_width'

launch_observed_layout9_worker 995100 10995100
launch_observed_layout9_worker 995101 10995101
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -v pocket_mask="${canonical_pocket_mask}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_operation,duplicate_nonce,scheduler_priority,resume_requested,
    scheduler_resume_origin,worker_control_state,feature_ablation_mask,
    model_input_width,model_input_semantic_layout_version,updated_at
) VALUES(
    995102,'gbpusdrmp',1,0.0008,1,1,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    'pending','train','train',995102,'normal',false,'none','running',
    :'pocket_mask',114,10,clock_timestamp()
);
SQL

test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE capacity_class='train' AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = 2
run_scheduler "${test_dir}/full.out"
grep -Fq "SCHEDULER_TRAIN_WORKER_SELECTED,experiment_id=995102,model_input_width=114,model_input_semantic_layout_version=10,worker_input_width=114,worker_semantic_layout_version=10,worker_executable=${layout10_worker},reason=current_published_semantic_worker" \
    "${test_dir}/full.out"
grep -q 'SCHEDULER_PREEMPTION_DEFERRED,candidate_experiment_id=995102,candidate_priority=normal,phase=train,reason=no_strictly_lower_pause_safe_victim' \
    "${test_dir}/full.out"
grep -q 'SCHEDULER_SKIP_TRAIN,experiment_id=995102,reason=global_train_slots_full' \
    "${test_dir}/full.out"
! grep -q 'semantic_worker_incompatible' "${test_dir}/full.out"
! grep -q 'FEATURE_ABLATION_MASK_UNKNOWN_FEATURE' "${test_dir}/full.out"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=995102")" = pending
test "$(scalar "SELECT count(*) FROM experiment WHERE experiment_id IN (995100,995101) AND status='running'")" = 2

# Free exactly one disposable fixture slot, then use scheduler dry-run so the
# real Layout-10 semantic worker is selected and its launch command is built
# without starting training alongside live experiments.
retire_observed_worker 0 995100 10995100
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE capacity_class='train' AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = 1
run_scheduler "${test_dir}/available.out" --dry-run
grep -Fq "SCHEDULER_TRAIN_WORKER_SELECTED,experiment_id=995102,model_input_width=114,model_input_semantic_layout_version=10,worker_input_width=114,worker_semantic_layout_version=10,worker_executable=${layout10_worker},reason=current_published_semantic_worker" \
    "${test_dir}/available.out"
grep -q 'EXPERIMENT_CHILD_COMMAND,experiment_id=995102,phase=train,dry_run=1,argv=.*--scheduler-experiment-id.*995102' \
    "${test_dir}/available.out"
! grep -q 'semantic_worker_incompatible' "${test_dir}/available.out"
! grep -q 'FEATURE_ABLATION_MASK_UNKNOWN_FEATURE' "${test_dir}/available.out"
test "$(scalar "SELECT feature_ablation_mask FROM experiment WHERE experiment_id=995102")" = "${canonical_pocket_mask}"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=995101")" = running
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=995102")" = pending

printf '%s\n' 'SchedulerSemanticRolloverIntegrationTests passed'
