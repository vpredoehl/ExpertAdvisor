#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:-${repo_root}/DerivedData/Development/SchedulerRecoveryRequeuePreemption/Build/Products/Debug/LSTM_Release}"
test_db="ea_scheduler_preemption_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-scheduler-preemption.XXXXXX")"
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
            kill -CONT -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
            kill -TERM -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
        fi
        wait "${worker_pids[${index}]}" 2>/dev/null || true
    done
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

diagnose_error() {
    local result=$?
    for diagnostic in "${test_dir}"/*.out; do
        [[ -f "${diagnostic}" ]] || continue
        printf '%s\n' "--- ${diagnostic} ---" >&2
        tail -100 "${diagnostic}" >&2 || true
    done
    psql -X -Atq -d "${test_db}" -c \
        "SELECT e.experiment_id,e.status,e.phase,e.scheduler_priority,
                e.resume_requested,e.scheduler_resume_origin,
                e.active_scheduler_worker_attempt_id,a.lifecycle_state
         FROM experiment e LEFT JOIN experiment_scheduler_worker_attempt a
           ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id
         WHERE e.experiment_id BETWEEN 994000 AND 994999
         ORDER BY e.experiment_id" >&2 || true
    return "${result}"
}
trap diagnose_error ERR

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
for migration in \
    051_scheduler_ownership_and_worker_attempts.sql \
    052_scheduler_protocol_and_exact_attempt_hardening.sql \
    071_resume_input_width_expansion.sql \
    078_operator_forced_final_inference_rerun.sql \
    086_scheduler_pause_resume_priority.sql \
    093_scheduler_priority_preemption.sql; do
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/${migration}"
done
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE
SET desired_state='running',active_request_id=NULL,current_pause_request_id=NULL;
UPDATE experiment_scheduler_protocol
SET required_generation=52,cutover_state='complete',
    cutover_completed_at=clock_timestamp(),
    cutover_completed_by='scheduler-priority-preemption-integration',
    cutover_executable_path='/test/LSTM_Release',
    cutover_process_evidence='disposable-test-database',
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

scalar() { psql -X -Atq -d "${test_db}" -c "$1"; }

run_scheduler() {
    local train_capacity="$1" infer_capacity="$2" output="$3"
    shift 3
    LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
        --schedule-experiments --scheduler-once \
        --max-train-procs="${train_capacity}" \
        --max-infer-procs="${infer_capacity}" --max-analyze-procs=0 \
        --scheduler-log-dir="${test_dir}/logs" "$@" >"${output}" 2>&1
}

run_cli() { LSTM_DB_NAME="${test_db}" "${scheduler_binary}" "$@"; }

wait_for_state() {
    local pid="$1" prefix="$2" state=""
    for _ in {1..150}; do
        state="$(ps -o state= -p "${pid}" 2>/dev/null | tr -d ' ' || true)"
        [[ "${state}" == "${prefix}"* ]] && return 0
        sleep 0.02
    done
    return 1
}

launch_and_persist() {
    local experiment_id="$1" attempt_id="$2" phase="$3" status="$4"
    local priority="$5" origin="$6" started_at="$7"
    local worker_link="${test_dir}/LSTM_Release_${experiment_id}"
    local ready="${test_dir}/${experiment_id}.ready"
    local identity="" pid="" pgid="" start="" executable="" command=""
    local state="running" resume=false control=running
    [[ "${status}" = pending ]] && state=stopped && resume=true && control=paused
    ln -sf "${process_binary}" "${worker_link}"
    "${worker_link}" --managed-test-worker --self-session \
        --"${phase}" --scheduler-experiment-id="${experiment_id}" \
        --scheduler-worker-attempt-id="${attempt_id}" --ready-fd=9 \
        9>"${ready}" &
    local launched_pid=$!
    for _ in {1..150}; do
        identity="$("${process_binary}" \
            --inspect-managed-test-process="${launched_pid}" 2>/dev/null || true)"
        [[ -s "${ready}" && -n "${identity}" ]] && break
        sleep 0.02
    done
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    test "${pid}" = "${launched_pid}"
    test "${pgid}" = "${launched_pid}"
    if [[ "${state}" = stopped ]]; then
        kill -STOP -- "-${pgid}"
        wait_for_state "${pid}" T
    fi
    worker_pids+=("${pid}")
    worker_pgids+=("${pgid}")
    worker_starts+=("${start}")
    worker_executables+=("${executable}")
    worker_commands+=("${command}")
    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v attempt_id="${attempt_id}" \
        -v phase="${phase}" -v status="${status}" -v priority="${priority}" \
        -v origin="${origin}" -v resume="${resume}" -v control="${control}" \
        -v lifecycle="${state}" -v started_at="${started_at}" \
        -v pid="${pid}" -v pgid="${pgid}" -v start="${start}" \
        -v executable="${executable}" -v command="${command}" <<'SQL'
INSERT INTO experiment(
    experiment_id,symbol,prediction_horizon,c_next_threshold,
    core_lr_mult,head_lr_mult,target_epochs,checkpoint_interval,
    train_start,train_end,infer_start,infer_end,status,phase,
    current_operation,duplicate_nonce,scheduler_priority,resume_requested,
    scheduler_resume_origin,worker_pid,worker_process_group_id,
    worker_process_start_identity,worker_executable,worker_command_line,
    worker_control_state,worker_started_at,model_input_width,
    model_input_semantic_layout_version
) VALUES(
    :experiment_id,'preempt'||:experiment_id,1,0.0008,1,1,80,20,
    '2020-01-01','2021-01-01','2021-01-01','2022-01-01',
    :'status',:'phase',:'phase',:experiment_id,:'priority',:resume,
    :'origin',:pid,:pgid,:'start',:'executable',:'command',:'control',
    :'started_at'::timestamptz,75,5
);
INSERT INTO experiment_scheduler_worker_attempt(
    worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
    lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
    worker_pid,worker_process_group_id,worker_process_start_identity,
    canonical_executable_path,command_line,command_identity,
    spawned_at,registered_at
) VALUES(
    :attempt_id,'preemption-'||:attempt_id,:experiment_id,'experiment',
    :'phase',:'phase','prior_scheduler_observed',:'lifecycle',
    :pid,:pgid,:'start',:'executable',:'command',
    'experiment:'||:experiment_id||':'||:'phase',
    :'started_at'::timestamptz,:'started_at'::timestamptz
);
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=:experiment_id;
SQL
}

insert_checkpoint_model() {
    local experiment_id="$1" symbol="$2" completed_epochs="$3"
    local model_id
    model_id="$(psql -X -Atq -d "${test_db}" -c \
        "INSERT INTO model(experiment_id,name,comment)
         VALUES(${experiment_id},'preemption-${experiment_id}-${completed_epochs}',
                'periodic training checkpoint') RETURNING model_id")"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<SQL
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_config_meta',1,14,0,col_idx,
       CASE col_idx WHEN 0 THEN 1 WHEN 1 THEN 1 WHEN 2 THEN 0.0008
           WHEN 10 THEN ${completed_epochs} WHEN 11 THEN 1.0
           WHEN 12 THEN 1.0 WHEN 13 THEN 1.0 ELSE 1.0 END
FROM generate_series(0,13) AS columns(col_idx);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_symbol_meta',1,length('${symbol}'),0,position-1,
       ascii(substr('${symbol}',position,1))
FROM generate_series(1,length('${symbol}')) AS chars(position);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'train_range_meta',1,length('2020-01-01|2021-01-01'),0,
       position-1,ascii(substr('2020-01-01|2021-01-01',position,1))
FROM generate_series(1,length('2020-01-01|2021-01-01')) AS chars(position);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'model_meta',1,3,0,i-1,v[i]
FROM (SELECT ARRAY[1.0,75.0,1.0] v) data,generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'param',76,4,(i-1)/4,(i-1)%4,0.0
FROM generate_series(1,304) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'bias',1,4,0,i-1,0.0 FROM generate_series(1,4) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
VALUES
    (${model_id},'returnHeadWeight',1,1,0,0,0.0),
    (${model_id},'returnHeadBias',1,1,0,0,0.0);
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'returnHeadDirWeight',1,3,0,i-1,0.0
FROM generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'returnHeadDirBias',1,3,0,i-1,0.0
FROM generate_series(1,3) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'target_meta',1,6,0,i-1,v[i]
FROM (SELECT ARRAY[2.0,1.0,0.0,0.0,0.0,1.0] v) data,
     generate_series(1,6) i;
INSERT INTO matrix(model_id,param_name,n_rows,n_cols,row_idx,col_idx,value)
SELECT ${model_id},'optimizer_meta',1,5,0,i-1,v[i]
FROM (SELECT ARRAY[1.0,1.0,${completed_epochs}::double precision,0.0,0.0] v) data,
     generate_series(1,5) i;
SQL
    printf '%s\n' "${model_id}"
}

retire_all() {
    local index
    for index in "${!worker_pids[@]}"; do
        if kill -0 "${worker_pids[${index}]}" >/dev/null 2>&1; then
            kill -CONT -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
            kill -TERM -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
        fi
        wait "${worker_pids[${index}]}" 2>/dev/null || true
    done
    PGOPTIONS='-c expertadvisor.scheduler_protocol_generation=52' \
        psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
UPDATE experiment
SET status='failed',resume_requested=false,scheduler_resume_origin='none',
    worker_pid=NULL,worker_process_group_id=NULL,
    worker_process_start_identity=NULL,worker_executable=NULL,
    worker_command_line=NULL,active_scheduler_worker_attempt_id=NULL,
    worker_control_state='running',completed_at=clock_timestamp()
WHERE experiment_id BETWEEN 994000 AND 994999
  AND active_scheduler_worker_attempt_id IS NOT NULL;
UPDATE experiment_scheduler_worker_attempt
SET lifecycle_state='abandoned',completed_at=clock_timestamp()
WHERE experiment_id BETWEEN 994000 AND 994999
  AND lifecycle_state IN ('reserved','spawned','running','observed','stopped',
                          'identity_ambiguous');
SQL
    worker_pids=()
    worker_pgids=()
    worker_starts=()
    worker_executables=()
    worker_commands=()
}

assert_preemption_pair() {
    local base="$1" victim_priority="$2" candidate_priority="$3" phase="$4"
    local victim="${base}" candidate="$((base + 1))" output="${test_dir}/${base}.out"
    launch_and_persist "${victim}" "$((victim + 9000000))" "${phase}" \
        running "${victim_priority}" none '2026-03-01 00:00:00+00'
    launch_and_persist "${candidate}" "$((candidate + 9000000))" "${phase}" \
        pending "${candidate_priority}" operator '2026-03-01 00:00:01+00'
    if [[ "${phase}" = train ]]; then
        run_scheduler 1 0 "${output}"
    else
        run_scheduler 0 1 "${output}"
    fi
    grep -q "SCHEDULER_PRIORITY_PREEMPTED,candidate_experiment_id=${candidate}.*victim_experiment_id=${victim}" "${output}"
    grep -q "SCHEDULER_STOPPED_WORKER_ADMITTED,experiment_id=${candidate}" "${output}"
    test "$(scalar "SELECT status||':'||resume_requested::text||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=${victim}")" = 'pending:true:preemption'
    test "$(scalar "SELECT status||':'||resume_requested::text||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=${candidate}")" = 'running:false:none'
    wait_for_state "${worker_pids[0]}" T
    [[ "$(ps -o state= -p "${worker_pids[1]}" | tr -d ' ')" != T* ]]
    retire_all
}

assert_no_equal_preemption() {
    local base="$1" priority="$2"
    local output="${test_dir}/${base}.out"
    launch_and_persist "${base}" "$((base + 9000000))" train running \
        "${priority}" none '2026-03-02 00:00:00+00'
    launch_and_persist "$((base + 1))" "$((base + 9000001))" train pending \
        "${priority}" operator '2026-03-02 00:00:01+00'
    run_scheduler 1 0 "${output}"
    ! grep -q 'SCHEDULER_PRIORITY_PREEMPTED' "${output}"
    test "$(scalar "SELECT status FROM experiment WHERE experiment_id=${base}")" = running
    test "$(scalar "SELECT status FROM experiment WHERE experiment_id=$((base + 1))")" = pending
    [[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" != T* ]]
    retire_all
}

# C1-C3 and capacity-class locality.
assert_preemption_pair 994010 low high train
assert_preemption_pair 994020 normal high train
assert_preemption_pair 994030 low normal train
assert_preemption_pair 994040 low high infer

# C4-C6: equal priority never preempts.
assert_no_equal_preemption 994050 high
assert_no_equal_preemption 994060 normal
assert_no_equal_preemption 994070 low

# C8/C19: one high request selects exactly one low victim ahead of normal.
launch_and_persist 994080 9994080 train running normal none '2026-03-03 00:00:00+00'
launch_and_persist 994081 9994081 train running low none '2026-03-03 00:00:01+00'
launch_and_persist 994082 9994082 train pending high operator '2026-03-03 00:00:02+00'
run_scheduler 2 0 "${test_dir}/victim-priority.out"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=994080")" = running
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=994081")" = pending
test "$(scalar "SELECT count(*) FROM experiment WHERE experiment_id IN (994080,994081) AND scheduler_resume_origin='preemption'")" = 1
retire_all

# C9: newest durable worker_started_at wins within the same victim priority.
launch_and_persist 994090 9994090 train running low none '2026-03-04 00:00:00+00'
launch_and_persist 994091 9994091 train running low none '2026-03-04 00:00:10+00'
launch_and_persist 994092 9994092 train pending high operator '2026-03-04 00:00:20+00'
run_scheduler 2 0 "${test_dir}/victim-youngest.out"
grep -q 'victim_experiment_id=994091' "${test_dir}/victim-youngest.out"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=994090")" = running
retire_all

# C10: coherent durable but wrong OS start identity fails closed before SIGSTOP.
launch_and_persist 994100 9994100 train running low none '2026-03-05 00:00:00+00'
launch_and_persist 994101 9994101 train pending high operator '2026-03-05 00:00:01+00'
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET worker_process_start_identity='mismatch' WHERE experiment_id=994100; UPDATE experiment_scheduler_worker_attempt SET worker_process_start_identity='mismatch' WHERE worker_attempt_id=9994100"
run_scheduler 1 0 "${test_dir}/identity-mismatch.out"
! grep -q 'SCHEDULER_PRIORITY_PREEMPTED' "${test_dir}/identity-mismatch.out"
test "$(scalar "SELECT status||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=994100")" = 'running:none'
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" != T* ]]
retire_all

# C17: administrative stop-after-checkpoint state is not a safe victim.
launch_and_persist 994110 9994110 train running low none '2026-03-06 00:00:00+00'
launch_and_persist 994111 9994111 train pending high operator '2026-03-06 00:00:01+00'
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET stop_after_checkpoint_epoch=40 WHERE experiment_id=994110"
run_scheduler 1 0 "${test_dir}/unsafe-victim.out"
! grep -q 'SCHEDULER_PRIORITY_PREEMPTED' "${test_dir}/unsafe-victim.out"
test "$(scalar "SELECT status FROM experiment WHERE experiment_id=994111")" = pending
retire_all

# C18: injected post-SIGSTOP DB failure aborts and identity-safely compensates.
launch_and_persist 994120 9994120 train running low none '2026-03-07 00:00:00+00'
launch_and_persist 994121 9994121 train pending high operator '2026-03-07 00:00:01+00'
set +e
trap - ERR
EA_SCHEDULER_OWNERSHIP_TEST_ENABLE=1 \
EA_SCHEDULER_OWNERSHIP_TEST_BOUNDARY=preemption_after_sigstop_before_db \
    run_scheduler 1 0 "${test_dir}/rollback-compensation.out"
rollback_result=$?
trap diagnose_error ERR
set -e
test "${rollback_result}" -ne 0
grep -q 'SCHEDULER_PREEMPTION_ROLLBACK_COMPENSATION.*restored=1' \
    "${test_dir}/rollback-compensation.out"
test "$(scalar "SELECT status||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=994120")" = 'running:none'
test "$(scalar "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9994120")" = observed
rollback_process_state=''
for _ in {1..150}; do
    rollback_process_state="$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')"
    [[ "${rollback_process_state}" != T* ]] && break
    sleep 0.02
done
[[ "${rollback_process_state}" != T* ]]
retire_all

# C7/C11/C16 and the integrated two-slot case: L1 is displaced before N1,
# both highs enter, restart preserves stopped attempts, then N1 resumes first.
launch_and_persist 994200 9994200 train running low none '2026-03-08 00:00:00+00'
launch_and_persist 994201 9994201 train running normal none '2026-03-08 00:00:01+00'
launch_and_persist 994202 9994202 train pending high operator '2026-03-08 00:00:02+00'
launch_and_persist 994203 9994203 train pending high operator '2026-03-08 00:00:03+00'
run_scheduler 2 0 "${test_dir}/two-slot.out"
test "$(scalar "SELECT string_agg(experiment_id||':'||status||':'||scheduler_resume_origin,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 994200 AND 994203")" = \
    '994200:pending:preemption,994201:pending:preemption,994202:running:none,994203:running:none'
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id IN (9994200,9994201) AND lifecycle_state='stopped'")" = 2
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE capacity_class='train' AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = 2
grep -q 'candidate_experiment_id=994202.*victim_experiment_id=994200' "${test_dir}/two-slot.out"
grep -q 'candidate_experiment_id=994203.*victim_experiment_id=994201' "${test_dir}/two-slot.out"
run_scheduler 2 0 "${test_dir}/restart-preserves.out" --recover-orphans-only
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id IN (9994200,9994201) AND lifecycle_state='stopped'")" = 2
run_cli --pause-experiment=994202 --yes >"${test_dir}/pause-h1.out"
run_cli --pause-experiment=994203 --yes >"${test_dir}/pause-h2.out"
run_scheduler 2 0 "${test_dir}/resume-order.out"
normal_line="$(rg -n 'SCHEDULER_STOPPED_WORKER_ADMITTED,experiment_id=994201' "${test_dir}/resume-order.out" | cut -d: -f1)"
low_line="$(rg -n 'SCHEDULER_STOPPED_WORKER_ADMITTED,experiment_id=994200' "${test_dir}/resume-order.out" | cut -d: -f1)"
test "${normal_line}" -lt "${low_line}"
test "$(scalar "SELECT string_agg(experiment_id||':'||active_scheduler_worker_attempt_id||':'||scheduler_resume_origin,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id IN (994200,994201)")" = \
    '994200:9994200:none,994201:9994201:none'
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id IN (994200,994201)")" = 2
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" != T* ]]
[[ "$(ps -o state= -p "${worker_pids[1]}" | tr -d ' ')" != T* ]]
retire_all

# C12: a disappeared preempted train worker retains automatic continuation
# urgency and uses the Phase A selector, including invalid-newest fallback.
launch_and_persist 994300 9994300 train running low none '2026-03-09 00:00:00+00'
launch_and_persist 994301 9994301 train pending high operator '2026-03-09 00:00:01+00'
valid_checkpoint="$(insert_checkpoint_model 994300 preempt994300 40)"
invalid_checkpoint="$(insert_checkpoint_model 994300 wrongsymbol 60)"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET last_model_id=${invalid_checkpoint},resume_model_id=${invalid_checkpoint},current_epoch=60 WHERE experiment_id=994300"
run_scheduler 1 0 "${test_dir}/missing-preempted-setup.out"
test "$(scalar "SELECT status||':'||scheduler_resume_origin FROM experiment WHERE experiment_id=994300")" = 'pending:preemption'
kill -KILL -- "-${worker_pgids[0]}"
wait "${worker_pids[0]}" 2>/dev/null || true
run_scheduler 1 0 "${test_dir}/missing-preempted-recovery.out" --recover-orphans-only
grep -q 'SCHEDULER_STOPPED_WORKER_RECONCILED.*experiment_id=994300.*result=phase_a_checkpoint_restart' \
    "${test_dir}/missing-preempted-recovery.out"
test "$(scalar "SELECT status||':'||phase||':'||last_model_id||':'||resume_model_id||':'||resume_requested::text||':'||scheduler_resume_origin||':'||(active_scheduler_worker_attempt_id IS NULL)::text FROM experiment WHERE experiment_id=994300")" = \
    "pending:train:${valid_checkpoint}:${valid_checkpoint}:true:preemption:true"
test "$(scalar "SELECT lifecycle_state||':'||reconciliation_result FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9994300")" = \
    'abandoned:stopped_process_missing'
retire_all

printf '%s\n' \
    'SchedulerPriorityPreemptionIntegrationTests passed' \
    'strict_priority_preemption=C1,C2,C3,C4,C5,C6:PASS' \
    'two_slot_L1_N1_H1_H2=C7:PASS' \
    'victim_order=C8,C9:PASS' \
    'identity_fail_closed=C10:PASS' \
    'same_attempt_sigcont=C11:PASS' \
    'missing_worker_checkpoint_fallback=C12:PASS' \
    'scheduler_restart=C16:PASS' \
    'unsafe_victim=C17:PASS' \
    'rollback_compensation=C18:PASS' \
    'no_over_preemption=C19:PASS' \
    'normal_before_low_resume=PASS' \
    'production_processes_signaled=0'
