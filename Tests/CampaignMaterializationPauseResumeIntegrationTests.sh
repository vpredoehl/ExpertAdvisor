#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_root="${repo_root}/DerivedData/Development/SchedulerCampaignMaterializationPauseResumeValidation"
harness="${1:-${build_root}/Tests/CampaignMaterializationControlCliHarness}"
process_binary="${2:-${build_root}/Tests/GlobalExperimentControlProcessTests}"
test_db="ea_campaign_materialization_control_${$}"
test_dir="$(mktemp -d "${TMPDIR:-/tmp}/ea-campaign-control.XXXXXX")"
worker_pids=()
worker_pgids=()
worker_starts=()
worker_executables=()
worker_commands=()
export PGOPTIONS=

case "${test_db}" in
    ea_campaign_materialization_control_[0-9]*) ;;
    *) exit 90 ;;
esac

inspect_process() {
    "${process_binary}" --inspect-managed-test-process="$1" 2>/dev/null
}

worker_identity_matches() {
    local index="$1" identity="" pid="" pgid="" start="" executable="" command=""
    identity="$(inspect_process "${worker_pids[${index}]}" || true)"
    IFS='|' read -r pid pgid start executable command <<<"${identity}"
    [[ "${pid}" = "${worker_pids[${index}]}" ]] &&
        [[ "${pgid}" = "${worker_pgids[${index}]}" ]] &&
        [[ "${start}" = "${worker_starts[${index}]}" ]] &&
        [[ "${executable}" = "${worker_executables[${index}]}" ]] &&
        [[ "${command}" == *"--managed-test-worker"* ]]
}

cleanup() {
    local index=""
    for index in "${!worker_pids[@]}"; do
        if kill -0 "${worker_pids[${index}]}" >/dev/null 2>&1 &&
           worker_identity_matches "${index}"; then
            kill -CONT -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
            kill -TERM -- "-${worker_pgids[${index}]}" >/dev/null 2>&1 || true
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
        "SELECT experiment_id,status,phase,resume_requested,
                scheduler_priority,active_scheduler_worker_attempt_id
         FROM experiment WHERE experiment_id BETWEEN 887100 AND 887999
         ORDER BY experiment_id" >&2 || true
    return "${result}"
}
trap diagnose_error ERR

mkdir -p "${build_root}/Tests"
read -r -a pqxx_compile_flags <<<"$(pkg-config --cflags libpqxx)"
read -r -a pqxx_link_flags <<<"$(pkg-config --libs libpqxx)"
"${CXX:-clang++}" -std=c++20 -O0 -g -Wall -Wextra -Werror \
    -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" "${pqxx_compile_flags[@]}" \
    "${repo_root}/Tests/CampaignMaterializationControlCliHarness.cpp" \
    "${repo_root}/Sources/GlobalExperimentControl.cpp" \
    "${pqxx_link_flags[@]}" -o "${harness}"
"${CXX:-clang++}" -std=c++20 -O0 -g \
    -Wno-deprecated-declarations -Wno-c++23-attribute-extensions \
    -I"${repo_root}/Headers" "${pqxx_compile_flags[@]}" \
    "${repo_root}/Tests/GlobalExperimentControlProcessTests.cpp" \
    "${repo_root}/Sources/GlobalExperimentControl.cpp" \
    "${pqxx_link_flags[@]}" -o "${process_binary}"

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/086_scheduler_pause_resume_priority.sql"
if [[ "$(psql -X -Atq -d "${test_db}" -c \
    "SELECT to_regclass('public.experiment_campaign_materialization_control_operation') IS NOT NULL")" != "t" ]]; then
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -f "${repo_root}/Database/migrations/087_campaign_materialization_pause_resume.sql"
fi
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/093_scheduler_priority_preemption.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Tests/CampaignMaterializationControlMigrationTests.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE
SET desired_state='running',active_request_id=NULL,current_pause_request_id=NULL;
UPDATE experiment_scheduler_protocol
SET cutover_state='complete',cutover_completed_at=clock_timestamp(),
    cutover_completed_by='campaign_materialization_control_integration',
    cutover_executable_path='/tmp/isolated-test',
    cutover_process_evidence='isolated_disposable_database',
    failure_diagnostic=NULL,updated_at=clock_timestamp()
WHERE singleton=true;
SQL

scalar() {
    psql -X -Atq -d "${test_db}" -c "$1"
}

run_control() {
    LSTM_DB_NAME="${test_db}" "${harness}" "$@"
}

insert_materialization() {
    local materialization_id="$1" member_count="$2" member_base="$3" proposal_base="$4"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v materialization_id="${materialization_id}" \
        -v member_count="${member_count}" \
        -v member_base="${member_base}" \
        -v proposal_base="${proposal_base}" <<'SQL'
SET session_replication_role=replica;
INSERT INTO experiment_recommendation_campaign_materialization(
 recommendation_campaign_materialization_id,recommendation_campaign_approval_id,
 materialization_contract_version,approval_identity_canonical,approval_identity_hash,
 recommendation_ranking_snapshot_id,ranking_snapshot_identity_canonical,
 ranking_snapshot_identity_hash,planning_policy_canonical,planning_policy_hash,
 planning_scope_canonical,campaign_plan_identity_canonical,campaign_plan_identity_hash,
 campaign_review_identity_canonical,campaign_review_identity_hash,approval_decision,
 approval_reviewer_identity,approval_reason_text,materialized_by,
 materialization_reason_text,selected_member_count,initially_created_proposal_count,
 initially_reused_proposal_count,materialization_identity_canonical,
 materialization_identity_hash)
VALUES(:materialization_id,:materialization_id * 10,1,'approval',
'fnv1a64:0000000000000001',:materialization_id * 10,'ranking',
'fnv1a64:0000000000000001','policy','fnv1a64:0000000000000001',
'scope','plan','fnv1a64:0000000000000001','review',
'fnv1a64:0000000000000001','approved','fixture','fixture','fixture',
'fixture',:member_count,:member_count,0,'materialization',
'fnv1a64:0000000000000001');
INSERT INTO experiment_recommendation_campaign_materialization_member(
 recommendation_campaign_materialization_member_id,
 recommendation_campaign_materialization_id,member_ordinal,
 recommendation_ranking_member_id,recommendation_id,source_experiment_id,
 ranking_position,selected_member_identity_canonical,selected_member_identity_hash,
 recommendation_conversion_proposal_id,proposal_identity_canonical,
 proposal_identity_hash)
SELECT :member_base + ordinal,:materialization_id,ordinal,
       :member_base + 100 + ordinal,:member_base + 200 + ordinal,
       :member_base + 300 + ordinal,ordinal,'selected:'||ordinal,
       'fnv1a64:'||lpad(ordinal::text,16,'0'),
       :proposal_base + ordinal,'proposal:'||ordinal,
       'fnv1a64:'||lpad(ordinal::text,16,'0')
FROM generate_series(1,:member_count) AS ordinal;
SET session_replication_role=origin;
SQL
}

insert_experiment() {
    local experiment_id="$1" status="$2" phase="$3" priority="$4" updated="$5"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v status="${status}" \
        -v phase="${phase}" -v priority="${priority}" -v updated="${updated}" <<'SQL'
INSERT INTO experiment(
 experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,
 head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,
 status,phase,current_operation,duplicate_nonce,scheduler_priority,updated_at,
 model_input_width,model_input_semantic_layout_version)
VALUES(:experiment_id,'campaignfixture'||:experiment_id,1,0.0008,1,1,2,1,
 '2020-01-01','2020-02-01',:'status',:'phase',
 CASE WHEN :'phase' IN ('train','infer','analyze') THEN :'phase' ELSE NULL END,
 :experiment_id,:'priority',:'updated'::timestamptz,77,6);
SQL
}

link_member() {
    local proposal_id="$1" execution_id="$2" experiment_id="$3"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v proposal_id="${proposal_id}" -v execution_id="${execution_id}" \
        -v experiment_id="${experiment_id}" <<'SQL'
SET session_replication_role=replica;
INSERT INTO experiment_recommendation_conversion_execution(
 recommendation_conversion_execution_id,recommendation_conversion_proposal_id,
 recommendation_conversion_review_decision_id,experiment_id,
 execution_contract_version,authorization_decision,execution_identity_canonical,
 execution_identity_hash)
VALUES(:execution_id,:proposal_id,:execution_id + 100000,:experiment_id,
       1,'approve','execution:'||:execution_id,'hash:'||:execution_id);
SET session_replication_role=origin;
SQL
}

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
        identity="$(inspect_process "${launched_pid}" || true)"
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
    local index="$1" experiment_id="$2" attempt_id="$3" attempt_state="${4:-running}"
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
        -v experiment_id="${experiment_id}" -v attempt_id="${attempt_id}" \
        -v attempt_state="${attempt_state}" \
        -v pid="${worker_pids[${index}]}" -v pgid="${worker_pgids[${index}]}" \
        -v start="${worker_starts[${index}]}" \
        -v executable="${worker_executables[${index}]}" \
        -v command="${worker_commands[${index}]}" <<'SQL'
SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false);
INSERT INTO experiment(
 experiment_id,symbol,prediction_horizon,c_next_threshold,core_lr_mult,
 head_lr_mult,target_epochs,checkpoint_interval,train_start,train_end,
 status,phase,current_operation,duplicate_nonce,scheduler_priority,
 worker_pid,worker_process_group_id,worker_process_start_identity,
 worker_executable,worker_command_line,worker_control_state,
 model_input_width,model_input_semantic_layout_version)
VALUES(:experiment_id,'campaignworker'||:experiment_id,1,0.0008,1,1,2,1,
 '2020-01-01','2020-02-01','running','train','train',:experiment_id,'low',
 :pid,:pgid,:'start',:'executable',:'command','running',77,6);
INSERT INTO experiment_scheduler_worker_attempt(
 worker_attempt_id,launch_attempt_identity,experiment_id,worker_kind,
 lifecycle_phase,capacity_class,ownership_origin,lifecycle_state,
 worker_pid,worker_process_group_id,worker_process_start_identity,
 canonical_executable_path,command_line,command_identity,spawned_at,registered_at)
VALUES(:attempt_id,'campaign-control-'||:attempt_id,:experiment_id,'experiment',
 'train','train','prior_scheduler_observed',:'attempt_state',:pid,:pgid,:'start',
 :'executable',:'command','experiment:'||:experiment_id||':train',
 clock_timestamp(),clock_timestamp());
UPDATE experiment SET active_scheduler_worker_attempt_id=:attempt_id
WHERE experiment_id=:experiment_id;
SQL
}

# Pending, infer, independently paused, terminal, and unresolved members.
insert_materialization 887001 5 887010 887020
insert_experiment 887101 pending train low '2026-01-01 00:00:01+00'
insert_experiment 887102 pending infer high '2026-01-01 00:00:02+00'
insert_experiment 887103 paused train normal '2026-01-01 00:00:03+00'
insert_experiment 887104 completed done high '2026-01-01 00:00:04+00'
link_member 887021 887031 887101
link_member 887022 887032 887102
link_member 887023 887033 887103
link_member 887024 887034 887104

before_dry_run="$(scalar "SELECT string_agg(experiment_id||':'||status||':'||resume_requested,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887101 AND 887104")"
run_control --pause-campaign-materialization=887001 --dry-run >"${test_dir}/dry-run.out"
grep -q 'result=dry_run.*target_count=5.*changed_count=2' "${test_dir}/dry-run.out"
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")" = 0
test "${before_dry_run}" = "$(scalar "SELECT string_agg(experiment_id||':'||status||':'||resume_requested,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887101 AND 887104")"

run_control --pause-campaign-materialization=887001 >"${test_dir}/confirmation.out"
grep -q 'result=confirmation_required' "${test_dir}/confirmation.out"
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")" = 0

run_control --pause-campaign-materialization=887001 --yes >"${test_dir}/pause-pending.out"
grep -q 'target_count=5,changed_count=2,already_paused_count=1,terminal_non_applicable_count=2' "${test_dir}/pause-pending.out"
test "$(scalar "SELECT string_agg(experiment_id||':'||status||':'||resume_requested||':'||scheduler_priority,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887101 AND 887104")" = \
    '887101:paused:false:low,887102:paused:false:high,887103:paused:false:normal,887104:completed:false:high'
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE ownership_state='active'")" = 2

run_control --pause-campaign-materialization=887001 --yes >"${test_dir}/repeat-pause.out"
grep -q 'changed_count=0,already_paused_count=3' "${test_dir}/repeat-pause.out"
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE ownership_state='active'")" = 2

# A downstream execution created after the frozen pause population must not be
# broadened into the later resume operation.
insert_experiment 887105 pending train normal '2026-01-01 00:00:05+00'
link_member 887025 887035 887105
run_control --resume-campaign-materialization=887001 --yes >"${test_dir}/resume-pending.out"
grep -q 'target_count=5,changed_count=2.*not_group_owned_count=3' "${test_dir}/resume-pending.out"
test "$(scalar "SELECT string_agg(experiment_id||':'||status||':'||resume_requested||':'||scheduler_priority,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887101 AND 887105")" = \
    '887101:pending:true:low,887102:pending:true:high,887103:paused:false:normal,887104:completed:false:high,887105:pending:false:normal'
test "$(scalar "SELECT string_agg(experiment_id||':'||scheduler_resume_origin,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887101 AND 887105")" = \
    '887101:operator,887102:operator,887103:none,887104:none,887105:none'
insert_experiment 887106 pending train high '2025-01-01 00:00:00+00'
test "$(scalar "SELECT string_agg(experiment_id::text,',' ORDER BY CASE scheduler_priority WHEN 'high' THEN 0 WHEN 'normal' THEN 1 ELSE 2 END,CASE scheduler_resume_origin WHEN 'operator' THEN 0 WHEN 'preemption' THEN 1 ELSE 2 END,updated_at,experiment_id) FROM experiment WHERE experiment_id IN (887101,887106)")" = '887106,887101'
run_control --resume-campaign-materialization=887001 --yes >"${test_dir}/repeat-resume.out"
grep -q 'changed_count=0.*not_group_owned_count=5' "${test_dir}/repeat-resume.out"

# Individual pause and resume both supersede active group ownership.
insert_materialization 887002 3 887040 887050
for ordinal in 1 2 3; do
    experiment_id=$((887110 + ordinal))
    insert_experiment "${experiment_id}" pending train normal "2026-02-01 00:00:0${ordinal}+00"
    link_member "$((887050 + ordinal))" "$((887060 + ordinal))" "${experiment_id}"
done
run_control --pause-campaign-materialization=887002 --yes >"${test_dir}/pause-supersession.out"
run_control --pause-experiment=887111 --yes >"${test_dir}/individual-pause.out"
run_control --resume-experiment=887112 --yes >"${test_dir}/individual-resume.out"
test "$(scalar "SELECT string_agg(experiment_id||':'||ownership_state||':'||COALESCE(superseded_action,'NULL'),',' ORDER BY experiment_id) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id BETWEEN 887111 AND 887113")" = \
    '887111:superseded:pause,887112:superseded:resume,887113:active:NULL'
run_control --resume-campaign-materialization=887002 --yes >"${test_dir}/resume-supersession.out"
grep -q 'changed_count=1.*not_group_owned_count=2' "${test_dir}/resume-supersession.out"
test "$(scalar "SELECT string_agg(experiment_id||':'||status||':'||resume_requested,',' ORDER BY experiment_id) FROM experiment WHERE experiment_id BETWEEN 887111 AND 887113")" = \
    '887111:paused:false,887112:pending:true,887113:pending:true'

# Controlled live workers: one is stopped and releases capacity, while an
# identity_ambiguous attempt fails closed and receives no signal.
insert_materialization 887003 2 887070 887080
launch_worker 887121 9887121
persist_worker 0 887121 9887121 running
link_member 887081 887091 887121
launch_worker 887122 9887122
persist_worker 1 887122 9887122 identity_ambiguous
link_member 887082 887092 887122
operations_before_worker_dry_run="$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")"
worker_dry_run_result=0
run_control --pause-campaign-materialization=887003 --dry-run \
    >"${test_dir}/pause-workers-dry-run.out" 2>&1 || worker_dry_run_result=$?
test "${worker_dry_run_result}" = 1
grep -q 'result=dry_run.*changed_count=1.*identity_failure_count=1' \
    "${test_dir}/pause-workers-dry-run.out"
test "${operations_before_worker_dry_run}" = \
    "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")"
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" != T* ]]
[[ "$(ps -o state= -p "${worker_pids[1]}" | tr -d ' ')" != T* ]]
pause_workers_result=0
run_control --pause-campaign-materialization=887003 --yes \
    >"${test_dir}/pause-workers.out" 2>&1 || pause_workers_result=$?
test "${pause_workers_result}" = 1
grep -q 'changed_count=1.*identity_failure_count=1' "${test_dir}/pause-workers.out"
for _ in {1..100}; do
    worker_state="$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')"
    [[ "${worker_state}" == T* ]] && break
    sleep 0.02
done
[[ "${worker_state}" == T* ]]
[[ "$(ps -o state= -p "${worker_pids[1]}" | tr -d ' ')" != T* ]]
test "$(scalar "SELECT lifecycle_state FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9887121")" = stopped
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9887121 AND lifecycle_state IN ('reserved','spawned','running','observed','identity_ambiguous')")" = 0
run_control --resume-campaign-materialization=887003 --yes >"${test_dir}/resume-workers.out"
test "$(scalar "SELECT status||':'||resume_requested FROM experiment WHERE experiment_id=887121")" = 'pending:true'
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]

# Campaign re-pause retains the exact stopped worker awaiting admission. Its
# dry run does not mutate durable provenance, ownership, attempt state, or the
# stopped process.
repause_attempt="$(scalar "SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id=887121")"
repause_signal="$(scalar "SELECT signal_number FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9887121")"
repause_operations="$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")"
repause_active_ownership="$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id=887121 AND ownership_state='active'")"
repause_dry_run_state="$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887121")"
repause_dry_run_result=0
run_control --pause-campaign-materialization=887003 --dry-run \
    >"${test_dir}/repause-workers-dry-run.out" 2>&1 || repause_dry_run_result=$?
test "${repause_dry_run_result}" = 1
grep -q 'outcome=changed_by_group_pause,changed=1,reason=would_retain_stopped_worker' "${test_dir}/repause-workers-dry-run.out"
test "${repause_operations}" = "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation")"
test "${repause_active_ownership}" = "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id=887121 AND ownership_state='active'")"
test "${repause_dry_run_state}" = "$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887121")"
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]
repause_result=0
run_control --pause-campaign-materialization=887003 --yes \
    >"${test_dir}/repause-workers.out" 2>&1 || repause_result=$?
test "${repause_result}" = 1
grep -q 'outcome=changed_by_group_pause,changed=1,reason=stopped_worker_retained' "${test_dir}/repause-workers.out"
test "$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887121")" = \
    "paused:false:low:${repause_attempt}:stopped:${repause_signal}"
test "$(scalar "SELECT outcome_kind||':'||changed_by_operation||':'||identity_result FROM experiment_campaign_materialization_control_outcome o JOIN experiment_campaign_materialization_control_member m USING(campaign_materialization_control_member_id) WHERE m.experiment_id=887121 ORDER BY campaign_materialization_control_outcome_id DESC LIMIT 1")" = 'changed_by_group_pause:true:validated'
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id=887121 AND ownership_state='active'")" = 1
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=887121")" = 1
[[ "$(ps -o state= -p "${worker_pids[0]}" | tr -d ' ')" == T* ]]

# A pending stopped attempt with a stale exact identity fails closed without
# changing lifecycle or signalling the controlled process.
insert_materialization 887004 1 8871000 8871100
launch_worker 887131 9887131
persist_worker 2 887131 9887131 running
link_member 8871101 8871201 887131
run_control --pause-experiment=887131 --yes >"${test_dir}/stale-setup-pause.out"
run_control --resume-experiment=887131 --yes >"${test_dir}/stale-setup-resume.out"
[[ "$(ps -o state= -p "${worker_pids[2]}" | tr -d ' ')" == T* ]]
individual_repause_attempt="$(scalar "SELECT active_scheduler_worker_attempt_id FROM experiment WHERE experiment_id=887131")"
individual_repause_signal="$(scalar "SELECT signal_number FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=9887131")"
individual_repause_state="$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887131")"
run_control --pause-experiment=887131 --dry-run >"${test_dir}/individual-repause-dry-run.out"
grep -q 'worker_action=validate_and_retain_stopped' "${test_dir}/individual-repause-dry-run.out"
test "${individual_repause_state}" = "$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887131")"
[[ "$(ps -o state= -p "${worker_pids[2]}" | tr -d ' ')" == T* ]]
run_control --pause-experiment=887131 --yes >"${test_dir}/individual-repause.out"
grep -q 'new_status=paused,resume_requested=false,worker_state=stopped' "${test_dir}/individual-repause.out"
test "$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.scheduler_priority||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state||':'||COALESCE(a.signal_number::text,'NULL') FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887131")" = \
    "paused:false:low:${individual_repause_attempt}:stopped:${individual_repause_signal}"
test "$(scalar "SELECT count(*) FROM experiment_scheduler_worker_attempt WHERE experiment_id=887131")" = 1
[[ "$(ps -o state= -p "${worker_pids[2]}" | tr -d ' ')" == T* ]]
run_control --resume-experiment=887131 --yes >"${test_dir}/stale-setup-second-resume.out"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "SELECT set_config('expertadvisor.scheduler_protocol_generation','52',false); UPDATE experiment_scheduler_worker_attempt SET worker_process_start_identity='stale:start' WHERE worker_attempt_id=9887131; UPDATE experiment SET worker_process_start_identity='stale:start' WHERE experiment_id=887131;"
stale_result=0
run_control --pause-campaign-materialization=887004 --yes \
    >"${test_dir}/stale-pid.out" 2>&1 || stale_result=$?
test "${stale_result}" = 1
grep -q 'identity_failure_count=1' "${test_dir}/stale-pid.out"
test "$(scalar "SELECT e.status||':'||e.resume_requested||':'||e.active_scheduler_worker_attempt_id||':'||a.lifecycle_state FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887131")" = 'pending:true:9887131:stopped'
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id=887131 AND ownership_state='active'")" = 0
[[ "$(ps -o state= -p "${worker_pids[2]}" | tr -d ' ')" == T* ]]

# Process disappearance after a durable group pause preserves resume priority
# and the stopped attempt for scheduler admission/restart fallback.
insert_materialization 887005 1 8871300 8871400
launch_worker 887141 9887141
persist_worker 3 887141 9887141 running
link_member 8871401 8871501 887141
run_control --pause-campaign-materialization=887005 --yes >"${test_dir}/pause-disappear.out"
[[ "$(ps -o state= -p "${worker_pids[3]}" | tr -d ' ')" == T* ]]
kill -KILL -- "-${worker_pgids[3]}"
wait "${worker_pids[3]}" 2>/dev/null || true
run_control --resume-campaign-materialization=887005 --yes >"${test_dir}/resume-disappear.out"
test "$(scalar "SELECT e.status||':'||e.resume_requested||':'||a.lifecycle_state FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id=887141")" = 'pending:true:stopped'

# A later lifecycle transition cannot be overridden merely because ownership
# still exists; resume records the exact predicate mismatch and fails closed.
insert_materialization 887006 1 8871600 8871700
insert_experiment 887151 pending train normal '2026-03-01 00:00:00+00'
link_member 8871701 8871801 887151
run_control --pause-campaign-materialization=887006 --yes >"${test_dir}/pause-predicate.out"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" -c \
    "UPDATE experiment SET status='completed',phase='done',updated_at=clock_timestamp() WHERE experiment_id=887151"
predicate_result=0
run_control --resume-campaign-materialization=887006 --yes \
    >"${test_dir}/resume-predicate.out" 2>&1 || predicate_result=$?
test "${predicate_result}" = 1
grep -q 'resume_predicate_mismatch_count=1' "${test_dir}/resume-predicate.out"
test "$(scalar "SELECT status||':'||resume_requested FROM experiment WHERE experiment_id=887151")" = 'completed:false'

# A later unexpected database exception rolls back group provenance and state;
# every process newly SIGSTOP'd by the aborted transaction is compensated.
insert_materialization 887008 2 8872200 8872300
launch_worker 887171 9887171
persist_worker 4 887171 9887171 running
link_member 8872301 8872401 887171
launch_worker 887172 9887172
persist_worker 5 887172 9887172 running
link_member 8872302 8872402 887172
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
CREATE FUNCTION campaign_pause_fault_after_sigstop() RETURNS trigger
LANGUAGE plpgsql AS $$
BEGIN
    IF NEW.worker_attempt_id = 9887172 AND
       NEW.reconciliation_result = 'paused_by_campaign_materialization' THEN
        RAISE EXCEPTION 'injected campaign pause persistence failure';
    END IF;
    RETURN NEW;
END;
$$;
CREATE TRIGGER campaign_pause_fault_after_sigstop
BEFORE UPDATE ON experiment_scheduler_worker_attempt
FOR EACH ROW EXECUTE FUNCTION campaign_pause_fault_after_sigstop();
SQL
rollback_result=0
run_control --pause-campaign-materialization=887008 --yes \
    >"${test_dir}/rollback-compensation.out" 2>&1 || rollback_result=$?
test "${rollback_result}" = 2
test "$(grep -c 'ROLLBACK_COMPENSATION.*restored=1' "${test_dir}/rollback-compensation.out")" = 2
for index in 4 5; do
    for _ in {1..100}; do
        rollback_state="$(ps -o state= -p "${worker_pids[${index}]}" | tr -d ' ')"
        [[ "${rollback_state}" != T* ]] && break
        sleep 0.02
    done
    [[ "${rollback_state}" != T* ]]
done
test "$(scalar "SELECT string_agg(e.experiment_id||':'||e.status||':'||e.resume_requested||':'||a.lifecycle_state,',' ORDER BY e.experiment_id) FROM experiment e JOIN experiment_scheduler_worker_attempt a ON a.worker_attempt_id=e.active_scheduler_worker_attempt_id WHERE e.experiment_id IN (887171,887172)")" = \
    '887171:running:false:running,887172:running:false:running'
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation WHERE recommendation_campaign_materialization_id=887008")" = 0
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_pause_ownership WHERE experiment_id IN (887171,887172)")" = 0

test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_operation WHERE resolution_status='resolved'")" -ge 10
test "$(scalar "SELECT count(*) FROM experiment_campaign_materialization_control_member m LEFT JOIN experiment_campaign_materialization_control_outcome o USING(campaign_materialization_control_member_id) WHERE o.campaign_materialization_control_outcome_id IS NULL")" = 0

printf '%s\n' 'CampaignMaterializationPauseResumeIntegrationTests passed'
