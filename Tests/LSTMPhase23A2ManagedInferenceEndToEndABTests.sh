#!/usr/bin/env bash
# Deterministic, disposable Phase 23A2 managed-inference parity regression.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
products="${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release"
scheduler="${1:-${products}/lstm-scheduler}"
compatibility="${2:-${products}/LSTM_Release}"
standalone="${3:-${products}/lstm-infer-worker}"
build_dir="${repo_root}/DerivedData/ExpertAdvisor/Phase23A2"
seeder="${build_dir}/fixture-seeder"
test_dir="$(mktemp -d /tmp/ea_phase23a2_managed_inference.XXXXXX)"
base_tag="ea_phase23a2_managed_inference_${$}"
source_commit="$(git -C "${repo_root}" rev-parse HEAD)"

for binary in "${scheduler}" "${compatibility}" "${standalone}"; do
    test -x "${binary}"
done
live_scheduler_pid=""

cleanup() {
    local status=$?
    if [[ -n "${live_scheduler_pid}" ]] && kill -0 "${live_scheduler_pid}" >/dev/null 2>&1; then kill -TERM "${live_scheduler_pid}" >/dev/null 2>&1 || true; wait "${live_scheduler_pid}" >/dev/null 2>&1 || true; fi
    while IFS= read -r database; do
        [[ -n "${database}" ]] || continue
        case "${database}" in
            ${base_tag}_a[0-9]_*|${base_tag}_b[0-9]_*|${base_tag}_live[0-9]_*)
                dropdb --if-exists "${database}" >/dev/null 2>&1 || true
                ;;
            *) printf 'refusing unexpected Phase23A2 cleanup target: %s\n' "${database}" >&2 ;;
        esac
    done <"${database_list:-/dev/null}"
    if [[ "${status}" -ne 0 ]]; then
        find "${test_dir}" -maxdepth 2 -name '*.out' -type f -exec sed -n '1,220p' {} \; >&2 || true
    fi
    rm -rf -- "${test_dir}"
    return "${status}"
}
database_list="${test_dir}/databases"
: >"${database_list}"
trap cleanup EXIT

read -r -a pqxx_cflags <<<"$(pkg-config --cflags libpqxx)"
read -r -a pqxx_libs <<<"$(pkg-config --libs libpqxx)"
mkdir -p "${build_dir}"
clang++ -std=c++20 -Wall -Wextra -Werror -Wno-deprecated-declarations \
    -I"${repo_root}/Headers" -I"${repo_root}/Sources" "${pqxx_cflags[@]}" \
    "${repo_root}/Sources/EconomicEventRepository.cpp" \
    "${repo_root}/Tests/Phase23A2ManagedInferenceFixtureSeeder.cpp" \
    "${pqxx_libs[@]}" -o "${seeder}"

scalar() { psql -X -At -q -d "$1" -c "$2"; }

make_registry() {
    local root="$1" executable="$2"
    local executable_sha
    executable_sha="$(shasum -a 256 "${executable}" | awk '{print $1}')"
    /usr/bin/python3 - "${repo_root}" "${root}" "${executable}" \
        "${executable_sha}" "${source_commit}" <<'PY'
import json
import os
from pathlib import Path
import shutil
import sys

repo = Path(sys.argv[1])
root = Path(sys.argv[2])
worker = Path(sys.argv[3])
digest = sys.argv[4]
source_commit = sys.argv[5]
registry_source = repo / "Builds/SemanticWorkers/registry.json"
source = json.loads(registry_source.read_text(encoding="utf-8"))
root.mkdir(parents=True, exist_ok=False)
runtime = source["runtimes"][0]
runtime_source = registry_source.parent / runtime["directory"]
runtime_destination = root / runtime["directory"]
shutil.copytree(runtime_source, runtime_destination)
train = next(item for item in source["workers"]
             if item["semantic_layout"] == 7 and item["worker_role"] == "train")
for field in ("executable", "manifest"):
    src = registry_source.parent / train[field]
    dst = root / train[field]
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
train_dir = (root / train["executable"]).parent
for name in ("default.metallib", "MetaNN.metallib"):
    (train_dir / name).symlink_to(os.path.relpath(runtime_destination / name, train_dir))
relative = Path("layout7/infer") / source_commit / digest
worker_dir = root / relative
worker_dir.mkdir(parents=True)
shutil.copy2(worker, worker_dir / "lstm-infer-worker")
for name in ("default.metallib", "MetaNN.metallib"):
    (worker_dir / name).symlink_to(os.path.relpath(runtime_destination / name, worker_dir))
manifest = {
    "schema_version": 2,
    "semantic_layout": 7,
    "storage": "immutable",
    "model_input_width": 77,
    "source_commit": source_commit,
    "sha256": digest,
    "executable_identity": "lstm-infer-worker",
    "worker_role": "infer",
    "capabilities": ["infer"],
}
(worker_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
registry = {
    "schema_version": 4,
    "current_layout": 7,
    "runtimes": [runtime],
    "workers": [
        {
            "semantic_layout": 7,
            "worker_role": "infer",
            "artifact_manifest_schema_version": 2,
            "worker_rule": "current",
            "model_input_width": 77,
            "source_commit": source_commit,
            "sha256": digest,
            "executable": str(relative / "lstm-infer-worker"),
            "manifest": str(relative / "manifest.json"),
            "runtime_identity": runtime["identity"],
            "capabilities": ["infer"],
        },
        train,
    ],
}
(root / "registry.json").write_text(json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY
}

wait_for_exit() {
    /usr/bin/python3 - "$1" <<'PY'
import os
import select
import sys

pid = int(sys.argv[1])
try:
    os.kill(pid, 0)
except ProcessLookupError:
    raise SystemExit(0)
kq = select.kqueue()
event = select.kevent(pid, filter=select.KQ_FILTER_PROC,
                      flags=select.KQ_EV_ADD | select.KQ_EV_ONESHOT,
                      fflags=select.KQ_NOTE_EXIT)
if not kq.control([event], 0, 0):
    pass
if not kq.control(None, 1, 60):
    raise SystemExit("timed out waiting for disposable managed-inference worker")
PY
}

run_path() {
    local label="$1" executable="$2" iteration="$3"
    local lstm_db="${base_tag}_${label}${iteration}_lstm"
    local forex_db="${base_tag}_${label}${iteration}_forex"
    local root="${test_dir}/${label}${iteration}-registry"
    local log_dir="${test_dir}/${label}${iteration}-logs"
    printf '%s\n%s\n' "${lstm_db}" "${forex_db}" >>"${database_list}"
    createdb "${lstm_db}"
    createdb "${forex_db}"
    PGOPTIONS='-c default_transaction_read_only=on' \
        pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
        psql -X -v ON_ERROR_STOP=1 -q -d "${lstm_db}"
    PGOPTIONS='-c default_transaction_read_only=on' \
        pg_dump -s -h 127.0.0.1 -U vjp -d forex |
        psql -X -v ON_ERROR_STOP=1 -q -d "${forex_db}"
    psql -X -v ON_ERROR_STOP=1 -q -d "${lstm_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
INSERT INTO experiment_scheduler_lease(singleton)
VALUES(true) ON CONFLICT(singleton) DO NOTHING;
INSERT INTO experiment_scheduler_protocol(
    singleton,required_generation,cutover_state,cutover_completed_at,
    cutover_completed_by,cutover_executable_path,cutover_process_evidence,
    failure_diagnostic,updated_at)
VALUES(true,52,'complete',clock_timestamp(),'phase23a2-disposable-fixture',
       '/phase23a2/disposable/lstm-scheduler','disposable fixture',NULL,
       clock_timestamp())
ON CONFLICT(singleton) DO UPDATE SET required_generation=52,
    cutover_state='complete',cutover_completed_at=clock_timestamp(),
    cutover_completed_by='phase23a2-disposable-fixture',
    cutover_executable_path='/phase23a2/disposable/lstm-scheduler',
    cutover_process_evidence='disposable fixture',failure_diagnostic=NULL,
    updated_at=clock_timestamp();
SQL
    psql -X -v ON_ERROR_STOP=1 -q -d "${forex_db}" <<'SQL'
CREATE TABLE phase23a2audrmp(
    time timestamp without time zone NOT NULL,
    bid numeric(10,6) NOT NULL, ask numeric(10,6) NOT NULL,
    vol smallint NOT NULL
);
INSERT INTO phase23a2audrmp(time,bid,ask,vol)
SELECT timestamp '2024-01-01 00:00:00' + i * interval '5 minutes',
       (1.000000 + i * 0.000001)::numeric(10,6),
       (1.000200 + i * 0.000001)::numeric(10,6),1
FROM generate_series(0,2303) AS g(i);
SQL
    local fixture
    fixture="$(LSTM_DB_NAME="${lstm_db}" "${seeder}")"
    grep -Eq '^PHASE23A2_FIXTURE,experiment_id=[0-9]+,model_id=[0-9]+,snapshot_id=[0-9]+,snapshot_hash=fnv1a64:[0-9a-f]{16}$' <<<"${fixture}"
    local experiment_id model_id
    experiment_id="$(sed -n 's/.*experiment_id=\([0-9]*\).*/\1/p' <<<"${fixture}")"
    model_id="$(sed -n 's/.*model_id=\([0-9]*\).*/\1/p' <<<"${fixture}")"
    make_registry "${root}" "${executable}"
    LSTM_DB_NAME="${lstm_db}" FOREX_DB_NAME="${forex_db}" "${scheduler}" \
        --schedule-experiments --scheduler-once --max-train-procs=0 \
        --max-infer-procs=1 --max-analyze-procs=0 \
        --semantic-worker-registry="${root}/registry.json" \
        --scheduler-log-dir="${log_dir}" >"${test_dir}/${label}${iteration}.out" 2>&1
    grep -q "SCHEDULER_CHILD_LAUNCHED,.*experiment_id=${experiment_id},.*phase=infer" \
        "${test_dir}/${label}${iteration}.out"
    local attempt pid
    attempt="$(scalar "${lstm_db}" "SELECT worker_attempt_id FROM experiment_scheduler_worker_attempt WHERE experiment_id=${experiment_id} ORDER BY worker_attempt_id DESC LIMIT 1")"
    pid="$(scalar "${lstm_db}" "SELECT worker_pid FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${attempt}")"
    test -n "${attempt}" && test -n "${pid}"
    wait_for_exit "${pid}"
    LSTM_DB_NAME="${lstm_db}" FOREX_DB_NAME="${forex_db}" "${scheduler}" \
        --schedule-experiments --scheduler-once --recover-orphans-only \
        --semantic-worker-registry="${root}/registry.json" \
        --scheduler-log-dir="${log_dir}" >>"${test_dir}/${label}${iteration}.out" 2>&1
    # The terminal durable result is the scheduler's authoritative completion
    # evidence; the worker's stage stream is retained in its scheduler log.
    test "$(scalar "${lstm_db}" "SELECT status||':'||phase FROM experiment WHERE experiment_id=${experiment_id}")" = 'pending:analyze'
    test "$(scalar "${lstm_db}" "SELECT lifecycle_state||':'||reconciliation_result||':'||COALESCE(exit_code::text,'NULL') FROM experiment_scheduler_worker_attempt WHERE worker_attempt_id=${attempt}")" = 'completed:process_missing_result_recovered:NULL'
    test "$(scalar "${lstm_db}" "SELECT count(*) FROM inference_eval_result WHERE model_id=${model_id} AND inference_scope='final' AND status='completed'")" = 1
    test "$(scalar "${lstm_db}" "SELECT count(*) FROM inference_profitability_observation WHERE model_id=${model_id} AND inference_scope='final'")" = 1
    local worker_log="${log_dir}/experiment_${experiment_id}_phase23a2audrmp_infer.log"
    for stage in \
        SCHEDULER_WORKER_REGISTERED \
        detached_materialization_read_begun \
        detached_materialization_read_completed \
        repeatable_read_transaction_committed \
        evaluation_begun \
        evaluation_completed \
        fresh_read_write_persistence_transaction_begun \
        scheduler_identity_state_revalidation_completed \
        result_persistence_completed \
        profitability_persistence_completed \
        result_profitability_persistence_completed \
        persistence_transaction_committed \
        managed_application_success_return; do
        grep -q "${stage}" "${worker_log}"
    done
    # The generated model's surrogate ID is intentionally excluded: each
    # iteration creates a new disposable database.  Everything retained below
    # is a durable semantic inference/profitability field.
    path_result="$(scalar "${lstm_db}" "SELECT e.symbol||'|'||e.prediction_horizon||'|'||e.threshold_logret||'|'||e.window_size||'|'||e.label_rule_id||'|'||e.target_type||'|'||e.from_date||'|'||e.to_date||'|'||e.completed_epochs||'|'||e.accuracy||'|'||e.accept_model||'|'||COALESCE(e.reject_reason,'')||'|'||e.pred_down||'|'||e.pred_neutral||'|'||e.pred_up||'|'||p.prediction_count||'|'||p.actionable_count||'|'||p.winning_actionable_count||'|'||p.losing_actionable_count||'|'||p.gross_positive_terminal_horizon_log_return_sum||'|'||p.gross_negative_terminal_horizon_log_return_sum||'|'||p.aggregate_terminal_horizon_log_return_sum||'|'||COALESCE(p.average_terminal_horizon_log_return_per_actionable_prediction::text,'')||'|'||p.metric_definition_canonical||'|'||p.metric_definition_hash||'|'||p.source_content_hash FROM inference_eval_result e JOIN inference_profitability_observation p ON p.inference_eval_result_id=e.id WHERE e.model_id=${model_id} AND e.inference_scope='final'")"
    printf 'PHASE23A2_PATH,label=%s,iteration=%s,fixture_model_id=%s,worker_attempt_id=%s,durable_semantic_sha256=%s\n' \
        "${label}" "${iteration}" "${model_id}" "${attempt}" \
        "$(printf '%s' "${path_result}" | shasum -a 256 | awk '{print $1}')"
}

run_live_reaper_regression() {
    local lstm_db="${base_tag}_live1_lstm"
    local forex_db="${base_tag}_live1_forex"
    local root="${test_dir}/live1-registry"
    local log_dir="${test_dir}/live1-logs"
    local output="${test_dir}/live1.out"
    printf '%s\n%s\n' "${lstm_db}" "${forex_db}" >>"${database_list}"

    createdb "${lstm_db}"
    createdb "${forex_db}"
    PGOPTIONS='-c default_transaction_read_only=on' \
        pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
        psql -X -v ON_ERROR_STOP=1 -q -d "${lstm_db}"
    PGOPTIONS='-c default_transaction_read_only=on' \
        pg_dump -s -h 127.0.0.1 -U vjp -d forex |
        psql -X -v ON_ERROR_STOP=1 -q -d "${forex_db}"

    psql -X -v ON_ERROR_STOP=1 -q -d "${lstm_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO UPDATE SET desired_state='running';
INSERT INTO experiment_scheduler_lease(singleton)
VALUES(true) ON CONFLICT(singleton) DO NOTHING;
INSERT INTO experiment_scheduler_protocol(
    singleton,required_generation,cutover_state,cutover_completed_at,
    cutover_completed_by,cutover_executable_path,cutover_process_evidence,
    failure_diagnostic,updated_at)
VALUES(true,52,'complete',clock_timestamp(),'phase23a2-live-reaper-fixture',
       '/phase23a2/disposable/lstm-scheduler','disposable fixture',NULL,
       clock_timestamp())
ON CONFLICT(singleton) DO UPDATE SET required_generation=52,
    cutover_state='complete',cutover_completed_at=clock_timestamp(),
    cutover_completed_by='phase23a2-live-reaper-fixture',
    cutover_executable_path='/phase23a2/disposable/lstm-scheduler',
    cutover_process_evidence='disposable fixture',failure_diagnostic=NULL,
    updated_at=clock_timestamp();
SQL

    psql -X -v ON_ERROR_STOP=1 -q -d "${forex_db}" <<'SQL'
CREATE TABLE phase23a2audrmp(
    time timestamp without time zone NOT NULL,
    bid numeric(10,6) NOT NULL, ask numeric(10,6) NOT NULL,
    vol smallint NOT NULL
);
INSERT INTO phase23a2audrmp(time,bid,ask,vol)
SELECT timestamp '2024-01-01 00:00:00' + i * interval '5 minutes',
       (1.000000 + i * 0.000001)::numeric(10,6),
       (1.000200 + i * 0.000001)::numeric(10,6),1
FROM generate_series(0,2303) AS g(i);
SQL

    local fixture experiment_id model_id
    fixture="$(LSTM_DB_NAME="${lstm_db}" "${seeder}")"
    grep -Eq '^PHASE23A2_FIXTURE,experiment_id=[0-9]+,model_id=[0-9]+,snapshot_id=[0-9]+,snapshot_hash=fnv1a64:[0-9a-f]{16}$' <<<"${fixture}"
    experiment_id="$(sed -n 's/.*experiment_id=\([0-9]*\).*/\1/p' <<<"${fixture}")"
    model_id="$(sed -n 's/.*model_id=\([0-9]*\).*/\1/p' <<<"${fixture}")"
    make_registry "${root}" "${standalone}"

    LSTM_DB_NAME="${lstm_db}" FOREX_DB_NAME="${forex_db}" "${scheduler}" \
        --schedule-experiments \
        --max-train-procs=0 --max-infer-procs=1 --max-analyze-procs=0 \
        --scheduler-poll-seconds=1 \
        --semantic-worker-registry="${root}/registry.json" \
        --scheduler-log-dir="${log_dir}" >"${output}" 2>&1 &
    live_scheduler_pid=$!

    local state=""
    for _ in {1..6000}; do
        state="$(scalar "${lstm_db}" \
            "SELECT status||':'||phase FROM experiment WHERE experiment_id=${experiment_id}" 2>/dev/null || true)"
        [[ "${state}" = "pending:analyze" ]] && break
        if ! kill -0 "${live_scheduler_pid}" >/dev/null 2>&1; then
            wait "${live_scheduler_pid}" || true
            printf 'live-reaper scheduler exited before durable completion; state=%s\n' "${state}" >&2
            return 1
        fi
        sleep 0.02
    done
    test "${state}" = "pending:analyze"

    kill -TERM "${live_scheduler_pid}"
    wait "${live_scheduler_pid}"
    live_scheduler_pid=""

    grep -q "SCHEDULER_CHILD_EXITED,experiment_id=${experiment_id},.*phase=infer,.*exit_code=0" \
        "${output}"
    grep -q "SCHEDULER_CHILD_RESULT_PERSISTED,experiment_id=${experiment_id},model_id=${model_id},phase=infer,reason=completed_inference_eval_result" \
        "${output}"

    local completed_attempt
    completed_attempt="$(scalar "${lstm_db}" \
        "SELECT worker_attempt_id FROM experiment_scheduler_worker_attempt
         WHERE experiment_id=${experiment_id}
         ORDER BY worker_attempt_id DESC LIMIT 1")"

    test "$(scalar "${lstm_db}" \
        "SELECT lifecycle_state||':'||COALESCE(exit_code::text,'NULL')||':'||
                reconciliation_result
         FROM experiment_scheduler_worker_attempt
         WHERE worker_attempt_id=${completed_attempt}")" = \
        "completed:0:parent_observed_exit"

    test "$(scalar "${lstm_db}" \
        "SELECT count(*)
         FROM experiment e
         JOIN inference_eval_result r ON r.model_id=e.last_model_id
         WHERE e.experiment_id=${experiment_id}
           AND r.inference_scope='final'
           AND r.status='completed'
           AND r.threshold_logret <> e.c_next_threshold
           AND abs(r.threshold_logret-e.c_next_threshold) <= 1e-7")" = 1

    test "$(scalar "${lstm_db}" \
        "SELECT count(*) FROM inference_profitability_observation
         WHERE experiment_id=${experiment_id}
           AND model_id=${model_id}
           AND inference_scope='final'")" = 1

    printf 'PHASE23A2_LIVE_REAPER,experiment_id=%s,model_id=%s,worker_attempt_id=%s,state=%s\n' \
        "${experiment_id}" "${model_id}" "${completed_attempt}" "${state}"
}

run_live_reaper_regression

for iteration in 1 2; do
    run_path a "${compatibility}" "${iteration}"
    compatibility_result="${path_result}"
    if [[ "${iteration}" -eq 1 ]]; then
        compatibility_first_result="${compatibility_result}"
    else
        test "${compatibility_first_result}" = "${compatibility_result}"
    fi
    run_path b "${standalone}" "${iteration}"
    standalone_result="${path_result}"
    if [[ "${iteration}" -eq 1 ]]; then
        standalone_first_result="${standalone_result}"
    else
        test "${standalone_first_result}" = "${standalone_result}"
    fi
    test "${compatibility_result}" = "${standalone_result}"
done

printf '%s\n' 'LSTMPhase23A2ManagedInferenceEndToEndABTests passed'
