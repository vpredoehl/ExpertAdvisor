#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
scheduler_binary="${1:?usage: $0 /path/to/isolated/LSTM_Release}"
test_db="ea_training_objective_scheduler_${$}"
test_dir="$(mktemp -d /tmp/ea_training_objective_scheduler.XXXXXX)"
safe_ps_path="${repo_root}/Tests/fixtures/scheduler_protocol_cutover_bin"

case "${test_db}" in
    ea_training_objective_scheduler_[0-9]*) ;;
    *) exit 90 ;;
esac

cleanup() {
    dropdb --if-exists "${test_db}" >/dev/null 2>&1 || true
    rm -rf -- "${test_dir}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

createdb "${test_db}"
pg_dump -s -h 127.0.0.1 -U vjp -d LSTM |
    psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/051_scheduler_ownership_and_worker_attempts.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" \
    -f "${repo_root}/Database/migrations/052_scheduler_protocol_and_exact_attempt_hardening.sql"
psql -X -v ON_ERROR_STOP=1 -q -d "${test_db}" <<'SQL'
INSERT INTO experiment_global_control(singleton,desired_state)
VALUES(true,'running') ON CONFLICT(singleton) DO NOTHING;
SQL

PATH="${safe_ps_path}:${PATH}" \
EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --complete-scheduler-protocol-cutover --yes \
    >"${test_dir}/cutover.out" 2>&1
grep -Eq 'result=(completed|already_complete)' "${test_dir}/cutover.out"

common=(
    --queue-experiment
    --symbol=usdcadrmp
    --prediction-horizon=6
    --threshold=0.0008
    --core-lr=120
    --head-lr=25
    --checkpoint-interval=20
    --train-start=2010-01-01
    --train-end=2025-01-01
    --infer-start=2025-01-01
    --infer-end=2026-01-01
    --donchian20-mode=enabled
    --feature-warmup-scope=legacy_cold_boundary
    --donchian-lookback=20
)

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    "${common[@]}" --target-epochs=81 \
    >"${test_dir}/default.out" 2>&1
grep -q 'training_objective_id=legacy_first_hit_weighted_ce_v1' \
    "${test_dir}/default.out"

LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    "${common[@]}" --target-epochs=82 --training-objective=legacy \
    >"${test_dir}/explicit-legacy.out" 2>&1
grep -q 'training_objective_id=legacy_first_hit_weighted_ce_v1' \
    "${test_dir}/explicit-legacy.out"

# Identical queue identity except for objective must create two experiments.
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    "${common[@]}" --target-epochs=83 --training-objective=legacy \
    >"${test_dir}/pair-control.out" 2>&1
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    "${common[@]}" --target-epochs=83 \
    --training-objective=profitability_auxiliary_v1 \
    >"${test_dir}/pair-treatment.out" 2>&1
grep -q 'training_objective_id=first_hit_weighted_ce_terminal_log_return_huber_aux_v1' \
    "${test_dir}/pair-treatment.out"

test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE target_epochs=83")" = 2
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(DISTINCT training_objective_hash) FROM experiment WHERE target_epochs=83")" = 2
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE target_epochs IN (81,82) AND training_objective_id='legacy_first_hit_weighted_ce_v1' AND training_objective_hash='fnv1a64:65818f2e1fa1a324'")" = 2
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE target_epochs=83 AND training_objective_id='first_hit_weighted_ce_terminal_log_return_huber_aux_v1' AND training_objective_hash='fnv1a64:f7a9a20f7f72eee5' AND auxiliary_loss_coefficient=0.1 AND robust_loss_delta=1")" = 1

# Repeating the treatment is an objective-aware duplicate, not a new row.
set +e
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    "${common[@]}" --target-epochs=83 \
    --training-objective=profitability_auxiliary_v1 \
    >"${test_dir}/duplicate-treatment.out" 2>&1
duplicate_status=$?
set -e
test "${duplicate_status}" = 3
grep -q '^QUEUE_ALREADY_EXISTS' "${test_dir}/duplicate-treatment.out"
test "$(psql -X -At -q -d "${test_db}" -c \
    "SELECT count(*) FROM experiment WHERE target_epochs=83")" = 2

set +e
"${scheduler_binary}" "${common[@]}" --target-epochs=84 --dry-run \
    --training-objective=unknown_v99 \
    >"${test_dir}/unknown.out" 2>&1
unknown_status=$?
set -e
test "${unknown_status}" -ne 0
grep -q 'unsupported --training-objective: unknown_v99' \
    "${test_dir}/unknown.out"

# Queue dry-run is non-mutating and names both objective identifier and hash.
"${scheduler_binary}" "${common[@]}" --target-epochs=84 --dry-run \
    --training-objective=profitability_auxiliary_v1 \
    >"${test_dir}/queue-dry-run.out" 2>&1
grep -q 'training_objective_id=first_hit_weighted_ce_terminal_log_return_huber_aux_v1' \
    "${test_dir}/queue-dry-run.out"
grep -q 'training_objective_hash=fnv1a64:f7a9a20f7f72eee5' \
    "${test_dir}/queue-dry-run.out"

# Scheduler dry-run constructs, but never launches, both persisted child
# commands. Each argv carries the exact persisted objective identifier.
PATH="${safe_ps_path}:${PATH}" \
EA_SCHEDULER_PROTOCOL_TEST_PS_MODE=safe \
LSTM_DB_NAME="${test_db}" "${scheduler_binary}" \
    --schedule-experiments --scheduler-once --dry-run \
    --max-train-procs=10 --max-infer-procs=1 --max-analyze-procs=1 \
    --scheduler-log-dir="${test_dir}/logs" \
    >"${test_dir}/scheduler-dry-run.out" 2>&1
grep -Eq 'EXPERIMENT_CHILD_COMMAND,.*phase=train,dry_run=1,argv=.*--training-objective[ =]legacy_first_hit_weighted_ce_v1' \
    "${test_dir}/scheduler-dry-run.out"
grep -Eq 'EXPERIMENT_CHILD_COMMAND,.*phase=train,dry_run=1,argv=.*--training-objective[ =]first_hit_weighted_ce_terminal_log_return_huber_aux_v1' \
    "${test_dir}/scheduler-dry-run.out"

printf '%s\n' 'TrainingObjectiveSchedulerIntegrationTests passed'
