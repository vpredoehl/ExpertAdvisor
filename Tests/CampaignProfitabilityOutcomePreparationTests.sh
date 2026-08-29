#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_root}/DerivedData/Release/ProfitabilityPhase12Validation/Build/Products/Debug/LSTM_Release}"
cohort="fnv1a64:fe7aee4a1aed8a5e"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_name="${LSTM_DB_NAME:-LSTM}"
tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

fingerprint() {
    psql -X -Atq -h "${db_host}" -U pqxx -d "${db_name}" <<'SQL'
BEGIN READ ONLY;
SELECT count(*) FROM experiment;
SELECT count(*) FROM model;
SELECT count(*) FROM experiment_recommendation;
SELECT count(*) FROM experiment_recommendation_ranking_snapshot;
SELECT count(*) FROM inference_eval_result;
SELECT count(*) FROM inference_profitability_observation;
COMMIT;
SQL
}

before="$(fingerprint)"
"${binary}" --prepare-campaign-profitability-outcome-jobs="${cohort}" \
    >"${tmp_dir}/first.txt"
"${binary}" --prepare-campaign-profitability-outcome-jobs="${cohort}" \
    >"${tmp_dir}/second.txt"
cmp "${tmp_dir}/first.txt" "${tmp_dir}/second.txt"

test "$(grep -c '^CAMPAIGN_PROFITABILITY_OUTCOME_JOB,' "${tmp_dir}/first.txt")" = "23"
test "$(grep -c 'readiness=waiting_for_outcome_data' "${tmp_dir}/first.txt")" = "22"
test "$(grep -c 'readiness=incompatible_source' "${tmp_dir}/first.txt")" = "1"
grep -q 'artifact_identity_verified=true' "${tmp_dir}/first.txt"
grep -q 'artifact_sha256=8d2176ef26513f6b69d690fa5b550a870aab7190221bace24a18be9200a89bbc' \
    "${tmp_dir}/first.txt"
grep -q 'proof_future_outcome_not_executed=true' "${tmp_dir}/first.txt"
grep -q 'training_started=false' "${tmp_dir}/first.txt"
grep -q 'scheduler_modified=false' "${tmp_dir}/first.txt"

if "${binary}" --prepare-campaign-profitability-outcome-jobs=fnv1a64:0000000000000000 \
    >"${tmp_dir}/bad-cohort.out" 2>"${tmp_dir}/bad-cohort.err"; then
    echo "expected cohort hash mismatch rejection" >&2
    exit 1
fi
grep -q 'phase12_validation_cohort_hash_mismatch' "${tmp_dir}/bad-cohort.err"

job_line="$(grep '^CAMPAIGN_PROFITABILITY_OUTCOME_JOB,' "${tmp_dir}/first.txt" | \
    grep 'compatibility_state=compatible,' | head -1)"
field() {
    printf '%s\n' "${job_line}" | tr ',' '\n' | sed -n "s/^${1}=//p"
}
experiment_id="$(field source_experiment_id)"
model_id="$(field source_model_id)"
job_hash="$(field job_hash)"
execution_option="--run-frozen-model-outcome-inference=${cohort},${experiment_id},${model_id},2026-08-31,2026-09-30,${job_hash}"

if "${binary}" "${execution_option}" \
    >"${tmp_dir}/early.out" 2>"${tmp_dir}/early.err"; then
    echo "expected incomplete future-window rejection" >&2
    exit 1
fi
grep -q 'phase12_execution_waiting_for_outcome_data' "${tmp_dir}/early.err"
test "$(grep -c 'CAMPAIGN_PROFITABILITY_OUTCOME_EXECUTION_START' \
    "${tmp_dir}/early.out" || true)" = "0"

if "${binary}" "${execution_option}" --epochs=1 \
    >"${tmp_dir}/override.out" 2>"${tmp_dir}/override.err"; then
    echo "expected training override rejection" >&2
    exit 1
fi
grep -q 'rejects training, model, feature, scheduler, and inference semantic overrides' \
    "${tmp_dir}/override.err"

after="$(fingerprint)"
test "${after}" = "${before}"

echo "CAMPAIGN_PROFITABILITY_PHASE12_PREPARATION_TESTS_PASS"
