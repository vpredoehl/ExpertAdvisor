#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_root}/DerivedData/ExpertAdvisor/Build/Products/Release/LSTM_Release}"
cohort="fnv1a64:fe7aee4a1aed8a5e"
db_host="${LSTM_DB_HOST:-127.0.0.1}"
db_name="${LSTM_DB_NAME:-LSTM}"
tmp_dir="$(mktemp -d /tmp/ea_profitability_phase13_integration.XXXXXX)"
trap 'rm -rf -- "${tmp_dir}"' EXIT

fingerprint() {
    PGOPTIONS='-c default_transaction_read_only=on' \
    psql -X -Atq -h "${db_host}" -U pqxx -d "${db_name}" <<'SQL'
BEGIN ISOLATION LEVEL REPEATABLE READ READ ONLY;
SELECT count(*) FROM experiment;
SELECT count(*) FROM model;
SELECT count(*) FROM experiment_recommendation;
SELECT count(*) FROM experiment_recommendation_ranking_snapshot;
SELECT to_regclass('public.campaign_profitability_prospective_outcome_result');
COMMIT;
SQL
}

before="$(fingerprint)"
set +e
PGOPTIONS='-c default_transaction_read_only=on' \
    "${binary}" --compare-campaign-profitability-prospective="${cohort}" \
    >"${tmp_dir}/first.txt" 2>"${tmp_dir}/first.err"
first_status=$?
PGOPTIONS='-c default_transaction_read_only=on' \
    "${binary}" --compare-campaign-profitability-prospective="${cohort}" \
    >"${tmp_dir}/second.txt" 2>"${tmp_dir}/second.err"
second_status=$?
set -e

test "${first_status}" = "4"
test "${second_status}" = "4"
cmp "${tmp_dir}/first.txt" "${tmp_dir}/second.txt"
cmp "${tmp_dir}/first.err" "${tmp_dir}/second.err"
test ! -s "${tmp_dir}/first.err"

grep -q 'phase11_artifact_sha256=8d2176ef26513f6b69d690fa5b550a870aab7190221bace24a18be9200a89bbc' \
    "${tmp_dir}/first.txt"
grep -q 'phase12_preparation_artifact_sha256=441a1957ac2ffd53a0693e2332d2b9f7af1c1f9977ff688f9d8e043199f581e1' \
    "${tmp_dir}/first.txt"
grep -q 'metric_hash=fnv1a64:5894ecab3036bb93' "${tmp_dir}/first.txt"
grep -q 'required_source_model_count=23,covered_source_model_count=0' \
    "${tmp_dir}/first.txt"
grep -q 'known_incompatible_source_model_id=499' "${tmp_dir}/first.txt"
grep -q 'top_n=5,.*entrant_recommendation_ids=410:417:418,exit_recommendation_ids=359:360:361' \
    "${tmp_dir}/first.txt"
grep -q 'top_n=10,.*entrant_recommendation_ids=407:411:417,exit_recommendation_ids=360:361:378' \
    "${tmp_dir}/first.txt"
grep -q 'top_n=20,.*entrant_recommendation_ids=408:409,exit_recommendation_ids=369:370' \
    "${tmp_dir}/first.txt"
grep -q 'top_n=5,.*changed_selection_required_source_model_count=3,changed_selection_covered_source_model_count=0' \
    "${tmp_dir}/first.txt"
grep -q 'top_n=10,.*changed_selection_required_source_model_count=5,changed_selection_covered_source_model_count=0' \
    "${tmp_dir}/first.txt"
grep -q 'top_n=20,.*changed_selection_required_source_model_count=2,changed_selection_covered_source_model_count=0' \
    "${tmp_dir}/first.txt"
test "$(grep -c 'candidate_minus_control_incremental_profitability=PENDING' \
    "${tmp_dir}/first.txt")" = "3"
test "$(grep -c 'readiness=pending_outcomes' "${tmp_dir}/first.txt")" = "4"
test "$(grep -c 'final=false' "${tmp_dir}/first.txt")" = "4"
test "$(grep -c 'economic_weighting=recommendation_selection_slot' \
    "${tmp_dir}/first.txt")" = "3"
test "$(grep -c 'duplicate_recommendations_are_independent=false' \
    "${tmp_dir}/first.txt")" = "3"
test "$(grep -c 'training_started=false' "${tmp_dir}/first.txt")" = "5"
test "$(grep -c 'scheduler_modified=false' "${tmp_dir}/first.txt")" = "5"
test "$(grep -c 'database_write=false' "${tmp_dir}/first.txt")" = "5"

after="$(fingerprint)"
test "${after}" = "${before}"

echo "CAMPAIGN_PROFITABILITY_PHASE13_INTEGRATION_TESTS_PASS"
