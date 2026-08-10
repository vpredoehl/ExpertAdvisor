#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h3-compatibility.XXXXXX)"
cluster_root=""
cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

baseline="$tmp_root/baseline"
H3_EXTRA_FIXTURE_IDS=73,74 \
H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
H2_PRESERVE_BASELINE_ROOT="$baseline" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/baseline.log" 2>&1
IFS='|' read -r cluster_root database socket < "$baseline"
target=(-h "$socket" -p 5432 -U campaign_manager_login)
# The repository runner sees the working-tree migration list while producing
# its H2 fixture.  Reconstruct the exact accepted 057 schema on this
# disposable database before creating the historical operation; 058 is purely
# additive, so removing only its objects and ledger leaves the H2 fixture
# intact.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
DROP TRIGGER IF EXISTS campaign_operations_manager_attempt_complete
    ON campaign_operations_dispatch_attempt;
DROP FUNCTION IF EXISTS validate_campaign_operations_manager_attempt_completeness();
DROP TABLE IF EXISTS campaign_operations_h2_manager_key_compatibility CASCADE;
DROP TABLE IF EXISTS campaign_operations_dispatch_manager_operation CASCADE;
DROP FUNCTION IF EXISTS reject_campaign_operations_h2_manager_key_compat_mutation();
DROP FUNCTION IF EXISTS reject_campaign_operations_manager_operation_mutation();
DELETE FROM schema_migrations WHERE version='058';
SQL
[[ "$(psql "${target[@]}" -At "$database" -c "SELECT to_regclass('campaign_operations_h2_manager_key_compatibility') IS NULL AND to_regclass('campaign_operations_dispatch_manager_operation') IS NULL;")" == t ]]
enabler="host=$socket port=5432 dbname=$database user=h2_enabler_login application_name=h3-compat-enabler"
manager="host=$socket port=5432 dbname=$database user=h2_manager_login application_name=h3-compat-manager"

includes=(-I "$repo_root/Sources" -I "$repo_root/Headers"
  -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include)
sources=(
  "$repo_root/Sources/CampaignOperations.cpp"
  "$repo_root/Sources/CampaignOperationsDispatch.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp"
  "$repo_root/Sources/CampaignOperationsRepository.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchRepository.cpp"
  "$repo_root/Sources/CampaignOperationsManager.cpp"
  "$repo_root/Sources/CampaignOperationsManagerService.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchService.cpp"
  "$repo_root/Sources/CampaignOperationsBindingRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp"
  "$repo_root/Sources/ExperimentRecommendation.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignActivation.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignActivationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignMaterializationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoff.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignHandoffRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalRatificationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionExecutionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionActivation.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionActivationRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignExecution.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignExecutionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunch.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignLaunchRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReview.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionProposalReviewRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCandidateGenerator.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp")
harness="$tmp_root/CampaignOperationsPhaseH3CompatibilityTests"
clang++ -std=c++20 -Wall -Wextra -Werror \
  -DCAMPAIGN_OPERATIONS_H2_TESTING -DCAMPAIGN_OPERATIONS_H3_TESTING \
  "${includes[@]}" "$repo_root/Tests/CampaignOperationsPhaseH3CompatibilityTests.cpp" \
  "${sources[@]}" -L /opt/homebrew/opt/libpqxx@7.10.1/lib -L /opt/homebrew/opt/libpq/lib \
  -lpqxx -lpq -Wl,-dead_strip -o "$harness"

"$harness" pre "$enabler" "$manager" | tee "$tmp_root/pre.log"
rg -q 'H3_COMPAT_PRE058 exact_h2_replay=PASS recoverable_h2_attempt=PASS' "$tmp_root/pre.log"

pre_bytes="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT attempt_identity_canonical||'|'||attempt_identity_hash FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND operation_key='mgr-v1:legacy-h2-exact';")"
pre_counts="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE operational_request_id IN (71,72,73,74);")"

psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -f \
  "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql"
checksum="$(shasum -a 256 "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql" | awk '{print $1}')"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "INSERT INTO schema_migrations(version,filename,checksum) VALUES ('058','058_campaign_operations_h3_manager_run_once.sql','$checksum');"
[[ "$(psql "${target[@]}" -At "$database" -c "SELECT count(*) FROM campaign_operations_h2_manager_key_compatibility WHERE operation_key LIKE 'mgr-v1:legacy-h2-%';")" == 2 ]]
[[ "$(psql "${target[@]}" -At "$database" -c "SELECT count(*) FROM campaign_operations_dispatch_manager_operation;")" == 0 ]]

"$harness" post "$enabler" "$manager" | tee "$tmp_root/post.log"
rg -q 'H3_COMPAT_POST058 historical_exact=PASS historical_conflict=PASS historical_recovery=PASS new_prefix_rejected=PASS ordinary_h2=PASS manager_only=PASS' "$tmp_root/post.log"

post_bytes="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT attempt_identity_canonical||'|'||attempt_identity_hash FROM campaign_operations_dispatch_attempt WHERE operational_request_id=71 AND operation_key='mgr-v1:legacy-h2-exact';")"
post_counts="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT count(*) FROM campaign_operations_dispatch_attempt WHERE operational_request_id IN (71,72,73,74);")"
[[ "$pre_bytes" == "$post_bytes" ]]
[[ "$post_counts" == "$((pre_counts + 2))" ]]
[[ "$(psql "${target[@]}" -At "$database" -c "SELECT count(*) FROM campaign_operations_dispatch_manager_operation m JOIN campaign_operations_dispatch_attempt a ON a.dispatch_attempt_id=m.dispatch_attempt_id WHERE a.operation_key LIKE 'mgr-v1:legacy-h2-%';")" == 0 ]]
echo "H3_COMPAT_057_TO_058 inventory=PASS canonical_bytes=UNCHANGED no_historical_manager_source=PASS new_prefix_no_attempt=PASS"
