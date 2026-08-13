#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h3-runtime.XXXXXX)"
baseline_marker="$tmp_root/baseline"
cluster_root=""

cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

# H2 owns the repository-native disposable cluster and Phase E graph.  H3
# opts into four more complete candidates; no production database or role is
# referenced by this workflow.
H3_EXTRA_FIXTURE_IDS=73,74,75,76 \
H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
H2_PRESERVE_BASELINE_ROOT="$baseline_marker" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/baseline.log" 2>&1

IFS='|' read -r cluster_root database cluster_socket < "$baseline_marker"
[[ -n "$cluster_root" && -n "$database" && -n "$cluster_socket" ]]
target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)
[[ -z "$(psql "${target[@]}" -At postgres -c \
  "SELECT pid FROM pg_stat_activity WHERE datname='$database' AND pid<>pg_backend_pid();")" ]]

# H2 deliberately leaves H3's ledger entry untouched.  Install 058 only now,
# against the already-quiescent disposable database, and prove its ledger
# identity before cloning the template.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "DELETE FROM schema_migrations WHERE version='058';"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" \
  -f "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql"
migration_checksum="$(shasum -a 256 "$repo_root/Database/migrations/058_campaign_operations_h3_manager_run_once.sql" | awk '{print $1}')"
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" -c \
  "INSERT INTO schema_migrations(version,filename,checksum) VALUES \
   ('058','058_campaign_operations_h3_manager_run_once.sql','$migration_checksum');"
ledger="$(psql "${target[@]}" -At -v ON_ERROR_STOP=1 "$database" -c \
  "SELECT version||'|'||filename||'|'||checksum FROM schema_migrations WHERE version='058';")"
[[ "$ledger" == "058|058_campaign_operations_h3_manager_run_once.sql|$migration_checksum" ]]
echo "H3_MIGRATION058_DISPOSABLE_INSTALL table=PASS trigger=PASS ledger=PASS"

# The repository-native H1 fixture also contains historical request 7.  Keep
# that unrelated row in the disposable database, but make it ineligible so
# every H3 assertion names only the six candidates created by this harness.
psql "${target[@]}" -q -v ON_ERROR_STOP=1 "$database" <<'SQL'
SET session_replication_role=replica;
UPDATE campaign_operations_operational_request
   SET request_state='dispatching'
 WHERE operational_request_id NOT IN (71,72,73,74,75,76);
SET session_replication_role=origin;
SQL

clone_database() {
  local clone="$1"
  createdb "${target[@]}" -T "$database" "$clone"
}

suffix="$$"
databases=()
for label in local disable scheduler privilege database overlap unrelated source; do
  clone="expertadvisor_h3_${label}_${suffix}"
  clone_database "$clone"
  databases+=("$clone")
done

includes=(-I "$repo_root/Sources" -I "$repo_root/Headers"
  -I /opt/homebrew/opt/libpqxx@7.10.1/include
  -I /opt/homebrew/opt/libpq/include)
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

harness="$tmp_root/CampaignOperationsPhaseH3RuntimeConcurrencyTests"
clang++ -std=c++20 -Wall -Wextra -Werror -DNDEBUG \
  -DCAMPAIGN_OPERATIONS_H2_TESTING -DCAMPAIGN_OPERATIONS_H3_TESTING \
  "${includes[@]}" \
  "$repo_root/Tests/CampaignOperationsPhaseH3RuntimeConcurrencyTests.cpp" \
  "${sources[@]}" -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -Wl,-dead_strip \
  -o "$harness"

"$repo_root/Tests/CampaignOperationsPhaseH3ContractTests.sh"
"$harness" "$cluster_socket" "${databases[@]}" \
  | tee "$tmp_root/h3-runtime.log"
rg -q 'H3_RUNTIME_HARNESS_OK A=PASS B=PASS C=PASS D=PASS E=PASS F=PASS G=PASS H=PASS I=PASS J=PASS' \
  "$tmp_root/h3-runtime.log"
echo "H3_RUNTIME_CONCURRENCY_HARNESS_OK disposable_cluster=PASS migration058=PASS "\
  "actual_manager_run_once=PASS explicit_barriers=PASS blocking_catalog=PASS "\
  "scenarios_A_to_J=PASS"
