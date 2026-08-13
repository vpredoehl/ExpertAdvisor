#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h2-concurrency.XXXXXX)"
baseline_marker="$tmp_root/baseline"
cluster_root=""

cleanup() {
  if [[ -n "$cluster_root" && -f "$cluster_root/data/postmaster.pid" ]]; then
    pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null 2>&1 || true
  fi
  if [[ -n "$cluster_root" && -d "$cluster_root" ]]; then
    rm -rf -- "$cluster_root"
  fi
  rm -rf -- "$tmp_root"
}
trap cleanup EXIT

# The workflow harness builds the canonical Phase E graph and exits only after
# all setup sessions have disconnected and the database has been proven quiet.
H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
H2_PRESERVE_BASELINE_ROOT="$baseline_marker" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/baseline.log" 2>&1

IFS='|' read -r cluster_root database cluster_socket < "$baseline_marker"
[[ -n "$cluster_root" && -n "$database" && -n "$cluster_socket" ]]
rg -q 'H2_BASELINE_QUIESCENT .*sessions=0' "$tmp_root/baseline.log"

target=(-h "$cluster_socket" -p 5432 -U campaign_manager_login)
active_sessions="$(psql "${target[@]}" -At postgres -c \
  "SELECT pid||'|'||application_name||'|'||state
     FROM pg_stat_activity
    WHERE datname='$database' AND pid <> pg_backend_pid()
    ORDER BY pid;")"
[[ -z "$active_sessions" ]]
echo "H2_TEMPLATE_QUIESCENT database=$database sessions=0"

clone_database() {
  local clone="$1"
  createdb "${target[@]}" -T "$database" "$clone"
  echo "H2_RACE_DATABASE clone=$clone source=$database status=created"
}

suffix="$$"
c1="expertadvisor_h2_c1_${suffix}"
c2="expertadvisor_h2_c2_${suffix}"
d1="expertadvisor_h2_d1_${suffix}"
d2="expertadvisor_h2_d2_${suffix}"
same="expertadvisor_h2_same_${suffix}"
different="expertadvisor_h2_different_${suffix}"
for clone in "$c1" "$c2" "$d1" "$d2" "$same" "$different"; do
  clone_database "$clone"
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

race_test="$tmp_root/CampaignOperationsPhaseH2ConcurrencyTests"
clang++ -std=c++20 -Wall -Wextra -Werror -ffunction-sections -fdata-sections \
  -DCAMPAIGN_OPERATIONS_H2_TESTING "${includes[@]}" \
  "$repo_root/Tests/CampaignOperationsPhaseH2ConcurrencyTests.cpp" \
  "${sources[@]}" -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -Wl,-dead_strip \
  -o "$race_test"

"$race_test" "$cluster_socket" "$c1" "$c2" "$d1" "$d2" "$same" \
  "$different" campaign_manager_login | tee "$tmp_root/races.log"
rg -q 'H2_FOUR_RACE_SUITE_OK C1=PASS C2=PASS D1=PASS D2=PASS E_SAME_KEY=PASS F_DIFFERENT_KEY=PASS' \
  "$tmp_root/races.log"

echo "H2_CONCURRENCY_HARNESS_OK quiescent_template=PASS independent_sessions=PASS "\
  "blocking_catalog=PASS exact_final_state=PASS four_races=PASS"
