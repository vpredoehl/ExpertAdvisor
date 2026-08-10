#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
tmp_root="$(mktemp -d /tmp/ea-h2-replay.XXXXXX)"
trap 'rm -rf -- "$tmp_root"' EXIT

H2_PRESERVE_BASELINE_ROOT="$tmp_root/baseline" \
  H2_PRESERVE_CLUSTER_ROOT="$tmp_root/cluster-root" \
  bash "$repo_root/Tests/CampaignOperationsPhaseH2WorkflowTests.sh" \
  >"$tmp_root/baseline.log" 2>&1
IFS='|' read -r cluster_root database cluster_socket < "$tmp_root/baseline"

includes=(-I "$repo_root/Sources" -I "$repo_root/Headers"
  -I /opt/homebrew/opt/libpqxx@7.10.1/include -I /opt/homebrew/opt/libpq/include)
sources=(
  "$repo_root/Sources/CampaignOperations.cpp"
  "$repo_root/Sources/CampaignOperationsDispatch.cpp"
  "$repo_root/Sources/CampaignOperationsControl.cpp"
  "$repo_root/Sources/CampaignOperationsCompletion.cpp"
  "$repo_root/Sources/CampaignOperationsRepository.cpp"
  "$repo_root/Sources/CampaignOperationsService.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchRepository.cpp"
  "$repo_root/Sources/CampaignOperationsManager.cpp"
  "$repo_root/Sources/CampaignOperationsDispatchService.cpp"
  "$repo_root/Sources/CampaignOperationsBindingRepository.cpp"
  "$repo_root/Sources/CampaignOperationsControlRepository.cpp"
  "$repo_root/Sources/CampaignOperationsControlService.cpp"
  "$repo_root/Sources/CampaignOperationsCompletionRepository.cpp"
  "$repo_root/Sources/CampaignOperationsCompletionService.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmission.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionRepository.cpp"
  "$repo_root/Sources/CampaignOperationsProductionAdmissionService.cpp"
  "$repo_root/Sources/ExperimentRecommendation.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposal.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalRatification.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalReview.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignFollowUpProposalReviewRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignOutcomeAssessment.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignOutcomePolicy.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignReview.cpp"
  "$repo_root/Sources/ExperimentRecommendationRanking.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignMaterialization.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignApproval.cpp"
  "$repo_root/Sources/ExperimentRecommendationCampaignPlanning.cpp"
  "$repo_root/Sources/ExperimentRecommendationReview.cpp"
  "$repo_root/Sources/ExperimentRecommendationEvaluation.cpp"
  "$repo_root/Sources/ExperimentRecommendationScoring.cpp"
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
  "$repo_root/Sources/ExperimentRecommendationConversion.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionRepository.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflow.cpp"
  "$repo_root/Sources/ExperimentRecommendationConversionWorkflowRepository.cpp")
binary="$tmp_root/CampaignOperationsPhaseH2ReplayAssociationTests"
clang++ -std=c++20 -Wall -Wextra -Werror -DCAMPAIGN_OPERATIONS_H2_TESTING \
  "${includes[@]}" \
  "$repo_root/Tests/CampaignOperationsPhaseH2ReplayAssociationTests.cpp" \
  "${sources[@]}" -L /opt/homebrew/opt/libpqxx@7.10.1/lib \
  -L /opt/homebrew/opt/libpq/lib -lpqxx -lpq -o "$binary"

connection_for() {
  printf 'host=%s port=5432 dbname=%s user=%s application_name=h2-replay-%s' \
    "$cluster_socket" "$database" "$1" "$2"
}
"$binary" "$(connection_for h2_enabler_login enabler)" \
  "$(connection_for h2_manager_login manager)" \
  "$(connection_for campaign_manager_login recovery)"

pg_ctl -D "$cluster_root/data" -m immediate stop >/dev/null
echo "H2_REPLAY_ASSOCIATION_SCRIPT_OK disposable_database=$database"
