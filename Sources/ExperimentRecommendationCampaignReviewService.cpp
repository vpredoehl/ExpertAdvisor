#include "ExperimentRecommendationCampaignReviewService.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ExperimentRecommendationCampaignReview.hpp"
#include "ExperimentRecommendationService.hpp"

#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void PrintSafety(std::ostream& output)
{
    output << "read_only=true,campaign_created=false,proposal_created=false,"
              "experiment_created=false,experiment_modified=false,"
              "activation_created=false,scheduler_started=false,"
              "worker_started=false";
}

std::string ReasonsText(
    const std::vector<RecommendationCampaignReason>& reasons)
{
    if (reasons.empty()) return "NULL";
    std::ostringstream out;
    for (std::size_t index = 0; index < reasons.size(); ++index)
    {
        if (index != 0) out << ':';
        out << RecommendationCampaignReasonText(reasons[index]);
    }
    return RecommendationMachineText(out.str());
}

std::string RecommendationIdsText(const std::vector<long long>& ids)
{
    std::ostringstream out;
    for (std::size_t index = 0; index < ids.size(); ++index)
    {
        if (index != 0) out << ':';
        out << ids[index];
    }
    return RecommendationMachineText(out.str());
}

void PrintCandidate(
    std::ostream& output,
    const RecommendationCampaignReviewCandidate& candidate)
{
    output << "RECOMMENDATION_CAMPAIGN_REVIEW_CANDIDATE"
           << ",ordinal=" << candidate.ordinal
           << ",decision="
           << RecommendationCampaignDecisionText(candidate.decision)
           << ",recommendation_id=" << candidate.recommendationId
           << ",source_experiment_id=" << candidate.sourceExperimentId
           << ",ranking_member_id=" << candidate.rankingMemberId
           << ",ranking_position=" << candidate.rankingPosition
           << ",symbol=" << RecommendationMachineText(candidate.symbol)
           << ",prediction_horizon=" << candidate.predictionHorizon
           << ",family=" << RecommendationMachineText(candidate.family)
           << ",duplicate=" << (candidate.duplicate ? "true" : "false")
           << ",duplicate_invocation_identity_hash="
           << (candidate.duplicateIdentityHash
                   ? RecommendationMachineText(*candidate.duplicateIdentityHash)
                   : "NULL")
           << ",reason_count=" << candidate.reasons.size()
           << ",reason_codes=" << ReasonsText(candidate.reasons) << ',';
    PrintSafety(output);
    output << '\n';
}

void PrintTextCoverage(
    std::ostream& output,
    const char* event,
    const RecommendationCampaignReviewTextCoverage& coverage)
{
    output << event
           << ",value=" << RecommendationMachineText(coverage.value)
           << ",considered_count=" << coverage.consideredCount
           << ",selected_count=" << coverage.selectedCount
           << ",excluded_count=" << coverage.excludedCount << ',';
    PrintSafety(output);
    output << '\n';
}

} // namespace

int RunRecommendationCampaignReviewCommand(
    const std::string& connectionString,
    const RecommendationCampaignPlanningPolicy& policy,
    const RecommendationCampaignPlanningScope& scope,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        if (const auto error = ValidateRecommendationCampaignPlanningPolicy(
                policy))
            throw std::invalid_argument(*error);
        if (const auto error = ValidateRecommendationCampaignPlanningScope(
                scope))
            throw std::invalid_argument(*error);

        pqxx::connection connection{connectionString};
        if (!RecommendationCampaignPlanningSchemasExist(connection))
            throw std::runtime_error(
                "recommendation_campaign_planning_schemas_required");
        const RecommendationCampaignPlanInput input =
            LoadRecommendationCampaignPlanInput(connection, policy, scope);
        const RecommendationCampaignPlan plan =
            PlanRecommendationCampaign(input);
        const RecommendationCampaignReview review =
            ReviewRecommendationCampaignPlan(plan);

        output << "RECOMMENDATION_CAMPAIGN_REVIEW"
               << ",campaign_review_contract_version="
               << review.contractVersion
               << ",campaign_review_identity_hash="
               << RecommendationMachineText(review.identityHash)
               << ",campaign_plan_identity_hash="
               << RecommendationMachineText(review.campaignPlanIdentityHash)
               << ",policy_identity_hash="
               << RecommendationMachineText(review.policyIdentityHash)
               << ",scope_canonical="
               << RecommendationMachineText(review.scopeCanonical)
               << ",ranking_snapshot_id=" << scope.rankingSnapshotId
               << ",generated_at="
               << RecommendationMachineText(review.generatedAt)
               << ",candidate_count=" << review.summary.candidateCount
               << ",selected_count=" << review.summary.selectedCount
               << ",excluded_count=" << review.summary.excludedCount
               << ",duplicate_group_count="
               << review.summary.duplicateGroupCount
               << ",duplicate_candidate_count="
               << review.summary.duplicateCandidateCount
               << ",considered_family_count="
               << review.summary.consideredFamilyCount
               << ",selected_family_count="
               << review.summary.selectedFamilyCount
               << ",considered_symbol_count="
               << review.summary.consideredSymbolCount
               << ",selected_symbol_count="
               << review.summary.selectedSymbolCount
               << ",considered_horizon_count="
               << review.summary.consideredHorizonCount
               << ",selected_horizon_count="
               << review.summary.selectedHorizonCount
               << ",deterministic_ordering_verified="
               << (review.summary.deterministicOrderingVerified
                       ? "true" : "false") << ',';
        PrintSafety(output);
        output << '\n';

        for (const auto& candidate : review.selected)
            PrintCandidate(output, candidate);
        for (const auto& candidate : review.excluded)
            PrintCandidate(output, candidate);
        for (const auto& group : review.duplicateGroups)
        {
            output << "RECOMMENDATION_CAMPAIGN_REVIEW_DUPLICATE"
                   << ",recommendation_invocation_identity_hash="
                   << RecommendationMachineText(
                          group.recommendationInvocationIdentityHash)
                   << ",member_count=" << group.recommendationIds.size()
                   << ",recommendation_ids="
                   << RecommendationIdsText(group.recommendationIds) << ',';
            PrintSafety(output);
            output << '\n';
        }
        for (const auto& coverage : review.familyCoverage)
            PrintTextCoverage(
                output, "RECOMMENDATION_CAMPAIGN_REVIEW_FAMILY", coverage);
        for (const auto& coverage : review.symbolCoverage)
            PrintTextCoverage(
                output, "RECOMMENDATION_CAMPAIGN_REVIEW_SYMBOL", coverage);
        for (const auto& coverage : review.horizonCoverage)
        {
            output << "RECOMMENDATION_CAMPAIGN_REVIEW_HORIZON"
                   << ",prediction_horizon=" << coverage.horizon
                   << ",considered_count=" << coverage.consideredCount
                   << ",selected_count=" << coverage.selectedCount
                   << ",excluded_count=" << coverage.excludedCount << ',';
            PrintSafety(output);
            output << '\n';
        }

        output << "RECOMMENDATION_CAMPAIGN_REVIEW_COMPLETE"
               << ",campaign_review_identity_hash="
               << RecommendationMachineText(review.identityHash)
               << ",candidate_count=" << review.summary.candidateCount
               << ",selected_count=" << review.summary.selectedCount
               << ",excluded_count=" << review.summary.excludedCount << ',';
        PrintSafety(output);
        output << "\nCampaign review is read-only; it explains the existing "
                  "deterministic plan and performs no campaign or workflow "
                  "action.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_REVIEW_INVALID,error="
               << RecommendationMachineText(error.what()) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
    catch (const std::exception& error)
    {
        const std::string message = error.what();
        if (message == "recommendation_campaign_ranking_snapshot_not_found")
        {
            errors << "RECOMMENDATION_CAMPAIGN_RANKING_SNAPSHOT_NOT_FOUND"
                   << ",ranking_snapshot_id=" << scope.rankingSnapshotId << ',';
            PrintSafety(errors);
            errors << '\n';
            return 1;
        }
        errors << "RECOMMENDATION_CAMPAIGN_REVIEW_FAILED,error="
               << RecommendationMachineText(message) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
