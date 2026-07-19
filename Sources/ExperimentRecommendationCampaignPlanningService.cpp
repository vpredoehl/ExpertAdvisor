#include "ExperimentRecommendationCampaignPlanningService.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ExperimentRecommendationService.hpp"

#include <cmath>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

template <typename Value>
std::string OptionalNumber(const std::optional<Value>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalDouble(const std::optional<double>& value)
{
    if (!value) return "NULL";
    return std::isfinite(*value)
        ? CanonicalRecommendationDouble(*value)
        : "INVALID_NONFINITE";
}

std::string Number(double value)
{
    return std::isfinite(value)
        ? CanonicalRecommendationDouble(value)
        : "INVALID_NONFINITE";
}

std::string OptionalText(const std::optional<std::string>& value)
{
    return value ? RecommendationMachineText(*value) : "NULL";
}

std::string ReasonsText(
    const std::vector<RecommendationCampaignReason>& reasons)
{
    if (reasons.empty()) return "NULL";
    std::ostringstream out;
    for (std::size_t i = 0; i < reasons.size(); ++i)
    {
        if (i != 0) out << ':';
        out << RecommendationCampaignReasonText(reasons[i]);
    }
    return RecommendationMachineText(out.str());
}

void PrintSafety(std::ostream& output)
{
    output << "read_only=true,proposal_created=false,experiment_created=false,"
              "experiment_modified=false,activation_created=false,"
              "scheduler_started=false,worker_started=false";
}

void PrintCandidate(
    std::ostream& output,
    const RecommendationCampaignPlanCandidate& candidate)
{
    const auto& value = candidate.input;
    const RecommendationCampaignWorkflowEvidence* workflow =
        value.workflows.empty() ? nullptr : &value.workflows.back();
    output << "RECOMMENDATION_CAMPAIGN_PLAN_CANDIDATE"
           << ",ordinal=" << candidate.ordinal
           << ",decision="
           << RecommendationCampaignDecisionText(candidate.decision)
           << ",recommendation_id=" << value.recommendationId
           << ",source_experiment_id=" << value.sourceExperimentId
           << ",symbol=" << RecommendationMachineText(value.symbol)
           << ",prediction_horizon=" << value.predictionHorizon
           << ",family=" << RecommendationMachineText(value.family)
           << ",target_epochs=" << OptionalNumber(value.targetEpochs)
           << ",ranking_snapshot_id=" << value.rankingSnapshotId
           << ",ranking_member_id=" << value.rankingMemberId
           << ",ranking_position=" << value.rankingPosition
           << ",ranking_score=" << OptionalDouble(value.rankingScore)
           << ",recommendation_semantic_hash="
           << RecommendationMachineText(value.recommendationSemanticHash)
           << ",recommendation_invocation_hash="
           << RecommendationMachineText(value.recommendationInvocationHash)
           << ",leader_score=" << Number(value.leaderScore)
           << ",inference_accuracy=" << Number(value.inferenceAccuracy)
           << ",predicted_neutral_proportion="
           << OptionalDouble(value.predictedNeutralProportion)
           << ",profitability_metric="
           << OptionalDouble(value.profitabilityMetric)
           << ",profitability_metric_identity="
           << OptionalText(value.profitabilityMetricIdentity)
           << ",proposal_id="
           << (workflow ? std::to_string(workflow->proposalId) : "NULL")
           << ",latest_review_decision_id="
           << (workflow ? OptionalNumber(workflow->latestReviewDecisionId)
                        : "NULL")
           << ",execution_id="
           << (workflow ? OptionalNumber(workflow->executionId) : "NULL")
           << ",activation_id="
           << (workflow ? OptionalNumber(workflow->activationId) : "NULL")
           << ",converted_experiment_id="
           << (workflow ? OptionalNumber(workflow->convertedExperimentId)
                        : "NULL")
           << ",workflow_state="
           << (workflow
                   ? RecommendationConversionWorkflowStateText(workflow->state)
                   : "NULL")
           << ",workflow_integrity="
           << (workflow
                   ? RecommendationConversionWorkflowIntegrityText(
                         workflow->integrity)
                   : "NULL")
           << ",reason_count=" << candidate.reasons.size()
           << ",reason_codes=" << ReasonsText(candidate.reasons) << ',';
    PrintSafety(output);
    output << '\n';
}

} // namespace

int RunRecommendationCampaignPlanningCommand(
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
        RecommendationCampaignPlanInput input =
            LoadRecommendationCampaignPlanInput(connection, policy, scope);
        const RecommendationCampaignPlan plan =
            PlanRecommendationCampaign(input);

        output << "RECOMMENDATION_CAMPAIGN_PLAN"
               << ",campaign_plan_contract_version=" << plan.contractVersion
               << ",campaign_plan_identity_hash="
               << RecommendationMachineText(plan.identityHash)
               << ",policy_identity_hash="
               << RecommendationMachineText(plan.policyHash)
               << ",policy_canonical="
               << RecommendationMachineText(plan.policyCanonical)
               << ",scope_canonical="
               << RecommendationMachineText(plan.scopeCanonical)
               << ",ranking_snapshot_id=" << scope.rankingSnapshotId
               << ",ranking_snapshot_identity_hash="
               << RecommendationMachineText(
                      input.rankingSnapshotIdentityHash)
               << ",generated_at="
               << RecommendationMachineText(plan.generatedAt)
               << ",candidate_count=" << plan.summary.candidateCount
               << ",selected_count=" << plan.summary.selectedCount
               << ",excluded_count=" << plan.summary.excludedCount << ',';
        PrintSafety(output);
        output << '\n';
        for (const auto& candidate : plan.candidates)
            PrintCandidate(output, candidate);
        output << "RECOMMENDATION_CAMPAIGN_PLAN_COMPLETE,count="
               << plan.summary.candidateCount
               << ",selected_count=" << plan.summary.selectedCount
               << ",excluded_count=" << plan.summary.excludedCount << ',';
        PrintSafety(output);
        output << "\nCampaign planning is read-only; selected candidates were "
                  "not proposed, reviewed, executed, activated, queued, or "
                  "started.\n";
        return 0;
    }
    catch (const std::invalid_argument& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_PLAN_INVALID,error="
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
        errors << "RECOMMENDATION_CAMPAIGN_PLAN_FAILED,error="
               << RecommendationMachineText(message) << ',';
        PrintSafety(errors);
        errors << '\n';
        return 2;
    }
}

} // namespace EA::ExperimentRecommendation
