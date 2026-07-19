#include "ExperimentRecommendationCampaignPlanning.hpp"

#include "CanonicalSymbol.hpp"
#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <cmath>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalDoubleText(const std::optional<double>& value)
{
    if (!value) return "NULL";
    return std::isfinite(*value)
        ? CanonicalRecommendationDouble(*value)
        : "INVALID_NONFINITE";
}

std::string FiniteDoubleText(double value)
{
    return std::isfinite(value)
        ? CanonicalRecommendationDouble(value)
        : "INVALID_NONFINITE";
}

std::string OptionalIntText(const std::optional<int>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalLongText(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalFramedText(const std::optional<std::string>& value)
{
    return value ? LengthText(*value) : "NULL";
}

bool FiniteUnit(double value)
{
    return std::isfinite(value) && value >= 0.0 && value <= 1.0;
}

void AddReason(
    RecommendationCampaignPlanCandidate& candidate,
    RecommendationCampaignReason reason)
{
    if (std::find(candidate.reasons.begin(), candidate.reasons.end(), reason) ==
        candidate.reasons.end())
        candidate.reasons.push_back(reason);
}

bool IdentityPairValid(
    const std::string& canonical,
    const std::string& hash)
{
    return !canonical.empty() && hash == RecommendationCanonicalHash(canonical);
}

bool WorkflowIdentityValid(
    const RecommendationCampaignWorkflowEvidence& workflow)
{
    if (workflow.proposalId <= 0 ||
        !IdentityPairValid(
            workflow.proposalIdentityCanonical,
            workflow.proposalIdentityHash))
        return false;
    if (workflow.executionId)
    {
        if (!workflow.executionIdentityCanonical ||
            !workflow.executionIdentityHash ||
            !IdentityPairValid(
                *workflow.executionIdentityCanonical,
                *workflow.executionIdentityHash))
            return false;
    }
    if (workflow.activationId)
    {
        if (!workflow.activationIdentityCanonical ||
            !workflow.activationIdentityHash ||
            !IdentityPairValid(
                *workflow.activationIdentityCanonical,
                *workflow.activationIdentityHash))
            return false;
    }
    return true;
}

bool CandidateIdentityValid(const RecommendationCampaignCandidateInput& value)
{
    if (!value.persistedProvenanceValid ||
        value.rankingSnapshotId <= 0 || value.rankingMemberId <= 0 ||
        value.recommendationId <= 0 || value.sourceExperimentId <= 0 ||
        value.rankingPosition <= 0 || value.predictionHorizon <= 0 ||
        value.symbol.empty() || value.family.empty() ||
        value.rankingVersion <= 0)
        return false;
    if (!IdentityPairValid(
            value.rankingSnapshotIdentityCanonical,
            value.rankingSnapshotIdentityHash) ||
        !IdentityPairValid(
            value.rankingPolicyCanonical, value.rankingPolicyHash) ||
        !IdentityPairValid(
            value.recommendationSemanticCanonical,
            value.recommendationSemanticHash) ||
        !IdentityPairValid(
            value.recommendationInvocationCanonical,
            value.recommendationInvocationHash))
        return false;
    if (!FiniteUnit(value.leaderScore) ||
        !FiniteUnit(value.inferenceAccuracy) ||
        (value.predictedNeutralProportion &&
         !FiniteUnit(*value.predictedNeutralProportion)) ||
        (value.rankingScore && !FiniteUnit(*value.rankingScore)) ||
        (value.profitabilityMetric &&
         !std::isfinite(*value.profitabilityMetric)))
        return false;
    if (value.profitabilityMetric.has_value() !=
        value.profitabilityMetricIdentity.has_value())
        return false;
    for (const auto& workflow : value.workflows)
        if (!WorkflowIdentityValid(workflow)) return false;
    return true;
}

std::string WorkflowCanonicalText(
    const RecommendationCampaignWorkflowEvidence& workflow)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "proposal_id=" << workflow.proposalId
        << ";proposal_identity="
        << LengthText(workflow.proposalIdentityCanonical)
        << ";proposal_hash=" << LengthText(workflow.proposalIdentityHash)
        << ";review_id=" << OptionalLongText(workflow.latestReviewDecisionId)
        << ";review_disposition="
        << OptionalFramedText(workflow.latestReviewDisposition)
        << ";execution_id=" << OptionalLongText(workflow.executionId)
        << ";execution_identity="
        << OptionalFramedText(workflow.executionIdentityCanonical)
        << ";execution_hash="
        << OptionalFramedText(workflow.executionIdentityHash)
        << ";activation_id=" << OptionalLongText(workflow.activationId)
        << ";activation_identity="
        << OptionalFramedText(workflow.activationIdentityCanonical)
        << ";activation_hash="
        << OptionalFramedText(workflow.activationIdentityHash)
        << ";experiment_id="
        << OptionalLongText(workflow.convertedExperimentId)
        << ";state="
        << RecommendationConversionWorkflowStateText(workflow.state)
        << ";integrity="
        << RecommendationConversionWorkflowIntegrityText(workflow.integrity)
        << ";diagnostic_count=" << workflow.diagnosticCodes.size();
    for (std::size_t i = 0; i < workflow.diagnosticCodes.size(); ++i)
        out << ";diagnostic[" << i << "]="
            << LengthText(workflow.diagnosticCodes[i]);
    return out.str();
}

std::string CandidateEvidenceCanonicalText(
    const RecommendationCampaignCandidateInput& candidate)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "ranking_snapshot_id=" << candidate.rankingSnapshotId
        << ";ranking_policy="
        << LengthText(candidate.rankingPolicyCanonical)
        << ";ranking_policy_hash="
        << LengthText(candidate.rankingPolicyHash)
        << ";ranking_version=" << candidate.rankingVersion
        << ";ranking_member_id=" << candidate.rankingMemberId
        << ";ranking_position=" << candidate.rankingPosition
        << ";ranking_bucket=" << LengthText(candidate.rankingBucket)
        << ";ranking_score=" << OptionalDoubleText(candidate.rankingScore)
        << ";recommendation_id=" << candidate.recommendationId
        << ";source_experiment_id=" << candidate.sourceExperimentId
        << ";symbol=" << LengthText(candidate.symbol)
        << ";horizon=" << candidate.predictionHorizon
        << ";family=" << LengthText(candidate.family)
        << ";target_epochs=" << OptionalIntText(candidate.targetEpochs)
        << ";leader_score=" << FiniteDoubleText(candidate.leaderScore)
        << ";inference_accuracy="
        << FiniteDoubleText(candidate.inferenceAccuracy)
        << ";neutral="
        << OptionalDoubleText(candidate.predictedNeutralProportion)
        << ";profitability="
        << OptionalDoubleText(candidate.profitabilityMetric)
        << ";profitability_identity="
        << OptionalFramedText(candidate.profitabilityMetricIdentity)
        << ";semantic="
        << LengthText(candidate.recommendationSemanticCanonical)
        << ";semantic_hash="
        << LengthText(candidate.recommendationSemanticHash)
        << ";invocation="
        << LengthText(candidate.recommendationInvocationCanonical)
        << ";invocation_hash="
        << LengthText(candidate.recommendationInvocationHash)
        << ";persisted_provenance_valid="
        << (candidate.persistedProvenanceValid ? 1 : 0)
        << ";workflow_count=" << candidate.workflows.size();
    for (std::size_t i = 0; i < candidate.workflows.size(); ++i)
        out << ";workflow[" << i << "]="
            << LengthText(WorkflowCanonicalText(candidate.workflows[i]));
    return out.str();
}

bool CampaignCandidateLess(
    const RecommendationCampaignCandidateInput& left,
    const RecommendationCampaignCandidateInput& right)
{
    if (left.rankingPosition != right.rankingPosition)
        return left.rankingPosition < right.rankingPosition;
    if (left.recommendationId != right.recommendationId)
        return left.recommendationId < right.recommendationId;
    if (left.rankingMemberId != right.rankingMemberId)
        return left.rankingMemberId < right.rankingMemberId;
    return CandidateEvidenceCanonicalText(left) <
           CandidateEvidenceCanonicalText(right);
}

void AddWorkflowReasons(
    RecommendationCampaignPlanCandidate& candidate,
    const RecommendationCampaignPlanningPolicy& policy)
{
    for (const auto& workflow : candidate.input.workflows)
    {
        if (workflow.integrity ==
                RecommendationConversionWorkflowIntegrity::inconsistent ||
            workflow.state == RecommendationConversionWorkflowState::inconsistent)
        {
            AddReason(candidate, RecommendationCampaignReason::workflowInconsistent);
            continue;
        }
        switch (workflow.state)
        {
            case RecommendationConversionWorkflowState::proposed:
            case RecommendationConversionWorkflowState::pendingReview:
                AddReason(candidate,
                          RecommendationCampaignReason::proposalAlreadyExists);
                AddReason(candidate,
                          RecommendationCampaignReason::workflowPendingReview);
                break;
            case RecommendationConversionWorkflowState::rejected:
                if (!policy.reconsiderRejectedWorkflows)
                {
                    AddReason(candidate,
                              RecommendationCampaignReason::proposalAlreadyExists);
                    AddReason(candidate, RecommendationCampaignReason::
                              workflowRejectedNotReconsiderable);
                }
                break;
            case RecommendationConversionWorkflowState::approvedNotExecuted:
                AddReason(candidate,
                          RecommendationCampaignReason::proposalAlreadyExists);
                AddReason(candidate, RecommendationCampaignReason::
                          workflowApprovedNotExecuted);
                break;
            case RecommendationConversionWorkflowState::executedPaused:
                AddReason(candidate,
                          RecommendationCampaignReason::proposalAlreadyExists);
                AddReason(candidate,
                          RecommendationCampaignReason::workflowExecutedPaused);
                break;
            case RecommendationConversionWorkflowState::activatedPending:
            case RecommendationConversionWorkflowState::schedulerClaimedOrRunning:
                AddReason(candidate,
                          RecommendationCampaignReason::proposalAlreadyExists);
                AddReason(candidate, RecommendationCampaignReason::
                          workflowActivatedOrSchedulerOwned);
                break;
            case RecommendationConversionWorkflowState::completed:
                if (policy.completedWorkflowsExcludeSelection)
                {
                    AddReason(candidate,
                              RecommendationCampaignReason::proposalAlreadyExists);
                    AddReason(candidate,
                              RecommendationCampaignReason::workflowCompleted);
                }
                break;
            case RecommendationConversionWorkflowState::failed:
                if (!policy.reconsiderFailedWorkflows)
                {
                    AddReason(candidate,
                              RecommendationCampaignReason::proposalAlreadyExists);
                    AddReason(candidate, RecommendationCampaignReason::
                              workflowFailedNotReconsiderable);
                }
                break;
            case RecommendationConversionWorkflowState::cancelled:
                if (!policy.reconsiderCancelledWorkflows)
                {
                    AddReason(candidate,
                              RecommendationCampaignReason::proposalAlreadyExists);
                    AddReason(candidate, RecommendationCampaignReason::
                              workflowCancelledNotReconsiderable);
                }
                break;
            case RecommendationConversionWorkflowState::inconsistent:
                break;
        }
    }
}

std::string PlanIdentityCanonicalText(
    const RecommendationCampaignPlan& plan,
    const std::string& snapshotCanonical,
    const std::string& snapshotHash)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_plan_v1"
        << ";contract_version=" << plan.contractVersion
        << ";policy=" << LengthText(plan.policyCanonical)
        << ";scope=" << LengthText(plan.scopeCanonical)
        << ";ranking_snapshot=" << LengthText(snapshotCanonical)
        << ";ranking_snapshot_hash=" << LengthText(snapshotHash)
        << ";candidate_count=" << plan.candidates.size();
    for (std::size_t i = 0; i < plan.candidates.size(); ++i)
    {
        const auto& candidate = plan.candidates[i];
        out << ";candidate[" << i << "].evidence="
            << LengthText(CandidateEvidenceCanonicalText(candidate.input))
            << ";candidate[" << i << "].decision="
            << RecommendationCampaignDecisionText(candidate.decision)
            << ";candidate[" << i << "].reason_count="
            << candidate.reasons.size();
        for (std::size_t r = 0; r < candidate.reasons.size(); ++r)
            out << ";candidate[" << i << "].reason[" << r << "]="
                << RecommendationCampaignReasonText(candidate.reasons[r]);
    }
    return out.str();
}

} // namespace

std::optional<std::string> ValidateRecommendationCampaignPlanningPolicy(
    const RecommendationCampaignPlanningPolicy& policy)
{
    if (policy.contractVersion != kRecommendationCampaignPlanContractVersion)
        return "recommendation_campaign_contract_version_unsupported";
    if (policy.maximumSelectedRecommendations <= 0 ||
        policy.maximumSelectedRecommendations >
            kMaximumRecommendationCampaignCandidates)
        return "recommendation_campaign_limit_invalid";
    if (policy.maximumCandidatesConsidered <= 0 ||
        policy.maximumCandidatesConsidered >
            kMaximumRecommendationCampaignCandidates)
        return "recommendation_campaign_candidate_limit_invalid";
    if (!FiniteUnit(policy.minimumLeaderScore))
        return "recommendation_campaign_minimum_leader_score_invalid";
    if (!FiniteUnit(policy.minimumInferenceAccuracy))
        return "recommendation_campaign_minimum_inference_accuracy_invalid";
    if (!FiniteUnit(policy.maximumPredictedNeutralProportion))
        return "recommendation_campaign_maximum_neutral_invalid";
    if (policy.minimumProfitability &&
        !std::isfinite(*policy.minimumProfitability))
        return "recommendation_campaign_minimum_profitability_invalid";
    for (const auto value : {
             policy.maximumPerSymbol,
             policy.maximumPerHorizon,
             policy.maximumPerSourceExperiment})
        if (value && (*value <= 0 ||
                      *value > kMaximumRecommendationCampaignCandidates))
            return "recommendation_campaign_group_limit_invalid";
    if (!policy.inconsistentWorkflowsAlwaysExclude)
        return "recommendation_campaign_inconsistent_workflow_exclusion_required";
    if (policy.tieBreaking !=
        "ranking_global_ordinal_then_recommendation_id_then_ranking_member_id_"
        "then_canonical_evidence")
        return "recommendation_campaign_tie_breaking_unsupported";
    return std::nullopt;
}

std::string RecommendationCampaignPlanningPolicyCanonicalText(
    const RecommendationCampaignPlanningPolicy& policy)
{
    if (const auto error = ValidateRecommendationCampaignPlanningPolicy(policy))
        throw std::invalid_argument(*error);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_planning_policy_v1"
        << ";contract_version=" << policy.contractVersion
        << ";enabled=" << (policy.enabled ? 1 : 0)
        << ";maximum_selected=" << policy.maximumSelectedRecommendations
        << ";maximum_candidates=" << policy.maximumCandidatesConsidered
        << ";minimum_leader_score="
        << CanonicalRecommendationDouble(policy.minimumLeaderScore)
        << ";minimum_inference_accuracy="
        << CanonicalRecommendationDouble(policy.minimumInferenceAccuracy)
        << ";maximum_neutral="
        << CanonicalRecommendationDouble(
               policy.maximumPredictedNeutralProportion)
        << ";minimum_profitability="
        << OptionalDoubleText(policy.minimumProfitability)
        << ";maximum_per_symbol="
        << OptionalIntText(policy.maximumPerSymbol)
        << ";maximum_per_horizon="
        << OptionalIntText(policy.maximumPerHorizon)
        << ";maximum_per_source="
        << OptionalIntText(policy.maximumPerSourceExperiment)
        << ";reconsider_rejected="
        << (policy.reconsiderRejectedWorkflows ? 1 : 0)
        << ";reconsider_failed="
        << (policy.reconsiderFailedWorkflows ? 1 : 0)
        << ";reconsider_cancelled="
        << (policy.reconsiderCancelledWorkflows ? 1 : 0)
        << ";completed_excludes="
        << (policy.completedWorkflowsExcludeSelection ? 1 : 0)
        << ";inconsistent_excludes="
        << (policy.inconsistentWorkflowsAlwaysExclude ? 1 : 0)
        << ";tie_breaking=" << LengthText(policy.tieBreaking);
    return out.str();
}

std::string RecommendationCampaignPlanningPolicyHash(
    const RecommendationCampaignPlanningPolicy& policy)
{
    return RecommendationCanonicalHash(
        RecommendationCampaignPlanningPolicyCanonicalText(policy));
}

std::optional<std::string> ValidateRecommendationCampaignPlanningScope(
    const RecommendationCampaignPlanningScope& scope)
{
    if (scope.rankingSnapshotId <= 0)
        return "recommendation_campaign_ranking_snapshot_id_invalid";
    if (scope.symbol)
    {
        const auto canonical = EA::CanonicalSymbol::TryNormalize(*scope.symbol);
        if (scope.symbol->size() > 64 || !canonical ||
            *canonical != *scope.symbol)
            return "recommendation_campaign_symbol_invalid";
    }
    if (scope.horizon && *scope.horizon <= 0)
        return "recommendation_campaign_horizon_invalid";
    if (scope.recommendationId && *scope.recommendationId <= 0)
        return "recommendation_campaign_recommendation_id_invalid";
    return std::nullopt;
}

std::string RecommendationCampaignPlanningScopeCanonicalText(
    const RecommendationCampaignPlanningScope& scope)
{
    if (const auto error = ValidateRecommendationCampaignPlanningScope(scope))
        throw std::invalid_argument(*error);
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_scope_v1"
        << ";ranking_snapshot_id=" << scope.rankingSnapshotId
        << ";symbol=" << OptionalFramedText(scope.symbol)
        << ";horizon=" << OptionalIntText(scope.horizon)
        << ";recommendation_id="
        << OptionalLongText(scope.recommendationId);
    return out.str();
}

std::string RecommendationCampaignDecisionText(
    RecommendationCampaignDecision decision)
{
    switch (decision)
    {
        case RecommendationCampaignDecision::include: return "include";
        case RecommendationCampaignDecision::exclude: return "exclude";
    }
    throw std::invalid_argument("recommendation_campaign_decision_invalid");
}

std::string RecommendationCampaignReasonText(RecommendationCampaignReason reason)
{
    switch (reason)
    {
        case RecommendationCampaignReason::selected: return "selected";
        case RecommendationCampaignReason::planningDisabled: return "planning_disabled";
        case RecommendationCampaignReason::recommendationNotRanked: return "recommendation_not_ranked";
        case RecommendationCampaignReason::rankingNotAdvisoryReady: return "ranking_not_advisory_ready";
        case RecommendationCampaignReason::belowMinimumLeaderScore: return "below_minimum_leader_score";
        case RecommendationCampaignReason::belowMinimumInferenceAccuracy: return "below_minimum_inference_accuracy";
        case RecommendationCampaignReason::predictedNeutralMetricUnavailable: return "predicted_neutral_metric_unavailable";
        case RecommendationCampaignReason::predictedNeutralAboveMaximum: return "predicted_neutral_above_maximum";
        case RecommendationCampaignReason::profitabilityMetricUnavailable: return "profitability_metric_unavailable";
        case RecommendationCampaignReason::belowMinimumProfitability: return "below_minimum_profitability";
        case RecommendationCampaignReason::proposalAlreadyExists: return "proposal_already_exists";
        case RecommendationCampaignReason::workflowPendingReview: return "workflow_pending_review";
        case RecommendationCampaignReason::workflowRejectedNotReconsiderable: return "workflow_rejected_not_reconsiderable";
        case RecommendationCampaignReason::workflowApprovedNotExecuted: return "workflow_approved_not_executed";
        case RecommendationCampaignReason::workflowExecutedPaused: return "workflow_executed_paused";
        case RecommendationCampaignReason::workflowActivatedOrSchedulerOwned: return "workflow_activated_or_scheduler_owned";
        case RecommendationCampaignReason::workflowCompleted: return "workflow_completed";
        case RecommendationCampaignReason::workflowFailedNotReconsiderable: return "workflow_failed_not_reconsiderable";
        case RecommendationCampaignReason::workflowCancelledNotReconsiderable: return "workflow_cancelled_not_reconsiderable";
        case RecommendationCampaignReason::workflowInconsistent: return "workflow_inconsistent";
        case RecommendationCampaignReason::duplicateConversionIdentity: return "duplicate_conversion_identity";
        case RecommendationCampaignReason::sourceExperimentConflict: return "source_experiment_conflict";
        case RecommendationCampaignReason::symbolLimitReached: return "symbol_limit_reached";
        case RecommendationCampaignReason::horizonLimitReached: return "horizon_limit_reached";
        case RecommendationCampaignReason::sourceExperimentLimitReached: return "source_experiment_limit_reached";
        case RecommendationCampaignReason::campaignLimitReached: return "campaign_limit_reached";
        case RecommendationCampaignReason::candidateLimitReached: return "candidate_limit_reached";
        case RecommendationCampaignReason::unsupportedContractVersion: return "unsupported_contract_version";
        case RecommendationCampaignReason::identityValidationFailed: return "identity_validation_failed";
    }
    throw std::invalid_argument("recommendation_campaign_reason_invalid");
}

std::string RecommendationCampaignReasonExplanation(
    RecommendationCampaignReason reason)
{
    switch (reason)
    {
        case RecommendationCampaignReason::selected:
            return "The candidate is included in this read-only plan.";
        case RecommendationCampaignReason::planningDisabled:
            return "Campaign planning is disabled by policy.";
        case RecommendationCampaignReason::profitabilityMetricUnavailable:
            return "No authoritative profitability metric exists in the current schema.";
        case RecommendationCampaignReason::workflowInconsistent:
            return "Phase 4C workflow provenance or lifecycle integrity is inconsistent.";
        case RecommendationCampaignReason::campaignLimitReached:
            return "A higher-ranked candidate filled the campaign selection limit.";
        default:
            return "The candidate was excluded by the named deterministic campaign rule.";
    }
}

RecommendationCampaignPlan PlanRecommendationCampaign(
    const RecommendationCampaignPlanInput& input)
{
    if (const auto error = ValidateRecommendationCampaignPlanningPolicy(
            input.policy))
        throw std::invalid_argument(*error);
    if (const auto error = ValidateRecommendationCampaignPlanningScope(
            input.scope))
        throw std::invalid_argument(*error);
    if (!IdentityPairValid(
            input.rankingSnapshotIdentityCanonical,
            input.rankingSnapshotIdentityHash))
        throw std::invalid_argument(
            "recommendation_campaign_snapshot_identity_invalid");
    if (input.candidates.size() >
        static_cast<std::size_t>(kMaximumRecommendationCampaignCandidates))
        throw std::invalid_argument(
            "recommendation_campaign_input_limit_exceeded");

    RecommendationCampaignPlan plan;
    plan.policy = input.policy;
    plan.scope = input.scope;
    plan.policyCanonical = RecommendationCampaignPlanningPolicyCanonicalText(
        input.policy);
    plan.policyHash = RecommendationCanonicalHash(plan.policyCanonical);
    plan.scopeCanonical = RecommendationCampaignPlanningScopeCanonicalText(
        input.scope);
    plan.generatedAt = input.generatedAt;

    std::vector<RecommendationCampaignCandidateInput> ordered = input.candidates;
    for (auto& candidate : ordered)
        std::sort(
            candidate.workflows.begin(), candidate.workflows.end(),
            [](const auto& left, const auto& right)
            { return left.proposalId < right.proposalId; });
    std::sort(
        ordered.begin(), ordered.end(),
        CampaignCandidateLess);

    bool rankingPolicyConsistent = true;
    if (!ordered.empty())
    {
        const std::string& expectedCanonical =
            ordered.front().rankingPolicyCanonical;
        const std::string& expectedHash = ordered.front().rankingPolicyHash;
        rankingPolicyConsistent = std::all_of(
            ordered.begin(), ordered.end(),
            [&](const auto& candidate)
            {
                return candidate.rankingPolicyCanonical == expectedCanonical &&
                       candidate.rankingPolicyHash == expectedHash;
            });
    }

    std::set<std::string> selectedInvocationCanonicals;
    std::map<std::string, int> symbolCounts;
    std::map<int, int> horizonCounts;
    std::map<long long, int> sourceCounts;

    for (std::size_t index = 0; index < ordered.size(); ++index)
    {
        RecommendationCampaignPlanCandidate candidate;
        candidate.ordinal = static_cast<int>(index + 1);
        candidate.input = std::move(ordered[index]);

        if (!input.policy.enabled)
            AddReason(candidate, RecommendationCampaignReason::planningDisabled);
        else
        {
            if (candidate.ordinal > input.policy.maximumCandidatesConsidered)
                AddReason(candidate,
                          RecommendationCampaignReason::candidateLimitReached);
            if (candidate.input.rankingSnapshotId !=
                    input.scope.rankingSnapshotId ||
                candidate.input.rankingPosition <= 0)
                AddReason(candidate,
                          RecommendationCampaignReason::recommendationNotRanked);
            if (candidate.input.rankingBucket != "advisory_ready" ||
                !candidate.input.rankingScore)
                AddReason(candidate,
                          RecommendationCampaignReason::rankingNotAdvisoryReady);
            if (candidate.input.rankingVersion != 1)
                AddReason(candidate,
                          RecommendationCampaignReason::unsupportedContractVersion);
            if (!rankingPolicyConsistent ||
                !CandidateIdentityValid(candidate.input) ||
                candidate.input.rankingSnapshotIdentityCanonical !=
                    input.rankingSnapshotIdentityCanonical ||
                candidate.input.rankingSnapshotIdentityHash !=
                    input.rankingSnapshotIdentityHash)
                AddReason(candidate,
                          RecommendationCampaignReason::identityValidationFailed);
            if (std::isfinite(candidate.input.leaderScore) &&
                candidate.input.leaderScore < input.policy.minimumLeaderScore)
                AddReason(candidate, RecommendationCampaignReason::
                          belowMinimumLeaderScore);
            if (std::isfinite(candidate.input.inferenceAccuracy) &&
                candidate.input.inferenceAccuracy <
                    input.policy.minimumInferenceAccuracy)
                AddReason(candidate, RecommendationCampaignReason::
                          belowMinimumInferenceAccuracy);
            if (!candidate.input.predictedNeutralProportion)
                AddReason(candidate, RecommendationCampaignReason::
                          predictedNeutralMetricUnavailable);
            else if (*candidate.input.predictedNeutralProportion >
                     input.policy.maximumPredictedNeutralProportion)
                AddReason(candidate, RecommendationCampaignReason::
                          predictedNeutralAboveMaximum);
            if (input.policy.minimumProfitability)
            {
                if (!candidate.input.profitabilityMetric)
                    AddReason(candidate, RecommendationCampaignReason::
                              profitabilityMetricUnavailable);
                else if (*candidate.input.profitabilityMetric <
                         *input.policy.minimumProfitability)
                    AddReason(candidate, RecommendationCampaignReason::
                              belowMinimumProfitability);
            }
            AddWorkflowReasons(candidate, input.policy);
        }

        if (candidate.reasons.empty())
        {
            if (selectedInvocationCanonicals.contains(
                    candidate.input.recommendationInvocationCanonical))
                AddReason(candidate, RecommendationCampaignReason::
                          duplicateConversionIdentity);
            const int sourceCount = sourceCounts[candidate.input.sourceExperimentId];
            if (input.policy.maximumPerSourceExperiment && sourceCount > 0 &&
                *input.policy.maximumPerSourceExperiment == 1)
                AddReason(candidate, RecommendationCampaignReason::
                          sourceExperimentConflict);
            else if (input.policy.maximumPerSourceExperiment &&
                     sourceCount >= *input.policy.maximumPerSourceExperiment)
                AddReason(candidate, RecommendationCampaignReason::
                          sourceExperimentLimitReached);
            if (input.policy.maximumPerSymbol &&
                symbolCounts[candidate.input.symbol] >=
                    *input.policy.maximumPerSymbol)
                AddReason(candidate,
                          RecommendationCampaignReason::symbolLimitReached);
            if (input.policy.maximumPerHorizon &&
                horizonCounts[candidate.input.predictionHorizon] >=
                    *input.policy.maximumPerHorizon)
                AddReason(candidate,
                          RecommendationCampaignReason::horizonLimitReached);
            if (plan.summary.selectedCount >=
                input.policy.maximumSelectedRecommendations)
                AddReason(candidate,
                          RecommendationCampaignReason::campaignLimitReached);
        }

        if (candidate.reasons.empty())
        {
            candidate.decision = RecommendationCampaignDecision::include;
            candidate.reasons.push_back(RecommendationCampaignReason::selected);
            ++plan.summary.selectedCount;
            selectedInvocationCanonicals.insert(
                candidate.input.recommendationInvocationCanonical);
            ++symbolCounts[candidate.input.symbol];
            ++horizonCounts[candidate.input.predictionHorizon];
            ++sourceCounts[candidate.input.sourceExperimentId];
        }
        else
        {
            candidate.decision = RecommendationCampaignDecision::exclude;
            ++plan.summary.excludedCount;
        }
        plan.candidates.push_back(std::move(candidate));
    }

    plan.summary.candidateCount = static_cast<int>(plan.candidates.size());
    plan.identityCanonical = PlanIdentityCanonicalText(
        plan, input.rankingSnapshotIdentityCanonical,
        input.rankingSnapshotIdentityHash);
    plan.identityHash = RecommendationCanonicalHash(plan.identityCanonical);
    return plan;
}

bool RecommendationCampaignPlanOrderingIsDeterministic(
    const RecommendationCampaignPlan& plan)
{
    for (std::size_t index = 0; index < plan.candidates.size(); ++index)
    {
        if (plan.candidates[index].ordinal != static_cast<int>(index + 1))
            return false;
        if (index != 0 && CampaignCandidateLess(
                plan.candidates[index].input,
                plan.candidates[index - 1].input))
            return false;
    }
    return true;
}

} // namespace EA::ExperimentRecommendation
