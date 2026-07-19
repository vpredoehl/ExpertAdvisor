#include "ExperimentRecommendationConversionWorkflow.hpp"

#include "ExperimentRecommendation.hpp"

#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

void Diagnose(
    RecommendationConversionWorkflowDerivation& result,
    const char* code)
{
    result.diagnosticCodes.emplace_back(code);
}

bool HasSchedulerActivity(
    const RecommendationConversionWorkflowExperimentFact& experiment)
{
    return experiment.workerPid.has_value() ||
        experiment.currentOperation.has_value() ||
        experiment.currentEpoch.has_value();
}

} // namespace

std::string RecommendationConversionWorkflowStateText(
    RecommendationConversionWorkflowState state)
{
    switch (state)
    {
        case RecommendationConversionWorkflowState::proposed:
            return "proposed";
        case RecommendationConversionWorkflowState::pendingReview:
            return "pending_review";
        case RecommendationConversionWorkflowState::rejected:
            return "rejected";
        case RecommendationConversionWorkflowState::approvedNotExecuted:
            return "approved_not_executed";
        case RecommendationConversionWorkflowState::executedPaused:
            return "executed_paused";
        case RecommendationConversionWorkflowState::activatedPending:
            return "activated_pending";
        case RecommendationConversionWorkflowState::schedulerClaimedOrRunning:
            return "scheduler_claimed_or_running";
        case RecommendationConversionWorkflowState::completed:
            return "completed";
        case RecommendationConversionWorkflowState::failed:
            return "failed";
        case RecommendationConversionWorkflowState::cancelled:
            return "cancelled";
        case RecommendationConversionWorkflowState::inconsistent:
            return "inconsistent";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_workflow_state");
}

std::optional<RecommendationConversionWorkflowState>
ParseRecommendationConversionWorkflowState(const std::string& text)
{
    for (const auto state : {
             RecommendationConversionWorkflowState::proposed,
             RecommendationConversionWorkflowState::pendingReview,
             RecommendationConversionWorkflowState::rejected,
             RecommendationConversionWorkflowState::approvedNotExecuted,
             RecommendationConversionWorkflowState::executedPaused,
             RecommendationConversionWorkflowState::activatedPending,
             RecommendationConversionWorkflowState::schedulerClaimedOrRunning,
             RecommendationConversionWorkflowState::completed,
             RecommendationConversionWorkflowState::failed,
             RecommendationConversionWorkflowState::cancelled,
             RecommendationConversionWorkflowState::inconsistent})
    {
        if (RecommendationConversionWorkflowStateText(state) == text)
            return state;
    }
    return std::nullopt;
}

std::string RecommendationConversionWorkflowIntegrityText(
    RecommendationConversionWorkflowIntegrity integrity)
{
    switch (integrity)
    {
        case RecommendationConversionWorkflowIntegrity::consistent:
            return "consistent";
        case RecommendationConversionWorkflowIntegrity::inconsistent:
            return "inconsistent";
    }
    throw std::invalid_argument(
        "invalid_recommendation_conversion_workflow_integrity");
}

bool IsValidRecommendationConversionWorkflowReviewFact(
    const RecommendationConversionWorkflowReviewFact& review,
    long long proposalId)
{
    return review.reviewDecisionId > 0 && review.proposalId == proposalId &&
        (review.decision == "approve" || review.decision == "reject");
}

RecommendationConversionWorkflowDerivation
DeriveRecommendationConversionWorkflow(
    const RecommendationConversionWorkflowFacts& facts)
{
    RecommendationConversionWorkflowDerivation result;
    if (facts.proposalId <= 0)
        Diagnose(result, "proposal_id_invalid");
    if (facts.proposalContractVersion != 1)
        Diagnose(result, "proposal_contract_version_unsupported");
    if (facts.proposalIdentityCanonical.empty() ||
        facts.proposalIdentityHash !=
            RecommendationCanonicalHash(facts.proposalIdentityCanonical))
        Diagnose(result, "proposal_identity_invalid");
    if (facts.executionCount < 0 || facts.executionCount > 1 ||
        (facts.execution && facts.executionCount != 1) ||
        (!facts.execution && facts.executionCount != 0))
        Diagnose(result, "execution_cardinality_invalid");
    if (facts.activationCount < 0 || facts.activationCount > 1 ||
        (facts.activation && facts.activationCount != 1) ||
        (!facts.activation && facts.activationCount != 0))
        Diagnose(result, "activation_cardinality_invalid");

    if (facts.latestReview &&
        !IsValidRecommendationConversionWorkflowReviewFact(
            *facts.latestReview, facts.proposalId))
        Diagnose(result, "latest_review_invalid");

    if (facts.execution)
    {
        const auto& execution = *facts.execution;
        if (execution.executionId <= 0 ||
            execution.proposalId != facts.proposalId ||
            execution.reviewDecisionId <= 0 || execution.experimentId <= 0)
            Diagnose(result, "execution_provenance_mismatch");
        if (execution.contractVersion != 1)
            Diagnose(result, "execution_contract_version_unsupported");
        if (execution.authorizationDecision != "approve")
            Diagnose(result, "execution_not_approved");
        if (execution.identityCanonical.empty() ||
            execution.identityHash !=
                RecommendationCanonicalHash(execution.identityCanonical))
            Diagnose(result, "execution_identity_invalid");
        if (!facts.executionReview ||
            !IsValidRecommendationConversionWorkflowReviewFact(
                *facts.executionReview, facts.proposalId) ||
            facts.executionReview->reviewDecisionId !=
                execution.reviewDecisionId ||
            facts.executionReview->decision != "approve")
            Diagnose(result, "execution_review_invalid");
        if (!facts.latestReview)
            Diagnose(result, "latest_review_missing_for_execution");
    }

    if (facts.activation)
    {
        const auto& activation = *facts.activation;
        if (!facts.execution)
            Diagnose(result, "activation_without_execution");
        else if (activation.executionId != facts.execution->executionId ||
                 activation.proposalId != facts.execution->proposalId ||
                 activation.reviewDecisionId !=
                     facts.execution->reviewDecisionId ||
                 activation.experimentId != facts.execution->experimentId)
            Diagnose(result, "activation_provenance_mismatch");
        if (activation.activationId <= 0 || activation.contractVersion != 1)
            Diagnose(result, "activation_contract_version_unsupported");
        if (activation.previousStatus != "paused" ||
            activation.previousPhase != "train" ||
            activation.resultingStatus != "pending" ||
            activation.resultingPhase != "train")
            Diagnose(result, "activation_transition_invalid");
        if (activation.identityCanonical.empty() ||
            activation.identityHash !=
                RecommendationCanonicalHash(activation.identityCanonical))
            Diagnose(result, "activation_identity_invalid");
    }

    if (facts.execution)
    {
        if (!facts.experiment)
            Diagnose(result, "experiment_missing");
        else if (facts.experiment->experimentId !=
                 facts.execution->experimentId)
            Diagnose(result, "experiment_provenance_mismatch");
    }
    else if (facts.experiment)
    {
        Diagnose(result, "experiment_without_execution");
    }

    if (!result.diagnosticCodes.empty())
    {
        result.state = RecommendationConversionWorkflowState::inconsistent;
        return result;
    }

    // The greatest review-decision ID remains the current manual disposition,
    // even when an earlier approval already authorized durable downstream
    // evidence. Preserve that evidence in the view, but do not present a later
    // explicit rejection as an active conversion stage.
    if (facts.latestReview && facts.latestReview->decision == "reject")
    {
        result.integrity = RecommendationConversionWorkflowIntegrity::consistent;
        result.state = RecommendationConversionWorkflowState::rejected;
        return result;
    }

    if (!facts.execution)
    {
        result.integrity = RecommendationConversionWorkflowIntegrity::consistent;
        if (!facts.latestReview)
            result.state = RecommendationConversionWorkflowState::pendingReview;
        else
            result.state =
                RecommendationConversionWorkflowState::approvedNotExecuted;
        return result;
    }

    const auto& experiment = *facts.experiment;
    if (!facts.activation)
    {
        if (experiment.status == "paused" && experiment.phase == "train" &&
            !HasSchedulerActivity(experiment))
        {
            result.integrity =
                RecommendationConversionWorkflowIntegrity::consistent;
            result.state =
                RecommendationConversionWorkflowState::executedPaused;
            return result;
        }
        Diagnose(result, "unactivated_experiment_lifecycle_incompatible");
        result.state = RecommendationConversionWorkflowState::inconsistent;
        return result;
    }

    if (experiment.status == "completed")
        result.state = RecommendationConversionWorkflowState::completed;
    else if (experiment.status == "failed")
        result.state = RecommendationConversionWorkflowState::failed;
    else if (experiment.status == "cancelled")
        result.state = RecommendationConversionWorkflowState::cancelled;
    else if (experiment.status == "pending" && experiment.phase == "train" &&
             !HasSchedulerActivity(experiment))
        result.state = RecommendationConversionWorkflowState::activatedPending;
    else if (experiment.status == "running" ||
             HasSchedulerActivity(experiment) ||
             (experiment.status == "pending" && experiment.phase != "train"))
        result.state =
            RecommendationConversionWorkflowState::schedulerClaimedOrRunning;
    else
    {
        Diagnose(result, "activated_experiment_lifecycle_incompatible");
        result.state = RecommendationConversionWorkflowState::inconsistent;
        return result;
    }
    result.integrity = RecommendationConversionWorkflowIntegrity::consistent;
    return result;
}

} // namespace EA::ExperimentRecommendation
