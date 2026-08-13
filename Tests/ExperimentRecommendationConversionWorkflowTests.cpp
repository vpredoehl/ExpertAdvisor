#include "../Sources/ExperimentRecommendationConversionWorkflow.hpp"

#include "../Sources/ExperimentRecommendation.hpp"

#include <cassert>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationConversionWorkflowFacts Proposal()
{
    RecommendationConversionWorkflowFacts facts;
    facts.proposalId = 10;
    facts.proposalContractVersion = 1;
    facts.proposalIdentityCanonical = "proposal-canonical";
    facts.proposalIdentityHash = RecommendationCanonicalHash(
        facts.proposalIdentityCanonical);
    return facts;
}

void AddReview(
    RecommendationConversionWorkflowFacts& facts,
    long long id,
    const char* decision)
{
    facts.latestReview = RecommendationConversionWorkflowReviewFact{
        id, facts.proposalId, decision};
}

void AddExecution(RecommendationConversionWorkflowFacts& facts)
{
    if (!facts.latestReview)
        facts.latestReview = RecommendationConversionWorkflowReviewFact{
            11, facts.proposalId, "approve"};
    facts.executionCount = 1;
    RecommendationConversionWorkflowExecutionFact execution;
    execution.executionId = 20;
    execution.proposalId = facts.proposalId;
    execution.reviewDecisionId = 11;
    execution.experimentId = 30;
    execution.contractVersion = 1;
    execution.authorizationDecision = "approve";
    execution.identityCanonical = "execution-canonical";
    execution.identityHash = RecommendationCanonicalHash(
        execution.identityCanonical);
    facts.execution = execution;
    facts.executionReview = RecommendationConversionWorkflowReviewFact{
        11, facts.proposalId, "approve"};
    facts.experiment = RecommendationConversionWorkflowExperimentFact{
        30, "paused", "train", std::nullopt, std::nullopt, std::nullopt};
}

void AddActivation(RecommendationConversionWorkflowFacts& facts)
{
    facts.activationCount = 1;
    RecommendationConversionWorkflowActivationFact activation;
    activation.activationId = 40;
    activation.executionId = facts.execution->executionId;
    activation.proposalId = facts.proposalId;
    activation.reviewDecisionId = facts.execution->reviewDecisionId;
    activation.experimentId = facts.execution->experimentId;
    activation.contractVersion = 1;
    activation.previousStatus = "paused";
    activation.previousPhase = "train";
    activation.resultingStatus = "pending";
    activation.resultingPhase = "train";
    activation.identityCanonical = "activation-canonical";
    activation.identityHash = RecommendationCanonicalHash(
        activation.identityCanonical);
    facts.activation = activation;
    facts.experiment->status = "pending";
}

void ExpectState(
    const RecommendationConversionWorkflowFacts& facts,
    RecommendationConversionWorkflowState state)
{
    const auto result = DeriveRecommendationConversionWorkflow(facts);
    assert(result.state == state);
    assert(result.integrity ==
           RecommendationConversionWorkflowIntegrity::consistent);
    assert(result.diagnosticCodes.empty());
}

} // namespace

int main()
{
    auto facts = Proposal();
    ExpectState(facts, RecommendationConversionWorkflowState::pendingReview);

    AddReview(facts, 11, "reject");
    ExpectState(facts, RecommendationConversionWorkflowState::rejected);

    AddReview(facts, 12, "approve");
    ExpectState(
        facts, RecommendationConversionWorkflowState::approvedNotExecuted);

    AddExecution(facts);
    ExpectState(facts, RecommendationConversionWorkflowState::executedPaused);

    auto laterRejected = facts;
    AddReview(laterRejected, 13, "reject");
    ExpectState(
        laterRejected, RecommendationConversionWorkflowState::rejected);

    AddActivation(facts);
    ExpectState(
        facts, RecommendationConversionWorkflowState::activatedPending);

    laterRejected = facts;
    AddReview(laterRejected, 14, "reject");
    ExpectState(
        laterRejected, RecommendationConversionWorkflowState::rejected);

    facts.experiment->status = "running";
    facts.experiment->workerPid = 123;
    facts.experiment->currentOperation = "train";
    ExpectState(
        facts,
        RecommendationConversionWorkflowState::schedulerClaimedOrRunning);

    facts.experiment->workerPid.reset();
    facts.experiment->currentOperation.reset();
    for (const auto& terminal : {
             std::pair{"completed",
                       RecommendationConversionWorkflowState::completed},
             std::pair{"failed", RecommendationConversionWorkflowState::failed},
             std::pair{"cancelled",
                       RecommendationConversionWorkflowState::cancelled}})
    {
        facts.experiment->status = terminal.first;
        ExpectState(facts, terminal.second);
    }

    facts.experiment->status = "pending";
    facts.experiment->phase = "infer";
    ExpectState(
        facts,
        RecommendationConversionWorkflowState::schedulerClaimedOrRunning);

    auto inconsistent = facts;
    inconsistent.activation->proposalId = 999;
    auto derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.state == RecommendationConversionWorkflowState::inconsistent);
    assert(derived.diagnosticCodes.front() ==
           "activation_provenance_mismatch");

    inconsistent = facts;
    inconsistent.activation.reset();
    inconsistent.activationCount = 1;
    derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.diagnosticCodes.front() == "activation_cardinality_invalid");

    inconsistent = facts;
    inconsistent.experiment.reset();
    derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.diagnosticCodes.front() == "experiment_missing");

    inconsistent = facts;
    inconsistent.execution->identityHash = "bad";
    derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.diagnosticCodes.front() == "execution_identity_invalid");

    inconsistent = facts;
    inconsistent.activation->identityHash = "bad";
    derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.diagnosticCodes.front() == "activation_identity_invalid");

    inconsistent = Proposal();
    AddExecution(inconsistent);
    inconsistent.experiment->status = "pending";
    derived = DeriveRecommendationConversionWorkflow(inconsistent);
    assert(derived.diagnosticCodes.front() ==
           "unactivated_experiment_lifecycle_incompatible");

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
        const auto text = RecommendationConversionWorkflowStateText(state);
        assert(ParseRecommendationConversionWorkflowState(text) == state);
    }
    assert(!ParseRecommendationConversionWorkflowState("unknown"));
}
