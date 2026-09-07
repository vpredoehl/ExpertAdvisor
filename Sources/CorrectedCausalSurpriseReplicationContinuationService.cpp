#include "CorrectedCausalSurpriseReplicationContinuationService.hpp"

#include "FeatureAblation.hpp"
#include "FeatureAblationPairEvaluationRepository.hpp"

#include <algorithm>
#include <sstream>
#include <stdexcept>

namespace EA::CorrectedCausalSurpriseReplicationContinuation
{
namespace
{

ScientificConfiguration ConfigurationFrom(
    const FeatureAblationPairEvaluation::ArmEvidence& control)
{
    const auto& persisted = control.authoritative.configuration;
    const auto& extended = control.extended;
    if (!extended.configuredModelInputWidth ||
        !extended.configuredModelInputLayoutVersion ||
        !extended.economicCalendarSnapshotId ||
        !extended.economicCalendarSnapshotHash)
        throw std::invalid_argument(
            "corrected_replication_anchor_provenance_missing");
    ScientificConfiguration result;
    result.initialSymbol = persisted.symbol;
    result.initialPredictionHorizon = persisted.predictionHorizon;
    result.targetEpochs = persisted.targetEpochs;
    result.threshold = persisted.threshold;
    result.coreLearningRateMultiplier =
        persisted.coreLearningRateMultiplier;
    result.headLearningRateMultiplier =
        persisted.headLearningRateMultiplier;
    result.checkpointInterval = persisted.checkpointInterval;
    result.trainingObjectiveCanonical =
        persisted.experimentObjective.canonical;
    result.trainingObjectiveHash = persisted.experimentObjective.hash;
    result.trainStart = persisted.trainStart;
    result.trainEnd = persisted.trainEnd;
    result.inferenceStart = persisted.inferenceStart;
    result.inferenceEnd = persisted.inferenceEnd;
    result.donchianMode = persisted.donchianMode;
    result.featureWarmupScope = persisted.featureWarmupScope;
    result.donchianLookback = persisted.donchianLookback;
    result.modelInputWidth = *extended.configuredModelInputWidth;
    result.semanticLayoutVersion =
        *extended.configuredModelInputLayoutVersion;
    result.economicCalendarSnapshotId =
        *extended.economicCalendarSnapshotId;
    result.economicCalendarSnapshotHash =
        *extended.economicCalendarSnapshotHash;
    result.baseLearningRate = extended.baseLearningRate;
    result.batchSize = extended.batchSize;
    result.freshInitializationSeed = extended.freshInitializationSeed;
    result.checkpointInferenceEnabled =
        extended.checkpointInferenceEnabled;
    result.checkpointInferenceMinimumEpoch =
        extended.checkpointInferenceMinimumEpoch;
    result.checkpointInferenceInterval =
        extended.checkpointInferenceInterval;
    result.checkpointPolicyEnabled = extended.checkpointPolicyEnabled;
    result.checkpointPolicyMinimumLeaderScore =
        extended.checkpointPolicyMinimumLeaderScore;
    result.checkpointPolicyMinimumInferenceAccuracy =
        extended.checkpointPolicyMinimumInferenceAccuracy;
    result.checkpointPolicyTopN = extended.checkpointPolicyTopN;
    result.checkpointPolicyScope = extended.checkpointPolicyScope;
    result.checkpointPolicyStopMode = extended.checkpointPolicyStopMode;
    result.checkpointPolicyGraceEvaluations =
        extended.checkpointPolicyGraceEvaluations;
    result.checkpointPolicyRevision = extended.checkpointPolicyRevision;
    result.checkpointPolicyHash = extended.checkpointPolicyHash;
    result.continuationPolicyEnabled = extended.continuationPolicyEnabled;
    result.continuationPolicyScientificIdentity =
        extended.continuationPolicyScientificIdentity;
    result.schedulerPriority = "high";
    return result;
}

} // namespace

Assessment EvaluateCommand(pqxx::transaction_base& transaction,
                           const Command& command)
{
    if (command.evidencePairs.empty())
        throw std::invalid_argument("corrected_replication_evidence_empty");
    const auto anchor = std::find(
        command.evidencePairs.begin(), command.evidencePairs.end(),
        command.anchorPair);
    if (anchor == command.evidencePairs.end())
        throw std::invalid_argument(
            "corrected_replication_anchor_not_in_membership");

    Assessment result;
    FeatureAblationReplicationEvaluation::ComparisonCommand comparison;
    comparison.experimentIdPairs = command.evidencePairs;
    comparison.expectedAblationMask =
        std::string{kCausalEconomicEventSurpriseAblationMaskText};
    result.replication =
        FeatureAblationReplicationEvaluation::EvaluateComparison(
            transaction, comparison);

    result.anchorControl =
        FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
            transaction, command.anchorPair.first);
    result.anchorTreatment =
        FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
            transaction, command.anchorPair.second);
    result.anchorComparison = FeatureAblationPairEvaluation::Compare(
        result.anchorControl, result.anchorTreatment,
        kCausalEconomicEventSurpriseAblationMaskText);
    result.plan = MakePlan(ConfigurationFrom(result.anchorControl));

    const std::size_t anchorIndex = static_cast<std::size_t>(
        std::distance(command.evidencePairs.begin(), anchor));
    result.gate = EvaluateGate(
        result.replication, result.replication.members.at(anchorIndex));
    return result;
}

std::string RenderAssessment(const Assessment& assessment)
{
    std::ostringstream output;
    output << FeatureAblationReplicationEvaluation::RenderComparisonOutput(
                  assessment.replication)
           << RenderGate(assessment.gate)
           << RenderPlan(assessment.plan);
    output << "CORRECTED_CAUSAL_SURPRISE_CONTINUATION_GATE"
           << ",plan_hash=" << assessment.plan.hash
           << ",next_action="
           << NextActionText(assessment.gate.nextAction)
           << ",materialization_permitted="
           << (assessment.gate.nextAction ==
                       NextAction::PrepareAdditionalReplications
                   ? "true" : "false")
           << ",materialization_performed=false"
           << ",selection_uses_outcome_metrics=false"
           << ",read_only=true\n";
    return output.str();
}

int RunStatusCommand(const std::string& connectionString,
                     const Command& command,
                     std::ostream& output,
                     std::ostream& errors)
{
    (void)errors;
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const Assessment assessment = EvaluateCommand(transaction, command);
    output << RenderAssessment(assessment);
    switch (assessment.gate.nextAction)
    {
        case NextAction::ReplicationThresholdSatisfied:
        case NextAction::PrepareAdditionalReplications:
            return 0;
        case NextAction::AwaitPairCompletion:
            return 4;
        case NextAction::InvalidPairRequiresReview:
            return 3;
    }
    return 3;
}

} // namespace EA::CorrectedCausalSurpriseReplicationContinuation
