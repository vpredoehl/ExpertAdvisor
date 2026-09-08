#include "CorrectedCausalSurpriseReplicationContinuationService.hpp"

#include "FeatureAblation.hpp"
#include "FeatureAblationPairEvaluationRepository.hpp"

#include <algorithm>
#include <set>
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
    result.schedulerPriority = control.operational.schedulerPriority;
    return result;
}

std::string InvocationMode(pqxx::transaction_base& transaction,
                           long long experimentId)
{
    const pqxx::result rows = transaction.exec(
        "SELECT invocation_mode FROM experiment WHERE experiment_id=$1;",
        pqxx::params{experimentId});
    if (rows.size() != 1 || rows.one_row()[0].is_null())
        throw std::invalid_argument(
            "corrected_replication_follow_on_provenance_missing");
    return rows.one_row()[0].as<std::string>();
}

void ValidateDeclaredFollowOnMembership(
    pqxx::transaction_base& transaction,
    const Command& command,
    const Assessment& assessment)
{
    std::set<std::size_t> ordinals;
    std::size_t previousOrdinal = 0;
    for (std::size_t index = 0; index < command.evidencePairs.size(); ++index)
    {
        const auto ids = command.evidencePairs[index];
        if (ids == command.anchorPair) continue;
        const auto& member = assessment.replication.members.at(index);
        if (member.evidenceState ==
                Replication::MemberEvidenceState::HistoricalPreFix)
            continue;
        if (member.evidenceClassification !=
                FeatureAblationPairEvaluation::EvidenceClassification::
                    CorrectedCausalSurprisePairEvidence)
            continue;

        const auto control =
            FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
                transaction, ids.first);
        const auto treatment =
            FeatureAblationPairEvaluation::LoadAuthoritativeArmEvidence(
                transaction, ids.second);
        const Pair* matched = nullptr;
        for (const Pair& candidate : assessment.plan.pairs)
        {
            try
            {
                ValidatePlannedArmEvidence(
                    assessment.plan, candidate, candidate.control, control);
                ValidatePlannedArmEvidence(
                    assessment.plan, candidate, candidate.treatment,
                    treatment);
                matched = &candidate;
                break;
            }
            catch (const std::invalid_argument&)
            {
            }
        }
        if (!matched)
            throw std::invalid_argument(
                "corrected_replication_follow_on_not_in_predeclared_plan");
        if (!ordinals.insert(matched->ordinal).second ||
            matched->ordinal != previousOrdinal + 1)
            throw std::invalid_argument(
                "corrected_replication_follow_on_prefix_invalid");
        previousOrdinal = matched->ordinal;
        if (InvocationMode(transaction, ids.first) !=
            MaterializationProvenance(
                assessment.plan, *matched, matched->control))
            throw std::invalid_argument(
                "corrected_replication_control_materialization_"
                "provenance_mismatch");
        if (InvocationMode(transaction, ids.second) !=
            MaterializationProvenance(
                assessment.plan, *matched, matched->treatment))
            throw std::invalid_argument(
                "corrected_replication_treatment_materialization_"
                "provenance_mismatch");
    }

    if (assessment.gate.correctedValidPairCount != ordinals.size() + 1 &&
        assessment.gate.anchorPairValidCorrected)
        throw std::invalid_argument(
            "corrected_replication_valid_count_not_plan_prefix");
}

} // namespace

Assessment EvaluateCommand(pqxx::transaction_base& transaction,
                           const Command& command)
{
    if (command.evidencePairs.empty())
        throw std::invalid_argument("corrected_replication_evidence_empty");
    if (command.anchorPair != std::pair<long long, long long>{
            kAnchorControlExperimentId, kAnchorTreatmentExperimentId})
        throw std::invalid_argument(
            "corrected_replication_anchor_identity_not_predeclared");
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

    if (result.anchorControl.operational.schedulerPriority != "high" ||
        result.anchorTreatment.operational.schedulerPriority != "high")
        throw std::invalid_argument(
            "corrected_replication_anchor_priority_invalid");

    const std::size_t anchorIndex = static_cast<std::size_t>(
        std::distance(command.evidencePairs.begin(), anchor));
    result.gate = EvaluateGate(
        result.replication, result.replication.members.at(anchorIndex));
    ValidateDeclaredFollowOnMembership(transaction, command, result);
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
