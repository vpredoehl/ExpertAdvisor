#include "CorrectedCausalSurpriseReplicationContinuation.hpp"

#include "TrainingObjective.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>

namespace Continuation =
    EA::CorrectedCausalSurpriseReplicationContinuation;
namespace Replication = EA::FeatureAblationReplicationEvaluation;
namespace Pair = EA::FeatureAblationPairEvaluation;

namespace
{

Continuation::ScientificConfiguration Configuration()
{
    Continuation::ScientificConfiguration value;
    value.initialSymbol = "eurusdrmp";
    value.initialPredictionHorizon = 4;
    value.targetEpochs = 80;
    value.threshold = 0.0008;
    value.coreLearningRateMultiplier = 119.75;
    value.headLearningRateMultiplier = 25.0;
    value.checkpointInterval = 20;
    value.trainingObjectiveCanonical = EA::TrainingObjective::CanonicalText(
        EA::TrainingObjective::Legacy());
    value.trainingObjectiveHash = EA::TrainingObjective::Identity(
        EA::TrainingObjective::Legacy());
    value.trainStart = "2010-01-01";
    value.trainEnd = "2025-01-01";
    value.inferenceStart = "2025-01-01";
    value.inferenceEnd = "2026-01-01";
    value.donchianMode = "enabled";
    value.featureWarmupScope = "full_history_warmup";
    value.donchianLookback = 20;
    value.modelInputWidth = 77;
    value.semanticLayoutVersion = 7;
    value.economicCalendarSnapshotId = 1;
    value.economicCalendarSnapshotHash =
        "fnv1a64:67610f94f5c8e7cc";
    value.baseLearningRate = 1.0e-3 / 3.0;
    value.batchSize = 256;
    value.freshInitializationSeed = 42U;
    value.checkpointPolicyScope = "symbol_horizon";
    value.checkpointPolicyStopMode = "next_checkpoint";
    value.checkpointPolicyGraceEvaluations = 1;
    value.checkpointPolicyRevision = 1;
    value.continuationPolicyScientificIdentity = "disabled";
    return value;
}

Replication::MemberEvaluation Corrected(
    std::size_t ordinal,
    long long firstId,
    double aggregate,
    double average)
{
    Replication::MemberEvaluation member;
    member.ordinal = ordinal;
    member.controlExperimentId = firstId;
    member.treatmentExperimentId = firstId + 1;
    member.symbol = firstId == 624 ? "eurusdrmp" :
        (firstId == 626 ? "gbpusdrmp" : "usdcadrmp");
    member.predictionHorizon = firstId == 624 ? 4 : 6;
    member.evidenceState = Replication::MemberEvidenceState::Complete;
    member.pairDisposition = Pair::Disposition::ComparableComplete;
    member.evidenceClassification =
        Pair::EvidenceClassification::CorrectedCausalSurprisePairEvidence;
    member.pairEvaluationIdentityHash =
        "fnv1a64:000000000000000" + std::to_string(ordinal);
    member.ablationIdentityHash = "fnv1a64:aaaaaaaaaaaaaaaa";
    member.controlEconomicCalendarSnapshotId = 1;
    member.controlEconomicCalendarSnapshotHash =
        "fnv1a64:67610f94f5c8e7cc";
    member.treatmentEconomicCalendarSnapshotId = 1;
    member.treatmentEconomicCalendarSnapshotHash =
        "fnv1a64:67610f94f5c8e7cc";
    member.controlModelInputWidth = 77;
    member.controlModelInputLayoutVersion = 7;
    member.treatmentModelInputWidth = 77;
    member.treatmentModelInputLayoutVersion = 7;
    member.scientificallyValidComplete = true;
    member.comparison.disposition = Pair::Disposition::ComparableComplete;
    member.comparison.evidenceClassification = member.evidenceClassification;
    member.comparison.aggregateProfitability.controlMinusAblation = aggregate;
    member.comparison.averageProfitability.controlMinusAblation = average;
    member.comparison.inferenceAccuracy.controlMinusAblation = 0.0;
    member.comparison.leaderScore.controlMinusAblation = 0.0;
    member.comparison.neutralProportion.controlMinusAblation = 0.0;
    member.comparison.actionableCount.controlMinusAblation = 0.0;
    return member;
}

Replication::MemberEvaluation Historical(std::size_t ordinal,
                                         long long firstId)
{
    auto member = Corrected(ordinal, firstId, 0.0, 0.0);
    member.evidenceState = Replication::MemberEvidenceState::HistoricalPreFix;
    member.evidenceClassification =
        Pair::EvidenceClassification::PreFixCausalSurpriseEvidence;
    member.comparison.evidenceClassification = member.evidenceClassification;
    member.controlModelInputLayoutVersion = 6;
    member.treatmentModelInputLayoutVersion = 6;
    member.scientificallyValidComplete = false;
    return member;
}

Replication::MemberEvaluation Incomplete(std::size_t ordinal,
                                         long long firstId)
{
    auto member = Corrected(ordinal, firstId, 0.0, 0.0);
    member.evidenceState = Replication::MemberEvidenceState::Incomplete;
    member.pairDisposition = Pair::Disposition::ComparableIncomplete;
    member.scientificallyValidComplete = false;
    member.comparison = {};
    member.comparison.evidenceClassification = member.evidenceClassification;
    return member;
}

template <typename Function>
void AssertInvalid(Function&& function)
{
    bool threw = false;
    try { function(); }
    catch (const std::invalid_argument&) { threw = true; }
    assert(threw);
}

Pair::ArmEvidence PlannedEvidence(
    const Continuation::Plan& plan,
    const Continuation::Pair& pair,
    const Continuation::Arm& arm)
{
    Pair::ArmEvidence evidence;
    const auto& configured = plan.configuration;
    auto& actual = evidence.authoritative.configuration;
    actual.experimentId = arm.role == "control" ? 7001 : 7002;
    actual.symbol = pair.symbol;
    actual.predictionHorizon = pair.predictionHorizon;
    actual.targetEpochs = configured.targetEpochs;
    actual.threshold = configured.threshold;
    actual.coreLearningRateMultiplier = configured.coreLearningRateMultiplier;
    actual.headLearningRateMultiplier = configured.headLearningRateMultiplier;
    actual.checkpointInterval = configured.checkpointInterval;
    actual.trainStart = configured.trainStart;
    actual.trainEnd = configured.trainEnd;
    actual.inferenceStart = configured.inferenceStart;
    actual.inferenceEnd = configured.inferenceEnd;
    actual.donchianMode = configured.donchianMode;
    actual.featureWarmupScope = configured.featureWarmupScope;
    actual.donchianLookback = configured.donchianLookback;
    actual.featureAblationMask = arm.featureAblationMask;
    actual.experimentObjective = {
        configured.trainingObjectiveCanonical,
        configured.trainingObjectiveHash};
    auto& extended = evidence.extended;
    extended.configuredModelInputWidth = configured.modelInputWidth;
    extended.configuredModelInputLayoutVersion =
        configured.semanticLayoutVersion;
    extended.economicCalendarSnapshotId =
        configured.economicCalendarSnapshotId;
    extended.economicCalendarSnapshotHash =
        configured.economicCalendarSnapshotHash;
    extended.baseLearningRate = configured.baseLearningRate;
    extended.batchSize = configured.batchSize;
    extended.freshInitializationSeed = configured.freshInitializationSeed;
    extended.checkpointInferenceEnabled =
        configured.checkpointInferenceEnabled;
    extended.checkpointInferenceMinimumEpoch =
        configured.checkpointInferenceMinimumEpoch;
    extended.checkpointInferenceInterval =
        configured.checkpointInferenceInterval;
    extended.checkpointPolicyEnabled = configured.checkpointPolicyEnabled;
    extended.checkpointPolicyMinimumLeaderScore =
        configured.checkpointPolicyMinimumLeaderScore;
    extended.checkpointPolicyMinimumInferenceAccuracy =
        configured.checkpointPolicyMinimumInferenceAccuracy;
    extended.checkpointPolicyTopN = configured.checkpointPolicyTopN;
    extended.checkpointPolicyScope = configured.checkpointPolicyScope;
    extended.checkpointPolicyStopMode = configured.checkpointPolicyStopMode;
    extended.checkpointPolicyGraceEvaluations =
        configured.checkpointPolicyGraceEvaluations;
    extended.checkpointPolicyRevision = configured.checkpointPolicyRevision;
    extended.checkpointPolicyHash = configured.checkpointPolicyHash;
    extended.continuationPolicyEnabled =
        configured.continuationPolicyEnabled;
    extended.continuationPolicyScientificIdentity =
        configured.continuationPolicyScientificIdentity;
    evidence.operational.schedulerPriority = configured.schedulerPriority;
    return evidence;
}

} // namespace

int main()
{
    const auto plan = Continuation::MakePlan(Configuration());
    const auto repeated = Continuation::MakePlan(Configuration());
    assert(plan == repeated);
    assert(plan.canonical == repeated.canonical);
    assert(plan.hash == repeated.hash);
    assert(plan.pairs.size() == 2);
    assert(plan.pairs[0].ordinal == 1);
    assert(plan.pairs[0].symbol == "gbpusdrmp");
    assert(plan.pairs[0].predictionHorizon == 6);
    assert(plan.pairs[1].ordinal == 2);
    assert(plan.pairs[1].symbol == "usdcadrmp");
    assert(plan.pairs[1].predictionHorizon == 6);
    assert(plan.hash == Continuation::kPredeclaredPlanHash);
    assert(plan.pairs[0].replicationUnitHash ==
           Continuation::kFirstReplicationUnitHash);
    assert(plan.pairs[0].control.scientificIdentityHash ==
           Continuation::kFirstControlIdentityHash);
    assert(plan.pairs[0].treatment.scientificIdentityHash ==
           Continuation::kFirstTreatmentIdentityHash);
    assert(plan.pairs[1].replicationUnitHash ==
           Continuation::kSecondReplicationUnitHash);
    assert(plan.pairs[1].control.scientificIdentityHash ==
           Continuation::kSecondControlIdentityHash);
    assert(plan.pairs[1].treatment.scientificIdentityHash ==
           Continuation::kSecondTreatmentIdentityHash);
    assert(plan.pairs[0].control.role == "control");
    assert(plan.pairs[0].control.featureAblationMask.empty());
    assert(plan.pairs[0].treatment.role == "treatment");
    assert(plan.pairs[0].treatment.featureAblationMask ==
           "causal_first_release_surprise_available,"
           "causal_first_release_surprise");
    assert(plan.canonical.find("plan_semantic_version=1;") !=
           std::string::npos);
    const std::string hypotheticalVersion2 =
        "corrected_causal_surprise_replication_plan_v2;" + plan.canonical;
    assert(EA::TrainingObjective::DeterministicHash(hypotheticalVersion2) !=
           plan.hash);
    const std::string controlProvenance =
        Continuation::MaterializationProvenance(
            plan, plan.pairs[0], plan.pairs[0].control);
    assert(controlProvenance.find("plan_hash=" + plan.hash) !=
           std::string::npos);
    assert(controlProvenance.find("pair_ordinal=1") != std::string::npos);
    assert(controlProvenance.find(
               "replication_unit_hash=" + plan.pairs[0].replicationUnitHash) !=
           std::string::npos);
    assert(controlProvenance.find("arm_role=control") != std::string::npos);
    assert(controlProvenance.find("outcome_blind=true") != std::string::npos);

    const auto plannedControl = PlannedEvidence(
        plan, plan.pairs[0], plan.pairs[0].control);
    const auto plannedTreatment = PlannedEvidence(
        plan, plan.pairs[0], plan.pairs[0].treatment);
    Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, plannedControl);
    Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].treatment, plannedTreatment);
    auto tamperedArm = plannedControl;
    tamperedArm.authoritative.configuration.symbol = "usdcadrmp";
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.authoritative.configuration.predictionHorizon = 4;
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.authoritative.configuration.featureAblationMask = "close";
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.extended.configuredModelInputWidth = 75;
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.extended.configuredModelInputLayoutVersion = 6;
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.extended.economicCalendarSnapshotId = 2;
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.extended.economicCalendarSnapshotHash =
        "fnv1a64:0000000000000000";
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.authoritative.configuration.experimentObjective.hash =
        "fnv1a64:0000000000000000";
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    tamperedArm = plannedControl;
    tamperedArm.authoritative.configuration.trainEnd = "2024-12-31";
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, tamperedArm); });
    AssertInvalid([&] { Continuation::ValidatePlannedArmEvidence(
        plan, plan.pairs[0], plan.pairs[0].control, plannedTreatment); });

    // Planning accepts scientific identity only. Favorable, unfavorable,
    // mixed, and absent metrics therefore cannot alter membership or hashes.
    const auto favorable = Corrected(2, 624, 1.0, 1.0);
    const auto unfavorable = Corrected(2, 624, -1.0, -1.0);
    const auto mixed = Corrected(2, 624, 1.0, -1.0);
    const auto absent = Incomplete(2, 624);
    assert(Continuation::MakePlan(Configuration()).hash == plan.hash);
    (void)favorable;
    (void)unfavorable;
    (void)mixed;
    (void)absent;

    auto historical = Historical(1, 622);
    auto incomplete = Incomplete(2, 624);
    const auto awaitingEvaluation = Replication::Evaluate(
        {historical, incomplete}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    const auto awaiting = Continuation::EvaluateGate(
        awaitingEvaluation, awaitingEvaluation.members[1]);
    assert(awaiting.correctedValidPairCount == 0);
    assert(awaiting.historicalPreFixPairCount == 1);
    assert(awaiting.nextAction == Continuation::NextAction::AwaitPairCompletion);
    assert(!awaiting.nextPlanPairOrdinal);

    for (const auto& anchor : {favorable, unfavorable, mixed})
    {
        const auto evaluation = Replication::Evaluate(
            {historical, anchor}, {}, {},
            Replication::EvidenceScope::CorrectedCausalSurprise);
        const auto gate = Continuation::EvaluateGate(
            evaluation, evaluation.members[1]);
        assert(gate.correctedValidPairCount == 1);
        assert(gate.nextAction ==
               Continuation::NextAction::PrepareAdditionalReplications);
        assert(gate.nextPlanPairOrdinal == 1);
    }

    const auto twoEvaluation = Replication::Evaluate(
        {historical, favorable, Corrected(3, 626, -2.0, -2.0)}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    const auto twoGate = Continuation::EvaluateGate(
        twoEvaluation, twoEvaluation.members[1]);
    assert(twoGate.correctedValidPairCount == 2);
    assert(twoGate.nextPlanPairOrdinal == 2);

    const auto threeEvaluation = Replication::Evaluate(
        {historical, favorable, Corrected(3, 626, -2.0, -2.0),
         Corrected(4, 628, 0.0, 0.0)}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    const auto threeGate = Continuation::EvaluateGate(
        threeEvaluation, threeEvaluation.members[1]);
    assert(threeGate.correctedValidPairCount == 3);
    assert(threeGate.correctedReplicationMinimumSatisfied);
    assert(threeGate.nextAction ==
           Continuation::NextAction::ReplicationThresholdSatisfied);
    assert(!threeGate.nextPlanPairOrdinal);

    auto invalidAnchor = unfavorable;
    invalidAnchor.evidenceState = Replication::MemberEvidenceState::Invalid;
    invalidAnchor.scientificallyValidComplete = false;
    invalidAnchor.evidenceClassification =
        Pair::EvidenceClassification::IncompatibleOrInvalidEvidence;
    invalidAnchor.comparison.evidenceClassification =
        invalidAnchor.evidenceClassification;
    const auto invalidEvaluation = Replication::Evaluate(
        {historical, invalidAnchor}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    const auto invalidGate = Continuation::EvaluateGate(
        invalidEvaluation, invalidEvaluation.members[1]);
    assert(invalidGate.nextAction ==
           Continuation::NextAction::InvalidPairRequiresReview);

    auto bad = Configuration();
    bad.semanticLayoutVersion = 6;
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.modelInputWidth = 75;
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.economicCalendarSnapshotHash.clear();
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.economicCalendarSnapshotId = 0;
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.trainingObjectiveHash = "fnv1a64:0000000000000000";
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.trainEnd = bad.trainStart;
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });
    bad = Configuration();
    bad.initialSymbol = "gbpusdrmp";
    AssertInvalid([&] { (void)Continuation::MakePlan(bad); });

    auto invalidPlan = plan;
    invalidPlan.pairs[0].control.featureAblationMask = "close";
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.pairs[0].treatment.featureAblationMask =
        "causal_first_release_surprise";
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.pairs[1].replicationUnitHash =
        invalidPlan.pairs[0].replicationUnitHash;
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.hash = "fnv1a64:0000000000000000";
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.pairs[0].control.scientificIdentityCanonical +=
        "tampered=true;";
    invalidPlan.pairs[0].control.scientificIdentityHash =
        EA::TrainingObjective::DeterministicHash(
            invalidPlan.pairs[0].control.scientificIdentityCanonical);
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.pairs[0].replicationUnitCanonical += "tampered=true;";
    invalidPlan.pairs[0].replicationUnitHash =
        EA::TrainingObjective::DeterministicHash(
            invalidPlan.pairs[0].replicationUnitCanonical);
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });
    invalidPlan = plan;
    invalidPlan.canonical += "tampered=true;";
    invalidPlan.hash = EA::TrainingObjective::DeterministicHash(
        invalidPlan.canonical);
    AssertInvalid([&] { Continuation::ValidatePlan(invalidPlan); });

    const std::string rendered = Continuation::RenderPlan(plan);
    assert(rendered.find("outcome_metrics_in_plan=false") != std::string::npos);
    assert(rendered.find("materialized=false") != std::string::npos);
    assert(rendered.find(plan.hash) != std::string::npos);
    assert(Continuation::RenderGate(awaiting).find(
               "next_action=await_pair_completion") != std::string::npos);

    std::cout <<
        "CorrectedCausalSurpriseReplicationContinuationTests passed\n";
    return 0;
}
