#include "FeatureAblationReplicationEvaluation.hpp"

#include <cassert>
#include <iostream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace Replication = EA::FeatureAblationReplicationEvaluation;
namespace Pair = EA::FeatureAblationPairEvaluation;

namespace
{

Replication::MemberEvaluation Complete(
    std::size_t ordinal,
    double aggregate,
    double average,
    double inference = 0.0,
    double leader = 0.0,
    double neutral = 0.0,
    double actionable = 0.0)
{
    Replication::MemberEvaluation member;
    member.ordinal = ordinal;
    member.controlExperimentId = 600 + static_cast<long long>(ordinal * 2 - 1);
    member.treatmentExperimentId = 600 + static_cast<long long>(ordinal * 2);
    member.symbol = "symbol" + std::to_string(ordinal);
    member.predictionHorizon = ordinal % 2 == 0 ? 6 : 4;
    member.evidenceState = Replication::MemberEvidenceState::Complete;
    member.pairDisposition = Pair::Disposition::ComparableComplete;
    member.pairEvaluationIdentityHash =
        "fnv1a64:000000000000000" + std::to_string(ordinal);
    member.ablationIdentityHash = "fnv1a64:aaaaaaaaaaaaaaaa";
    member.scientificallyValidComplete = true;
    if (ordinal >= 3)
    {
        member.controlEconomicCalendarSnapshotId = 1;
        member.controlEconomicCalendarSnapshotHash =
            "fnv1a64:67610f94f5c8e7cc";
        member.treatmentEconomicCalendarSnapshotId = 1;
        member.treatmentEconomicCalendarSnapshotHash =
            "fnv1a64:67610f94f5c8e7cc";
    }
    member.comparison.disposition = Pair::Disposition::ComparableComplete;
    member.comparison.aggregateProfitability.controlMinusAblation = aggregate;
    member.comparison.averageProfitability.controlMinusAblation = average;
    member.comparison.inferenceAccuracy.controlMinusAblation = inference;
    member.comparison.leaderScore.controlMinusAblation = leader;
    member.comparison.neutralProportion.controlMinusAblation = neutral;
    member.comparison.actionableCount.controlMinusAblation = actionable;
    return member;
}

Replication::MemberEvaluation Incomplete(std::size_t ordinal)
{
    auto member = Complete(ordinal, 0.0, 0.0);
    member.evidenceState = Replication::MemberEvidenceState::Incomplete;
    member.pairDisposition = Pair::Disposition::ComparableIncomplete;
    member.scientificallyValidComplete = false;
    member.incompleteReasons = {"control_experiment_not_complete"};
    member.comparison = {};
    return member;
}

Replication::MemberEvaluation Corrected(std::size_t ordinal,
                                        double aggregate,
                                        double average)
{
    auto member = Complete(ordinal, aggregate, average);
    member.evidenceClassification = Pair::EvidenceClassification::
        CorrectedCausalSurprisePairEvidence;
    member.comparison.evidenceClassification =
        member.evidenceClassification;
    member.controlModelInputWidth = 77;
    member.controlModelInputLayoutVersion = 7;
    member.treatmentModelInputWidth = 77;
    member.treatmentModelInputLayoutVersion = 7;
    member.controlEconomicCalendarSnapshotId = 1;
    member.controlEconomicCalendarSnapshotHash =
        "fnv1a64:67610f94f5c8e7cc";
    member.treatmentEconomicCalendarSnapshotId = 1;
    member.treatmentEconomicCalendarSnapshotHash =
        "fnv1a64:67610f94f5c8e7cc";
    return member;
}

Replication::MemberEvaluation PreFix(std::size_t ordinal)
{
    auto member = Complete(ordinal, 0.5, 0.05);
    member.evidenceClassification =
        Pair::EvidenceClassification::PreFixCausalSurpriseEvidence;
    member.comparison.evidenceClassification =
        member.evidenceClassification;
    member.evidenceState = Replication::MemberEvidenceState::HistoricalPreFix;
    member.scientificallyValidComplete = false;
    member.controlModelInputWidth = 77;
    member.controlModelInputLayoutVersion = 6;
    member.treatmentModelInputWidth = 77;
    member.treatmentModelInputLayoutVersion = 6;
    member.exclusionReasons = {
        "semantic_layout_6_excluded_from_corrected_replication"};
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

} // namespace

int main()
{
    using Decision = Replication::ReplicationDecision;
    using Action = Replication::ProductionizationAction;

    const auto ids = Replication::ParseExperimentIdPairs(
        "601:602,603:604,605:606,607:608");
    assert(ids.size() == 4);
    assert(ids.front() == std::make_pair(601LL, 602LL));
    assert(ids.back() == std::make_pair(607LL, 608LL));
    AssertInvalid([] { (void)Replication::ParseExperimentIdPairs(""); });
    AssertInvalid([] {
        (void)Replication::ParseExperimentIdPairs("601:602,601:604");
    });
    AssertInvalid([] {
        (void)Replication::ParseExperimentIdPairs("601:602,603:602");
    });
    AssertInvalid([] {
        (void)Replication::ParseExperimentIdPairs("601:602,601:602");
    });

    const auto early = Replication::Evaluate(
        {Complete(1, 0.1, 0.01), Incomplete(2), Incomplete(3), Incomplete(4)});
    assert(early.decision == Decision::InsufficientEvidence);
    assert(early.softwareReady);
    assert(early.action == Action::AwaitReplication);
    assert(early.population.completeComparablePairCount == 1);
    assert(early.population.incompletePairCount == 3);
    assert(early.population.distinctEconomicCalendarCorpusCount == 2);
    assert(early.evaluationIdentityCanonical.find(
               "pair_evaluation_semantic_version=3;") !=
           std::string::npos);
    assert(early.evaluationIdentityCanonical.find(
               "economic_calendar_snapshot=provenance_not_treatment;") !=
           std::string::npos);
    assert(Replication::ExitCode(early) == 4);

    const auto promising = Replication::Evaluate({
        Complete(1, 0.1, 0.01, -0.5),
        Complete(2, 0.2, 0.02, -0.4),
        Complete(3, 0.3, 0.03, -0.3)});
    assert(promising.decision == Decision::Promising);
    assert(promising.action == Action::EligibleForActivationReview);
    assert(promising.inferenceAccuracy.negativeCount == 3);
    assert(promising.aggregateProfitability.median == 0.2);

    const auto notPromising = Replication::Evaluate({
        Complete(1, -0.1, -0.01, 0.5),
        Complete(2, -0.2, -0.02, 0.4),
        Complete(3, 0.01, 0.001, 0.3)});
    assert(notPromising.decision == Decision::NotPromising);
    assert(notPromising.action == Action::DoNotEnable);

    const auto conflicting = Replication::Evaluate({
        Complete(1, 0.3, -0.03),
        Complete(2, 0.2, -0.02),
        Complete(3, -0.1, 0.04)});
    assert(conflicting.decision == Decision::Mixed);
    assert(conflicting.action == Action::DoNotEnable);

    const auto outlier = Replication::Evaluate({
        Complete(1, -1.0, -1.0),
        Complete(2, -1.0, -1.0),
        Complete(3, 100.0, 100.0)});
    assert(outlier.decision == Decision::Mixed);
    assert(outlier.decision != Decision::Promising);

    auto invalidMember = Incomplete(4);
    invalidMember.evidenceState = Replication::MemberEvidenceState::Invalid;
    invalidMember.pairDisposition = Pair::Disposition::InvalidAblationPair;
    invalidMember.invalidReasons = {"invalid_ablation_pair"};
    const auto invalid = Replication::Evaluate({
        Complete(1, 0.1, 0.01), Complete(2, 0.2, 0.02),
        Complete(3, 0.3, 0.03), invalidMember});
    assert(invalid.population.invalidPairCount == 1);
    assert(invalid.evidenceIntegrityFailure);
    assert(invalid.decision == Decision::Mixed);
    assert(Replication::ExitCode(invalid) == 3);

    auto ablationMismatch = Complete(3, 0.3, 0.03);
    ablationMismatch.ablationIdentityHash = "fnv1a64:bbbbbbbbbbbbbbbb";
    const auto mismatch = Replication::Evaluate({
        Complete(1, 0.1, 0.01), Complete(2, 0.2, 0.02),
        ablationMismatch});
    assert(mismatch.population.invalidPairCount == 1);
    assert(mismatch.members[2].invalidReasons.front() ==
           "replication_ablation_identity_mismatch");

    const auto repeat = Replication::Evaluate({
        Complete(1, 0.1, 0.01), Incomplete(2), Incomplete(3), Incomplete(4)});
    assert(repeat.membershipIdentityCanonical ==
           early.membershipIdentityCanonical);
    assert(repeat.membershipIdentityHash == early.membershipIdentityHash);
    assert(repeat.evaluationIdentityHash == early.evaluationIdentityHash);

    auto permutedFirst = Complete(1, 0.2, 0.02);
    auto permutedSecond = Complete(2, 0.1, 0.01);
    std::swap(permutedFirst.controlExperimentId,
              permutedSecond.controlExperimentId);
    std::swap(permutedFirst.treatmentExperimentId,
              permutedSecond.treatmentExperimentId);
    const auto permuted = Replication::Evaluate(
        {permutedFirst, permutedSecond, Incomplete(3), Incomplete(4)});
    assert(permuted.membershipIdentityHash != early.membershipIdentityHash);

    auto badOrdinal = Complete(2, 0.1, 0.01);
    AssertInvalid([&] { (void)Replication::Evaluate({badOrdinal}); });

    Replication::SoftwareReadinessAudit notReadyAudit;
    notReadyAudit.finiteFeatureValuesEnforced = false;
    const auto blocked = Replication::Evaluate(
        {Complete(1, 0.1, 0.01)}, {}, notReadyAudit);
    assert(!blocked.softwareReady);
    assert(blocked.action == Action::BlockedSoftwareReadiness);

    const auto correctedReplicated = Replication::Evaluate(
        {Corrected(1, 0.1, 0.01), Corrected(2, 0.2, 0.02),
         Corrected(3, 0.3, 0.03)}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    assert(correctedReplicated.population.correctedValidPairCount == 3);
    assert(correctedReplicated.population.historicalPreFixPairCount == 0);
    assert(correctedReplicated.evidenceClassification ==
           Replication::EvidenceClassification::
               CorrectedCausalSurpriseReplicationEvidence);

    const auto twoCorrectedOneHistorical = Replication::Evaluate(
        {Corrected(1, 0.1, 0.01), Corrected(2, 0.2, 0.02), PreFix(3)},
        {}, {}, Replication::EvidenceScope::CorrectedCausalSurprise);
    assert(twoCorrectedOneHistorical.population.correctedValidPairCount == 2);
    assert(twoCorrectedOneHistorical.population.historicalPreFixPairCount == 1);
    assert(twoCorrectedOneHistorical.decision == Decision::InsufficientEvidence);
    assert(twoCorrectedOneHistorical.evidenceClassification ==
           Replication::EvidenceClassification::
               CorrectedCausalSurprisePairEvidence);

    const auto oneCorrectedTwoHistorical = Replication::Evaluate(
        {Corrected(1, 0.1, 0.01), PreFix(2), PreFix(3)}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    assert(oneCorrectedTwoHistorical.population.correctedValidPairCount == 1);
    assert(oneCorrectedTwoHistorical.population.historicalPreFixPairCount == 2);

    auto invalidCorrected = Corrected(3, 0.3, 0.03);
    invalidCorrected.treatmentModelInputLayoutVersion = 6;
    const auto invalidCorrectedSet = Replication::Evaluate(
        {Corrected(1, 0.1, 0.01), Corrected(2, 0.2, 0.02),
         invalidCorrected}, {}, {},
        Replication::EvidenceScope::CorrectedCausalSurprise);
    assert(invalidCorrectedSet.population.correctedValidPairCount == 2);
    assert(invalidCorrectedSet.population.invalidPairCount == 1);
    assert(invalidCorrectedSet.evidenceClassification ==
           Replication::EvidenceClassification::IncompatibleOrInvalidEvidence);
    assert(invalidCorrectedSet.members[2].invalidReasons.front() ==
           "corrected_causal_surprise_input_identity_invalid");

    // Every action is advisory. No state represents automatic activation.
    for (const Action action : {
             early.action, promising.action, notPromising.action,
             conflicting.action, blocked.action})
    {
        assert(Replication::ProductionizationActionText(action) !=
               "enable_production");
    }

    std::cout << "FeatureAblationReplicationEvaluationTests passed\n";
    return 0;
}
