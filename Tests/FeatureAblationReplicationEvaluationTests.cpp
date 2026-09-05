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
    assert(early.evaluationIdentityCanonical.find(
               "pair_evaluation_semantic_version=2;") !=
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
