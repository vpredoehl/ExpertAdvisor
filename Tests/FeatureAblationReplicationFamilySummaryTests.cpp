#include "FeatureAblationReplicationFamilySummary.hpp"

#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Family = EA::FeatureAblationReplicationFamilySummary;

namespace
{

Family::ReplicationMember Member(unsigned int seed,
                                 double leader,
                                 double aggregate,
                                 double inference = 0.0)
{
    Family::ReplicationMember member;
    member.freshInitializationSeed = seed;
    member.controlExperimentIdentity = "control-identity-" + std::to_string(seed);
    member.ablationExperimentIdentity = "ablation-identity-" + std::to_string(seed);
    member.pairMemberIdentity = "pair-evidence-" + std::to_string(seed);
    member.interventionIdentity = "feature-set:example";
    member.scientificConfigurationIdentity = "science-config:example";
    member.pairValid = true;
    member.pairReady = true;
    member.pairComplete = true;
    member.pairEvidence.leaderScore.controlMinusAblation = leader;
    member.pairEvidence.aggregateProfitability.controlMinusAblation = aggregate;
    member.pairEvidence.inferenceAccuracy.controlMinusAblation = inference;
    return member;
}

const Family::MetricFamilySummary& Metric(
    const Family::FamilySummary& family, Family::Metric metric)
{
    for (const auto& summary : family.metrics)
        if (summary.metric == metric) return summary;
    assert(false);
    return family.metrics.front();
}

void ExpectInvalid(const auto& function, const std::string& reason)
{
    try
    {
        function();
        assert(false);
    }
    catch (const std::invalid_argument& error)
    {
        assert(std::string{error.what()}.find(reason) != std::string::npos);
    }
}

} // namespace

int main()
{
    // Zero members fail closed. One member has a population standard deviation
    // of zero and leave-one-out explicitly contains zero remaining members.
    ExpectInvalid([] { (void)Family::Summarize({}); }, "replication_family_empty");
    const auto one = Family::Summarize({Member(9, 2.0, 3.0)});
    const auto& oneLeader = Metric(one, Family::Metric::LeaderScore);
    assert(oneLeader.descriptive->memberCount == 1);
    assert(*oneLeader.descriptive->medianDelta == 2.0);
    assert(*oneLeader.descriptive->populationStandardDeviation == 0.0);
    assert(oneLeader.leaveOneOut.size() == 1);
    assert(oneLeader.leaveOneOut[0].summary.memberCount == 0);

    // The input order does not affect members, descriptive values, or the
    // leave-one-out ordering (all are canonicalized by initialization seed).
    const auto canonical = Family::Summarize({
        Member(30, 3.0, 30.0), Member(10, -1.0, 10.0), Member(20, 0.0, 20.0)});
    const auto permuted = Family::Summarize({
        Member(20, 0.0, 20.0), Member(30, 3.0, 30.0), Member(10, -1.0, 10.0)});
    assert(canonical.members[0].freshInitializationSeed == 10);
    assert(canonical.members[1].freshInitializationSeed == 20);
    assert(canonical.members[2].freshInitializationSeed == 30);
    const auto& leaders = Metric(canonical, Family::Metric::LeaderScore);
    const auto& permutedLeaders = Metric(permuted, Family::Metric::LeaderScore);
    assert(leaders.category == Family::MetricCategory::ModelPerformance);
    assert(Metric(canonical, Family::Metric::PredictionCount).category ==
           Family::MetricCategory::BehavioralCoverage);
    assert(Metric(canonical, Family::Metric::ActionableCount).category ==
           Family::MetricCategory::Profitability);
    assert(leaders.descriptive->memberCount == 3);
    assert(leaders.descriptive->positiveCount == 1);
    assert(leaders.descriptive->zeroCount == 1);
    assert(leaders.descriptive->negativeCount == 1);
    assert(*leaders.descriptive->meanDelta == 2.0 / 3.0);
    assert(*leaders.descriptive->medianDelta == 0.0);
    assert(*leaders.descriptive->minimumDelta == -1.0);
    assert(*leaders.descriptive->maximumDelta == 3.0);
    assert(std::fabs(*leaders.descriptive->populationStandardDeviation -
                     std::sqrt(26.0 / 9.0)) < 1.0e-15);
    assert(*leaders.descriptive->meanDelta == *permutedLeaders.descriptive->meanDelta);
    assert(leaders.leaveOneOut.size() == 3);
    assert(leaders.leaveOneOut[0].omittedFreshInitializationSeed == 10);
    assert(leaders.leaveOneOut[0].summary.memberCount == 2);
    assert(*leaders.leaveOneOut[0].summary.meanDelta == 1.5);
    assert(*leaders.leaveOneOut[0].summary.medianDelta == 1.5);
    assert(leaders.leaveOneOut[0].summary.positiveCount == 1);
    assert(leaders.leaveOneOut[0].summary.zeroCount == 1);
    assert(leaders.leaveOneOut[0].summary.negativeCount == 0);
    assert(leaders.leaveOneOut[2].omittedFreshInitializationSeed == 30);
    assert(*leaders.leaveOneOut[2].summary.meanDelta == -0.5);

    // Mixed finite magnitudes retain finite, deterministic summaries with the
    // long-double accumulation path on platforms whose long double is either
    // wider than, or equivalent to, double.
    const auto mixedMagnitude = Family::Summarize({
        Member(3, 1.0e150, 3.0), Member(1, -1.0e150, 1.0),
        Member(2, 1.0, 2.0)});
    const auto mixedMagnitudePermuted = Family::Summarize({
        Member(2, 1.0, 2.0), Member(3, 1.0e150, 3.0),
        Member(1, -1.0e150, 1.0)});
    const auto& mixedLeaders = Metric(mixedMagnitude, Family::Metric::LeaderScore);
    const auto& mixedPermutedLeaders =
        Metric(mixedMagnitudePermuted, Family::Metric::LeaderScore);
    assert(std::isfinite(*mixedLeaders.descriptive->meanDelta));
    assert(std::isfinite(*mixedLeaders.descriptive->populationStandardDeviation));
    assert(*mixedLeaders.descriptive->meanDelta ==
           *mixedPermutedLeaders.descriptive->meanDelta);
    assert(*mixedLeaders.descriptive->populationStandardDeviation ==
           *mixedPermutedLeaders.descriptive->populationStandardDeviation);

    // Exact even median: no epsilon is used for either signs or identities.
    const auto two = Family::Summarize({Member(1, -2.0, -4.0), Member(2, 4.0, 8.0)});
    const auto& twoLeaders = Metric(two, Family::Metric::LeaderScore);
    assert(*twoLeaders.descriptive->medianDelta == 1.0);
    assert(*twoLeaders.descriptive->meanDelta == 1.0);
    assert(*twoLeaders.descriptive->populationStandardDeviation == 3.0);
    assert(twoLeaders.leaveOneOut[0].summary.memberCount == 1);
    assert(*twoLeaders.leaveOneOut[0].summary.medianDelta == 4.0);

    // All unavailable is represented, while partial availability fails closed.
    const auto& absentAcceptRate = Metric(canonical, Family::Metric::AcceptRate);
    assert(!absentAcceptRate.descriptive.has_value());
    assert(absentAcceptRate.leaveOneOut.empty());
    auto partial = Member(1, 1.0, 1.0);
    auto partialOther = Member(2, 2.0, 2.0);
    partial.pairEvidence.acceptRate.controlMinusAblation = 0.2;
    ExpectInvalid([&] { (void)Family::Summarize({partial, partialOther}); },
                  "metric_availability_inconsistent:accept_rate");

    auto duplicateSeed = Member(1, 1.0, 1.0);
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), duplicateSeed}); },
                  "duplicate_seed");
    auto duplicateMember = Member(2, 2.0, 2.0);
    duplicateMember.pairMemberIdentity = "pair-evidence-1";
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), duplicateMember}); },
                  "duplicate_member");
    auto duplicatePair = Member(2, 2.0, 2.0);
    duplicatePair.controlExperimentIdentity = "control-identity-1";
    duplicatePair.ablationExperimentIdentity = "ablation-identity-1";
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), duplicatePair}); },
                  "duplicate_member");
    auto reusedArm = Member(2, 2.0, 2.0);
    reusedArm.controlExperimentIdentity = "control-identity-1";
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), reusedArm}); },
                  "duplicate_member");

    auto wrongIntervention = Member(2, 2.0, 2.0);
    wrongIntervention.interventionIdentity = "feature-set:other";
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), wrongIntervention}); },
                  "intervention_identity_mismatch");
    auto wrongConfiguration = Member(2, 2.0, 2.0);
    wrongConfiguration.scientificConfigurationIdentity = "science-config:other";
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, 1.0, 1.0), wrongConfiguration}); },
                  "scientific_configuration_identity_mismatch");

    auto invalid = Member(1, 1.0, 1.0);
    invalid.pairValid = false;
    ExpectInvalid([&] { (void)Family::Summarize({invalid}); }, "member_invalid");
    auto notReady = Member(1, 1.0, 1.0);
    notReady.pairReady = false;
    ExpectInvalid([&] { (void)Family::Summarize({notReady}); }, "member_not_ready");
    auto incomplete = Member(1, 1.0, 1.0);
    incomplete.pairComplete = false;
    ExpectInvalid([&] { (void)Family::Summarize({incomplete}); }, "member_incomplete");
    auto nonfinite = Member(1, std::numeric_limits<double>::infinity(), 1.0);
    ExpectInvalid([&] { (void)Family::Summarize({nonfinite}); }, "metric_nonfinite");
    const double large = std::numeric_limits<double>::max();
    ExpectInvalid([&] { (void)Family::Summarize({Member(1, large, 1.0),
                                                  Member(2, -large, 2.0)}); },
                  "metric_dispersion_nonfinite");

    assert(Family::MetricCategoryText(Family::MetricCategory::Profitability) ==
           "profitability");
    assert(Family::MetricText(Family::Metric::AggregateProfitability) ==
           "aggregate_terminal_horizon_log_return");
    std::cout << "FeatureAblationReplicationFamilySummaryTests passed\n";
    return 0;
}
