#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "CausalTrendLineBreakRetestBehavior.hpp"

namespace
{
using EA::TG1A::Candle;
using EA::TG1A::TrendLineDirection;
using EA::TG1B::CalibrationConfiguration;
using EA::TG1B::ClassifiedTrendLineCandidate;
using EA::TG1B::SteepnessClassification;
using EA::TG2::BreakObservation;
using EA::TG2::CausalTrendLineBreakRetestBehavior;
using EA::TG2::CensorReason;
using EA::TG2::Configuration;
using EA::TG2::ResolutionState;
using EA::TG2::TrendLineBehaviorTracker;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kHour = 3'600;

Candle Bar(std::size_t index, double open, double high,
           double low, double close)
{
    return {kStart + static_cast<std::int64_t>(index) * kHour,
            open, high, low, close};
}

ClassifiedTrendLineCandidate Candidate(
    TrendLineDirection direction,
    SteepnessClassification classification,
    std::size_t identityOffset,
    double anchorPrice,
    double slope = 0.0)
{
    ClassifiedTrendLineCandidate result;
    result.direction = direction;
    result.anchor1Bar = identityOffset;
    result.anchor1Timestamp = kStart - static_cast<std::int64_t>(
        20 - identityOffset) * kHour;
    result.anchor1Price = anchorPrice;
    result.anchor1ConfirmationBar = identityOffset + 2;
    result.anchor1ConfirmationTimestamp = result.anchor1Timestamp + 2 * kHour;
    result.anchor2Bar = identityOffset + 5;
    result.anchor2Timestamp = result.anchor1Timestamp + 5 * kHour;
    result.anchor2Price = anchorPrice + slope * 5.0;
    result.anchor2ConfirmationBar = identityOffset + 7;
    result.anchor2ConfirmationTimestamp = result.anchor2Timestamp + 2 * kHour;
    result.anchorSeparationBars = 5;
    result.creationBar = 0;
    result.creationTimestamp = kStart;
    result.rawPriceSlopePerBar = slope;
    result.creationAtr = 1.0;
    result.creationAtrNormalizedSlope = slope;
    result.calibrationReferenceBarScale = 1.0;
    result.classification = classification;
    return result;
}

std::vector<ClassifiedTrendLineCandidate> UtlPair()
{
    return {
        Candidate(TrendLineDirection::UTL,
                  SteepnessClassification::Inner, 0, 10.0),
        Candidate(TrendLineDirection::UTL,
                  SteepnessClassification::Outer, 1, 8.0),
        Candidate(TrendLineDirection::UTL,
                  SteepnessClassification::Outer, 2, 9.0)};
}

std::vector<ClassifiedTrendLineCandidate> DtlPair()
{
    return {
        Candidate(TrendLineDirection::DTL,
                  SteepnessClassification::Inner, 0, 20.0),
        Candidate(TrendLineDirection::DTL,
                  SteepnessClassification::Outer, 1, 22.0),
        Candidate(TrendLineDirection::DTL,
                  SteepnessClassification::Outer, 2, 21.0)};
}

Configuration Config(std::size_t retestHorizon = 3,
                     std::size_t outerHorizon = 4)
{
    Configuration result;
    result.retestHorizonBars = retestHorizon;
    result.outerTargetHorizonBars = outerHorizon;
    result.maxActiveBreakObservations = 128;
    result.maxRetainedBreakObservations = 128;
    return result;
}

void Initialize(TrendLineBehaviorTracker& tracker,
                const std::vector<ClassifiedTrendLineCandidate>& candidates,
                const Candle& candle)
{
    const auto update = tracker.ObserveCompletedBar(0, candle, candidates);
    assert(update.newBreakEvents.empty());
}

void TestUtlBreakTimingNoPreBreakAndEpisodeRearm()
{
    TrendLineBehaviorTracker tracker(Config(), {"EURUSD", "1h"});
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.5, 10.5));
    assert(tracker.Observations().empty());

    const auto valid = tracker.ObserveCompletedBar(
        1, Bar(1, 10.4, 10.8, 9.8, 10.2), candidates);
    assert(valid.newBreakEvents.empty());

    const auto broken = tracker.ObserveCompletedBar(
        2, Bar(2, 9.7, 9.9, 9.4, 9.5), candidates);
    assert(broken.newBreakEvents.size() == 1);
    assert(broken.newBreakEvents[0].bar == 2);
    assert(broken.newBreakEvents[0].policy ==
           EA::TG2::BreakPolicy::CompletedCloseBeyondLine);
    assert(broken.newBreakEvents[0].frozenClassification ==
           SteepnessClassification::Inner);
    assert(broken.newBreakEvents[0].penetration == 0.5);

    const auto stillBroken = tracker.ObserveCompletedBar(
        3, Bar(3, 9.4, 9.8, 9.2, 9.3), candidates);
    assert(stillBroken.newBreakEvents.empty());
    assert(tracker.Observations().size() == 1);

    const auto rearmed = tracker.ObserveCompletedBar(
        4, Bar(4, 10.1, 10.4, 9.8, 10.1), candidates);
    assert(rearmed.newBreakEvents.empty());
    const auto secondBreak = tracker.ObserveCompletedBar(
        5, Bar(5, 9.7, 9.9, 9.4, 9.6), candidates);
    assert(secondBreak.newBreakEvents.size() == 1);
    assert(secondBreak.newBreakEvents[0].eventSequence == 2);
}

void TestDtlBreakTimingAndSymmetry()
{
    TrendLineBehaviorTracker tracker(Config());
    const auto candidates = DtlPair();
    Initialize(tracker, candidates, Bar(0, 19.5, 20.5, 19.0, 19.5));
    const auto valid = tracker.ObserveCompletedBar(
        1, Bar(1, 19.6, 20.4, 19.2, 19.8), candidates);
    assert(valid.newBreakEvents.empty());
    const auto broken = tracker.ObserveCompletedBar(
        2, Bar(2, 20.2, 20.6, 20.1, 20.5), candidates);
    assert(broken.newBreakEvents.size() == 1);
    assert(broken.newBreakEvents[0].candidate.direction ==
           TrendLineDirection::DTL);
    assert(broken.newBreakEvents[0].penetration == 0.5);
    assert(tracker.Observations()[0].pairedOuter.has_value());
    assert(tracker.Observations()[0].pairedOuter->projectedPriceAtBreak ==
           21.0);
}

void TestExplicitWickBreakPolicy()
{
    Configuration configuration = Config();
    configuration.breakPolicy = EA::TG2::BreakPolicy::CompletedWickBeyondLine;
    configuration.breakPriceTolerance = 0.1;
    TrendLineBehaviorTracker tracker(configuration);
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 10.0, 10.5));

    const auto boundary = tracker.ObserveCompletedBar(
        1, Bar(1, 10.2, 10.5, 9.9, 10.2), candidates);
    assert(boundary.newBreakEvents.empty());
    const auto breakUpdate = tracker.ObserveCompletedBar(
        2, Bar(2, 10.2, 10.5,
               std::nextafter(9.9, 0.0), 10.2), candidates);
    assert(breakUpdate.newBreakEvents.size() == 1);
    assert(breakUpdate.newBreakEvents[0].observedBreakComponent ==
           std::nextafter(9.9, 0.0));
}

void TestRetestStrictlyAfterBreakAndToleranceBoundaries()
{
    Configuration exactConfig = Config();
    exactConfig.retestPriceTolerance = 0.1;
    TrendLineBehaviorTracker exact(exactConfig);
    const auto candidates = UtlPair();
    Initialize(exact, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    const auto breakUpdate = exact.ObserveCompletedBar(
        1, Bar(1, 9.7, 10.5, 9.4, 9.5), candidates);
    assert(breakUpdate.newBreakEvents.size() == 1);
    assert(breakUpdate.newRetestContacts.empty());
    assert(exact.Observations()[0].retest.state == ResolutionState::Pending);

    const auto boundary = exact.ObserveCompletedBar(
        2, Bar(2, 9.4, 9.9, 9.2, 9.4), candidates);
    assert(boundary.newRetestContacts.size() == 1);
    assert(exact.Observations()[0].retest.state == ResolutionState::Succeeded);
    assert(exact.Observations()[0].retest.latencyBars == 1);

    TrendLineBehaviorTracker inside(exactConfig);
    Initialize(inside, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    inside.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const auto insideContact = inside.ObserveCompletedBar(
        2, Bar(2, 9.4, std::nextafter(9.9, 11.0), 9.2, 9.4),
        candidates);
    assert(insideContact.newRetestContacts.size() == 1);

    TrendLineBehaviorTracker outside(exactConfig);
    Initialize(outside, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    outside.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const double justOutside = std::nextafter(9.9, 0.0);
    const auto missed = outside.ObserveCompletedBar(
        2, Bar(2, 9.4, justOutside, 9.2, 9.4), candidates);
    assert(missed.newRetestContacts.empty());
    assert(outside.Observations()[0].retest.state == ResolutionState::Pending);

    TrendLineBehaviorTracker gap(exactConfig);
    Initialize(gap, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    gap.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const auto gapAcross = gap.ObserveCompletedBar(
        2, Bar(2, 10.3, 10.5, 10.2, 10.3), candidates);
    assert(gapAcross.newRetestContacts.empty());
}

void TestDtlRetestAndOuterTargetSymmetry()
{
    TrendLineBehaviorTracker tracker(Config(3, 4));
    const auto candidates = DtlPair();
    Initialize(tracker, candidates, Bar(0, 19.5, 20.0, 19.0, 19.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 20.3, 20.6, 20.2, 20.5), candidates);
    assert(tracker.Observations().size() == 1);
    assert(tracker.Observations()[0].pairedOuter.has_value());

    const auto retest = tracker.ObserveCompletedBar(
        2, Bar(2, 20.4, 20.8, 20.0, 20.4), candidates);
    assert(retest.newRetestContacts.size() == 1);
    assert(retest.newOuterTargetContacts.empty());
    const auto target = tracker.ObserveCompletedBar(
        3, Bar(3, 20.7, 21.0, 20.4, 20.7), candidates);
    assert(target.newOuterTargetContacts.size() == 1);
    assert(target.newOuterTargetAfterRetestContacts.size() == 1);
}

void TestRetestAndOuterHorizonExactBoundaries()
{
    Configuration configuration = Config(2, 3);
    TrendLineBehaviorTracker tracker(configuration);
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    tracker.ObserveCompletedBar(
        2, Bar(2, 9.5, 9.8, 9.2, 9.4), candidates);
    const auto retestAtDeadline = tracker.ObserveCompletedBar(
        3, Bar(3, 9.8, 10.0, 9.2, 9.7), candidates);
    assert(retestAtDeadline.newRetestContacts.size() == 1);
    assert(tracker.Observations()[0].retest.latencyBars == 2);

    const auto outerAtDeadline = tracker.ObserveCompletedBar(
        4, Bar(4, 9.1, 9.3, 9.0, 9.1), candidates);
    assert(outerAtDeadline.newOuterTargetContacts.size() == 1);
    assert(outerAtDeadline.newOuterTargetAfterRetestContacts.size() == 1);
    assert(tracker.Observations()[0].outerTarget.latencyBars == 3);
    assert(tracker.Observations()[0].outerTargetAfterRetest.state ==
           ResolutionState::Succeeded);
}

void TestCausalPairingAndExactOuterContactBoundary()
{
    Configuration configuration = Config();
    configuration.outerTargetPriceTolerance = 0.05;
    TrendLineBehaviorTracker tracker(configuration);
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const BreakObservation& observation = tracker.Observations()[0];
    assert(observation.pairedOuter.has_value());
    assert(observation.pairedOuter->candidate.anchor1Bar == 2);
    assert(observation.pairedOuter->projectedPriceAtBreak == 9.0);

    const auto contact = tracker.ObserveCompletedBar(
        2, Bar(2, 9.2, 9.4, 9.05, 9.2), candidates);
    assert(contact.newOuterTargetContacts.size() == 1);
    assert(tracker.Observations()[0].outerTarget.state ==
           ResolutionState::Succeeded);
}

void TestOutcomesCensoringUnpairedAndDenominators()
{
    const auto candidates = UtlPair();

    TrendLineBehaviorTracker success(Config(2, 2));
    Initialize(success, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    success.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    success.ObserveCompletedBar(
        2, Bar(2, 9.1, 9.3, 9.0, 9.1), candidates);
    success.ObserveCompletedBar(
        3, Bar(3, 9.2, 9.5, 9.1, 9.2), candidates);
    const auto successSummary = success.AggregateSummary();
    assert(successSummary.innerBreaks == 1);
    assert(successSummary.pairedInnerBreaks == 1);
    assert(successSummary.allEligibleInnerToOuter.successes == 1);
    assert(successSummary.allEligibleInnerToOuter.failures == 0);
    assert(successSummary.allEligibleInnerToOuter.empiricalRate == 1.0);

    TrendLineBehaviorTracker failure(Config(2, 2));
    Initialize(failure, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    failure.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    failure.ObserveCompletedBar(
        2, Bar(2, 9.4, 9.7, 9.2, 9.4), candidates);
    failure.ObserveCompletedBar(
        3, Bar(3, 9.4, 9.7, 9.2, 9.4), candidates);
    const auto failureSummary = failure.AggregateSummary();
    assert(failureSummary.allEligibleInnerToOuter.failures == 1);
    assert(failureSummary.allEligibleInnerToOuter.empiricalRate == 0.0);

    TrendLineBehaviorTracker censored(Config(2, 2));
    Initialize(censored, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    censored.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    censored.Finalize();
    const auto censoredSummary = censored.AggregateSummary();
    assert(censoredSummary.allEligibleInnerToOuter.censored == 1);
    assert(censoredSummary.allEligibleInnerToOuter.resolved == 0);
    assert(!censoredSummary.allEligibleInnerToOuter.empiricalRate.has_value());
    assert(censored.Observations()[0].outerTarget.censorReason ==
           CensorReason::EndOfInput);

    TrendLineBehaviorTracker unpaired(Config());
    const std::vector<ClassifiedTrendLineCandidate> innerOnly =
        {candidates.front()};
    Initialize(unpaired, innerOnly, Bar(0, 10.5, 11.0, 9.8, 10.5));
    unpaired.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), innerOnly);
    const auto unpairedSummary = unpaired.AggregateSummary();
    assert(unpairedSummary.unpairedInnerBreaks == 1);
    assert(unpairedSummary.allEligibleInnerToOuter.eligible == 0);
    assert(unpairedSummary.allEligibleInnerToOuter.failures == 0);
    assert(unpairedSummary.allEligibleInnerToOuter.structurallyIneligible == 1);
}

void TestRetestConditionedAggregationAndCausalOrdering()
{
    TrendLineBehaviorTracker tracker(Config(3, 4));
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    // This candle contacts both lines. Overall Outer succeeds, but the
    // retest-conditioned outcome cannot succeed on the retest candle.
    const auto sameBar = tracker.ObserveCompletedBar(
        2, Bar(2, 9.5, 10.0, 9.0, 9.4), candidates);
    assert(sameBar.newRetestContacts.size() == 1);
    assert(sameBar.newOuterTargetContacts.size() == 1);
    assert(sameBar.newOuterTargetAfterRetestContacts.empty());
    assert(tracker.Observations()[0].outerTargetAfterRetest.state ==
           ResolutionState::Pending);

    const auto later = tracker.ObserveCompletedBar(
        3, Bar(3, 9.2, 9.4, 9.0, 9.2), candidates);
    assert(later.newOuterTargetAfterRetestContacts.size() == 1);
    const auto summary = tracker.AggregateSummary();
    assert(summary.retestedInnerBreaks == 1);
    assert(summary.retestedPairedInnerBreaks == 1);
    assert(summary.retestThenOuter.eligible == 1);
    assert(summary.retestThenOuter.successes == 1);
    assert(summary.retestThenOuter.resolved == 1);
    assert(summary.retestThenOuter.empiricalRate == 1.0);
}

auto EventSnapshot(const BreakObservation& observation)
{
    const auto& event = observation.breakEvent;
    return std::tuple{
        event.eventSequence, event.candidate.direction,
        event.candidate.anchor1Bar, event.candidate.anchor1Timestamp,
        event.candidate.anchor2Bar, event.candidate.anchor2Timestamp,
        event.candidate.creationBar, event.candidate.creationTimestamp,
        event.frozenClassification, event.policy, event.bar, event.timestamp,
        event.projectedLinePrice, event.open, event.high, event.low, event.close,
        event.observedBreakComponent, event.penetration,
        event.directionalDistance};
}

void TestFutureBarsDoNotRewriteEventIdentity()
{
    TrendLineBehaviorTracker tracker(Config());
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const auto fixed = EventSnapshot(tracker.Observations()[0]);
    tracker.ObserveCompletedBar(
        2, Bar(2, 9.2, 10.0, 9.0, 9.2), candidates);
    tracker.ObserveCompletedBar(
        3, Bar(3, 9.2, 9.4, 9.0, 9.2), candidates);
    assert(EventSnapshot(tracker.Observations()[0]) == fixed);
}

void TestCandidateExpiryIsNotBreakAndDoesNotCancelOutcome()
{
    TrendLineBehaviorTracker noBreak(Config());
    const auto candidates = UtlPair();
    Initialize(noBreak, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    const auto expired = noBreak.ObserveCompletedBar(
        1, Bar(1, 10.4, 10.7, 9.8, 10.2), {});
    assert(expired.newBreakEvents.empty());
    assert(noBreak.Observations().empty());

    TrendLineBehaviorTracker pending(Config());
    Initialize(pending, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    pending.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const auto afterExpiry = pending.ObserveCompletedBar(
        2, Bar(2, 9.1, 9.3, 9.0, 9.1), {});
    assert(afterExpiry.newBreakEvents.empty());
    assert(afterExpiry.newOuterTargetContacts.size() == 1);
    assert(pending.Observations()[0].outerTarget.state ==
           ResolutionState::Succeeded);
}

std::vector<Candle> ValidUtlCandles()
{
    const std::vector<double> lows =
        {10.0, 9.0, 5.0, 8.0, 10.0, 9.0, 10.0, 7.0, 10.0, 10.0};
    std::vector<Candle> candles;
    for (std::size_t index = 0; index < lows.size(); ++index)
    {
        const double close = 0.5 * (20.0 + lows[index]);
        candles.push_back(Bar(index, close, 20.0, lows[index], close));
    }
    return candles;
}

using ClassificationIdentity =
    std::tuple<TrendLineDirection, std::size_t, std::size_t, std::size_t,
               SteepnessClassification>;

std::vector<ClassificationIdentity> ClassificationSnapshot(
    const std::vector<ClassifiedTrendLineCandidate>& candidates)
{
    std::vector<ClassificationIdentity> result;
    for (const auto& candidate : candidates)
        result.emplace_back(candidate.direction, candidate.anchor1Bar,
                            candidate.anchor2Bar, candidate.creationBar,
                            candidate.classification);
    return result;
}

void TestIntegratedHistoricalStreamingPrefixAndTG1IdentityStability()
{
    auto candles = ValidUtlCandles();
    candles.push_back(Bar(10, 8.0, 8.1, 7.0, 7.5));
    candles.push_back(Bar(11, 7.8, 8.0, 7.1, 7.6));
    EA::TG1A::Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    Configuration behaviorConfiguration = Config(2, 3);

    CausalTrendLineBreakRetestBehavior streaming(
        CalibrationConfiguration(20.0), geometryConfiguration,
        behaviorConfiguration, {"EURUSD", "1h"});
    EA::TG1B::CausalFractalTrendLineAngleClassification baseline(
        CalibrationConfiguration(20.0), geometryConfiguration,
        {"EURUSD", "1h"});

    for (std::size_t index = 0; index < candles.size(); ++index)
    {
        streaming.AddCompletedBar(candles[index]);
        baseline.AddCompletedBar(candles[index]);
        assert(ClassificationSnapshot(
                   streaming.Classification().ClassifiedCandidates()) ==
               ClassificationSnapshot(baseline.ClassifiedCandidates()));

        std::vector<Candle> prefix(candles.begin(),
                                   candles.begin() + index + 1);
        auto historical = CausalTrendLineBreakRetestBehavior::FromHistorical(
            prefix, CalibrationConfiguration(20.0), geometryConfiguration,
            behaviorConfiguration, {"EURUSD", "1h"});
        assert(historical.Observations().size() ==
               streaming.Observations().size());
        for (std::size_t event = 0;
             event < streaming.Observations().size(); ++event)
            assert(EventSnapshot(historical.Observations()[event]) ==
                   EventSnapshot(streaming.Observations()[event]));
    }
    assert(!streaming.Observations().empty());
}

void TestDeterministicBoundsAndLongStreamPerformance()
{
    Configuration configuration = Config(5, 5);
    configuration.maxActiveBreakObservations = 8;
    configuration.maxRetainedBreakObservations = 16;
    TrendLineBehaviorTracker first(configuration);
    TrendLineBehaviorTracker second(configuration);
    const std::vector<ClassifiedTrendLineCandidate> candidate =
        {Candidate(TrendLineDirection::UTL,
                   SteepnessClassification::Inner, 0, 10.0)};
    Initialize(first, candidate, Bar(0, 10.5, 11.0, 10.0, 10.5));
    Initialize(second, candidate, Bar(0, 10.5, 11.0, 10.0, 10.5));

    constexpr std::size_t barCount = 50'000;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 1; index < barCount; ++index)
    {
        const bool broken = index % 2 == 1;
        const Candle candle = broken
            ? Bar(index, 9.5, 9.8, 9.2, 9.5)
            : Bar(index, 10.2, 10.5, 9.8, 10.2);
        first.ObserveCompletedBar(index, candle, candidate);
        second.ObserveCompletedBar(index, candle, candidate);
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    assert(first.Observations().size() <=
           configuration.maxRetainedBreakObservations);
    assert(first.ActiveObservationCount() <=
           configuration.maxActiveBreakObservations);
    const auto firstSummary = first.AggregateSummary();
    const auto secondSummary = second.AggregateSummary();
    assert(firstSummary.breakEvents == (barCount - 1) / 2 + 1);
    assert(firstSummary.breakEvents == secondSummary.breakEvents);
    assert(firstSummary.innerBreaks == secondSummary.innerBreaks);
    assert(firstSummary.allEligibleInnerToOuter.structurallyIneligible ==
           firstSummary.innerBreaks);
    assert(elapsed.count() < 10'000);
    std::cout << "TG2_BOUNDED_STREAM bars=" << barCount
              << ",milliseconds=" << elapsed.count()
              << ",retained=" << first.Observations().size()
              << ",active=" << first.ActiveObservationCount()
              << ",breaks=" << firstSummary.breakEvents << '\n';
}

void TestDiagnosticAndValidation()
{
    TrendLineBehaviorTracker tracker(Config(), {"EURUSD", "1h"});
    const auto candidates = UtlPair();
    Initialize(tracker, candidates, Bar(0, 10.5, 11.0, 9.8, 10.5));
    tracker.ObserveCompletedBar(
        1, Bar(1, 9.7, 9.9, 9.4, 9.5), candidates);
    const std::string diagnostic =
        tracker.FormatDiagnostic(tracker.Observations()[0]);
    assert(diagnostic.find("symbol=EURUSD") != std::string::npos);
    assert(diagnostic.find("break_policy=completed_close_beyond_line") !=
           std::string::npos);
    assert(diagnostic.find("paired_outer=yes") != std::string::npos);
    assert(diagnostic.find("outer_horizon_bars=4") != std::string::npos);

    bool rejectedTolerance = false;
    bool rejectedHorizon = false;
    bool rejectedBounds = false;
    try
    {
        Configuration invalid = Config();
        invalid.retestPriceTolerance = -1.0;
        (void)TrendLineBehaviorTracker(invalid);
    }
    catch (const std::invalid_argument&) { rejectedTolerance = true; }
    try
    {
        Configuration invalid = Config(4, 3);
        (void)TrendLineBehaviorTracker(invalid);
    }
    catch (const std::invalid_argument&) { rejectedHorizon = true; }
    try
    {
        Configuration invalid = Config();
        invalid.maxRetainedBreakObservations = 0;
        (void)TrendLineBehaviorTracker(invalid);
    }
    catch (const std::invalid_argument&) { rejectedBounds = true; }
    assert(rejectedTolerance && rejectedHorizon && rejectedBounds);
}
} // namespace

int main()
{
    TestUtlBreakTimingNoPreBreakAndEpisodeRearm();
    TestDtlBreakTimingAndSymmetry();
    TestExplicitWickBreakPolicy();
    TestRetestStrictlyAfterBreakAndToleranceBoundaries();
    TestDtlRetestAndOuterTargetSymmetry();
    TestRetestAndOuterHorizonExactBoundaries();
    TestCausalPairingAndExactOuterContactBoundary();
    TestOutcomesCensoringUnpairedAndDenominators();
    TestRetestConditionedAggregationAndCausalOrdering();
    TestFutureBarsDoNotRewriteEventIdentity();
    TestCandidateExpiryIsNotBreakAndDoesNotCancelOutcome();
    TestIntegratedHistoricalStreamingPrefixAndTG1IdentityStability();
    TestDeterministicBoundsAndLongStreamPerformance();
    TestDiagnosticAndValidation();
    std::cout << "TG2TrendLineBreakRetestBehaviorTests passed\n";
    return 0;
}
