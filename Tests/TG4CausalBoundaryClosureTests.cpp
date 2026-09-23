#include <algorithm>
#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <vector>

#include "CausalFibonacciConfluenceIntegration.hpp"

namespace
{
using EA::TG1A::Candle;
using EA::TG1A::TrendLineDirection;
using EA::TG1B::SteepnessClassification;
using EA::TG2::BreakObservation;
using EA::TG2::ResolutionState;
using EA::TG3::ConfluenceObservation;
using EA::TG3::ConfluenceState;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kHour = 3'600;

struct Pulse
{
    std::size_t row = 0;
    std::int64_t timestamp = 0;
    std::array<unsigned char, 3> bits{};

    bool operator==(const Pulse&) const = default;
};

std::int64_t Timestamp(std::size_t bar)
{
    return kStart + static_cast<std::int64_t>(bar) * kHour;
}

Candle Bar(std::size_t index, double high, double low, double close)
{
    return {Timestamp(index), close, high, low, close};
}

std::vector<Candle> IntegratedCandles()
{
    const std::vector<double> lows{
        10.0, 9.0, 5.0, 8.0, 10.0, 9.0,
        10.0, 7.0, 10.0, 10.0, 7.0, 7.1};
    const std::vector<double> highs{
        20.0, 20.5, 21.0, 23.0, 25.0, 22.0,
        21.0, 20.5, 21.0, 20.0, 8.1, 8.0};
    std::vector<Candle> result;
    for (std::size_t index = 0; index < lows.size(); ++index)
        result.push_back(Bar(index, highs[index], lows[index],
                             0.5 * (highs[index] + lows[index])));
    return result;
}

EA::TG3::Configuration FibonacciConfiguration()
{
    EA::TG3::Configuration result;
    result.retracementRatios = {0.25, 0.5};
    result.absolutePriceTolerance = 0.0;
    result.maxConfirmedFractalsPerKind = 32;
    result.maxABAgeBars = 100;
    result.maxActiveABStructures = 32;
    result.maxActiveConfluenceObservations = 64;
    result.maxRetainedConfluenceObservations = 64;
    return result;
}

EA::TG1A::Configuration GeometryConfiguration()
{
    EA::TG1A::Configuration result;
    result.atrPeriod = 3;
    result.maxCandidates = 64;
    return result;
}

EA::TG2::Configuration BehaviorConfiguration()
{
    EA::TG2::Configuration result;
    result.retestHorizonBars = 2;
    result.outerTargetHorizonBars = 3;
    result.maxActiveBreakObservations = 64;
    result.maxRetainedBreakObservations = 64;
    return result;
}

const ConfluenceObservation& FindObservation(
    const EA::TG3::CausalFibonacciConfluenceIntegration& integration,
    std::uint64_t sequence)
{
    const auto& observations = integration.Observations();
    const auto found = std::find_if(
        observations.begin(), observations.end(),
        [sequence](const ConfluenceObservation& observation)
        {
            return observation.breakEventSequence == sequence;
        });
    assert(found != observations.end());
    return *found;
}

Pulse PulseForObservation(const ConfluenceObservation& observation)
{
    const bool eligible = observation.confluenceState !=
        ConfluenceState::StructurallyIneligible;
    return {observation.innerBreakBar, observation.innerBreakTimestamp,
            {1, static_cast<unsigned char>(eligible),
             static_cast<unsigned char>(observation.confluenceState ==
                                        ConfluenceState::Confluence)}};
}

std::array<unsigned char, 3> AggregateSameBar(
    std::vector<ConfluenceState> states)
{
    std::array<unsigned char, 3> result{};
    for (const ConfluenceState state : states)
    {
        result[0] |= 1;
        result[1] |= static_cast<unsigned char>(
            state != ConfluenceState::StructurallyIneligible);
        result[2] |= static_cast<unsigned char>(
            state == ConfluenceState::Confluence);
    }
    return result;
}

std::vector<Pulse> Replay(const std::vector<Candle>& candles)
{
    EA::TG3::CausalFibonacciConfluenceIntegration integration(
        EA::TG1B::CalibrationConfiguration(50.0), FibonacciConfiguration(),
        GeometryConfiguration(), BehaviorConfiguration(), {"EURUSD", "1h"});
    std::vector<Pulse> result;
    for (std::size_t index = 0; index < candles.size(); ++index)
    {
        const auto update = integration.AddCompletedBar(candles[index]);
        for (const std::uint64_t sequence : update.newConfluenceObservations)
        {
            const Pulse pulse = PulseForObservation(
                FindObservation(integration, sequence));
            // A Tensor row is appended once, in input order, for this same
            // completed market bar.  This test fixes the adapter's row key.
            assert(pulse.row == index);
            assert(pulse.timestamp == candles[index].timestamp);
            assert(pulse.bits[2] <= pulse.bits[1]);
            assert(pulse.bits[1] <= pulse.bits[0]);
            result.push_back(pulse);
        }
    }
    return result;
}

BreakObservation InnerBreak(std::uint64_t sequence, std::size_t bar)
{
    BreakObservation result;
    result.breakEvent.eventSequence = sequence;
    result.breakEvent.candidate.direction = TrendLineDirection::UTL;
    result.breakEvent.candidate.anchor1Bar = 1;
    result.breakEvent.candidate.anchor2Bar = 4;
    result.breakEvent.candidate.creationBar = 6;
    result.breakEvent.frozenClassification = SteepnessClassification::Inner;
    result.breakEvent.bar = bar;
    result.breakEvent.timestamp = Timestamp(bar);
    result.retest.state = ResolutionState::Pending;
    result.outerTarget.state = ResolutionState::StructurallyIneligible;
    result.outerTargetAfterRetest.state = ResolutionState::StructurallyIneligible;
    return result;
}

void AssertOutcomeEqual(const EA::TG2::OutcomeResolution& actual,
                        const EA::TG2::OutcomeResolution& expected)
{
    assert(actual.state == expected.state);
    assert(actual.resolutionBar == expected.resolutionBar);
    assert(actual.resolutionTimestamp == expected.resolutionTimestamp);
    assert(actual.latencyBars == expected.latencyBars);
    assert(actual.projectedLinePrice == expected.projectedLinePrice);
    assert(actual.censorReason == expected.censorReason);
}

void AssertOutcomeSnapshotsEqual(
    const std::vector<ConfluenceObservation>& actual,
    const std::vector<ConfluenceObservation>& expected)
{
    assert(actual.size() == expected.size());
    for (std::size_t index = 0; index < actual.size(); ++index)
    {
        assert(actual[index].breakEventSequence == expected[index].breakEventSequence);
        AssertOutcomeEqual(actual[index].retest, expected[index].retest);
        AssertOutcomeEqual(actual[index].outerTarget, expected[index].outerTarget);
        AssertOutcomeEqual(actual[index].outerTargetAfterRetest,
                           expected[index].outerTargetAfterRetest);
    }
}

bool Pending(const EA::TG2::OutcomeResolution& outcome)
{
    return outcome.state == ResolutionState::Pending;
}

void ReferenceCensor(EA::TG2::OutcomeResolution& outcome,
                     std::size_t currentBar, std::int64_t currentTimestamp,
                     std::size_t breakBar)
{
    if (!Pending(outcome)) return;
    outcome.state = ResolutionState::Censored;
    outcome.censorReason = EA::TG2::CensorReason::CapacityEviction;
    outcome.resolutionBar = currentBar;
    outcome.resolutionTimestamp = currentTimestamp;
    if (currentBar >= breakBar)
        outcome.latencyBars = currentBar - breakBar;
}

// Reference the pre-optimization full rescan exactly. It is test-local so
// production keeps only the incremental implementation.
std::size_t ReferenceSynchronizeFullRescan(
    std::vector<ConfluenceObservation>& observations,
    const std::vector<BreakObservation>& behaviorObservations,
    std::size_t currentBar, std::int64_t currentTimestamp)
{
    std::size_t comparisons = 0;
    for (ConfluenceObservation& observation : observations)
    {
        const BreakObservation* found = nullptr;
        for (const BreakObservation& candidate : behaviorObservations)
        {
            ++comparisons;
            if (candidate.breakEvent.eventSequence == observation.breakEventSequence)
            {
                found = &candidate;
                break;
            }
        }
        if (found != nullptr)
        {
            observation.retest = found->retest;
            observation.outerTarget = found->outerTarget;
            observation.outerTargetAfterRetest = found->outerTargetAfterRetest;
        }
        else
        {
            ReferenceCensor(observation.retest, currentBar, currentTimestamp,
                            observation.innerBreakBar);
            ReferenceCensor(observation.outerTarget, currentBar, currentTimestamp,
                            observation.innerBreakBar);
            ReferenceCensor(observation.outerTargetAfterRetest, currentBar,
                            currentTimestamp, observation.innerBreakBar);
        }
    }
    return comparisons;
}

void TestOutcomeSynchronizationMatchesFullRescanAndCensoring()
{
    EA::TG3::FibonacciConfluenceTracker tracker(FibonacciConfiguration());
    std::vector<BreakObservation> behavior;

    tracker.Advance(1, Timestamp(1));
    BreakObservation first = InnerBreak(1, 1);
    behavior.push_back(first);
    assert(tracker.ObserveInnerBreak(first).has_value());

    tracker.Advance(2, Timestamp(2));
    behavior.front().retest.state = ResolutionState::Succeeded;
    behavior.front().retest.resolutionBar = 2;
    behavior.front().retest.resolutionTimestamp = Timestamp(2);
    behavior.front().retest.latencyBars = 1;
    std::vector<ConfluenceObservation> expected = tracker.Observations();
    assert(ReferenceSynchronizeFullRescan(expected, behavior, 2, Timestamp(2)) == 1);
    tracker.SynchronizeOutcomes(behavior, 2, Timestamp(2));
    AssertOutcomeSnapshotsEqual(tracker.Observations(), expected);

    // Once terminal, a removed TG2 record cannot alter the immutable snapshot.
    tracker.Advance(3, Timestamp(3));
    behavior.clear();
    expected = tracker.Observations();
    assert(ReferenceSynchronizeFullRescan(expected, behavior, 3, Timestamp(3)) == 0);
    tracker.SynchronizeOutcomes(behavior, 3, Timestamp(3));
    AssertOutcomeSnapshotsEqual(tracker.Observations(), expected);

    // A still-pending record removed by TG2 capacity is censored at this bar.
    tracker.Advance(4, Timestamp(4));
    const BreakObservation pending = InnerBreak(2, 4);
    assert(tracker.ObserveInnerBreak(pending).has_value());
    expected = tracker.Observations();
    assert(ReferenceSynchronizeFullRescan(expected, behavior, 4, Timestamp(4)) == 0);
    tracker.SynchronizeOutcomes(behavior, 4, Timestamp(4));
    AssertOutcomeSnapshotsEqual(tracker.Observations(), expected);
    const auto censored = std::find_if(
        tracker.Observations().begin(), tracker.Observations().end(),
        [](const ConfluenceObservation& observation)
        {
            return observation.breakEventSequence == 2;
        });
    assert(censored != tracker.Observations().end());
    assert(censored->retest.state == ResolutionState::Censored);
    assert(censored->retest.censorReason ==
           EA::TG2::CensorReason::CapacityEviction);
}

void TestOutcomeSynchronizationDoesNotRescanTerminalHistory()
{
    constexpr std::size_t observationCount = 256;
    constexpr std::size_t synchronizationCalls = 32;
    EA::TG3::Configuration configuration = FibonacciConfiguration();
    configuration.maxActiveConfluenceObservations = observationCount;
    configuration.maxRetainedConfluenceObservations = observationCount;
    EA::TG3::FibonacciConfluenceTracker tracker(configuration);
    std::vector<BreakObservation> behavior;
    behavior.reserve(observationCount);

    for (std::size_t index = 0; index < observationCount; ++index)
    {
        const std::size_t bar = index + 1;
        tracker.Advance(bar, Timestamp(bar));
        BreakObservation observation = InnerBreak(index + 1, bar);
        observation.retest.state = ResolutionState::Succeeded;
        observation.retest.resolutionBar = bar;
        observation.retest.resolutionTimestamp = Timestamp(bar);
        observation.retest.latencyBars = 0;
        behavior.push_back(observation);
        assert(tracker.ObserveInnerBreak(observation).has_value());
    }

    for (std::size_t index = 0; index < synchronizationCalls; ++index)
    {
        const std::size_t bar = observationCount + index + 1;
        tracker.SynchronizeOutcomes(behavior, bar, Timestamp(bar));
    }
    const auto& work = tracker.OutcomeSynchronizationWork();
    assert(work.calls == synchronizationCalls);
    assert(work.retainedObservationsExamined ==
           synchronizationCalls * observationCount);
    assert(work.pendingObservationsVisited == 0);
    assert(work.behaviorObservationsCompared == 0);

    // The former implementation performed this deterministic full-rescan
    // work even though every retained snapshot was already terminal.
    const std::size_t formerComparisons = synchronizationCalls *
        observationCount * (observationCount + 1) / 2;
    assert(formerComparisons == 1'052'672);
}

void TestOutcomeSynchronizationUsesLogarithmicPendingLookup()
{
    constexpr std::size_t observationCount = 256;
    EA::TG3::Configuration configuration = FibonacciConfiguration();
    configuration.maxActiveConfluenceObservations = observationCount;
    configuration.maxRetainedConfluenceObservations = observationCount;
    EA::TG3::FibonacciConfluenceTracker tracker(configuration);
    std::vector<BreakObservation> behavior;
    behavior.reserve(observationCount);

    for (std::size_t index = 0; index < observationCount; ++index)
    {
        const std::size_t bar = index + 1;
        tracker.Advance(bar, Timestamp(bar));
        const BreakObservation observation = InnerBreak(index + 1, bar);
        behavior.push_back(observation);
        assert(tracker.ObserveInnerBreak(observation).has_value());
    }

    tracker.SynchronizeOutcomes(behavior, observationCount + 1,
                                Timestamp(observationCount + 1));
    const auto& work = tracker.OutcomeSynchronizationWork();
    assert(work.calls == 1);
    assert(work.retainedObservationsExamined == observationCount);
    assert(work.pendingObservationsVisited == observationCount);
    // Each successful lower-bound lookup needs at most log2(256) probes plus
    // one equality check, rather than the old triangular full-rescan count.
    assert(work.behaviorObservationsCompared <= observationCount * 10);
    const std::size_t formerComparisons = observationCount *
        (observationCount + 1) / 2;
    assert(work.behaviorObservationsCompared < formerComparisons / 10);
    std::cout << "TG3_SYNC_WORK retained=" << work.retainedObservationsExamined
              << ",pending=" << work.pendingObservationsVisited
              << ",tg2_comparisons=" << work.behaviorObservationsCompared
              << ",former_tg2_comparisons=" << formerComparisons << '\n';
}

void TestReplayTimestampAndFutureTailInvariance()
{
    const auto candles = IntegratedCandles();
    const auto first = Replay(candles);
    const auto second = Replay(candles);
    assert(!first.empty());
    assert(first == second);

    const std::size_t cutoff = candles.size() - 1;
    const std::vector<Candle> prefix(candles.begin(), candles.begin() + cutoff);
    const auto prefixPulses = Replay(prefix);
    const auto fullPulses = Replay(candles);
    for (const Pulse& pulse : fullPulses)
        if (pulse.row < cutoff)
            assert(std::find(prefixPulses.begin(), prefixPulses.end(), pulse) !=
                   prefixPulses.end());
}

void TestSameBarOrAggregationIsOrderIndependent()
{
    const std::vector<ConfluenceState> states{
        ConfluenceState::StructurallyIneligible,
        ConfluenceState::NoConfluence,
        ConfluenceState::Confluence};
    const std::array<unsigned char, 3> expected{1, 1, 1};
    assert(AggregateSameBar(states) == expected);
    std::vector<ConfluenceState> reversed = states;
    std::reverse(reversed.begin(), reversed.end());
    assert(AggregateSameBar(reversed) == expected);
    const std::array<unsigned char, 3> ineligibleExpected{1, 0, 0};
    assert(AggregateSameBar({ConfluenceState::StructurallyIneligible}) ==
           ineligibleExpected);
}

void TestOutcomeCapacityCannotRewriteAlreadyEmittedPulse()
{
    EA::TG3::Configuration bounded = FibonacciConfiguration();
    bounded.maxActiveConfluenceObservations = 1;
    bounded.maxRetainedConfluenceObservations = 1;
    EA::TG3::FibonacciConfluenceTracker tracker(bounded);

    tracker.Advance(8, Timestamp(8));
    const auto first = tracker.ObserveInnerBreak(InnerBreak(1, 8));
    assert(first.has_value());
    const Pulse emitted = PulseForObservation(tracker.Observations().front());

    tracker.Advance(9, Timestamp(9));
    const auto second = tracker.ObserveInnerBreak(InnerBreak(2, 9));
    assert(second.has_value());
    assert(tracker.Observations().size() == 1);
    assert(tracker.Observations().front().breakEventSequence == 2);
    // The first record was evicted only after its pulse was emitted.  Its
    // captured row is immutable; capacity only censors future outcomes.
    const Pulse expected{8, Timestamp(8), {1, 0, 0}};
    assert(emitted == expected);
}

} // namespace

int main()
{
    TestReplayTimestampAndFutureTailInvariance();
    TestSameBarOrAggregationIsOrderIndependent();
    TestOutcomeCapacityCannotRewriteAlreadyEmittedPulse();
    TestOutcomeSynchronizationMatchesFullRescanAndCensoring();
    TestOutcomeSynchronizationDoesNotRescanTerminalHistory();
    TestOutcomeSynchronizationUsesLogarithmicPendingLookup();
    std::cout << "TG4CausalBoundaryClosureTests passed\n";
    return 0;
}
