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
    std::cout << "TG4CausalBoundaryClosureTests passed\n";
    return 0;
}
