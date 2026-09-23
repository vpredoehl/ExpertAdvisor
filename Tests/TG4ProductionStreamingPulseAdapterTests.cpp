#include "TG4ProductionStreamingPulseAdapter.hpp"

#include <algorithm>
#include <array>
#include <cassert>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <random>
#include <string>
#include <vector>

namespace
{

Feature Bar(std::size_t index, double open, double high, double low, double close)
{
    return {static_cast<float>(open), static_cast<float>(close),
            static_cast<float>(high), static_cast<float>(low),
            PriceTP{std::chrono::seconds{
                1'700'000'100 + static_cast<std::int64_t>(index) * 900}},
            0.0f};
}

std::vector<Feature> DeterministicBars()
{
    std::mt19937_64 generator{0x54473450554c5345ULL};
    std::uniform_int_distribution<int> direction(-6, 6);
    std::uniform_int_distribution<int> wick(1, 8);
    std::vector<Feature> bars;
    bars.reserve(5'000);
    double close = 1.2000;
    for (std::size_t index = 0; index < 5'000; ++index)
    {
        const double open = close;
        close += static_cast<double>(direction(generator)) * 0.0001;
        const double high = std::max(open, close) +
            static_cast<double>(wick(generator)) * 0.0001;
        const double low = std::min(open, close) -
            static_cast<double>(wick(generator)) * 0.0001;
        bars.push_back(Bar(index, open, high, low, close));
    }
    return bars;
}

std::string StablePulseBytes(const std::vector<EA::TG4Pulse::Pulse>& pulses)
{
    std::string result;
    result.reserve(pulses.size() * 11);
    for (const auto& pulse : pulses)
    {
        const auto timestamp = static_cast<std::uint64_t>(
            pulse.barStart.time_since_epoch().count());
        for (unsigned int byte = 0; byte != 8; ++byte)
            result.push_back(static_cast<char>(timestamp >> (byte * 8)));
        for (const std::uint8_t bit : pulse.bits)
            result.push_back(static_cast<char>(bit));
    }
    return result;
}

void TestProductionReplayAndCausalInvariants()
{
    const auto bars = DeterministicBars();
    const auto pulses = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", bars);
    const auto repeated = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", bars);
    assert(pulses == repeated);
    assert(StablePulseBytes(pulses) == StablePulseBytes(repeated));
    assert(pulses.size() == bars.size());
    assert((pulses.front().bits == std::array<std::uint8_t, 3>{0, 0, 0}));

    std::array<std::size_t, 4> counts{};
    for (std::size_t index = 0; index < pulses.size(); ++index)
    {
        const auto& pulse = pulses[index];
        assert(pulse.barStart == bars[index].time);
        assert(pulse.bits[2] <= pulse.bits[1]);
        assert(pulse.bits[1] <= pulse.bits[0]);
        if (pulse.bits == std::array<std::uint8_t, 3>{0, 0, 0}) ++counts[0];
        if (pulse.bits == std::array<std::uint8_t, 3>{1, 0, 0}) ++counts[1];
        if (pulse.bits == std::array<std::uint8_t, 3>{1, 1, 0}) ++counts[2];
        if (pulse.bits == std::array<std::uint8_t, 3>{1, 1, 1}) ++counts[3];
    }
    // This stream drives the real production TG1->TG3 composition and
    // deterministically covers every externally visible pulse state.
    for (const std::size_t count : counts) assert(count != 0);

    const std::size_t prefixLength = 2'000;
    const std::vector<Feature> prefix(bars.begin(),
                                      bars.begin() + prefixLength);
    const auto prefixPulses = EA::TG4Pulse::ReplayCanonicalCompletedBars(
        "eurusdrmp", prefix);
    assert(std::equal(prefixPulses.begin(), prefixPulses.end(), pulses.begin()));

    EA::TG4Pulse::ProductionStreamingAdapter instrumented{"EURUSDRMP"};
    std::vector<EA::TG4Pulse::Pulse> instrumentedPulses;
    instrumentedPulses.reserve(bars.size());
    for (const Feature& bar : bars)
        instrumentedPulses.push_back(instrumented.AddCompletedCanonicalBar(bar));
    assert(instrumentedPulses == pulses);
    const auto& synchronization =
        instrumented.OutcomeSynchronizationWorkForTesting();
    assert(synchronization.calls == bars.size());
    assert(synchronization.pendingObservationsVisited <=
           synchronization.retainedObservationsExamined);
    std::cout << "TG3_PRODUCTION_REPLAY_SYNC calls=" << synchronization.calls
              << ",retained_examined="
              << synchronization.retainedObservationsExamined
              << ",pending_visited="
              << synchronization.pendingObservationsVisited
              << ",tg2_comparisons="
              << synchronization.behaviorObservationsCompared
              << ",behavior_available="
              << synchronization.behaviorObservationsAvailable
              << ",max_retained="
              << synchronization.maxRetainedObservations
              << ",max_pending=" << synchronization.maxPendingObservations
              << ",max_behavior="
              << synchronization.maxBehaviorObservations << '\n';

    EA::TG4Pulse::ProductionStreamingAdapter adapter{"EURUSDRMP"};
    const auto expectedConfiguration =
        EA::ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    assert(adapter.ConfigurationIdentity().canonicalPayload ==
           expectedConfiguration.identity().canonicalPayload);
    assert(adapter.ConfigurationIdentity().hash ==
           expectedConfiguration.identity().hash);
    assert(adapter.Configuration().values().geometry.maxCandidates == 4096);
    assert(adapter.Configuration().values().fibonacci.maxActiveABStructures == 512);
}

void TestSameBarOrAggregation()
{
    using State = EA::TG3::ConfluenceState;
    const std::vector<State> states{
        State::StructurallyIneligible, State::NoConfluence, State::Confluence};
    const std::array<std::uint8_t, 3> expected{1, 1, 1};
    assert(EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(states) ==
           expected);
    std::vector<State> reversed = states;
    std::reverse(reversed.begin(), reversed.end());
    assert(EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(reversed) ==
           expected);
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar(
               {State::StructurallyIneligible}) ==
           std::array<std::uint8_t, 3>{1, 0, 0}));
    assert((EA::TG4Pulse::ProductionStreamingAdapter::AggregateSameBar({}) ==
           std::array<std::uint8_t, 3>{0, 0, 0}));
}

void TestProductionPolicyRejectsDtlConfluence()
{
    const auto configuration =
        EA::ProductionTG1TG3Pulse::Configuration::TG4ADerivedSourceUTLUpABOnlyV1();
    EA::TG3::FibonacciConfluenceTracker tracker(
        configuration.FibonacciConfigurationForSymbol("eurusdrmp"));
    const auto timestamp = [](std::size_t bar)
    {
        return 1'700'000'100 + static_cast<std::int64_t>(bar) * 900;
    };
    for (std::size_t bar = 0; bar <= 4; ++bar) tracker.Advance(bar, timestamp(bar));
    tracker.ObserveConfirmedFractals({
        {EA::TG1A::FractalKind::High, 2, timestamp(2), 1.2200,
         4, timestamp(4)}});
    for (std::size_t bar = 5; bar <= 7; ++bar) tracker.Advance(bar, timestamp(bar));
    tracker.ObserveConfirmedFractals({
        {EA::TG1A::FractalKind::Low, 5, timestamp(5), 1.2000,
         7, timestamp(7)}});
    tracker.Advance(8, timestamp(8));

    EA::TG2::BreakObservation breakObservation;
    breakObservation.breakEvent.eventSequence = 1;
    breakObservation.breakEvent.candidate.direction = EA::TG1A::TrendLineDirection::DTL;
    breakObservation.breakEvent.frozenClassification =
        EA::TG1B::SteepnessClassification::Inner;
    breakObservation.breakEvent.bar = 8;
    breakObservation.breakEvent.timestamp = timestamp(8);
    EA::TG2::PairedOuterLine outer;
    outer.candidate.direction = EA::TG1A::TrendLineDirection::DTL;
    outer.projectedPriceAtBreak = 1.212360677749498;
    breakObservation.pairedOuter = outer;
    assert(tracker.ObserveInnerBreak(breakObservation).has_value());
    const auto& observation = tracker.Observations().back();
    assert(observation.confluenceState ==
           EA::TG3::ConfluenceState::StructurallyIneligible);
    assert(observation.ineligibleReason ==
           EA::TG3::IneligibleReason::DirectionUnsupportedByStudy);
}

} // namespace

int main()
{
    TestProductionReplayAndCausalInvariants();
    TestSameBarOrAggregation();
    TestProductionPolicyRejectsDtlConfluence();
    std::cout << "TG4ProductionStreamingPulseAdapterTests passed\n";
}
