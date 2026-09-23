#pragma once

#include "CanonicalSymbol.hpp"
#include "PricePoint.hpp"
#include "ProductionTG1TG3PulseConfiguration.hpp"

#include <array>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

// This adapter is intentionally independent of Tensor/model construction.
// Its input is one completed bar returned by the canonical market-data API;
// its output is a same-bar, three-channel TG4A-derived pulse.
namespace EA::TG4Pulse
{

struct Pulse final
{
    PriceTP barStart;
    std::array<std::uint8_t, 3> bits{};

    bool operator==(const Pulse&) const = default;
};

class ProductionStreamingAdapter final
{
public:
    explicit ProductionStreamingAdapter(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol)),
          configuration_(
              ProductionTG1TG3Pulse::Configuration::
                  TG4ADerivedSourceUTLUpABOnlyV1()),
          integration_(
              TG1B::CalibrationConfiguration(
                  configuration_.values().referenceBarScale),
              configuration_.FibonacciConfigurationForSymbol(symbol_),
              configuration_.values().geometry,
              configuration_.values().behavior,
              {symbol_, configuration_.values().timeframe})
    {
    }

    // `bar.time` is the absolute UTC start of the completed [T,T+15m)
    // canonical bar.  The returned pulse is keyed to that same T.
    Pulse AddCompletedCanonicalBar(const Feature& bar)
    {
        const std::int64_t timestamp = bar.time.time_since_epoch().count();
        const TG3::Update update = integration_.AddCompletedBar(
            {timestamp, static_cast<double>(bar.open),
             static_cast<double>(bar.high), static_cast<double>(bar.low),
             static_cast<double>(bar.close)});

        if (update.newConfluenceObservations.size() !=
            update.newlyCreatedConfluenceObservations.size())
            throw std::logic_error(
                "TG4 pulse update did not retain every new confluence observation");
        std::vector<TG3::ConfluenceState> states;
        states.reserve(update.newlyCreatedConfluenceObservations.size());
        for (const TG3::ConfluenceObservation& observation :
             update.newlyCreatedConfluenceObservations)
        {
            if (observation.innerBreakBar != update.bar ||
                observation.innerBreakTimestamp != timestamp ||
                observation.observationBar != update.bar ||
                observation.observationTimestamp != timestamp)
                throw std::logic_error(
                    "TG4 pulse observation was not created on its completed bar");

            states.push_back(observation.confluenceState);
        }
        return {bar.time, AggregateSameBar(states)};
    }

    static std::array<std::uint8_t, 3> AggregateSameBar(
        const std::vector<TG3::ConfluenceState>& states)
    {
        std::array<std::uint8_t, 3> result{};
        for (const TG3::ConfluenceState state : states)
        {
            result[0] = 1;
            if (state != TG3::ConfluenceState::StructurallyIneligible)
                result[1] = 1;
            if (state == TG3::ConfluenceState::Confluence)
                result[2] = 1;
        }
        if (result[2] > result[1] || result[1] > result[0])
            throw std::logic_error("TG4 pulse hierarchy invariant failed");
        return result;
    }

    const ProductionTG1TG3Pulse::Identity& ConfigurationIdentity() const noexcept
    {
        return configuration_.identity();
    }

    const ProductionTG1TG3Pulse::Configuration& Configuration() const noexcept
    {
        return configuration_;
    }

#if defined(EA_TG3_SYNCHRONIZATION_WORK_INSTRUMENTATION)
    const TG3::SynchronizationWork& OutcomeSynchronizationWorkForTesting()
        const noexcept
    {
        return integration_.Tracker().OutcomeSynchronizationWork();
    }
#endif

private:
    std::string symbol_;
    ProductionTG1TG3Pulse::Configuration configuration_;
    TG3::CausalFibonacciConfluenceIntegration integration_;
};

inline std::vector<Pulse> ReplayCanonicalCompletedBars(
    const std::string& symbol, const std::vector<Feature>& bars)
{
    ProductionStreamingAdapter adapter(symbol);
    std::vector<Pulse> pulses;
    pulses.reserve(bars.size());
    for (const Feature& bar : bars)
        pulses.push_back(adapter.AddCompletedCanonicalBar(bar));
    return pulses;
}

} // namespace EA::TG4Pulse
