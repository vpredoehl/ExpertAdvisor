#pragma once

#include "CanonicalSymbol.hpp"
#include "MarketStructureRegistry.hpp"
#include "ProductionTG1TG3PulseConfiguration.hpp"
#include "TG4ProductionStreamingPulseAdapter.hpp"

#include <array>
#include <chrono>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// This boundary adapts an already-created production TG4 pulse.  It does not
// own, invoke, or alter the source detector; Tensor's existing raw TG4 path
// remains independent of these descriptive observations.
namespace EA::MarketStructure::Production
{

inline constexpr std::string_view kTG4FamilyId = "tg_structure";
inline constexpr std::string_view kTG4DetectorVersion =
    "tg4-production-pulse-v1";
inline constexpr std::string_view kDescriptorSchemaVersion =
    "tg4-production-pulse-observation-v1";
inline constexpr std::string_view kInnerBreakRole = "inner_break";
inline constexpr std::string_view kStructuralEligibilityRole =
    "structural_eligibility";
inline constexpr std::string_view kFibonacciRetracementRelationRole =
    "fibonacci_retracement_relation";
inline constexpr std::chrono::seconds kCompletedCanonicalBarDuration{900};

inline void ValidateTG4Pulse(const TG4Pulse::Pulse& pulse)
{
    for (const std::uint8_t bit : pulse.bits)
        if (bit > 1)
            throw std::invalid_argument(
                "MARKET_STRUCTURE_TG4_PULSE_BIT_INVALID");
    if (pulse.bits[2] > pulse.bits[1] || pulse.bits[1] > pulse.bits[0])
        throw std::invalid_argument(
            "MARKET_STRUCTURE_TG4_PULSE_HIERARCHY_INVALID");
}

// Converts only explicit production pulse states.  A negative descriptor is
// emitted only for an applicable source state: no eligibility after an inner
// break, or no Fibonacci relation after structural eligibility.  The source
// flags do not define a normalized confidence, so confidence is always absent.
class TG4PulseObservationAdapter final
{
public:
    explicit TG4PulseObservationAdapter(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol))
    {
        const auto configuration =
            ProductionTG1TG3Pulse::Configuration::
                TG4ADerivedSourceUTLUpABOnlyV1();
        sourceProvenance_ = "tg4-production-pulse-observation-bridge-v1;symbol=" +
            symbol_ + ";configuration_hash=" + configuration.identity().hash;
    }

    const std::string& sourceProvenance() const noexcept
    {
        return sourceProvenance_;
    }

    std::vector<Observation> Adapt(const TG4Pulse::Pulse& pulse) const
    {
        ValidateTG4Pulse(pulse);
        if (pulse.bits[0] == 0) return {};

        const std::chrono::sys_seconds completedAt{
            pulse.barStart.time_since_epoch() + kCompletedCanonicalBarDuration};
        std::vector<Observation> observations;
        observations.reserve(3);
        observations.push_back(Make(pulse, completedAt, kInnerBreakRole,
                                    DescriptorPolarity::positive));
        observations.push_back(Make(pulse, completedAt,
            kStructuralEligibilityRole, pulse.bits[1] == 1
                ? DescriptorPolarity::positive : DescriptorPolarity::negative));
        if (pulse.bits[1] == 1)
        {
            observations.push_back(Make(pulse, completedAt,
                kFibonacciRetracementRelationRole, pulse.bits[2] == 1
                    ? DescriptorPolarity::positive : DescriptorPolarity::negative));
        }
        return observations;
    }

private:
    Observation Make(const TG4Pulse::Pulse& pulse,
                     std::chrono::sys_seconds completedAt,
                     std::string_view role,
                     DescriptorPolarity polarity) const
    {
        return {std::string{kTG4FamilyId}, std::string{kTG4DetectorVersion},
                completedAt, completedAt, sourceProvenance_,
                "tg4-pulse;bar_start=" + std::to_string(
                    pulse.barStart.time_since_epoch().count()) + ";role=" +
                    std::string{role},
                {std::string{kDescriptorSchemaVersion}, std::string{role},
                 polarity, std::nullopt}};
    }

    std::string symbol_;
    std::string sourceProvenance_;
};

inline std::array<ConfluenceDefinition, 2> FrozenConfluenceDefinitions()
{
    return {{
        {"tg4-structural-fibonacci-retracement-support", "v1",
         RelationKind::support,
         {std::string{kTG4FamilyId}, std::string{kStructuralEligibilityRole}},
         {std::string{kTG4FamilyId},
          std::string{kFibonacciRetracementRelationRole}},
         1},
        {"tg4-structural-fibonacci-retracement-contradiction", "v1",
         RelationKind::contradiction,
         {std::string{kTG4FamilyId}, std::string{kStructuralEligibilityRole}},
         {std::string{kTG4FamilyId},
          std::string{kFibonacciRetracementRelationRole}},
         1},
    }};
}

struct ProductionConfluenceDescription
{
    std::vector<Observation> sourceObservations;
    std::vector<ConfluenceReplay> replays;
    std::vector<ConfluenceObservation> outputs;
};

// A reusable, non-persistent production bridge.  Its per-pulse input and
// frozen cap keep each evaluation bounded; clients may retain returned
// descriptions only when their own non-model diagnostic workflow requires it.
class TG4ProductionConfluenceBridge final
{
public:
    explicit TG4ProductionConfluenceBridge(std::string symbol)
        : adapter_(std::move(symbol)), definitions_(FrozenConfluenceDefinitions()),
          engines_{DescriptiveConfluenceEngine{definitions_[0]},
                   DescriptiveConfluenceEngine{definitions_[1]}}
    {
    }

    const std::array<ConfluenceDefinition, 2>& definitions() const noexcept
    {
        return definitions_;
    }

    ProductionConfluenceDescription Describe(
        const TG4Pulse::Pulse& pulse,
        std::chrono::sys_seconds decisionTime) const
    {
        ProductionConfluenceDescription result;
        result.sourceObservations = adapter_.Adapt(pulse);
        result.replays.reserve(engines_.size());
        for (const DescriptiveConfluenceEngine& engine : engines_)
        {
            ConfluenceReplay replay = engine.Evaluate(result.sourceObservations,
                                                       decisionTime);
            for (const ConfluenceObservation& output : replay.outputs)
                result.outputs.push_back(output);
            result.replays.push_back(std::move(replay));
        }
        return result;
    }

private:
    TG4PulseObservationAdapter adapter_;
    std::array<ConfluenceDefinition, 2> definitions_;
    std::array<DescriptiveConfluenceEngine, 2> engines_;
};

} // namespace EA::MarketStructure::Production
