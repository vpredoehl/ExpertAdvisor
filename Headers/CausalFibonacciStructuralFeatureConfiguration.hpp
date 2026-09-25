#pragma once

#include "CanonicalSymbol.hpp"
#include "ProductionTG1TG3PulseConfiguration.hpp"

#include <string>
#include <string_view>

// Frozen layout-9 source contract.  This deliberately does not derive from
// the mutable production UpAB-only TG4 pulse configuration.
namespace EA::CausalFibonacciStructuralFeatureConfiguration {

inline constexpr std::string_view kName =
    "causal-fibonacci-layout9-symmetric-structural-v1";
inline constexpr std::string_view kTimeframe = "15m";
inline constexpr std::size_t kMaxABAgeBars = 2048;
inline constexpr std::size_t kMaxActiveABStructures = 512;
inline constexpr double kRetracementRatio = 0.6180339887498949;
inline constexpr double kTolerancePips = 1.0;

class Configuration final {
public:
    const TG1A::Configuration& geometry() const noexcept { return geometry_; }
    std::string_view timeframe() const noexcept { return kTimeframe; }

    TG3::Configuration FibonacciConfigurationForSymbol(
        const std::string& rawSymbol) const
    {
        const std::string symbol = CanonicalSymbol::Normalize(rawSymbol);
        const auto& pips = ProductionTG1TG3Pulse::ProductionCanonicalFxPipSizes();
        const auto found = pips.find(symbol);
        if (found == pips.end())
            throw std::invalid_argument("unsupported canonical FX pip symbol: " + symbol);
        TG3::Configuration result;
        result.retracementRatios = {kRetracementRatio};
        result.absolutePriceTolerance = kTolerancePips * found->second;
        result.anchorSelectionPolicy =
            TG3::AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal;
        result.directionalStudyPolicy =
            TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
        result.confluencePolicy =
            TG3::ConfluencePolicy::AbsolutePriceToleranceAroundExactRetracementLevel;
        result.maxConfirmedFractalsPerKind = 128;
        result.maxABAgeBars = kMaxABAgeBars;
        result.maxActiveABStructures = kMaxActiveABStructures;
        result.maxActiveConfluenceObservations = 4096;
        result.maxRetainedConfluenceObservations = 4096;
        return result;
    }

    double CanonicalPipSize(const std::string& rawSymbol) const
    {
        const std::string symbol = CanonicalSymbol::Normalize(rawSymbol);
        return ProductionTG1TG3Pulse::ProductionCanonicalFxPipSizes().at(symbol);
    }

private:
    const TG1A::Configuration geometry_{0.0, 0.0, 512, 64, 512, 4096, 14};
};

} // namespace EA::CausalFibonacciStructuralFeatureConfiguration
