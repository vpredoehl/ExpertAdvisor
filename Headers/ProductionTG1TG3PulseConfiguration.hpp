#pragma once

#include "CanonicalMarketDataRange.hpp"
#include "CausalFibonacciConfluenceIntegration.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// Source-owned semantic contract for the future isolated TG1->TG3 pulse
// adapter.  It deliberately has no artifact/file loader and no TG4 reporting
// or outcome-selection inputs.
namespace EA::ProductionTG1TG3Pulse
{
inline constexpr std::string_view kConfigurationSchema =
    "tg1-tg3-causal-pulse-configuration-v1";
inline constexpr std::string_view kConfigurationName =
    "tg4a-derived-source-utl-up-ab-only-v1";

struct Identity final
{
    std::string canonicalPayload;
    std::string hash;
};

struct Input final
{
    std::string configurationSchema;
    std::string name;
    std::string marketRangeContractVersion;
    std::string timeframe;
    int candlePeriod = 0;
    std::string candleUnit;
    std::int64_t expectedIntervalSeconds = 0;
    std::string barIdentity;
    std::string ordering;
    std::string gapPolicy;

    std::string fractalConfirmationSemantics;
    TG1A::Configuration geometry;
    double referenceBarScale = 0.0;
    std::string angleBands;
    std::string classificationSnapshotSemantics;

    TG2::Configuration behavior;
    TG3::Configuration fibonacci;
    std::string fibonacciToleranceConvention;
    double fibonacciTolerancePips = 0.0;
    std::map<std::string, double> canonicalFxPipSizes;
};

inline std::string StableDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("TG pulse configuration has non-finite value");
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << std::hexfloat << value;
    return out.str();
}

inline std::string Fnv1a64(std::string_view text)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char character : text)
    {
        value ^= character;
        value *= 1099511628211ULL;
    }
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
        << value;
    return out.str();
}

inline void Append(std::ostringstream& out, std::string_view key,
                   const std::string& value)
{
    out << key << '=' << value << '\n';
}

inline void Append(std::ostringstream& out, std::string_view key,
                   std::size_t value)
{
    out << key << '=' << value << '\n';
}

inline void Append(std::ostringstream& out, std::string_view key,
                   std::int64_t value)
{
    out << key << '=' << value << '\n';
}

inline void Append(std::ostringstream& out, std::string_view key, int value)
{
    out << key << '=' << value << '\n';
}

inline void AppendDouble(std::ostringstream& out, std::string_view key,
                         double value)
{
    Append(out, key, StableDouble(value));
}

inline std::string BreakPolicyName(TG2::BreakPolicy value)
{
    switch (value)
    {
        case TG2::BreakPolicy::CompletedCloseBeyondLine:
            return "completed_close_beyond_line";
        case TG2::BreakPolicy::CompletedWickBeyondLine:
            return "completed_wick_beyond_line";
    }
    throw std::invalid_argument("unsupported TG2 break policy");
}

inline std::string RearmPolicyName(TG2::RearmPolicy value)
{
    switch (value)
    {
        case TG2::RearmPolicy::CompletedBarReturnsToValidSide:
            return "completed_bar_returns_to_valid_side";
    }
    throw std::invalid_argument("unsupported TG2 rearm policy");
}

inline std::string OuterPairingPolicyName(TG2::OuterPairingPolicy value)
{
    switch (value)
    {
        case TG2::OuterPairingPolicy::NearestCoexistingOuterBeyondBreakCandle:
            return "nearest_coexisting_outer_beyond_break_candle";
    }
    throw std::invalid_argument("unsupported TG2 outer pairing policy");
}

inline std::string AnchorSelectionPolicyName(TG3::AnchorSelectionPolicy value)
{
    switch (value)
    {
        case TG3::AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal:
            return "most_recent_prior_opposite_confirmed_fractal";
    }
    throw std::invalid_argument("unsupported TG3 anchor selection policy");
}

inline std::string DirectionalPolicyName(TG3::DirectionalStudyPolicy value)
{
    switch (value)
    {
        case TG3::DirectionalStudyPolicy::SourceUTLUpABOnly:
            return "source_utl_up_ab_only";
        case TG3::DirectionalStudyPolicy::SymmetricDirectionalDiagnostic:
            return "symmetric_directional_diagnostic";
    }
    throw std::invalid_argument("unsupported TG3 directional policy");
}

inline std::string ConfluencePolicyName(TG3::ConfluencePolicy value)
{
    switch (value)
    {
        case TG3::ConfluencePolicy::AbsolutePriceToleranceAroundExactRetracementLevel:
            return "absolute_price_tolerance_around_exact_retracement_level";
    }
    throw std::invalid_argument("unsupported TG3 confluence policy");
}

inline std::string CanonicalPayload(const Input& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    Append(out, "configuration_schema", value.configurationSchema);
    Append(out, "name", value.name);
    Append(out, "market_range_contract", value.marketRangeContractVersion);
    Append(out, "timeframe", value.timeframe);
    Append(out, "candle_period", value.candlePeriod);
    Append(out, "candle_unit", value.candleUnit);
    Append(out, "expected_interval_seconds", value.expectedIntervalSeconds);
    Append(out, "bar_identity", value.barIdentity);
    Append(out, "ordering", value.ordering);
    Append(out, "gap_policy", value.gapPolicy);
    Append(out, "tg1_fractal_confirmation", value.fractalConfirmationSemantics);
    AppendDouble(out, "tg1_intervening_price_tolerance",
                 value.geometry.interveningPriceTolerance);
    AppendDouble(out, "tg1_touch_price_tolerance",
                 value.geometry.touchPriceTolerance);
    Append(out, "tg1_max_fractal_anchor_lookback_bars",
           value.geometry.maxFractalAnchorLookbackBars);
    Append(out, "tg1_max_confirmed_fractals_per_kind",
           value.geometry.maxConfirmedFractalsPerKind);
    Append(out, "tg1_max_candidate_age_bars", value.geometry.maxCandidateAgeBars);
    Append(out, "tg1_max_candidates", value.geometry.maxCandidates);
    Append(out, "tg1_atr_period", value.geometry.atrPeriod);
    AppendDouble(out, "tg1b_reference_bar_scale", value.referenceBarScale);
    Append(out, "tg1b_angle_bands", value.angleBands);
    Append(out, "tg1b_classification_snapshot",
           value.classificationSnapshotSemantics);
    Append(out, "tg2_break_policy", BreakPolicyName(value.behavior.breakPolicy));
    Append(out, "tg2_rearm_policy", RearmPolicyName(value.behavior.rearmPolicy));
    Append(out, "tg2_outer_pairing_policy",
           OuterPairingPolicyName(value.behavior.outerPairingPolicy));
    AppendDouble(out, "tg2_break_price_tolerance",
                 value.behavior.breakPriceTolerance);
    Append(out, "tg3_anchor_selection",
           AnchorSelectionPolicyName(value.fibonacci.anchorSelectionPolicy));
    std::vector<double> ratios = value.fibonacci.retracementRatios;
    std::sort(ratios.begin(), ratios.end());
    out << "tg3_retracement_ratios=";
    for (std::size_t index = 0; index < ratios.size();
         ++index)
    {
        if (index != 0) out << ',';
        out << StableDouble(ratios[index]);
    }
    out << '\n';
    Append(out, "tg3_directional_policy",
           DirectionalPolicyName(value.fibonacci.directionalStudyPolicy));
    Append(out, "tg3_confluence_policy",
           ConfluencePolicyName(value.fibonacci.confluencePolicy));
    Append(out, "tg3_tolerance_convention", value.fibonacciToleranceConvention);
    AppendDouble(out, "tg3_tolerance_pips", value.fibonacciTolerancePips);
    for (const auto& [symbol, pipSize] : value.canonicalFxPipSizes)
        AppendDouble(out, "tg3_pip_size." + symbol, pipSize);
    Append(out, "tg3_max_confirmed_fractals_per_kind",
           value.fibonacci.maxConfirmedFractalsPerKind);
    Append(out, "tg3_max_ab_age_bars", value.fibonacci.maxABAgeBars);
    Append(out, "tg3_max_active_ab_structures",
           value.fibonacci.maxActiveABStructures);
    return out.str();
}

// The outcome tracking/retention fields below are intentionally excluded from
// CanonicalPayload: they cannot alter a current bar's three immediate pulse
// bits after TG2/TG3 creation.  Keep an independently auditable payload so a
// future adapter cannot silently confuse operational limits with semantics.
inline std::string OperationalRetentionPayload(const Input& value)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    switch (value.behavior.retestContactPolicy)
    {
        case TG2::RetestContactPolicy::WickReachesProjectedLineFromBrokenSide:
            Append(out, "tg2_retest_contact_policy",
                   "wick_reaches_projected_line_from_broken_side");
            break;
    }
    AppendDouble(out, "tg2_retest_price_tolerance",
                 value.behavior.retestPriceTolerance);
    AppendDouble(out, "tg2_outer_target_price_tolerance",
                 value.behavior.outerTargetPriceTolerance);
    Append(out, "tg2_retest_horizon_bars", value.behavior.retestHorizonBars);
    Append(out, "tg2_outer_target_horizon_bars",
           value.behavior.outerTargetHorizonBars);
    Append(out, "tg2_max_active_break_observations",
           value.behavior.maxActiveBreakObservations);
    Append(out, "tg2_max_retained_break_observations",
           value.behavior.maxRetainedBreakObservations);
    Append(out, "tg3_max_active_confluence_observations",
           value.fibonacci.maxActiveConfluenceObservations);
    Append(out, "tg3_max_retained_confluence_observations",
           value.fibonacci.maxRetainedConfluenceObservations);
    return out.str();
}

inline void Validate(const Input& value)
{
    if (value.configurationSchema != kConfigurationSchema ||
        value.name != kConfigurationName ||
        value.marketRangeContractVersion.empty() ||
        value.timeframe != "15m" ||
        value.candlePeriod != CanonicalMarketData::kCanonicalCandlePeriodMinutes ||
        value.candleUnit != "minute" ||
        value.expectedIntervalSeconds != CanonicalMarketData::kCanonicalIntervalSeconds ||
        value.fibonacciToleranceConvention != "canonical_fx_pips" ||
        !std::isfinite(value.fibonacciTolerancePips) ||
        value.fibonacciTolerancePips < 0.0 || value.canonicalFxPipSizes.empty())
        throw std::invalid_argument("invalid production TG1-TG3 pulse configuration");
    (void)TG1B::CalibrationConfiguration(value.referenceBarScale);
    (void)TG1A::CausalFractalTrendLineGeometry(value.geometry);
    (void)TG2::TrendLineBehaviorTracker(value.behavior);
    TG3::Configuration fibonacci = value.fibonacci;
    fibonacci.absolutePriceTolerance = 0.0;
    (void)TG3::FibonacciConfluenceTracker(std::move(fibonacci));
    for (const auto& [symbol, pipSize] : value.canonicalFxPipSizes)
    {
        if (symbol.empty() || !std::isfinite(pipSize) || pipSize <= 0.0)
            throw std::invalid_argument("invalid canonical FX pip map");
    }
}

inline Input TG4ADerivedSourceUTLUpABOnlyV1Input()
{
    Input value;
    value.configurationSchema = std::string{kConfigurationSchema};
    value.name = std::string{kConfigurationName};
    value.marketRangeContractVersion =
        CanonicalMarketData::kAbsoluteHalfOpenContractVersion;
    value.timeframe = "15m";
    value.candlePeriod = CanonicalMarketData::kCanonicalCandlePeriodMinutes;
    value.candleUnit = "minute";
    value.expectedIntervalSeconds = CanonicalMarketData::kCanonicalIntervalSeconds;
    value.barIdentity = "new_york_civil_bar_start_converted_to_utc_pricetp";
    value.ordering = "strictly_ascending_bar_start";
    value.gapPolicy = "retain_available_bars_no_synthetic_fill";
    value.fractalConfirmationSemantics =
        "strict_five_completed_candles_radius_2";
    value.geometry = {0.0, 0.0, 512, 64, 512, 4096, 14};
    value.referenceBarScale = 14.0;
    value.angleBands =
        "long_term_12_20_outer_25_40_inner_45_85_degrees_inclusive";
    value.classificationSnapshotSemantics =
        "creation_time_atr_normalized_slope";
    value.behavior = {};
    value.fibonacci.retracementRatios = {0.6180339887498949};
    value.fibonacci.anchorSelectionPolicy =
        TG3::AnchorSelectionPolicy::MostRecentPriorOppositeConfirmedFractal;
    value.fibonacci.directionalStudyPolicy =
        TG3::DirectionalStudyPolicy::SourceUTLUpABOnly;
    value.fibonacci.confluencePolicy =
        TG3::ConfluencePolicy::AbsolutePriceToleranceAroundExactRetracementLevel;
    value.fibonacci.maxConfirmedFractalsPerKind = 128;
    value.fibonacci.maxABAgeBars = 2048;
    value.fibonacci.maxActiveABStructures = 512;
    value.fibonacci.maxActiveConfluenceObservations = 4096;
    value.fibonacci.maxRetainedConfluenceObservations = 4096;
    value.fibonacciToleranceConvention = "canonical_fx_pips";
    value.fibonacciTolerancePips = 1.0;
    value.canonicalFxPipSizes = {{"audcadrmp", 0.0001}, {"audusdrmp", 0.0001},
                                 {"eurusdrmp", 0.0001}, {"gbpusdrmp", 0.0001},
                                 {"usdcadrmp", 0.0001}, {"usdjpyrmp", 0.01}};
    Validate(value);
    return value;
}

class Configuration final
{
public:
    explicit Configuration(Input value) : value_(std::move(value))
    {
        Validate(value_);
        identity_ = {CanonicalPayload(value_), Fnv1a64(CanonicalPayload(value_))};
    }

    static Configuration TG4ADerivedSourceUTLUpABOnlyV1()
    {
        return Configuration{TG4ADerivedSourceUTLUpABOnlyV1Input()};
    }

    const Input& values() const noexcept { return value_; }
    const Identity& identity() const noexcept { return identity_; }

    TG3::Configuration FibonacciConfigurationForSymbol(
        const std::string& symbol) const
    {
        const auto found = value_.canonicalFxPipSizes.find(symbol);
        if (found == value_.canonicalFxPipSizes.end())
            throw std::invalid_argument("unsupported canonical FX pip symbol: " + symbol);
        TG3::Configuration result = value_.fibonacci;
        result.absolutePriceTolerance =
            value_.fibonacciTolerancePips * found->second;
        return result;
    }

private:
    Input value_;
    Identity identity_;
};
} // namespace EA::ProductionTG1TG3Pulse
