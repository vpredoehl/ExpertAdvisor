#pragma once

#include "CausalFibonacciExtensionResearch.hpp"
#include "CausalFibonacciStructuralFeatureConfiguration.hpp"
#include "CausalFibonacciStructuralPrimitives.hpp"
#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <optional>
#include <string>
#include <vector>

namespace EA::CausalFibonacciFeatures {

inline constexpr std::size_t kFeatureCount = 23;
inline constexpr std::size_t kHorizonBars = 20;

enum Column : std::size_t {
    RecentPriceScaleValid = 0,
    UpRecentUnionCountLog,
    UpRecentH1CountLog,
    UpRecentH2CountLog,
    UpRecentBothCountLog,
    UpRecentH1YoungestAge20,
    UpRecentH2YoungestAge20,
    UpRecentMedian1272,
    UpRecentMedian1618,
    UpRecentMedianPullback0382,
    UpRecentMedianPullback0500,
    UpRecentMedianPullback0618,
    DownRecentUnionCountLog,
    DownRecentH1CountLog,
    DownRecentH2CountLog,
    DownRecentBothCountLog,
    DownRecentH1YoungestAge20,
    DownRecentH2YoungestAge20,
    DownRecentMedian1272,
    DownRecentMedian1618,
    DownRecentMedianPullback0382,
    DownRecentMedianPullback0500,
    DownRecentMedianPullback0618,
};

static_assert(DownRecentMedianPullback0618 + 1 == kFeatureCount);

struct StructureState {
    bool upAB = true;
    std::array<double, 5> levels{};
    CausalFibonacciStructural::H1H2EventState events;
};

inline double ExactNearestRankSignedMedian(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    // Nearest-rank p50: one-based rank = ceil(0.5 * N).  This is the lower
    // middle for even N; it is not an arithmetic midpoint.
    return values[(values.size() - 1) / 2];
}

inline bool IsValidPriceScale(double close, const std::optional<double>& atr,
                              double pip)
{
    if (!std::isfinite(close) || !std::isfinite(pip) || pip <= 0.0)
        return false;
    const double denominator = atr && std::isfinite(*atr) && *atr > pip ? *atr : pip;
    return std::isfinite(denominator) && denominator > 0.0;
}

inline std::array<float, kFeatureCount> Aggregate(
    const std::vector<StructureState>& structures, std::size_t currentBar,
    double close, const std::optional<double>& atr, double canonicalPipSize)
{
    std::array<float, kFeatureCount> out{};
    const bool validScale = IsValidPriceScale(close, atr, canonicalPipSize);
    out[RecentPriceScaleValid] = validScale ? 1.0f : 0.0f;
    for (std::size_t direction = 0; direction != 2; ++direction) {
        const bool up = direction == 0;
        const std::size_t base = up ? UpRecentUnionCountLog : DownRecentUnionCountLog;
        std::size_t unionCount = 0, h1Count = 0, h2Count = 0, bothCount = 0;
        std::optional<std::size_t> youngestH1, youngestH2;
        std::array<std::vector<double>, 5> distances;
        for (const auto& structure : structures) {
            if (structure.upAB != up) continue;
            const auto relevance = CausalFibonacciStructural::ClassifyEventRelevance(
                structure.events.h1BeyondBar, structure.events.h2RejectionBar,
                currentBar, kHorizonBars);
            if (relevance == CausalFibonacciStructural::EventRelevance::None) continue;
            for (const double level : structure.levels)
                if (!std::isfinite(level))
                    throw std::runtime_error(
                        "CAUSAL_FIBONACCI_REQUIRED_LEVEL_INVALID");
            ++unionCount;
            const bool hasH1 = relevance == CausalFibonacciStructural::EventRelevance::H1Only ||
                relevance == CausalFibonacciStructural::EventRelevance::Both;
            const bool hasH2 = relevance == CausalFibonacciStructural::EventRelevance::H2Only ||
                relevance == CausalFibonacciStructural::EventRelevance::Both;
            if (hasH1) {
                ++h1Count;
                const auto age = CausalFibonacciStructural::EventAgeAtBar(
                    *structure.events.h1BeyondBar, currentBar);
                youngestH1 = youngestH1 ? std::min(*youngestH1, age) : age;
            }
            if (hasH2) {
                ++h2Count;
                const auto age = CausalFibonacciStructural::EventAgeAtBar(
                    *structure.events.h2RejectionBar, currentBar);
                youngestH2 = youngestH2 ? std::min(*youngestH2, age) : age;
            }
            if (hasH1 && hasH2) ++bothCount;
            if (validScale)
                for (std::size_t level = 0; level != distances.size(); ++level) {
                    const auto normalized = CausalFibonacciStructural::NormalizeDistance(
                        up, structure.levels[level], close, atr, canonicalPipSize);
                    if (!normalized)
                        throw std::runtime_error(
                            "CAUSAL_FIBONACCI_REQUIRED_LEVEL_INVALID");
                    distances[level].push_back(normalized->value);
                }
        }
        out[base] = static_cast<float>(std::log1p(static_cast<double>(unionCount)));
        out[base + 1] = static_cast<float>(std::log1p(static_cast<double>(h1Count)));
        out[base + 2] = static_cast<float>(std::log1p(static_cast<double>(h2Count)));
        out[base + 3] = static_cast<float>(std::log1p(static_cast<double>(bothCount)));
        out[base + 4] = youngestH1 ? static_cast<float>(*youngestH1) / kHorizonBars : 0.0f;
        out[base + 5] = youngestH2 ? static_cast<float>(*youngestH2) / kHorizonBars : 0.0f;
        for (std::size_t level = 0; level != distances.size(); ++level)
            out[base + 6 + level] = static_cast<float>(
                ExactNearestRankSignedMedian(std::move(distances[level])));
    }
    return out;
}

class Producer final {
public:
    explicit Producer(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol)),
          geometry_(configuration_.geometry(), {symbol_, std::string{configuration_.timeframe()}}),
          tracker_(configuration_.FibonacciConfigurationForSymbol(symbol_),
                   {symbol_, std::string{configuration_.timeframe()}}) {}

    std::array<float, kFeatureCount> AddCompletedBar(const TG1A::Candle& candle)
    {
        const auto update = geometry_.AddCompletedBar(candle);
        tracker_.Advance(update.bar, candle.timestamp);
        tracker_.ObserveConfirmedFractals(update.newlyConfirmedFractals);
        std::map<std::string, bool> live;
        for (const auto& ab : tracker_.ABStructures()) {
            const auto key = Key(ab.identity);
            live[key] = true;
            if (!states_.contains(key)) states_.emplace(key, MakeState(ab));
        }
        for (auto it = states_.begin(); it != states_.end(); )
            if (!live.contains(it->first)) it = states_.erase(it); else ++it;
        for (auto& [key, state] : states_) {
            const auto& far = state.levels[0];
            CausalFibonacciStructural::AdvanceH1H2EventState(
                state.events, update.bar,
                state.upAB ? candle.high >= far.zoneLowerPrice : candle.low <= far.zoneUpperPrice,
                state.upAB ? candle.close > far.zoneUpperPrice : candle.close < far.zoneLowerPrice,
                state.upAB ? candle.close < far.zoneLowerPrice : candle.close > far.zoneUpperPrice);
        }
        std::vector<StructureState> current;
        current.reserve(states_.size());
        for (const auto& [key, state] : states_)
            current.push_back({state.upAB, {state.levels[0].price, state.levels[1].price,
                state.levels[2].price, state.levels[3].price, state.levels[4].price}, state.events});
        return Aggregate(current, update.bar, candle.close, geometry_.CurrentAtr(),
                         CanonicalPipSize());
    }

private:
    struct RuntimeState {
        bool upAB = true;
        std::array<FibonacciResearch::FibonacciLevel, 5> levels;
        CausalFibonacciStructural::H1H2EventState events;
    };
    static std::string Key(const TG3::ABIdentity& identity) {
        return std::to_string(identity.availabilityBar) + ':' + std::to_string(identity.bBar) + ':' +
            std::to_string(static_cast<int>(identity.direction)) + ':' +
            std::to_string(identity.aTimestamp) + ':' + std::to_string(identity.bTimestamp);
    }
    RuntimeState MakeState(const TG3::ABStructure& ab) const {
        const double tolerance =
            configuration_.FibonacciConfigurationForSymbol(symbol_).absolutePriceTolerance;
        return {ab.identity.direction == TG3::ABDirection::UpAB,
                {FibonacciResearch::CalculateExtensionLevel(ab, 1.272, tolerance),
                 FibonacciResearch::CalculateExtensionLevel(ab, 1.618, tolerance),
                 FibonacciResearch::CalculatePullbackLevel(ab, .382, tolerance),
                 FibonacciResearch::CalculatePullbackLevel(ab, .500, tolerance),
                 FibonacciResearch::CalculatePullbackLevel(ab, .618, tolerance)}, {}};
    }
    double CanonicalPipSize() const {
        return configuration_.CanonicalPipSize(symbol_);
    }
    std::string symbol_;
    CausalFibonacciStructuralFeatureConfiguration::Configuration configuration_;
    TG1A::CausalFractalTrendLineGeometry geometry_;
    TG3::FibonacciConfluenceTracker tracker_;
    std::map<std::string, RuntimeState> states_;
};

} // namespace EA::CausalFibonacciFeatures
