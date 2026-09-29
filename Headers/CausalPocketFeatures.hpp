#pragma once

// Frozen Phase Pocket 5A causal representation.  This producer consumes the
// immutable Phase Pocket 2 detector stream; it is not an active-Pocket,
// execution, outcome, touch/fill, or lifecycle model.

#include "CanonicalSymbol.hpp"
#include "CausalFibonacciStructuralFeatureConfiguration.hpp"
#include "CausalFractalTrendLineGeometry.hpp"
#include "CausalPocketDetector.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <deque>
#include <optional>
#include <set>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace EA::CausalPocketFeatures
{

inline constexpr std::size_t kFeatureCount = 11;
inline constexpr std::size_t kRecentObservationHorizonBars = 20;
inline constexpr std::string_view kDetectorSourceTimeframe = "15m_completed";

enum Column : std::size_t
{
    RecentPriceScaleValid = 0,
    BullRecentCountLog,
    BullYoungestAge20,
    BullMedianTouchDistance,
    BullMedianCloseDistance,
    BullMedianWidth,
    BearRecentCountLog,
    BearYoungestAge20,
    BearMedianTouchDistance,
    BearMedianCloseDistance,
    BearMedianWidth,
};

static_assert(BearMedianWidth + 1 == kFeatureCount);

inline double ExactNearestRankSignedMedian(std::vector<double> values)
{
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    // One-based nearest-rank p50 = ceil(0.5*N), the lower middle for even N.
    return values[(values.size() - 1) / 2];
}

inline bool IsValidPriceScale(double close, const std::optional<double>& atr,
                              double canonicalPipSize)
{
    if (!std::isfinite(close) || !std::isfinite(canonicalPipSize) ||
        canonicalPipSize <= 0.0)
    {
        return false;
    }
    const double denominator = atr && std::isfinite(*atr) &&
            *atr > canonicalPipSize
        ? *atr
        : canonicalPipSize;
    return std::isfinite(denominator) && denominator > 0.0;
}

inline std::string ObservationIdentity(const Pocket::PocketObservation& value,
                                       const std::string& symbol)
{
    return CanonicalSymbol::Normalize(symbol) + ':' + value.sourceTimeframe + ':' +
        std::to_string(value.eventBar) + ':' +
        std::to_string(value.eventTimestamp) + ':' +
        std::to_string(value.informationCutoffBar) + ':' +
        std::to_string(value.informationCutoffTimestamp) + ':' +
        std::to_string(static_cast<int>(value.direction));
}

inline void ValidateObservation(const Pocket::PocketObservation& value)
{
    if (value.sourceTimeframe.empty() ||
        value.informationCutoffBar != value.confirmationBar ||
        value.informationCutoffTimestamp != value.confirmationTimestamp ||
        value.eventBar >= value.informationCutoffBar ||
        !std::isfinite(value.range.lower) || !std::isfinite(value.range.upper) ||
        !(value.range.upper > value.range.lower))
    {
        throw std::invalid_argument("CAUSAL_POCKET_OBSERVATION_INVALID");
    }
}

inline std::array<float, kFeatureCount> Aggregate(
    std::span<const Pocket::PocketObservation> observations,
    std::size_t currentBar, double currentClose,
    const std::optional<double>& currentAtr, double canonicalPipSize)
{
    struct DirectionValues
    {
        std::size_t count = 0;
        std::optional<std::size_t> youngestAge;
        std::vector<double> touchDistances;
        std::vector<double> closeDistances;
        std::vector<double> widths;
    };

    std::array<float, kFeatureCount> out{};
    const bool validScale = IsValidPriceScale(currentClose, currentAtr,
                                               canonicalPipSize);
    out[RecentPriceScaleValid] = validScale ? 1.0F : 0.0F;
    const double denominator = currentAtr && std::isfinite(*currentAtr) &&
            *currentAtr > canonicalPipSize
        ? *currentAtr
        : canonicalPipSize;

    DirectionValues bullish;
    DirectionValues bearish;
    std::set<std::string> identities;
    for (const Pocket::PocketObservation& observation : observations)
    {
        ValidateObservation(observation);
        const std::string identity = observation.sourceTimeframe + ':' +
            std::to_string(observation.eventBar) + ':' +
            std::to_string(observation.eventTimestamp) + ':' +
            std::to_string(observation.informationCutoffBar) + ':' +
            std::to_string(observation.informationCutoffTimestamp) + ':' +
            std::to_string(static_cast<int>(observation.direction));
        if (!identities.insert(identity).second)
            throw std::invalid_argument("CAUSAL_POCKET_OBSERVATION_DUPLICATE");
        if (observation.informationCutoffBar > currentBar)
            throw std::invalid_argument("CAUSAL_POCKET_OBSERVATION_FROM_FUTURE");

        const std::size_t age = currentBar - observation.informationCutoffBar;
        if (age > kRecentObservationHorizonBars) continue;
        DirectionValues& values = observation.direction == Pocket::PocketDirection::Bullish
            ? bullish : bearish;
        ++values.count;
        values.youngestAge = values.youngestAge
            ? std::min(*values.youngestAge, age) : age;
        if (!validScale) continue;

        const double orientation = observation.direction == Pocket::PocketDirection::Bullish
            ? 1.0 : -1.0;
        const double touchDistance = orientation *
            (observation.TouchPrice() - currentClose) / denominator;
        const double closeDistance = orientation *
            (observation.ClosePrice() - currentClose) / denominator;
        const double width =
            (observation.range.upper - observation.range.lower) / denominator;
        if (!std::isfinite(touchDistance) || !std::isfinite(closeDistance) ||
            !std::isfinite(width) || width < 0.0)
        {
            throw std::runtime_error("CAUSAL_POCKET_NORMALIZATION_INVALID");
        }
        values.touchDistances.push_back(touchDistance);
        values.closeDistances.push_back(closeDistance);
        values.widths.push_back(width);
    }

    const auto write = [&out](const DirectionValues& values, std::size_t base)
    {
        out[base] = static_cast<float>(std::log1p(static_cast<double>(values.count)));
        out[base + 1] = values.youngestAge
            ? static_cast<float>(*values.youngestAge) /
                static_cast<float>(kRecentObservationHorizonBars)
            : 0.0F;
        out[base + 2] = static_cast<float>(
            ExactNearestRankSignedMedian(values.touchDistances));
        out[base + 3] = static_cast<float>(
            ExactNearestRankSignedMedian(values.closeDistances));
        out[base + 4] = static_cast<float>(
            ExactNearestRankSignedMedian(values.widths));
        for (std::size_t index = base; index != base + 5; ++index)
            if (!std::isfinite(out[index]))
                throw std::runtime_error("CAUSAL_POCKET_FEATURE_NONFINITE");
    };
    write(bullish, BullRecentCountLog);
    write(bearish, BearRecentCountLog);
    return out;
}

inline std::array<float, kFeatureCount> Aggregate(
    const std::vector<Pocket::PocketObservation>& observations,
    std::size_t currentBar, double currentClose,
    const std::optional<double>& currentAtr, double canonicalPipSize)
{
    return Aggregate(std::span<const Pocket::PocketObservation>{observations},
                     currentBar, currentClose, currentAtr, canonicalPipSize);
}

class Producer final
{
public:
    explicit Producer(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol)),
          geometry_(configuration_.geometry(),
                    {symbol_, std::string{configuration_.timeframe()}}),
          detector_(std::string{kDetectorSourceTimeframe},
                    Pocket::CausalPocketDetector::kSourceDefaultLookbackBars)
    {
    }

    std::array<float, kFeatureCount> AddCompletedBar(const TG1A::Candle& candle)
    {
        // The frozen normalization uses the existing causal geometry ATR after
        // accepting this completed bar, never Tensor's unrelated legacy ATR.
        const TG1A::Update update = geometry_.AddCompletedBar(candle);
        const auto emitted = detector_.AddCompletedBar({
            candle.timestamp, candle.open, candle.high, candle.low, candle.close});
        if (detector_.CompletedBarCount() == 0 ||
            update.bar != detector_.CompletedBarCount() - 1)
        {
            throw std::logic_error("CAUSAL_POCKET_BAR_COORDINATE_MISMATCH");
        }
        if (emitted)
        {
            ValidateObservation(*emitted);
            if (emitted->sourceTimeframe != kDetectorSourceTimeframe)
                throw std::logic_error("CAUSAL_POCKET_TIMEFRAME_MISMATCH");
            const std::string identity = ObservationIdentity(*emitted, symbol_);
            if (!emittedIdentities_.insert(identity).second)
                throw std::logic_error("CAUSAL_POCKET_OBSERVATION_DUPLICATE");
            retained_.push_back(*emitted);
        }
        while (!retained_.empty())
        {
            const std::size_t cutoff = retained_.front().informationCutoffBar;
            if (cutoff > update.bar)
                throw std::logic_error("CAUSAL_POCKET_RETAINED_OBSERVATION_FROM_FUTURE");
            if (update.bar - cutoff <= kRecentObservationHorizonBars) break;
            retained_.pop_front();
        }
        const std::vector<Pocket::PocketObservation> current(
            retained_.begin(), retained_.end());
        return Aggregate(current, update.bar, candle.close,
                         geometry_.CurrentAtr(),
                         configuration_.CanonicalPipSize(symbol_));
    }

private:
    std::string symbol_;
    CausalFibonacciStructuralFeatureConfiguration::Configuration configuration_;
    TG1A::CausalFractalTrendLineGeometry geometry_;
    Pocket::CausalPocketDetector detector_;
    std::deque<Pocket::PocketObservation> retained_;
    std::set<std::string> emittedIdentities_;
};

} // namespace EA::CausalPocketFeatures
