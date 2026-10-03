#pragma once

// Research-only Price-Level Phase 5A helpers.  This header does not alter the
// causal-price-level/v1 detector or expose any model feature.  It measures the
// v1 strict-pivot rule and causal, preceding-range scale candidates.

#include "CausalPriceLevelEngine.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <map>
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace EA::PriceLevel::Characterization
{
inline constexpr const char* kStudyContract =
    "price-level-phase5a-characterization-v1";

inline void ValidateCompletedBar(const CompletedBar& bar)
{
    if (!std::isfinite(bar.open) || !std::isfinite(bar.high) ||
        !std::isfinite(bar.low) || !std::isfinite(bar.close) ||
        bar.high < bar.low || bar.open < bar.low || bar.open > bar.high ||
        bar.close < bar.low || bar.close > bar.high)
    {
        throw std::invalid_argument("PRICE_LEVEL_CHARACTERIZATION_BAR_INVALID");
    }
}

struct PivotCounts final
{
    std::uint64_t highs = 0;
    std::uint64_t lows = 0;

    std::uint64_t total() const noexcept { return highs + lows; }
    bool operator==(const PivotCounts&) const = default;
};

// The sample is taken before AddCompletedRange consumes the current bar.  It
// consequently contains ranges for exactly the preceding completed bars and
// cannot include the pivot bar or any later confirmation bar.
class PrecedingRangeScale final
{
public:
    explicit PrecedingRangeScale(std::size_t lookback)
        : lookback_(lookback)
    {
        if (lookback_ == 0)
            throw std::invalid_argument("PRICE_LEVEL_SCALE_LOOKBACK_INVALID");
    }

    struct Sample final
    {
        std::size_t count = 0;
        double mean = 0.0;
        double median = 0.0;
    };

    Sample BeforeCurrentBar() const
    {
        if (ranges_.empty()) return {};
        const auto middle = std::next(ordered_.begin(),
            static_cast<std::ptrdiff_t>((ordered_.size() - 1) / 2));
        double median = *middle;
        if (ordered_.size() % 2 == 0)
            median = (median + *std::next(middle)) / 2.0;
        return {ranges_.size(), sum_ / static_cast<double>(ranges_.size()), median};
    }

    void AddCompletedRange(double value)
    {
        if (!std::isfinite(value) || value < 0.0)
            throw std::invalid_argument("PRICE_LEVEL_SCALE_RANGE_INVALID");
        ranges_.push_back(value);
        ordered_.insert(value);
        sum_ += value;
        if (ranges_.size() > lookback_)
        {
            const double removed = ranges_.front();
            ranges_.pop_front();
            const auto found = ordered_.find(removed);
            if (found == ordered_.end())
                throw std::logic_error("PRICE_LEVEL_SCALE_INTERNAL_ORDERING");
            ordered_.erase(found);
            sum_ -= removed;
        }
    }

private:
    std::size_t lookback_;
    std::deque<double> ranges_;
    std::multiset<double> ordered_;
    double sum_ = 0.0;
};

struct PivotScaleSample final
{
    std::size_t pivotBar = 0;
    PivotKind kind = PivotKind::high;
    std::size_t pivotRadius = 0;
    std::chrono::sys_seconds pivotBarStart{};
    std::chrono::sys_seconds confirmationBarStart{};
    std::size_t scaleLookback = 0;
    PrecedingRangeScale::Sample pivotTimeScale;
    PrecedingRangeScale::Sample confirmationTimeScale;
};

// One pass covers every supplied radius.  Its comparison is deliberately the
// same strict <= / >= rule used by CausalPriceLevelEngine::IsStrictPivot.
class StrictPivotCharacterizer final
{
public:
    explicit StrictPivotCharacterizer(std::vector<std::size_t> radii)
        : radii_(std::move(radii))
    {
        if (radii_.empty())
            throw std::invalid_argument("PRICE_LEVEL_PIVOT_RADII_EMPTY");
        std::sort(radii_.begin(), radii_.end());
        if (std::adjacent_find(radii_.begin(), radii_.end()) != radii_.end() ||
            radii_.front() == 0)
            throw std::invalid_argument("PRICE_LEVEL_PIVOT_RADII_INVALID");
        maxRadius_ = radii_.back();
        counts_.resize(radii_.size());
    }

    template <typename Callback>
    void AddCompletedBar(const CompletedBar& bar,
                         const std::vector<PrecedingRangeScale::Sample>&
                             scaleBeforeBar,
                         const std::vector<std::size_t>& scaleLookbacks,
                         Callback&& onPivot)
    {
        ValidateCompletedBar(bar);
        if (hasPreviousTime_ && bar.barStart <= previousTime_)
            throw std::invalid_argument(
                "PRICE_LEVEL_CHARACTERIZATION_BARS_NOT_STRICTLY_INCREASING");
        if (scaleBeforeBar.size() != scaleLookbacks.size())
            throw std::invalid_argument("PRICE_LEVEL_SCALE_CONFIGURATION_MISMATCH");
        hasPreviousTime_ = true;
        previousTime_ = bar.barStart;
        bars_.push_back({bar, scaleBeforeBar});
        const std::size_t currentBar = barsSeen_++;
        for (std::size_t radiusIndex = 0; radiusIndex < radii_.size(); ++radiusIndex)
        {
            const std::size_t radius = radii_[radiusIndex];
            if (bars_.size() < radius * 2 + 1) continue;
            const std::size_t pivotOffset = bars_.size() - 1 - radius;
            const std::size_t pivotBar = currentBar - radius;
            if (IsStrict(pivotOffset, radius, PivotKind::high))
            {
                ++counts_[radiusIndex].highs;
                PublishSamples(pivotBar, PivotKind::high, radius, pivotOffset,
                               scaleLookbacks, onPivot);
            }
            if (IsStrict(pivotOffset, radius, PivotKind::low))
            {
                ++counts_[radiusIndex].lows;
                PublishSamples(pivotBar, PivotKind::low, radius, pivotOffset,
                               scaleLookbacks, onPivot);
            }
        }
        const std::size_t maximum = maxRadius_ * 2 + 1;
        while (bars_.size() > maximum) bars_.pop_front();
    }

    const std::vector<std::size_t>& radii() const noexcept { return radii_; }
    const std::vector<PivotCounts>& counts() const noexcept { return counts_; }
    std::size_t barsSeen() const noexcept { return barsSeen_; }

private:
    struct RetainedBar final
    {
        CompletedBar bar;
        std::vector<PrecedingRangeScale::Sample> scaleBefore;
    };

    bool IsStrict(std::size_t pivotOffset, std::size_t radius,
                  PivotKind kind) const
    {
        const double price = kind == PivotKind::high ? bars_[pivotOffset].bar.high :
                                                       bars_[pivotOffset].bar.low;
        for (std::size_t index = pivotOffset - radius;
             index <= pivotOffset + radius; ++index)
        {
            if (index == pivotOffset) continue;
            const double other = kind == PivotKind::high ? bars_[index].bar.high :
                                                           bars_[index].bar.low;
            if (kind == PivotKind::high ? price <= other : price >= other)
                return false;
        }
        return true;
    }

    template <typename Callback>
    void PublishSamples(std::size_t pivotBar, PivotKind kind,
                        std::size_t pivotRadius, std::size_t pivotOffset,
                        const std::vector<std::size_t>& scaleLookbacks,
                        Callback&& onPivot) const
    {
        for (std::size_t scaleIndex = 0; scaleIndex < scaleLookbacks.size();
             ++scaleIndex)
        {
            onPivot(PivotScaleSample{pivotBar, kind, pivotRadius,
                bars_[pivotOffset].bar.barStart, bars_.back().bar.barStart,
                scaleLookbacks[scaleIndex],
                bars_[pivotOffset].scaleBefore[scaleIndex],
                bars_.back().scaleBefore[scaleIndex]});
        }
    }

    std::vector<std::size_t> radii_;
    std::size_t maxRadius_ = 0;
    std::vector<PivotCounts> counts_;
    std::deque<RetainedBar> bars_;
    std::size_t barsSeen_ = 0;
    bool hasPreviousTime_ = false;
    std::chrono::sys_seconds previousTime_{};
};

struct RunningDistribution final
{
    std::uint64_t count = 0;
    double sum = 0.0;
    double minimum = std::numeric_limits<double>::infinity();
    double maximum = -std::numeric_limits<double>::infinity();
    // A deterministic hash-ranked sample bounds memory while retaining
    // reproducible p50/p90 estimates for long ranges.  A partition with at
    // most kMaximumRetained values is exact.
    static constexpr std::size_t kMaximumRetained = 65'536;

    void Add(double value)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("PRICE_LEVEL_DISTRIBUTION_NONFINITE");
        ++count;
        sum += value;
        minimum = std::min(minimum, value);
        maximum = std::max(maximum, value);
        const std::uint64_t rank = SplitMix64(count);
        if (retained_.size() < kMaximumRetained)
            retained_.push({rank, value});
        else if (rank < retained_.top().first)
        {
            retained_.pop();
            retained_.push({rank, value});
        }
    }

    double mean() const noexcept
    {
        return count == 0 ? 0.0 : sum / static_cast<double>(count);
    }

    double Quantile(double fraction) const
    {
        if (retained_.empty() || fraction < 0.0 || fraction > 1.0)
            throw std::invalid_argument("PRICE_LEVEL_DISTRIBUTION_QUANTILE_INVALID");
        auto copy = retained_;
        std::vector<double> ordered;
        ordered.reserve(copy.size());
        while (!copy.empty())
        {
            ordered.push_back(copy.top().second);
            copy.pop();
        }
        std::sort(ordered.begin(), ordered.end());
        const std::size_t index = static_cast<std::size_t>(
            std::floor(fraction * static_cast<double>(ordered.size() - 1)));
        return ordered[index];
    }

    std::size_t retainedSampleSize() const noexcept { return retained_.size(); }

private:
    static std::uint64_t SplitMix64(std::uint64_t value) noexcept
    {
        value += 0x9e3779b97f4a7c15ULL;
        value = (value ^ (value >> 30U)) * 0xbf58476d1ce4e5b9ULL;
        value = (value ^ (value >> 27U)) * 0x94d049bb133111ebULL;
        return value ^ (value >> 31U);
    }

    std::priority_queue<std::pair<std::uint64_t, double>> retained_;
};

// The adaptive detector below is deliberately research-only.  In particular,
// it does not share the causal-price-level/v1 definition identifier, level
// identities, or configuration type.  It exists solely to characterize a
// candidate contract without changing the frozen production engine.
inline constexpr const char* kAdaptiveStudyContract =
    "price-level-adaptive-width-study-v1";
inline constexpr const char* kAgeTimingStudyContract =
    "price-level-phase5b-age-timing-study-v1";
inline constexpr const char* kBoundsConfirmationStudyContract =
    "price-level-phase5c-bounds-confirmation-study-v1";

enum class ScaleTiming { pivot_time, confirmation_time };

inline std::string_view CanonicalScaleTiming(ScaleTiming value)
{
    switch (value)
    {
        case ScaleTiming::pivot_time: return "pivot_time";
        case ScaleTiming::confirmation_time: return "confirmation_time";
    }
    throw std::invalid_argument("PRICE_LEVEL_ADAPTIVE_SCALE_TIMING_INVALID");
}

struct AdaptiveConfiguration final
{
    std::size_t pivotRadiusBars = 0;
    std::size_t scaleLookbackBars = 0;
    double scaleMultiplier = 0.0;
    ScaleTiming scaleTiming = ScaleTiming::pivot_time;
    std::size_t maxActiveLevels = 0;
    std::size_t maxAgeBars = 0;
    std::size_t maxRetainedPivotEvidence = 0;
    std::chrono::seconds completedBarDuration{0};
};

inline void ValidateAdaptiveConfiguration(const AdaptiveConfiguration& value)
{
    if (value.pivotRadiusBars == 0 || value.scaleLookbackBars == 0 ||
        !std::isfinite(value.scaleMultiplier) || value.scaleMultiplier < 0.0 ||
        value.maxActiveLevels == 0 || value.maxAgeBars == 0 ||
        value.maxRetainedPivotEvidence == 0 ||
        value.completedBarDuration <= std::chrono::seconds::zero())
    {
        throw std::invalid_argument("PRICE_LEVEL_ADAPTIVE_CONFIGURATION_INVALID");
    }
}

// Phase 5C varies only bounded-state limits.  Keeping the grid beside the
// research detector makes both the CLI and focused replay tests use the exact
// same fixed provisional core.
inline std::array<AdaptiveConfiguration, 9> Phase5CBoundsConfirmationConfigurations()
{
    constexpr std::array<std::size_t, 3> activeCaps{32, 48, 64};
    constexpr std::array<std::size_t, 3> evidenceCaps{8, 16, 32};
    std::array<AdaptiveConfiguration, 9> result{};
    std::size_t index = 0;
    for (const std::size_t maxActive : activeCaps)
    {
        for (const std::size_t maxEvidence : evidenceCaps)
        {
            result[index++] = {3, 64, 1.0, ScaleTiming::pivot_time, maxActive,
                               512, maxEvidence, std::chrono::seconds{900}};
        }
    }
    return result;
}

inline std::string CanonicalAdaptiveConfigurationIdentity(
    const AdaptiveConfiguration& value)
{
    ValidateAdaptiveConfiguration(value);
    return std::string{kAdaptiveStudyContract} + ";pivot_radius=" +
        std::to_string(value.pivotRadiusBars) + ";scale_statistic=median_high_low" +
        ";scale_lookback=" + std::to_string(value.scaleLookbackBars) +
        ";scale_multiplier=" + CanonicalDouble(value.scaleMultiplier) +
        ";scale_timing=" + std::string{CanonicalScaleTiming(value.scaleTiming)} +
        ";max_active=" + std::to_string(value.maxActiveLevels) +
        ";max_age=" + std::to_string(value.maxAgeBars) +
        ";max_pivot_evidence=" + std::to_string(value.maxRetainedPivotEvidence) +
        ";bar_duration=" + std::to_string(value.completedBarDuration.count());
}

struct AdaptiveLevel final
{
    std::string identity;
    PivotKind originatingPivot = PivotKind::high;
    Role currentRole = Role::resistance_like;
    double anchorPrice = 0.0;
    double zoneHalfWidth = 0.0;
    std::size_t originBar = 0;
    std::size_t availableBar = 0;
    std::size_t pivotObservationCount = 0;
    std::size_t retainedPivotEvidenceCount = 0;
};

struct AdaptiveObservation final
{
    InteractionKind kind = InteractionKind::level_established;
    AdaptiveLevel level;
    // Present only for a retest.  It is measured from the bar that emitted
    // the most recent cross to this later retest bar.
    std::optional<std::size_t> retestLatencyBars;
    // True only when a reinforcing pivot could not be retained because the
    // bounded pivot-evidence collection was already full.
    bool retainedEvidenceAlreadySaturated = false;
};

struct AdaptiveUpdate final
{
    std::vector<AdaptiveObservation> observations;
    // Counted before bounded active-state or retained-evidence policies are
    // applied.  This is a test-only seam proving those policies cannot alter
    // the upstream strict-pivot decision.
    std::size_t strictPivotsConfirmed = 0;
    std::size_t activeLevelCount = 0;
    // A research-only end-of-bar census.  It lets the characterization
    // harness distinguish levels still active at the requested window end
    // (right-censored) from levels that actually terminated.
    std::vector<AdaptiveLevel> activeLevels;
};

// Candidate pivot P merges into an existing level only if P falls inside that
// existing level's frozen zone: abs(P - anchor) <= existing zoneHalfWidth.
// Candidate width is consequently used only when establishing a new level.
// This keeps anchors non-chainable and prevents a later scale change from
// altering any established boundary.
class AdaptiveWidthResearchDetector final
{
public:
    AdaptiveWidthResearchDetector(std::string symbol, AdaptiveConfiguration configuration)
        : symbol_(std::move(symbol)), configuration_(configuration),
          configurationIdentity_(CanonicalAdaptiveConfigurationIdentity(configuration_))
    {
        ValidateAdaptiveConfiguration(configuration_);
    }

    const std::string& configurationIdentity() const noexcept
    {
        return configurationIdentity_;
    }

    AdaptiveUpdate AddCompletedBar(const CompletedBar& bar,
                                   PrecedingRangeScale::Sample scaleBeforeBar)
    {
        ValidateCompletedBar(bar);
        if (hasPreviousTime_ && bar.barStart <= previousTime_)
            throw std::invalid_argument("PRICE_LEVEL_ADAPTIVE_BARS_NOT_STRICTLY_INCREASING");
        if (scaleBeforeBar.count > configuration_.scaleLookbackBars ||
            !std::isfinite(scaleBeforeBar.median) || scaleBeforeBar.median < 0.0)
            throw std::invalid_argument("PRICE_LEVEL_ADAPTIVE_SCALE_SAMPLE_INVALID");
        hasPreviousTime_ = true;
        previousTime_ = bar.barStart;
        bars_.push_back({bar, scaleBeforeBar});
        const std::size_t currentBar = nextBar_++;
        AdaptiveUpdate result;
        Expire(currentBar, result.observations);
        if (currentBar > 0) ObserveInteractions(currentBar, result.observations);
        ConfirmLatestPivots(currentBar, result.strictPivotsConfirmed, result.observations);
        std::sort(result.observations.begin(), result.observations.end(),
            [](const AdaptiveObservation& left, const AdaptiveObservation& right) {
                if (left.level.identity != right.level.identity)
                    return left.level.identity < right.level.identity;
                return static_cast<int>(left.kind) < static_cast<int>(right.kind);
            });
        result.activeLevelCount = levels_.size();
        result.activeLevels.reserve(levels_.size());
        for (const InternalLevel& level : levels_)
            result.activeLevels.push_back(level.value);
        const std::size_t retained = configuration_.pivotRadiusBars * 2 + 1;
        while (bars_.size() > retained) bars_.pop_front();
        return result;
    }

private:
    struct RetainedBar final { CompletedBar bar; PrecedingRangeScale::Sample scale; };
    struct InternalLevel final
    {
        AdaptiveLevel value;
        bool touching = false;
        bool awaitingRetest = false;
        std::size_t crossBar = 0;
    };

    std::string symbol_;
    AdaptiveConfiguration configuration_;
    std::string configurationIdentity_;
    std::deque<RetainedBar> bars_;
    std::vector<InternalLevel> levels_;
    std::size_t nextBar_ = 0;
    bool hasPreviousTime_ = false;
    std::chrono::sys_seconds previousTime_{};

    bool IsStrict(std::size_t pivotOffset, PivotKind kind) const
    {
        const std::size_t radius = configuration_.pivotRadiusBars;
        const double price = kind == PivotKind::high ? bars_[pivotOffset].bar.high :
                                                       bars_[pivotOffset].bar.low;
        for (std::size_t index = pivotOffset - radius;
             index <= pivotOffset + radius; ++index)
        {
            if (index == pivotOffset) continue;
            const double other = kind == PivotKind::high ? bars_[index].bar.high :
                                                          bars_[index].bar.low;
            if (kind == PivotKind::high ? price <= other : price >= other) return false;
        }
        return true;
    }

    void Expire(std::size_t currentBar, std::vector<AdaptiveObservation>& observations)
    {
        for (auto iterator = levels_.begin(); iterator != levels_.end(); )
        {
            // This deliberately matches the frozen v1 boundary: a level made
            // available at A remains active through A + maxAgeBars and is
            // forcibly expired before interactions at A + maxAgeBars + 1.
            if (currentBar - iterator->value.availableBar > configuration_.maxAgeBars)
            {
                observations.push_back({InteractionKind::level_expired, iterator->value,
                                        std::nullopt, false});
                iterator = levels_.erase(iterator);
            }
            else ++iterator;
        }
    }

    void ObserveInteractions(std::size_t currentBar,
                             std::vector<AdaptiveObservation>& observations)
    {
        const CompletedBar& previous = bars_[bars_.size() - 2].bar;
        const CompletedBar& current = bars_.back().bar;
        for (InternalLevel& internal : levels_)
        {
            AdaptiveLevel& level = internal.value;
            const double lower = level.anchorPrice - level.zoneHalfWidth;
            const double upper = level.anchorPrice + level.zoneHalfWidth;
            const bool touching = current.low <= upper && current.high >= lower;
            if (touching && !internal.touching)
                observations.push_back({InteractionKind::touch, level, std::nullopt, false});
            internal.touching = touching;
            const bool crossedUp = previous.close < lower && current.close > upper;
            const bool crossedDown = previous.close > upper && current.close < lower;
            if (!crossedUp && !crossedDown)
            {
                if (internal.awaitingRetest && currentBar > internal.crossBar && touching)
                {
                    observations.push_back({InteractionKind::retest, level,
                        currentBar - internal.crossBar, false});
                    internal.awaitingRetest = false;
                }
                continue;
            }
            const Role before = level.currentRole;
            level.currentRole = crossedUp ? Role::support_like : Role::resistance_like;
            observations.push_back({crossedUp ? InteractionKind::cross_up :
                                    InteractionKind::cross_down, level, std::nullopt, false});
            internal.awaitingRetest = true;
            internal.crossBar = currentBar;
            if (before != level.currentRole)
                observations.push_back({InteractionKind::role_reversal, level,
                                        std::nullopt, false});
        }
    }

    void ConfirmLatestPivots(std::size_t currentBar, std::size_t& strictPivotsConfirmed,
                             std::vector<AdaptiveObservation>& observations)
    {
        const std::size_t radius = configuration_.pivotRadiusBars;
        if (bars_.size() < radius * 2 + 1) return;
        const std::size_t pivotOffset = bars_.size() - 1 - radius;
        if (IsStrict(pivotOffset, PivotKind::high))
        {
            ++strictPivotsConfirmed;
            AddPivot(currentBar, pivotOffset, PivotKind::high, bars_[pivotOffset].bar.high,
                     observations);
        }
        if (IsStrict(pivotOffset, PivotKind::low))
        {
            ++strictPivotsConfirmed;
            AddPivot(currentBar, pivotOffset, PivotKind::low, bars_[pivotOffset].bar.low,
                     observations);
        }
    }

    void AddPivot(std::size_t currentBar, std::size_t pivotOffset, PivotKind kind,
                  double price, std::vector<AdaptiveObservation>& observations)
    {
        const PrecedingRangeScale::Sample& scale =
            configuration_.scaleTiming == ScaleTiming::pivot_time
                ? bars_[pivotOffset].scale : bars_.back().scale;
        // A startup prefix is valid evidence, but no level can be established
        // before at least one preceding completed range exists.
        if (scale.count == 0) return;
        const double width = configuration_.scaleMultiplier * scale.median;
        if (!std::isfinite(width) || width < 0.0)
            throw std::invalid_argument("PRICE_LEVEL_ADAPTIVE_WIDTH_INVALID");
        auto merge = levels_.end();
        for (auto candidate = levels_.begin(); candidate != levels_.end(); ++candidate)
        {
            const double distance = std::fabs(candidate->value.anchorPrice - price);
            if (distance > candidate->value.zoneHalfWidth) continue;
            if (merge == levels_.end() ||
                distance < std::fabs(merge->value.anchorPrice - price) ||
                (distance == std::fabs(merge->value.anchorPrice - price) &&
                 candidate->value.identity < merge->value.identity))
            {
                merge = candidate;
            }
        }
        if (merge != levels_.end())
        {
            AdaptiveLevel& level = merge->value;
            const bool saturated = level.retainedPivotEvidenceCount >=
                configuration_.maxRetainedPivotEvidence;
            ++level.pivotObservationCount;
            if (!saturated) ++level.retainedPivotEvidenceCount;
            observations.push_back({InteractionKind::level_reinforced, level, std::nullopt,
                                    saturated});
            return;
        }
        if (levels_.size() == configuration_.maxActiveLevels)
        {
            const auto evicted = std::min_element(levels_.begin(), levels_.end(),
                [](const InternalLevel& left, const InternalLevel& right) {
                    return left.value.availableBar != right.value.availableBar
                        ? left.value.availableBar < right.value.availableBar
                        : left.value.identity < right.value.identity;
                });
            observations.push_back({InteractionKind::level_evicted, evicted->value,
                                    std::nullopt, false});
            levels_.erase(evicted);
        }
        const std::size_t pivotBar = currentBar - configuration_.pivotRadiusBars;
        const std::string identity = std::string{kAdaptiveStudyContract} + ";symbol=" +
            symbol_ + ";definition=" + configurationIdentity_ + ";origin_bar=" +
            std::to_string(pivotBar) + ";origin_time=" +
            CanonicalTimestamp(bars_[pivotOffset].bar.barStart) + ";kind=" +
            std::string{CanonicalPivotKind(kind)} + ";price=" + CanonicalDouble(price);
        AdaptiveLevel level{identity, kind,
            kind == PivotKind::high ? Role::resistance_like : Role::support_like,
            price, width, pivotBar, currentBar, 1, 1};
        observations.push_back({InteractionKind::level_established, level, std::nullopt, false});
        levels_.push_back({std::move(level), false, false, 0});
    }
};

} // namespace EA::PriceLevel::Characterization
