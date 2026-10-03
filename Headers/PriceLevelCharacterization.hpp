#pragma once

// Research-only Price-Level Phase 5A helpers.  This header does not alter the
// causal-price-level/v1 detector or expose any model feature.  It measures the
// v1 strict-pivot rule and causal, preceding-range scale candidates.

#include "CausalPriceLevelEngine.hpp"

#include <algorithm>
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

} // namespace EA::PriceLevel::Characterization
