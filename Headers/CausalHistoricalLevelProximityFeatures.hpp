#ifndef CausalHistoricalLevelProximityFeatures_hpp
#define CausalHistoricalLevelProximityFeatures_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

// Causal, symbol-local proximity to corroborated weekly swing levels.  Each
// instance owns one source series; callers must feed completed bars in time
// order.  Weekly candidates are confirmed with two completed weeks on either
// side, so the current active week and all future bars are excluded from level
// discovery.
class CausalHistoricalLevelProximity
{
public:
    static constexpr std::size_t kPivotRadiusWeeks = 2;
    static constexpr std::size_t kBandwidthWeeks = 26;
    static constexpr std::size_t kMinimumHistoryWeeks = 104;
    static constexpr std::int64_t kRetentionWeeks = 260;
    static constexpr std::int64_t kTaperWeeks = 26;
    static constexpr double kBandwidthFraction = 0.25;
    static constexpr double kMinimumLogBandwidth = 1.0e-6;

    float AddCompletedBar(float high,
                          float low,
                          float close,
                          std::int64_t epochSeconds)
    {
        if (!ValidBar(high, low, close)) return 0.0f;

        const std::int64_t week = UtcMondayWeek(epochSeconds);
        if (hasActiveWeek && week < activeWeek.week) return 0.0f;

        if (!hasActiveWeek)
        {
            activeWeek = MakeWeeklyBar(week, high, low, close);
            hasActiveWeek = true;
        }
        else if (week > activeWeek.week)
        {
            FinalizeActiveWeek();
            activeWeek = MakeWeeklyBar(week, high, low, close);
            EvictExpiredCandidates(week);
            RecomputeCorroboration(week);
        }
        else
        {
            activeWeek.high = std::max(activeWeek.high, static_cast<double>(high));
            activeWeek.low = std::min(activeWeek.low, static_cast<double>(low));
            activeWeek.close = static_cast<double>(close);
        }

        return CurrentValue(close, week);
    }

    std::size_t CompletedWeekCount() const { return completedWeeks.size(); }
    std::size_t CandidateCount() const { return candidates.size(); }

private:
    struct WeeklyBar
    {
        std::int64_t week = 0;
        double high = 0.0;
        double low = 0.0;
        double close = 0.0;
    };

    struct Candidate
    {
        std::int64_t pivotWeek = 0;
        double logLevel = 0.0;
        double bandwidth = 0.0;
        double corroboration = 0.0;
    };

    static bool ValidBar(float high, float low, float close)
    {
        return std::isfinite(high) && std::isfinite(low) &&
               std::isfinite(close) && high > 0.0f && low > 0.0f &&
               close > 0.0f && high >= low && close >= low && close <= high;
    }

    static WeeklyBar MakeWeeklyBar(std::int64_t week,
                                   float high,
                                   float low,
                                   float close)
    {
        return {week, static_cast<double>(high), static_cast<double>(low),
                static_cast<double>(close)};
    }

    static std::int64_t FloorDivide(std::int64_t numerator,
                                    std::int64_t denominator)
    {
        const std::int64_t quotient = numerator / denominator;
        const std::int64_t remainder = numerator % denominator;
        return quotient - ((remainder < 0) ? 1 : 0);
    }

    static std::int64_t UtcMondayWeek(std::int64_t epochSeconds)
    {
        constexpr std::int64_t kSecondsPerDay = 24 * 60 * 60;
        constexpr std::int64_t kSecondsPerWeek = 7 * kSecondsPerDay;
        // 1970-01-01 was Thursday; this offset makes Monday the boundary.
        constexpr std::int64_t kMondayOffset = 3 * kSecondsPerDay;
        if (epochSeconds > std::numeric_limits<std::int64_t>::max() - kMondayOffset)
            return FloorDivide(epochSeconds, kSecondsPerWeek);
        return FloorDivide(epochSeconds + kMondayOffset, kSecondsPerWeek);
    }

    void FinalizeActiveWeek()
    {
        completedWeeks.push_back(activeWeek);
        DiscoverLatestPivot();

        const std::int64_t newestWeek = completedWeeks.back().week;
        while (!completedWeeks.empty() &&
               newestWeek - completedWeeks.front().week > kRetentionWeeks)
            completedWeeks.pop_front();
    }

    bool LatestPivotHasContinuousHistory(std::size_t pivotIndex) const
    {
        if (pivotIndex < kBandwidthWeeks - 1 ||
            pivotIndex + kPivotRadiusWeeks >= completedWeeks.size())
            return false;
        const std::size_t start = pivotIndex - (kBandwidthWeeks - 1);
        for (std::size_t index = start + 1;
             index <= pivotIndex + kPivotRadiusWeeks;
             ++index)
            if (completedWeeks[index].week != completedWeeks[index - 1].week + 1)
                return false;
        return true;
    }

    double PivotBandwidth(std::size_t pivotIndex) const
    {
        std::vector<double> ranges;
        ranges.reserve(kBandwidthWeeks);
        const std::size_t start = pivotIndex - (kBandwidthWeeks - 1);
        for (std::size_t index = start; index <= pivotIndex; ++index)
        {
            const WeeklyBar& week = completedWeeks[index];
            const double range = std::log(week.high / week.low);
            if (!std::isfinite(range) || range < 0.0)
                return 0.0;
            ranges.push_back(range);
        }
        const auto middle = ranges.begin() +
            static_cast<std::ptrdiff_t>(ranges.size() / 2);
        std::nth_element(ranges.begin(), middle, ranges.end());
        double median = *middle;
        if (ranges.size() % 2 == 0)
        {
            const auto lower = std::max_element(ranges.begin(), middle);
            median = 0.5 * (median + *lower);
        }
        const double bandwidth = std::max(
            kMinimumLogBandwidth, kBandwidthFraction * median);
        return std::isfinite(bandwidth) ? bandwidth : 0.0;
    }

    void AddCandidate(std::int64_t pivotWeek,
                      double level,
                      double bandwidth)
    {
        const double logLevel = std::log(level);
        if (!std::isfinite(logLevel) || !std::isfinite(bandwidth) ||
            bandwidth < kMinimumLogBandwidth)
            return;
        candidates.push_back({pivotWeek, logLevel, bandwidth, 0.0});
    }

    void DiscoverLatestPivot()
    {
        if (completedWeeks.size() < 2 * kPivotRadiusWeeks + 1) return;
        const std::size_t pivotIndex =
            completedWeeks.size() - 1 - kPivotRadiusWeeks;
        if (!LatestPivotHasContinuousHistory(pivotIndex)) return;

        const WeeklyBar& pivot = completedWeeks[pivotIndex];
        bool strictHigh = true;
        bool strictLow = true;
        for (std::size_t offset = 1; offset <= kPivotRadiusWeeks; ++offset)
        {
            strictHigh = strictHigh &&
                pivot.high > completedWeeks[pivotIndex - offset].high &&
                pivot.high > completedWeeks[pivotIndex + offset].high;
            strictLow = strictLow &&
                pivot.low < completedWeeks[pivotIndex - offset].low &&
                pivot.low < completedWeeks[pivotIndex + offset].low;
        }

        const double bandwidth = PivotBandwidth(pivotIndex);
        if (bandwidth == 0.0) return;
        if (strictHigh) AddCandidate(pivot.week, pivot.high, bandwidth);
        if (strictLow) AddCandidate(pivot.week, pivot.low, bandwidth);
    }

    static double CandidateAgeWeight(const Candidate& candidate,
                                     std::int64_t currentWeek)
    {
        const std::int64_t age = currentWeek - candidate.pivotWeek;
        if (age < 0 || age >= kRetentionWeeks) return 0.0;
        const std::int64_t fullWeightWeeks = kRetentionWeeks - kTaperWeeks;
        if (age <= fullWeightWeeks) return 1.0;
        return static_cast<double>(kRetentionWeeks - age) /
               static_cast<double>(kTaperWeeks);
    }

    void EvictExpiredCandidates(std::int64_t currentWeek)
    {
        candidates.erase(
            std::remove_if(candidates.begin(), candidates.end(),
                [currentWeek](const Candidate& candidate)
                {
                    return currentWeek - candidate.pivotWeek >= kRetentionWeeks;
                }),
            candidates.end());
    }

    void RecomputeCorroboration(std::int64_t currentWeek)
    {
        for (std::size_t index = 0; index < candidates.size(); ++index)
        {
            double support = 0.0;
            Candidate& candidate = candidates[index];
            for (std::size_t otherIndex = 0;
                 otherIndex < candidates.size(); ++otherIndex)
            {
                if (otherIndex == index) continue;
                const Candidate& other = candidates[otherIndex];
                const double denominator = std::hypot(candidate.bandwidth,
                                                      other.bandwidth);
                if (!std::isfinite(denominator) || denominator <= 0.0) continue;
                const double distance =
                    (candidate.logLevel - other.logLevel) / denominator;
                if (!std::isfinite(distance) || std::fabs(distance) > 16.0) continue;
                support += CandidateAgeWeight(other, currentWeek) *
                           std::exp(-0.5 * distance * distance);
            }
            candidate.corroboration = std::isfinite(support)
                ? 1.0 - std::exp(-support)
                : 0.0;
        }
    }

    bool HasSufficientHistory() const
    {
        return completedWeeks.size() >= kMinimumHistoryWeeks &&
               completedWeeks.back().week - completedWeeks.front().week >=
                   static_cast<std::int64_t>(kMinimumHistoryWeeks - 1);
    }

    float CurrentValue(float close, std::int64_t currentWeek) const
    {
        if (!HasSufficientHistory() || candidates.size() < 2) return 0.0f;
        const double logClose = std::log(static_cast<double>(close));
        if (!std::isfinite(logClose)) return 0.0f;

        double density = 0.0;
        for (const Candidate& candidate : candidates)
        {
            const double distance =
                (logClose - candidate.logLevel) / candidate.bandwidth;
            if (!std::isfinite(distance) || std::fabs(distance) > 16.0) continue;
            density += CandidateAgeWeight(candidate, currentWeek) *
                       candidate.corroboration *
                       std::exp(-0.5 * distance * distance);
        }
        if (!std::isfinite(density) || density <= 0.0) return 0.0f;
        const double proximity = 1.0 - std::exp(-0.5 * density);
        if (!std::isfinite(proximity) || proximity <= 0.0) return 0.0f;
        const float result = static_cast<float>(proximity);
        return std::min(result, std::nextafter(1.0f, 0.0f));
    }

    bool hasActiveWeek = false;
    WeeklyBar activeWeek;
    std::deque<WeeklyBar> completedWeeks;
    std::vector<Candidate> candidates;
};

#endif /* CausalHistoricalLevelProximityFeatures_hpp */
