#pragma once

#include "CausalPriceLevelEngine.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <deque>
#include <limits>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

// Production causal-price-level/v2.  This is deliberately separate from the
// frozen v1 engine: v2 has an adaptive, pivot-time width contract whereas v1
// has a caller-provided fixed width contract.
namespace EA::PriceLevel::V2
{
inline constexpr std::string_view kDefinitionId = "causal-price-level";
inline constexpr std::string_view kDefinitionVersion = "v2";

struct Configuration final
{
    std::size_t pivotRadiusBars = 0;
    std::size_t scaleLookbackBars = 0;
    double scaleMultiplier = 0.0;
    std::size_t maxAgeBars = 0;
    std::size_t maxActiveLevels = 0;
    std::size_t maxRetainedPivotEvidence = 0;
    std::chrono::seconds completedBarDuration{0};
};

inline constexpr Configuration ProductionConfiguration()
{
    return {3, 64, 1.0, 512, 48, 32, std::chrono::seconds{900}};
}

inline void ValidateConfiguration(const Configuration& value)
{
    if (value.pivotRadiusBars == 0 ||
        value.pivotRadiusBars > std::numeric_limits<std::size_t>::max() / 2 ||
        value.scaleLookbackBars == 0 || !std::isfinite(value.scaleMultiplier) ||
        value.scaleMultiplier < 0.0 || value.maxAgeBars == 0 ||
        value.maxActiveLevels == 0 || value.maxRetainedPivotEvidence == 0 ||
        value.completedBarDuration <= std::chrono::seconds::zero())
    {
        throw std::invalid_argument("PRICE_LEVEL_V2_CONFIGURATION_INVALID");
    }
}

inline std::string CanonicalConfigurationIdentity(const Configuration& value)
{
    ValidateConfiguration(value);
    return "price-level-definition-v2;id=" + std::string{kDefinitionId} +
        ";version=" + std::string{kDefinitionVersion} + ";pivot_radius=" +
        std::to_string(value.pivotRadiusBars) +
        ";scale_statistic=median_high_low;scale_lookback=" +
        std::to_string(value.scaleLookbackBars) + ";scale_multiplier=" +
        CanonicalDouble(value.scaleMultiplier) + ";scale_timing=pivot_time" +
        ";max_active=" + std::to_string(value.maxActiveLevels) +
        ";max_age=" + std::to_string(value.maxAgeBars) +
        ";max_pivot_evidence=" + std::to_string(value.maxRetainedPivotEvidence) +
        ";bar_duration=" + std::to_string(value.completedBarDuration.count());
}

struct ScaleSample final
{
    std::size_t count = 0;
    double median = 0.0;

    bool operator==(const ScaleSample&) const = default;
};

struct Level final
{
    std::string identity;
    PivotKind originatingPivot = PivotKind::high;
    Role currentRole = Role::resistance_like;
    double anchorPrice = 0.0;
    double zoneHalfWidth = 0.0;
    double lower = 0.0;
    double upper = 0.0;
    std::size_t originBar = 0;
    std::chrono::sys_seconds originObservedAt{};
    std::size_t availableBar = 0;
    std::chrono::sys_seconds availableAt{};
    std::size_t pivotObservationCount = 0;
    std::size_t touchCount = 0;
    std::vector<std::string> retainedPivotEvidence;

    bool operator==(const Level&) const = default;
};

struct Observation final
{
    std::string identity;
    std::string levelIdentity;
    InteractionKind kind = InteractionKind::level_established;
    std::chrono::sys_seconds observedAt{};
    std::chrono::sys_seconds availableAt{};
    std::string sourceProvenance;
    std::string sourceObservationId;
    std::optional<Role> roleBefore;
    std::optional<Role> roleAfter;
    bool retainedEvidenceAlreadySaturated = false;
    Level level;

    bool operator==(const Observation&) const = default;
};

struct Update final
{
    std::chrono::sys_seconds decisionTime{};
    // The causal prefix that was available immediately before this bar.  It is
    // an auditable detector seam, not a model-facing observation.
    ScaleSample scaleBeforeCurrentBar;
    std::size_t strictPivotsConfirmed = 0;
    std::vector<Observation> observations;
    std::vector<Level> activeLevels;

    std::string CanonicalRepresentation() const
    {
        std::string result = "price-level-update-v2;decision=" +
            CanonicalTimestamp(decisionTime) + ";scale_count=" +
            std::to_string(scaleBeforeCurrentBar.count) + ";scale_median=" +
            CanonicalDouble(scaleBeforeCurrentBar.median) + ";strict_pivots=" +
            std::to_string(strictPivotsConfirmed) + ";observations=" +
            std::to_string(observations.size()) + ";levels=" +
            std::to_string(activeLevels.size());
        for (const Observation& observation : observations)
            result += ";observation=" + observation.identity;
        for (const Level& level : activeLevels)
            result += ";level=" + level.identity + ";role=" +
                std::string{CanonicalRole(level.currentRole)} + ";anchor=" +
                CanonicalDouble(level.anchorPrice) + ";width=" +
                CanonicalDouble(level.zoneHalfWidth) + ";pivots=" +
                std::to_string(level.pivotObservationCount) + ";touches=" +
                std::to_string(level.touchCount);
        return result;
    }
};

class CausalPriceLevelEngine final
{
public:
    CausalPriceLevelEngine(std::string symbol, Configuration configuration)
        : symbol_(CanonicalSymbol::Normalize(symbol)), configuration_(configuration),
          configurationIdentity_(CanonicalConfigurationIdentity(configuration_)),
          sourceProvenance_("causal-price-level-engine-v2;symbol=" + symbol_ +
              ";definition=" + configurationIdentity_)
    {
    }

    const Configuration& configuration() const noexcept { return configuration_; }
    const std::string& configurationIdentity() const noexcept
    {
        return configurationIdentity_;
    }
    const std::string& sourceProvenance() const noexcept { return sourceProvenance_; }
    std::size_t completedBarCount() const noexcept { return nextBar_; }

    Update AddCompletedBar(const CompletedBar& bar)
    {
        ValidateBar(bar);
        if (hasPreviousTime_ && bar.barStart <= previousTime_)
            throw std::invalid_argument("PRICE_LEVEL_V2_BARS_NOT_STRICTLY_INCREASING");
        const ScaleSample scaleBefore = ScaleBeforeCurrentBar();
        hasPreviousTime_ = true;
        previousTime_ = bar.barStart;
        bars_.push_back({bar, scaleBefore});
        const std::size_t currentBar = nextBar_++;
        const std::chrono::sys_seconds decision = CompletedAt(bar);
        Update result;
        result.decisionTime = decision;
        result.scaleBeforeCurrentBar = scaleBefore;

        Expire(currentBar, decision, result.observations);
        if (currentBar > 0) ObserveInteractions(currentBar, decision, result.observations);
        ConfirmLatestPivots(currentBar, decision, result.strictPivotsConfirmed,
                            result.observations);
        SortObservations(result.observations);
        result.activeLevels = ActiveLevels();
        AddCompletedRange(bar.high - bar.low);
        const std::size_t retained = configuration_.pivotRadiusBars * 2 + 1;
        while (bars_.size() > retained) bars_.pop_front();
        return result;
    }

    std::vector<Level> ActiveLevels() const
    {
        std::vector<Level> result;
        result.reserve(levels_.size());
        for (const InternalLevel& level : levels_) result.push_back(level.value);
        std::sort(result.begin(), result.end(), [](const Level& left, const Level& right) {
            return left.anchorPrice != right.anchorPrice
                ? left.anchorPrice < right.anchorPrice : left.identity < right.identity;
        });
        return result;
    }

private:
    struct RetainedBar final { CompletedBar bar; ScaleSample scaleBefore; };
    struct InternalLevel final
    {
        Level value;
        bool touching = false;
        bool awaitingRetest = false;
        std::size_t crossBar = 0;
    };

    std::string symbol_;
    Configuration configuration_;
    std::string configurationIdentity_;
    std::string sourceProvenance_;
    std::deque<RetainedBar> bars_;
    std::vector<InternalLevel> levels_;
    std::deque<double> ranges_;
    std::multiset<double> orderedRanges_;
    std::size_t nextBar_ = 0;
    bool hasPreviousTime_ = false;
    std::chrono::sys_seconds previousTime_{};

    std::chrono::sys_seconds CompletedAt(const CompletedBar& bar) const
    {
        return bar.barStart + configuration_.completedBarDuration;
    }

    static void ValidateBar(const CompletedBar& bar)
    {
        if (!std::isfinite(bar.open) || !std::isfinite(bar.high) ||
            !std::isfinite(bar.low) || !std::isfinite(bar.close) ||
            bar.high < bar.low || bar.open < bar.low || bar.open > bar.high ||
            bar.close < bar.low || bar.close > bar.high)
        {
            throw std::invalid_argument("PRICE_LEVEL_V2_BAR_INVALID");
        }
    }

    ScaleSample ScaleBeforeCurrentBar() const
    {
        if (ranges_.empty()) return {};
        const auto middle = std::next(orderedRanges_.begin(),
            static_cast<std::ptrdiff_t>((orderedRanges_.size() - 1) / 2));
        double median = *middle;
        if (orderedRanges_.size() % 2 == 0)
            median = (median + *std::next(middle)) / 2.0;
        return {ranges_.size(), median};
    }

    void AddCompletedRange(double range)
    {
        ranges_.push_back(range);
        orderedRanges_.insert(range);
        if (ranges_.size() > configuration_.scaleLookbackBars)
        {
            const double removed = ranges_.front();
            ranges_.pop_front();
            const auto found = orderedRanges_.find(removed);
            if (found == orderedRanges_.end())
                throw std::logic_error("PRICE_LEVEL_V2_SCALE_INTERNAL_ORDERING");
            orderedRanges_.erase(found);
        }
    }

    std::string PivotSourceId(PivotKind kind, std::size_t bar,
                              std::chrono::sys_seconds barStart, double price) const
    {
        return "price-level-pivot-v2;kind=" + std::string{CanonicalPivotKind(kind)} +
            ";bar=" + std::to_string(bar) + ";bar_start=" +
            CanonicalTimestamp(barStart) + ";price=" + CanonicalDouble(price);
    }

    std::string LevelIdentity(std::string_view pivotSourceId) const
    {
        return "price-level-v2;symbol=" + symbol_ + ";definition=" +
            configurationIdentity_ + ";origin=" + std::string{pivotSourceId};
    }

    Observation MakeObservation(const Level& level, InteractionKind kind,
                                std::chrono::sys_seconds observed,
                                std::chrono::sys_seconds available,
                                std::string sourceId,
                                std::optional<Role> before = std::nullopt,
                                std::optional<Role> after = std::nullopt,
                                bool saturated = false) const
    {
        const std::string identity = "price-level-observation-v2;level=" + level.identity +
            ";kind=" + std::string{CanonicalInteractionKind(kind)} + ";observed=" +
            CanonicalTimestamp(observed) + ";available=" + CanonicalTimestamp(available) +
            ";source=" + sourceId + ";before=" +
            (before ? std::string{CanonicalRole(*before)} : "absent") + ";after=" +
            (after ? std::string{CanonicalRole(*after)} : "absent") + ";saturated=" +
            (saturated ? "true" : "false");
        return {identity, level.identity, kind, observed, available, sourceProvenance_,
                std::move(sourceId), before, after, saturated, level};
    }

    static void SortObservations(std::vector<Observation>& observations)
    {
        std::sort(observations.begin(), observations.end(),
            [](const Observation& left, const Observation& right) {
                return left.identity < right.identity;
            });
    }

    void Expire(std::size_t currentBar, std::chrono::sys_seconds decision,
                std::vector<Observation>& observations)
    {
        for (auto iterator = levels_.begin(); iterator != levels_.end(); )
        {
            if (currentBar - iterator->value.availableBar > configuration_.maxAgeBars)
            {
                observations.push_back(MakeObservation(iterator->value,
                    InteractionKind::level_expired, decision, decision,
                    "price-level-expiry-v2;bar=" + std::to_string(currentBar)));
                iterator = levels_.erase(iterator);
            }
            else ++iterator;
        }
    }

    void ObserveInteractions(std::size_t currentBar,
                             std::chrono::sys_seconds decision,
                             std::vector<Observation>& observations)
    {
        const CompletedBar& previous = bars_[bars_.size() - 2].bar;
        const CompletedBar& current = bars_.back().bar;
        const std::string source = "price-level-bar-v2;bar=" +
            std::to_string(currentBar) + ";bar_start=" + CanonicalTimestamp(current.barStart);
        for (InternalLevel& internal : levels_)
        {
            Level& level = internal.value;
            const bool touching = current.low <= level.upper && current.high >= level.lower;
            if (touching && !internal.touching)
            {
                SaturatingIncrement(level.touchCount);
                observations.push_back(MakeObservation(level, InteractionKind::touch,
                    decision, decision, source));
            }
            internal.touching = touching;
            const bool crossedUp = previous.close < level.lower && current.close > level.upper;
            const bool crossedDown = previous.close > level.upper && current.close < level.lower;
            if (!crossedUp && !crossedDown)
            {
                if (internal.awaitingRetest && currentBar > internal.crossBar && touching)
                {
                    observations.push_back(MakeObservation(level, InteractionKind::retest,
                        decision, decision, source));
                    internal.awaitingRetest = false;
                }
                continue;
            }
            const Role before = level.currentRole;
            const Role after = crossedUp ? Role::support_like : Role::resistance_like;
            // Phase 5 records the post-cross role with the cross observation.
            level.currentRole = after;
            observations.push_back(MakeObservation(level,
                crossedUp ? InteractionKind::cross_up : InteractionKind::cross_down,
                decision, decision, source, before, after));
            internal.awaitingRetest = true;
            internal.crossBar = currentBar;
            if (before != after)
                observations.push_back(MakeObservation(level, InteractionKind::role_reversal,
                    decision, decision, source, before, after));
        }
    }

    void ConfirmLatestPivots(std::size_t currentBar, std::chrono::sys_seconds decision,
                             std::size_t& strictPivotsConfirmed,
                             std::vector<Observation>& observations)
    {
        const std::size_t radius = configuration_.pivotRadiusBars;
        if (bars_.size() < radius * 2 + 1) return;
        const std::size_t pivotOffset = bars_.size() - 1 - radius;
        if (IsStrictPivot(pivotOffset, PivotKind::high))
        {
            ++strictPivotsConfirmed;
            AddPivot(currentBar, pivotOffset, PivotKind::high,
                     bars_[pivotOffset].bar.high, decision, observations);
        }
        if (IsStrictPivot(pivotOffset, PivotKind::low))
        {
            ++strictPivotsConfirmed;
            AddPivot(currentBar, pivotOffset, PivotKind::low,
                     bars_[pivotOffset].bar.low, decision, observations);
        }
    }

    bool IsStrictPivot(std::size_t pivotOffset, PivotKind kind) const
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

    void AddPivot(std::size_t currentBar, std::size_t pivotOffset, PivotKind kind,
                  double price, std::chrono::sys_seconds available,
                  std::vector<Observation>& observations)
    {
        const ScaleSample& scale = bars_[pivotOffset].scaleBefore;
        // A startup predecessor prefix is valid.  No level can exist where
        // the originating pivot has no completed predecessor range at all.
        if (scale.count == 0) return;
        const double width = configuration_.scaleMultiplier * scale.median;
        if (!std::isfinite(width) || width < 0.0)
            throw std::invalid_argument("PRICE_LEVEL_V2_WIDTH_INVALID");
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
        const std::size_t pivotBar = currentBar - configuration_.pivotRadiusBars;
        const std::string pivotId = PivotSourceId(kind, pivotBar,
            bars_[pivotOffset].bar.barStart, price);
        if (merge != levels_.end())
        {
            Level& level = merge->value;
            const bool saturated = level.retainedPivotEvidence.size() >=
                configuration_.maxRetainedPivotEvidence;
            SaturatingIncrement(level.pivotObservationCount);
            if (!saturated) level.retainedPivotEvidence.push_back(pivotId);
            observations.push_back(MakeObservation(level, InteractionKind::level_reinforced,
                bars_[pivotOffset].bar.barStart + configuration_.completedBarDuration,
                available, pivotId, std::nullopt, std::nullopt, saturated));
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
            observations.push_back(MakeObservation(evicted->value,
                InteractionKind::level_evicted, available, available,
                "price-level-capacity-eviction-v2;new_pivot=" + pivotId));
            levels_.erase(evicted);
        }
        const double lower = price - width;
        const double upper = price + width;
        if (!std::isfinite(lower) || !std::isfinite(upper))
            throw std::invalid_argument("PRICE_LEVEL_V2_ZONE_NONFINITE");
        const Role role = kind == PivotKind::high ? Role::resistance_like :
                                                   Role::support_like;
        Level level{LevelIdentity(pivotId), kind, role, price, width, lower, upper,
            pivotBar, bars_[pivotOffset].bar.barStart + configuration_.completedBarDuration,
            currentBar, available, 1, 0, {pivotId}};
        observations.push_back(MakeObservation(level, InteractionKind::level_established,
            level.originObservedAt, available, pivotId));
        levels_.push_back({std::move(level), false, false, 0});
    }

    static void SaturatingIncrement(std::size_t& value)
    {
        if (value != std::numeric_limits<std::size_t>::max()) ++value;
    }
};
} // namespace EA::PriceLevel::V2
