#pragma once

#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

// A detector-only, bounded, causal description of confirmed swing-price
// zones.  It intentionally has no Tensor or MarketStructure dependency; a
// later bridge may copy its immutable observations into that boundary.
namespace EA::PriceLevel
{
inline constexpr std::string_view kDefinitionId = "causal-price-level";
inline constexpr std::string_view kDefinitionVersion = "v1";

struct CompletedBar final
{
    std::chrono::sys_seconds barStart{};
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;

    bool operator==(const CompletedBar&) const = default;
};

enum class PivotKind { high, low };
enum class Role { resistance_like, support_like };
enum class InteractionKind
{
    level_established,
    level_reinforced,
    touch,
    cross_up,
    cross_down,
    retest,
    role_reversal,
    level_expired,
    level_evicted,
};

inline std::string_view CanonicalPivotKind(PivotKind value)
{
    switch (value)
    {
        case PivotKind::high: return "high";
        case PivotKind::low: return "low";
    }
    throw std::invalid_argument("PRICE_LEVEL_PIVOT_KIND_INVALID");
}

inline std::string_view CanonicalRole(Role value)
{
    switch (value)
    {
        case Role::resistance_like: return "resistance_like";
        case Role::support_like: return "support_like";
    }
    throw std::invalid_argument("PRICE_LEVEL_ROLE_INVALID");
}

inline std::string_view CanonicalInteractionKind(InteractionKind value)
{
    switch (value)
    {
        case InteractionKind::level_established: return "level_established";
        case InteractionKind::level_reinforced: return "level_reinforced";
        case InteractionKind::touch: return "touch";
        case InteractionKind::cross_up: return "cross_up";
        case InteractionKind::cross_down: return "cross_down";
        case InteractionKind::retest: return "retest";
        case InteractionKind::role_reversal: return "role_reversal";
        case InteractionKind::level_expired: return "level_expired";
        case InteractionKind::level_evicted: return "level_evicted";
    }
    throw std::invalid_argument("PRICE_LEVEL_INTERACTION_KIND_INVALID");
}

inline std::string CanonicalDouble(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("PRICE_LEVEL_NONFINITE_VALUE");
    if (value == 0.0) value = 0.0;
    std::ostringstream result;
    result.imbue(std::locale::classic());
    result << std::setprecision(17) << value;
    return result.str();
}

inline std::string CanonicalTimestamp(std::chrono::sys_seconds value)
{
    return std::to_string(value.time_since_epoch().count());
}

struct Configuration final
{
    // A pivot at i is confirmed only after pivotRadiusBars later completed
    // bars.  The half width is both the fixed zone half width and the fixed,
    // non-chainable anchor merge tolerance.
    std::size_t pivotRadiusBars = 0;
    double zoneHalfWidth = 0.0;
    std::size_t maxActiveLevels = 0;
    std::size_t maxAgeBars = 0;
    std::size_t maxRetainedPivotEvidence = 0;
    std::chrono::seconds completedBarDuration{0};
};

inline void ValidateConfiguration(const Configuration& value)
{
    if (value.pivotRadiusBars == 0 ||
        value.pivotRadiusBars > std::numeric_limits<std::size_t>::max() / 2 ||
        !std::isfinite(value.zoneHalfWidth) ||
        value.zoneHalfWidth < 0.0 || value.maxActiveLevels == 0 ||
        value.maxAgeBars == 0 || value.maxRetainedPivotEvidence == 0 ||
        value.completedBarDuration <= std::chrono::seconds::zero())
    {
        throw std::invalid_argument("PRICE_LEVEL_CONFIGURATION_INVALID");
    }
}

inline std::string CanonicalConfigurationIdentity(const Configuration& value)
{
    ValidateConfiguration(value);
    return "price-level-definition-v1;id=" + std::string{kDefinitionId} +
        ";version=" + std::string{kDefinitionVersion} + ";pivot_radius=" +
        std::to_string(value.pivotRadiusBars) + ";zone_half_width=" +
        CanonicalDouble(value.zoneHalfWidth) + ";max_active=" +
        std::to_string(value.maxActiveLevels) + ";max_age=" +
        std::to_string(value.maxAgeBars) + ";max_pivot_evidence=" +
        std::to_string(value.maxRetainedPivotEvidence) + ";bar_duration=" +
        std::to_string(value.completedBarDuration.count());
}

struct Level final
{
    std::string identity;
    PivotKind originatingPivot = PivotKind::high;
    Role currentRole = Role::resistance_like;
    double anchorPrice = 0.0;
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

    bool operator==(const Observation&) const = default;
};

struct Update final
{
    std::chrono::sys_seconds decisionTime{};
    std::vector<Observation> observations;
    // Ascending anchor price, then canonical level identity. This gives a
    // stable representation from which nearest-above/below queries are simple.
    std::vector<Level> activeLevels;

    std::string CanonicalRepresentation() const
    {
        std::string result = "price-level-update-v1;decision=" +
            CanonicalTimestamp(decisionTime) + ";observations=" +
            std::to_string(observations.size()) + ";levels=" +
            std::to_string(activeLevels.size());
        for (const Observation& observation : observations)
            result += ";observation=" + observation.identity;
        for (const Level& level : activeLevels)
            result += ";level=" + level.identity + ";role=" +
                std::string{CanonicalRole(level.currentRole)} + ";anchor=" +
                CanonicalDouble(level.anchorPrice) + ";pivots=" +
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
          sourceProvenance_("causal-price-level-engine-v1;symbol=" + symbol_ +
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
        if (!bars_.empty() && bar.barStart <= bars_.back().barStart)
            throw std::invalid_argument("PRICE_LEVEL_BARS_NOT_STRICTLY_INCREASING");
        bars_.push_back(bar);
        const std::size_t currentBar = nextBar_++;
        const std::chrono::sys_seconds decision = CompletedAt(bar);
        Update result;
        result.decisionTime = decision;

        Expire(currentBar, decision, result.observations);
        if (currentBar > 0)
            ObserveInteractions(currentBar, decision, result.observations);
        ConfirmLatestPivots(decision, result.observations);
        SortObservations(result.observations);
        result.activeLevels = ActiveLevels();
        const std::size_t maximumRetainedBars =
            configuration_.pivotRadiusBars * 2 + 1;
        while (bars_.size() > maximumRetainedBars)
        {
            bars_.erase(bars_.begin());
            ++firstRetainedBar_;
        }
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
    std::vector<CompletedBar> bars_;
    std::vector<InternalLevel> levels_;
    std::size_t firstRetainedBar_ = 0;
    std::size_t nextBar_ = 0;

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
            throw std::invalid_argument("PRICE_LEVEL_BAR_INVALID");
        }
    }

    std::string PivotSourceId(PivotKind kind, std::size_t bar,
                              std::chrono::sys_seconds barStart,
                              double price) const
    {
        return "price-level-pivot-v1;kind=" + std::string{CanonicalPivotKind(kind)} +
            ";bar=" + std::to_string(bar) + ";bar_start=" +
            CanonicalTimestamp(barStart) + ";price=" +
            CanonicalDouble(price);
    }

    std::string LevelIdentity(std::string_view pivotSourceId) const
    {
        return "price-level-v1;symbol=" + symbol_ + ";definition=" +
            configurationIdentity_ + ";origin=" + std::string{pivotSourceId};
    }

    Observation MakeObservation(const Level& level, InteractionKind kind,
                                std::chrono::sys_seconds observed,
                                std::chrono::sys_seconds available,
                                std::string sourceId,
                                std::optional<Role> before = std::nullopt,
                                std::optional<Role> after = std::nullopt) const
    {
        const std::string identity = "price-level-observation-v1;level=" +
            level.identity + ";kind=" + std::string{CanonicalInteractionKind(kind)} +
            ";observed=" + CanonicalTimestamp(observed) + ";available=" +
            CanonicalTimestamp(available) + ";source=" + sourceId + ";before=" +
            (before ? std::string{CanonicalRole(*before)} : "absent") + ";after=" +
            (after ? std::string{CanonicalRole(*after)} : "absent");
        return {identity, level.identity, kind, observed, available,
                sourceProvenance_, std::move(sourceId), before, after};
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
                    "price-level-expiry-v1;bar=" + std::to_string(currentBar)));
                iterator = levels_.erase(iterator);
            }
            else ++iterator;
        }
    }

    void ObserveInteractions(std::size_t currentBar,
                             std::chrono::sys_seconds decision,
                             std::vector<Observation>& observations)
    {
        const CompletedBar& previous = bars_[bars_.size() - 2];
        const CompletedBar& current = bars_.back();
        for (InternalLevel& internal : levels_)
        {
            Level& level = internal.value;
            const bool touching = current.low <= level.upper && current.high >= level.lower;
            const std::string barSource = "price-level-bar-v1;bar=" +
                std::to_string(currentBar) + ";bar_start=" +
                CanonicalTimestamp(current.barStart);
            if (touching && !internal.touching)
            {
                SaturatingIncrement(level.touchCount);
                observations.push_back(MakeObservation(level, InteractionKind::touch,
                    decision, decision, barSource));
            }
            internal.touching = touching;

            const bool crossedUp = previous.close < level.lower && current.close > level.upper;
            const bool crossedDown = previous.close > level.upper && current.close < level.lower;
            if (!crossedUp && !crossedDown)
            {
                if (internal.awaitingRetest && currentBar > internal.crossBar && touching)
                {
                    observations.push_back(MakeObservation(level, InteractionKind::retest,
                        decision, decision, barSource));
                    internal.awaitingRetest = false;
                }
                continue;
            }

            const Role before = level.currentRole;
            const Role after = crossedUp ? Role::support_like : Role::resistance_like;
            const InteractionKind kind = crossedUp ? InteractionKind::cross_up :
                                                    InteractionKind::cross_down;
            observations.push_back(MakeObservation(level, kind, decision, decision,
                barSource, before, after));
            level.currentRole = after;
            internal.awaitingRetest = true;
            internal.crossBar = currentBar;
            if (before != after)
                observations.push_back(MakeObservation(level,
                    InteractionKind::role_reversal, decision, decision, barSource,
                    before, after));
        }
    }

    void ConfirmLatestPivots(std::chrono::sys_seconds decision,
                             std::vector<Observation>& observations)
    {
        const std::size_t radius = configuration_.pivotRadiusBars;
        if (bars_.size() < radius * 2 + 1) return;
        const std::size_t pivotOffset = bars_.size() - 1 - radius;
        const std::size_t pivotBar = firstRetainedBar_ + pivotOffset;
        if (IsStrictPivot(pivotOffset, PivotKind::high))
            AddPivot(PivotKind::high, pivotBar, pivotOffset,
                     bars_[pivotOffset].high, decision,
                     observations);
        if (IsStrictPivot(pivotOffset, PivotKind::low))
            AddPivot(PivotKind::low, pivotBar, pivotOffset,
                     bars_[pivotOffset].low, decision,
                     observations);
    }

    bool IsStrictPivot(std::size_t pivotBar, PivotKind kind) const
    {
        const std::size_t radius = configuration_.pivotRadiusBars;
        const double price = kind == PivotKind::high ? bars_[pivotBar].high :
                                                      bars_[pivotBar].low;
        for (std::size_t index = pivotBar - radius; index <= pivotBar + radius; ++index)
        {
            if (index == pivotBar) continue;
            const double other = kind == PivotKind::high ? bars_[index].high :
                                                          bars_[index].low;
            if (kind == PivotKind::high ? price <= other : price >= other)
                return false;
        }
        return true;
    }

    void AddPivot(PivotKind kind, std::size_t pivotBar, std::size_t pivotOffset,
                  double price,
                  std::chrono::sys_seconds available,
                  std::vector<Observation>& observations)
    {
        const std::string pivotId = PivotSourceId(kind, pivotBar,
            bars_[pivotOffset].barStart, price);
        const auto merge = std::min_element(levels_.begin(), levels_.end(),
            [&price](const InternalLevel& left, const InternalLevel& right) {
                const double leftDistance = std::fabs(left.value.anchorPrice - price);
                const double rightDistance = std::fabs(right.value.anchorPrice - price);
                if (leftDistance != rightDistance) return leftDistance < rightDistance;
                return left.value.identity < right.value.identity;
            });
        if (merge != levels_.end() &&
            std::fabs(merge->value.anchorPrice - price) <= configuration_.zoneHalfWidth)
        {
            Level& level = merge->value;
            SaturatingIncrement(level.pivotObservationCount);
            if (level.retainedPivotEvidence.size() < configuration_.maxRetainedPivotEvidence)
                level.retainedPivotEvidence.push_back(pivotId);
            observations.push_back(MakeObservation(level, InteractionKind::level_reinforced,
            bars_[pivotOffset].barStart + configuration_.completedBarDuration,
                available, pivotId));
            return;
        }

        if (levels_.size() == configuration_.maxActiveLevels)
        {
            const auto evicted = std::min_element(levels_.begin(), levels_.end(),
                [](const InternalLevel& left, const InternalLevel& right) {
                    if (left.value.availableAt != right.value.availableAt)
                        return left.value.availableAt < right.value.availableAt;
                    return left.value.identity < right.value.identity;
                });
            observations.push_back(MakeObservation(evicted->value,
                InteractionKind::level_evicted, available, available,
                "price-level-capacity-eviction-v1;new_pivot=" + pivotId));
            levels_.erase(evicted);
        }

        const Role role = kind == PivotKind::high ? Role::resistance_like :
                                                   Role::support_like;
        const double lower = price - configuration_.zoneHalfWidth;
        const double upper = price + configuration_.zoneHalfWidth;
        if (!std::isfinite(lower) || !std::isfinite(upper))
            throw std::invalid_argument("PRICE_LEVEL_ZONE_NONFINITE");
        Level level{LevelIdentity(pivotId), kind, role, price,
            lower, upper,
            pivotBar, bars_[pivotOffset].barStart + configuration_.completedBarDuration,
            nextBar_ - 1, available, 1, 0, {pivotId}};
        observations.push_back(MakeObservation(level, InteractionKind::level_established,
            level.originObservedAt, available, pivotId));
        levels_.push_back({std::move(level), false, false, 0});
    }

    static void SaturatingIncrement(std::size_t& value)
    {
        if (value != std::numeric_limits<std::size_t>::max()) ++value;
    }
};
} // namespace EA::PriceLevel
