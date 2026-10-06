#pragma once

#include "CausalFibonacciRetracementLifecycle.hpp"
#include "CausalFibonacciStructuralFeatureConfiguration.hpp"
#include "CanonicalSymbol.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace EA::CausalFibonacciLifecycleFeatures
{

inline constexpr std::size_t kFeatureCount = 44;

// Layout 13 deliberately exposes causal lifecycle observations rather than
// an outcome-tuned interpretation of those observations. Counts use log1p.
// Unbounded causal ages also use log1p; zero means no qualifying observation,
// with the corresponding count channel providing the availability signal.
enum Column : std::size_t
{
    Up0382ReachedCountLog = 0,
    Up0382DirectionalCloseCountLog,
    Up0382DirectionalBreakCountLog,
    Up0382CloseBackThroughCountLog,
    Up0382ReachedYoungestAgeLog,
    Up0382DirectionalCloseYoungestAgeLog,
    Up0500ReachedCountLog,
    Up0500DirectionalCloseCountLog,
    Up0500DirectionalBreakCountLog,
    Up0500CloseBackThroughCountLog,
    Up0500ReachedYoungestAgeLog,
    Up0500DirectionalCloseYoungestAgeLog,
    Up0618ReachedCountLog,
    Up0618DirectionalCloseCountLog,
    Up0618DirectionalBreakCountLog,
    Up0618CloseBackThroughCountLog,
    Up0618ReachedYoungestAgeLog,
    Up0618DirectionalCloseYoungestAgeLog,
    UpAPenetrationCountLog,
    UpACloseBeyondCountLog,
    UpAPenetrationYoungestAgeLog,
    UpACloseBeyondYoungestAgeLog,

    Down0382ReachedCountLog,
    Down0382DirectionalCloseCountLog,
    Down0382DirectionalBreakCountLog,
    Down0382CloseBackThroughCountLog,
    Down0382ReachedYoungestAgeLog,
    Down0382DirectionalCloseYoungestAgeLog,
    Down0500ReachedCountLog,
    Down0500DirectionalCloseCountLog,
    Down0500DirectionalBreakCountLog,
    Down0500CloseBackThroughCountLog,
    Down0500ReachedYoungestAgeLog,
    Down0500DirectionalCloseYoungestAgeLog,
    Down0618ReachedCountLog,
    Down0618DirectionalCloseCountLog,
    Down0618DirectionalBreakCountLog,
    Down0618CloseBackThroughCountLog,
    Down0618ReachedYoungestAgeLog,
    Down0618DirectionalCloseYoungestAgeLog,
    DownAPenetrationCountLog,
    DownACloseBeyondCountLog,
    DownAPenetrationYoungestAgeLog,
    DownACloseBeyondYoungestAgeLog,
};

static_assert(DownACloseBeyondYoungestAgeLog + 1 == kFeatureCount);

inline float CountTransform(std::size_t count)
{
    return static_cast<float>(std::log1p(static_cast<double>(count)));
}

inline float AgeTransform(const std::optional<std::size_t>& age)
{
    return age.has_value()
        ? static_cast<float>(std::log1p(static_cast<double>(*age)))
        : 0.0f;
}

inline void ObserveYoungest(std::optional<std::size_t>& youngest,
                            const std::optional<FibonacciResearch::
                                RetracementLifecycle::Occurrence>& occurrence,
                            std::size_t currentBar)
{
    if (!occurrence.has_value()) return;
    if (occurrence->bar > currentBar)
        throw std::logic_error(
            "CAUSAL_FIBONACCI_LIFECYCLE_FUTURE_OCCURRENCE");
    const std::size_t age = currentBar - occurrence->bar;
    youngest = youngest.has_value() ? std::min(*youngest, age) : age;
}

struct LevelAggregate
{
    std::size_t reached = 0;
    std::size_t directionalClose = 0;
    std::size_t directionalBreak = 0;
    std::size_t closeBackThrough = 0;
    std::optional<std::size_t> youngestReached;
    std::optional<std::size_t> youngestDirectionalClose;
};

struct DirectionAggregate
{
    std::array<LevelAggregate, 3> levels{};
    std::size_t aPenetration = 0;
    std::size_t aCloseBeyond = 0;
    std::optional<std::size_t> youngestAPenetration;
    std::optional<std::size_t> youngestACloseBeyond;
};

inline std::array<float, kFeatureCount> Aggregate(
    const std::vector<FibonacciResearch::RetracementLifecycle::Record>& records,
    std::size_t currentBar)
{
    using namespace FibonacciResearch::RetracementLifecycle;
    std::array<DirectionAggregate, 2> directions{};

    for (const Record& record : records)
    {
        const bool up =
            record.sourceAB.identity.direction == TG3::ABDirection::UpAB;
        DirectionAggregate& aggregate = directions[up ? 0 : 1];

        const std::array<const LevelState*, 3> levels{{
            &record.retracement0382,
            &record.retracement0500,
            &record.retracement0618,
        }};

        for (std::size_t index = 0; index < levels.size(); ++index)
        {
            const LevelState& level = *levels[index];
            LevelAggregate& output = aggregate.levels[index];

            if (level.reached.has_value())
            {
                ++output.reached;
                ObserveYoungest(output.youngestReached, level.reached, currentBar);
            }
            if (level.firstDirectionalClose.has_value())
            {
                ++output.directionalClose;
                ObserveYoungest(output.youngestDirectionalClose,
                                level.firstDirectionalClose, currentBar);
            }
            if (level.firstDirectionalBreak.has_value())
                ++output.directionalBreak;
            if (level.firstCloseBackThroughLevel.has_value())
                ++output.closeBackThrough;
        }

        if (record.firstAPenetration.has_value())
        {
            ++aggregate.aPenetration;
            ObserveYoungest(aggregate.youngestAPenetration,
                            record.firstAPenetration, currentBar);
        }
        if (record.firstCloseBeyondA.has_value())
        {
            ++aggregate.aCloseBeyond;
            ObserveYoungest(aggregate.youngestACloseBeyond,
                            record.firstCloseBeyondA, currentBar);
        }
    }

    std::array<float, kFeatureCount> result{};
    for (std::size_t direction = 0; direction < directions.size(); ++direction)
    {
        const DirectionAggregate& input = directions[direction];
        const std::size_t base = direction == 0 ? Up0382ReachedCountLog
                                                 : Down0382ReachedCountLog;
        std::size_t offset = base;
        for (const LevelAggregate& level : input.levels)
        {
            result[offset++] = CountTransform(level.reached);
            result[offset++] = CountTransform(level.directionalClose);
            result[offset++] = CountTransform(level.directionalBreak);
            result[offset++] = CountTransform(level.closeBackThrough);
            result[offset++] = AgeTransform(level.youngestReached);
            result[offset++] = AgeTransform(level.youngestDirectionalClose);
        }
        result[offset++] = CountTransform(input.aPenetration);
        result[offset++] = CountTransform(input.aCloseBeyond);
        result[offset++] = AgeTransform(input.youngestAPenetration);
        result[offset++] = AgeTransform(input.youngestACloseBeyond);
    }
    return result;
}

class Producer final
{
public:
    explicit Producer(std::string symbol)
        : symbol_(CanonicalSymbol::Normalize(symbol)),
          geometry_(configuration_.geometry(),
                    {symbol_, std::string{configuration_.timeframe()}}),
          tracker_(configuration_.FibonacciConfigurationForSymbol(symbol_),
                   {symbol_, std::string{configuration_.timeframe()}})
    {
    }

    std::array<float, kFeatureCount> AddCompletedBar(
        const TG1A::Candle& candle)
    {
        using namespace FibonacciResearch;
        using namespace FibonacciResearch::RetracementLifecycle;

        const auto update = geometry_.AddCompletedBar(candle);
        tracker_.Advance(update.bar, candle.timestamp);
        const auto created =
            tracker_.ObserveConfirmedFractals(update.newlyConfirmedFractals);

        const double tolerance =
            configuration_.FibonacciConfigurationForSymbol(symbol_)
                .absolutePriceTolerance;

        for (const TG3::ABIdentity& identity : created)
        {
            const auto found = std::find_if(
                tracker_.ABStructures().begin(), tracker_.ABStructures().end(),
                [&identity](const TG3::ABStructure& ab)
                {
                    return ab.identity == identity;
                });
            if (found == tracker_.ABStructures().end())
                throw std::logic_error(
                    "CAUSAL_FIBONACCI_LIFECYCLE_NEW_AB_NOT_RETAINED");

            // Layout 13 freezes the lifecycle target hypothesis at the 1.272
            // extension. This matches the audited retracement population and
            // makes D terminal without inventing an arbitrary age timeout.
            const double dPrice = CalculateExtensionLevel(
                *found, kExtension1272, tolerance).price;
            states_.push_back({identity, Tracker(*found, dPrice)});
        }

        for (State& state : states_)
            state.tracker.AddCompletedBar(update.bar, candle);

        std::vector<Record> active;
        active.reserve(states_.size());
        for (const State& state : states_)
        {
            const Record& record = state.tracker.GetRecord();
            // The first D candle remains observable. Later completed bars do
            // not expose this terminal lifecycle candidate.
            if (!record.firstDReached.has_value() ||
                record.firstDReached->bar == update.bar)
            {
                active.push_back(record);
            }
        }

        const auto result = Aggregate(active, update.bar);

        // Retention is independent of TG3 age/capacity pruning. Only the
        // lifecycle's natural D terminal removes a state, after its D candle
        // has contributed to the just-completed row.
        states_.erase(
            std::remove_if(states_.begin(), states_.end(),
                [bar = update.bar](const State& state)
                {
                    const auto& d = state.tracker.GetRecord().firstDReached;
                    return d.has_value() && d->bar <= bar;
                }),
            states_.end());

        return result;
    }

private:
    struct State
    {
        TG3::ABIdentity identity;
        FibonacciResearch::RetracementLifecycle::Tracker tracker;
    };

    std::string symbol_;
    CausalFibonacciStructuralFeatureConfiguration::Configuration configuration_;
    TG1A::CausalFractalTrendLineGeometry geometry_;
    TG3::FibonacciConfluenceTracker tracker_;
    std::vector<State> states_;
};

} // namespace EA::CausalFibonacciLifecycleFeatures
