#pragma once

#include "CausalFibonacciExtensionResearch.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace EA::FibonacciResearch::RetracementLifecycle
{

inline constexpr std::string_view kProtocolVersion =
    "causal-fibonacci-retracement-lifecycle-v1";

enum class RetracementLevel
{
    R0382,
    R0500,
    R0618
};

struct Occurrence
{
    std::size_t bar = 0;
    std::int64_t timestamp = 0;
    double price = 0.0;

    bool operator==(const Occurrence&) const = default;
};

struct LevelState
{
    RetracementLevel level = RetracementLevel::R0382;
    double ratio = 0.0;
    double price = 0.0;

    std::optional<Occurrence> reached;
    std::optional<std::size_t> barsBToReached;

    // Candidate rally confirmations. These are observations, not yet
    // declarations of which entry policy is best.
    std::optional<Occurrence> firstDirectionalClose;
    std::optional<Occurrence> firstDirectionalBreak;
    std::optional<Occurrence> firstCloseBackThroughLevel;
};

// Candidate observations remain separate: this research path does not select
// an entry policy before empirical comparison.
struct EntryExcursion
{
    std::optional<Occurrence> entry;
    std::optional<std::size_t> barsBToEntry;
    std::optional<std::size_t> barsRetracementToEntry;
    std::optional<std::size_t> barsEntryToD;
    std::optional<std::size_t> barsEntryToAPenetration;
    std::optional<std::size_t> barsEntryToACloseBeyond;
    std::optional<double> highestHigh;
    std::optional<double> lowestLow;
    std::optional<double> maximumFavorableExcursion;
    std::optional<double> maximumAdverseExcursion;
    std::optional<double> maximumFavorableExcursionABRanges;
    std::optional<double> maximumAdverseExcursionABRanges;
};

struct Record
{
    TG3::ABStructure sourceAB;

    LevelState retracement0382;
    LevelState retracement0500;
    LevelState retracement0618;

    EntryExcursion directionalClose0382;
    EntryExcursion directionalBreak0382;
    EntryExcursion closeBackThrough0382;
    EntryExcursion directionalClose0500;
    EntryExcursion directionalBreak0500;
    EntryExcursion closeBackThrough0500;
    EntryExcursion directionalClose0618;
    EntryExcursion directionalBreak0618;
    EntryExcursion closeBackThrough0618;

    // D is deliberately explicit. The existing research contract does not
    // contain an authoritative D definition.
    double dPrice = 0.0;

    std::optional<Occurrence> firstAPenetration;
    std::optional<Occurrence> firstCloseBeyondA;
    std::optional<Occurrence> firstDReached;

    std::optional<std::size_t> barsBToD;
    std::optional<std::size_t> barsBToAPenetration;
    std::optional<std::size_t> barsBToACloseBeyond;

    bool rightCensored = false;
};

class Tracker
{
public:
    Tracker(TG3::ABStructure sourceAB, double dPrice)
    {
        ValidateAB(sourceAB);

        if (!std::isfinite(dPrice))
            throw std::invalid_argument(
                "Fibonacci retracement lifecycle D price must be finite");

        record_.sourceAB = std::move(sourceAB);
        record_.dPrice = dPrice;

        record_.retracement0382 =
            MakeLevel(RetracementLevel::R0382, kPullback0382);
        record_.retracement0500 =
            MakeLevel(RetracementLevel::R0500, kPullback0500);
        record_.retracement0618 =
            MakeLevel(RetracementLevel::R0618, kPullback0618);
    }

    void AddCompletedBar(std::size_t bar, const TG1A::Candle& candle)
    {
        if (finalized_)
            throw std::logic_error(
                "Fibonacci retracement lifecycle tracker is finalized");

        ValidateCandle(candle);

        if (lastBar_.has_value() && bar != *lastBar_ + 1)
            throw std::invalid_argument(
                "Fibonacci retracement lifecycle bars must be consecutive");

        if (lastTimestamp_.has_value() &&
            candle.timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "Fibonacci retracement lifecycle timestamps must increase");

        lastBar_ = bar;
        lastTimestamp_ = candle.timestamp;

        const auto& identity = record_.sourceAB.identity;

        if (bar < identity.availabilityBar)
            return;

        if (bar == identity.availabilityBar &&
            candle.timestamp != identity.availabilityTimestamp)
            throw std::invalid_argument(
                "Fibonacci retracement lifecycle availability timestamp mismatch");

        if (candle.timestamp < identity.availabilityTimestamp)
            return;

        ObserveStructuralEvents(bar, candle);

        ObserveLevel(record_.retracement0382, bar, candle);
        ObserveLevel(record_.retracement0500, bar, candle);
        ObserveLevel(record_.retracement0618, bar, candle);
        ObserveExcursions(bar, candle);

        previousCandle_ = candle;
    }

    void Finalize()
    {
        if (finalized_)
            return;

        record_.rightCensored = !record_.firstDReached.has_value();
        finalized_ = true;
    }

    const Record& GetRecord() const
    {
        return record_;
    }

    bool IsFinalized() const
    {
        return finalized_;
    }

private:
    Record record_;
    std::optional<TG1A::Candle> previousCandle_;
    std::optional<std::size_t> lastBar_;
    std::optional<std::int64_t> lastTimestamp_;
    bool finalized_ = false;

    bool Up() const
    {
        return record_.sourceAB.identity.direction == TG3::ABDirection::UpAB;
    }

    static void ValidateAB(const TG3::ABStructure& ab)
    {
        if (!std::isfinite(ab.aPrice) ||
            !std::isfinite(ab.bPrice) ||
            !std::isfinite(ab.priceRange) ||
            ab.priceRange <= 0.0)
            throw std::invalid_argument(
                "invalid Fibonacci retracement lifecycle A/B structure");

        if (ab.identity.availabilityBar < ab.identity.bBar ||
            ab.identity.availabilityTimestamp < ab.identity.bTimestamp)
            throw std::invalid_argument(
                "noncausal Fibonacci retracement lifecycle A/B structure");

        if (ab.identity.direction == TG3::ABDirection::UpAB &&
            !(ab.bPrice > ab.aPrice))
            throw std::invalid_argument(
                "UpAB Fibonacci lifecycle requires B above A");

        if (ab.identity.direction == TG3::ABDirection::DownAB &&
            !(ab.bPrice < ab.aPrice))
            throw std::invalid_argument(
                "DownAB Fibonacci lifecycle requires B below A");
    }

    static void ValidateCandle(const TG1A::Candle& candle)
    {
        if (!std::isfinite(candle.open) ||
            !std::isfinite(candle.high) ||
            !std::isfinite(candle.low) ||
            !std::isfinite(candle.close) ||
            candle.high < candle.low ||
            candle.open < candle.low ||
            candle.open > candle.high ||
            candle.close < candle.low ||
            candle.close > candle.high)
            throw std::invalid_argument(
                "invalid completed candle in Fibonacci retracement lifecycle");
    }

    LevelState MakeLevel(RetracementLevel level, double ratio) const
    {
        const double range =
            std::abs(record_.sourceAB.bPrice - record_.sourceAB.aPrice);

        // Retracement is measured from B back toward A.
        const double price = Up()
            ? record_.sourceAB.bPrice - ratio * range
            : record_.sourceAB.bPrice + ratio * range;

        return {level, ratio, price, std::nullopt, std::nullopt,
                std::nullopt, std::nullopt, std::nullopt};
    }

    Occurrence At(std::size_t bar,
                  const TG1A::Candle& candle,
                  double price) const
    {
        return {bar, candle.timestamp, price};
    }

    bool RetracementReached(double levelPrice,
                            const TG1A::Candle& candle) const
    {
        return Up()
            ? candle.low <= levelPrice
            : candle.high >= levelPrice;
    }

    bool DirectionalClose(const TG1A::Candle& candle) const
    {
        return Up()
            ? candle.close > candle.open
            : candle.close < candle.open;
    }

    bool DirectionalBreak(const TG1A::Candle& candle) const
    {
        if (!previousCandle_.has_value())
            return false;

        return Up()
            ? candle.close > previousCandle_->high
            : candle.close < previousCandle_->low;
    }

    bool CloseBackThrough(double levelPrice,
                          const TG1A::Candle& candle) const
    {
        return Up()
            ? candle.close > levelPrice
            : candle.close < levelPrice;
    }

    void ObserveLevel(LevelState& level,
                      std::size_t bar,
                      const TG1A::Candle& candle)
    {
        if (!level.reached.has_value())
        {
            if (!RetracementReached(level.price, candle))
                return;

            level.reached = At(
                bar, candle, Up() ? candle.low : candle.high);
            level.barsBToReached = bar - record_.sourceAB.identity.bBar;

            // Do not permit the same completed candle that first reaches the
            // retracement to confirm a subsequent rally. This preserves
            // temporal ordering and avoids same-bar path assumptions.
            return;
        }

        if (bar <= level.reached->bar)
            return;

        if (!level.firstDirectionalClose.has_value() &&
            DirectionalClose(candle))
            level.firstDirectionalClose =
                At(bar, candle, candle.close);

        if (!level.firstDirectionalBreak.has_value() &&
            DirectionalBreak(candle))
            level.firstDirectionalBreak =
                At(bar, candle, candle.close);

        if (!level.firstCloseBackThroughLevel.has_value() &&
            CloseBackThrough(level.price, candle))
            level.firstCloseBackThroughLevel =
                At(bar, candle, candle.close);
    }

    EntryExcursion& ExcursionFor(LevelState& level, int kind)
    {
        if (&level == &record_.retracement0382)
            return kind == 0 ? record_.directionalClose0382 :
                kind == 1 ? record_.directionalBreak0382 :
                            record_.closeBackThrough0382;
        if (&level == &record_.retracement0500)
            return kind == 0 ? record_.directionalClose0500 :
                kind == 1 ? record_.directionalBreak0500 :
                            record_.closeBackThrough0500;
        return kind == 0 ? record_.directionalClose0618 :
            kind == 1 ? record_.directionalBreak0618 :
                        record_.closeBackThrough0618;
    }

    void ObserveEntry(LevelState& level, EntryExcursion& excursion,
                      const std::optional<Occurrence>& candidate,
                      std::size_t bar, const TG1A::Candle& candle)
    {
        if (!excursion.entry.has_value() && candidate.has_value())
        {
            excursion.entry = candidate;
            excursion.barsBToEntry = candidate->bar -
                record_.sourceAB.identity.bBar;
            if (level.reached.has_value())
                excursion.barsRetracementToEntry = candidate->bar -
                    level.reached->bar;

            // Preserve structural events already observed on the confirmation
            // candle. Their ordering within that candle is unknown, so a zero
            // value means same-candle coincidence, not a proven post-entry
            // outcome.
            if (record_.firstDReached.has_value() &&
                record_.firstDReached->bar == candidate->bar)
                excursion.barsEntryToD = 0;

            if (record_.firstAPenetration.has_value() &&
                record_.firstAPenetration->bar == candidate->bar)
                excursion.barsEntryToAPenetration = 0;

            if (record_.firstCloseBeyondA.has_value() &&
                record_.firstCloseBeyondA->bar == candidate->bar)
                excursion.barsEntryToACloseBeyond = 0;

            // Entry is established by this completed candle's close. Do not
            // use its high/low for MFE/MAE because their intrabar ordering
            // relative to the confirming close is unknowable.
            return;
        }

        // Post-entry MFE/MAE starts with the next completed candle. This avoids
        // inventing intrabar ordering on the confirmation candle.
        if (!excursion.entry.has_value() || bar <= excursion.entry->bar)
            return;

        excursion.highestHigh = !excursion.highestHigh.has_value()
            ? candle.high : std::max(*excursion.highestHigh, candle.high);
        excursion.lowestLow = !excursion.lowestLow.has_value()
            ? candle.low : std::min(*excursion.lowestLow, candle.low);
        const double favorable = Up() ? candle.high - excursion.entry->price
                                      : excursion.entry->price - candle.low;
        const double adverse = Up() ? excursion.entry->price - candle.low
                                    : candle.high - excursion.entry->price;
        excursion.maximumFavorableExcursion =
            !excursion.maximumFavorableExcursion.has_value()
                ? favorable : std::max(*excursion.maximumFavorableExcursion,
                                       favorable);
        excursion.maximumAdverseExcursion =
            !excursion.maximumAdverseExcursion.has_value()
                ? adverse : std::max(*excursion.maximumAdverseExcursion,
                                     adverse);
        excursion.maximumFavorableExcursionABRanges =
            *excursion.maximumFavorableExcursion / record_.sourceAB.priceRange;
        excursion.maximumAdverseExcursionABRanges =
            *excursion.maximumAdverseExcursion / record_.sourceAB.priceRange;
        if (!excursion.barsEntryToD.has_value() &&
            record_.firstDReached.has_value() &&
            record_.firstDReached->bar >= excursion.entry->bar)
            excursion.barsEntryToD = record_.firstDReached->bar -
                excursion.entry->bar;
        if (!excursion.barsEntryToAPenetration.has_value() &&
            record_.firstAPenetration.has_value() &&
            record_.firstAPenetration->bar >= excursion.entry->bar)
            excursion.barsEntryToAPenetration = record_.firstAPenetration->bar -
                excursion.entry->bar;
        if (!excursion.barsEntryToACloseBeyond.has_value() &&
            record_.firstCloseBeyondA.has_value() &&
            record_.firstCloseBeyondA->bar >= excursion.entry->bar)
            excursion.barsEntryToACloseBeyond =
                record_.firstCloseBeyondA->bar - excursion.entry->bar;
    }

    void ObserveExcursions(std::size_t bar, const TG1A::Candle& candle)
    {
        for (LevelState* level : {&record_.retracement0382,
                                  &record_.retracement0500,
                                  &record_.retracement0618})
        {
            ObserveEntry(*level, ExcursionFor(*level, 0),
                         level->firstDirectionalClose,
                         bar, candle);
            ObserveEntry(*level, ExcursionFor(*level, 1),
                         level->firstDirectionalBreak,
                         bar, candle);
            ObserveEntry(*level, ExcursionFor(*level, 2),
                         level->firstCloseBackThroughLevel, bar, candle);
        }
    }

    void ObserveStructuralEvents(std::size_t bar,
                                 const TG1A::Candle& candle)
    {
        const double a = record_.sourceAB.aPrice;

        if (!record_.firstAPenetration.has_value())
        {
            const bool penetrated = Up()
                ? candle.low < a
                : candle.high > a;

            if (penetrated)
            {
                record_.firstAPenetration =
                    At(bar, candle, Up() ? candle.low : candle.high);

                if (bar >= record_.sourceAB.identity.bBar)
                    record_.barsBToAPenetration =
                        bar - record_.sourceAB.identity.bBar;
            }
        }

        if (!record_.firstCloseBeyondA.has_value())
        {
            const bool closedBeyond = Up()
                ? candle.close < a
                : candle.close > a;

            if (closedBeyond)
            {
                record_.firstCloseBeyondA =
                    At(bar, candle, candle.close);

                if (bar >= record_.sourceAB.identity.bBar)
                    record_.barsBToACloseBeyond =
                        bar - record_.sourceAB.identity.bBar;
            }
        }

        if (!record_.firstDReached.has_value())
        {
            const bool reached = Up()
                ? candle.high >= record_.dPrice
                : candle.low <= record_.dPrice;

            if (reached)
            {
                record_.firstDReached =
                    At(bar, candle, Up() ? candle.high : candle.low);

                if (bar >= record_.sourceAB.identity.bBar)
                    record_.barsBToD =
                        bar - record_.sourceAB.identity.bBar;
            }
        }
    }
};

} // namespace EA::FibonacciResearch::RetracementLifecycle
