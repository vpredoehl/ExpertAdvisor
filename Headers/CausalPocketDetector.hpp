#ifndef CausalPocketDetector_hpp
#define CausalPocketDetector_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>

namespace EA::Pocket
{

// This model implements the frozen Phase Pocket 2 project operational
// definition.  It is deliberately not an execution, profitability, or
// lifecycle model.
enum class PocketDirection
{
    Bullish,
    Bearish
};

struct CompletedBar
{
    std::int64_t timestamp = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
};

struct PocketPriceRange
{
    double lower = 0.0;
    double upper = 0.0;
};

struct PocketObservation
{
    PocketDirection direction = PocketDirection::Bullish;
    PocketPriceRange range;
    std::size_t eventBar = 0;
    std::int64_t eventTimestamp = 0;
    std::size_t confirmationBar = 0;
    std::int64_t confirmationTimestamp = 0;
    std::size_t informationCutoffBar = 0;
    std::int64_t informationCutoffTimestamp = 0;
    std::string sourceTimeframe;

    double TouchPrice() const
    {
        return direction == PocketDirection::Bullish ? range.upper : range.lower;
    }

    double ClosePrice() const
    {
        return direction == PocketDirection::Bullish ? range.lower : range.upper;
    }
};

class CausalPocketDetector
{
public:
    static constexpr std::size_t kSourceDefaultLookbackBars = 15;
    static constexpr std::size_t kConfirmationLatencyBars = 1;

    explicit CausalPocketDetector(std::string sourceTimeframe)
        : sourceTimeframe_(std::move(sourceTimeframe))
    {
        if (sourceTimeframe_.empty())
            throw std::invalid_argument("Pocket source timeframe must be non-empty");
    }

    std::optional<PocketObservation> AddCompletedBar(const CompletedBar& bar)
    {
        ValidateBar(bar);
        if (lastTimestamp_.has_value() && bar.timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "Pocket completed bars must have unique increasing timestamps");

        const std::size_t barIndex = nextBar_++;
        lastTimestamp_ = bar.timestamp;
        history_.push_back({barIndex, bar});
        while (history_.size() > kRequiredHistoryBars)
            history_.pop_front();

        if (history_.size() != kRequiredHistoryBars)
            return std::nullopt;
        return DetectLatest();
    }

    const std::string& SourceTimeframe() const { return sourceTimeframe_; }
    std::size_t CompletedBarCount() const { return nextBar_; }

private:
    static constexpr std::size_t kRequiredHistoryBars =
        kSourceDefaultLookbackBars + 1 + kConfirmationLatencyBars;

    struct IndexedBar
    {
        std::size_t index = 0;
        CompletedBar bar;
    };

    std::string sourceTimeframe_;
    std::deque<IndexedBar> history_;
    std::optional<std::int64_t> lastTimestamp_;
    std::size_t nextBar_ = 0;

    static void ValidateBar(const CompletedBar& bar)
    {
        if (!std::isfinite(bar.open) || !std::isfinite(bar.high) ||
            !std::isfinite(bar.low) || !std::isfinite(bar.close) ||
            bar.high < bar.low || bar.open < bar.low || bar.open > bar.high ||
            bar.close < bar.low || bar.close > bar.high)
            throw std::invalid_argument("Pocket received an invalid completed bar");
    }

    std::optional<PocketObservation> DetectLatest() const
    {
        double referenceHigh = history_.front().bar.high;
        double referenceLow = history_.front().bar.low;
        for (std::size_t index = 1; index < kSourceDefaultLookbackBars; ++index)
        {
            referenceHigh = std::max(referenceHigh, history_[index].bar.high);
            referenceLow = std::min(referenceLow, history_[index].bar.low);
        }

        const IndexedBar& event = history_[kSourceDefaultLookbackBars];
        const IndexedBar& confirmation = history_.back();
        if (event.bar.close > event.bar.open && event.bar.close > referenceHigh &&
            confirmation.bar.low > referenceHigh)
            return MakeObservation(PocketDirection::Bullish, referenceHigh,
                confirmation.bar.low, event, confirmation);

        if (event.bar.close < event.bar.open && event.bar.close < referenceLow &&
            confirmation.bar.high < referenceLow)
            return MakeObservation(PocketDirection::Bearish, confirmation.bar.high,
                referenceLow, event, confirmation);

        return std::nullopt;
    }

    PocketObservation MakeObservation(PocketDirection direction, double lower,
        double upper, const IndexedBar& event, const IndexedBar& confirmation) const
    {
        return {direction, {lower, upper}, event.index, event.bar.timestamp,
            confirmation.index, confirmation.bar.timestamp, confirmation.index,
            confirmation.bar.timestamp, sourceTimeframe_};
    }
};

} // namespace EA::Pocket

#endif /* CausalPocketDetector_hpp */
