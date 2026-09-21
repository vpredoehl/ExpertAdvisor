#ifndef CausalFractalTrendLineGeometry_hpp
#define CausalFractalTrendLineGeometry_hpp

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <deque>
#include <iomanip>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace EA::TG1A
{

enum class FractalKind
{
    High,
    Low
};

enum class TrendLineDirection
{
    UTL,
    DTL
};

struct Candle
{
    std::int64_t timestamp = 0;
    double open = 0.0;
    double high = 0.0;
    double low = 0.0;
    double close = 0.0;
};

struct ConfirmedFractal
{
    FractalKind kind = FractalKind::High;
    std::size_t anchorBar = 0;
    std::int64_t anchorTimestamp = 0;
    double price = 0.0;
    std::size_t confirmationBar = 0;
    std::int64_t confirmationTimestamp = 0;
};

struct TrendLineCandidate
{
    TrendLineDirection direction = TrendLineDirection::UTL;
    std::size_t anchor1Bar = 0;
    std::int64_t anchor1Timestamp = 0;
    double anchor1Price = 0.0;
    std::size_t anchor1ConfirmationBar = 0;
    std::int64_t anchor1ConfirmationTimestamp = 0;
    std::size_t anchor2Bar = 0;
    std::int64_t anchor2Timestamp = 0;
    double anchor2Price = 0.0;
    std::size_t anchor2ConfirmationBar = 0;
    std::int64_t anchor2ConfirmationTimestamp = 0;
    std::size_t anchorSeparationBars = 0;
    double rawPriceSlopePerBar = 0.0;
    std::size_t creationBar = 0;
    std::int64_t creationTimestamp = 0;
    std::size_t candidateAgeBars = 0;
    std::size_t currentBar = 0;
    std::int64_t currentTimestamp = 0;
    double projectedLinePriceAtCurrentBar = 0.0;
    double rawPriceToLineDistance = 0.0;
    std::optional<double> currentAtr;
    std::optional<double> atrNormalizedSlope;
    std::optional<double> atrNormalizedPriceToLineDistance;
    std::size_t candleTouchCount = 0;
    std::size_t fractalTouchCount = 0;

    double ProjectedPrice(std::size_t bar) const
    {
        const double offset = static_cast<double>(bar) -
            static_cast<double>(anchor1Bar);
        return anchor1Price + rawPriceSlopePerBar * offset;
    }
};

struct Configuration
{
    // Absolute price-unit tolerances.  Zero means exact geometry.
    double interveningPriceTolerance = 0.0;
    double touchPriceTolerance = 0.0;

    // A new anchor can pair only with confirmed fractals in this bar horizon.
    std::size_t maxFractalAnchorLookbackBars = 512;
    std::size_t maxConfirmedFractalsPerKind = 64;

    // Candidate age is measured from anchor2 confirmation/creation.
    std::size_t maxCandidateAgeBars = 512;
    std::size_t maxCandidates = 4096;

    // Wilder recursive ATR period used only for normalized diagnostics.
    std::size_t atrPeriod = 14;
};

struct SeriesIdentity
{
    std::string symbol;
    std::string timeframe;
};

struct Update
{
    std::size_t bar = 0;
    std::vector<ConfirmedFractal> newlyConfirmedFractals;
    std::vector<TrendLineCandidate> newlyCreatedCandidates;
};

class CausalFractalTrendLineGeometry
{
public:
    explicit CausalFractalTrendLineGeometry(
        Configuration configuration = {}, SeriesIdentity identity = {})
        : configuration_(configuration), identity_(std::move(identity))
    {
        ValidateConfiguration();
    }

    static CausalFractalTrendLineGeometry FromHistorical(
        std::vector<Candle> candles,
        Configuration configuration = {},
        SeriesIdentity identity = {})
    {
        std::stable_sort(candles.begin(), candles.end(),
            [](const Candle& left, const Candle& right)
            {
                return left.timestamp < right.timestamp;
            });
        CausalFractalTrendLineGeometry result(
            configuration, std::move(identity));
        for (const Candle& candle : candles) result.AddCompletedBar(candle);
        return result;
    }

    Update AddCompletedBar(const Candle& candle)
    {
        ValidateCandle(candle);
        if (lastTimestamp_.has_value() && candle.timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "TG1A completed bars must have unique increasing timestamps");

        const std::size_t currentBar = nextBar_++;
        lastTimestamp_ = candle.timestamp;
        bars_.push_back({currentBar, candle});
        UpdateAtr(candle);

        ExpireCandidates(currentBar);
        UpdateExistingCandidateCandleTouches(currentBar, candle);

        Update update;
        update.bar = currentBar;
        if (bars_.size() >= kFractalWindowBars)
            update.newlyConfirmedFractals = DetectLatestFractals(currentBar);

        for (const ConfirmedFractal& fractal : update.newlyConfirmedFractals)
        {
            UpdateExistingCandidateFractalTouches(fractal);
            std::vector<TrendLineCandidate> created =
                AddFractalAndCreateCandidates(fractal, currentBar, candle);
            update.newlyCreatedCandidates.insert(
                update.newlyCreatedCandidates.end(),
                created.begin(), created.end());
        }

        UpdateCurrentGeometry(currentBar, candle);
        EnforceCandidateBound();
        SortCandidates(candidates_);
        PruneStoredHistory(currentBar);
        return update;
    }

    const std::vector<TrendLineCandidate>& Candidates() const
    {
        return candidates_;
    }

    std::vector<ConfirmedFractal> ConfirmedFractals() const
    {
        std::vector<ConfirmedFractal> result;
        result.reserve(highFractals_.size() + lowFractals_.size());
        result.insert(result.end(), highFractals_.begin(), highFractals_.end());
        result.insert(result.end(), lowFractals_.begin(), lowFractals_.end());
        std::sort(result.begin(), result.end(), FractalLess);
        return result;
    }

    const Configuration& GetConfiguration() const { return configuration_; }
    const SeriesIdentity& GetSeriesIdentity() const { return identity_; }
    std::size_t CompletedBarCount() const { return nextBar_; }

    std::string FormatDiagnostic(const TrendLineCandidate& candidate) const
    {
        std::ostringstream output;
        output << std::setprecision(12)
               << "symbol=" << identity_.symbol
               << ",timeframe=" << identity_.timeframe
               << ",direction=" << DirectionName(candidate.direction)
               << ",anchor1_bar=" << candidate.anchor1Bar
               << ",anchor1_timestamp="
               << FormatTimestamp(candidate.anchor1Timestamp)
               << ",anchor1_price=" << candidate.anchor1Price
               << ",anchor1_confirmation_timestamp="
               << FormatTimestamp(candidate.anchor1ConfirmationTimestamp)
               << ",anchor2_bar=" << candidate.anchor2Bar
               << ",anchor2_timestamp="
               << FormatTimestamp(candidate.anchor2Timestamp)
               << ",anchor2_price=" << candidate.anchor2Price
               << ",anchor2_confirmation_timestamp="
               << FormatTimestamp(candidate.anchor2ConfirmationTimestamp)
               << ",creation_timestamp="
               << FormatTimestamp(candidate.creationTimestamp)
               << ",separation_bars=" << candidate.anchorSeparationBars
               << ",raw_slope_per_bar=" << candidate.rawPriceSlopePerBar
               << ",atr_normalized_slope="
               << OptionalNumber(candidate.atrNormalizedSlope)
               << ",age_bars=" << candidate.candidateAgeBars
               << ",candle_touch_count=" << candidate.candleTouchCount
               << ",fractal_touch_count=" << candidate.fractalTouchCount
               << ",current_bar=" << candidate.currentBar
               << ",current_timestamp="
               << FormatTimestamp(candidate.currentTimestamp)
               << ",projected_price="
               << candidate.projectedLinePriceAtCurrentBar
               << ",raw_price_to_line_distance="
               << candidate.rawPriceToLineDistance
               << ",atr_normalized_distance="
               << OptionalNumber(
                      candidate.atrNormalizedPriceToLineDistance);
        return output.str();
    }

private:
    static constexpr std::size_t kFractalRadius = 2;
    static constexpr std::size_t kFractalWindowBars = 5;

    struct IndexedCandle
    {
        std::size_t bar = 0;
        Candle candle;
    };

    Configuration configuration_;
    SeriesIdentity identity_;
    std::deque<IndexedCandle> bars_;
    std::deque<ConfirmedFractal> highFractals_;
    std::deque<ConfirmedFractal> lowFractals_;
    std::vector<TrendLineCandidate> candidates_;
    std::optional<double> currentAtr_;
    double atrState_ = 0.0;
    bool hasAtrState_ = false;
    std::optional<double> previousClose_;
    std::optional<std::int64_t> lastTimestamp_;
    std::size_t nextBar_ = 0;

    static bool FiniteNonnegative(double value)
    {
        return std::isfinite(value) && value >= 0.0;
    }

    void ValidateConfiguration() const
    {
        if (!FiniteNonnegative(configuration_.interveningPriceTolerance) ||
            !FiniteNonnegative(configuration_.touchPriceTolerance))
            throw std::invalid_argument("TG1A tolerances must be finite and nonnegative");
        if (configuration_.maxFractalAnchorLookbackBars <
                kFractalWindowBars - 1 ||
            configuration_.maxConfirmedFractalsPerKind == 0 ||
            configuration_.maxCandidateAgeBars == 0 ||
            configuration_.maxCandidates == 0 ||
            configuration_.atrPeriod == 0)
            throw std::invalid_argument("TG1A bounds and ATR period must be positive");
    }

    static void ValidateCandle(const Candle& candle)
    {
        if (!std::isfinite(candle.open) || !std::isfinite(candle.high) ||
            !std::isfinite(candle.low) || !std::isfinite(candle.close) ||
            candle.high < candle.low || candle.open < candle.low ||
            candle.open > candle.high || candle.close < candle.low ||
            candle.close > candle.high)
            throw std::invalid_argument("TG1A received an invalid completed candle");
    }

    void UpdateAtr(const Candle& candle)
    {
        double trueRange = candle.high - candle.low;
        if (previousClose_.has_value())
        {
            trueRange = std::max(
                {trueRange, std::fabs(candle.high - *previousClose_),
                 std::fabs(candle.low - *previousClose_)});
        }
        previousClose_ = candle.close;
        const double alpha = 1.0 /
            static_cast<double>(configuration_.atrPeriod);
        if (!hasAtrState_)
        {
            atrState_ = trueRange;
            hasAtrState_ = true;
        }
        else
            atrState_ = alpha * trueRange + (1.0 - alpha) * atrState_;
        if (std::isfinite(atrState_) && atrState_ > 0.0)
            currentAtr_ = atrState_;
        else
            currentAtr_.reset();
    }

    std::vector<ConfirmedFractal> DetectLatestFractals(
        std::size_t confirmationBar) const
    {
        const std::size_t start = bars_.size() - kFractalWindowBars;
        const IndexedCandle& center = bars_[start + kFractalRadius];
        bool strictHigh = true;
        bool strictLow = true;
        for (std::size_t offset = 0; offset < kFractalWindowBars; ++offset)
        {
            if (offset == kFractalRadius) continue;
            const Candle& neighbor = bars_[start + offset].candle;
            strictHigh = strictHigh && center.candle.high > neighbor.high;
            strictLow = strictLow && center.candle.low < neighbor.low;
        }

        std::vector<ConfirmedFractal> result;
        if (strictHigh)
            result.push_back({FractalKind::High, center.bar,
                center.candle.timestamp, center.candle.high,
                confirmationBar, bars_.back().candle.timestamp});
        if (strictLow)
            result.push_back({FractalKind::Low, center.bar,
                center.candle.timestamp, center.candle.low,
                confirmationBar, bars_.back().candle.timestamp});
        return result;
    }

    std::vector<TrendLineCandidate> AddFractalAndCreateCandidates(
        const ConfirmedFractal& fractal,
        std::size_t currentBar,
        const Candle& currentCandle)
    {
        std::deque<ConfirmedFractal>& sameKind =
            fractal.kind == FractalKind::High ? highFractals_ : lowFractals_;
        PruneFractals(sameKind, currentBar);
        while (sameKind.size() >= configuration_.maxConfirmedFractalsPerKind)
            sameKind.pop_front();
        const std::vector<ConfirmedFractal> prior(
            sameKind.begin(), sameKind.end());
        sameKind.push_back(fractal);

        std::vector<TrendLineCandidate> created;
        for (const ConfirmedFractal& first : prior)
        {
            const bool pricesQualify = fractal.kind == FractalKind::Low
                ? fractal.price > first.price
                : fractal.price < first.price;
            if (!pricesQualify) continue;

            const TrendLineDirection direction =
                fractal.kind == FractalKind::Low
                ? TrendLineDirection::UTL : TrendLineDirection::DTL;
            TrendLineCandidate candidate = MakeCandidate(
                direction, first, fractal, currentBar, currentCandle);
            if (!InterveningPriceIsValid(candidate)) continue;
            InitializeTouchCounts(candidate);
            candidates_.push_back(candidate);
            created.push_back(candidate);
        }
        return created;
    }

    TrendLineCandidate MakeCandidate(
        TrendLineDirection direction,
        const ConfirmedFractal& first,
        const ConfirmedFractal& second,
        std::size_t currentBar,
        const Candle& currentCandle) const
    {
        TrendLineCandidate candidate;
        candidate.direction = direction;
        candidate.anchor1Bar = first.anchorBar;
        candidate.anchor1Timestamp = first.anchorTimestamp;
        candidate.anchor1Price = first.price;
        candidate.anchor1ConfirmationBar = first.confirmationBar;
        candidate.anchor1ConfirmationTimestamp = first.confirmationTimestamp;
        candidate.anchor2Bar = second.anchorBar;
        candidate.anchor2Timestamp = second.anchorTimestamp;
        candidate.anchor2Price = second.price;
        candidate.anchor2ConfirmationBar = second.confirmationBar;
        candidate.anchor2ConfirmationTimestamp = second.confirmationTimestamp;
        candidate.anchorSeparationBars = second.anchorBar - first.anchorBar;
        candidate.rawPriceSlopePerBar =
            (second.price - first.price) /
            static_cast<double>(candidate.anchorSeparationBars);
        candidate.creationBar = second.confirmationBar;
        candidate.creationTimestamp = second.confirmationTimestamp;
        SetCurrentGeometry(candidate, currentBar, currentCandle);
        return candidate;
    }

    bool InterveningPriceIsValid(
        const TrendLineCandidate& candidate) const
    {
        for (const IndexedCandle& entry : bars_)
        {
            if (entry.bar <= candidate.anchor1Bar ||
                entry.bar >= candidate.anchor2Bar)
                continue;
            const double line = candidate.ProjectedPrice(entry.bar);
            if (candidate.direction == TrendLineDirection::UTL)
            {
                if (entry.candle.low +
                        configuration_.interveningPriceTolerance < line)
                    return false;
            }
            else if (entry.candle.high -
                         configuration_.interveningPriceTolerance > line)
                return false;
        }
        return true;
    }

    void InitializeTouchCounts(TrendLineCandidate& candidate) const
    {
        candidate.candleTouchCount = 0;
        for (const IndexedCandle& entry : bars_)
        {
            if (entry.bar < candidate.anchor1Bar ||
                entry.bar > candidate.currentBar ||
                entry.bar == candidate.anchor1Bar ||
                entry.bar == candidate.anchor2Bar)
                continue;
            if (CandleTouches(candidate, entry.bar, entry.candle))
                ++candidate.candleTouchCount;
        }

        candidate.fractalTouchCount = 0;
        const std::deque<ConfirmedFractal>& fractals =
            candidate.direction == TrendLineDirection::UTL
            ? lowFractals_ : highFractals_;
        for (const ConfirmedFractal& fractal : fractals)
            if (fractal.anchorBar >= candidate.anchor1Bar &&
                fractal.confirmationBar <= candidate.currentBar &&
                FractalTouches(candidate, fractal))
                ++candidate.fractalTouchCount;
    }

    bool CandleTouches(const TrendLineCandidate& candidate,
                       std::size_t bar,
                       const Candle& candle) const
    {
        const double price = candidate.direction == TrendLineDirection::UTL
            ? candle.low : candle.high;
        return std::fabs(price - candidate.ProjectedPrice(bar)) <=
            configuration_.touchPriceTolerance;
    }

    bool FractalTouches(const TrendLineCandidate& candidate,
                        const ConfirmedFractal& fractal) const
    {
        const FractalKind expected =
            candidate.direction == TrendLineDirection::UTL
            ? FractalKind::Low : FractalKind::High;
        return fractal.kind == expected &&
            std::fabs(fractal.price -
                      candidate.ProjectedPrice(fractal.anchorBar)) <=
                configuration_.touchPriceTolerance;
    }

    void UpdateExistingCandidateCandleTouches(
        std::size_t currentBar, const Candle& candle)
    {
        for (TrendLineCandidate& candidate : candidates_)
            if (currentBar != candidate.anchor1Bar &&
                currentBar != candidate.anchor2Bar &&
                CandleTouches(candidate, currentBar, candle))
                ++candidate.candleTouchCount;
    }

    void UpdateExistingCandidateFractalTouches(
        const ConfirmedFractal& fractal)
    {
        for (TrendLineCandidate& candidate : candidates_)
            if (FractalTouches(candidate, fractal))
                ++candidate.fractalTouchCount;
    }

    void UpdateCurrentGeometry(std::size_t currentBar, const Candle& candle)
    {
        for (TrendLineCandidate& candidate : candidates_)
            SetCurrentGeometry(candidate, currentBar, candle);
    }

    void SetCurrentGeometry(TrendLineCandidate& candidate,
                            std::size_t currentBar,
                            const Candle& candle) const
    {
        candidate.currentBar = currentBar;
        candidate.currentTimestamp = candle.timestamp;
        candidate.candidateAgeBars = currentBar - candidate.creationBar;
        candidate.projectedLinePriceAtCurrentBar =
            candidate.ProjectedPrice(currentBar);
        candidate.rawPriceToLineDistance =
            candidate.direction == TrendLineDirection::UTL
            ? candle.low - candidate.projectedLinePriceAtCurrentBar
            : candidate.projectedLinePriceAtCurrentBar - candle.high;
        candidate.currentAtr = currentAtr_;
        if (currentAtr_.has_value())
        {
            candidate.atrNormalizedSlope =
                candidate.rawPriceSlopePerBar / *currentAtr_;
            candidate.atrNormalizedPriceToLineDistance =
                candidate.rawPriceToLineDistance / *currentAtr_;
        }
        else
        {
            candidate.atrNormalizedSlope.reset();
            candidate.atrNormalizedPriceToLineDistance.reset();
        }
    }

    void ExpireCandidates(std::size_t currentBar)
    {
        candidates_.erase(
            std::remove_if(candidates_.begin(), candidates_.end(),
                [this, currentBar](const TrendLineCandidate& candidate)
                {
                    return currentBar > candidate.creationBar &&
                        currentBar - candidate.creationBar >
                            configuration_.maxCandidateAgeBars;
                }),
            candidates_.end());
    }

    void PruneFractals(std::deque<ConfirmedFractal>& fractals,
                       std::size_t currentBar) const
    {
        while (!fractals.empty() &&
               currentBar > fractals.front().anchorBar &&
               currentBar - fractals.front().anchorBar >
                   configuration_.maxFractalAnchorLookbackBars)
            fractals.pop_front();
        while (fractals.size() > configuration_.maxConfirmedFractalsPerKind)
            fractals.pop_front();
    }

    void PruneStoredHistory(std::size_t currentBar)
    {
        PruneFractals(highFractals_, currentBar);
        PruneFractals(lowFractals_, currentBar);
        const std::size_t retainBars =
            configuration_.maxFractalAnchorLookbackBars + kFractalRadius;
        while (!bars_.empty() && currentBar > bars_.front().bar &&
               currentBar - bars_.front().bar > retainBars)
            bars_.pop_front();
    }

    void EnforceCandidateBound()
    {
        while (candidates_.size() > configuration_.maxCandidates)
        {
            const auto oldest = std::min_element(
                candidates_.begin(), candidates_.end(),
                [](const TrendLineCandidate& left,
                   const TrendLineCandidate& right)
                {
                    return EvictionKey(left) < EvictionKey(right);
                });
            candidates_.erase(oldest);
        }
    }

    static std::tuple<std::size_t, int, std::size_t, std::size_t>
    EvictionKey(const TrendLineCandidate& candidate)
    {
        return std::tuple(candidate.creationBar,
                          static_cast<int>(candidate.direction),
                          candidate.anchor1Bar, candidate.anchor2Bar);
    }

    static bool CandidateLess(const TrendLineCandidate& left,
                              const TrendLineCandidate& right)
    {
        return std::tuple(static_cast<int>(left.direction), left.anchor1Bar,
                          left.anchor2Bar, left.creationBar) <
               std::tuple(static_cast<int>(right.direction), right.anchor1Bar,
                          right.anchor2Bar, right.creationBar);
    }

    static void SortCandidates(std::vector<TrendLineCandidate>& candidates)
    {
        std::sort(candidates.begin(), candidates.end(), CandidateLess);
    }

    static bool FractalLess(const ConfirmedFractal& left,
                            const ConfirmedFractal& right)
    {
        return std::tuple(left.anchorBar, static_cast<int>(left.kind),
                          left.confirmationBar) <
               std::tuple(right.anchorBar, static_cast<int>(right.kind),
                          right.confirmationBar);
    }

    static const char* DirectionName(TrendLineDirection direction)
    {
        return direction == TrendLineDirection::UTL ? "UTL" : "DTL";
    }

    static std::string OptionalNumber(const std::optional<double>& value)
    {
        if (!value.has_value()) return "unavailable";
        std::ostringstream output;
        output << std::setprecision(12) << *value;
        return output.str();
    }

    static std::string FormatTimestamp(std::int64_t epochSeconds)
    {
        const std::time_t raw = static_cast<std::time_t>(epochSeconds);
        std::tm utc{};
        if (gmtime_r(&raw, &utc) == nullptr)
            return std::to_string(epochSeconds);
        std::ostringstream output;
        output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
        return output.str();
    }
};

} // namespace EA::TG1A

#endif /* CausalFractalTrendLineGeometry_hpp */
