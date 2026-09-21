#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "CausalFractalTrendLineGeometry.hpp"

namespace
{
using EA::TG1A::Candle;
using EA::TG1A::CausalFractalTrendLineGeometry;
using EA::TG1A::Configuration;
using EA::TG1A::ConfirmedFractal;
using EA::TG1A::FractalKind;
using EA::TG1A::SeriesIdentity;
using EA::TG1A::TrendLineCandidate;
using EA::TG1A::TrendLineDirection;

constexpr std::int64_t kStart = 1'700'000'000;
constexpr std::int64_t kHour = 3'600;

bool Near(double left, double right, double tolerance = 1.0e-10)
{
    return std::fabs(left - right) <= tolerance;
}

Candle Bar(std::size_t index, double high, double low)
{
    const double middle = 0.5 * (high + low);
    return {kStart + static_cast<std::int64_t>(index) * kHour,
            middle, high, low, middle};
}

std::vector<Candle> BarsFromLows(const std::vector<double>& lows)
{
    std::vector<Candle> result;
    result.reserve(lows.size());
    for (std::size_t index = 0; index < lows.size(); ++index)
        result.push_back(Bar(index, 20.0, lows[index]));
    return result;
}

std::vector<Candle> BarsFromHighs(const std::vector<double>& highs)
{
    std::vector<Candle> result;
    result.reserve(highs.size());
    for (std::size_t index = 0; index < highs.size(); ++index)
        result.push_back(Bar(index, highs[index], 1.0));
    return result;
}

void Feed(CausalFractalTrendLineGeometry& geometry,
          const std::vector<Candle>& candles,
          std::size_t count)
{
    for (std::size_t index = 0; index < count; ++index)
        geometry.AddCompletedBar(candles[index]);
}

std::optional<ConfirmedFractal> FindFractal(
    const std::vector<ConfirmedFractal>& fractals,
    FractalKind kind,
    std::size_t anchorBar)
{
    const auto found = std::find_if(fractals.begin(), fractals.end(),
        [kind, anchorBar](const ConfirmedFractal& fractal)
        {
            return fractal.kind == kind && fractal.anchorBar == anchorBar;
        });
    return found == fractals.end()
        ? std::optional<ConfirmedFractal>{}
        : std::optional<ConfirmedFractal>{*found};
}

const TrendLineCandidate* FindCandidate(
    const std::vector<TrendLineCandidate>& candidates,
    TrendLineDirection direction,
    std::size_t anchor1,
    std::size_t anchor2)
{
    const auto found = std::find_if(candidates.begin(), candidates.end(),
        [direction, anchor1, anchor2](const TrendLineCandidate& candidate)
        {
            return candidate.direction == direction &&
                candidate.anchor1Bar == anchor1 &&
                candidate.anchor2Bar == anchor2;
        });
    return found == candidates.end() ? nullptr : &*found;
}

std::vector<std::string> Diagnostics(
    const CausalFractalTrendLineGeometry& geometry)
{
    std::vector<std::string> result;
    for (const TrendLineCandidate& candidate : geometry.Candidates())
        result.push_back(geometry.FormatDiagnostic(candidate));
    return result;
}

void TestStrictFractalHighLowAndEqualBoundaries()
{
    CausalFractalTrendLineGeometry high;
    const std::vector<double> highs{10.0, 11.0, 15.0, 12.0, 9.0};
    Feed(high, BarsFromHighs(highs), highs.size());
    const auto highFractals = high.ConfirmedFractals();
    const auto detectedHigh =
        FindFractal(highFractals, FractalKind::High, 2);
    assert(detectedHigh.has_value());
    assert(Near(detectedHigh->price, 15.0));

    CausalFractalTrendLineGeometry low;
    const std::vector<double> lows{5.0, 4.0, 1.0, 3.0, 5.0};
    Feed(low, BarsFromLows(lows), lows.size());
    const auto detectedLow =
        FindFractal(low.ConfirmedFractals(), FractalKind::Low, 2);
    assert(detectedLow.has_value());
    assert(Near(detectedLow->price, 1.0));

    CausalFractalTrendLineGeometry equalHigh;
    const std::vector<double> equalHighs{10.0, 15.0, 15.0, 12.0, 9.0};
    Feed(equalHigh, BarsFromHighs(equalHighs), equalHighs.size());
    assert(!FindFractal(equalHigh.ConfirmedFractals(),
                        FractalKind::High, 2).has_value());

    CausalFractalTrendLineGeometry equalLow;
    const std::vector<double> equalLows{5.0, 1.0, 1.0, 3.0, 5.0};
    Feed(equalLow, BarsFromLows(equalLows), equalLows.size());
    assert(!FindFractal(equalLow.ConfirmedFractals(),
                        FractalKind::Low, 2).has_value());
}

void TestCausalConfirmationAndAnchor()
{
    CausalFractalTrendLineGeometry geometry;
    const auto candles = BarsFromLows({5.0, 4.0, 1.0, 3.0, 5.0});
    Feed(geometry, candles, 4); // center i has only reached i+1
    assert(geometry.ConfirmedFractals().empty());

    const auto update = geometry.AddCompletedBar(candles[4]); // i+2 closes
    assert(update.newlyConfirmedFractals.size() == 1);
    const ConfirmedFractal& fractal = update.newlyConfirmedFractals.front();
    assert(fractal.kind == FractalKind::Low);
    assert(fractal.anchorBar == 2);
    assert(fractal.anchorTimestamp == candles[2].timestamp);
    assert(fractal.confirmationBar == 4);
    assert(fractal.confirmationTimestamp == candles[4].timestamp);
}

void TestHistoricalChronologicalOrdering()
{
    const auto ordered = BarsFromHighs({10.0, 11.0, 15.0, 12.0, 9.0});
    std::vector<Candle> shuffled{
        ordered[3], ordered[0], ordered[4], ordered[1], ordered[2]};
    const auto historical =
        CausalFractalTrendLineGeometry::FromHistorical(shuffled);
    const auto fractal =
        FindFractal(historical.ConfirmedFractals(), FractalKind::High, 2);
    assert(fractal.has_value());
    assert(fractal->anchorTimestamp == ordered[2].timestamp);
    assert(fractal->confirmationTimestamp == ordered[4].timestamp);

    bool rejectedDuplicateTimestamp = false;
    try
    {
        auto duplicate = ordered;
        duplicate[4].timestamp = duplicate[3].timestamp;
        (void)CausalFractalTrendLineGeometry::FromHistorical(duplicate);
    }
    catch (const std::invalid_argument&)
    {
        rejectedDuplicateTimestamp = true;
    }
    assert(rejectedDuplicateTimestamp);
}

std::vector<Candle> ValidUtlCandles()
{
    return BarsFromLows(
        {10.0, 9.0, 5.0, 8.0, 10.0, 9.0, 10.0, 7.0, 10.0, 10.0});
}

void TestUtlCreationInvalidSecondLowAndInterveningCross()
{
    Configuration config;
    config.interveningPriceTolerance = 0.05;
    config.touchPriceTolerance = 0.01;
    config.atrPeriod = 3;

    auto valid = CausalFractalTrendLineGeometry::FromHistorical(
        ValidUtlCandles(), config, {"EURUSD", "1h"});
    const TrendLineCandidate* candidate = FindCandidate(
        valid.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    assert(candidate->anchor1ConfirmationBar == 4);
    assert(candidate->anchor2ConfirmationBar == 9);
    assert(candidate->creationBar == 9);
    assert(candidate->anchorSeparationBars == 5);
    assert(Near(candidate->rawPriceSlopePerBar, 0.4));
    assert(candidate->fractalTouchCount == 2);
    assert(candidate->currentAtr.has_value());
    assert(candidate->atrNormalizedSlope.has_value());

    auto lowerSecond = ValidUtlCandles();
    lowerSecond[7] = Bar(7, 20.0, 4.0);
    auto invalid = CausalFractalTrendLineGeometry::FromHistorical(
        lowerSecond, config);
    assert(FindCandidate(invalid.Candidates(),
                         TrendLineDirection::UTL, 2, 7) == nullptr);

    auto crossed = ValidUtlCandles();
    crossed[3] = Bar(3, 20.0, 5.2); // line=5.4; 0.05 tolerance is exceeded
    auto rejected = CausalFractalTrendLineGeometry::FromHistorical(
        crossed, config);
    assert(FindCandidate(rejected.Candidates(),
                         TrendLineDirection::UTL, 2, 7) == nullptr);

    crossed[3] = Bar(3, 20.0, 5.36); // 0.04 below line is tolerated
    auto tolerated = CausalFractalTrendLineGeometry::FromHistorical(
        crossed, config);
    assert(FindCandidate(tolerated.Candidates(),
                         TrendLineDirection::UTL, 2, 7) != nullptr);
}

void TestDtlCreationAndInterveningCross()
{
    Configuration config;
    config.interveningPriceTolerance = 0.05;
    config.touchPriceTolerance = 0.01;
    const auto validBars = BarsFromHighs(
        {10.0, 11.0, 15.0, 14.0, 10.0, 12.0, 11.0, 13.0, 11.0, 10.0});
    auto valid = CausalFractalTrendLineGeometry::FromHistorical(
        validBars, config);
    const TrendLineCandidate* candidate = FindCandidate(
        valid.Candidates(), TrendLineDirection::DTL, 2, 7);
    assert(candidate != nullptr);
    assert(Near(candidate->rawPriceSlopePerBar, -0.4));
    assert(candidate->fractalTouchCount == 2);

    auto crossed = validBars;
    crossed[3] = Bar(3, 14.8, 1.0); // line=14.6; 0.05 tolerance exceeded
    auto rejected = CausalFractalTrendLineGeometry::FromHistorical(
        crossed, config);
    assert(FindCandidate(rejected.Candidates(),
                         TrendLineDirection::DTL, 2, 7) == nullptr);
}

void TestProjectionAndSeparateTouchCounts()
{
    Configuration config;
    config.touchPriceTolerance = 1.0e-9;
    config.atrPeriod = 3;
    auto candles = ValidUtlCandles();
    candles[3] = Bar(3, 20.0, 5.4); // ordinary touch between anchors
    candles[9] = Bar(9, 20.0, 7.8); // ordinary touch at confirmation bar
    CausalFractalTrendLineGeometry geometry(config, {"EURUSD", "1h"});
    Feed(geometry, candles, candles.size());

    const TrendLineCandidate* candidate = FindCandidate(
        geometry.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    assert(Near(candidate->ProjectedPrice(12), 9.0));
    assert(Near(candidate->projectedLinePriceAtCurrentBar, 7.8));
    assert(Near(candidate->rawPriceToLineDistance, 0.0));
    assert(candidate->candleTouchCount == 2);
    assert(candidate->fractalTouchCount == 2);

    const auto extension = BarsFromLows(
        {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
         11.0, 11.0, 9.0, 11.0, 12.0});
    for (std::size_t index = 10; index < extension.size(); ++index)
        geometry.AddCompletedBar(extension[index]);
    candidate = FindCandidate(
        geometry.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    assert(candidate->fractalTouchCount == 3);
    assert(candidate->candleTouchCount >= 2);
}

void TestNoLookAheadThroughTrendGeometry()
{
    Configuration config;
    config.touchPriceTolerance = 1.0e-9;
    auto prefix = ValidUtlCandles();
    prefix.push_back(Bar(10, 20.0, 11.0));
    prefix.push_back(Bar(11, 20.0, 11.0));
    prefix.push_back(Bar(12, 20.0, 9.0));
    prefix.push_back(Bar(13, 20.0, 11.0)); // third fractal is only at i+1

    CausalFractalTrendLineGeometry confirms(config, {"EURUSD", "1h"});
    CausalFractalTrendLineGeometry invalidates(config, {"EURUSD", "1h"});
    Feed(confirms, prefix, prefix.size());
    Feed(invalidates, prefix, prefix.size());
    assert(Diagnostics(confirms) == Diagnostics(invalidates));
    const TrendLineCandidate* before = FindCandidate(
        confirms.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(before != nullptr && before->fractalTouchCount == 2);
    assert(!FindFractal(confirms.ConfirmedFractals(),
                        FractalKind::Low, 12).has_value());

    confirms.AddCompletedBar(Bar(14, 20.0, 12.0)); // confirms center 12
    invalidates.AddCompletedBar(Bar(14, 20.0, 8.0)); // prevents center 12
    const TrendLineCandidate* afterConfirmation = FindCandidate(
        confirms.Candidates(), TrendLineDirection::UTL, 2, 7);
    const TrendLineCandidate* afterInvalidation = FindCandidate(
        invalidates.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(afterConfirmation != nullptr && afterInvalidation != nullptr);
    assert(afterConfirmation->fractalTouchCount == 3);
    assert(afterInvalidation->fractalTouchCount == 2);
    assert(FindFractal(confirms.ConfirmedFractals(),
                       FractalKind::Low, 12).has_value());
    assert(!FindFractal(invalidates.ConfirmedFractals(),
                        FractalKind::Low, 12).has_value());
}

void TestMultipleCandidatesDeterminismAndBounds()
{
    Configuration config;
    config.touchPriceTolerance = 1.0e-9;
    auto candles = ValidUtlCandles();
    candles.push_back(Bar(10, 20.0, 11.0));
    candles.push_back(Bar(11, 20.0, 11.0));
    candles.push_back(Bar(12, 20.0, 9.0));
    candles.push_back(Bar(13, 20.0, 11.0));
    candles.push_back(Bar(14, 20.0, 12.0));

    auto first = CausalFractalTrendLineGeometry::FromHistorical(
        candles, config, {"EURUSD", "1h"});
    auto second = CausalFractalTrendLineGeometry::FromHistorical(
        candles, config, {"EURUSD", "1h"});
    assert(Diagnostics(first) == Diagnostics(second));

    CausalFractalTrendLineGeometry streaming(config, {"EURUSD", "1h"});
    Feed(streaming, candles, candles.size());
    std::vector<Candle> reversePhysicalOrder(candles.rbegin(), candles.rend());
    auto historicalFromUnorderedRows =
        CausalFractalTrendLineGeometry::FromHistorical(
            reversePhysicalOrder, config, {"EURUSD", "1h"});
    assert(Diagnostics(first) == Diagnostics(streaming));
    assert(Diagnostics(first) == Diagnostics(historicalFromUnorderedRows));
    assert(FindCandidate(first.Candidates(),
                         TrendLineDirection::UTL, 2, 7) != nullptr);
    assert(FindCandidate(first.Candidates(),
                         TrendLineDirection::UTL, 2, 12) != nullptr);
    assert(FindCandidate(first.Candidates(),
                         TrendLineDirection::UTL, 7, 12) != nullptr);

    Configuration bounded = config;
    bounded.maxCandidates = 2;
    auto boundedFirst = CausalFractalTrendLineGeometry::FromHistorical(
        candles, bounded, {"EURUSD", "1h"});
    auto boundedSecond = CausalFractalTrendLineGeometry::FromHistorical(
        candles, bounded, {"EURUSD", "1h"});
    assert(boundedFirst.Candidates().size() == 2);
    assert(Diagnostics(boundedFirst) == Diagnostics(boundedSecond));
    assert(FindCandidate(boundedFirst.Candidates(),
                         TrendLineDirection::UTL, 2, 7) == nullptr);
    assert(FindCandidate(boundedFirst.Candidates(),
                         TrendLineDirection::UTL, 2, 12) != nullptr);
    assert(FindCandidate(boundedFirst.Candidates(),
                         TrendLineDirection::UTL, 7, 12) != nullptr);

    Configuration shortLived = config;
    shortLived.maxCandidateAgeBars = 1;
    CausalFractalTrendLineGeometry expiring(shortLived);
    const auto base = ValidUtlCandles();
    Feed(expiring, base, base.size());
    assert(FindCandidate(expiring.Candidates(),
                         TrendLineDirection::UTL, 2, 7) != nullptr);
    expiring.AddCompletedBar(Bar(10, 20.0, 11.0));
    assert(FindCandidate(expiring.Candidates(),
                         TrendLineDirection::UTL, 2, 7) != nullptr);
    expiring.AddCompletedBar(Bar(11, 20.0, 11.0));
    assert(FindCandidate(expiring.Candidates(),
                         TrendLineDirection::UTL, 2, 7) == nullptr);
}

void TestBoundedLongStream()
{
    Configuration config;
    config.maxFractalAnchorLookbackBars = 128;
    config.maxConfirmedFractalsPerKind = 16;
    config.maxCandidateAgeBars = 128;
    config.maxCandidates = 128;
    CausalFractalTrendLineGeometry geometry(config);

    constexpr std::size_t barCount = 50'000;
    const auto start = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < barCount; ++index)
    {
        const bool center = index % 5 == 2;
        const double low = center
            ? 5.0 + static_cast<double>(index / 5) * 0.0001
            : 10.0;
        geometry.AddCompletedBar(Bar(index, 20.0, low));
    }
    const auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(
        std::chrono::steady_clock::now() - start);
    assert(geometry.CompletedBarCount() == barCount);
    assert(geometry.ConfirmedFractals().size() <=
           2 * config.maxConfirmedFractalsPerKind);
    assert(geometry.Candidates().size() <= config.maxCandidates);
    std::cout << "TG1A_BOUNDED_STREAM bars=" << barCount
              << ",milliseconds=" << elapsed.count()
              << ",live_fractals=" << geometry.ConfirmedFractals().size()
              << ",live_candidates=" << geometry.Candidates().size() << '\n';
}

void PrintRepresentativeDiagnostic()
{
    Configuration config;
    config.touchPriceTolerance = 0.01;
    config.atrPeriod = 3;
    auto geometry = CausalFractalTrendLineGeometry::FromHistorical(
        ValidUtlCandles(), config, {"EURUSD", "1h"});
    const TrendLineCandidate* candidate = FindCandidate(
        geometry.Candidates(), TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    std::cout << "TG1A_DIAGNOSTIC "
              << geometry.FormatDiagnostic(*candidate) << '\n';
}
} // namespace

int main()
{
    TestStrictFractalHighLowAndEqualBoundaries();
    TestCausalConfirmationAndAnchor();
    TestHistoricalChronologicalOrdering();
    TestUtlCreationInvalidSecondLowAndInterveningCross();
    TestDtlCreationAndInterveningCross();
    TestProjectionAndSeparateTouchCounts();
    TestNoLookAheadThroughTrendGeometry();
    TestMultipleCandidatesDeterminismAndBounds();
    TestBoundedLongStream();
    PrintRepresentativeDiagnostic();
    std::cout << "TG1ACausalFractalTrendLineGeometryTests passed\n";
    return 0;
}
