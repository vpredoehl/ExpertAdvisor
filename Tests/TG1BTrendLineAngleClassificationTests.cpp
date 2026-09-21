#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "CausalFractalTrendLineAngleClassification.hpp"

namespace
{
using EA::TG1A::Candle;
using EA::TG1A::Configuration;
using EA::TG1A::TrendLineDirection;
using EA::TG1B::CalibrateNormalizedSlope;
using EA::TG1B::CalibrationConfiguration;
using EA::TG1B::CausalFractalTrendLineAngleClassification;
using EA::TG1B::ClassifiedTrendLineCandidate;
using EA::TG1B::ClassifyAngleMagnitude;
using EA::TG1B::SteepnessClassification;

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

std::vector<Candle> ValidUtlCandles()
{
    return BarsFromLows(
        {10.0, 9.0, 5.0, 8.0, 10.0, 9.0, 10.0, 7.0, 10.0, 10.0});
}

std::vector<Candle> MirrorCandles(const std::vector<Candle>& candles,
                                  double mirrorSum)
{
    std::vector<Candle> result;
    result.reserve(candles.size());
    for (const Candle& candle : candles)
    {
        result.push_back({candle.timestamp,
                          mirrorSum - candle.open,
                          mirrorSum - candle.low,
                          mirrorSum - candle.high,
                          mirrorSum - candle.close});
    }
    return result;
}

const ClassifiedTrendLineCandidate* FindCandidate(
    const std::vector<ClassifiedTrendLineCandidate>& candidates,
    TrendLineDirection direction,
    std::size_t anchor1,
    std::size_t anchor2)
{
    const auto found = std::find_if(candidates.begin(), candidates.end(),
        [direction, anchor1, anchor2](
            const ClassifiedTrendLineCandidate& candidate)
        {
            return candidate.direction == direction &&
                candidate.anchor1Bar == anchor1 &&
                candidate.anchor2Bar == anchor2;
        });
    return found == candidates.end() ? nullptr : &*found;
}

std::vector<std::string> Diagnostics(
    const CausalFractalTrendLineAngleClassification& classification)
{
    std::vector<std::string> result;
    for (const ClassifiedTrendLineCandidate& candidate :
         classification.ClassifiedCandidates())
        result.push_back(classification.FormatDiagnostic(candidate));
    return result;
}

void Expect(double angle, SteepnessClassification expected)
{
    assert(ClassifyAngleMagnitude(angle) == expected);
}

void TestExactBandBoundariesAndGaps()
{
    const double infinity = std::numeric_limits<double>::infinity();
    Expect(std::nextafter(12.0, 0.0),
           SteepnessClassification::Unclassified);
    Expect(12.0, SteepnessClassification::LongTerm);
    Expect(20.0, SteepnessClassification::LongTerm);
    Expect(std::nextafter(20.0, infinity),
           SteepnessClassification::Unclassified);

    Expect(std::nextafter(25.0, 0.0),
           SteepnessClassification::Unclassified);
    Expect(25.0, SteepnessClassification::Outer);
    Expect(40.0, SteepnessClassification::Outer);
    Expect(std::nextafter(40.0, infinity),
           SteepnessClassification::Unclassified);

    Expect(std::nextafter(45.0, 0.0),
           SteepnessClassification::Unclassified);
    Expect(45.0, SteepnessClassification::Inner);
    Expect(85.0, SteepnessClassification::Inner);
    Expect(std::nextafter(85.0, infinity),
           SteepnessClassification::Unclassified);

    Expect(0.0, SteepnessClassification::Unclassified);
    Expect(90.0, SteepnessClassification::Unclassified);
    Expect(-1.0, SteepnessClassification::Unclassified);
    Expect(std::numeric_limits<double>::quiet_NaN(),
           SteepnessClassification::Unclassified);
}

void TestSymmetryMissingInvalidScaleAndMonotonicity()
{
    const CalibrationConfiguration unitScale(1.0);
    const auto positive = CalibrateNormalizedSlope(0.5, unitScale);
    const auto negative = CalibrateNormalizedSlope(-0.5, unitScale);
    assert(positive.magnitudeDegrees.has_value());
    assert(negative.magnitudeDegrees.has_value());
    assert(Near(*positive.magnitudeDegrees, *negative.magnitudeDegrees));
    assert(positive.classification == negative.classification);

    const auto missing = CalibrateNormalizedSlope(std::nullopt, unitScale);
    const auto infinite = CalibrateNormalizedSlope(
        std::numeric_limits<double>::infinity(), unitScale);
    const auto nan = CalibrateNormalizedSlope(
        std::numeric_limits<double>::quiet_NaN(), unitScale);
    assert(!missing.magnitudeDegrees.has_value());
    assert(!infinite.magnitudeDegrees.has_value());
    assert(!nan.magnitudeDegrees.has_value());
    assert(missing.classification == SteepnessClassification::Unclassified);
    assert(infinite.classification == SteepnessClassification::Unclassified);
    assert(nan.classification == SteepnessClassification::Unclassified);

    bool rejectedZero = false;
    bool rejectedNegative = false;
    bool rejectedInfinite = false;
    try { (void)CalibrationConfiguration(0.0); }
    catch (const std::invalid_argument&) { rejectedZero = true; }
    try { (void)CalibrationConfiguration(-1.0); }
    catch (const std::invalid_argument&) { rejectedNegative = true; }
    try { (void)CalibrationConfiguration(
              std::numeric_limits<double>::infinity()); }
    catch (const std::invalid_argument&) { rejectedInfinite = true; }
    assert(rejectedZero && rejectedNegative && rejectedInfinite);

    double previous = -1.0;
    for (const double slope : {0.0, 0.01, 0.1, 0.5, 1.0, 10.0})
    {
        const auto calibrated = CalibrateNormalizedSlope(slope, unitScale);
        assert(calibrated.magnitudeDegrees.has_value());
        assert(*calibrated.magnitudeDegrees > previous);
        previous = *calibrated.magnitudeDegrees;
    }
}

void TestDeterministicCalibrationScaleEffect()
{
    const auto scaleOne = CalibrateNormalizedSlope(
        0.3, CalibrationConfiguration(1.0));
    const auto scaleTwo = CalibrateNormalizedSlope(
        0.3, CalibrationConfiguration(2.0));
    const auto scaleThree = CalibrateNormalizedSlope(
        0.3, CalibrationConfiguration(3.0));
    assert(scaleOne.magnitudeDegrees.has_value());
    assert(scaleTwo.magnitudeDegrees.has_value());
    assert(scaleThree.magnitudeDegrees.has_value());
    assert(*scaleOne.magnitudeDegrees < *scaleTwo.magnitudeDegrees);
    assert(*scaleTwo.magnitudeDegrees < *scaleThree.magnitudeDegrees);
    assert(scaleOne.classification == SteepnessClassification::LongTerm);
    assert(scaleTwo.classification == SteepnessClassification::Outer);
    assert(scaleThree.classification ==
           SteepnessClassification::Unclassified);
}

double WilderAtrThrough(const std::vector<Candle>& candles,
                        std::size_t finalIndex,
                        std::size_t period)
{
    double state = 0.0;
    std::optional<double> previousClose;
    for (std::size_t index = 0; index <= finalIndex; ++index)
    {
        const Candle& candle = candles[index];
        double trueRange = candle.high - candle.low;
        if (previousClose.has_value())
            trueRange = std::max(
                {trueRange, std::fabs(candle.high - *previousClose),
                 std::fabs(candle.low - *previousClose)});
        previousClose = candle.close;
        if (index == 0)
            state = trueRange;
        else
            state = trueRange / static_cast<double>(period) +
                (1.0 - 1.0 / static_cast<double>(period)) * state;
    }
    return state;
}

void TestCreationTimeCausalityAndStableClassification()
{
    Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    const CalibrationConfiguration calibration(20.0);
    CausalFractalTrendLineAngleClassification classification(
        calibration, geometryConfiguration, {"EURUSD", "1h"});
    const auto candles = ValidUtlCandles();

    for (std::size_t index = 0; index < 9; ++index)
    {
        const auto update = classification.AddCompletedBar(candles[index]);
        assert(update.newlyClassifiedCandidates.empty());
    }
    assert(classification.ClassifiedCandidates().empty());

    const auto confirmation = classification.AddCompletedBar(candles[9]);
    assert(confirmation.bar == 9);
    assert(confirmation.newlyClassifiedCandidates.size() == 1);
    const ClassifiedTrendLineCandidate* candidate = FindCandidate(
        classification.ClassifiedCandidates(),
        TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    assert(candidate->creationBar == 9);
    assert(candidate->anchor2ConfirmationBar == 9);
    assert(candidate->creationAtr.has_value());
    assert(candidate->creationAtrNormalizedSlope.has_value());
    assert(candidate->calibratedAngleMagnitudeDegrees.has_value());

    const double expectedAtr = WilderAtrThrough(candles, 9, 3);
    assert(Near(*candidate->creationAtr, expectedAtr));
    assert(Near(*candidate->creationAtrNormalizedSlope, 0.4 / expectedAtr));
    const auto expectedAngle = CalibrateNormalizedSlope(
        0.4 / expectedAtr, calibration);
    assert(Near(*candidate->calibratedAngleMagnitudeDegrees,
                *expectedAngle.magnitudeDegrees));
    assert(candidate->classification == expectedAngle.classification);
    assert(candidate->classification == SteepnessClassification::Outer);

    const ClassifiedTrendLineCandidate fixed = *candidate;
    const std::string fixedDiagnostic =
        classification.FormatDiagnostic(*candidate);
    const double creationNormalized = *candidate->creationAtrNormalizedSlope;
    classification.AddCompletedBar(Bar(10, 100.0, 10.0));

    candidate = FindCandidate(classification.ClassifiedCandidates(),
                              TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    assert(Near(*candidate->creationAtr, *fixed.creationAtr));
    assert(Near(*candidate->creationAtrNormalizedSlope,
                creationNormalized));
    assert(Near(*candidate->calibratedAngleMagnitudeDegrees,
                *fixed.calibratedAngleMagnitudeDegrees));
    assert(candidate->classification == fixed.classification);
    assert(classification.FormatDiagnostic(*candidate) == fixedDiagnostic);

    const auto liveGeometry = std::find_if(
        classification.Geometry().Candidates().begin(),
        classification.Geometry().Candidates().end(),
        [](const EA::TG1A::TrendLineCandidate& value)
        {
            return value.direction == TrendLineDirection::UTL &&
                value.anchor1Bar == 2 && value.anchor2Bar == 7;
        });
    assert(liveGeometry != classification.Geometry().Candidates().end());
    assert(liveGeometry->atrNormalizedSlope.has_value());
    assert(!Near(*liveGeometry->atrNormalizedSlope, creationNormalized));
}

void TestEqualSteepnessUtlDtlEquivalence()
{
    Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    const CalibrationConfiguration calibration(20.0);
    const std::vector<Candle> utlCandles = ValidUtlCandles();
    const std::vector<Candle> dtlCandles = MirrorCandles(utlCandles, 25.0);
    const auto utl =
        CausalFractalTrendLineAngleClassification::FromHistorical(
            utlCandles, calibration, geometryConfiguration,
            {"SYMMETRIC", "1h"});
    const auto dtl =
        CausalFractalTrendLineAngleClassification::FromHistorical(
            dtlCandles, calibration, geometryConfiguration,
            {"SYMMETRIC", "1h"});
    const ClassifiedTrendLineCandidate* rising = FindCandidate(
        utl.ClassifiedCandidates(), TrendLineDirection::UTL, 2, 7);
    const ClassifiedTrendLineCandidate* falling = FindCandidate(
        dtl.ClassifiedCandidates(), TrendLineDirection::DTL, 2, 7);
    assert(rising != nullptr && falling != nullptr);
    assert(Near(rising->rawPriceSlopePerBar,
                -falling->rawPriceSlopePerBar));
    assert(Near(*rising->creationAtrNormalizedSlope,
                -*falling->creationAtrNormalizedSlope));
    assert(Near(*rising->calibratedAngleMagnitudeDegrees,
                *falling->calibratedAngleMagnitudeDegrees));
    assert(rising->classification == falling->classification);
}

void TestHistoricalStreamingEveryPrefixParity()
{
    Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    const CalibrationConfiguration calibration(20.0);
    std::vector<Candle> candles = ValidUtlCandles();
    candles.push_back(Bar(10, 20.0, 11.0));
    candles.push_back(Bar(11, 20.0, 11.0));
    candles.push_back(Bar(12, 20.0, 9.0));
    candles.push_back(Bar(13, 20.0, 11.0));
    candles.push_back(Bar(14, 20.0, 12.0));

    CausalFractalTrendLineAngleClassification streaming(
        calibration, geometryConfiguration, {"EURUSD", "1h"});
    for (std::size_t length = 1; length <= candles.size(); ++length)
    {
        streaming.AddCompletedBar(candles[length - 1]);
        const std::vector<Candle> prefix(candles.begin(),
                                         candles.begin() + length);
        const auto historical =
            CausalFractalTrendLineAngleClassification::FromHistorical(
                prefix, calibration, geometryConfiguration,
                {"EURUSD", "1h"});
        assert(Diagnostics(streaming) == Diagnostics(historical));
    }
}

void TestClassificationLifecycleMirrorsGeometry()
{
    Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    geometryConfiguration.maxCandidateAgeBars = 1;
    CausalFractalTrendLineAngleClassification classification(
        CalibrationConfiguration(20.0), geometryConfiguration);
    const auto candles = ValidUtlCandles();
    for (const Candle& candle : candles)
        classification.AddCompletedBar(candle);
    assert(classification.Geometry().Candidates().size() == 1);
    assert(classification.ClassifiedCandidates().size() == 1);

    classification.AddCompletedBar(Bar(10, 20.0, 11.0));
    assert(classification.Geometry().Candidates().size() == 1);
    assert(classification.ClassifiedCandidates().size() == 1);
    classification.AddCompletedBar(Bar(11, 20.0, 11.0));
    assert(classification.Geometry().Candidates().empty());
    assert(classification.ClassifiedCandidates().empty());
}

void TestDiagnosticExplainsClassification()
{
    Configuration geometryConfiguration;
    geometryConfiguration.atrPeriod = 3;
    auto classification =
        CausalFractalTrendLineAngleClassification::FromHistorical(
            ValidUtlCandles(), CalibrationConfiguration(20.0),
            geometryConfiguration, {"EURUSD", "1h"});
    const ClassifiedTrendLineCandidate* candidate = FindCandidate(
        classification.ClassifiedCandidates(),
        TrendLineDirection::UTL, 2, 7);
    assert(candidate != nullptr);
    const std::string diagnostic = classification.FormatDiagnostic(*candidate);
    assert(diagnostic.find("symbol=EURUSD,timeframe=1h,direction=UTL") !=
           std::string::npos);
    assert(diagnostic.find("anchor1_bar=2") != std::string::npos);
    assert(diagnostic.find("anchor2_bar=7") != std::string::npos);
    assert(diagnostic.find("raw_slope_per_bar=0.4") != std::string::npos);
    assert(diagnostic.find("creation_atr_normalized_slope=") !=
           std::string::npos);
    assert(diagnostic.find("calibration_reference_bar_scale=20") !=
           std::string::npos);
    assert(diagnostic.find("calibrated_angle_magnitude_degrees=") !=
           std::string::npos);
    assert(diagnostic.find("classification=Outer") != std::string::npos);
    assert(diagnostic.find(
        "classification_timing=second_anchor_confirmation") !=
        std::string::npos);
    std::cout << "TG1B_DIAGNOSTIC " << diagnostic << '\n';
}

} // namespace

int main()
{
    TestExactBandBoundariesAndGaps();
    TestSymmetryMissingInvalidScaleAndMonotonicity();
    TestDeterministicCalibrationScaleEffect();
    TestCreationTimeCausalityAndStableClassification();
    TestEqualSteepnessUtlDtlEquivalence();
    TestHistoricalStreamingEveryPrefixParity();
    TestClassificationLifecycleMirrorsGeometry();
    TestDiagnosticExplainsClassification();
    std::cout << "TG1BTrendLineAngleClassificationTests passed\n";
    return 0;
}
