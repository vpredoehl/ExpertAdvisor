#ifndef CausalFractalTrendLineAngleClassification_hpp
#define CausalFractalTrendLineAngleClassification_hpp

#include "CausalFractalTrendLineGeometry.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <ctime>
#include <iomanip>
#include <numbers>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

namespace EA::TG1B
{

enum class SteepnessClassification
{
    Unclassified,
    LongTerm,
    Outer,
    Inner
};

// The scale is an explicit normalized-coordinate convention.  A value S maps
// one horizontal display-independent unit to S sequential completed bars while
// the vertical unit is one creation-time ATR.  Thus normalizedSlope * S is the
// ATR-unit rise over that reference-bar horizon.  It is not a chart-pixel angle.
class CalibrationConfiguration
{
public:
    explicit CalibrationConfiguration(double referenceBarScale)
        : referenceBarScale_(referenceBarScale)
    {
        if (!std::isfinite(referenceBarScale) || referenceBarScale <= 0.0)
            throw std::invalid_argument(
                "TG1B calibration reference-bar scale must be finite and positive");
    }

    double ReferenceBarScale() const { return referenceBarScale_; }

private:
    double referenceBarScale_;
};

struct CalibratedAngle
{
    std::optional<double> magnitudeDegrees;
    SteepnessClassification classification =
        SteepnessClassification::Unclassified;
};

inline SteepnessClassification ClassifyAngleMagnitude(double angleDegrees)
{
    if (!std::isfinite(angleDegrees) || angleDegrees < 0.0)
        return SteepnessClassification::Unclassified;
    if (angleDegrees >= 12.0 && angleDegrees <= 20.0)
        return SteepnessClassification::LongTerm;
    if (angleDegrees >= 25.0 && angleDegrees <= 40.0)
        return SteepnessClassification::Outer;
    if (angleDegrees >= 45.0 && angleDegrees <= 85.0)
        return SteepnessClassification::Inner;
    return SteepnessClassification::Unclassified;
}

inline CalibratedAngle CalibrateNormalizedSlope(
    const std::optional<double>& signedAtrNormalizedSlope,
    const CalibrationConfiguration& configuration)
{
    if (!signedAtrNormalizedSlope.has_value() ||
        !std::isfinite(*signedAtrNormalizedSlope))
        return {};

    // The magnitude is symmetric for UTL and DTL.  Direction remains an
    // independent property of the underlying TG1A candidate.
    const double normalizedRise =
        std::fabs(*signedAtrNormalizedSlope) *
        configuration.ReferenceBarScale();
    const double angleDegrees = std::atan(normalizedRise) *
        180.0 / std::numbers::pi_v<double>;
    return {angleDegrees, ClassifyAngleMagnitude(angleDegrees)};
}

struct ClassifiedTrendLineCandidate
{
    TG1A::TrendLineDirection direction = TG1A::TrendLineDirection::UTL;
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
    std::size_t creationBar = 0;
    std::int64_t creationTimestamp = 0;
    double rawPriceSlopePerBar = 0.0;
    std::optional<double> creationAtr;
    std::optional<double> creationAtrNormalizedSlope;
    double calibrationReferenceBarScale = 0.0;
    std::optional<double> calibratedAngleMagnitudeDegrees;
    SteepnessClassification classification =
        SteepnessClassification::Unclassified;
};

struct Update
{
    std::size_t bar = 0;
    TG1A::Update geometryUpdate;
    std::vector<ClassifiedTrendLineCandidate> newlyClassifiedCandidates;
};

class CausalFractalTrendLineAngleClassification
{
public:
    explicit CausalFractalTrendLineAngleClassification(
        CalibrationConfiguration calibration,
        TG1A::Configuration geometryConfiguration = {},
        TG1A::SeriesIdentity identity = {})
        : geometry_(geometryConfiguration, std::move(identity)),
          calibration_(calibration)
    {
    }

    static CausalFractalTrendLineAngleClassification FromHistorical(
        std::vector<TG1A::Candle> candles,
        CalibrationConfiguration calibration,
        TG1A::Configuration geometryConfiguration = {},
        TG1A::SeriesIdentity identity = {})
    {
        std::stable_sort(candles.begin(), candles.end(),
            [](const TG1A::Candle& left, const TG1A::Candle& right)
            {
                return left.timestamp < right.timestamp;
            });
        CausalFractalTrendLineAngleClassification result(
            calibration, geometryConfiguration, std::move(identity));
        for (const TG1A::Candle& candle : candles)
            result.AddCompletedBar(candle);
        return result;
    }

    Update AddCompletedBar(const TG1A::Candle& candle)
    {
        Update result;
        result.geometryUpdate = geometry_.AddCompletedBar(candle);
        result.bar = result.geometryUpdate.bar;
        for (const TG1A::TrendLineCandidate& candidate :
             result.geometryUpdate.newlyCreatedCandidates)
        {
            ClassifiedTrendLineCandidate classified =
                ClassifyAtCreation(candidate);
            classifiedCandidates_.push_back(classified);
            result.newlyClassifiedCandidates.push_back(std::move(classified));
        }
        SynchronizeCandidateLifecycle();
        std::sort(classifiedCandidates_.begin(), classifiedCandidates_.end(),
                  ClassifiedCandidateLess);
        return result;
    }

    const std::vector<ClassifiedTrendLineCandidate>& ClassifiedCandidates()
        const
    {
        return classifiedCandidates_;
    }

    const TG1A::CausalFractalTrendLineGeometry& Geometry() const
    {
        return geometry_;
    }

    const CalibrationConfiguration& GetCalibrationConfiguration() const
    {
        return calibration_;
    }

    std::string FormatDiagnostic(
        const ClassifiedTrendLineCandidate& candidate) const
    {
        const TG1A::SeriesIdentity& identity = geometry_.GetSeriesIdentity();
        std::ostringstream output;
        output << std::setprecision(12)
               << "symbol=" << identity.symbol
               << ",timeframe=" << identity.timeframe
               << ",direction=" << DirectionName(candidate.direction)
               << ",anchor1_bar=" << candidate.anchor1Bar
               << ",anchor1_timestamp="
               << FormatTimestamp(candidate.anchor1Timestamp)
               << ",anchor1_price=" << candidate.anchor1Price
               << ",anchor1_confirmation_bar="
               << candidate.anchor1ConfirmationBar
               << ",anchor1_confirmation_timestamp="
               << FormatTimestamp(candidate.anchor1ConfirmationTimestamp)
               << ",anchor2_bar=" << candidate.anchor2Bar
               << ",anchor2_timestamp="
               << FormatTimestamp(candidate.anchor2Timestamp)
               << ",anchor2_price=" << candidate.anchor2Price
               << ",anchor2_confirmation_bar="
               << candidate.anchor2ConfirmationBar
               << ",anchor2_confirmation_timestamp="
               << FormatTimestamp(candidate.anchor2ConfirmationTimestamp)
               << ",creation_bar=" << candidate.creationBar
               << ",creation_timestamp="
               << FormatTimestamp(candidate.creationTimestamp)
               << ",separation_bars=" << candidate.anchorSeparationBars
               << ",raw_slope_per_bar="
               << candidate.rawPriceSlopePerBar
               << ",creation_atr=" << OptionalNumber(candidate.creationAtr)
               << ",creation_atr_normalized_slope="
               << OptionalNumber(candidate.creationAtrNormalizedSlope)
               << ",calibration_reference_bar_scale="
               << candidate.calibrationReferenceBarScale
               << ",calibrated_angle_magnitude_degrees="
               << OptionalNumber(
                      candidate.calibratedAngleMagnitudeDegrees)
               << ",classification="
               << ClassificationName(candidate.classification)
               << ",classification_timing=second_anchor_confirmation";
        return output.str();
    }

private:
    using CandidateIdentity =
        std::tuple<int, std::size_t, std::size_t, std::size_t>;

    TG1A::CausalFractalTrendLineGeometry geometry_;
    CalibrationConfiguration calibration_;
    std::vector<ClassifiedTrendLineCandidate> classifiedCandidates_;

    ClassifiedTrendLineCandidate ClassifyAtCreation(
        const TG1A::TrendLineCandidate& candidate) const
    {
        const CalibratedAngle angle = CalibrateNormalizedSlope(
            candidate.atrNormalizedSlope, calibration_);
        return {
            candidate.direction,
            candidate.anchor1Bar,
            candidate.anchor1Timestamp,
            candidate.anchor1Price,
            candidate.anchor1ConfirmationBar,
            candidate.anchor1ConfirmationTimestamp,
            candidate.anchor2Bar,
            candidate.anchor2Timestamp,
            candidate.anchor2Price,
            candidate.anchor2ConfirmationBar,
            candidate.anchor2ConfirmationTimestamp,
            candidate.anchorSeparationBars,
            candidate.creationBar,
            candidate.creationTimestamp,
            candidate.rawPriceSlopePerBar,
            candidate.currentAtr,
            candidate.atrNormalizedSlope,
            calibration_.ReferenceBarScale(),
            angle.magnitudeDegrees,
            angle.classification};
    }

    void SynchronizeCandidateLifecycle()
    {
        const std::vector<TG1A::TrendLineCandidate>& live =
            geometry_.Candidates();
        std::vector<CandidateIdentity> liveIdentities;
        liveIdentities.reserve(live.size());
        for (const TG1A::TrendLineCandidate& candidate : live)
            liveIdentities.push_back(Identity(candidate));
        std::sort(liveIdentities.begin(), liveIdentities.end());
        classifiedCandidates_.erase(
            std::remove_if(classifiedCandidates_.begin(),
                           classifiedCandidates_.end(),
                [&liveIdentities](
                    const ClassifiedTrendLineCandidate& classified)
                {
                    return !std::binary_search(
                        liveIdentities.begin(), liveIdentities.end(),
                        Identity(classified));
                }),
            classifiedCandidates_.end());
    }

    static CandidateIdentity Identity(
        const ClassifiedTrendLineCandidate& candidate)
    {
        return {static_cast<int>(candidate.direction), candidate.anchor1Bar,
                candidate.anchor2Bar, candidate.creationBar};
    }

    static CandidateIdentity Identity(
        const TG1A::TrendLineCandidate& candidate)
    {
        return {static_cast<int>(candidate.direction), candidate.anchor1Bar,
                candidate.anchor2Bar, candidate.creationBar};
    }

    static bool ClassifiedCandidateLess(
        const ClassifiedTrendLineCandidate& left,
        const ClassifiedTrendLineCandidate& right)
    {
        return std::tuple(static_cast<int>(left.direction), left.anchor1Bar,
                          left.anchor2Bar, left.creationBar) <
               std::tuple(static_cast<int>(right.direction), right.anchor1Bar,
                          right.anchor2Bar, right.creationBar);
    }

    static const char* DirectionName(TG1A::TrendLineDirection direction)
    {
        return direction == TG1A::TrendLineDirection::UTL ? "UTL" : "DTL";
    }

    static const char* ClassificationName(
        SteepnessClassification classification)
    {
        switch (classification)
        {
            case SteepnessClassification::LongTerm: return "LongTerm";
            case SteepnessClassification::Outer: return "Outer";
            case SteepnessClassification::Inner: return "Inner";
            case SteepnessClassification::Unclassified:
                return "Unclassified";
        }
        return "Unclassified";
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

} // namespace EA::TG1B

#endif /* CausalFractalTrendLineAngleClassification_hpp */
