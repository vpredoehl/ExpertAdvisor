#pragma once

#include "CausalFibonacciConfluenceIntegration.hpp"

#include <numeric>
#include <span>

// Read-only geometry research. Nothing in the production pulse consumes this
// evaluator. SymmetricDirectionalDiagnostic is an explicit research hypothesis.
namespace EA::TG3::TrendlineDiagnostic
{

struct Configuration
{
    explicit Configuration(double intersectionThreshold)
        : intersectionAtrThreshold(intersectionThreshold) {}

    std::vector<double> retracementRatios{0.382, 0.5, 0.6180339887498949};
    double intersectionAtrThreshold;
    std::optional<double> marketProximityAtrThreshold;
    DirectionalStudyPolicy directionalStudyPolicy =
        DirectionalStudyPolicy::SourceUTLUpABOnly;
    std::size_t maxABStructures = 512;
    std::size_t maxTrendlines = 4096;
    std::size_t maxRatios = 16;
    std::size_t maxPairEvaluations = 65536;
    std::size_t maxMeasurements = 4096;
};

struct Identity
{
    ABIdentity ab;
    TG2::CandidateIdentity line;
    double ratio = 0.0;
    std::size_t decisionBar = 0;

    bool operator==(const Identity&) const = default;
};

struct Measurement
{
    Identity identity;
    double fibPrice = 0.0;
    double linePrice = 0.0;
    std::optional<double> normalizedStructureDistance;
    std::optional<double> normalizedMarketDistance;
    bool intersectionThresholdMatch = false;
    // Absent when the caller has not requested a market threshold. Even a
    // true value reports proximity only, not observed interaction or an outcome.
    std::optional<bool> marketProximityThresholdMatch;
    bool configuredThresholdMatch = false;

    bool operator==(const Measurement&) const = default;
};

struct Result
{
    std::vector<Measurement> measurements;
    std::size_t pairsExamined = 0;
    bool pairCapacityReached = false;
    bool measurementCapacityReached = false;

    bool operator==(const Result&) const = default;
};

namespace Detail
{
inline auto Key(const ABIdentity& value)
{
    return std::tuple{value.direction, value.aBar, value.aTimestamp,
        value.bBar, value.bTimestamp, value.availabilityBar,
        value.availabilityTimestamp};
}

inline TG2::CandidateIdentity LineIdentity(const TG1A::TrendLineCandidate& line)
{
    return {line.direction, line.anchor1Bar, line.anchor1Timestamp,
        line.anchor2Bar, line.anchor2Timestamp, line.creationBar,
        line.creationTimestamp};
}

inline auto Key(const TG2::CandidateIdentity& value)
{
    return std::tuple{value.direction, value.anchor1Bar, value.anchor1Timestamp,
        value.anchor2Bar, value.anchor2Timestamp, value.creationBar,
        value.creationTimestamp};
}

inline bool Eligible(const ABStructure& ab, std::size_t bar)
{
    const auto& id = ab.identity;
    const bool priceDirection =
        (id.direction == ABDirection::UpAB && ab.aPrice < ab.bPrice) ||
        (id.direction == ABDirection::DownAB && ab.aPrice > ab.bPrice);
    return std::isfinite(ab.aPrice) && std::isfinite(ab.bPrice) &&
        priceDirection && id.aBar < id.bBar &&
        id.aBar <= ab.aConfirmationBar && id.bBar <= ab.bConfirmationBar &&
        ab.aConfirmationBar <= id.availabilityBar &&
        ab.bConfirmationBar <= id.availabilityBar && id.availabilityBar <= bar;
}

inline bool Eligible(const TG1A::TrendLineCandidate& line, std::size_t bar)
{
    return (line.direction == TG1A::TrendLineDirection::UTL ||
            line.direction == TG1A::TrendLineDirection::DTL) &&
        std::isfinite(line.anchor1Price) && std::isfinite(line.anchor2Price) &&
        std::isfinite(line.rawPriceSlopePerBar) &&
        line.anchor1Bar < line.anchor2Bar &&
        line.anchor1Bar <= line.anchor1ConfirmationBar &&
        line.anchor2Bar <= line.anchor2ConfirmationBar &&
        line.anchor1ConfirmationBar <= line.creationBar &&
        line.anchor2ConfirmationBar <= line.creationBar &&
        line.creationBar <= bar && line.currentBar == bar &&
        std::isfinite(line.ProjectedPrice(bar));
}

inline std::optional<double> FiniteDouble(long double value)
{
    if (!std::isfinite(value) ||
        std::fabs(value) > std::numeric_limits<double>::max())
        return std::nullopt;
    return static_cast<double>(value);
}

inline std::optional<double> NormalizedDistance(double left, double right, double atr)
{
    const long double difference = std::fabs(static_cast<long double>(left) - right);
    // On macOS arm64 long double has double precision/range. An overflowing
    // subtraction of finite endpoints has opposing signs, so scale each term
    // before adding; this can still yield a representable normalized distance.
    return FiniteDouble(std::isfinite(difference) ? difference / atr :
        std::fabs(static_cast<long double>(left) / atr) +
        std::fabs(static_cast<long double>(right) / atr));
}

inline auto Snapshot(const ABStructure& ab)
{
    return std::tuple{ab.aPrice, ab.bPrice, ab.aConfirmationBar,
        ab.aConfirmationTimestamp, ab.bConfirmationBar, ab.bConfirmationTimestamp};
}

inline auto Snapshot(const TG1A::TrendLineCandidate& line)
{
    return std::tuple{line.anchor1Price, line.anchor2Price,
        line.rawPriceSlopePerBar, line.anchor1ConfirmationBar,
        line.anchor1ConfirmationTimestamp, line.anchor2ConfirmationBar,
        line.anchor2ConfirmationTimestamp};
}

// Only fields used by this diagnostic participate in duplicate consistency.
template<typename T, typename KeyFunction>
std::vector<const T*> CanonicalInputs(std::span<const T> input,
                                     std::size_t bar, KeyFunction key)
{
    std::vector<const T*> result;
    for (const auto& value : input)
        if (Eligible(value, bar)) result.push_back(&value);
    std::sort(result.begin(), result.end(), [&](const T* left, const T* right)
        { return key(*left) < key(*right); });
    for (std::size_t i = 1; i < result.size(); ++i)
        if (key(*result[i - 1]) == key(*result[i]) &&
            Snapshot(*result[i - 1]) != Snapshot(*result[i]))
            throw std::invalid_argument(
                "geometric diagnostic received conflicting snapshots for one identity");
    result.erase(std::unique(result.begin(), result.end(),
        [&](const T* left, const T* right)
        { return key(*left) == key(*right); }), result.end());
    return result;
}
} // namespace Detail

// Oversized input spans/ratio lists are rejected before copying or pairing.
// Unique causal inputs are ordered by AB identity, then line identity, then
// ascending unique ratio. Work/output caps retain that deterministic prefix;
// caps count every pair considered, including unsupported direction pairs.
inline Result Evaluate(std::size_t decisionBar, double close,
                       std::optional<double> atr,
                       std::span<const ABStructure> structures,
                       std::span<const TG1A::TrendLineCandidate> lines,
                       const Configuration& configuration)
{
    const auto validThreshold = [](double value)
        { return std::isfinite(value) && value >= 0.0; };
    if (!std::isfinite(close) ||
        !validThreshold(configuration.intersectionAtrThreshold) ||
        (configuration.marketProximityAtrThreshold &&
         !validThreshold(*configuration.marketProximityAtrThreshold)) ||
        (configuration.directionalStudyPolicy != DirectionalStudyPolicy::SourceUTLUpABOnly &&
         configuration.directionalStudyPolicy != DirectionalStudyPolicy::SymmetricDirectionalDiagnostic) ||
        configuration.maxABStructures == 0 || configuration.maxTrendlines == 0 ||
        configuration.maxRatios == 0 || configuration.maxPairEvaluations == 0 ||
        configuration.maxMeasurements == 0 ||
        structures.size() > configuration.maxABStructures ||
        lines.size() > configuration.maxTrendlines ||
        configuration.retracementRatios.empty() ||
        configuration.retracementRatios.size() > configuration.maxRatios)
        throw std::invalid_argument("invalid geometric diagnostic configuration or input bounds");
    for (double ratio : configuration.retracementRatios)
        if (!std::isfinite(ratio) || ratio < 0.0 || ratio > 1.0)
            throw std::invalid_argument("diagnostic retracement ratios must be finite in [0,1]");
    auto ratios = configuration.retracementRatios;
    // Canonicalize signed zero as well as duplicate ratios.
    for (double& ratio : ratios) if (ratio == 0.0) ratio = 0.0;
    std::sort(ratios.begin(), ratios.end());
    ratios.erase(std::unique(ratios.begin(), ratios.end()), ratios.end());
    const auto abs = Detail::CanonicalInputs(structures, decisionBar,
        [](const ABStructure& ab) { return Detail::Key(ab.identity); });
    const auto candidates = Detail::CanonicalInputs(lines, decisionBar,
        [](const TG1A::TrendLineCandidate& line)
        { return Detail::Key(Detail::LineIdentity(line)); });
    const bool validAtr = atr && std::isfinite(*atr) && *atr > 0.0;
    Result result;
    for (const auto* ab : abs)
        for (const auto* line : candidates)
        {
            if (result.pairsExamined == configuration.maxPairEvaluations)
            {
                result.pairCapacityReached = true;
                return result;
            }
            ++result.pairsExamined;
            const bool sourceDirection = ab->identity.direction == ABDirection::UpAB &&
                line->direction == TG1A::TrendLineDirection::UTL;
            const bool diagnosticDirection = ab->identity.direction == ABDirection::DownAB &&
                line->direction == TG1A::TrendLineDirection::DTL &&
                configuration.directionalStudyPolicy ==
                    DirectionalStudyPolicy::SymmetricDirectionalDiagnostic;
            if (!sourceDirection && !diagnosticDirection) continue;
            for (double ratio : ratios)
            {
                if (result.measurements.size() == configuration.maxMeasurements)
                {
                    result.measurementCapacityReached = true;
                    return result;
                }
                // lerp implements b + (a-b)*ratio without overflowing a-b.
                const auto fib = Detail::FiniteDouble(std::lerp(ab->bPrice, ab->aPrice, ratio));
                if (!fib) continue;
                Measurement measurement;
                measurement.identity = {ab->identity, Detail::LineIdentity(*line), ratio, decisionBar};
                measurement.fibPrice = *fib;
                measurement.linePrice = line->ProjectedPrice(decisionBar);
                if (validAtr)
                {
                    measurement.normalizedStructureDistance = Detail::NormalizedDistance(
                        *fib, measurement.linePrice, *atr);
                    const double midpoint = std::midpoint(*fib, measurement.linePrice);
                    measurement.normalizedMarketDistance = Detail::NormalizedDistance(close, midpoint, *atr);
                }
                measurement.intersectionThresholdMatch =
                    measurement.normalizedStructureDistance &&
                    *measurement.normalizedStructureDistance <= configuration.intersectionAtrThreshold;
                if (configuration.marketProximityAtrThreshold)
                    measurement.marketProximityThresholdMatch =
                        measurement.normalizedMarketDistance &&
                        *measurement.normalizedMarketDistance <= *configuration.marketProximityAtrThreshold;
                measurement.configuredThresholdMatch = measurement.intersectionThresholdMatch &&
                    measurement.marketProximityThresholdMatch.value_or(true);
                result.measurements.push_back(std::move(measurement));
            }
        }
    return result;
}
} // namespace EA::TG3::TrendlineDiagnostic
