#ifndef CausalFibonacciExtensionResearch_hpp
#define CausalFibonacciExtensionResearch_hpp

#include "TG4HistoricalEmpiricalEvaluation.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iomanip>
#include <limits>
#include <map>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <utility>
#include <vector>

namespace EA::FibonacciResearch
{

inline constexpr std::string_view kObservationSchemaVersion =
    "causal-fibonacci-extension-observation-v2";
inline constexpr std::string_view kAggregateSchemaVersion =
    "causal-fibonacci-extension-aggregate-v2";
inline constexpr std::string_view kStudyContractVersion =
    "causal-fibonacci-extension-h1-h2-policy-frozen-v2";
inline constexpr std::string_view kEvaluationPolicyVersion =
    "tg2-compatible-20-completed-bar-no-invalidation-v1";
inline constexpr std::size_t kPrimaryEvaluationHorizonBars = 20;
inline constexpr double kFrozenCanonicalFxTolerancePips = 1.0;
inline constexpr double kExtension1272 = 1.272;
inline constexpr double kExtension1618 = 1.618;
inline constexpr double kPullback0382 = 0.382;
inline constexpr double kPullback0500 = 0.500;
inline constexpr double kPullback0618 = 0.618;
inline constexpr double kExpertBenchmarkH2 = 0.80;
inline constexpr double kExpertBenchmarkH3 = 0.80;

enum class LevelType
{
    Extension,
    PullbackRetracement
};

enum class EventType
{
    Extension1272Touched,
    Extension1272Beyond,
    Extension1272RejectionConfirmed,
    Extension1618ReachedAfterBeyond,
    Pullback0382ReachedAfterRejection,
    Pullback0500ReachedAfterRejection,
    Pullback0618ReachedAfterRejection
};

enum class OutcomeState
{
    NotEligible,
    Pending,
    Success,
    Failure,
    RightCensored,
    AmbiguousIneligible
};

enum class CensorReason
{
    None,
    EndOfDataset,
    StudyWindowBoundary
};

enum class BoundarySide
{
    AtOrBelow,
    AtOrAbove
};

enum class DContractStatus
{
    MissingAuthoritativeDefinition
};

enum class Endpoint
{
    H1Extension1618,
    H2Pullback0382,
    H2Pullback0500Descriptive,
    H2Pullback0618Descriptive,
    H3Pullback0382,
    H3Pullback0500,
    H3Pullback0382Or0500
};

enum class CountingUnit
{
    Observation,
    UniqueStructuralEvent
};

struct FibonacciLevel
{
    LevelType type = LevelType::Extension;
    double ratio = 0.0;
    double price = 0.0;
    double zoneLowerPrice = 0.0;
    double zoneUpperPrice = 0.0;
    TG3::ABIdentity sourceLeg;
    std::size_t availabilityBar = 0;
    std::int64_t availabilityTimestamp = 0;
};

struct EventOccurrence
{
    EventType type = EventType::Extension1272Touched;
    std::size_t bar = 0;
    std::int64_t timestamp = 0;
    double observedPrice = 0.0;

    bool operator==(const EventOccurrence&) const = default;
};

struct InvalidationBoundary
{
    BoundarySide side = BoundarySide::AtOrBelow;
    double price = 0.0;

    bool operator==(const InvalidationBoundary&) const = default;
};

// Generic infrastructure retains optional policies for explicitly versioned
// future sensitivity work. The first outcome-bearing H1/H2 study is restricted
// to FrozenProspectiveEvaluationConfiguration below.
struct OutcomePolicy
{
    std::optional<std::size_t> maxBarsAfterEligibility;
    std::optional<InvalidationBoundary> invalidation;

    bool operator==(const OutcomePolicy&) const = default;
};

struct Configuration
{
    // Same deterministic absolute-price convention used by TG3. Touch/reach
    // checks include the near tolerance boundary; close-beyond is strict past
    // the far tolerance boundary.
    double absolutePriceTolerance = 0.0;
    OutcomePolicy h1;
    OutcomePolicy h2;

    bool operator==(const Configuration&) const = default;
};

// Twenty completed bars reuses the already-frozen TG2/TG4A outcome convention.
// No unsupported price invalidation boundary is introduced.
inline Configuration FrozenProspectiveEvaluationConfiguration(
    std::string_view canonicalSymbol)
{
    Configuration result;
    result.absolutePriceTolerance =
        TG4::CanonicalFxPipSize(canonicalSymbol) *
        kFrozenCanonicalFxTolerancePips;
    result.h1.maxBarsAfterEligibility = kPrimaryEvaluationHorizonBars;
    result.h2.maxBarsAfterEligibility = kPrimaryEvaluationHorizonBars;
    return result;
}

inline bool IsFrozenProspectiveEvaluationConfiguration(
    const Configuration& configuration, std::string_view canonicalSymbol)
{
    try
    {
        return configuration.absolutePriceTolerance ==
                TG4::CanonicalFxPipSize(canonicalSymbol) *
                    kFrozenCanonicalFxTolerancePips &&
            configuration.h1.maxBarsAfterEligibility ==
                kPrimaryEvaluationHorizonBars &&
            configuration.h2.maxBarsAfterEligibility ==
                kPrimaryEvaluationHorizonBars &&
            !configuration.h1.invalidation.has_value() &&
            !configuration.h2.invalidation.has_value();
    }
    catch (const std::invalid_argument&)
    {
        return false;
    }
}

struct CausalLabel
{
    std::string value;
    std::size_t availabilityBar = 0;
    std::int64_t availabilityTimestamp = 0;
};

struct CausalBoolean
{
    bool value = false;
    std::size_t availabilityBar = 0;
    std::int64_t availabilityTimestamp = 0;
};

struct GroupingMetadata
{
    std::string symbol;
    std::string timeframe;
    std::string calendarPeriod;
    std::optional<CausalLabel> preexistingVolatilityRegime;
    std::optional<CausalBoolean> tg3Confluence0618;
};

struct DPoint
{
    std::size_t bar = 0;
    std::int64_t timestamp = 0;
    double price = 0.0;
    std::size_t confirmationBar = 0;
    std::int64_t confirmationTimestamp = 0;
};

struct Outcome
{
    OutcomeState state = OutcomeState::NotEligible;
    std::optional<EventOccurrence> eligibility;
    std::optional<EventOccurrence> resolution;
    std::optional<std::size_t> barsToTarget;
    CensorReason censorReason = CensorReason::None;
    std::string detail;
};

struct ObservationRecord
{
    std::string eventIdentity;
    GroupingMetadata grouping;
    Configuration configuration;
    TG3::ABStructure sourceAB;
    FibonacciLevel extension1272;
    FibonacciLevel extension1618;
    FibonacciLevel pullback0382;
    FibonacciLevel pullback0500;
    FibonacciLevel pullback0618;
    std::optional<DPoint> dPoint;
    DContractStatus dContractStatus =
        DContractStatus::MissingAuthoritativeDefinition;

    std::optional<EventOccurrence> extension1272Touched;
    std::optional<EventOccurrence> extension1272Beyond;
    std::optional<EventOccurrence> rejectionConfirmed;
    std::optional<EventOccurrence> extension1618ReachedAfterBeyond;
    std::optional<EventOccurrence> pullback0382ReachedAfterRejection;
    std::optional<EventOccurrence> pullback0500ReachedAfterRejection;
    std::optional<EventOccurrence> pullback0618ReachedAfterRejection;

    Outcome h1;
    Outcome h2;
    Outcome h2Pullback0500Descriptive;
    Outcome h2Pullback0618Descriptive;
    Outcome h3Pullback0382;
    Outcome h3Pullback0500;
    Outcome h3Pullback0382Or0500;
};

struct TimeToTargetSummary
{
    std::size_t count = 0;
    std::optional<double> minimumBars;
    std::optional<double> medianBars;
    std::optional<double> maximumBars;
};

struct EndpointSummary
{
    std::size_t observations = 0;
    std::size_t uniqueStructuralEvents = 0;
    std::size_t eligible = 0;
    std::size_t successes = 0;
    std::size_t failures = 0;
    std::size_t censored = 0;
    std::size_t pending = 0;
    std::size_t ambiguousOrIneligible = 0;
    TG4::WilsonInterval measured;
    std::optional<double> expertBenchmark;
    std::optional<double> measuredMinusBenchmark;
    TimeToTargetSummary timeToTarget;
};

struct CohortKey
{
    std::string symbol;
    std::string direction;
    std::string calendarPeriod;
    std::string volatilityRegime;
    std::string tg3Confluence0618;

    bool operator<(const CohortKey& other) const
    {
        return std::tie(symbol, direction, calendarPeriod, volatilityRegime,
                        tg3Confluence0618) <
            std::tie(other.symbol, other.direction, other.calendarPeriod,
                     other.volatilityRegime, other.tg3Confluence0618);
    }
};

namespace Detail
{

inline bool Finite(double value) { return std::isfinite(value); }

inline std::string Number(double value)
{
    std::ostringstream output;
    output << std::setprecision(std::numeric_limits<double>::max_digits10)
           << value;
    return output.str();
}

inline std::string Csv(std::string_view value)
{
    if (value.find_first_of(",\"\r\n") == std::string_view::npos)
        return std::string(value);
    std::string result = "\"";
    for (const char character : value)
    {
        if (character == '\"') result += '\"';
        result += character;
    }
    result += '\"';
    return result;
}

inline std::uint64_t Fnv1a64(std::string_view text)
{
    std::uint64_t value = 14695981039346656037ULL;
    for (const unsigned char character : text)
    {
        value ^= character;
        value *= 1099511628211ULL;
    }
    return value;
}

inline std::string HashIdentity(std::string_view canonical)
{
    std::ostringstream output;
    output << "fnv1a64:" << std::hex << std::setw(16) << std::setfill('0')
           << Fnv1a64(canonical);
    return output.str();
}

inline int DirectionSign(TG3::ABDirection direction)
{
    return direction == TG3::ABDirection::UpAB ? 1 : -1;
}

inline std::string DirectionName(TG3::ABDirection direction)
{
    return direction == TG3::ABDirection::UpAB ? "up_ab" : "down_ab";
}

inline void ValidateAB(const TG3::ABStructure& structure)
{
    if (!Finite(structure.aPrice) || !Finite(structure.bPrice) ||
        !Finite(structure.priceRange) || !(structure.priceRange > 0.0))
        throw std::invalid_argument(
            "Fibonacci extension geometry requires finite nondegenerate TG3 A/B");
    if ((structure.identity.direction == TG3::ABDirection::UpAB &&
         !(structure.bPrice > structure.aPrice)) ||
        (structure.identity.direction == TG3::ABDirection::DownAB &&
         !(structure.bPrice < structure.aPrice)))
        throw std::invalid_argument(
            "Fibonacci extension A/B prices contradict the TG3 direction");
    if (structure.identity.aBar >= structure.identity.bBar ||
        structure.identity.aTimestamp >= structure.identity.bTimestamp ||
        structure.aConfirmationBar < structure.identity.aBar ||
        structure.bConfirmationBar < structure.identity.bBar ||
        structure.aConfirmationBar > structure.bConfirmationBar ||
        structure.aConfirmationTimestamp < structure.identity.aTimestamp ||
        structure.bConfirmationTimestamp < structure.identity.bTimestamp ||
        structure.aConfirmationTimestamp >
            structure.bConfirmationTimestamp ||
        structure.bConfirmationBar != structure.identity.availabilityBar ||
        structure.bConfirmationTimestamp !=
            structure.identity.availabilityTimestamp)
        throw std::invalid_argument(
            "Fibonacci extension geometry requires causally ordered TG3 A/B");
    const double expected = std::fabs(structure.bPrice - structure.aPrice);
    const double scale = std::max({1.0, expected, structure.priceRange});
    if (std::fabs(expected - structure.priceRange) >
        4.0 * std::numeric_limits<double>::epsilon() * scale)
        throw std::invalid_argument(
            "Fibonacci extension TG3 A/B range is inconsistent with anchors");
}

inline bool BoundaryReached(const InvalidationBoundary& boundary,
                            const TG1A::Candle& candle)
{
    return boundary.side == BoundarySide::AtOrBelow
        ? candle.low <= boundary.price
        : candle.high >= boundary.price;
}

inline std::string OptionalOccurrenceBar(
    const std::optional<EventOccurrence>& occurrence)
{
    return occurrence.has_value() ? std::to_string(occurrence->bar) : "";
}

inline std::string OptionalOccurrenceTimestamp(
    const std::optional<EventOccurrence>& occurrence)
{
    return occurrence.has_value() ? std::to_string(occurrence->timestamp) : "";
}

} // namespace Detail

inline FibonacciLevel CalculateExtensionLevel(
    const TG3::ABStructure& structure, double ratio, double tolerance)
{
    Detail::ValidateAB(structure);
    if (!Detail::Finite(ratio) || ratio <= 1.0)
        throw std::invalid_argument(
            "Fibonacci extension ratio must be finite and greater than one");
    if (!Detail::Finite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument(
            "Fibonacci extension tolerance must be finite and nonnegative");
    const double price = structure.aPrice +
        static_cast<double>(Detail::DirectionSign(structure.identity.direction)) *
        ratio * structure.priceRange;
    if (!Detail::Finite(price))
        throw std::invalid_argument("Fibonacci extension price is non-finite");
    if (!Detail::Finite(price - tolerance) ||
        !Detail::Finite(price + tolerance))
        throw std::invalid_argument(
            "Fibonacci extension tolerance zone is non-finite");
    return {LevelType::Extension, ratio, price, price - tolerance,
            price + tolerance, structure.identity,
            structure.identity.availabilityBar,
            structure.identity.availabilityTimestamp};
}

inline FibonacciLevel CalculatePullbackLevel(
    const TG3::ABStructure& structure, double ratio, double tolerance)
{
    Detail::ValidateAB(structure);
    if (!Detail::Finite(tolerance) || tolerance < 0.0)
        throw std::invalid_argument(
            "Fibonacci pullback tolerance must be finite and nonnegative");
    const auto levels = TG3::FibonacciConfluenceTracker::
        CalculateRetracementLevels(structure, {ratio});
    const double price = levels.front().price;
    if (!Detail::Finite(price - tolerance) ||
        !Detail::Finite(price + tolerance))
        throw std::invalid_argument(
            "Fibonacci pullback tolerance zone is non-finite");
    return {LevelType::PullbackRetracement, ratio, price,
            price - tolerance, price + tolerance, structure.identity,
            structure.identity.availabilityBar,
            structure.identity.availabilityTimestamp};
}

inline std::string StructuralEventIdentity(
    const GroupingMetadata& grouping, const TG3::ABStructure& structure)
{
    Detail::ValidateAB(structure);
    const auto& id = structure.identity;
    const std::string canonical = std::string(kObservationSchemaVersion) + "|" +
        grouping.symbol + "|" + grouping.timeframe + "|" +
        Detail::DirectionName(id.direction) + "|" +
        std::to_string(id.aBar) + "|" + std::to_string(id.aTimestamp) + "|" +
        std::to_string(id.bBar) + "|" + std::to_string(id.bTimestamp) + "|" +
        std::to_string(id.availabilityBar) + "|" +
        std::to_string(id.availabilityTimestamp);
    return Detail::HashIdentity(canonical);
}

class CausalExtensionTracker
{
public:
    CausalExtensionTracker(TG3::ABStructure sourceAB,
                           GroupingMetadata grouping,
                           Configuration configuration)
        : configuration_(std::move(configuration))
    {
        Detail::ValidateAB(sourceAB);
        ValidateConfiguration();
        ValidateCausalGrouping(grouping, sourceAB);
        record_.eventIdentity = StructuralEventIdentity(grouping, sourceAB);
        record_.grouping = std::move(grouping);
        record_.configuration = configuration_;
        record_.sourceAB = std::move(sourceAB);
        record_.extension1272 = CalculateExtensionLevel(
            record_.sourceAB, kExtension1272,
            configuration_.absolutePriceTolerance);
        record_.extension1618 = CalculateExtensionLevel(
            record_.sourceAB, kExtension1618,
            configuration_.absolutePriceTolerance);
        record_.pullback0382 = CalculatePullbackLevel(
            record_.sourceAB, kPullback0382,
            configuration_.absolutePriceTolerance);
        record_.pullback0500 = CalculatePullbackLevel(
            record_.sourceAB, kPullback0500,
            configuration_.absolutePriceTolerance);
        record_.pullback0618 = CalculatePullbackLevel(
            record_.sourceAB, kPullback0618,
            configuration_.absolutePriceTolerance);
        FailClosedH3(record_.h3Pullback0382, "h3_0.382");
        FailClosedH3(record_.h3Pullback0500, "h3_0.500");
        FailClosedH3(record_.h3Pullback0382Or0500, "h3_0.382_or_0.500");
    }

    void AddCompletedBar(std::size_t bar, const TG1A::Candle& candle)
    {
        if (finalized_)
            throw std::logic_error(
                "Fibonacci extension tracker cannot accept bars after finalization");
        ValidateCandle(candle);
        if (lastBar_.has_value() && bar != *lastBar_ + 1)
            throw std::invalid_argument(
                "Fibonacci extension completed bars must be consecutive");
        if (lastTimestamp_.has_value() && candle.timestamp <= *lastTimestamp_)
            throw std::invalid_argument(
                "Fibonacci extension completed bars need increasing timestamps");
        lastBar_ = bar;
        lastTimestamp_ = candle.timestamp;

        const auto& identity = record_.sourceAB.identity;
        if (bar < identity.availabilityBar) return;
        if (bar == identity.availabilityBar &&
            candle.timestamp != identity.availabilityTimestamp)
            throw std::invalid_argument(
                "Fibonacci extension source availability timestamp mismatch");
        if (candle.timestamp < identity.availabilityTimestamp) return;

        ObservePrimaryEvents(bar, candle);
        ResolveOutcome(record_.h1, configuration_.h1,
                       record_.extension1272Beyond,
                       record_.extension1618ReachedAfterBeyond,
                       bar, candle, "h1_1.618_after_1.272_beyond");
        ResolveOutcome(record_.h2, configuration_.h2,
                       record_.rejectionConfirmed,
                       record_.pullback0382ReachedAfterRejection,
                       bar, candle, "h2_0.382_after_rejection");
        ResolveOutcome(record_.h2Pullback0500Descriptive, configuration_.h2,
                       record_.rejectionConfirmed,
                       record_.pullback0500ReachedAfterRejection,
                       bar, candle,
                       "h2_descriptive_0.500_after_rejection");
        ResolveOutcome(record_.h2Pullback0618Descriptive, configuration_.h2,
                       record_.rejectionConfirmed,
                       record_.pullback0618ReachedAfterRejection,
                       bar, candle,
                       "h2_secondary_0.618_after_rejection");
    }

    void Finalize(CensorReason reason = CensorReason::EndOfDataset)
    {
        if (finalized_) return;
        if (reason == CensorReason::None)
            throw std::invalid_argument(
                "Fibonacci extension finalization requires a censor reason");
        CensorPending(record_.h1, reason);
        CensorPending(record_.h2, reason);
        CensorPending(record_.h2Pullback0500Descriptive, reason);
        CensorPending(record_.h2Pullback0618Descriptive, reason);
        finalized_ = true;
    }

    const ObservationRecord& Record() const { return record_; }
    const Configuration& GetConfiguration() const { return configuration_; }
    bool IsFinalized() const { return finalized_; }

private:
    Configuration configuration_;
    ObservationRecord record_;
    std::optional<std::size_t> lastBar_;
    std::optional<std::int64_t> lastTimestamp_;
    bool finalized_ = false;

    static void ValidateCandle(const TG1A::Candle& candle)
    {
        if (!Detail::Finite(candle.open) || !Detail::Finite(candle.high) ||
            !Detail::Finite(candle.low) || !Detail::Finite(candle.close) ||
            candle.high < candle.low || candle.open < candle.low ||
            candle.open > candle.high || candle.close < candle.low ||
            candle.close > candle.high)
            throw std::invalid_argument(
                "Fibonacci extension tracker received an invalid completed bar");
    }

    void ValidateConfiguration() const
    {
        if (!Detail::Finite(configuration_.absolutePriceTolerance) ||
            configuration_.absolutePriceTolerance < 0.0)
            throw std::invalid_argument(
                "Fibonacci extension tolerance must be finite and nonnegative");
        ValidatePolicy(configuration_.h1);
        ValidatePolicy(configuration_.h2);
    }

    static void ValidatePolicy(const OutcomePolicy& policy)
    {
        if (policy.maxBarsAfterEligibility.has_value() &&
            *policy.maxBarsAfterEligibility == 0)
            throw std::invalid_argument(
                "Fibonacci extension outcome horizon must be positive");
        if (policy.invalidation.has_value() &&
            !Detail::Finite(policy.invalidation->price))
            throw std::invalid_argument(
                "Fibonacci extension invalidation price must be finite");
    }

    static void ValidateCausalGrouping(
        const GroupingMetadata& grouping, const TG3::ABStructure& source)
    {
        const auto validate = [&source](std::size_t bar,
                                        std::int64_t timestamp)
        {
            if (bar > source.identity.availabilityBar ||
                timestamp > source.identity.availabilityTimestamp)
                throw std::invalid_argument(
                    "Fibonacci extension grouping must be available no later than the source leg");
        };
        if (grouping.preexistingVolatilityRegime.has_value())
            validate(grouping.preexistingVolatilityRegime->availabilityBar,
                     grouping.preexistingVolatilityRegime->availabilityTimestamp);
        if (grouping.tg3Confluence0618.has_value())
            validate(grouping.tg3Confluence0618->availabilityBar,
                     grouping.tg3Confluence0618->availabilityTimestamp);
    }

    bool ExtensionReached(const FibonacciLevel& level,
                          const TG1A::Candle& candle) const
    {
        return record_.sourceAB.identity.direction == TG3::ABDirection::UpAB
            ? candle.high >= level.zoneLowerPrice
            : candle.low <= level.zoneUpperPrice;
    }

    bool CloseBeyond(const FibonacciLevel& level,
                     const TG1A::Candle& candle) const
    {
        return record_.sourceAB.identity.direction == TG3::ABDirection::UpAB
            ? candle.close > level.zoneUpperPrice
            : candle.close < level.zoneLowerPrice;
    }

    bool CloseReturnedThroughNearEdge(const FibonacciLevel& level,
                                      const TG1A::Candle& candle) const
    {
        return record_.sourceAB.identity.direction == TG3::ABDirection::UpAB
            ? candle.close < level.zoneLowerPrice
            : candle.close > level.zoneUpperPrice;
    }

    bool PullbackReached(const FibonacciLevel& level,
                         const TG1A::Candle& candle) const
    {
        return record_.sourceAB.identity.direction == TG3::ABDirection::UpAB
            ? candle.low <= level.zoneUpperPrice
            : candle.high >= level.zoneLowerPrice;
    }

    EventOccurrence Occurrence(EventType type, std::size_t bar,
                               const TG1A::Candle& candle,
                               double observedPrice) const
    {
        return {type, bar, candle.timestamp, observedPrice};
    }

    void ObservePrimaryEvents(std::size_t bar, const TG1A::Candle& candle)
    {
        const bool up = record_.sourceAB.identity.direction ==
            TG3::ABDirection::UpAB;
        if (!record_.extension1272Touched.has_value() &&
            ExtensionReached(record_.extension1272, candle))
            record_.extension1272Touched = Occurrence(
                EventType::Extension1272Touched, bar, candle,
                up ? candle.high : candle.low);
        if (!record_.extension1272Beyond.has_value() &&
            CloseBeyond(record_.extension1272, candle))
            record_.extension1272Beyond = Occurrence(
                EventType::Extension1272Beyond, bar, candle, candle.close);

        if (!record_.rejectionConfirmed.has_value() &&
            record_.extension1272Touched.has_value() &&
            bar > record_.extension1272Touched->bar &&
            CloseReturnedThroughNearEdge(record_.extension1272, candle))
            record_.rejectionConfirmed = Occurrence(
                EventType::Extension1272RejectionConfirmed,
                bar, candle, candle.close);

        if (record_.h1.state == OutcomeState::Pending &&
            !record_.extension1618ReachedAfterBeyond.has_value() &&
            record_.extension1272Beyond.has_value() &&
            bar > record_.extension1272Beyond->bar &&
            ExtensionReached(record_.extension1618, candle))
            record_.extension1618ReachedAfterBeyond = Occurrence(
                EventType::Extension1618ReachedAfterBeyond, bar, candle,
                up ? candle.high : candle.low);

        if (record_.rejectionConfirmed.has_value() &&
            bar > record_.rejectionConfirmed->bar)
        {
            if (record_.h2.state == OutcomeState::Pending)
                ObservePullback(record_.pullback0382,
                    EventType::Pullback0382ReachedAfterRejection,
                    record_.pullback0382ReachedAfterRejection, bar, candle);
            if (record_.h2Pullback0500Descriptive.state ==
                OutcomeState::Pending)
                ObservePullback(record_.pullback0500,
                    EventType::Pullback0500ReachedAfterRejection,
                    record_.pullback0500ReachedAfterRejection, bar, candle);
            if (record_.h2Pullback0618Descriptive.state ==
                OutcomeState::Pending)
                ObservePullback(record_.pullback0618,
                    EventType::Pullback0618ReachedAfterRejection,
                    record_.pullback0618ReachedAfterRejection, bar, candle);
        }
    }

    void ObservePullback(const FibonacciLevel& level, EventType type,
                         std::optional<EventOccurrence>& destination,
                         std::size_t bar, const TG1A::Candle& candle)
    {
        if (destination.has_value() || !PullbackReached(level, candle)) return;
        const bool up = record_.sourceAB.identity.direction ==
            TG3::ABDirection::UpAB;
        destination = Occurrence(type, bar, candle,
                                 up ? candle.low : candle.high);
    }

    static void StartOutcome(Outcome& outcome,
                             const std::optional<EventOccurrence>& eligibility,
                             std::string_view detail)
    {
        if (outcome.state == OutcomeState::NotEligible && eligibility.has_value())
        {
            outcome.state = OutcomeState::Pending;
            outcome.eligibility = eligibility;
            outcome.detail = std::string(detail);
        }
    }

    static void ResolveOutcome(
        Outcome& outcome, const OutcomePolicy& policy,
        const std::optional<EventOccurrence>& eligibility,
        const std::optional<EventOccurrence>& target,
        std::size_t bar, const TG1A::Candle& candle, std::string_view detail)
    {
        StartOutcome(outcome, eligibility, detail);
        if (outcome.state != OutcomeState::Pending ||
            !outcome.eligibility.has_value() ||
            bar <= outcome.eligibility->bar)
            return;
        const bool targetNow = target.has_value() && target->bar == bar;
        const bool invalidatedNow = policy.invalidation.has_value() &&
            Detail::BoundaryReached(*policy.invalidation, candle);
        if (targetNow && invalidatedNow)
        {
            outcome.state = OutcomeState::AmbiguousIneligible;
            outcome.resolution = target;
            outcome.detail += ":same_bar_target_and_invalidation_order_unknown";
            return;
        }
        if (targetNow)
        {
            outcome.state = OutcomeState::Success;
            outcome.resolution = target;
            outcome.barsToTarget = bar - outcome.eligibility->bar;
            return;
        }
        if (invalidatedNow)
        {
            outcome.state = OutcomeState::Failure;
            outcome.resolution = EventOccurrence{
                eligibility->type, bar, candle.timestamp,
                policy.invalidation->price};
            return;
        }
        if (policy.maxBarsAfterEligibility.has_value() &&
            bar - outcome.eligibility->bar >=
                *policy.maxBarsAfterEligibility)
        {
            outcome.state = OutcomeState::Failure;
            outcome.resolution = EventOccurrence{
                eligibility->type, bar, candle.timestamp, candle.close};
            outcome.detail += ":frozen_horizon_expired";
        }
    }

    static void FailClosedH3(Outcome& outcome, std::string_view endpoint)
    {
        outcome.state = OutcomeState::AmbiguousIneligible;
        outcome.detail = std::string(endpoint) +
            ":missing_authoritative_d_extension_contract";
    }

    static void CensorPending(Outcome& outcome, CensorReason reason)
    {
        if (outcome.state != OutcomeState::Pending) return;
        outcome.state = OutcomeState::RightCensored;
        outcome.censorReason = reason;
    }
};

inline const Outcome& OutcomeFor(const ObservationRecord& record,
                                 Endpoint endpoint)
{
    switch (endpoint)
    {
        case Endpoint::H1Extension1618: return record.h1;
        case Endpoint::H2Pullback0382: return record.h2;
        case Endpoint::H2Pullback0500Descriptive:
            return record.h2Pullback0500Descriptive;
        case Endpoint::H2Pullback0618Descriptive:
            return record.h2Pullback0618Descriptive;
        case Endpoint::H3Pullback0382: return record.h3Pullback0382;
        case Endpoint::H3Pullback0500: return record.h3Pullback0500;
        case Endpoint::H3Pullback0382Or0500:
            return record.h3Pullback0382Or0500;
    }
    throw std::logic_error("Unknown Fibonacci research endpoint");
}

inline std::string ObservationCsvRow(const ObservationRecord& record);

class AggregateAccumulator
{
public:
    void Add(const ObservationRecord& record)
    {
        if (!IsFrozenProspectiveEvaluationConfiguration(
                record.configuration, record.grouping.symbol))
            throw std::invalid_argument(
                "Fibonacci aggregates require the frozen prospective H1/H2 configuration");
        const auto [position, inserted] =
            unique_.try_emplace(record.eventIdentity, record);
        if (!inserted && ObservationCsvRow(position->second) !=
                         ObservationCsvRow(record))
            throw std::invalid_argument(
                "Fibonacci structural event identity has inconsistent materializations");
        observations_.push_back(record);
    }

    EndpointSummary Summarize(Endpoint endpoint, CountingUnit unit) const
    {
        EndpointSummary result;
        result.observations = observations_.size();
        result.uniqueStructuralEvents = unique_.size();
        std::vector<std::size_t> times;
        const auto add = [&](const ObservationRecord& record)
        {
            const Outcome& outcome = OutcomeFor(record, endpoint);
            switch (outcome.state)
            {
                case OutcomeState::NotEligible:
                case OutcomeState::AmbiguousIneligible:
                    ++result.ambiguousOrIneligible;
                    break;
                case OutcomeState::Pending:
                    ++result.eligible;
                    ++result.pending;
                    break;
                case OutcomeState::Success:
                    ++result.eligible;
                    ++result.successes;
                    if (outcome.barsToTarget.has_value())
                        times.push_back(*outcome.barsToTarget);
                    break;
                case OutcomeState::Failure:
                    ++result.eligible;
                    ++result.failures;
                    break;
                case OutcomeState::RightCensored:
                    ++result.eligible;
                    ++result.censored;
                    break;
            }
        };
        if (unit == CountingUnit::Observation)
            for (const auto& record : observations_) add(record);
        else
            for (const auto& [identity, record] : unique_)
            {
                (void)identity;
                add(record);
            }
        result.measured = TG4::Wilson95(result.successes, result.failures);
        if (endpoint == Endpoint::H2Pullback0382)
            result.expertBenchmark = kExpertBenchmarkH2;
        else if (endpoint == Endpoint::H3Pullback0382 ||
                 endpoint == Endpoint::H3Pullback0500 ||
                 endpoint == Endpoint::H3Pullback0382Or0500)
            result.expertBenchmark = kExpertBenchmarkH3;
        if (result.measured.rate.has_value() &&
            result.expertBenchmark.has_value())
            result.measuredMinusBenchmark =
                *result.measured.rate - *result.expertBenchmark;
        result.timeToTarget = SummarizeTimes(std::move(times));
        return result;
    }

    std::map<CohortKey, EndpointSummary> SummarizeCohorts(
        Endpoint endpoint, CountingUnit unit) const;

private:
    std::vector<ObservationRecord> observations_;
    std::map<std::string, ObservationRecord> unique_;

    static TimeToTargetSummary SummarizeTimes(std::vector<std::size_t> values)
    {
        TimeToTargetSummary result;
        result.count = values.size();
        if (values.empty()) return result;
        std::sort(values.begin(), values.end());
        result.minimumBars = static_cast<double>(values.front());
        result.maximumBars = static_cast<double>(values.back());
        const std::size_t middle = values.size() / 2;
        result.medianBars = values.size() % 2 == 1
            ? static_cast<double>(values[middle])
            : (static_cast<double>(values[middle - 1]) +
               static_cast<double>(values[middle])) / 2.0;
        return result;
    }
};

inline std::string OutcomeTemporalPartitionName(const Outcome& outcome)
{
    return outcome.eligibility.has_value()
        ? TG4::TemporalPartitionName(TG4::PartitionForTimestamp(
              outcome.eligibility->timestamp))
        : std::string{};
}

inline CohortKey CohortFor(const ObservationRecord& record, Endpoint endpoint)
{
    const auto& grouping = record.grouping;
    const Outcome& outcome = OutcomeFor(record, endpoint);
    const std::string outcomePeriod = OutcomeTemporalPartitionName(outcome);
    return {
        grouping.symbol,
        Detail::DirectionName(record.sourceAB.identity.direction),
        outcomePeriod.empty() ? grouping.calendarPeriod : outcomePeriod,
        grouping.preexistingVolatilityRegime.has_value()
            ? grouping.preexistingVolatilityRegime->value : "not_supplied",
        grouping.tg3Confluence0618.has_value()
            ? (grouping.tg3Confluence0618->value ? "confluent" :
                                                   "not_confluent")
            : "not_supplied"};
}

inline std::map<CohortKey, EndpointSummary>
AggregateAccumulator::SummarizeCohorts(
    Endpoint endpoint, CountingUnit unit) const
{
    std::map<CohortKey, AggregateAccumulator> accumulators;
    for (const ObservationRecord& record : observations_)
        accumulators[CohortFor(record, endpoint)].Add(record);
    std::map<CohortKey, EndpointSummary> result;
    for (const auto& [key, accumulator] : accumulators)
        result.emplace(key, accumulator.Summarize(endpoint, unit));
    return result;
}

inline std::string OutcomeStateName(OutcomeState state)
{
    switch (state)
    {
        case OutcomeState::NotEligible: return "not_eligible";
        case OutcomeState::Pending: return "pending";
        case OutcomeState::Success: return "success";
        case OutcomeState::Failure: return "failure";
        case OutcomeState::RightCensored: return "right_censored";
        case OutcomeState::AmbiguousIneligible:
            return "ambiguous_or_ineligible";
    }
    return "ambiguous_or_ineligible";
}

inline std::string CensorReasonName(CensorReason reason)
{
    switch (reason)
    {
        case CensorReason::None: return "none";
        case CensorReason::EndOfDataset: return "end_of_dataset";
        case CensorReason::StudyWindowBoundary:
            return "study_window_boundary";
    }
    return "none";
}

inline std::string ObservationCsvHeader()
{
    return "schema_version,study_contract_version,evaluation_policy_version,"
        "event_identity,symbol,timeframe,source_calendar_period,"
        "h1_temporal_partition,h2_temporal_partition,"
        "absolute_price_tolerance,h1_horizon_bars,"
        "h1_invalidation_side,h1_invalidation_price,h2_horizon_bars,"
        "h2_invalidation_side,h2_invalidation_price,"
        "direction,source_a_bar,source_a_timestamp,source_a_price,"
        "source_a_confirmation_bar,source_a_confirmation_timestamp,"
        "source_b_bar,source_b_timestamp,source_b_price,"
        "source_b_confirmation_bar,source_b_confirmation_timestamp,"
        "source_availability_bar,source_availability_timestamp,"
        "extension_1_272_price,extension_1_618_price,pullback_0_382_price,"
        "pullback_0_500_price,pullback_0_618_price,d_bar,d_timestamp,d_price,"
        "d_confirmation_bar,d_confirmation_timestamp,d_contract_status,"
        "touch_1_272_bar,touch_1_272_timestamp,beyond_1_272_bar,"
        "beyond_1_272_timestamp,rejection_bar,rejection_timestamp,"
        "hit_1_618_bar,hit_1_618_timestamp,hit_0_382_bar,hit_0_382_timestamp,"
        "hit_0_500_bar,hit_0_500_timestamp,hit_0_618_bar,hit_0_618_timestamp,"
        "h1_state,h1_censor_reason,h1_bars_to_target,h1_detail,h2_state,"
        "h2_censor_reason,h2_bars_to_target,h2_detail,h2_0_500_state,"
        "h2_0_500_censor_reason,h2_0_500_bars_to_target,h2_0_500_detail,"
        "h2_0_618_state,h2_0_618_censor_reason,h2_0_618_bars_to_target,"
        "h2_0_618_detail,h3_0_382_state,"
        "h3_0_500_state,h3_union_state,h3_detail,volatility_regime,"
        "tg3_confluence_0_618";
}

inline std::string ObservationCsvRow(const ObservationRecord& record)
{
    const auto& ab = record.sourceAB;
    const auto& id = ab.identity;
    const auto optionalSize = [](const std::optional<std::size_t>& value)
    {
        return value.has_value() ? std::to_string(*value) : std::string{};
    };
    const auto boundarySide = [](const OutcomePolicy& policy)
    {
        if (!policy.invalidation.has_value()) return std::string{};
        return policy.invalidation->side == BoundarySide::AtOrBelow
            ? std::string("at_or_below") : std::string("at_or_above");
    };
    const auto boundaryPrice = [](const OutcomePolicy& policy)
    {
        return policy.invalidation.has_value()
            ? Detail::Number(policy.invalidation->price) : std::string{};
    };
    const auto dBar = record.dPoint.has_value()
        ? std::to_string(record.dPoint->bar) : std::string{};
    const auto dTimestamp = record.dPoint.has_value()
        ? std::to_string(record.dPoint->timestamp) : std::string{};
    const auto dPrice = record.dPoint.has_value()
        ? Detail::Number(record.dPoint->price) : std::string{};
    const auto dConfirmationBar = record.dPoint.has_value()
        ? std::to_string(record.dPoint->confirmationBar) : std::string{};
    const auto dConfirmationTimestamp = record.dPoint.has_value()
        ? std::to_string(record.dPoint->confirmationTimestamp) : std::string{};
    const auto regime = record.grouping.preexistingVolatilityRegime.has_value()
        ? record.grouping.preexistingVolatilityRegime->value : std::string{};
    const auto confluence = record.grouping.tg3Confluence0618.has_value()
        ? (record.grouping.tg3Confluence0618->value ? "true" : "false")
        : "";
    std::vector<std::string> fields{
        std::string(kObservationSchemaVersion),
        std::string(kStudyContractVersion),
        std::string(kEvaluationPolicyVersion), record.eventIdentity,
        record.grouping.symbol, record.grouping.timeframe,
        record.grouping.calendarPeriod,
        OutcomeTemporalPartitionName(record.h1),
        OutcomeTemporalPartitionName(record.h2),
        Detail::Number(record.configuration.absolutePriceTolerance),
        optionalSize(record.configuration.h1.maxBarsAfterEligibility),
        boundarySide(record.configuration.h1),
        boundaryPrice(record.configuration.h1),
        optionalSize(record.configuration.h2.maxBarsAfterEligibility),
        boundarySide(record.configuration.h2),
        boundaryPrice(record.configuration.h2),
        Detail::DirectionName(id.direction),
        std::to_string(id.aBar), std::to_string(id.aTimestamp),
        Detail::Number(ab.aPrice), std::to_string(ab.aConfirmationBar),
        std::to_string(ab.aConfirmationTimestamp), std::to_string(id.bBar),
        std::to_string(id.bTimestamp), Detail::Number(ab.bPrice),
        std::to_string(ab.bConfirmationBar),
        std::to_string(ab.bConfirmationTimestamp),
        std::to_string(id.availabilityBar),
        std::to_string(id.availabilityTimestamp),
        Detail::Number(record.extension1272.price),
        Detail::Number(record.extension1618.price),
        Detail::Number(record.pullback0382.price),
        Detail::Number(record.pullback0500.price),
        Detail::Number(record.pullback0618.price), dBar, dTimestamp, dPrice,
        dConfirmationBar, dConfirmationTimestamp,
        "missing_authoritative_definition",
        Detail::OptionalOccurrenceBar(record.extension1272Touched),
        Detail::OptionalOccurrenceTimestamp(record.extension1272Touched),
        Detail::OptionalOccurrenceBar(record.extension1272Beyond),
        Detail::OptionalOccurrenceTimestamp(record.extension1272Beyond),
        Detail::OptionalOccurrenceBar(record.rejectionConfirmed),
        Detail::OptionalOccurrenceTimestamp(record.rejectionConfirmed),
        Detail::OptionalOccurrenceBar(record.extension1618ReachedAfterBeyond),
        Detail::OptionalOccurrenceTimestamp(record.extension1618ReachedAfterBeyond),
        Detail::OptionalOccurrenceBar(record.pullback0382ReachedAfterRejection),
        Detail::OptionalOccurrenceTimestamp(record.pullback0382ReachedAfterRejection),
        Detail::OptionalOccurrenceBar(record.pullback0500ReachedAfterRejection),
        Detail::OptionalOccurrenceTimestamp(record.pullback0500ReachedAfterRejection),
        Detail::OptionalOccurrenceBar(record.pullback0618ReachedAfterRejection),
        Detail::OptionalOccurrenceTimestamp(record.pullback0618ReachedAfterRejection),
        OutcomeStateName(record.h1.state), CensorReasonName(record.h1.censorReason),
        optionalSize(record.h1.barsToTarget), record.h1.detail,
        OutcomeStateName(record.h2.state), CensorReasonName(record.h2.censorReason),
        optionalSize(record.h2.barsToTarget), record.h2.detail,
        OutcomeStateName(record.h2Pullback0500Descriptive.state),
        CensorReasonName(record.h2Pullback0500Descriptive.censorReason),
        optionalSize(record.h2Pullback0500Descriptive.barsToTarget),
        record.h2Pullback0500Descriptive.detail,
        OutcomeStateName(record.h2Pullback0618Descriptive.state),
        CensorReasonName(record.h2Pullback0618Descriptive.censorReason),
        optionalSize(record.h2Pullback0618Descriptive.barsToTarget),
        record.h2Pullback0618Descriptive.detail,
        OutcomeStateName(record.h3Pullback0382.state),
        OutcomeStateName(record.h3Pullback0500.state),
        OutcomeStateName(record.h3Pullback0382Or0500.state),
        record.h3Pullback0382Or0500.detail, regime,
        confluence};
    std::ostringstream output;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0) output << ',';
        output << Detail::Csv(fields[index]);
    }
    return output.str();
}

inline std::string EndpointName(Endpoint endpoint)
{
    switch (endpoint)
    {
        case Endpoint::H1Extension1618:
            return "h1_1.618_after_1.272_beyond";
        case Endpoint::H2Pullback0382:
            return "h2_0.382_after_rejection";
        case Endpoint::H2Pullback0500Descriptive:
            return "h2_descriptive_0.500_after_rejection";
        case Endpoint::H2Pullback0618Descriptive:
            return "h2_secondary_0.618_after_rejection";
        case Endpoint::H3Pullback0382: return "h3_0.382";
        case Endpoint::H3Pullback0500: return "h3_0.500";
        case Endpoint::H3Pullback0382Or0500:
            return "h3_0.382_or_0.500";
    }
    return "unknown";
}

inline std::string EndpointSummaryCsvHeader()
{
    return "schema_version,study_contract_version,evaluation_policy_version,"
        "endpoint,counting_unit,"
        "observation_rows,unique_structural_events,eligible_n,successes,"
        "failures,censored,pending,ambiguous_or_ineligible,resolved_n,"
        "measured_probability,wilson_95_lower,wilson_95_upper,"
        "expert_benchmark,measured_minus_benchmark,time_to_target_n,"
        "minimum_bars_to_target,median_bars_to_target,maximum_bars_to_target";
}

inline std::string EndpointSummaryCsvRow(
    Endpoint endpoint, CountingUnit unit, const EndpointSummary& summary)
{
    const auto optional = [](const std::optional<double>& value)
    {
        return value.has_value() ? Detail::Number(*value) : std::string{};
    };
    std::vector<std::string> fields{
        std::string(kAggregateSchemaVersion), std::string(kStudyContractVersion),
        std::string(kEvaluationPolicyVersion), EndpointName(endpoint),
        unit == CountingUnit::Observation ? "observation" :
                                            "unique_structural_event",
        std::to_string(summary.observations),
        std::to_string(summary.uniqueStructuralEvents),
        std::to_string(summary.eligible), std::to_string(summary.successes),
        std::to_string(summary.failures), std::to_string(summary.censored),
        std::to_string(summary.pending),
        std::to_string(summary.ambiguousOrIneligible),
        std::to_string(summary.measured.denominator),
        optional(summary.measured.rate), optional(summary.measured.lower95),
        optional(summary.measured.upper95), optional(summary.expertBenchmark),
        optional(summary.measuredMinusBenchmark),
        std::to_string(summary.timeToTarget.count),
        optional(summary.timeToTarget.minimumBars),
        optional(summary.timeToTarget.medianBars),
        optional(summary.timeToTarget.maximumBars)};
    std::ostringstream output;
    for (std::size_t index = 0; index < fields.size(); ++index)
    {
        if (index != 0) output << ',';
        output << Detail::Csv(fields[index]);
    }
    return output.str();
}

} // namespace EA::FibonacciResearch

#endif /* CausalFibonacciExtensionResearch_hpp */
