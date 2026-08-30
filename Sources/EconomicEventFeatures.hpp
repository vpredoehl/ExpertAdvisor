#pragma once

#include "EconomicEventBarAlignment.hpp"
#include "EconomicEventFeatureLayout.hpp"

#include <array>
#include <cstddef>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace EA::EconomicCalendar
{

enum class EconomicEventModelFamily : std::size_t
{
    inflation = 0,
    employment = 1,
    growth = 2,
    fedPolicy = 3,
    consumerDemand = 4,
    count = 5,
};

inline constexpr std::size_t kEconomicEventModelFamilyCount =
    static_cast<std::size_t>(EconomicEventModelFamily::count);

inline constexpr std::string_view kEconomicEventFeatureCurrency = "USD";

// For a completed bar cutoff and the most recent causal event timestamp:
//
//     decay = exp(-(cutoff - eventTime) / 86400 seconds)
//
// A family with no causal history has decay zero.
inline constexpr std::chrono::seconds kEconomicEventRecencyTimeConstant{
    24 * 60 * 60};

// Static normalization constants. Canonical percentage values are percentage
// points; count values have already had their persisted source scale applied.
// These constants never depend on future observations or dataset statistics.
inline constexpr double kEconomicPercentNormalizationScale = 10.0;
inline constexpr double kEconomicEmploymentNormalizationScale = 1'000'000.0;
inline constexpr double kEconomicJoltsNormalizationScale = 10'000'000.0;

// Named fields are the authoritative public representation. Ordered() is the
// stable append-only Tensor integration order used by both training and
// inference.
struct EconomicEventFeatureValues
{
    float inflationEvent = 0.0F;
    float employmentEvent = 0.0F;
    float growthEvent = 0.0F;
    float fedPolicyEvent = 0.0F;
    float consumerDemandEvent = 0.0F;

    float inflationRecencyDecay = 0.0F;
    float employmentRecencyDecay = 0.0F;
    float growthRecencyDecay = 0.0F;
    float fedPolicyRecencyDecay = 0.0F;
    float consumerDemandRecencyDecay = 0.0F;

    // Consensus context is the event exactly at the completed information
    // cutoff, if any, otherwise the most recent causally released event. This
    // avoids leaking a final pre-release forecast into arbitrarily early bars.
    // Scalar forecasts duplicate their value in low/high; range forecasts
    // retain both endpoints and set relevantEventConsensusIsRange.
    float relevantEventHasConsensus = 0.0F;
    float relevantEventConsensusLow = 0.0F;
    float relevantEventConsensusHigh = 0.0F;
    float relevantEventConsensusIsRange = 0.0F;

    // Closed semantic-layout-v4 channels. Width-71 models were trained while
    // these positions were reserved zeros, so they remain zero permanently.
    float releasedEventHasSurprise = 0.0F;
    float releasedEventSurprise = 0.0F;
    float releasedEventSurpriseAbs = 0.0F;
    float releasedEventSurpriseDirection = 0.0F;

    // Semantic-layout-v5 append. These activate only for a scalar selected
    // forecast paired with the provenance-certified authoritative initial
    // actual after its strict available-at boundary. Provider actuals, later
    // revisions, ranges, and incompatible semantics remain unavailable.
    float authoritativeInitialHasSurprise = 0.0F;
    float authoritativeInitialSurprise = 0.0F;
    float authoritativeInitialSurpriseAbs = 0.0F;
    float authoritativeInitialSurpriseDirection = 0.0F;

    std::array<float, kEconomicEventFeatureWidth> Ordered() const noexcept;
};

// Read-only row-level availability diagnostics. Provider identity never
// changes model values, but retaining its distribution here makes the unified
// selected-consensus provenance auditable in production verification.
struct EconomicEventFeatureAvailabilityDiagnostics
{
    std::size_t completedBarCount = 0;
    std::size_t relevantEventRowCount = 0;
    std::size_t selectedConsensusRowCount = 0;
    std::size_t scalarConsensusRowCount = 0;
    std::size_t rangeConsensusRowCount = 0;
    std::size_t missingConsensusRowCount = 0;
    std::map<std::string, std::size_t> selectedConsensusProviderRowCounts;
    std::size_t selectedInitialActualRowCount = 0;
    std::size_t notYetAvailableInitialActualRowCount = 0;
    std::size_t incompatibleInitialActualRowCount = 0;
    std::size_t availableSurpriseRowCount = 0;
    std::map<std::string, std::size_t> initialActualSourceRowCounts;
    bool operator==(const EconomicEventFeatureAvailabilityDiagnostics&) const =
        default;
};

EconomicEventModelFamily MapEconomicEventModelFamily(
    std::string_view sourceAgency,
    std::string_view canonicalEventFamily);

double EconomicEventNormalizationScale(
    std::string_view canonicalEventFamily,
    std::string_view canonicalUnit);

// Shared chronological implementation for future training and inference use.
//
// input15m timestamps are bar starts. AdvanceCompletedBar(barStart) evaluates
// the completed information set at barStart + 15 minutes. An event is causal
// only when eventTime < that cutoff. Consequently, an event exactly on the
// next bar boundary cannot leak into the preceding bar and is first included
// on the bar beginning at that boundary. The binary indicator is one only for
// an event contained in the actual interval [barStart, cutoff); an event that
// occurred during a market-data gap updates recency without being relabeled as
// an event that occurred inside the first post-gap bar.
class EconomicEventFeatureEngine
{
public:
    explicit EconomicEventFeatureEngine(
        std::vector<EconomicEvent> chronologicalEvents);

    EconomicEventFeatureValues AdvanceCompletedBar(
        PriceTP barStart);

    std::size_t ConsumedEventCount() const noexcept;

    const EconomicEventFeatureAvailabilityDiagnostics& Diagnostics()
        const noexcept;

private:
    struct MappedReleaseActual
    {
        // Preserve the database's microsecond timestamp exactly. Converting
        // it to the second-resolution market timestamp would risk making a
        // sub-second publication visible before its actual known-at instant.
        std::int64_t availableAtUnixMicros = 0;
        EconomicEventReleaseActual provenance;
    };

    struct MappedEvent
    {
        long long economicEventId = 0;
        PriceTP timestamp{};
        EconomicEventModelFamily family =
            EconomicEventModelFamily::inflation;
        std::string eventFamily;
        int eventImportance = 0;
        std::optional<EconomicEventSelectedConsensus> selectedConsensus;
        std::optional<MappedReleaseActual> releaseActual;
    };

    std::vector<MappedEvent> events_;
    std::size_t nextEventIndex_ = 0;

    std::array<std::optional<PriceTP>, kEconomicEventModelFamilyCount>
        mostRecentEventTimes_{};

    std::optional<std::size_t> mostRecentReleasedEventIndex_;

    std::optional<PriceTP> previousBarStart_;
    EconomicEventFeatureAvailabilityDiagnostics diagnostics_;
};

} // namespace EA::EconomicCalendar
