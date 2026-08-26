#pragma once

#include "EconomicEventBarAlignment.hpp"
#include "EconomicEventFeatureLayout.hpp"

#include <array>
#include <cstddef>
#include <optional>
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

    // Reserved append-only channels. Persisted provider actuals do not prove
    // first-release/revision provenance, so the current v4 contract leaves all
    // four values zero for every event. They may be activated only by a future
    // explicit persisted release-time actual provenance contract.
    float releasedEventHasSurprise = 0.0F;
    float releasedEventSurprise = 0.0F;
    float releasedEventSurpriseAbs = 0.0F;
    float releasedEventSurpriseDirection = 0.0F;

    std::array<float, kEconomicEventFeatureWidth> Ordered() const noexcept;
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

private:
    struct MappedEvent
    {
        long long economicEventId = 0;
        PriceTP timestamp{};
        EconomicEventModelFamily family =
            EconomicEventModelFamily::inflation;
        std::string eventFamily;
        int eventImportance = 0;
        std::optional<EconomicEventSelectedConsensus> selectedConsensus;
    };

    std::vector<MappedEvent> events_;
    std::size_t nextEventIndex_ = 0;

    std::array<std::optional<PriceTP>, kEconomicEventModelFamilyCount>
        mostRecentEventTimes_{};

    std::optional<std::size_t> mostRecentReleasedEventIndex_;

    std::optional<PriceTP> previousBarStart_;
};

} // namespace EA::EconomicCalendar
