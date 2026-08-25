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

    std::array<float, kEconomicEventFeatureWidth> Ordered() const noexcept;
};

EconomicEventModelFamily MapEconomicEventModelFamily(
    std::string_view sourceAgency,
    std::string_view canonicalEventFamily);

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
        PriceTP timestamp{};
        EconomicEventModelFamily family =
            EconomicEventModelFamily::inflation;
    };

    std::vector<MappedEvent> events_;
    std::size_t nextEventIndex_ = 0;

    std::array<std::optional<PriceTP>, kEconomicEventModelFamilyCount>
        mostRecentEventTimes_{};

    std::optional<PriceTP> previousBarStart_;
};

} // namespace EA::EconomicCalendar
