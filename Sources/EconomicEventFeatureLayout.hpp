#pragma once

#include <cstddef>

namespace EA::EconomicCalendar
{

enum class EconomicEventFeatureIndex : std::size_t
{
    inflationEvent = 0,
    employmentEvent = 1,
    growthEvent = 2,
    fedPolicyEvent = 3,
    consumerDemandEvent = 4,
    inflationRecencyDecay = 5,
    employmentRecencyDecay = 6,
    growthRecencyDecay = 7,
    fedPolicyRecencyDecay = 8,
    consumerDemandRecencyDecay = 9,
    relevantEventHasConsensus = 10,
    relevantEventConsensusLow = 11,
    relevantEventConsensusHigh = 12,
    relevantEventConsensusIsRange = 13,
    releasedEventHasSurprise = 14,
    releasedEventSurprise = 15,
    releasedEventSurpriseAbs = 16,
    releasedEventSurpriseDirection = 17,
    authoritativeInitialHasSurprise = 18,
    authoritativeInitialSurprise = 19,
    authoritativeInitialSurpriseAbs = 20,
    authoritativeInitialSurpriseDirection = 21,
};

inline constexpr std::size_t kPreConsensusEconomicEventFeatureWidth = 10;
inline constexpr std::size_t kEconomicEventConsensusFeatureWidth = 8;
inline constexpr std::size_t kEconomicEventReleaseActualFeatureWidth = 4;
inline constexpr std::size_t kEconomicEventFeatureWidth =
    kPreConsensusEconomicEventFeatureWidth +
    kEconomicEventConsensusFeatureWidth +
    kEconomicEventReleaseActualFeatureWidth;

} // namespace EA::EconomicCalendar
