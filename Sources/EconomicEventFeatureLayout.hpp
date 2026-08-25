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
};

inline constexpr std::size_t kEconomicEventFeatureWidth = 10;

} // namespace EA::EconomicCalendar
