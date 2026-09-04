#pragma once

#include <array>
#include <cstddef>
#include <string_view>

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
    causalFirstReleaseSurpriseAvailable = 22,
    causalFirstReleaseSurprise = 23,
};

inline constexpr std::size_t kPreConsensusEconomicEventFeatureWidth = 10;
inline constexpr std::size_t kEconomicEventConsensusFeatureWidth = 8;
inline constexpr std::size_t kEconomicEventReleaseActualFeatureWidth = 4;
inline constexpr std::size_t kCausalEconomicEventSurpriseFeatureWidth = 2;
inline constexpr std::size_t kEconomicEventFeatureWidth =
    kPreConsensusEconomicEventFeatureWidth +
    kEconomicEventConsensusFeatureWidth +
    kEconomicEventReleaseActualFeatureWidth +
    kCausalEconomicEventSurpriseFeatureWidth;

// Authoritative append-only names in the exact order returned by
// EconomicEventFeatureValues::Ordered(). Persisted expansion provenance and
// experiment/model feature identity use this registry rather than maintaining
// a second spelling of the economic-event layout.
inline constexpr std::array<std::string_view, kEconomicEventFeatureWidth>
    kEconomicEventFeatureNames{{
        "inflation_event",
        "employment_event",
        "growth_event",
        "fed_policy_event",
        "consumer_demand_event",
        "inflation_recency_decay",
        "employment_recency_decay",
        "growth_recency_decay",
        "fed_policy_recency_decay",
        "consumer_demand_recency_decay",
        "relevant_event_has_consensus",
        "relevant_event_consensus_low",
        "relevant_event_consensus_high",
        "relevant_event_consensus_is_range",
        "released_event_has_surprise",
        "released_event_surprise",
        "released_event_surprise_abs",
        "released_event_surprise_direction",
        "authoritative_initial_has_surprise",
        "authoritative_initial_surprise",
        "authoritative_initial_surprise_abs",
        "authoritative_initial_surprise_direction",
        "causal_first_release_surprise_available",
        "causal_first_release_surprise",
    }};

} // namespace EA::EconomicCalendar
