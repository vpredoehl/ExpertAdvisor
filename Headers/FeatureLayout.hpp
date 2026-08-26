#ifndef FeatureLayout_hpp
#define FeatureLayout_hpp

#include <cstddef>

#include "DonchianLookback.hpp"
#include "../Sources/EconomicEventFeatureLayout.hpp"

// The first 32 Tensor columns are the pre-Donchian, persisted model prefix.
// Subsequent feature increments append columns so historical model prefixes
// retain their exact meaning.
inline constexpr std::size_t legacy_feature_size = 32;
// Closed Donchian-20 compatibility constant; runtime lookback is persisted
// separately and supplied to Tensor.
inline constexpr std::size_t donchian_lookback = kDefaultDonchianLookback;
inline constexpr std::size_t donchianUpCol = legacy_feature_size;
inline constexpr std::size_t donchianDownCol = legacy_feature_size + 1;
inline constexpr std::size_t donchian_feature_size = legacy_feature_size + 2;
inline constexpr std::size_t sessionPhaseSinCol = donchian_feature_size;
inline constexpr std::size_t sessionPhaseCosCol = donchian_feature_size + 1;
inline constexpr std::size_t session_phase_feature_size = donchian_feature_size + 2;
inline constexpr std::size_t relativeTickVolumeCol = session_phase_feature_size;
inline constexpr std::size_t relative_tick_volume_feature_size = session_phase_feature_size + 1;
inline constexpr std::size_t causalReturnSurpriseCol = relative_tick_volume_feature_size;
inline constexpr std::size_t causal_return_surprise_feature_size =
    relative_tick_volume_feature_size + 1;
inline constexpr std::size_t causalVolatilityRegimeCol =
    causal_return_surprise_feature_size;
inline constexpr std::size_t causal_volatility_regime_feature_size =
    causal_return_surprise_feature_size + 1;
inline constexpr std::size_t causalDirectionalRangeCol =
    causal_volatility_regime_feature_size;
inline constexpr std::size_t causal_directional_range_feature_size =
    causal_volatility_regime_feature_size + 1;
inline constexpr std::size_t causalCloseLocationCol =
    causal_directional_range_feature_size;
inline constexpr std::size_t causal_close_location_feature_size =
    causal_directional_range_feature_size + 1;
inline constexpr std::size_t causalDirectionalPersistenceCol =
    causal_close_location_feature_size;
inline constexpr std::size_t causalReturnSignPersistenceCol =
    causalDirectionalPersistenceCol + 1;
inline constexpr std::size_t causalReturnDirectionImbalanceCol =
    causalReturnSignPersistenceCol + 1;
inline constexpr std::size_t causalDirectionalAdverseExcursionCol =
    causalReturnDirectionImbalanceCol + 1;
inline constexpr std::size_t causalMultiBarRangePressureCol =
    causalDirectionalAdverseExcursionCol + 1;
inline constexpr std::size_t causal_multi_bar_range_pressure_feature_size =
    causalMultiBarRangePressureCol + 1;
inline constexpr std::size_t causalRollingRangeExpansionCol =
    causal_multi_bar_range_pressure_feature_size;
inline constexpr std::size_t causal_rolling_range_expansion_feature_size =
    causalRollingRangeExpansionCol + 1;
inline constexpr std::size_t historicalLevelProximityCol =
    causal_rolling_range_expansion_feature_size;
inline constexpr std::size_t historical_level_proximity_feature_size =
    historicalLevelProximityCol + 1;
inline constexpr std::size_t returnAutocorrelationCol =
    historical_level_proximity_feature_size;
inline constexpr std::size_t return_autocorrelation_feature_size =
    returnAutocorrelationCol + 1;
inline constexpr std::size_t economicEventFeatureStartCol =
    return_autocorrelation_feature_size;
inline constexpr std::size_t inflationEventCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::inflationEvent);
inline constexpr std::size_t employmentEventCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::employmentEvent);
inline constexpr std::size_t growthEventCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::growthEvent);
inline constexpr std::size_t fedPolicyEventCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::fedPolicyEvent);
inline constexpr std::size_t consumerDemandEventCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::consumerDemandEvent);
inline constexpr std::size_t inflationRecencyDecayCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::inflationRecencyDecay);
inline constexpr std::size_t employmentRecencyDecayCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::employmentRecencyDecay);
inline constexpr std::size_t growthRecencyDecayCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::growthRecencyDecay);
inline constexpr std::size_t fedPolicyRecencyDecayCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::fedPolicyRecencyDecay);
inline constexpr std::size_t consumerDemandRecencyDecayCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::consumerDemandRecencyDecay);
inline constexpr std::size_t relevantEventHasConsensusCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::relevantEventHasConsensus);
inline constexpr std::size_t relevantEventConsensusLowCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::relevantEventConsensusLow);
inline constexpr std::size_t relevantEventConsensusHighCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::relevantEventConsensusHigh);
inline constexpr std::size_t relevantEventConsensusIsRangeCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::relevantEventConsensusIsRange);
inline constexpr std::size_t releasedEventHasSurpriseCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::releasedEventHasSurprise);
inline constexpr std::size_t releasedEventSurpriseCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::releasedEventSurprise);
inline constexpr std::size_t releasedEventSurpriseAbsCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::releasedEventSurpriseAbs);
inline constexpr std::size_t releasedEventSurpriseDirectionCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::releasedEventSurpriseDirection);
inline constexpr std::size_t pre_consensus_economic_event_feature_size =
    economicEventFeatureStartCol +
    EA::EconomicCalendar::kPreConsensusEconomicEventFeatureWidth;
inline constexpr std::size_t economic_event_feature_size =
    economicEventFeatureStartCol +
    EA::EconomicCalendar::kEconomicEventFeatureWidth;
inline constexpr std::size_t feature_size =
    economic_event_feature_size;

static_assert(economicEventFeatureStartCol ==
              return_autocorrelation_feature_size);
static_assert(feature_size == return_autocorrelation_feature_size +
              EA::EconomicCalendar::kEconomicEventFeatureWidth);
static_assert(relevantEventHasConsensusCol ==
              pre_consensus_economic_event_feature_size);

#endif /* FeatureLayout_hpp */
