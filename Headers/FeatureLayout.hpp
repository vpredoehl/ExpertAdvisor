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
inline constexpr std::size_t authoritativeInitialHasSurpriseCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::authoritativeInitialHasSurprise);
inline constexpr std::size_t authoritativeInitialSurpriseCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::authoritativeInitialSurprise);
inline constexpr std::size_t authoritativeInitialSurpriseAbsCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::authoritativeInitialSurpriseAbs);
inline constexpr std::size_t authoritativeInitialSurpriseDirectionCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::authoritativeInitialSurpriseDirection);
inline constexpr std::size_t causalFirstReleaseSurpriseAvailableCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::causalFirstReleaseSurpriseAvailable);
inline constexpr std::size_t causalFirstReleaseSurpriseCol =
    economicEventFeatureStartCol + static_cast<std::size_t>(
        EA::EconomicCalendar::EconomicEventFeatureIndex::causalFirstReleaseSurprise);
inline constexpr std::size_t pre_consensus_economic_event_feature_size =
    economicEventFeatureStartCol +
    EA::EconomicCalendar::kPreConsensusEconomicEventFeatureWidth;
inline constexpr std::size_t consensus_economic_event_feature_size =
    economicEventFeatureStartCol +
    EA::EconomicCalendar::kPreConsensusEconomicEventFeatureWidth +
    EA::EconomicCalendar::kEconomicEventConsensusFeatureWidth;
inline constexpr std::size_t economic_event_feature_size =
    consensus_economic_event_feature_size +
    EA::EconomicCalendar::kEconomicEventReleaseActualFeatureWidth;
inline constexpr std::size_t causal_economic_event_surprise_feature_size =
    economic_event_feature_size +
    EA::EconomicCalendar::kCausalEconomicEventSurpriseFeatureWidth;
// TG4 production pulses are one-bar categorical observations of the same
// completed canonical bar that produces the Tensor row. They intentionally
// precede the model-only multi-horizon return suffix.
inline constexpr std::size_t tg4InnerBreakAnyCol =
    causal_economic_event_surprise_feature_size;
inline constexpr std::size_t tg4SourceTg3StructurallyEligibleCol =
    tg4InnerBreakAnyCol + 1;
inline constexpr std::size_t tg4SourceTg3ConfluentCol =
    tg4SourceTg3StructurallyEligibleCol + 1;
inline constexpr std::size_t tg4_production_pulse_feature_size =
    tg4SourceTg3ConfluentCol + 1;
// Frozen causal Fibonacci H1/H2 structural aggregates are a separate,
// symmetric producer.  They append after the unchanged layout-8 TG4 pulses.
inline constexpr std::size_t fibRecentPriceScaleValidCol =
    tg4_production_pulse_feature_size;
inline constexpr std::size_t fibUpRecentUnionCountLogCol = fibRecentPriceScaleValidCol + 1;
inline constexpr std::size_t fibUpRecentH1CountLogCol = fibUpRecentUnionCountLogCol + 1;
inline constexpr std::size_t fibUpRecentH2CountLogCol = fibUpRecentH1CountLogCol + 1;
inline constexpr std::size_t fibUpRecentH1H2BothCountLogCol = fibUpRecentH2CountLogCol + 1;
inline constexpr std::size_t fibUpRecentH1YoungestAge20Col = fibUpRecentH1H2BothCountLogCol + 1;
inline constexpr std::size_t fibUpRecentH2YoungestAge20Col = fibUpRecentH1YoungestAge20Col + 1;
inline constexpr std::size_t fibUpRecentMedian1272Col = fibUpRecentH2YoungestAge20Col + 1;
inline constexpr std::size_t fibUpRecentMedian1618Col = fibUpRecentMedian1272Col + 1;
inline constexpr std::size_t fibUpRecentMedianPullback0382Col = fibUpRecentMedian1618Col + 1;
inline constexpr std::size_t fibUpRecentMedianPullback0500Col = fibUpRecentMedianPullback0382Col + 1;
inline constexpr std::size_t fibUpRecentMedianPullback0618Col = fibUpRecentMedianPullback0500Col + 1;
inline constexpr std::size_t fibDownRecentUnionCountLogCol = fibUpRecentMedianPullback0618Col + 1;
inline constexpr std::size_t fibDownRecentH1CountLogCol = fibDownRecentUnionCountLogCol + 1;
inline constexpr std::size_t fibDownRecentH2CountLogCol = fibDownRecentH1CountLogCol + 1;
inline constexpr std::size_t fibDownRecentH1H2BothCountLogCol = fibDownRecentH2CountLogCol + 1;
inline constexpr std::size_t fibDownRecentH1YoungestAge20Col = fibDownRecentH1H2BothCountLogCol + 1;
inline constexpr std::size_t fibDownRecentH2YoungestAge20Col = fibDownRecentH1YoungestAge20Col + 1;
inline constexpr std::size_t fibDownRecentMedian1272Col = fibDownRecentH2YoungestAge20Col + 1;
inline constexpr std::size_t fibDownRecentMedian1618Col = fibDownRecentMedian1272Col + 1;
inline constexpr std::size_t fibDownRecentMedianPullback0382Col = fibDownRecentMedian1618Col + 1;
inline constexpr std::size_t fibDownRecentMedianPullback0500Col = fibDownRecentMedianPullback0382Col + 1;
inline constexpr std::size_t fibDownRecentMedianPullback0618Col = fibDownRecentMedianPullback0500Col + 1;
inline constexpr std::size_t causal_fibonacci_structural_feature_size =
    fibDownRecentMedianPullback0618Col + 1;
inline constexpr std::size_t pocketRecentPriceScaleValidCol =
    causal_fibonacci_structural_feature_size;
inline constexpr std::size_t pocketBullRecentCountLogCol =
    pocketRecentPriceScaleValidCol + 1;
inline constexpr std::size_t pocketBullYoungestAge20Col =
    pocketBullRecentCountLogCol + 1;
inline constexpr std::size_t pocketBullMedianTouchDistanceCol =
    pocketBullYoungestAge20Col + 1;
inline constexpr std::size_t pocketBullMedianCloseDistanceCol =
    pocketBullMedianTouchDistanceCol + 1;
inline constexpr std::size_t pocketBullMedianWidthCol =
    pocketBullMedianCloseDistanceCol + 1;
inline constexpr std::size_t pocketBearRecentCountLogCol =
    pocketBullMedianWidthCol + 1;
inline constexpr std::size_t pocketBearYoungestAge20Col =
    pocketBearRecentCountLogCol + 1;
inline constexpr std::size_t pocketBearMedianTouchDistanceCol =
    pocketBearYoungestAge20Col + 1;
inline constexpr std::size_t pocketBearMedianCloseDistanceCol =
    pocketBearMedianTouchDistanceCol + 1;
inline constexpr std::size_t pocketBearMedianWidthCol =
    pocketBearMedianCloseDistanceCol + 1;
inline constexpr std::size_t causal_pocket_recent_observation_feature_size =
    pocketBearMedianWidthCol + 1;
// Layout 11 appends only the two fixed availability bits from the frozen
// Phase-2B production descriptive confluence definitions.  Raw TG4, Fibonacci
// and Pocket columns above retain their exact layouts and meanings.
inline constexpr std::size_t confluenceTg4StructuralFibonacciRetracementSupportAvailableCol =
    causal_pocket_recent_observation_feature_size;
inline constexpr std::size_t confluenceTg4StructuralFibonacciRetracementContradictionAvailableCol =
    confluenceTg4StructuralFibonacciRetracementSupportAvailableCol + 1;
inline constexpr std::size_t fixed_confluence_tensor_feature_size =
    confluenceTg4StructuralFibonacciRetracementContradictionAvailableCol + 1;
// Layout 12 appends the frozen raw causal-price-level/v2 model exposure.
// The prior Layout-11 confluence columns above retain their exact positions.
inline constexpr std::size_t priceLevelAvailableCol =
    fixed_confluence_tensor_feature_size;
inline constexpr std::size_t priceLevelZoneScaleValidCol =
    priceLevelAvailableCol + 1;
inline constexpr std::size_t priceLevelZoneGapSignedClippedCol =
    priceLevelZoneScaleValidCol + 1;
inline constexpr std::size_t priceLevelZoneRelationCol =
    priceLevelZoneGapSignedClippedCol + 1;
inline constexpr std::size_t priceLevelCurrentRoleCol =
    priceLevelZoneRelationCol + 1;
inline constexpr std::size_t priceLevelAgeFractionCol =
    priceLevelCurrentRoleCol + 1;
inline constexpr std::size_t priceLevelPriorEvidenceSaturationCol =
    priceLevelAgeFractionCol + 1;
inline constexpr std::size_t priceLevelTouchNowCol =
    priceLevelPriorEvidenceSaturationCol + 1;
inline constexpr std::size_t priceLevelCrossDirectionNowCol =
    priceLevelTouchNowCol + 1;
inline constexpr std::size_t priceLevelRetestNowCol =
    priceLevelCrossDirectionNowCol + 1;
inline constexpr std::size_t priceLevelRoleReversalNowCol =
    priceLevelRetestNowCol + 1;
inline constexpr std::size_t causal_price_level_raw_feature_size =
    priceLevelRoleReversalNowCol + 1;
// Layout 13 appends causal Fibonacci retracement-lifecycle state. Layout 12
// columns remain byte-for-byte and semantically unchanged.
inline constexpr std::size_t fibLifecycleFeatureStartCol =
    causal_price_level_raw_feature_size;
inline constexpr std::size_t fibLifecycleUp0382ReachedCountLogCol = fibLifecycleFeatureStartCol;
inline constexpr std::size_t fibLifecycleUp0382DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 1;
inline constexpr std::size_t fibLifecycleUp0382DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 2;
inline constexpr std::size_t fibLifecycleUp0382CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 3;
inline constexpr std::size_t fibLifecycleUp0382ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 4;
inline constexpr std::size_t fibLifecycleUp0382DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 5;
inline constexpr std::size_t fibLifecycleUp0500ReachedCountLogCol = fibLifecycleFeatureStartCol + 6;
inline constexpr std::size_t fibLifecycleUp0500DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 7;
inline constexpr std::size_t fibLifecycleUp0500DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 8;
inline constexpr std::size_t fibLifecycleUp0500CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 9;
inline constexpr std::size_t fibLifecycleUp0500ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 10;
inline constexpr std::size_t fibLifecycleUp0500DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 11;
inline constexpr std::size_t fibLifecycleUp0618ReachedCountLogCol = fibLifecycleFeatureStartCol + 12;
inline constexpr std::size_t fibLifecycleUp0618DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 13;
inline constexpr std::size_t fibLifecycleUp0618DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 14;
inline constexpr std::size_t fibLifecycleUp0618CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 15;
inline constexpr std::size_t fibLifecycleUp0618ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 16;
inline constexpr std::size_t fibLifecycleUp0618DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 17;
inline constexpr std::size_t fibLifecycleUpAPenetrationCountLogCol = fibLifecycleFeatureStartCol + 18;
inline constexpr std::size_t fibLifecycleUpACloseBeyondCountLogCol = fibLifecycleFeatureStartCol + 19;
inline constexpr std::size_t fibLifecycleUpAPenetrationYoungestAgeLogCol = fibLifecycleFeatureStartCol + 20;
inline constexpr std::size_t fibLifecycleUpACloseBeyondYoungestAgeLogCol = fibLifecycleFeatureStartCol + 21;
inline constexpr std::size_t fibLifecycleDown0382ReachedCountLogCol = fibLifecycleFeatureStartCol + 22;
inline constexpr std::size_t fibLifecycleDown0382DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 23;
inline constexpr std::size_t fibLifecycleDown0382DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 24;
inline constexpr std::size_t fibLifecycleDown0382CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 25;
inline constexpr std::size_t fibLifecycleDown0382ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 26;
inline constexpr std::size_t fibLifecycleDown0382DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 27;
inline constexpr std::size_t fibLifecycleDown0500ReachedCountLogCol = fibLifecycleFeatureStartCol + 28;
inline constexpr std::size_t fibLifecycleDown0500DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 29;
inline constexpr std::size_t fibLifecycleDown0500DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 30;
inline constexpr std::size_t fibLifecycleDown0500CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 31;
inline constexpr std::size_t fibLifecycleDown0500ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 32;
inline constexpr std::size_t fibLifecycleDown0500DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 33;
inline constexpr std::size_t fibLifecycleDown0618ReachedCountLogCol = fibLifecycleFeatureStartCol + 34;
inline constexpr std::size_t fibLifecycleDown0618DirectionalCloseCountLogCol = fibLifecycleFeatureStartCol + 35;
inline constexpr std::size_t fibLifecycleDown0618DirectionalBreakCountLogCol = fibLifecycleFeatureStartCol + 36;
inline constexpr std::size_t fibLifecycleDown0618CloseBackThroughCountLogCol = fibLifecycleFeatureStartCol + 37;
inline constexpr std::size_t fibLifecycleDown0618ReachedYoungestAgeLogCol = fibLifecycleFeatureStartCol + 38;
inline constexpr std::size_t fibLifecycleDown0618DirectionalCloseYoungestAgeLogCol = fibLifecycleFeatureStartCol + 39;
inline constexpr std::size_t fibLifecycleDownAPenetrationCountLogCol = fibLifecycleFeatureStartCol + 40;
inline constexpr std::size_t fibLifecycleDownACloseBeyondCountLogCol = fibLifecycleFeatureStartCol + 41;
inline constexpr std::size_t fibLifecycleDownAPenetrationYoungestAgeLogCol = fibLifecycleFeatureStartCol + 42;
inline constexpr std::size_t fibLifecycleDownACloseBeyondYoungestAgeLogCol = fibLifecycleFeatureStartCol + 43;
inline constexpr std::size_t causal_fibonacci_lifecycle_feature_size =
    fibLifecycleFeatureStartCol + 44;
inline constexpr std::size_t feature_size =
    causal_fibonacci_lifecycle_feature_size;

static_assert(economicEventFeatureStartCol ==
              return_autocorrelation_feature_size);
static_assert(causal_economic_event_surprise_feature_size ==
              return_autocorrelation_feature_size +
              EA::EconomicCalendar::kEconomicEventFeatureWidth);
static_assert(relevantEventHasConsensusCol ==
              pre_consensus_economic_event_feature_size);
static_assert(authoritativeInitialHasSurpriseCol ==
              consensus_economic_event_feature_size);
static_assert(causalFirstReleaseSurpriseAvailableCol ==
              economic_event_feature_size);
static_assert(tg4InnerBreakAnyCol ==
              causal_economic_event_surprise_feature_size);
static_assert(tg4SourceTg3StructurallyEligibleCol ==
              tg4InnerBreakAnyCol + 1);
static_assert(tg4SourceTg3ConfluentCol == tg4InnerBreakAnyCol + 2);
static_assert(fibRecentPriceScaleValidCol == 76);
static_assert(causal_fibonacci_structural_feature_size -
                  tg4_production_pulse_feature_size ==
              23);
static_assert(causal_fibonacci_structural_feature_size == 99);
static_assert(pocketRecentPriceScaleValidCol == 99);
static_assert(pocketBearMedianWidthCol == 109);
static_assert(causal_pocket_recent_observation_feature_size -
                  causal_fibonacci_structural_feature_size ==
              11);
static_assert(confluenceTg4StructuralFibonacciRetracementSupportAvailableCol == 110);
static_assert(confluenceTg4StructuralFibonacciRetracementContradictionAvailableCol == 111);
static_assert(fixed_confluence_tensor_feature_size -
                  causal_pocket_recent_observation_feature_size ==
              2);
static_assert(priceLevelAvailableCol == 112);
static_assert(priceLevelRoleReversalNowCol == 122);
static_assert(causal_price_level_raw_feature_size -
                  fixed_confluence_tensor_feature_size ==
              11);
static_assert(causal_price_level_raw_feature_size == 123);
static_assert(fibLifecycleFeatureStartCol == 123);
static_assert(causal_fibonacci_lifecycle_feature_size -
                  causal_price_level_raw_feature_size == 44);
static_assert(feature_size == 167);

#endif /* FeatureLayout_hpp */
