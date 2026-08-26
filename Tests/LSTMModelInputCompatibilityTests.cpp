#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

#include "../Headers/ModelInputContract.hpp"

int main()
{
    static_assert(legacy_feature_size == 32);
    static_assert(donchian_feature_size == 34);
    static_assert(session_phase_feature_size == 36);
    static_assert(relativeTickVolumeCol == 36);
    static_assert(relative_tick_volume_feature_size == 37);
    static_assert(causalReturnSurpriseCol == 37);
    static_assert(causal_return_surprise_feature_size == 38);
    static_assert(causalVolatilityRegimeCol == 38);
    static_assert(causal_volatility_regime_feature_size == 39);
    static_assert(causalDirectionalRangeCol == 39);
    static_assert(causal_directional_range_feature_size == 40);
    static_assert(causalCloseLocationCol == 40);
    static_assert(causal_close_location_feature_size == 41);
    static_assert(causalDirectionalPersistenceCol == 41);
    static_assert(causalReturnSignPersistenceCol == 42);
    static_assert(causalReturnDirectionImbalanceCol == 43);
    static_assert(causalDirectionalAdverseExcursionCol == 44);
    static_assert(causalMultiBarRangePressureCol == 45);
    static_assert(causal_multi_bar_range_pressure_feature_size == 46);
    static_assert(causalRollingRangeExpansionCol == 46);
    static_assert(causal_rolling_range_expansion_feature_size == 47);
    static_assert(historicalLevelProximityCol == 47);
    static_assert(historical_level_proximity_feature_size == 48);
    static_assert(returnAutocorrelationCol == 48);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(economicEventFeatureStartCol == 49);
    static_assert(pre_consensus_economic_event_feature_size == 59);
    static_assert(feature_size == 67);
    static_assert(EA::kLegacyModelInputWidth == 36);
    static_assert(EA::kDonchianModelInputWidth == 38);
    static_assert(EA::kSessionPhaseModelInputWidth == 40);
    static_assert(EA::kRelativeTickVolumeModelInputWidth == 41);
    static_assert(EA::kCausalReturnSurpriseModelInputWidth == 42);
    static_assert(EA::kCausalVolatilityRegimeModelInputWidth == 43);
    static_assert(EA::kCausalDirectionalRangeModelInputWidth == 44);
    static_assert(EA::kCausalCloseLocationModelInputWidth == 45);
    static_assert(EA::kCausalDirectionalPersistenceModelInputWidth == 46);
    static_assert(EA::kCausalReturnSignPersistenceModelInputWidth == 47);
    static_assert(EA::kCausalReturnDirectionImbalanceModelInputWidth == 48);
    static_assert(EA::kCausalDirectionalAdverseExcursionModelInputWidth == 49);
    static_assert(EA::kCausalMultiBarRangePressureModelInputWidth == 50);
    static_assert(EA::kCausalRollingRangeExpansionModelInputWidth == 51);
    static_assert(EA::kHistoricalLevelProximityModelInputWidth == 52);
    static_assert(EA::kReturnAutocorrelationModelInputWidth == 53);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kEconomicEventModelInputWidth == 63);
    static_assert(EA::kCurrentModelInputWidth == 71);

    std::vector<float> physicalTensor(feature_size, 0.0f);
    for (std::size_t i = 0; i < physicalTensor.size(); ++i)
        physicalTensor[i] = static_cast<float>(100 + i);

    // Legacy persisted model: Donchian columns 32 and 33 are not part of
    // the model input projection, even though they exist in today's tensor.
    const auto legacy = EA::ResolveModelInputContract(36, physicalTensor.size());
    assert(legacy.tensorFeatureCount == 32);
    assert(legacy.modelInputWidth == 36);
    std::vector<float> legacyInput(36, -1.0f);
    EA::CopyTensorFeaturesForModelInput(legacyInput.data(),
                                        physicalTensor.data(),
                                        legacy);
    for (std::size_t i = 0; i < 32; ++i)
        assert(legacyInput[i] == physicalTensor[i]);
    for (std::size_t i = 32; i < 36; ++i)
        assert(legacyInput[i] == -1.0f);
    assert(physicalTensor[donchianUpCol] == 132.0f);
    assert(physicalTensor[donchianDownCol] == 133.0f);

    // Donchian-era persisted model: the two later session-phase columns are
    // absent from its learned input projection.
    const auto donchian = EA::ResolveModelInputContract(
        38, physicalTensor.size());
    assert(donchian.tensorFeatureCount == donchian_feature_size);
    std::vector<float> donchianInput(38, -1.0f);
    EA::CopyTensorFeaturesForModelInput(donchianInput.data(),
                                        physicalTensor.data(),
                                        donchian);
    for (std::size_t i = 0; i < 34; ++i)
        assert(donchianInput[i] == physicalTensor[i]);
    for (std::size_t i = donchian_feature_size; i < donchianInput.size(); ++i)
        assert(donchianInput[i] == -1.0f);

    // Session-phase-era persisted models do not include the later relative
    // tick-volume column in their learned input projection.
    const auto sessionPhase = EA::ResolveModelInputContract(
        40, physicalTensor.size());
    assert(sessionPhase.tensorFeatureCount == session_phase_feature_size);
    std::vector<float> sessionPhaseInput(40, -1.0f);
    EA::CopyTensorFeaturesForModelInput(sessionPhaseInput.data(),
                                        physicalTensor.data(), sessionPhase);
    for (std::size_t i = 0; i < session_phase_feature_size; ++i)
        assert(sessionPhaseInput[i] == physicalTensor[i]);
    assert(sessionPhaseInput[sessionPhaseSinCol] == physicalTensor[sessionPhaseSinCol]);
    assert(sessionPhaseInput[sessionPhaseCosCol] == physicalTensor[sessionPhaseCosCol]);

    // Relative-tick-volume-era persisted models retain their exact 37-column
    // prefix and cannot consume the later return-surprise column.
    const auto relativeTickVolume = EA::ResolveModelInputContract(
        41, physicalTensor.size());
    assert(relativeTickVolume.tensorFeatureCount == relative_tick_volume_feature_size);
    std::vector<float> relativeTickVolumeInput(41, -1.0f);
    EA::CopyTensorFeaturesForModelInput(relativeTickVolumeInput.data(),
                                        physicalTensor.data(), relativeTickVolume);
    for (std::size_t i = 0; i < relative_tick_volume_feature_size; ++i)
        assert(relativeTickVolumeInput[i] == physicalTensor[i]);
    assert(relativeTickVolumeInput[relativeTickVolumeCol] ==
           physicalTensor[relativeTickVolumeCol]);
    assert(relativeTickVolumeInput[causalReturnSurpriseCol] == -1.0f);

    // Return-surprise-era persisted models retain their exact 38-column
    // prefix and cannot consume the later volatility-regime column.
    const auto returnSurprise = EA::ResolveModelInputContract(
        42, physicalTensor.size());
    assert(returnSurprise.tensorFeatureCount == causal_return_surprise_feature_size);
    std::vector<float> returnSurpriseInput(42, -1.0f);
    EA::CopyTensorFeaturesForModelInput(returnSurpriseInput.data(),
                                        physicalTensor.data(), returnSurprise);
    for (std::size_t i = 0; i < causal_return_surprise_feature_size; ++i)
        assert(returnSurpriseInput[i] == physicalTensor[i]);
    assert(returnSurpriseInput[causalReturnSurpriseCol] ==
           physicalTensor[causalReturnSurpriseCol]);
    assert(returnSurpriseInput[causalVolatilityRegimeCol] == -1.0f);

    // Volatility-regime-era persisted models retain their exact 39-column
    // prefix and cannot consume the later directional-range column.
    const auto volatilityRegime = EA::ResolveModelInputContract(
        43, physicalTensor.size());
    assert(volatilityRegime.tensorFeatureCount == causal_volatility_regime_feature_size);
    std::vector<float> volatilityRegimeInput(43, -1.0f);
    EA::CopyTensorFeaturesForModelInput(volatilityRegimeInput.data(),
                                        physicalTensor.data(), volatilityRegime);
    for (std::size_t i = 0; i < causal_volatility_regime_feature_size; ++i)
        assert(volatilityRegimeInput[i] == physicalTensor[i]);
    assert(volatilityRegimeInput[causalVolatilityRegimeCol] ==
           physicalTensor[causalVolatilityRegimeCol]);
    assert(volatilityRegimeInput[causalDirectionalRangeCol] == -1.0f);

    // Directional-range-era persisted models retain their exact 40-column
    // prefix and cannot consume the later close-location column.
    const auto directionalRange = EA::ResolveModelInputContract(
        44, physicalTensor.size());
    assert(directionalRange.tensorFeatureCount == causal_directional_range_feature_size);
    std::vector<float> directionalRangeInput(44, -1.0f);
    EA::CopyTensorFeaturesForModelInput(directionalRangeInput.data(),
                                        physicalTensor.data(), directionalRange);
    for (std::size_t i = 0; i < causal_directional_range_feature_size; ++i)
        assert(directionalRangeInput[i] == physicalTensor[i]);
    assert(directionalRangeInput[causalDirectionalRangeCol] ==
           physicalTensor[causalDirectionalRangeCol]);
    assert(directionalRangeInput[causalCloseLocationCol] == -1.0f);

    // Close-location-era persisted models do not consume the later
    // directional-persistence column.
    const auto current = EA::ResolveModelInputContract(45, physicalTensor.size());
    assert(current.tensorFeatureCount == causal_close_location_feature_size);
    std::vector<float> currentInput(45, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentInput.data(),
                                        physicalTensor.data(), current);
    for (std::size_t i = 0; i < causal_close_location_feature_size; ++i)
        assert(currentInput[i] == physicalTensor[i]);
    assert(currentInput[causalCloseLocationCol] ==
           physicalTensor[causalCloseLocationCol]);
    assert(currentInput[causalDirectionalPersistenceCol] == -1.0f);

    // Directional-persistence-era persisted models retain their exact 42-column
    // prefix and do not consume the later return-sign-persistence column.
    const auto currentPersistence = EA::ResolveModelInputContract(46, physicalTensor.size());
    assert(currentPersistence.tensorFeatureCount == causalReturnSignPersistenceCol);
    std::vector<float> currentPersistenceInput(46, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentPersistenceInput.data(),
                                        physicalTensor.data(), currentPersistence);
    for (std::size_t i = 0; i < causalReturnSignPersistenceCol; ++i)
        assert(currentPersistenceInput[i] == physicalTensor[i]);
    assert(currentPersistenceInput[causalDirectionalPersistenceCol] ==
           physicalTensor[causalDirectionalPersistenceCol]);
    assert(currentPersistenceInput[causalReturnSignPersistenceCol] == -1.0f);

    // Return-sign-persistence-era persisted models retain their exact
    // 43-column prefix and do not consume the later imbalance column.
    const auto currentSignPersistence = EA::ResolveModelInputContract(47, physicalTensor.size());
    std::vector<float> currentSignPersistenceInput(47, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentSignPersistenceInput.data(),
                                        physicalTensor.data(), currentSignPersistence);
    for (std::size_t i = 0; i < causalReturnDirectionImbalanceCol; ++i)
        assert(currentSignPersistenceInput[i] == physicalTensor[i]);
    assert(currentSignPersistenceInput[causalReturnSignPersistenceCol] ==
           physicalTensor[causalReturnSignPersistenceCol]);
    assert(currentSignPersistenceInput[causalReturnDirectionImbalanceCol] == -1.0f);

    const auto currentDirectionImbalance = EA::ResolveModelInputContract(
        48, physicalTensor.size());
    std::vector<float> currentDirectionImbalanceInput(48, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentDirectionImbalanceInput.data(),
                                        physicalTensor.data(), currentDirectionImbalance);
    for (std::size_t i = 0; i < causalDirectionalAdverseExcursionCol; ++i)
        assert(currentDirectionImbalanceInput[i] == physicalTensor[i]);
    assert(currentDirectionImbalanceInput[causalReturnDirectionImbalanceCol] ==
           physicalTensor[causalReturnDirectionImbalanceCol]);
    assert(currentDirectionImbalanceInput[causalDirectionalAdverseExcursionCol] == -1.0f);

    // Directional-adverse-excursion-era persisted models retain their exact
    // 45-column prefix and cannot consume multi-bar-range-pressure.
    const auto adverseExcursion = EA::ResolveModelInputContract(
        49, physicalTensor.size());
    assert(adverseExcursion.tensorFeatureCount == causalMultiBarRangePressureCol);
    std::vector<float> adverseExcursionInput(49, -1.0f);
    EA::CopyTensorFeaturesForModelInput(adverseExcursionInput.data(), physicalTensor.data(),
                                        adverseExcursion);
    for (std::size_t i = 0; i < causalMultiBarRangePressureCol; ++i)
        assert(adverseExcursionInput[i] == physicalTensor[i]);
    assert(adverseExcursionInput[causalDirectionalAdverseExcursionCol] ==
           physicalTensor[causalDirectionalAdverseExcursionCol]);
    assert(adverseExcursionInput[causalMultiBarRangePressureCol] == -1.0f);

    const auto multiBarRangePressure = EA::ResolveModelInputContract(
        50, physicalTensor.size());
    std::vector<float> multiBarRangePressureInput(50, -1.0f);
    EA::CopyTensorFeaturesForModelInput(multiBarRangePressureInput.data(),
                                        physicalTensor.data(), multiBarRangePressure);
    for (std::size_t i = 0; i < causal_multi_bar_range_pressure_feature_size; ++i)
        assert(multiBarRangePressureInput[i] == physicalTensor[i]);
    assert(multiBarRangePressureInput[causalMultiBarRangePressureCol] ==
           physicalTensor[causalMultiBarRangePressureCol]);
    assert(multiBarRangePressureInput[causalRollingRangeExpansionCol] == -1.0f);

    // The multi-bar-range era preserves its 46-column prefix and cannot
    // consume the later range-expansion column.
    assert(multiBarRangePressure.tensorFeatureCount == causalRollingRangeExpansionCol);
    const auto rollingRangeExpansion = EA::ResolveModelInputContract(
        51, physicalTensor.size());
    std::vector<float> rollingRangeExpansionInput(51, -1.0f);
    EA::CopyTensorFeaturesForModelInput(rollingRangeExpansionInput.data(),
                                        physicalTensor.data(), rollingRangeExpansion);
    for (std::size_t i = 0;
         i < causal_rolling_range_expansion_feature_size; ++i)
        assert(rollingRangeExpansionInput[i] == physicalTensor[i]);
    assert(rollingRangeExpansionInput[causalRollingRangeExpansionCol] ==
           physicalTensor[causalRollingRangeExpansionCol]);
    assert(rollingRangeExpansionInput[historicalLevelProximityCol] == -1.0f);

    // The historical-level-proximity model retains its exact 48-column Tensor
    // prefix and cannot consume the later autocorrelation column.
    const auto historicalLevelProximity = EA::ResolveModelInputContract(
        52, physicalTensor.size());
    std::vector<float> historicalLevelProximityInput(52, -1.0f);
    EA::CopyTensorFeaturesForModelInput(historicalLevelProximityInput.data(),
                                        physicalTensor.data(),
                                        historicalLevelProximity);
    for (std::size_t i = 0;
         i < historical_level_proximity_feature_size; ++i)
        assert(historicalLevelProximityInput[i] == physicalTensor[i]);
    assert(historicalLevelProximityInput[historicalLevelProximityCol] ==
           physicalTensor[historicalLevelProximityCol]);
    assert(historicalLevelProximityInput[returnAutocorrelationCol] == -1.0f);

    // The return-autocorrelation model appends that feature without changing the
    // exact width-52 predecessor contract above.
    const auto returnAutocorrelation = EA::ResolveModelInputContract(
        53, physicalTensor.size());
    std::vector<float> returnAutocorrelationInput(53, -1.0f);
    EA::CopyTensorFeaturesForModelInput(returnAutocorrelationInput.data(),
                                        physicalTensor.data(),
                                        returnAutocorrelation);
    for (std::size_t i = 0; i < return_autocorrelation_feature_size; ++i)
        assert(returnAutocorrelationInput[i] == physicalTensor[i]);
    assert(returnAutocorrelationInput[returnAutocorrelationCol] ==
           physicalTensor[returnAutocorrelationCol]);

    // Phase 2 appends exactly the ten economic-event Tensor features. The old
    // width remains a registered projection and cannot silently consume them.
    const auto economicEvents = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, physicalTensor.size());
    assert(economicEvents.tensorFeatureCount == feature_size);
    std::vector<float> economicEventInput(EA::kCurrentModelInputWidth, -1.0f);
    EA::CopyTensorFeaturesForModelInput(economicEventInput.data(),
                                        physicalTensor.data(), economicEvents);
    for (std::size_t i = 0; i < feature_size; ++i)
        assert(economicEventInput[i] == physicalTensor[i]);
    assert(returnAutocorrelationInput[inflationEventCol] == -1.0f);
    assert(economicEventInput[inflationEventCol] ==
           physicalTensor[inflationEventCol]);
    assert(economicEventInput[consumerDemandRecencyDecayCol] ==
           physicalTensor[consumerDemandRecencyDecayCol]);

    // Canonical feature identities are independent of physical offsets and
    // masking happens after projection without mutating Tensor storage.
    const auto emptyMask = EA::FeatureAblationMask::Parse("");
    assert(emptyMask.empty());
    const auto fullMask = EA::FeatureAblationMask::Parse(
        "directional_adverse_excursion,directional_efficiency,close_location,"
        "directional_range,volatility_regime,rms_return_surprise,"
        "relative_tick_volume,return_direction_imbalance,return_sign_persistence,"
        "multi_bar_range_pressure,rolling_range_expansion,"
        "historical_level_proximity,return_autocorrelation");
    assert(fullMask.CanonicalText() ==
           "relative_tick_volume,rms_return_surprise,volatility_regime,"
           "directional_range,close_location,directional_efficiency,"
           "return_sign_persistence,return_direction_imbalance,"
           "directional_adverse_excursion,multi_bar_range_pressure,"
           "rolling_range_expansion,historical_level_proximity,"
           "return_autocorrelation");
    const auto mask = EA::FeatureAblationMask::Parse(
        " return_direction_imbalance,return_sign_persistence,return_direction_imbalance ");
    assert(mask.CanonicalText() == "return_sign_persistence,return_direction_imbalance");
    std::vector<float> ablatedInput(53, -1.0f);
    EA::CopyTensorFeaturesForModelInput(ablatedInput.data(), physicalTensor.data(),
                                        returnAutocorrelation, mask);
    assert(ablatedInput[causalReturnSignPersistenceCol] == 0.0f);
    assert(ablatedInput[causalReturnDirectionImbalanceCol] == 0.0f);
    assert(ablatedInput[causalDirectionalAdverseExcursionCol] ==
           physicalTensor[causalDirectionalAdverseExcursionCol]);
    assert(ablatedInput[causalMultiBarRangePressureCol] ==
           physicalTensor[causalMultiBarRangePressureCol]);
    assert(ablatedInput[causalRollingRangeExpansionCol] ==
           physicalTensor[causalRollingRangeExpansionCol]);
    assert(ablatedInput[historicalLevelProximityCol] ==
           physicalTensor[historicalLevelProximityCol]);
    assert(physicalTensor[causalReturnSignPersistenceCol] ==
           static_cast<float>(100 + causalReturnSignPersistenceCol));
    bool absentFeatureRejected = false;
    try
    {
        const auto lateMask = EA::FeatureAblationMask::Parse("multi_bar_range_pressure");
        std::vector<float> historical(49, -1.0f);
        EA::CopyTensorFeaturesForModelInput(historical.data(), physicalTensor.data(),
                                            adverseExcursion, lateMask);
    }
    catch (const std::runtime_error&)
    {
        absentFeatureRejected = true;
    }
    assert(absentFeatureRejected);
    bool latestFeatureRejectedByPreviousFullWidth = false;
    try
    {
        const auto latestMask = EA::FeatureAblationMask::Parse("rolling_range_expansion");
        std::vector<float> previousFullWidth(50, -1.0f);
        EA::CopyTensorFeaturesForModelInput(previousFullWidth.data(), physicalTensor.data(),
                                            multiBarRangePressure, latestMask);
    }
    catch (const std::runtime_error&)
    {
        latestFeatureRejectedByPreviousFullWidth = true;
    }
    assert(latestFeatureRejectedByPreviousFullWidth);
    bool historicalLevelRejectedByPreviousFullWidth = false;
    try
    {
        const auto latestMask = EA::FeatureAblationMask::Parse(
            "historical_level_proximity");
        std::vector<float> previousFullWidth(51, -1.0f);
        EA::CopyTensorFeaturesForModelInput(
            previousFullWidth.data(), physicalTensor.data(),
            rollingRangeExpansion, latestMask);
    }
    catch (const std::runtime_error&)
    {
        historicalLevelRejectedByPreviousFullWidth = true;
    }
    assert(historicalLevelRejectedByPreviousFullWidth);
    bool autocorrelationRejectedByPreviousFullWidth = false;
    try
    {
        const auto latestMask = EA::FeatureAblationMask::Parse(
            "return_autocorrelation");
        std::vector<float> previousFullWidth(52, -1.0f);
        EA::CopyTensorFeaturesForModelInput(
            previousFullWidth.data(), physicalTensor.data(),
            historicalLevelProximity, latestMask);
    }
    catch (const std::runtime_error&)
    {
        autocorrelationRejectedByPreviousFullWidth = true;
    }
    assert(autocorrelationRejectedByPreviousFullWidth);
    bool unknownFeatureRejected = false;
    try { (void)EA::FeatureAblationMask::Parse("unknown_feature"); }
    catch (const std::invalid_argument&) { unknownFeatureRejected = true; }
    assert(unknownFeatureRejected);

    bool unsupportedRejected = false;
    try
    {
        (void)EA::ResolveModelInputContract(39, physicalTensor.size());
    }
    catch (const std::exception& error)
    {
        unsupportedRejected =
            std::string{error.what()} ==
            "MODEL_INPUT_WIDTH_UNSUPPORTED,model_n_in=39,supported=36:38:40:41:42:43:44:45:46:47:48:49:50:51:52:53:63:71";
    }
    assert(unsupportedRejected);

    // A projection is read-only with respect to persisted parameter shape.
    struct ParameterShape { std::size_t rows; std::size_t cols; } shape{100, 256};
    const ParameterShape before = shape;
    (void)EA::ResolveModelInputContract(36, physicalTensor.size());
    assert(shape.rows == before.rows && shape.cols == before.cols);
    return 0;
}
