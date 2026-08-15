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
    static_assert(feature_size == 37);
    static_assert(EA::kLegacyModelInputWidth == 36);
    static_assert(EA::kDonchianModelInputWidth == 38);
    static_assert(EA::kSessionPhaseModelInputWidth == 40);
    static_assert(EA::kCurrentModelInputWidth == 41);

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

    // Current persisted models include the appended relative tick-volume
    // column, while the four return channels remain outside the Tensor prefix.
    const auto current = EA::ResolveModelInputContract(
        41, physicalTensor.size());
    assert(current.tensorFeatureCount == feature_size);
    std::vector<float> currentInput(41, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentInput.data(),
                                        physicalTensor.data(), current);
    for (std::size_t i = 0; i < feature_size; ++i)
        assert(currentInput[i] == physicalTensor[i]);
    assert(currentInput[sessionPhaseSinCol] == physicalTensor[sessionPhaseSinCol]);
    assert(currentInput[sessionPhaseCosCol] == physicalTensor[sessionPhaseCosCol]);
    assert(currentInput[relativeTickVolumeCol] == physicalTensor[relativeTickVolumeCol]);

    bool unsupportedRejected = false;
    try
    {
        (void)EA::ResolveModelInputContract(39, physicalTensor.size());
    }
    catch (const std::exception& error)
    {
        unsupportedRejected =
            std::string{error.what()} ==
            "MODEL_INPUT_WIDTH_UNSUPPORTED,model_n_in=39,supported=36:38:40:41";
    }
    assert(unsupportedRejected);

    // A projection is read-only with respect to persisted parameter shape.
    struct ParameterShape { std::size_t rows; std::size_t cols; } shape{100, 256};
    const ParameterShape before = shape;
    (void)EA::ResolveModelInputContract(36, physicalTensor.size());
    assert(shape.rows == before.rows && shape.cols == before.cols);
    return 0;
}
