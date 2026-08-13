#include <cassert>
#include <cstddef>
#include <string>
#include <vector>

#include "../Headers/ModelInputContract.hpp"

int main()
{
    static_assert(legacy_feature_size == 32);
    static_assert(feature_size == 34);
    static_assert(EA::kLegacyModelInputWidth == 36);
    static_assert(EA::kCurrentModelInputWidth == 38);

    std::vector<float> physicalTensor(34, 0.0f);
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
    assert(physicalTensor[32] == 132.0f);
    assert(physicalTensor[33] == 133.0f);

    // Current persisted model: all current tensor columns, including both
    // Donchian columns, are copied into the 38-wide model input prefix.
    const auto current = EA::ResolveModelInputContract(38, physicalTensor.size());
    assert(current.tensorFeatureCount == 34);
    std::vector<float> currentInput(38, -1.0f);
    EA::CopyTensorFeaturesForModelInput(currentInput.data(),
                                        physicalTensor.data(),
                                        current);
    for (std::size_t i = 0; i < 34; ++i)
        assert(currentInput[i] == physicalTensor[i]);
    assert(currentInput[32] == 132.0f);
    assert(currentInput[33] == 133.0f);

    bool unsupportedRejected = false;
    try
    {
        (void)EA::ResolveModelInputContract(37, physicalTensor.size());
    }
    catch (const std::exception& error)
    {
        unsupportedRejected =
            std::string{error.what()} ==
            "MODEL_INPUT_WIDTH_UNSUPPORTED,model_n_in=37,supported=36:38";
    }
    assert(unsupportedRejected);

    // A projection is read-only with respect to persisted parameter shape.
    struct ParameterShape { std::size_t rows; std::size_t cols; } shape{100, 256};
    const ParameterShape before = shape;
    (void)EA::ResolveModelInputContract(36, physicalTensor.size());
    assert(shape.rows == before.rows && shape.cols == before.cols);
    return 0;
}
