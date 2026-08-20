#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "FeatureAblation.hpp"
#include "ModelInputExpansion.hpp"

namespace
{

template <typename Callable>
void ExpectFailureContaining(Callable&& callable, const std::string& marker)
{
    try
    {
        callable();
        assert(false && "expected failure");
    }
    catch (const std::exception& error)
    {
        assert(std::string{error.what()}.find(marker) != std::string::npos);
    }
}

std::vector<float> OneStepGates(const std::vector<float>& input,
                                const std::vector<float>& hidden,
                                const std::vector<float>& parameter,
                                std::size_t hiddenSize)
{
    const std::size_t inputWidth = input.size();
    const std::size_t gateColumns = 4 * hiddenSize;
    assert(parameter.size() == (inputWidth + hiddenSize) * gateColumns);
    std::vector<float> gates(gateColumns, 0.0f);
    for (std::size_t row = 0; row < inputWidth; ++row)
        for (std::size_t col = 0; col < gateColumns; ++col)
            gates[col] += input[row] * parameter[row * gateColumns + col];
    for (std::size_t row = 0; row < hiddenSize; ++row)
        for (std::size_t col = 0; col < gateColumns; ++col)
            gates[col] += hidden[row] *
                parameter[(inputWidth + row) * gateColumns + col];
    return gates;
}

float Sigmoid(float value)
{
    return 1.0f / (1.0f + std::exp(-value));
}

float OneStepPrediction(const std::vector<float>& input,
                        const std::vector<float>& previousHidden,
                        const std::vector<float>& previousCell,
                        const std::vector<float>& parameter,
                        const std::vector<float>& bias,
                        const std::vector<float>& headWeight,
                        float headBias,
                        std::size_t hiddenSize)
{
    std::vector<float> gates =
        OneStepGates(input, previousHidden, parameter, hiddenSize);
    for (std::size_t i = 0; i < gates.size(); ++i) gates[i] += bias[i];
    float prediction = headBias;
    for (std::size_t unit = 0; unit < hiddenSize; ++unit)
    {
        const float inputGate = Sigmoid(gates[unit]);
        const float forgetGate = Sigmoid(gates[hiddenSize + unit]);
        const float candidate = std::tanh(gates[2 * hiddenSize + unit]);
        const float outputGate = Sigmoid(gates[3 * hiddenSize + unit]);
        const float cell = forgetGate * previousCell[unit] +
                           inputGate * candidate;
        const float hidden = outputGate * std::tanh(cell);
        prediction += hidden * headWeight[unit];
    }
    return prediction;
}

} // namespace

int main()
{
    constexpr std::size_t hiddenSize = 3;
    constexpr std::size_t gateColumns = 4 * hiddenSize;
    constexpr std::size_t sourceWidth =
        EA::kCausalRollingRangeExpansionModelInputWidth;
    constexpr std::size_t expandedWidth = EA::kCurrentModelInputWidth;
    static_assert(sourceWidth == 51);
    static_assert(expandedWidth == 52);

    const EA::InputWidthExpansionPlan plan =
        EA::BuildInputWidthExpansionPlan(sourceWidth);
    assert(plan.sourceTensorFeatureCount == 47);
    assert(plan.expandedTensorFeatureCount == 48);
    assert(plan.newlyIntroducedTensorFeatures ==
           std::vector<std::string>{"historical_level_proximity"});

    std::vector<float> source((sourceWidth + hiddenSize) * gateColumns);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>(i + 1) / 1024.0f;

    const std::vector<float> expanded =
        EA::ExpandFusedLstmParameterRowMajor(source, hiddenSize, plan);
    assert(expanded.size() == (expandedWidth + hiddenSize) * gateColumns);

    // Historical Tensor-feature rows are byte-identical at stable indices.
    assert(std::memcmp(expanded.data(), source.data(),
                       plan.sourceTensorFeatureCount * gateColumns *
                           sizeof(float)) == 0);

    // Newly appended Tensor-feature rows are exactly zero for all four gates.
    for (std::size_t row = plan.sourceTensorFeatureCount;
         row < plan.expandedTensorFeatureCount;
         ++row)
        for (std::size_t col = 0; col < gateColumns; ++col)
            assert(expanded[row * gateColumns + col] == 0.0f);

    // The stable four-return-feature suffix is relocated, not reinterpreted.
    for (std::size_t returnIndex = 0;
         returnIndex < EA::kModelReturnFeatureCount;
         ++returnIndex)
    {
        const std::size_t sourceRow =
            plan.sourceTensorFeatureCount + returnIndex;
        const std::size_t expandedRow =
            plan.expandedTensorFeatureCount + returnIndex;
        assert(std::memcmp(
                   expanded.data() + expandedRow * gateColumns,
                   source.data() + sourceRow * gateColumns,
                   gateColumns * sizeof(float)) == 0);
    }

    // Recurrent hidden-to-hidden rows remain byte-identical after relocation.
    assert(std::memcmp(
               expanded.data() + expandedWidth * gateColumns,
               source.data() + sourceWidth * gateColumns,
               hiddenSize * gateColumns * sizeof(float)) == 0);

    // Biases and heads are not shape-dependent and the expansion API never
    // touches them.  A checkpoint load retains these exact buffers.
    const std::vector<float> bias(gateColumns, 0.25f);
    constexpr std::size_t directionOutputSize = 3;
    const std::vector<float> headWeight(hiddenSize * directionOutputSize,
                                        -0.125f);
    const std::vector<float> headBias(directionOutputSize, 0.5f);
    const auto reloadedBias = bias;
    const auto reloadedHeadWeight = headWeight;
    const auto reloadedHeadBias = headBias;
    assert(std::memcmp(bias.data(), reloadedBias.data(),
                       bias.size() * sizeof(float)) == 0);
    assert(std::memcmp(headWeight.data(), reloadedHeadWeight.data(),
                       headWeight.size() * sizeof(float)) == 0);
    assert(std::memcmp(headBias.data(), reloadedHeadBias.data(),
                       headBias.size() * sizeof(float)) == 0);

    // Prediction/gate preservation with nonzero current-only feature input.
    std::vector<float> sourceInput(sourceWidth);
    for (std::size_t i = 0; i < plan.sourceTensorFeatureCount; ++i)
        sourceInput[i] = static_cast<float>(i + 1) / 128.0f;
    for (std::size_t i = 0; i < EA::kModelReturnFeatureCount; ++i)
        sourceInput[plan.sourceTensorFeatureCount + i] =
            static_cast<float>(i + 1) / 16.0f;
    std::vector<float> expandedInput(expandedWidth, 0.0f);
    std::copy_n(sourceInput.begin(), plan.sourceTensorFeatureCount,
                expandedInput.begin());
    expandedInput[plan.sourceTensorFeatureCount] = 7.25f;
    std::copy_n(
        sourceInput.begin() +
            static_cast<std::ptrdiff_t>(plan.sourceTensorFeatureCount),
        EA::kModelReturnFeatureCount,
        expandedInput.begin() +
            static_cast<std::ptrdiff_t>(plan.expandedTensorFeatureCount));
    const std::vector<float> previousHidden{0.1f, -0.2f, 0.3f};
    const auto legacyGates =
        OneStepGates(sourceInput, previousHidden, source, hiddenSize);
    const auto expandedGates =
        OneStepGates(expandedInput, previousHidden, expanded, hiddenSize);
    for (std::size_t i = 0; i < legacyGates.size(); ++i)
        assert(std::fabs(legacyGates[i] - expandedGates[i]) <= 1.0e-6f);

    const std::vector<float> previousCell{-0.15f, 0.25f, 0.05f};
    const std::vector<float> regressionHead{0.75f, -0.5f, 0.125f};
    const float legacyPrediction = OneStepPrediction(
        sourceInput, previousHidden, previousCell, source, bias,
        regressionHead, -0.2f, hiddenSize);
    const float expandedPrediction = OneStepPrediction(
        expandedInput, previousHidden, previousCell, expanded, bias,
        regressionHead, -0.2f, hiddenSize);
    assert(std::memcmp(&legacyPrediction, &expandedPrediction,
                       sizeof(float)) == 0);

    // Zero is trainable: production BPTT computes dW=[x|h]^T*d_gates.
    // A nonzero new input and nonzero gate derivative produce a nonzero SGD
    // update without depending on the current (zero) weight value.
    std::vector<float> trained = expanded;
    const std::vector<float> dGates{
        0.1f, -0.2f, 0.3f, 0.4f, -0.5f, 0.6f,
        0.7f, -0.8f, 0.9f, 1.0f, -1.1f, 1.2f};
    constexpr float learningRate = 0.01f;
    const std::size_t newRow = plan.sourceTensorFeatureCount;
    bool changed = false;
    for (std::size_t col = 0; col < gateColumns; ++col)
    {
        const float gradient = expandedInput[newRow] * dGates[col];
        trained[newRow * gateColumns + col] -= learningRate * gradient;
        changed = changed || trained[newRow * gateColumns + col] != 0.0f;
        assert(std::isfinite(trained[newRow * gateColumns + col]));
    }
    assert(changed);

    // Expanded data materializes the current width; ablation zeros the value
    // without shrinking either the row or the model.
    const auto currentContract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::vector<float> physical(feature_size, 2.0f);
    std::vector<float> currentInput(EA::kCurrentModelInputWidth, -1.0f);
    const auto ablation = EA::FeatureAblationMask::Parse(
        "historical_level_proximity");
    EA::CopyTensorFeaturesForModelInput(currentInput.data(), physical.data(),
                                        currentContract, ablation);
    assert(currentInput.size() == expandedWidth);
    assert(currentInput[historicalLevelProximityCol] == 0.0f);

    // Ordinary historical-width projection remains unchanged.
    const auto legacyContract = EA::ResolveModelInputContract(
        EA::kLegacyModelInputWidth, feature_size);
    assert(legacyContract.modelInputWidth == EA::kLegacyModelInputWidth);
    assert(legacyContract.tensorFeatureCount == legacy_feature_size);

    // Expanded checkpoint payload and provenance round-trip exactly.
    const EA::InputWidthExpansionProvenance provenance =
        EA::MakeInputWidthExpansionProvenance(9876543210LL, plan);
    const std::string encoded = provenance.CanonicalText();
    const auto decoded = EA::ParseInputWidthExpansionProvenance(encoded);
    assert(decoded.CanonicalText() == encoded);
    std::vector<double> persisted(expanded.begin(), expanded.end());
    std::vector<float> reloadedExpanded(persisted.begin(), persisted.end());
    assert(std::memcmp(expanded.data(), reloadedExpanded.data(),
                       expanded.size() * sizeof(float)) == 0);

    // Simulate the real append-only lifecycle beyond the currently compiled
    // layout.  B records its immutable 51 -> 52 event under V1.  V2 adds one
    // registered feature at width 53, and V3 adds another at width 54.
    constexpr std::size_t v2Width = expandedWidth + 1;
    constexpr std::size_t v3Width = v2Width + 1;
    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 3>
        futureRegistry{{
            {1, expandedWidth, 0},
            {2, v2Width, 1},
            {3, v3Width, 2},
        }};
    std::vector<std::size_t> futureWidths{
        EA::kRegisteredModelInputWidths.begin(),
        EA::kRegisteredModelInputWidths.end()};
    futureWidths.push_back(v2Width);
    futureWidths.push_back(v3Width);
    std::vector<EA::AppendedTensorFeatureSemantic> futureSemantics{
        EA::kAppendedTensorFeatureSemantics.begin(),
        EA::kAppendedTensorFeatureSemantics.end()};
    futureSemantics.push_back({expandedWidth - EA::kModelReturnFeatureCount,
                               "synthetic_v2_feature"});
    futureSemantics.push_back({v2Width - EA::kModelReturnFeatureCount,
                               "synthetic_v3_feature"});

    const auto historicalUnderV2 =
        EA::ParseInputWidthExpansionProvenance(
            encoded, futureRegistry, futureWidths, futureSemantics, 2,
            v2Width);
    const auto historicalUnderV3 =
        EA::ParseInputWidthExpansionProvenance(
            encoded, futureRegistry, futureWidths, futureSemantics, 3,
            v3Width);
    assert(historicalUnderV2.semanticLayoutVersion == 1);
    assert(historicalUnderV3.semanticLayoutVersion == 1);
    assert(historicalUnderV3.expandedInputWidth == expandedWidth);

    // B can be re-expanded to C under V2.  C records only the new V2 event;
    // B's already-persisted V1 evidence remains byte-for-byte unchanged.
    const EA::InputWidthExpansionPlan v2Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            expandedWidth, v2Width, futureWidths, futureSemantics);
    const auto v2Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543211LL, v2Plan, 2, futureRegistry, futureWidths,
        futureSemantics, 2, v2Width);
    assert(v2Provenance.sourceInputWidth == expandedWidth);
    assert(v2Provenance.expandedInputWidth == v2Width);
    assert(v2Provenance.semanticLayoutVersion == 2);
    assert(v2Provenance.newlyIntroducedTensorFeatures ==
           std::vector<std::string>{"synthetic_v2_feature"});
    assert(provenance.CanonicalText() == encoded);
    const auto v2UnderV3 = EA::ParseInputWidthExpansionProvenance(
        v2Provenance.CanonicalText(), futureRegistry, futureWidths,
        futureSemantics, 3, v3Width);
    assert(v2UnderV3.semanticLayoutVersion == 2);

    ExpectFailureContaining(
        [] { (void)EA::BuildInputWidthExpansionPlan(37); },
        "MODEL_INPUT_WIDTH_UNSUPPORTED");
    ExpectFailureContaining(
        [] { (void)EA::BuildInputWidthExpansionPlan(53); },
        "SOURCE_WIDER_THAN_TARGET");
    ExpectFailureContaining(
        [] {
            (void)EA::BuildInputWidthExpansionPlan(
                EA::kCurrentModelInputWidth);
        },
        "MODEL_INPUT_EXPANSION_NOT_REQUIRED");
    ExpectFailureContaining(
        [&] {
            std::vector<float> malformed(source.size() - 1);
            (void)EA::ExpandFusedLstmParameterRowMajor(
                malformed, hiddenSize, plan);
        },
        "PARAMETER_SHAPE_MISMATCH");
    EA::ValidateModelInputSemanticMetadataForExpansion(
        1, 1, expandedWidth, futureRegistry, futureWidths, 2, v2Width);
    EA::ValidateModelInputSemanticMetadataForExpansion(
        1, 1, expandedWidth, futureRegistry, futureWidths, 3,
        v3Width);

    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 999, expandedWidth, futureRegistry, futureWidths, 2,
                v2Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 1, v2Width, futureRegistry, futureWidths, 2,
                v2Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");

    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 2>
        incompatibleFutureRegistry{{
            {1, expandedWidth, 0},
            {2, v2Width, 0},
        }};
    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 1, expandedWidth, incompatibleFutureRegistry,
                futureWidths, 2, v2Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");

    const std::array<double, 2> validSemanticMetadata{{1.0, 1.0}};
    const auto parsedSemanticMetadata = EA::ParseModelInputSemanticMetadata(
        1, 2, validSemanticMetadata);
    assert(parsedSemanticMetadata.schemaVersion == 1);
    assert(parsedSemanticMetadata.layoutVersion == 1);
    ExpectFailureContaining(
        [&] {
            (void)EA::ParseModelInputSemanticMetadata(
                2, 1, validSemanticMetadata);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    const std::array<double, 2> fractionalSemanticMetadata{{1.0, 1.5}};
    ExpectFailureContaining(
        [&] {
            (void)EA::ParseModelInputSemanticMetadata(
                1, 2, fractionalSemanticMetadata);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                999, 1, expandedWidth, futureRegistry, futureWidths, 2,
                v2Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string unknownLayout = encoded;
            const std::string needle = "semantic_layout=1";
            unknownLayout.replace(unknownLayout.find(needle), needle.size(),
                                  "semantic_layout=999");
            (void)EA::ParseInputWidthExpansionProvenance(
                unknownLayout, futureRegistry, futureWidths, futureSemantics,
                2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string impossibleWidth = encoded;
            const std::string needle = "semantic_layout=1";
            impossibleWidth.replace(impossibleWidth.find(needle), needle.size(),
                                    "semantic_layout=2");
            (void)EA::ParseInputWidthExpansionProvenance(
                impossibleWidth, futureRegistry, futureWidths,
                futureSemantics, 2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            (void)EA::ParseInputWidthExpansionProvenance(
                encoded, incompatibleFutureRegistry, futureWidths,
                futureSemantics, 2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string wrongPolicy = encoded;
            const std::string needle = "initialization=zero";
            wrongPolicy.replace(wrongPolicy.find(needle), needle.size(),
                                "initialization=random");
            (void)EA::ParseInputWidthExpansionProvenance(
                wrongPolicy, futureRegistry, futureWidths, futureSemantics,
                2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string wrongColumns = encoded;
            const std::string needle = "new_tensor_columns=47:48";
            wrongColumns.replace(wrongColumns.find(needle), needle.size(),
                                 "new_tensor_columns=46:48");
            (void)EA::ParseInputWidthExpansionProvenance(
                wrongColumns, futureRegistry, futureWidths, futureSemantics,
                2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string wrongFeatures = encoded;
            const std::string needle =
                "new_tensor_features=historical_level_proximity";
            wrongFeatures.replace(
                wrongFeatures.find(needle), needle.size(),
                "new_tensor_features=wrong_feature");
            (void)EA::ParseInputWidthExpansionProvenance(
                wrongFeatures, futureRegistry, futureWidths, futureSemantics,
                2, v2Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string malformed = encoded;
            const std::string needle = "source_input_width=51";
            malformed.replace(malformed.find(needle), needle.size(),
                              "source_input_width=-1");
            (void)EA::ParseInputWidthExpansionProvenance(malformed);
        },
        "PROVENANCE_INVALID_FIELD:source_input_width");
    ExpectFailureContaining(
        [&] {
            (void)EA::ParseInputWidthExpansionProvenance(encoded + ";extra=1");
        },
        "PROVENANCE_NOT_CANONICAL");

    return 0;
}
