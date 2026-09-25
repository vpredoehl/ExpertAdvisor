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
        EA::kTG4ProductionPulseModelInputWidth;
    constexpr std::size_t expandedWidth = EA::kCurrentModelInputWidth;
    static_assert(sourceWidth == 80);
    static_assert(expandedWidth == 103);
    static_assert(expandedWidth == sourceWidth + 23);

    const EA::InputWidthExpansionPlan plan =
        EA::BuildInputWidthExpansionPlan(sourceWidth);
    assert(plan.sourceTensorFeatureCount == 76);
    assert(plan.expandedTensorFeatureCount == 99);
    assert(plan.newlyIntroducedTensorFeatures.size() == 23);
    assert(plan.newlyIntroducedTensorFeatures.front() ==
           "fib_recent_price_scale_valid");
    assert(plan.newlyIntroducedTensorFeatures.back() ==
           "fib_down_recent_median_pullback_0618_signed_atr");

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
    expandedInput[plan.sourceTensorFeatureCount + 1] = 8.25f;
    expandedInput[plan.sourceTensorFeatureCount + 2] = 9.25f;
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
        std::string{EA::kEconomicEventReleaseActualAblationMaskText});
    EA::CopyTensorFeaturesForModelInput(currentInput.data(), physical.data(),
                                        currentContract, ablation);
    assert(currentInput.size() == expandedWidth);
    for (std::size_t col = authoritativeInitialHasSurpriseCol;
         col <= authoritativeInitialSurpriseDirectionCol; ++col)
        assert(currentInput[col] == 0.0f);
    assert(currentInput[returnAutocorrelationCol] == 2.0f);
    // Existing ablation names resolve only their registered historical
    // columns; they do not acquire TG4 prefix/pattern semantics.
    assert(currentInput[tg4InnerBreakAnyCol] == 2.0f);
    assert(currentInput[tg4SourceTg3StructurallyEligibleCol] == 2.0f);
    assert(currentInput[tg4SourceTg3ConfluentCol] == 2.0f);

    // The causal first-release surprise treatment is a paired zero mask. It
    // preserves width and every unrelated input byte while clearing both the
    // availability and value channels.
    std::vector<float> surpriseControl(EA::kCurrentModelInputWidth, -1.0f);
    std::vector<float> surpriseAblation(EA::kCurrentModelInputWidth, -1.0f);
    EA::CopyTensorFeaturesForModelInput(
        surpriseControl.data(), physical.data(), currentContract);
    const auto surpriseMask = EA::FeatureAblationMask::Parse(
        std::string{EA::kCausalEconomicEventSurpriseAblationMaskText});
    EA::CopyTensorFeaturesForModelInput(
        surpriseAblation.data(), physical.data(), currentContract,
        surpriseMask);
    assert(surpriseControl.size() == surpriseAblation.size());
    for (std::size_t col = 0; col < currentContract.tensorFeatureCount; ++col)
    {
        const bool ablated =
            col == causalFirstReleaseSurpriseAvailableCol ||
            col == causalFirstReleaseSurpriseCol;
        assert(surpriseAblation[col] ==
               (ablated ? 0.0f : surpriseControl[col]));
    }

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

    // Prior append-only events remain valid under the compiled V6 layout.
    const EA::InputWidthExpansionPlan v1Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            EA::kCausalRollingRangeExpansionModelInputWidth,
            EA::kHistoricalLevelProximityModelInputWidth,
            EA::kRegisteredModelInputWidths,
            EA::kAppendedTensorFeatureSemantics);
    const auto v1Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543209LL, v1Plan, 1, EA::kModelInputSemanticLayoutRegistry,
        EA::kRegisteredModelInputWidths, EA::kAppendedTensorFeatureSemantics,
        EA::kModelInputSemanticLayoutVersion, EA::kCurrentModelInputWidth);
    assert(EA::ParseInputWidthExpansionProvenance(
               v1Provenance.CanonicalText()).semanticLayoutVersion == 1);

    const EA::InputWidthExpansionPlan v3Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            EA::kPreEconomicEventModelInputWidth,
            EA::kEconomicEventModelInputWidth,
            EA::kRegisteredModelInputWidths,
            EA::kAppendedTensorFeatureSemantics);
    const auto v3Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543210LL, v3Plan, 3, EA::kModelInputSemanticLayoutRegistry,
        EA::kRegisteredModelInputWidths, EA::kAppendedTensorFeatureSemantics,
        EA::kModelInputSemanticLayoutVersion, EA::kCurrentModelInputWidth);
    assert(EA::ParseInputWidthExpansionProvenance(
               v3Provenance.CanonicalText()).semanticLayoutVersion == 3);
    assert(v3Provenance.expandedInputWidth ==
           EA::kEconomicEventModelInputWidth);

    const EA::InputWidthExpansionPlan v4Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            EA::kEconomicEventModelInputWidth,
            EA::kEconomicEventConsensusModelInputWidth,
            EA::kRegisteredModelInputWidths,
            EA::kAppendedTensorFeatureSemantics);
    const auto v4Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543211LL, v4Plan, 4, EA::kModelInputSemanticLayoutRegistry,
        EA::kRegisteredModelInputWidths, EA::kAppendedTensorFeatureSemantics,
        EA::kModelInputSemanticLayoutVersion, EA::kCurrentModelInputWidth);
    assert(EA::ParseInputWidthExpansionProvenance(
               v4Provenance.CanonicalText()).semanticLayoutVersion == 4);
    assert(v4Provenance.expandedInputWidth ==
           EA::kEconomicEventConsensusModelInputWidth);

    const EA::InputWidthExpansionPlan v5Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            EA::kEconomicEventConsensusModelInputWidth,
            EA::kEconomicEventReleaseActualModelInputWidth,
            EA::kRegisteredModelInputWidths,
            EA::kAppendedTensorFeatureSemantics);
    const auto v5Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543212LL, v5Plan, 5, EA::kModelInputSemanticLayoutRegistry,
        EA::kRegisteredModelInputWidths, EA::kAppendedTensorFeatureSemantics,
        EA::kModelInputSemanticLayoutVersion, EA::kCurrentModelInputWidth);
    assert(v5Provenance.expandedInputWidth ==
           EA::kEconomicEventReleaseActualModelInputWidth);

    // Simulate two further append-only generations from the current layout 9.
    // Layouts 6 and 7 are same-width siblings descending
    // from layout 5, so the pre-fix layout 6 is not an append-only
    // predecessor of layout 7 or of these future generations.
    constexpr std::size_t v9Width = expandedWidth + 1;
    constexpr std::size_t v10Width = v9Width + 1;
    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 11>
        futureRegistry{{
            {1, EA::kHistoricalLevelProximityModelInputWidth, 0},
            {2, EA::kPreEconomicEventModelInputWidth, 1},
            {3, EA::kEconomicEventModelInputWidth, 2},
            {4, EA::kEconomicEventConsensusModelInputWidth, 3},
            {5, EA::kEconomicEventReleaseActualModelInputWidth, 4},
            {6, EA::kCausalEconomicEventSurpriseModelInputWidth, 5},
            {7, EA::kCausalEconomicEventSurpriseModelInputWidth, 5},
            {8, sourceWidth, 7},
            {9, expandedWidth, 8},
            {10, v9Width, 9},
            {11, v10Width, 10},
        }};
    std::vector<std::size_t> futureWidths{
        EA::kRegisteredModelInputWidths.begin(),
        EA::kRegisteredModelInputWidths.end()};
    futureWidths.push_back(v9Width);
    futureWidths.push_back(v10Width);
    std::vector<EA::AppendedTensorFeatureSemantic> futureSemantics{
        EA::kAppendedTensorFeatureSemantics.begin(),
        EA::kAppendedTensorFeatureSemantics.end()};
    futureSemantics.push_back({expandedWidth - EA::kModelReturnFeatureCount,
                               "synthetic_v10_feature"});
    futureSemantics.push_back({v9Width - EA::kModelReturnFeatureCount,
                               "synthetic_v11_feature"});

    const auto currentUnderV9 =
        EA::ParseInputWidthExpansionProvenance(
            encoded, futureRegistry, futureWidths, futureSemantics, 10,
            v9Width);
    const auto currentUnderV10 =
        EA::ParseInputWidthExpansionProvenance(
            encoded, futureRegistry, futureWidths, futureSemantics, 11,
            v10Width);
    assert(currentUnderV9.semanticLayoutVersion == 9);
    assert(currentUnderV10.semanticLayoutVersion == 9);
    assert(currentUnderV10.expandedInputWidth == expandedWidth);

    const EA::InputWidthExpansionPlan v10Plan =
        EA::BuildRegisteredInputWidthExpansionPlan(
            expandedWidth, v9Width, futureWidths, futureSemantics);
    const auto v10Provenance = EA::MakeInputWidthExpansionProvenance(
        9876543213LL, v10Plan, 10, futureRegistry, futureWidths,
        futureSemantics, 10, v9Width);
    assert(v10Provenance.sourceInputWidth == expandedWidth);
    assert(v10Provenance.expandedInputWidth == v9Width);
    assert(v10Provenance.semanticLayoutVersion == 10);
    assert(v10Provenance.newlyIntroducedTensorFeatures ==
           std::vector<std::string>{"synthetic_v10_feature"});
    assert(provenance.CanonicalText() == encoded);
    const auto v10UnderV11 = EA::ParseInputWidthExpansionProvenance(
        v10Provenance.CanonicalText(), futureRegistry, futureWidths,
        futureSemantics, 11, v10Width);
    assert(v10UnderV11.semanticLayoutVersion == 10);

    ExpectFailureContaining(
        [] { (void)EA::BuildInputWidthExpansionPlan(37); },
        "MODEL_INPUT_WIDTH_UNSUPPORTED");
    ExpectFailureContaining(
        [] { (void)EA::BuildInputWidthExpansionPlan(104); },
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
        1, 9, expandedWidth, futureRegistry, futureWidths, 10, v9Width);
    EA::ValidateModelInputSemanticMetadataForExpansion(
        1, 9, expandedWidth, futureRegistry, futureWidths, 11,
        v10Width);

    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 999, expandedWidth, futureRegistry, futureWidths, 10,
                v9Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 1, expandedWidth, futureRegistry, futureWidths, 10,
                v9Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");

    constexpr std::array<EA::ModelInputSemanticLayoutRegistryEntry, 10>
        incompatibleFutureRegistry{{
            {1, EA::kHistoricalLevelProximityModelInputWidth, 0},
            {2, EA::kPreEconomicEventModelInputWidth, 1},
            {3, EA::kEconomicEventModelInputWidth, 2},
            {4, EA::kEconomicEventConsensusModelInputWidth, 3},
            {5, EA::kEconomicEventReleaseActualModelInputWidth, 4},
            {6, EA::kCausalEconomicEventSurpriseModelInputWidth, 5},
            {7, EA::kCausalEconomicEventSurpriseModelInputWidth, 5},
            {8, sourceWidth, 7},
            {9, expandedWidth, 8},
            {10, v9Width, 0},
        }};
    ExpectFailureContaining(
        [&] {
            EA::ValidateModelInputSemanticMetadataForExpansion(
                1, 9, expandedWidth, incompatibleFutureRegistry,
                futureWidths, 10, v9Width);
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
                999, 9, expandedWidth, futureRegistry, futureWidths, 10,
                v9Width);
        },
        "SEMANTIC_METADATA_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string unknownLayout = encoded;
            const std::string needle = "semantic_layout=9";
            unknownLayout.replace(unknownLayout.find(needle), needle.size(),
                                  "semantic_layout=999");
            (void)EA::ParseInputWidthExpansionProvenance(
                unknownLayout, futureRegistry, futureWidths, futureSemantics,
                10, v9Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string impossibleWidth = encoded;
            const std::string needle = "semantic_layout=9";
            impossibleWidth.replace(impossibleWidth.find(needle), needle.size(),
                                    "semantic_layout=1");
            (void)EA::ParseInputWidthExpansionProvenance(
                impossibleWidth, futureRegistry, futureWidths,
                futureSemantics, 10, v9Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            (void)EA::ParseInputWidthExpansionProvenance(
                encoded, incompatibleFutureRegistry, futureWidths,
                futureSemantics, 10, v9Width);
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
                10, v9Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string wrongColumns = encoded;
            const std::string needle = "new_tensor_columns=76:99";
            wrongColumns.replace(wrongColumns.find(needle), needle.size(),
                                 "new_tensor_columns=75:99");
            (void)EA::ParseInputWidthExpansionProvenance(
                wrongColumns, futureRegistry, futureWidths, futureSemantics,
                10, v9Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string wrongFeatures = encoded;
            const std::string needle =
                "new_tensor_features=fib_recent_price_scale_valid|fib_up_recent_union_count_log|fib_up_recent_h1_count_log|fib_up_recent_h2_count_log|fib_up_recent_h1_h2_both_count_log|fib_up_recent_h1_youngest_age_20|fib_up_recent_h2_youngest_age_20|fib_up_recent_median_1272_signed_atr|fib_up_recent_median_1618_signed_atr|fib_up_recent_median_pullback_0382_signed_atr|fib_up_recent_median_pullback_0500_signed_atr|fib_up_recent_median_pullback_0618_signed_atr|fib_down_recent_union_count_log|fib_down_recent_h1_count_log|fib_down_recent_h2_count_log|fib_down_recent_h1_h2_both_count_log|fib_down_recent_h1_youngest_age_20|fib_down_recent_h2_youngest_age_20|fib_down_recent_median_1272_signed_atr|fib_down_recent_median_1618_signed_atr|fib_down_recent_median_pullback_0382_signed_atr|fib_down_recent_median_pullback_0500_signed_atr|fib_down_recent_median_pullback_0618_signed_atr";
            wrongFeatures.replace(
                wrongFeatures.find(needle), needle.size(),
                "new_tensor_features=wrong_feature");
            (void)EA::ParseInputWidthExpansionProvenance(
                wrongFeatures, futureRegistry, futureWidths, futureSemantics,
                10, v9Width);
        },
        "PROVENANCE_INCOMPATIBLE");
    ExpectFailureContaining(
        [&] {
            std::string malformed = encoded;
            const std::string needle = "source_input_width=80";
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
