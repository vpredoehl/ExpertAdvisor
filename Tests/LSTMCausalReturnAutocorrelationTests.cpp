#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstring>
#include <limits>
#include <vector>

#include <MetaNN/data/facilities/lower_access.h>

#include "CausalReturnAutocorrelationFeatures.hpp"
#include "FeatureAblation.hpp"
#include "FeatureLayout.hpp"
#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "PricePoint.hpp"
#include "Tensor.hpp"

extern "C" bool LstmRuntimeDiagnosticLoggingEnabled() { return false; }

namespace
{
float Advance(float close, double logReturn)
{
    return static_cast<float>(static_cast<double>(close) *
                              std::exp(logReturn));
}

float ValueForReturns(const std::vector<double>& returns)
{
    CausalReturnAutocorrelation32 feature;
    float close = 100.0f;
    float result = feature.AddCompletedClose(close);
    for (const double value : returns)
    {
        close = Advance(close, value);
        result = feature.AddCompletedClose(close);
    }
    return result;
}

void AssertBoundedFinite(float value)
{
    assert(std::isfinite(value));
    assert(value >= -1.0f && value <= 1.0f);
}

std::vector<double> PositiveReturns()
{
    std::vector<double> result;
    result.reserve(causalReturnAutocorrelationPairCount + 1);
    for (int index = -16; index <= 16; ++index)
        result.push_back(static_cast<double>(index) * 0.0001);
    return result;
}

std::vector<double> NegativeReturns()
{
    std::vector<double> result;
    result.reserve(causalReturnAutocorrelationPairCount + 1);
    for (std::size_t index = 0;
         index <= causalReturnAutocorrelationPairCount; ++index)
    {
        result.push_back(index % 2 == 0 ? 0.001 : -0.001);
    }
    return result;
}

std::vector<double> NearZeroReturns()
{
    // Eight complete [a,0,-a,0] cycles occur in each 32-value side.
    // Every adjacent product is zero and both side means are zero.
    std::vector<double> result;
    result.reserve(causalReturnAutocorrelationPairCount + 1);
    constexpr std::array<double, 4> cycle{{0.001, 0.0, -0.001, 0.0}};
    for (std::size_t index = 0;
         index <= causalReturnAutocorrelationPairCount; ++index)
    {
        result.push_back(cycle[index % cycle.size()]);
    }
    return result;
}

void TestDefinitionDegeneracyAndBounds()
{
    static_assert(causalReturnAutocorrelationPairCount == 32);
    CausalReturnAutocorrelation32 insufficient;
    float close = 100.0f;
    assert(insufficient.AddCompletedClose(close) == 0.0f);
    for (std::size_t index = 0;
         index < causalReturnAutocorrelationPairCount; ++index)
    {
        close = Advance(close, index % 2 == 0 ? 0.001 : -0.001);
        assert(insufficient.AddCompletedClose(close) == 0.0f);
    }
    assert(insufficient.CompletedReturnCount() ==
           causalReturnAutocorrelationPairCount);

    const float positive = ValueForReturns(PositiveReturns());
    const float negative = ValueForReturns(NegativeReturns());
    const float nearZero = ValueForReturns(NearZeroReturns());
    assert(positive > 0.99f);
    assert(negative < -0.99f);
    assert(std::fabs(nearZero) < 0.02f);

    CausalReturnAutocorrelation32 zeroVariance;
    for (std::size_t index = 0;
         index < causalReturnAutocorrelationPairCount + 2; ++index)
    {
        const float value = zeroVariance.AddCompletedClose(100.0f);
        assert(value == 0.0f);
        AssertBoundedFinite(value);
    }
    for (const float value : {positive, negative, nearZero})
        AssertBoundedFinite(value);

    CausalReturnAutocorrelation32 invalidCurrent;
    float invalidClose = 100.0f;
    (void)invalidCurrent.AddCompletedClose(invalidClose);
    for (const double value : PositiveReturns())
    {
        invalidClose = Advance(invalidClose, value);
        (void)invalidCurrent.AddCompletedClose(invalidClose);
    }
    assert(invalidCurrent.AddCompletedClose(
               std::numeric_limits<float>::quiet_NaN()) == 0.0f);
    assert(invalidCurrent.CompletedReturnCount() == 0);
}

Feature BarAt(std::size_t index, float close)
{
    Feature bar{close, close, close * 1.001f, close * 0.999f,
                PriceTP{std::chrono::seconds{
                    static_cast<long long>(index * 15 * 60)}}};
    bar.tickVolume = 100.0f + static_cast<float>(index % 7);
    return bar;
}

std::vector<float> ClosesForReturns(const std::vector<double>& returns)
{
    std::vector<float> closes;
    closes.reserve(returns.size() + 1);
    closes.push_back(100.0f);
    for (const double value : returns)
        closes.push_back(Advance(closes.back(), value));
    return closes;
}

Tensor TensorForCloses(const char* name, const std::vector<float>& closes)
{
    Tensor tensor{name};
    for (std::size_t index = 0; index < closes.size(); ++index)
        tensor.Add(BarAt(index, closes[index]));
    return tensor;
}

float RowValue(const Tensor& tensor, std::size_t row, std::size_t column)
{
    return MetaNN::LowerAccess(
        *(tensor.begin() + static_cast<std::ptrdiff_t>(row)))
        .RawMemory()[column];
}

void TestCausalitySensitivityAndExactBoundary()
{
    const auto baseReturns = PositiveReturns();
    const auto baseCloses = ClosesForReturns(baseReturns);
    Tensor futureA = TensorForCloses("autocorrelation-future-a", baseCloses);
    Tensor futureB = TensorForCloses("autocorrelation-future-b", baseCloses);
    const std::size_t observation = baseCloses.size() - 1;
    const float beforeFuture =
        RowValue(futureA, observation, returnAutocorrelationCol);
    futureA.Add(BarAt(observation + 1,
                      Advance(baseCloses.back(), 0.5)));
    futureB.Add(BarAt(observation + 1,
                      Advance(baseCloses.back(), -0.5)));
    assert(RowValue(futureA, observation, returnAutocorrelationCol) ==
           beforeFuture);
    assert(RowValue(futureB, observation, returnAutocorrelationCol) ==
           beforeFuture);

    auto insideChanged = baseReturns;
    insideChanged[16] = -0.003;
    const float changed = ValueForReturns(insideChanged);
    assert(changed != ValueForReturns(baseReturns));

    std::vector<double> boundaryReturns;
    boundaryReturns.reserve(causalReturnAutocorrelationPairCount + 2);
    for (std::size_t index = 0;
         index < causalReturnAutocorrelationPairCount + 2; ++index)
    {
        boundaryReturns.push_back(
            static_cast<double>(static_cast<int>(index % 7) - 3) * 0.0003);
    }
    const auto boundaryCloses = ClosesForReturns(boundaryReturns);
    auto outsideChangedCloses = boundaryCloses;
    outsideChangedCloses[0] *= 1.25f;
    Tensor boundaryBase = TensorForCloses("autocorrelation-boundary-base",
                                          boundaryCloses);
    Tensor boundaryOutside = TensorForCloses(
        "autocorrelation-boundary-outside", outsideChangedCloses);
    const std::size_t boundaryObservation = boundaryCloses.size() - 1;
    const float boundaryValue = RowValue(
        boundaryBase, boundaryObservation, returnAutocorrelationCol);
    assert(RowValue(boundaryOutside, boundaryObservation,
                    returnAutocorrelationCol) == boundaryValue);

    auto insideChangedCloses = boundaryCloses;
    insideChangedCloses[1] *= 1.01f;
    Tensor boundaryInside = TensorForCloses(
        "autocorrelation-boundary-inside", insideChangedCloses);
    assert(RowValue(boundaryInside, boundaryObservation,
                    returnAutocorrelationCol) != boundaryValue);
}

void TestTensorParityAblationAndHistoricalPrefix()
{
    static_assert(historicalLevelProximityCol == 47);
    static_assert(returnAutocorrelationCol == 48);
    static_assert(return_autocorrelation_feature_size == 49);
    static_assert(feature_size == 67);
    static_assert(EA::kHistoricalLevelProximityModelInputWidth == 52);
    static_assert(EA::kReturnAutocorrelationModelInputWidth == 53);
    static_assert(EA::kPreEconomicEventModelInputWidth == 53);
    static_assert(EA::kCurrentModelInputWidth == 71);

    const auto closes = ClosesForReturns(NegativeReturns());
    Tensor trainingTensor = TensorForCloses("autocorrelation-training", closes);
    Tensor inferenceTensor = TensorForCloses("autocorrelation-inference", closes);
    const std::size_t observation = closes.size() - 1;
    const float* trainingRow = MetaNN::LowerAccess(
        *(trainingTensor.begin() + static_cast<std::ptrdiff_t>(observation)))
        .RawMemory();
    const float* inferenceRow = MetaNN::LowerAccess(
        *(inferenceTensor.begin() + static_cast<std::ptrdiff_t>(observation)))
        .RawMemory();
    assert(std::memcmp(trainingRow, inferenceRow,
                       feature_size * sizeof(float)) == 0);
    assert(trainingRow[returnAutocorrelationCol] < -0.99f);

    const auto current = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::array<float, EA::kCurrentModelInputWidth> trainingInput{};
    std::array<float, EA::kCurrentModelInputWidth> inferenceInput{};
    EA::CopyTensorFeaturesForModelInput(
        trainingInput.data(), trainingRow, current);
    EA::CopyTensorFeaturesForModelInput(
        inferenceInput.data(), inferenceRow, current);
    assert(std::memcmp(trainingInput.data(), inferenceInput.data(),
                       feature_size * sizeof(float)) == 0);

    const auto mask = EA::FeatureAblationMask::Parse(
        "return_autocorrelation");
    assert(mask.CanonicalText() == "return_autocorrelation");
    auto ablated = trainingInput;
    EA::CopyTensorFeaturesForModelInput(
        ablated.data(), trainingRow, current, mask);
    assert(ablated.size() == trainingInput.size());
    assert(ablated[returnAutocorrelationCol] == 0.0f);
    for (std::size_t column = 0; column < ablated.size(); ++column)
        if (column != returnAutocorrelationCol)
            assert(ablated[column] == trainingInput[column]);

    const auto historical = EA::ResolveModelInputContract(
        EA::kHistoricalLevelProximityModelInputWidth, feature_size);
    assert(historical.tensorFeatureCount ==
           historical_level_proximity_feature_size);
    std::array<float, EA::kHistoricalLevelProximityModelInputWidth>
        historicalInput{};
    historicalInput.fill(-7.0f);
    EA::CopyTensorFeaturesForModelInput(
        historicalInput.data(), trainingRow, historical);
    assert(std::memcmp(historicalInput.data(), trainingRow,
                       historical_level_proximity_feature_size *
                           sizeof(float)) == 0);
    assert(historicalInput[returnAutocorrelationCol] == -7.0f);
}

void TestGenericInputWidthExpansion()
{
    constexpr std::size_t hiddenSize = 2;
    constexpr std::size_t gateColumns = 4 * hiddenSize;
    const auto plan = EA::BuildInputWidthExpansionPlan(
        EA::kHistoricalLevelProximityModelInputWidth);
    assert(plan.sourceTensorFeatureCount == returnAutocorrelationCol);
    assert(plan.expandedTensorFeatureCount == feature_size);
    assert((plan.newlyIntroducedTensorFeatures ==
            std::vector<std::string>{
                "return_autocorrelation",
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
                "released_event_surprise_direction"}));

    std::vector<float> source(
        (plan.sourceInputWidth + hiddenSize) * gateColumns);
    for (std::size_t index = 0; index < source.size(); ++index)
        source[index] = static_cast<float>(index + 1) / 1024.0f;
    auto expanded = EA::ExpandFusedLstmParameterRowMajor(
        source, hiddenSize, plan);
    assert(std::memcmp(expanded.data(), source.data(),
                       plan.sourceTensorFeatureCount * gateColumns *
                           sizeof(float)) == 0);
    for (std::size_t column = 0; column < gateColumns; ++column)
        assert(expanded[returnAutocorrelationCol * gateColumns + column] ==
               0.0f);
    for (std::size_t returnIndex = 0;
         returnIndex < EA::kModelReturnFeatureCount; ++returnIndex)
    {
        assert(std::memcmp(
            expanded.data() +
                (plan.expandedTensorFeatureCount + returnIndex) * gateColumns,
            source.data() +
                (plan.sourceTensorFeatureCount + returnIndex) * gateColumns,
            gateColumns * sizeof(float)) == 0);
    }

    bool trainable = false;
    constexpr float featureValue = 0.75f;
    for (std::size_t column = 0; column < gateColumns; ++column)
    {
        const float gradient = featureValue *
            (0.1f + static_cast<float>(column) * 0.05f);
        expanded[returnAutocorrelationCol * gateColumns + column] -=
            0.01f * gradient;
        trainable = trainable ||
            expanded[returnAutocorrelationCol * gateColumns + column] != 0.0f;
    }
    assert(trainable);
}
} // namespace

int main()
{
    TestDefinitionDegeneracyAndBounds();
    TestCausalitySensitivityAndExactBoundary();
    TestTensorParityAblationAndHistoricalPrefix();
    TestGenericInputWidthExpansion();
    return 0;
}
