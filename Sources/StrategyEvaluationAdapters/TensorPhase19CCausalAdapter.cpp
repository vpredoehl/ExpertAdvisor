#include "TensorPhase19CCausalAdapter.hpp"

#include "../../Headers/ModelInputContract.hpp"
#include "../../Headers/ModelInputFeatureSemantics.hpp"
#include "../../Headers/Params.hpp"
#include "../../Headers/ReturnFeatureHistory.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <stdexcept>

namespace EA::StrategyEvaluationAdapters
{
namespace
{

void ValidateContext(const TensorPhase19CCausalAdapterContext& context)
{
    if (context.inferenceWindowSize == 0)
        throw std::invalid_argument("phase19c_invalid_inference_window_size");
    if (!EA::IsModelInputSemanticLayoutWidthCompatible(
            context.semanticLayoutVersion, context.modelInputWidth,
            EA::kModelInputSemanticLayoutRegistry,
            EA::kRegisteredModelInputWidths,
            EA::kModelInputSemanticLayoutVersion,
            EA::kCurrentModelInputWidth, false))
        throw std::invalid_argument(
            "phase19c_feature_layout_identity_ambiguous");
}

} // namespace

TensorPhase19CCausalAdapterResult AdaptTensorPhase19CCausalEntryStates(
    const Tensor& tensor,
    const TensorPhase19CCausalAdapterContext& context,
    const std::vector<TensorInferenceDecision>& decisions)
{
    ValidateContext(context);
    const auto contract = EA::ResolveModelInputContract(
        context.modelInputWidth,
        tensor.begin() == tensor.end()
            ? 0 : static_cast<std::size_t>((*tensor.begin()).Shape()[1]));
    const auto semantics = EA::ModelInputFeatureSemantics(
        context.modelInputWidth);

    TensorPhase19CCausalAdapterResult result;
    result.featureAblationIdentity = context.featureAblationMask.CanonicalText();
    result.predictors.reserve(semantics.size());
    for (const auto& semantic : semantics)
        result.predictors.push_back({
            semantic.name, semantic.source, semantic.causalTiming,
            semantic.modelInputColumn, semantic.categorical});
    result.entryStates.reserve(decisions.size());

    for (std::size_t ordinal = 0; ordinal < decisions.size(); ++ordinal)
    {
        const auto& decision = decisions[ordinal];
        if (decision.windowStartRow >= tensor.RowCount() ||
            context.inferenceWindowSize - 1 >
                tensor.RowCount() - 1 - decision.windowStartRow)
            throw std::invalid_argument("phase19c_decision_row_out_of_bounds");
        const std::size_t decisionRow = decision.windowStartRow +
            context.inferenceWindowSize - 1;
        const auto iterator = tensor.begin() +
            static_cast<std::ptrdiff_t>(decisionRow);
        const auto row = MetaNN::LowerAccess(*iterator);
        std::vector<float> values(context.modelInputWidth, 0.0f);
        EA::CopyTensorFeaturesForModelInput(
            values.data(), row.RawMemory(), contract,
            context.featureAblationMask);
        const std::size_t appended =
            EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
                decisionRow, values.data(), contract.tensorFeatureCount,
                kFeatureScale,
                [&](std::size_t position) {
                    return tensor.RawCloseAtIterator(
                        tensor.begin() + static_cast<std::ptrdiff_t>(position));
                });
        if (appended != EA::kModelReturnFeatureCount)
            throw std::runtime_error("phase19c_return_feature_count_mismatch");

        const int predictedClass = StrategyEvaluation::
            PredictedClassForProbabilities(decision.probabilities);
        const double directionalProbability = predictedClass ==
                InferenceProfitability::kNeutralClass
            ? static_cast<double>(decision.probabilities.downNeutralUp[1])
            : static_cast<double>(decision.probabilities.downNeutralUp[
                  static_cast<std::size_t>(predictedClass)]);
        const double confidence = StrategyEvaluation::
            NormalizedDirectionalConfidenceForProbability(
                directionalProbability);
        const auto timestamp = tensor.RawTimeAtIterator(iterator);
        StrategyEvaluation::Phase19CCausalEntryState state;
        state.observationOrdinal = ordinal;
        state.entrySourceRow = decisionRow;
        state.entryTimestampUnixSeconds =
            std::chrono::duration_cast<std::chrono::seconds>(
                timestamp.time_since_epoch()).count();
        state.predictedClass = predictedClass;
        state.direction = predictedClass == InferenceProfitability::kUpClass
            ? "long" : (predictedClass == InferenceProfitability::kDownClass
                ? "short" : "flat");
        state.directionalProbability = directionalProbability;
        state.normalizedDirectionalConfidence = confidence;
        state.predictorValues.assign(values.begin(), values.end());
        result.entryStates.push_back(std::move(state));
    }
    return result;
}

TensorPhase19CCausalAdapterResult AdaptTensorPhase19CCausalJoinRequests(
    const Tensor& tensor,
    const TensorPhase19CCausalAdapterContext& context,
    std::vector<StrategyEvaluation::Phase19CCausalEntryState> requests)
{
    ValidateContext(context);
    const auto contract = EA::ResolveModelInputContract(
        context.modelInputWidth,
        tensor.begin() == tensor.end()
            ? 0 : static_cast<std::size_t>((*tensor.begin()).Shape()[1]));
    const auto semantics = EA::ModelInputFeatureSemantics(
        context.modelInputWidth);
    TensorPhase19CCausalAdapterResult result;
    result.featureAblationIdentity = context.featureAblationMask.CanonicalText();
    for (const auto& semantic : semantics)
        result.predictors.push_back({
            semantic.name, semantic.source, semantic.causalTiming,
            semantic.modelInputColumn, semantic.categorical});
    result.entryStates.reserve(requests.size());
    for (auto& request : requests)
    {
        if (request.entrySourceRow >= tensor.RowCount())
            throw std::runtime_error("phase19c_entry_source_row_out_of_bounds");
        const auto iterator = tensor.begin() +
            static_cast<std::ptrdiff_t>(request.entrySourceRow);
        const auto tensorTimestamp =
            std::chrono::duration_cast<std::chrono::seconds>(
                tensor.RawTimeAtIterator(iterator).time_since_epoch()).count();
        if (tensorTimestamp > request.entryTimestampUnixSeconds)
            throw std::runtime_error("phase19c_predictor_timestamp_after_entry");
        if (tensorTimestamp != request.entryTimestampUnixSeconds)
            throw std::runtime_error("phase19c_entry_timestamp_join_mismatch");
        const auto row = MetaNN::LowerAccess(*iterator);
        std::vector<float> values(context.modelInputWidth, 0.0f);
        EA::CopyTensorFeaturesForModelInput(
            values.data(), row.RawMemory(), contract,
            context.featureAblationMask);
        const std::size_t appended =
            EA::AppendMultiHorizonReturnFeaturesAtGlobalPosition(
                request.entrySourceRow, values.data(),
                contract.tensorFeatureCount, kFeatureScale,
                [&](std::size_t position) {
                    return tensor.RawCloseAtIterator(
                        tensor.begin() + static_cast<std::ptrdiff_t>(position));
                });
        if (appended != EA::kModelReturnFeatureCount)
            throw std::runtime_error("phase19c_return_feature_count_mismatch");
        request.predictorValues.assign(values.begin(), values.end());
        result.entryStates.push_back(std::move(request));
    }
    return result;
}

} // namespace EA::StrategyEvaluationAdapters
