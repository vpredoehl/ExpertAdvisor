#include "InferenceRuntime.hpp"

#include "LSTM.hpp"
#include "ModelInputContract.hpp"
#include "Tensor.hpp"

#include <cmath>
#include <stdexcept>
#include <utility>

namespace EA::Inference
{
ModelInputPreparation::Result PrepareInferenceInput(
    const ModelInputPreparation::Request& request,
    const ModelInputPreparation::DatabaseConnectionSettings& settings)
{
    return ModelInputPreparation::Prepare(request, settings);
}

RuntimeResult RunInferenceRuntime(const RuntimeRequest& request)
{
    const auto stage = [&request](const char* name)
    {
        if (request.stageObserver) request.stageObserver(name);
    };
    const auto& persisted = request.materialization;
    const auto& configuration = request.configuration;
    stage("model_config_validation_begun");
    if (configuration.windowSize == 0 || configuration.predictionHorizon == 0)
        throw std::invalid_argument("inference_runtime_invalid_window_or_horizon");
    if (persisted.identity.modelId <= 0)
        throw std::invalid_argument("inference_runtime_invalid_model_identity");
    if (persisted.modelMeta.hiddenSize == 0)
        throw std::invalid_argument("inference_runtime_invalid_model_hidden_size");
    if (persisted.targetMeta.has_value() &&
        persisted.targetMeta->targetType != configuration.targetType)
    {
        throw std::invalid_argument("inference_runtime_target_type_mismatch");
    }
    if (persisted.trainConfigMeta.has_value())
    {
        const auto& meta = *persisted.trainConfigMeta;
        if (meta.rows != 1 ||
            meta.cols < DBIO::PgModelIO::kTrainConfigMetaFieldCount ||
            meta.values.size() < static_cast<std::size_t>(
                DBIO::PgModelIO::kTrainConfigMetaFieldCount))
        {
            throw std::invalid_argument(
                "inference_runtime_invalid_train_config_meta");
        }
        if (std::llround(meta.values[1]) !=
                static_cast<long long>(configuration.predictionHorizon) ||
            std::fabs(meta.values[2] -
                      static_cast<double>(configuration.thresholdLogret)) >
                1e-7 ||
            std::llround(meta.values[3]) !=
                static_cast<long long>(configuration.windowSize))
        {
            throw std::invalid_argument(
                "inference_runtime_materialized_config_mismatch");
        }
    }

    // Validate the materialized model contract before construction.  This is
    // value-only and therefore remains valid after the RR/RO reader commits.
    const std::size_t tensorFeatureWidth =
        request.tensor.begin() != request.tensor.end()
            ? static_cast<std::size_t>((*request.tensor.begin()).Shape()[1])
            : 0;
    const auto inputContract = EA::ResolveModelInputContract(
        persisted.modelMeta.inputWidth, tensorFeatureWidth);
    stage("model_config_validation_completed");
    stage("lstm_construction_begun");
    auto model = std::make_unique<EA::LSTM>(
        request.tensor, persisted.modelMeta.hiddenSize, 1.0f, 0.0f,
        configuration.targetType, inputContract.modelInputWidth,
        persisted.identity.featureAblationMask);
    stage("lstm_construction_completed");

    // The detached applier deliberately accepts no transaction or connection.
    stage("detached_materialization_apply_begun");
    DBIO::PgModelIO::ApplyPersistedModelMaterialization(persisted, *model);
    stage("detached_materialization_apply_completed");

    RuntimeResult result;
    stage("evaluation_begun");
    result.evaluationFacts = InferenceEvaluationFacts::Evaluate(
        *model, request.tensor,
        {configuration.windowSize, configuration.predictionHorizon,
         configuration.thresholdLogret, configuration.logicalOutputStartIndex,
         configuration.captureStrategyDecisions});
    stage("evaluation_completed");
    result.model = std::move(model);
    return result;
}
} // namespace EA::Inference
