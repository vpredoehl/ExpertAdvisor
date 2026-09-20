#pragma once

#include "InferenceEvaluationFacts.hpp"
#include "LSTM.hpp"
#include "ModelInputPreparation.hpp"
#include "PgModelIO.hpp"

#include <cstddef>
#include <memory>
#include <optional>

namespace EA
{
namespace Inference
{
// Value-only inputs for the detached inference computation.  Applications
// choose and materialize the model before calling this boundary; no database
// transaction is accepted or retained here.
struct RuntimeConfiguration
{
    std::size_t windowSize = 0;
    std::size_t predictionHorizon = 0;
    float thresholdLogret = 0.0f;
    std::size_t logicalOutputStartIndex = 0;
    EA::LSTM::TargetType targetType =
        EA::LSTM::TargetType::UpNeutralDownReturn;
    bool captureStrategyDecisions = false;
};

struct RuntimeRequest
{
    const ::Tensor& tensor;
    const DBIO::PgModelIO::PersistedModelMaterialization& materialization;
    RuntimeConfiguration configuration;
};

struct RuntimeResult
{
    // The caller may use the loaded model only for presentation or
    // application-owned artifact workflows.  It is already detached and
    // fully applied by RunInferenceRuntime.
    std::unique_ptr<EA::LSTM> model;
    InferenceEvaluationFacts::EvaluationFacts evaluationFacts;
};

// Owns Tensor materialization for independently-invocable inference callers.
// Snapshot selection and lifecycle policy remain with the application.
ModelInputPreparation::Result PrepareInferenceInput(
    const ModelInputPreparation::Request& request,
    const ModelInputPreparation::DatabaseConnectionSettings& settings);

// Constructs the LSTM, applies a query-free detached persisted model, and
// obtains the Phase 22Z1 evaluation facts.  It has no CLI, scheduler,
// persistence, or transaction dependency.
RuntimeResult RunInferenceRuntime(const RuntimeRequest& request);

} // namespace Inference
} // namespace EA
