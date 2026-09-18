#pragma once

#include "SemanticWorkerRegistry.hpp"

namespace EA::Scheduler
{

using InferenceWorkerSelection = SemanticWorkerSelection;

inline InferenceWorkerSelection SelectInferenceWorker(
    const PersistedWorkerSemanticIdentity& persisted,
    const SemanticWorkerRegistry& registry)
{
    return registry.selectInferenceWorker(persisted);
}

} // namespace EA::Scheduler
