#pragma once

#include "SemanticWorkerRegistry.hpp"

namespace EA::Scheduler
{

using TrainingWorkerSelection = SemanticWorkerSelection;

inline TrainingWorkerSelection SelectTrainingWorker(
    const PersistedWorkerSemanticIdentity& persisted,
    const SemanticWorkerRegistry& registry)
{
    return registry.selectTrainingReferenceWorker(persisted);
}

} // namespace EA::Scheduler
