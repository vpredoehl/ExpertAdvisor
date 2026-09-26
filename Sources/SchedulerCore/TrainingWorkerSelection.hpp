#pragma once

#include "SemanticWorkerRegistry.hpp"

namespace EA::Scheduler
{

using TrainingWorkerSelection = SemanticWorkerSelection;

inline TrainingWorkerSelection SelectTrainingWorker(
    const PersistedWorkerSemanticIdentity& persisted,
    const SemanticWorkerRegistry& registry,
    const SemanticWorkerCapabilities& requiredCapabilities = {})
{
    return registry.selectTrainingReferenceWorker(
        persisted, requiredCapabilities);
}

inline SemanticWorkerCapabilities RequiredTrainingWorkerCapabilities(
    const std::string& canonicalFeatureAblationMask)
{
    if (canonicalFeatureAblationMask.empty()) return {};
    return {kTrainFeatureAblationCapability};
}

} // namespace EA::Scheduler
