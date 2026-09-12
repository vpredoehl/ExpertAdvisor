#pragma once

#include "../../Headers/ModelInputExpansion.hpp"

#include <cstddef>
#include <optional>
#include <span>
#include <string>

namespace EA::Scheduler
{

struct WorkerSemanticCapability
{
    int layoutVersion = EA::kModelInputSemanticLayoutVersion;
    std::size_t maximumInputWidth = EA::kCurrentModelInputWidth;
    std::span<const EA::ModelInputSemanticLayoutRegistryEntry> registry =
        EA::kModelInputSemanticLayoutRegistry;
    std::span<const std::size_t> registeredInputWidths =
        EA::kRegisteredModelInputWidths;
};

struct PersistedWorkerSemanticIdentity
{
    std::optional<std::size_t> inputWidth;
    std::optional<int> layoutVersion;
    bool modelIdentityExpected = false;
};

struct SemanticAdmissionDecision
{
    bool admissible = false;
    std::string diagnostic;
};

inline SemanticAdmissionDecision EvaluateLegacyMarkerlessModelAdmission(
    std::size_t inputWidth,
    const WorkerSemanticCapability& worker = {})
{
    std::size_t matches = 0;
    for (const std::size_t registered : worker.registeredInputWidths)
        if (registered == inputWidth) ++matches;

    if (matches != 1 || inputWidth > worker.maximumInputWidth)
        return {false, "legacy_model_registered_width_incompatible"};

    return {true, "legacy_model_registered_width_compatible"};
}

inline SemanticAdmissionDecision EvaluateSemanticWorkerAdmission(
    const std::string& phase,
    const PersistedWorkerSemanticIdentity& persisted,
    const WorkerSemanticCapability& worker = {})
{
    // Analysis consumes already persisted evidence and does not execute or
    // reconstruct model-input features in the current scheduler path.
    if (phase == "analyze") return {true, "analyze_not_model_bearing"};
    if (phase != "train" && phase != "infer")
        return {false, "unsupported_model_bearing_phase"};
    if (!persisted.inputWidth && !persisted.layoutVersion &&
        phase == "train" && !persisted.modelIdentityExpected)
        return {true, "legacy_identity_unavailable"};
    if (!persisted.inputWidth && !persisted.layoutVersion)
        return {false, "semantic_worker_identity_unavailable"};
    if (!persisted.inputWidth || !persisted.layoutVersion)
        return {false, "semantic_worker_identity_incomplete"};
    if (!EA::IsModelInputSemanticLayoutWidthCompatible(
            *persisted.layoutVersion, *persisted.inputWidth,
            worker.registry, worker.registeredInputWidths,
            worker.layoutVersion, worker.maximumInputWidth, false))
        return {false, "semantic_worker_incompatible"};
    return {true, "semantic_worker_compatible"};
}

} // namespace EA::Scheduler
