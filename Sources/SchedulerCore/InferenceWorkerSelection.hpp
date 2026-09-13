#pragma once

#include "SchedulerSemanticAdmission.hpp"

#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <optional>
#include <stdexcept>
#include <string>

#include <sys/stat.h>
#include <unistd.h>

namespace EA::Scheduler
{

inline constexpr int kLegacyInferenceSemanticLayoutVersion = 6;

struct InferenceWorkerRoutingConfiguration
{
    std::string currentCanonicalExecutablePath;
    std::optional<std::string> legacyLayout6CanonicalExecutablePath;
};

struct InferenceWorkerSelection
{
    bool selected = false;
    std::string diagnostic;
    std::string canonicalExecutablePath;
    int semanticLayoutVersion = 0;
    std::size_t maximumInputWidth = 0;
    std::string reason;
};

inline std::string ValidateAndCanonicalizeWorkerExecutable(
    const std::string& configuredPath,
    const std::string& optionName)
{
    if (configuredPath.empty() || configuredPath.front() != '/')
        throw std::invalid_argument(
            optionName + " requires an absolute executable path");

    errno = 0;
    char* resolved = ::realpath(configuredPath.c_str(), nullptr);
    if (resolved == nullptr)
        throw std::invalid_argument(
            optionName + " path cannot be canonicalized: " +
            std::string{std::strerror(errno)});
    std::string canonicalPath{resolved};
    std::free(resolved);

    struct stat status {};
    if (::stat(canonicalPath.c_str(), &status) != 0 ||
        !S_ISREG(status.st_mode) ||
        ::access(canonicalPath.c_str(), X_OK) != 0)
    {
        throw std::invalid_argument(
            optionName + " requires an existing executable regular file");
    }
    return canonicalPath;
}

inline WorkerSemanticCapability LegacyLayout6InferenceCapability()
{
    WorkerSemanticCapability capability;
    capability.layoutVersion = kLegacyInferenceSemanticLayoutVersion;
    capability.maximumInputWidth = EA::kCausalEconomicEventSurpriseModelInputWidth;
    capability.registry = EA::kModelInputSemanticLayoutRegistry;
    capability.registeredInputWidths = EA::kRegisteredModelInputWidths;
    return capability;
}

inline InferenceWorkerSelection SelectInferenceWorker(
    const PersistedWorkerSemanticIdentity& persisted,
    const InferenceWorkerRoutingConfiguration& configuration)
{
    if (!persisted.inputWidth && !persisted.layoutVersion)
        return {false, "semantic_worker_identity_unavailable", {}, 0, 0, {}};
    if (!persisted.inputWidth || !persisted.layoutVersion)
        return {false, "semantic_worker_identity_incomplete", {}, 0, 0, {}};

    WorkerSemanticCapability capability;
    std::string executable;
    std::string reason;
    if (*persisted.layoutVersion == EA::kModelInputSemanticLayoutVersion)
    {
        executable = configuration.currentCanonicalExecutablePath;
        reason = "current_semantic_layout";
    }
    else if (*persisted.layoutVersion ==
             kLegacyInferenceSemanticLayoutVersion)
    {
        if (!configuration.legacyLayout6CanonicalExecutablePath)
            return {false,
                    "semantic_worker_unavailable_for_layout=6",
                    {},
                    kLegacyInferenceSemanticLayoutVersion,
                    EA::kCausalEconomicEventSurpriseModelInputWidth,
                    {}};
        capability = LegacyLayout6InferenceCapability();
        executable = *configuration.legacyLayout6CanonicalExecutablePath;
        reason = "legacy_semantic_layout";
    }
    else
    {
        return {false, "semantic_worker_incompatible", {}, 0, 0, {}};
    }

    // Phase 20N routes only exact full-width layout-6 and layout-7 models.
    // Append-only training expansion compatibility must not become inference
    // compatibility across semantic workers.
    if (*persisted.inputWidth != capability.maximumInputWidth)
        return {false,
                "semantic_worker_incompatible",
                {},
                capability.layoutVersion,
                capability.maximumInputWidth,
                {}};

    const SemanticAdmissionDecision admission =
        EvaluateSemanticWorkerAdmission("infer", persisted, capability);
    if (!admission.admissible)
        return {false, admission.diagnostic, {}, 0, 0, {}};
    if (executable.empty() || executable.front() != '/')
        return {false, "semantic_worker_executable_invalid", {}, 0, 0, {}};

    return {true,
            "semantic_worker_compatible",
            std::move(executable),
            capability.layoutVersion,
            capability.maximumInputWidth,
            std::move(reason)};
}

} // namespace EA::Scheduler
