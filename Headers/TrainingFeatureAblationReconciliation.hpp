#pragma once

#include "FeatureAblation.hpp"

#include <optional>
#include <stdexcept>

namespace EA
{

inline FeatureAblationMask ReconcileSchedulerTrainFeatureAblationMask(
    const std::optional<FeatureAblationMask>& cliMask,
    const FeatureAblationMask& persistedMask,
    const std::optional<FeatureAblationMask>& ordinaryResumeSourceMask =
        std::nullopt)
{
    if (persistedMask.empty())
    {
        if (cliMask.has_value())
            throw std::runtime_error(
                "TRAIN_FEATURE_ABLATION_CLI_UNEXPECTED_FOR_CONTROL");
    }
    else
    {
        if (!cliMask.has_value())
            throw std::runtime_error("TRAIN_FEATURE_ABLATION_CLI_REQUIRED");
        if (cliMask->CanonicalText() != persistedMask.CanonicalText())
            throw std::runtime_error(
                "TRAIN_FEATURE_ABLATION_CLI_PERSISTED_MISMATCH");
    }

    if (ordinaryResumeSourceMask.has_value() &&
        ordinaryResumeSourceMask->CanonicalText() !=
            persistedMask.CanonicalText())
    {
        throw std::runtime_error(
            "TRAIN_FEATURE_ABLATION_RESUME_LINEAGE_MISMATCH");
    }
    return persistedMask;
}

} // namespace EA
