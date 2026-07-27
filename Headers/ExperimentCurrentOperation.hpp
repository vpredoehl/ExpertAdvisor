#pragma once

#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>

namespace EA::ExperimentLifecycle
{

inline constexpr std::string_view kTrainOperation = "train";
inline constexpr std::string_view kInferOperation = "infer";
inline constexpr std::string_view kAnalyzeOperation = "analyze";

inline bool IsCanonicalCurrentOperation(std::string_view operation)
{
    return operation == kTrainOperation ||
           operation == kInferOperation ||
           operation == kAnalyzeOperation;
}

inline std::optional<std::string> CanonicalCurrentOperationValue(
    std::string_view operation)
{
    if (operation == kTrainOperation || operation == "training")
        return std::string{kTrainOperation};
    if (operation == kInferOperation || operation == "inference")
        return std::string{kInferOperation};
    if (operation == kAnalyzeOperation || operation == "analysis")
        return std::string{kAnalyzeOperation};
    return std::nullopt;
}

inline std::optional<std::string> CurrentOperationForPhase(
    std::string_view phase)
{
    if (phase == kTrainOperation)
        return std::string{kTrainOperation};
    if (phase == kInferOperation)
        return std::string{kInferOperation};
    if (phase == kAnalyzeOperation)
        return std::string{kAnalyzeOperation};
    return std::nullopt;
}

inline std::string NormalizePersistedCurrentOperation(
    std::string_view operation)
{
    if (const auto canonical = CanonicalCurrentOperationValue(operation))
        return *canonical;
    return std::string{operation};
}

inline std::optional<std::string> NormalizeOptionalPersistedCurrentOperation(
    const std::optional<std::string>& operation)
{
    if (!operation)
        return std::nullopt;
    return NormalizePersistedCurrentOperation(*operation);
}

inline std::string RequireCanonicalCurrentOperationForPhase(
    std::string_view phase)
{
    const auto operation = CurrentOperationForPhase(phase);
    if (!operation || phase == "done")
        throw std::invalid_argument(
            "current_operation_requires_active_canonical_phase");
    return *operation;
}

inline std::string CurrentOperationForStatus(
    const std::optional<std::string>& operation,
    std::string_view phase)
{
    if (operation)
    {
        if (const auto canonical =
                CanonicalCurrentOperationValue(*operation))
            return *canonical;
        return "invalid(" + *operation + ")";
    }
    if (const auto phaseOperation = CurrentOperationForPhase(phase))
        return *phaseOperation;
    return "unknown";
}

} // namespace EA::ExperimentLifecycle
