#pragma once

#include <limits>
#include <optional>
#include <cstdint>
#include <iomanip>
#include <sstream>
#include <string>

namespace EA::ExperimentScheduler
{

struct ContinuationTargetDerivation
{
    std::optional<int> targetEpochs;
    std::string error;
};

struct BoundedContinuationTargetDerivation
{
    bool terminal = false;
    std::optional<int> inheritedPolicyTargetEpochs;
    std::string error;
};

inline ContinuationTargetDerivation DeriveContinuationChildPolicyTarget(
    const std::optional<int>& childTargetEpochs,
    const std::optional<int>& targetIncrement)
{
    if (!childTargetEpochs.has_value())
        return {std::nullopt, "inherit_to_child_requires_target_epochs"};
    if (!targetIncrement.has_value())
        return {std::nullopt, "inherit_to_child_requires_target_increment"};
    if (*targetIncrement <= 0)
        return {std::nullopt, "target_increment_must_be_positive"};
    if (*childTargetEpochs > std::numeric_limits<int>::max() - *targetIncrement)
        return {std::nullopt, "inherited_target_epochs_overflow"};
    const int derived = *childTargetEpochs + *targetIncrement;
    if (derived <= *childTargetEpochs)
        return {std::nullopt, "inherited_target_must_exceed_child_target"};
    return {derived, {}};
}

inline BoundedContinuationTargetDerivation DeriveBoundedContinuationChildPolicy(
    int childTrainingTargetEpochs,
    const std::optional<int>& targetIncrement,
    const std::optional<int>& maxTargetEpochs)
{
    if (maxTargetEpochs.has_value())
    {
        if (childTrainingTargetEpochs > *maxTargetEpochs)
            return {false, std::nullopt, "policy_target_exceeds_max_target"};
        if (childTrainingTargetEpochs == *maxTargetEpochs)
            return {true, std::nullopt, {}};
    }
    const ContinuationTargetDerivation next =
        DeriveContinuationChildPolicyTarget(
            childTrainingTargetEpochs,
            targetIncrement);
    if (!next.error.empty())
        return {false, std::nullopt, next.error};
    if (maxTargetEpochs.has_value() &&
        *next.targetEpochs > *maxTargetEpochs)
    {
        return {false, std::nullopt, "target_increment_would_skip_past_max_target"};
    }
    return {false, next.targetEpochs, {}};
}

inline std::string StableContinuationPolicyHash(const std::string& canonicalPolicy)
{
    uint64_t hash = UINT64_C(14695981039346656037);
    for (const unsigned char ch : canonicalPolicy)
    {
        hash ^= static_cast<uint64_t>(ch);
        hash *= UINT64_C(1099511628211);
    }
    std::ostringstream out;
    out << std::hex << std::setw(16) << std::setfill('0') << hash;
    return out.str();
}

struct ContinuationPolicyIdentityMaterial
{
    std::string semanticConfiguration;
    bool inherited = false;
    std::optional<long long> inheritedFromExperimentId;
    std::string inheritanceStatus = "not_requested";
};

inline std::string SemanticContinuationPolicyHash(
    const ContinuationPolicyIdentityMaterial& material)
{
    return StableContinuationPolicyHash(material.semanticConfiguration);
}

} // namespace EA::ExperimentScheduler
