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

} // namespace EA::ExperimentScheduler
