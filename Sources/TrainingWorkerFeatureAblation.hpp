#pragma once

#include <cstddef>

#include "FeatureAblation.hpp"

namespace EA::Training
{

// Preserve persisted feature-ablation decisions when a resumed model is
// widened to a newer model-input contract.  This deliberately receives the
// live tensor width from its caller rather than reading executable globals.
void ValidateExpandedResumeAblationComposition(
    const FeatureAblationMask& sourceMask,
    const FeatureAblationMask& requestedMask,
    std::size_t sourceInputWidth,
    std::size_t tensorFeatureCount);

} // namespace EA::Training
