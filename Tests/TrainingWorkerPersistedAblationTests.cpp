#include "PersistedModelRuntimeConfig.hpp"

#include <cassert>
#include <string_view>

namespace
{
void ExpectFailure(const auto& action, std::string_view diagnostic)
{
    try { action(); assert(false && "expected mask rejection"); }
    catch (const std::runtime_error& error)
    {
        assert(std::string_view{error.what()}.find(diagnostic) != std::string_view::npos);
    }
}
} // namespace

int main()
{
    using namespace EA::PersistedModelRuntimeConfig;
    const auto mask = EA::FeatureAblationMask::Parse("relative_tick_volume,fibonacci.*");
    const auto canonical = EA::FeatureAblationMask::Parse(mask.CanonicalText());
    const auto reordered = EA::FeatureAblationMask::Parse("fibonacci.*,relative_tick_volume");
    ValidateSchedulerModelFeatureAblationMask({}, {}, 42);
    ValidateSchedulerModelFeatureAblationMask(mask, canonical, 42);
    ValidateSchedulerModelFeatureAblationMask(mask, reordered, 42);
    for (const auto& pair : {std::pair{mask, EA::FeatureAblationMask{}},
                            std::pair{EA::FeatureAblationMask{}, mask}})
        ExpectFailure([&] { ValidateSchedulerModelFeatureAblationMask(
            pair.first, pair.second, 42); }, "FEATURE_ABLATION_MASK_LINEAGE_MISMATCH:model_id=42");

    ResumeCheckpointConfig resume;
    resume.sourceModelId = 42;
    resume.modelInputWidth = EA::kCausalFibonacciStructuralModelInputWidth;
    ValidateSchedulerResumeFeatureAblationMask(resume, {}); // Control continuation.
    resume.featureAblationMask = mask;
    ValidateSchedulerResumeFeatureAblationMask(resume, canonical);
    ExpectFailure([&] { ValidateSchedulerResumeFeatureAblationMask(resume, {}); },
                  "FEATURE_ABLATION_MASK_LINEAGE_MISMATCH");
    const auto widened = EA::FeatureAblationMask::Parse(
        "relative_tick_volume,fibonacci.*,fibonacci_lifecycle.*");
    ExpectFailure([&] { ValidateSchedulerResumeFeatureAblationMask(resume, widened); },
                  "FEATURE_ABLATION_MASK_LINEAGE_MISMATCH");
    resume.expandInputWidthRequested = true;
    ValidateSchedulerResumeFeatureAblationMask(resume, widened);
    ExpectFailure([&] { ValidateSchedulerResumeFeatureAblationMask(resume, {}); },
                  "FEATURE_ABLATION_MASK_EXPANSION_REMOVES_SOURCE_ABLATION");
    resume.featureAblationMask = {};
    ExpectFailure([&] { ValidateSchedulerResumeFeatureAblationMask(resume, mask); },
                  "FEATURE_ABLATION_MASK_EXPANSION_CHANGES_HISTORICAL_FEATURE");
}
