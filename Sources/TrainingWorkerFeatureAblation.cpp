#include "TrainingWorkerFeatureAblation.hpp"

#include <algorithm>
#include <stdexcept>
#include <vector>

#include "ModelInputContract.hpp"

namespace EA::Training
{

void ValidateExpandedResumeAblationComposition(
    const FeatureAblationMask& sourceMask,
    const FeatureAblationMask& requestedMask,
    std::size_t sourceInputWidth,
    std::size_t tensorFeatureCount)
{
    const auto sourceContract = ContractForModelInputWidth(sourceInputWidth);
    const auto contains = [](const std::vector<std::size_t>& values,
                             std::size_t value)
    {
        return std::find(values.begin(), values.end(), value) != values.end();
    };
    for (const std::size_t sourceColumn : sourceMask.tensorColumns())
    {
        if (!contains(requestedMask.tensorColumns(), sourceColumn))
            throw std::runtime_error(
                "FEATURE_ABLATION_MASK_EXPANSION_REMOVES_SOURCE_ABLATION");
    }
    for (const std::size_t requestedColumn : requestedMask.tensorColumns())
    {
        if (!contains(sourceMask.tensorColumns(), requestedColumn) &&
            requestedColumn < sourceContract.tensorFeatureCount)
        {
            throw std::runtime_error(
                "FEATURE_ABLATION_MASK_EXPANSION_CHANGES_HISTORICAL_FEATURE");
        }
    }
    requestedMask.ValidateForTensorFeatureCount(tensorFeatureCount);
}

} // namespace EA::Training
