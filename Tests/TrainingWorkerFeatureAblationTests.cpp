#include <cassert>
#include <stdexcept>

#include "ModelInputContract.hpp"
#include "TrainingWorkerFeatureAblation.hpp"

namespace
{

bool Throws(const auto& callback)
{
    try
    {
        callback();
    }
    catch (const std::runtime_error&)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    const EA::FeatureAblationMask historical =
        EA::FeatureAblationMask::Parse("fibonacci.*");
    const EA::FeatureAblationMask widened =
        EA::FeatureAblationMask::Parse(
            "fibonacci.*,fibonacci_lifecycle.*");

    EA::Training::ValidateExpandedResumeAblationComposition(
        historical, widened, EA::kCausalFibonacciStructuralModelInputWidth,
        EA::kCausalFibonacciLifecycleModelInputWidth -
            EA::kModelReturnFeatureCount);

    assert(Throws([&]
    {
        EA::Training::ValidateExpandedResumeAblationComposition(
            historical, EA::FeatureAblationMask{},
            EA::kCausalFibonacciStructuralModelInputWidth,
            EA::kCausalFibonacciLifecycleModelInputWidth -
                EA::kModelReturnFeatureCount);
    }));

    return 0;
}
