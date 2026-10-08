#include <cassert>
#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

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
    catch (const std::exception&)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    // Exercise the materialization hook used by the dedicated TRAIN LSTM,
    // including persisted canonical text, controls and every supported channel.
    const auto contract = EA::ResolveModelInputContract(
        EA::kCurrentModelInputWidth, feature_size);
    std::vector<float> source(feature_size);
    for (std::size_t i = 0; i < source.size(); ++i)
        source[i] = static_cast<float>(i + 1);
    const auto original = source;
    const auto project = [&](const EA::FeatureAblationMask& mask)
    {
        std::vector<float> output(EA::kCurrentModelInputWidth, -7.0f);
        EA::CopyTensorFeaturesForModelInput(
            output.data(), source.data(), contract, mask);
        assert(source == original);
        for (std::size_t i = 0; i < contract.tensorFeatureCount; ++i)
            assert(output[i] == (std::find(mask.tensorColumns().begin(),
                mask.tensorColumns().end(), i) != mask.tensorColumns().end()
                    ? 0.0f : source[i]));
        for (std::size_t i = contract.tensorFeatureCount; i < output.size(); ++i)
            assert(output[i] == -7.0f); // Appended returns remain untouched.
        return output;
    };
    assert(project({}) == project(EA::FeatureAblationMask::Parse("")));
    for (const auto& feature : EA::kAblatableFeatures)
    {
        const auto fresh = EA::FeatureAblationMask::Parse(std::string{feature.name});
        const auto persisted = EA::FeatureAblationMask::Parse(fresh.CanonicalText());
        assert(project(fresh) == project(persisted));
    }
    for (const auto& feature : {EA::kDirectionalEfficiencyFeature,
                               EA::kDirectionalAdverseExcursionFeature})
        (void)project(EA::FeatureAblationMask::Parse(std::string{feature.name}));
    const auto mixed = EA::FeatureAblationMask::Parse(
        " relative_tick_volume,relative_tick_volume,fibonacci.* ");
    assert(project(mixed) == project(EA::FeatureAblationMask::Parse(mixed.CanonicalText())));
    for (const char* invalid : {"unknown_feature", "fibonacci*", "fibonacci.*,",
                                "fibonacci.unknown.*"})
        assert(Throws([&] { (void)EA::FeatureAblationMask::Parse(invalid); }));
    assert(Throws([&]
    {
        const auto old = EA::ResolveModelInputContract(
            EA::kLegacyModelInputWidth, source.size());
        std::vector<float> output(EA::kLegacyModelInputWidth);
        EA::CopyTensorFeaturesForModelInput(output.data(), source.data(), old,
            EA::FeatureAblationMask::Parse("relative_tick_volume"));
    }));

    const EA::FeatureAblationMask historical =
        EA::FeatureAblationMask::Parse("fibonacci.*");
    const EA::FeatureAblationMask widened =
        EA::FeatureAblationMask::Parse(
            "fibonacci.*,fibonacci_lifecycle.*");

    EA::Training::ValidateExpandedResumeAblationComposition(
        historical, widened, EA::kCausalFibonacciStructuralModelInputWidth,
        EA::kCausalFibonacciLifecycleModelInputWidth -
            EA::kModelReturnFeatureCount);

    // A control continuation stays a control; same-mask continuation preserves
    // historical treatment. Only newly appended features may add treatment.
    EA::Training::ValidateExpandedResumeAblationComposition(
        {}, {}, EA::kCausalFibonacciStructuralModelInputWidth,
        contract.tensorFeatureCount);
    EA::Training::ValidateExpandedResumeAblationComposition(
        historical, historical, EA::kCausalFibonacciStructuralModelInputWidth,
        contract.tensorFeatureCount);
    assert(Throws([&]
    {
        EA::Training::ValidateExpandedResumeAblationComposition(
            {}, historical, EA::kCausalFibonacciStructuralModelInputWidth,
            contract.tensorFeatureCount);
    }));
    assert(Throws([&]
    {
        EA::Training::ValidateExpandedResumeAblationComposition(
            historical, widened, EA::kCausalFibonacciStructuralModelInputWidth,
            EA::ContractForModelInputWidth(
                EA::kCausalFibonacciStructuralModelInputWidth).tensorFeatureCount);
    }));

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
