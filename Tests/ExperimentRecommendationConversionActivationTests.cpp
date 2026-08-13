#include "../Sources/ExperimentRecommendationConversionActivation.hpp"

#include <cassert>
#include <stdexcept>
#include <string>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationConversionActivationIdentityInput Input()
{
    return {11, 22, 33, 44, "fnv1a64:0123456789abcdef"};
}

template <typename Function>
bool ThrowsInvalidArgument(Function&& function)
{
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        return true;
    }
    return false;
}

} // namespace

int main()
{
    const auto expected = BuildRecommendationConversionActivationIdentity(
        Input());
    assert(expected.canonicalText ==
           "experiment_recommendation_conversion_activation_v1;"
           "contract_version=1;execution_id=11;proposal_id=22;"
           "review_decision_id=33;experiment_id=44;previous_status=paused;"
           "previous_phase=train;resulting_status=pending;"
           "resulting_phase=train;execution_identity_hash="
           "24:fnv1a64:0123456789abcdef");
    assert(expected.hash ==
           BuildRecommendationConversionActivationIdentity(Input()).hash);

    auto changed = Input();
    ++changed.executionId;
    assert(BuildRecommendationConversionActivationIdentity(changed).hash !=
           expected.hash);
    changed = Input();
    ++changed.proposalId;
    assert(BuildRecommendationConversionActivationIdentity(changed).hash !=
           expected.hash);
    changed = Input();
    ++changed.reviewDecisionId;
    assert(BuildRecommendationConversionActivationIdentity(changed).hash !=
           expected.hash);
    changed = Input();
    ++changed.experimentId;
    assert(BuildRecommendationConversionActivationIdentity(changed).hash !=
           expected.hash);
    changed = Input();
    changed.executionIdentityHash = "fnv1a64:fedcba9876543210";
    assert(BuildRecommendationConversionActivationIdentity(changed).hash !=
           expected.hash);

    for (int field = 0; field < 4; ++field)
    {
        auto invalid = Input();
        if (field == 0) invalid.executionId = 0;
        if (field == 1) invalid.proposalId = 0;
        if (field == 2) invalid.reviewDecisionId = 0;
        if (field == 3) invalid.experimentId = 0;
        assert(ThrowsInvalidArgument([&] {
            (void)BuildRecommendationConversionActivationIdentity(invalid);
        }));
    }
    auto invalidText = Input();
    invalidText.executionIdentityHash.clear();
    assert(ThrowsInvalidArgument([&] {
        (void)BuildRecommendationConversionActivationIdentity(invalidText);
    }));
    invalidText = Input();
    invalidText.executionIdentityHash = std::string{"bad\0hash", 8};
    assert(ThrowsInvalidArgument([&] {
        (void)BuildRecommendationConversionActivationIdentity(invalidText);
    }));
    invalidText.executionIdentityHash.assign(257, 'x');
    assert(ThrowsInvalidArgument([&] {
        (void)BuildRecommendationConversionActivationIdentity(invalidText);
    }));
}
