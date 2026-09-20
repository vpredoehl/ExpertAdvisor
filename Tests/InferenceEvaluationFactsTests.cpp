#include "../Sources/InferenceEvaluationFacts.hpp"

#include <cassert>

int main()
{
    using EA::InferenceEvaluationFacts::ComputeAcceptanceSummary;
    using EA::InferenceEvaluationFacts::ConfusionMatrix;

    ConfusionMatrix accepted {{
        {{20, 2, 3}},
        {{2, 20, 2}},
        {{3, 2, 20}},
    }};
    const auto acceptedFacts = ComputeAcceptanceSummary(accepted);
    assert(acceptedFacts.acceptModel);
    assert(acceptedFacts.rejectReason == "none");
    assert(acceptedFacts.predFrac[0] > 0.15);
    assert(acceptedFacts.predFrac[2] > 0.15);

    ConfusionMatrix neutralOnly {{
        {{0, 1, 0}},
        {{0, 8, 0}},
        {{0, 1, 0}},
    }};
    const auto rejectedFacts = ComputeAcceptanceSummary(neutralOnly);
    assert(!rejectedFacts.acceptModel);
    assert(rejectedFacts.rejectReason ==
           "pred_neutral_gt_0.60;pred_down_lt_0.15;pred_up_lt_0.15");
    assert(rejectedFacts.predFrac[1] == 1.0);
    return 0;
}
