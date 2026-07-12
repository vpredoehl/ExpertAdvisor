#include <cassert>
#include <climits>
#include <iostream>

#include "../Sources/ContinuationPolicyInheritance.hpp"

using EA::ExperimentScheduler::DeriveContinuationChildPolicyTarget;
using EA::ExperimentScheduler::StableContinuationPolicyHash;

int main()
{
    const auto derived = DeriveContinuationChildPolicyTarget(130, 4);
    assert(derived.error.empty());
    assert(derived.targetEpochs == 134);

    const auto repeated = DeriveContinuationChildPolicyTarget(130, 4);
    assert(repeated.targetEpochs == derived.targetEpochs);
    assert(repeated.error == derived.error);

    const auto missingIncrement =
        DeriveContinuationChildPolicyTarget(130, std::nullopt);
    assert(!missingIncrement.targetEpochs.has_value());
    assert(missingIncrement.error == "inherit_to_child_requires_target_increment");

    const auto missingTarget =
        DeriveContinuationChildPolicyTarget(std::nullopt, 4);
    assert(!missingTarget.targetEpochs.has_value());
    assert(missingTarget.error == "inherit_to_child_requires_target_epochs");

    const auto overflow = DeriveContinuationChildPolicyTarget(INT_MAX - 1, 4);
    assert(!overflow.targetEpochs.has_value());
    assert(overflow.error == "inherited_target_epochs_overflow");

    const std::string parentCanonical =
        "target_epochs=130|min_evals=1|inherit_to_child=true|target_increment=4";
    const std::string childCanonical =
        "target_epochs=134|min_evals=1|inherit_to_child=true|target_increment=4";
    const std::string childHash = StableContinuationPolicyHash(childCanonical);
    assert(childHash == StableContinuationPolicyHash(childCanonical));
    assert(childHash != StableContinuationPolicyHash(parentCanonical));

    std::cout << "ContinuationPolicyInheritanceTests passed\n";
    return 0;
}
