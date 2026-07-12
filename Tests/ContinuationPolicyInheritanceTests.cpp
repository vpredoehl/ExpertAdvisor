#include <cassert>
#include <climits>
#include <iostream>

#include "../Sources/ContinuationPolicyInheritance.hpp"

using EA::ExperimentScheduler::DeriveContinuationChildPolicyTarget;
using EA::ExperimentScheduler::DeriveBoundedContinuationChildPolicy;
using EA::ExperimentScheduler::StableContinuationPolicyHash;
using EA::ExperimentScheduler::ContinuationPolicyIdentityMaterial;
using EA::ExperimentScheduler::SemanticContinuationPolicyHash;

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

    const std::string semanticPolicy =
        "target_epochs=138|min_evals=1|patience=2|min_infer_accuracy=0"
        "|source_mode=final_model|candidate_excluded=true|inherit_to_child=true"
        "|target_increment=4|max_target_epochs=142";
    const ContinuationPolicyIdentityMaterial provenanceA{
        semanticPolicy, true, 184, "valid"};
    const ContinuationPolicyIdentityMaterial provenanceB{
        semanticPolicy, true, 999, "max_target_reached"};
    const ContinuationPolicyIdentityMaterial noProvenance{
        semanticPolicy, false, std::nullopt, "not_requested"};
    assert(SemanticContinuationPolicyHash(provenanceA) ==
           SemanticContinuationPolicyHash(provenanceB));
    assert(SemanticContinuationPolicyHash(provenanceA) ==
           SemanticContinuationPolicyHash(noProvenance));

    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|target_epochs=142"));
    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|target_increment=8"));
    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|max_target_epochs=146"));
    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|min_infer_accuracy=0.5"));
    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|source_mode=best_checkpoint"));
    assert(StableContinuationPolicyHash(semanticPolicy) !=
           StableContinuationPolicyHash(semanticPolicy + "|candidate_excluded=false"));

    const auto child134 = DeriveBoundedContinuationChildPolicy(134, 4, 142);
    assert(!child134.terminal && child134.inheritedPolicyTargetEpochs == 138);
    const auto child138 = DeriveBoundedContinuationChildPolicy(138, 4, 142);
    assert(!child138.terminal && child138.inheritedPolicyTargetEpochs == 142);
    const auto child142 = DeriveBoundedContinuationChildPolicy(142, 4, 142);
    assert(child142.terminal && !child142.inheritedPolicyTargetEpochs.has_value());
    const auto child146 = DeriveBoundedContinuationChildPolicy(146, 4, 142);
    assert(child146.error == "policy_target_exceeds_max_target");
    const auto skippedMaximum = DeriveBoundedContinuationChildPolicy(140, 4, 142);
    assert(skippedMaximum.error == "target_increment_would_skip_past_max_target");

    std::cout << "ContinuationPolicyInheritanceTests passed\n";
    return 0;
}
