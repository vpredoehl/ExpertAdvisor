#include <array>
#include <cassert>
#include <cmath>
#include <functional>
#include <iostream>
#include <string>
#include <vector>

#include "TrainingObjective.hpp"

using namespace EA::TrainingObjective;

namespace
{

template <typename Callable>
void ExpectFailureContaining(Callable&& callable, const std::string& marker)
{
    try
    {
        callable();
        assert(false && "expected failure");
    }
    catch (const std::exception& error)
    {
        assert(std::string{error.what()}.find(marker) != std::string::npos);
    }
}

bool NearlyEqual(double left, double right, double tolerance = 1.0e-14)
{
    return std::fabs(left - right) <= tolerance;
}

} // namespace

int main()
{
    const Configuration legacy = Legacy();
    assert(!Validate(legacy));
    const std::string canonical = CanonicalText(legacy);
    const std::string identity = Identity(legacy);
    assert(canonical == CanonicalText(Legacy()));
    assert(identity == Identity(Legacy()));
    assert(identity.starts_with("fnv1a64:"));
    assert(identity == "fnv1a64:65818f2e1fa1a324");

    // The canonical legacy contract freezes the values reconstructed from the
    // active CalculateBatch implementation.
    assert(legacy.mode == Mode::LegacyFirstHitClassification);
    assert(legacy.objectiveIdentifier == kLegacyObjectiveIdentifier);
    assert(legacy.classificationTarget ==
           "up_neutral_down_return_high_low_first_hit_strict_threshold_up_tie_v1");
    assert((legacy.classWeights == std::array<double, 3>{1.0, 1.0, 1.0}));
    assert(legacy.classificationLogitGradientScale == 0.1);
    assert(legacy.sharedCoreClassificationGradientScale == 4.0);
    assert(legacy.optimizer == OptimizerFamily::Sgd);
    assert(legacy.gradientClipping ==
           GradientClippingMode::ComponentwiseAfterNormalization);
    assert(legacy.gradientClipThreshold == 10.0);
    assert(legacy.auxiliaryLossMode == AuxiliaryLossMode::Disabled);
    assert(legacy.auxiliaryLossCoefficient == 0.0);
    assert(!legacy.regressionTargetDefinition);
    assert(!legacy.robustLossDefinition);
    assert(!legacy.robustLossDelta);

    Configuration changed = legacy;
    changed.classificationLogitGradientScale = 0.2;
    assert(!Validate(changed));
    assert(CanonicalText(changed) != canonical);
    assert(Identity(changed) != identity);
    assert(!ResumeCompatible(legacy, changed));
    ExpectFailureContaining(
        [&] { RequireResumeCompatible(legacy, changed); },
        "training_objective_resume_incompatible");
    RequireResumeCompatible(legacy, legacy);

    // Model metadata round-trip uses this exact canonical/hash pair. Missing
    // metadata deterministically resolves marker-less historical models to
    // the legacy objective; partial or corrupted metadata fails closed.
    const Configuration roundTripped = ResolvePersisted(canonical, identity);
    assert(roundTripped == legacy);
    assert(Identity(roundTripped) == identity);
    assert(ResolvePersisted(std::nullopt, std::nullopt) == legacy);
    ExpectFailureContaining(
        [&] { (void)ResolvePersisted(canonical, std::nullopt); },
        "incomplete_training_objective_provenance");
    ExpectFailureContaining(
        [&] { (void)ResolvePersisted(canonical, "fnv1a64:0000000000000000"); },
        "training_objective_provenance_hash_mismatch");

    // Freeze the active weighted-CE and dlogit definitions.
    const std::array<double, 3> probabilities {0.2, 0.3, 0.5};
    const auto gradient = LegacyClassificationLogitGradient(
        probabilities, 2, legacy);
    assert(NearlyEqual(gradient[0], 0.02));
    assert(NearlyEqual(gradient[1], 0.03));
    assert(NearlyEqual(gradient[2], -0.05));

    const std::vector<std::array<double, 3>> batchProbabilities {
        {0.8, 0.1, 0.1},
        {0.1, 0.2, 0.7},
    };
    const std::vector<int> trueClasses {0, 2};
    const auto unitSummary = SummarizeLegacyClassificationBatch(
        batchProbabilities, trueClasses, legacy);
    assert(NearlyEqual(unitSummary.InternallyNormalizedLoss(),
                       unitSummary.CalculateBatchReturnValue()));

    // Freeze the existing naming/normalization mismatch without changing it:
    // internal/logged loss divides by weight sum, while CalculateBatch's
    // returned scalar divides by mseCount/example count.
    Configuration nonunit = legacy;
    nonunit.classWeights = {2.0, 1.0, 4.0};
    const auto weightedSummary = SummarizeLegacyClassificationBatch(
        batchProbabilities, trueClasses, nonunit);
    const double expectedWeightedLoss =
        2.0 * -std::log(0.8) + 4.0 * -std::log(0.7);
    assert(NearlyEqual(weightedSummary.weightedLossSum,
                       expectedWeightedLoss));
    assert(NearlyEqual(weightedSummary.InternallyNormalizedLoss(),
                       expectedWeightedLoss / 6.0));
    assert(NearlyEqual(weightedSummary.CalculateBatchReturnValue(),
                       expectedWeightedLoss / 2.0));
    assert(!NearlyEqual(weightedSummary.InternallyNormalizedLoss(),
                        weightedSummary.CalculateBatchReturnValue()));

    Configuration invalidAuxiliary = legacy;
    invalidAuxiliary.auxiliaryLossCoefficient = 0.1;
    assert(Validate(invalidAuxiliary) ==
           "phase4a_auxiliary_objective_must_be_disabled");

    std::cout << "LEGACY_TRAINING_OBJECTIVE_CANONICAL=" << canonical << '\n'
              << "LEGACY_TRAINING_OBJECTIVE_HASH=" << identity << '\n'
              << "TrainingObjectiveTests passed\n";
    return 0;
}
