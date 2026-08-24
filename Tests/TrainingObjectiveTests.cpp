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
    assert(ParseCliSelection("legacy") == legacy);
    assert(ParseCliSelection(kLegacyObjectiveIdentifier) == legacy);
    ExpectFailureContaining(
        [] { (void)ParseCliSelection("unknown_v99"); },
        "unsupported_training_objective_selection");
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
           "legacy_classification_only_contract_required");

    const Configuration auxiliary = ProfitabilityAuxiliary();
    assert(ParseCliSelection("profitability_auxiliary_v1") == auxiliary);
    assert(ParseCliSelection(kAuxiliaryObjectiveIdentifier) == auxiliary);
    assert(!Validate(auxiliary));
    assert(AuxiliaryEnabled(auxiliary));
    assert(auxiliary.mode ==
           Mode::FirstHitClassificationWithTerminalReturnAuxiliary);
    assert(auxiliary.auxiliaryLossCoefficient == 0.1);
    assert(auxiliary.robustLossDelta == 1.0);
    assert(auxiliary.targetClippingDefinition == "none");

    // Terminal targets use the same prediction-time and terminal-horizon
    // closes as the classification window identity, with fixed x1000 scaling.
    assert(NearlyEqual(TerminalHorizonLogReturn(100.0, 110.0),
                       std::log(1.1)));
    assert(AuxiliaryRegressionTarget(100.0, 110.0) > 0.0);
    assert(AuxiliaryRegressionTarget(100.0, 90.0) < 0.0);
    assert(AuxiliaryRegressionTarget(100.0, 100.0) == 0.0);
    assert(NearlyEqual(AuxiliaryRegressionTarget(100.0, 110.0),
                       1000.0 * std::log(1.1)));

    // Huber uses 0.5*r^2 inside delta and delta*(|r|-0.5*delta)
    // outside, with a signed clipped residual derivative.
    assert(NearlyEqual(HuberLoss(0.5), 0.125));
    assert(NearlyEqual(HuberLoss(2.0), 1.5));
    assert(NearlyEqual(HuberLoss(-2.0), 1.5));
    assert(NearlyEqual(HuberGradient(0.5), 0.5));
    assert(NearlyEqual(HuberGradient(2.0), 1.0));
    assert(NearlyEqual(HuberGradient(-2.0), -1.0));
    assert(NearlyEqual(WeightedAuxiliaryLoss(2.0, 0.0, 0.2), 0.3));
    assert(NearlyEqual(WeightedAuxiliaryOutputGradient(2.0, 0.0, 0.2),
                       0.2));

    // Disabled/zero auxiliary is exactly the classification loss and shared
    // gradient. Enabling it adds only the scalar-head projection to d_h.
    const double classificationLoss = -std::log(0.7);
    assert(CombinedExampleLoss(classificationLoss, 5.0, -5.0, legacy) ==
           classificationLoss);
    assert(NearlyEqual(
        CombinedExampleLoss(classificationLoss, 2.0, 0.0, auxiliary),
        classificationLoss + 0.15));
    const std::array<double, 2> classificationProjection {0.25, -0.5};
    const std::array<double, 2> auxiliaryWeight {2.0, 3.0};
    const auto legacyShared = CombineSharedCoreHeadGradients(
        classificationProjection, auxiliaryWeight, 0.1, legacy);
    assert((legacyShared == std::array<double, 2>{1.0, -2.0}));
    const auto combinedShared = CombineSharedCoreHeadGradients(
        classificationProjection, auxiliaryWeight, 0.1, auxiliary);
    assert(NearlyEqual(combinedShared[0], 1.8));
    assert(NearlyEqual(combinedShared[1], -0.8));
    assert(LegacyClassificationLogitGradient(probabilities, 2, auxiliary) ==
           gradient);

    // Exact objective identity round-trip and bidirectional resume rejection.
    const std::string auxiliaryCanonical = CanonicalText(auxiliary);
    const std::string auxiliaryIdentity = Identity(auxiliary);
    assert(auxiliaryIdentity == "fnv1a64:f7a9a20f7f72eee5");
    assert(ResolvePersisted(auxiliaryCanonical, auxiliaryIdentity) == auxiliary);
    assert(!ResumeCompatible(legacy, auxiliary));
    assert(!ResumeCompatible(auxiliary, legacy));
    ExpectFailureContaining(
        [&] { RequireResumeCompatible(legacy, auxiliary); },
        "training_objective_resume_incompatible");
    ExpectFailureContaining(
        [&] { RequireResumeCompatible(auxiliary, legacy); },
        "training_objective_resume_incompatible");

    std::string unknownAuxiliaryCanonical = auxiliaryCanonical;
    const std::string knownRobust =
        "robust_loss_definition=huber_half_squared_inside_linear_outside_v1";
    const std::size_t robustOffset =
        unknownAuxiliaryCanonical.find(knownRobust);
    assert(robustOffset != std::string::npos);
    unknownAuxiliaryCanonical.replace(
        robustOffset, knownRobust.size(),
        "robust_loss_definition=unknown_auxiliary_loss_v99");
    ExpectFailureContaining(
        [&] {
            (void)ResolvePersisted(
                unknownAuxiliaryCanonical,
                DeterministicHash(unknownAuxiliaryCanonical));
        },
        "unsupported_training_objective_canonical_configuration");

    std::cout << "LEGACY_TRAINING_OBJECTIVE_CANONICAL=" << canonical << '\n'
              << "LEGACY_TRAINING_OBJECTIVE_HASH=" << identity << '\n'
              << "AUXILIARY_TRAINING_OBJECTIVE_CANONICAL="
              << auxiliaryCanonical << '\n'
              << "AUXILIARY_TRAINING_OBJECTIVE_HASH="
              << auxiliaryIdentity << '\n'
              << "TrainingObjectiveTests passed\n";
    return 0;
}
