#include "PostPairReplicationPolicy.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <string>
#include <vector>

namespace Policy = EA::PostPairReplicationPolicy;
namespace Pair = EA::PairedTrainingObjectiveEvaluation;
namespace Objective = EA::TrainingObjective;

namespace
{

Pair::ObjectiveProvenance Provenance(
    const Objective::Configuration& configuration)
{
    const std::string canonical = Objective::CanonicalText(configuration);
    return {canonical, Objective::DeterministicHash(canonical)};
}

Pair::MaterialityPolicy Materiality(double improvement = 0.01)
{
    Pair::MaterialityPolicy result;
    result.minimumProfitabilityImprovement = improvement;
    result.maximumProfitabilityWorsening = 0.01;
    result.classification = Pair::ClassificationDegradationPolicy{
        0.02, 0.02, 0.05, 0.02, 0.05};
    return result;
}

Pair::MetricDelta Delta(double control, double treatment)
{
    return {control, treatment, treatment - control,
            control == 0.0
                ? std::optional<double>{}
                : std::optional<double>{
                      (treatment - control) / std::abs(control)}};
}

Policy::PairSummary Summary(
    std::string pairIdentity,
    std::string unitIdentity,
    std::string symbol,
    int horizon,
    Pair::Disposition disposition,
    std::vector<std::string> reasons = {})
{
    const Pair::MaterialityPolicy materiality = Materiality();
    Policy::PairSummary result;
    result.pairIdentity = std::move(pairIdentity);
    result.replicationUnitCanonical = std::move(unitIdentity);
    result.replicationUnitIdentity = Objective::DeterministicHash(
        result.replicationUnitCanonical);
    result.symbol = std::move(symbol);
    result.horizon = horizon;
    result.controlObjective = Provenance(Objective::Legacy());
    result.treatmentObjective =
        Provenance(Objective::ProfitabilityAuxiliary());
    result.materialityPolicyCanonical =
        Pair::MaterialityPolicyCanonicalText(materiality);
    result.materialityPolicyIdentity =
        Pair::MaterialityPolicyIdentity(materiality);
    result.primaryProfitability = Delta(0.10, 0.12);
    result.aggregateProfitability = result.primaryProfitability;
    result.averageProfitability = Delta(0.001, 0.0012);
    result.inferenceAccuracy = Delta(0.70, 0.70);
    result.acceptAccuracy = Delta(0.75, 0.75);
    result.acceptRate = Delta(0.60, 0.60);
    result.predictedNeutralProportion = Delta(0.40, 0.40);
    result.leaderScore = Delta(0.65, 0.65);
    result.disposition = disposition;
    result.interpretationReasons = std::move(reasons);
    if (disposition == Pair::Disposition::InvalidComparison)
        result.invalidReasons = {"synthetic_invalid"};
    if (disposition == Pair::Disposition::Incomplete)
        result.incompleteReasons = {"synthetic_incomplete"};
    return result;
}

Policy::Evaluation Evaluate(std::vector<Policy::PairSummary> summaries)
{
    return Policy::Evaluate(
        Provenance(Objective::ProfitabilityAuxiliary()),
        Policy::ConservativeFirstAuxiliaryScreening("usdcadrmp", 6),
        std::move(summaries));
}

bool Has(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

} // namespace

int main()
{
    // Materiality identity contains the primary metric and every Phase 4D
    // threshold in a stable, human-readable canonical form.
    const Pair::MaterialityPolicy materiality = Materiality();
    const std::string expectedCanonical =
        "paired_objective_materiality_policy_v1;"
        "primary_profitability_metric=aggregate_terminal_horizon_log_return_sum;"
        "minimum_profitability_improvement=0.01;"
        "maximum_profitability_worsening=0.01;"
        "classification_policy=configured;"
        "maximum_inference_accuracy_decrease=0.02;"
        "maximum_accept_accuracy_decrease=0.02;"
        "maximum_accept_rate_decrease=0.05;"
        "maximum_leader_score_decrease=0.02;"
        "maximum_neutral_proportion_increase=0.05;";
    assert(Pair::MaterialityPolicyCanonicalText(materiality) ==
           expectedCanonical);
    assert(Pair::MaterialityPolicyIdentity(materiality) ==
           "fnv1a64:de03318ec969e2ad");

    // 1. Zero valid pairs.
    const auto empty = Evaluate({});
    assert(empty.validPairCount == 0);
    assert(empty.nextAction == Policy::NextAction::WaitForValidResult);

    // 2. INCOMPLETE does not count.
    const auto incomplete = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::Incomplete)});
    assert(incomplete.validPairCount == 0);
    assert(incomplete.incompleteCount == 1);
    assert(incomplete.nextAction == Policy::NextAction::WaitForValidResult);

    // 3. INVALID_COMPARISON requires repair and does not count.
    const auto invalid = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6,
        Pair::Disposition::InvalidComparison)});
    assert(invalid.validPairCount == 0);
    assert(invalid.invalidCount == 1);
    assert(invalid.nextAction == Policy::NextAction::RepairComparability);

    // 4. One positive result only justifies broader replication.
    const auto onePromising = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising)});
    assert(onePromising.promisingCount == 1);
    assert(onePromising.nextAction == Policy::NextAction::ReplicateBroader);

    // 5. One negative result receives one diverse confirmation, not an
    // immediate stop.
    const auto oneNegative = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::NotPromising,
        {"primary_profitability_materially_worse"})});
    assert(oneNegative.notPromisingCount == 1);
    assert(oneNegative.nextAction == Policy::NextAction::ReplicateBroader);

    // 6. MIXED is targeted according to the Phase 4D interpretation reason.
    const auto oneMixed = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::Mixed,
        {"profitability_primitives_disagree_or_are_ambiguous"})});
    assert(oneMixed.mixedCount == 1);
    assert(oneMixed.nextAction == Policy::NextAction::Replicate);
    assert(Has(oneMixed.reasons,
               "replicate_profitability_metric_disagreement"));

    // 7. Multiple independent positive configurations at one symbol/horizon
    // still cannot satisfy geographic/horizon diversity.
    const auto sameSymbolHorizon = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p3", "u3", "usdcadrmp", 6, Pair::Disposition::Promising)});
    assert(sameSymbolHorizon.validPairCount == 3);
    assert(!sameSymbolHorizon.replicationDiversitySatisfied);
    assert(sameSymbolHorizon.nextAction ==
           Policy::NextAction::ReplicateBroader);

    // Exact deterministic reruns share a replication unit and count once.
    const auto sameUnit = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u1", "usdcadrmp", 6, Pair::Disposition::Promising)});
    assert(sameUnit.validPairCount == 1);
    assert(Has(sameUnit.reasons,
               "non_independent_replication_unit_ignored"));

    // 8 and 9. Symbol diversity alone and horizon diversity alone are each
    // insufficient, even with four PROMISING independent units.
    const auto symbolsOnly = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "eurusdrmp", 6, Pair::Disposition::Promising),
        Summary("p3", "u3", "gbpusdrmp", 6, Pair::Disposition::Promising),
        Summary("p4", "u4", "audusdrmp", 6, Pair::Disposition::Promising)});
    assert(symbolsOnly.distinctSymbolCount == 4);
    assert(symbolsOnly.distinctHorizonCount == 1);
    assert(symbolsOnly.nextAction == Policy::NextAction::ReplicateBroader);

    const auto horizonsOnly = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "usdcadrmp", 12, Pair::Disposition::Promising),
        Summary("p3", "u3", "usdcadrmp", 24, Pair::Disposition::Promising),
        Summary("p4", "u4", "usdcadrmp", 48, Pair::Disposition::Promising)});
    assert(horizonsOnly.distinctSymbolCount == 1);
    assert(horizonsOnly.distinctHorizonCount == 4);
    assert(horizonsOnly.nextAction == Policy::NextAction::ReplicateBroader);

    // The minimum satisfying program combines three symbols, two horizons,
    // and a 3/4 promising supermajority.
    const std::vector<Policy::PairSummary> sufficient{
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "eurusdrmp", 6, Pair::Disposition::Promising),
        Summary("p3", "u3", "gbpusdrmp", 12, Pair::Disposition::Promising),
        Summary("p4", "u4", "eurusdrmp", 12, Pair::Disposition::Mixed,
                {"profitability_effect_below_materiality_or_directionally_ambiguous"})};
    const auto eligible = Evaluate(sufficient);
    assert(eligible.validPairCount == 4);
    assert(eligible.promisingCount == 3);
    assert(eligible.mixedCount == 1);
    assert(eligible.distinctSymbolCount == 3);
    assert(eligible.distinctHorizonCount == 2);
    assert(eligible.replicationDiversitySatisfied);
    assert(eligible.nextAction ==
           Policy::NextAction::EligibleForCoefficientExploration);

    // 10. A positive/negative conflict is unresolved and goes broader.
    const auto conflicting = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "eurusdrmp", 6,
                Pair::Disposition::NotPromising,
                {"primary_profitability_materially_worse"})});
    assert(conflicting.promisingCount == 1);
    assert(conflicting.notPromisingCount == 1);
    assert(conflicting.nextAction == Policy::NextAction::ReplicateBroader);

    // 11. MIXED-heavy evidence is ineligible.
    const auto mixedHeavy = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        Summary("p2", "u2", "eurusdrmp", 6, Pair::Disposition::Mixed),
        Summary("p3", "u3", "gbpusdrmp", 12, Pair::Disposition::Mixed),
        Summary("p4", "u4", "eurusdrmp", 12, Pair::Disposition::Mixed)});
    assert(mixedHeavy.nextAction !=
           Policy::NextAction::EligibleForCoefficientExploration);

    // 12. Materiality policies are fail-closed aggregation strata.
    auto policyMismatch = Summary(
        "p2", "u2", "eurusdrmp", 12, Pair::Disposition::Promising);
    const auto otherMateriality = Materiality(0.02);
    policyMismatch.materialityPolicyCanonical =
        Pair::MaterialityPolicyCanonicalText(otherMateriality);
    policyMismatch.materialityPolicyIdentity =
        Pair::MaterialityPolicyIdentity(otherMateriality);
    const auto mismatched = Evaluate({Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising),
        policyMismatch});
    assert(mismatched.nextAction ==
           Policy::NextAction::RepairComparability);
    assert(Has(mismatched.reasons, "materiality_policy_identity_mismatch"));

    // 13. Exact duplicate pair identity is retained/counts only once.
    const auto duplicate = Summary(
        "p1", "u1", "usdcadrmp", 6, Pair::Disposition::Promising);
    const auto duplicated = Evaluate({duplicate, duplicate});
    assert(duplicated.validPairCount == 1);
    assert(duplicated.pairEvidence.size() == 1);
    assert(Has(duplicated.reasons, "duplicate_pair_identity_ignored"));

    // 14. Sorting makes both the decision and complete rendered output input
    // order independent.
    auto reversed = sufficient;
    std::reverse(reversed.begin(), reversed.end());
    assert(Policy::RenderMachineReadable(Evaluate(sufficient)) ==
           Policy::RenderMachineReadable(Evaluate(reversed)));

    // 15. Eligibility is impossible from one pair even under a malformed
    // attempted threshold relaxation: policy validation fails closed.
    auto unsafePolicy =
        Policy::ConservativeFirstAuxiliaryScreening("usdcadrmp", 6);
    unsafePolicy.minimumValidReplications = 1;
    unsafePolicy.minimumPromisingReplications = 1;
    const auto unsafe = Policy::Evaluate(
        Provenance(Objective::ProfitabilityAuxiliary()), unsafePolicy,
        {Summary("p1", "u1", "usdcadrmp", 6,
                 Pair::Disposition::Promising)});
    assert(unsafe.nextAction == Policy::NextAction::RepairComparability);

    // 16. Invalid and incomplete rows never contribute positive evidence.
    const auto nonEvidence = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6,
                Pair::Disposition::InvalidComparison),
        Summary("p2", "u2", "eurusdrmp", 12,
                Pair::Disposition::Incomplete),
        Summary("p3", "u3", "gbpusdrmp", 12,
                Pair::Disposition::Promising)});
    assert(nonEvidence.validPairCount == 1);
    assert(nonEvidence.promisingCount == 1);
    assert(nonEvidence.invalidCount == 1);
    assert(nonEvidence.incompleteCount == 1);
    assert(nonEvidence.nextAction ==
           Policy::NextAction::RepairComparability);

    // Two independently negative, diverse units stop the objective.
    const auto confirmedNegative = Evaluate({
        Summary("p1", "u1", "usdcadrmp", 6,
                Pair::Disposition::NotPromising,
                {"primary_profitability_materially_worse"}),
        Summary("p2", "u2", "eurusdrmp", 6,
                Pair::Disposition::NotPromising,
                {"primary_profitability_materially_worse"})});
    assert(confirmedNegative.nextAction == Policy::NextAction::StopObjective);

    std::cout << "post_pair_replication_policy_tests_passed\n";
    return 0;
}
