#include "ReplicationCandidatePlanning.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <cassert>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace Planning = EA::ReplicationCandidatePlanning;
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

std::string UnitCanonical(const std::string& family,
                          const std::string& symbol,
                          int horizon)
{
    return "synthetic_replication_unit_v1;family=" + family +
        ";symbol=" + symbol + ";horizon=" + std::to_string(horizon) + ";";
}

Policy::PairSummary Summary(
    std::string pairIdentity,
    std::string family,
    std::string symbol,
    int horizon,
    Pair::Disposition disposition,
    std::vector<std::string> interpretationReasons = {})
{
    const Pair::MaterialityPolicy materiality = Materiality();
    Policy::PairSummary result;
    result.pairIdentity = std::move(pairIdentity);
    result.symbol = std::move(symbol);
    result.horizon = horizon;
    result.replicationUnitCanonical = UnitCanonical(
        family, result.symbol, result.horizon);
    result.replicationUnitIdentity = Objective::DeterministicHash(
        result.replicationUnitCanonical);
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
    result.interpretationReasons = std::move(interpretationReasons);
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
        Policy::ConservativeFirstAuxiliaryScreening("USDCAD", 6),
        std::move(summaries));
}

Planning::CandidateUnit Candidate(
    std::string family,
    std::string symbol,
    int horizon)
{
    const std::string canonical = UnitCanonical(family, symbol, horizon);
    return Planning::MakeCandidateUnit(
        std::move(symbol), horizon, canonical);
}

bool Has(const std::vector<Planning::CandidateReason>& values,
         Planning::CandidateReason value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool Has(const std::vector<Planning::ExcludedCandidate>& values,
         const std::string& identity,
         Planning::ExclusionReason reason)
{
    return std::find(values.begin(), values.end(),
                     Planning::ExcludedCandidate{identity, reason}) !=
        values.end();
}

std::vector<Planning::CandidateUnit> BroadUniverse()
{
    return {
        Candidate("eur12", "EURUSD", 12),
        Candidate("aud12", "AUDUSD", 12),
        Candidate("gbp6", "GBPUSD", 6),
        Candidate("usd12", "USDCAD", 12),
        Candidate("usd6-other", "USDCAD", 6)};
}

} // namespace

int main()
{
    const auto existing = Summary(
        "pair-1", "initial", "USDCAD", 6,
        Pair::Disposition::Promising);
    const auto onePromising = Evaluate({existing});
    assert(onePromising.nextAction == Policy::NextAction::ReplicateBroader);

    // 1. WAIT_FOR_VALID_RESULT emits no candidates.
    const auto wait = Planning::MakePlan(Evaluate({}), BroadUniverse(), 3);
    assert(wait.status == Planning::PlanStatus::NoActionWait);
    assert(wait.selectedCandidates.empty());

    // 2 and 18. INVALID_COMPARISON and a Phase 4E materiality mismatch both
    // preserve the authoritative repair/no-action state.
    const auto repair = Planning::MakePlan(Evaluate({Summary(
        "pair-invalid", "invalid", "USDCAD", 6,
        Pair::Disposition::InvalidComparison)}), BroadUniverse(), 3);
    assert(repair.status == Planning::PlanStatus::NoActionRepair);
    assert(repair.selectedCandidates.empty());

    auto mismatched = Summary(
        "pair-2", "second", "EURUSD", 12,
        Pair::Disposition::Promising);
    const auto otherMateriality = Materiality(0.02);
    mismatched.materialityPolicyCanonical =
        Pair::MaterialityPolicyCanonicalText(otherMateriality);
    mismatched.materialityPolicyIdentity =
        Pair::MaterialityPolicyIdentity(otherMateriality);
    const auto mismatchPlan = Planning::MakePlan(
        Evaluate({existing, mismatched}), BroadUniverse(), 3);
    assert(mismatchPlan.status == Planning::PlanStatus::NoActionRepair);

    // 3. Confirmed diverse NOT_PROMISING evidence stops replication.
    const auto stop = Planning::MakePlan(Evaluate({
        Summary("negative-1", "n1", "USDCAD", 6,
                Pair::Disposition::NotPromising,
                {"primary_profitability_materially_worse"}),
        Summary("negative-2", "n2", "EURUSD", 6,
                Pair::Disposition::NotPromising,
                {"primary_profitability_materially_worse"})}),
        BroadUniverse(), 3);
    assert(stop.status == Planning::PlanStatus::NoActionStop);
    assert(stop.selectedCandidates.empty());

    // 4 and 20. Coefficient eligibility emits no Phase 4F candidate and no
    // coefficient recommendation.
    const auto eligibleEvaluation = Evaluate({
        Summary("eligible-1", "e1", "USDCAD", 6,
                Pair::Disposition::Promising),
        Summary("eligible-2", "e2", "EURUSD", 6,
                Pair::Disposition::Promising),
        Summary("eligible-3", "e3", "GBPUSD", 12,
                Pair::Disposition::Promising),
        Summary("eligible-4", "e4", "EURUSD", 12,
                Pair::Disposition::Mixed,
                {"profitability_effect_below_materiality_or_directionally_ambiguous"})});
    assert(eligibleEvaluation.nextAction ==
           Policy::NextAction::EligibleForCoefficientExploration);
    const auto coefficient = Planning::MakePlan(
        eligibleEvaluation, BroadUniverse(), 3);
    assert(coefficient.status ==
           Planning::PlanStatus::NoActionCoefficientEligible);
    assert(coefficient.selectedCandidates.empty());
    assert(Planning::RenderMachineReadable(coefficient).find(
               "COEFFICIENT_CANDIDATE") == std::string::npos);

    // 5. One PROMISING initial unit selects outside that unit, with a new
    // symbol/new horizon candidate first.
    const auto broad = Planning::MakePlan(
        onePromising, BroadUniverse(), 5);
    assert(broad.status == Planning::PlanStatus::PlanBroaderReplication);
    assert(broad.requiredDiversityCanBeSatisfied);
    assert(!broad.selectedCandidates.empty());
    assert(broad.selectedCandidates.front().unit.symbol == "AUDUSD");
    assert(broad.selectedCandidates.front().unit.predictionHorizon == 12);
    assert(broad.selectedCandidates.front().diversityClass ==
           Planning::DiversityClass::AddNewSymbolAndHorizon);

    // 6. Structured MIXED classification concern targets classification
    // replication; no free-form parsing is used.
    const auto mixedClassification = Evaluate({Summary(
        "mixed-classification", "mc", "USDCAD", 6,
        Pair::Disposition::Mixed,
        {"classification_policy_metric_unavailable"})});
    assert(mixedClassification.nextAction == Policy::NextAction::Replicate);
    const auto classificationPlan = Planning::MakePlan(
        mixedClassification, BroadUniverse(), 1);
    assert(classificationPlan.status ==
           Planning::PlanStatus::PlanReplication);
    assert(Has(classificationPlan.selectedCandidates.front().reasons,
               Planning::CandidateReason::ReplicateClassificationConcern));

    // Profitability disagreement and weak materiality use their structured
    // Phase 4E reason codes as deterministic targets.
    const auto disagreementPlan = Planning::MakePlan(Evaluate({Summary(
        "mixed-profit", "mp", "USDCAD", 6,
        Pair::Disposition::Mixed,
        {"profitability_primitives_disagree_or_are_ambiguous"})}),
        BroadUniverse(), 1);
    assert(Has(disagreementPlan.selectedCandidates.front().reasons,
               Planning::CandidateReason::
                   ReplicateProfitabilityDisagreement));
    const auto weakPlan = Planning::MakePlan(Evaluate({Summary(
        "mixed-weak", "mw", "USDCAD", 6,
        Pair::Disposition::Mixed,
        {"profitability_effect_below_materiality_or_directionally_ambiguous"})}),
        BroadUniverse(), 1);
    assert(Has(weakPlan.selectedCandidates.front().reasons,
               Planning::CandidateReason::ReplicateWeakMateriality));

    // 7. One NOT_PROMISING result gets a diverse confirmation reason.
    const auto negativePlan = Planning::MakePlan(Evaluate({Summary(
        "negative", "neg", "USDCAD", 6,
        Pair::Disposition::NotPromising,
        {"primary_profitability_materially_worse"})}),
        BroadUniverse(), 1);
    assert(negativePlan.status ==
           Planning::PlanStatus::PlanBroaderReplication);
    assert(Has(negativePlan.selectedCandidates.front().reasons,
               Planning::CandidateReason::ConfirmNegativeOnDiverseUnit));

    // 8 and 15. An exact deterministic rerun has the Phase 4E unit identity,
    // is excluded, and never becomes independent evidence.
    const auto exactRerun = Candidate("initial", "USDCAD", 6);
    assert(exactRerun.replicationUnitIdentity ==
           existing.replicationUnitIdentity);
    const auto exactPlan = Planning::MakePlan(
        onePromising, {exactRerun, Candidate("eur", "EURUSD", 12)}, 2);
    assert(Has(exactPlan.excludedCandidates,
               exactRerun.replicationUnitIdentity,
               Planning::ExclusionReason::
                   DuplicatesExistingReplicationUnit));
    const auto duplicateEvidence = Evaluate({existing, Summary(
        "pair-exact-rerun", "initial", "USDCAD", 6,
        Pair::Disposition::Promising)});
    assert(duplicateEvidence.validPairCount == 1);

    // 9. Duplicate caller candidates are deterministically deduplicated.
    const auto duplicateCandidate = Candidate("duplicate", "EURUSD", 12);
    const auto duplicatePlan = Planning::MakePlan(
        onePromising, {duplicateCandidate, duplicateCandidate}, 2);
    assert(duplicatePlan.selectedCandidates.size() == 1);
    assert(Has(duplicatePlan.excludedCandidates,
               duplicateCandidate.replicationUnitIdentity,
               Planning::ExclusionReason::DuplicateCandidateIdentity));

    // A reused hash with conflicting metadata is rejected as a whole.
    auto conflict = duplicateCandidate;
    conflict.symbol = "GBPUSD";
    const auto conflictPlan = Planning::MakePlan(
        onePromising, {duplicateCandidate, conflict}, 2);
    assert(conflictPlan.selectedCandidates.empty());
    assert(Has(conflictPlan.excludedCandidates,
               duplicateCandidate.replicationUnitIdentity,
               Planning::ExclusionReason::CandidateIdentityConflict));

    // 10. Both-dimension diversity ranks ahead of either single dimension.
    const auto ranking = Planning::MakePlan(onePromising, {
        Candidate("horizon", "USDCAD", 12),
        Candidate("symbol", "EURUSD", 6),
        Candidate("both", "GBPUSD", 24)}, 3);
    assert(ranking.selectedCandidates[0].unit.symbol == "GBPUSD");
    assert(ranking.selectedCandidates[1].unit.symbol == "EURUSD");
    assert(ranking.selectedCandidates[2].unit.symbol == "USDCAD");

    // 11. New symbols use lexical ordering after diversity class.
    const auto symbolRanking = Planning::MakePlan(onePromising, {
        Candidate("z", "ZARUSD", 6),
        Candidate("a", "AUDUSD", 6)}, 2);
    assert(symbolRanking.selectedCandidates[0].unit.symbol == "AUDUSD");
    assert(symbolRanking.selectedCandidates[1].unit.symbol == "ZARUSD");

    // 12. New horizons use numeric ordering after diversity class.
    const auto horizonRanking = Planning::MakePlan(onePromising, {
        Candidate("h24", "USDCAD", 24),
        Candidate("h12", "USDCAD", 12)}, 2);
    assert(horizonRanking.selectedCandidates[0].unit.predictionHorizon == 12);
    assert(horizonRanking.selectedCandidates[1].unit.predictionHorizon == 24);

    // 13. Broader action with only same-symbol/same-horizon candidates fails
    // explicitly; it does not silently fall back.
    const auto noDiversityCandidate = Candidate(
        "different-config-same-unit", "USDCAD", 6);
    const auto insufficient = Planning::MakePlan(
        onePromising, {noDiversityCandidate}, 1);
    assert(insufficient.status ==
           Planning::PlanStatus::InsufficientCandidateDiversity);
    assert(!insufficient.requiredDiversityCanBeSatisfied);
    assert(Has(insufficient.excludedCandidates,
               noDiversityCandidate.replicationUnitIdentity,
               Planning::ExclusionReason::DoesNotAddRequiredDiversity));

    // 14 and 19. Candidate input order cannot change deterministic plan
    // rendering.
    auto forward = BroadUniverse();
    auto reversed = forward;
    std::reverse(reversed.begin(), reversed.end());
    const std::string forwardRender = Planning::RenderMachineReadable(
        Planning::MakePlan(onePromising, forward, 4));
    const std::string reverseRender = Planning::RenderMachineReadable(
        Planning::MakePlan(onePromising, reversed, 4));
    assert(forwardRender == reverseRender);

    // 16. Candidate limit is honored after deterministic ranking.
    const auto limited = Planning::MakePlan(onePromising, BroadUniverse(), 2);
    assert(limited.selectedCandidates.size() == 2);
    assert(limited.selectedCandidates[0].unit.symbol == "AUDUSD");
    assert(limited.selectedCandidates[1].unit.symbol == "EURUSD");

    // 17. Invalid symbols/horizons and caller allowlist violations are
    // rejected rather than normalized or inferred.
    const auto invalidSymbol = Candidate("bad-symbol", "USD/CAD", 12);
    const auto invalidHorizon = Candidate("bad-horizon", "EURUSD", 0);
    const auto disallowed = Candidate("disallowed", "GBPUSD", 24);
    const auto disallowedHorizon = Candidate(
        "disallowed-horizon", "EURUSD", 24);
    auto emptyCanonical = Candidate("empty-canonical", "EURUSD", 12);
    emptyCanonical.replicationUnitCanonical.clear();
    emptyCanonical.replicationUnitIdentity =
        Objective::DeterministicHash(emptyCanonical.replicationUnitCanonical);
    auto corruptIdentity = Candidate("corrupt-identity", "EURUSD", 12);
    corruptIdentity.replicationUnitIdentity =
        "fnv1a64:0000000000000000";
    Planning::CandidateConstraints constraints;
    constraints.allowedSymbols = {"EURUSD"};
    constraints.allowedHorizons = {12};
    const auto filtered = Planning::MakePlan(onePromising, {
        invalidSymbol, invalidHorizon, disallowed, disallowedHorizon,
        emptyCanonical, corruptIdentity,
        Candidate("allowed", "EURUSD", 12)}, 4, constraints);
    assert(filtered.selectedCandidates.size() == 1);
    assert(filtered.selectedCandidates[0].unit.symbol == "EURUSD");
    assert(Has(filtered.excludedCandidates,
               invalidSymbol.replicationUnitIdentity,
               Planning::ExclusionReason::InvalidSymbol));
    assert(Has(filtered.excludedCandidates,
               invalidHorizon.replicationUnitIdentity,
               Planning::ExclusionReason::InvalidHorizon));
    assert(Has(filtered.excludedCandidates,
               disallowed.replicationUnitIdentity,
               Planning::ExclusionReason::SymbolNotAllowed));
    assert(Has(filtered.excludedCandidates,
               disallowedHorizon.replicationUnitIdentity,
               Planning::ExclusionReason::HorizonNotAllowed));
    assert(Has(filtered.excludedCandidates,
               emptyCanonical.replicationUnitIdentity,
               Planning::ExclusionReason::
                   InvalidReplicationUnitCanonical));
    assert(Has(filtered.excludedCandidates,
               corruptIdentity.replicationUnitIdentity,
               Planning::ExclusionReason::
                   InvalidReplicationUnitIdentity));

    // Exact canonical/hash determinism, including symbol, horizon, objective
    // pair, and objective-neutral Phase 4E unit identity.
    const auto identityCandidate = Candidate("identity", "EURUSD", 12);
    const auto control = Provenance(Objective::Legacy());
    const auto treatment = Provenance(Objective::ProfitabilityAuxiliary());
    assert(identityCandidate.replicationUnitCanonical ==
           "synthetic_replication_unit_v1;family=identity;symbol=EURUSD;"
           "horizon=12;");
    assert(identityCandidate.replicationUnitIdentity ==
           "fnv1a64:3d42c343ccf2955f");
    const std::string expectedScientificCanonical =
        "replication_candidate_unit_v1;symbol=EURUSD;prediction_horizon=12;"
        "replication_unit_identity=" +
        identityCandidate.replicationUnitIdentity +
        ";control_objective_hash=fnv1a64:65818f2e1fa1a324;"
        "objective_under_review_hash=fnv1a64:f7a9a20f7f72eee5;";
    assert(Planning::ScientificIdentityCanonical(
               identityCandidate, control, treatment) ==
           expectedScientificCanonical);
    assert(Planning::ScientificIdentity(
               identityCandidate, control, treatment) ==
           Objective::DeterministicHash(expectedScientificCanonical));
    assert(Planning::ScientificIdentity(
               identityCandidate, control, treatment) ==
           "fnv1a64:4c8a6a9b97ea2b92");

    // 21-23 are structural: CandidateUnit and MakePlan accept no database,
    // experiment ID, scheduler, or lifecycle state. This test executable is
    // linked only from pure Phase 4D/4E/4F sources.
    std::cout << "replication_candidate_planning_tests_passed\n";
    std::cout << "scientific_identity="
              << Planning::ScientificIdentity(
                     identityCandidate, control, treatment) << '\n';
    return 0;
}
