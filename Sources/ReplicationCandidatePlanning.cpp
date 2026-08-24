#include "ReplicationCandidatePlanning.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace EA::ReplicationCandidatePlanning
{
namespace
{

bool TaggedHash(const std::string& value)
{
    if (value.size() != 24 || !value.starts_with("fnv1a64:")) return false;
    return std::all_of(value.begin() + 8, value.end(), [](char character)
    {
        return (character >= '0' && character <= '9') ||
            (character >= 'a' && character <= 'f');
    });
}

bool ValidObjective(const Pair::ObjectiveProvenance& value)
{
    return !value.canonical.empty() && TaggedHash(value.hash) &&
        TrainingObjective::DeterministicHash(value.canonical) == value.hash;
}

bool ValidSymbol(std::string_view value)
{
    if (value.empty()) return false;
    return std::all_of(value.begin(), value.end(), [](char character)
    {
        return (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.';
    });
}

template <typename Value>
bool Allowed(const std::vector<Value>& allowlist, const Value& value)
{
    return allowlist.empty() ||
        std::find(allowlist.begin(), allowlist.end(), value) !=
            allowlist.end();
}

bool HasReason(const Policy::Evaluation& evaluation, std::string_view reason)
{
    return std::find(evaluation.reasons.begin(), evaluation.reasons.end(),
                     reason) != evaluation.reasons.end();
}

void Add(std::vector<std::string>& values, std::string value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(std::move(value));
}

void Add(std::vector<CandidateReason>& values, CandidateReason value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(value);
}

void AddExcluded(std::vector<ExcludedCandidate>& values,
                 std::string identity,
                 ExclusionReason reason)
{
    const ExcludedCandidate candidate{std::move(identity), reason};
    if (std::find(values.begin(), values.end(), candidate) == values.end())
        values.push_back(candidate);
}

int DiversityRank(DiversityClass value)
{
    switch (value)
    {
        case DiversityClass::AddNewSymbolAndHorizon: return 0;
        case DiversityClass::AddNewSymbol: return 1;
        case DiversityClass::AddNewHorizon: return 2;
        case DiversityClass::NoSymbolOrHorizonDiversity: return 3;
    }
    throw std::invalid_argument("unknown_replication_candidate_diversity");
}

CandidateReason DiversityReason(DiversityClass value)
{
    switch (value)
    {
        case DiversityClass::AddNewSymbolAndHorizon:
            return CandidateReason::AddNewSymbolAndHorizon;
        case DiversityClass::AddNewSymbol:
            return CandidateReason::AddNewSymbol;
        case DiversityClass::AddNewHorizon:
            return CandidateReason::AddNewHorizon;
        case DiversityClass::NoSymbolOrHorizonDiversity:
            return CandidateReason::NoSymbolOrHorizonDiversity;
    }
    throw std::invalid_argument("unknown_replication_candidate_diversity");
}

std::string Reasons(const std::vector<std::string>& values)
{
    if (values.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << values[index];
    }
    return output.str();
}

std::string Reasons(const std::vector<CandidateReason>& values)
{
    if (values.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << CandidateReasonText(values[index]);
    }
    return output.str();
}

} // namespace

CandidateUnit MakeCandidateUnit(
    std::string symbol,
    int predictionHorizon,
    std::string replicationUnitCanonical)
{
    CandidateUnit result;
    result.symbol = std::move(symbol);
    result.predictionHorizon = predictionHorizon;
    result.replicationUnitCanonical = std::move(replicationUnitCanonical);
    result.replicationUnitIdentity = TrainingObjective::DeterministicHash(
        result.replicationUnitCanonical);
    return result;
}

std::string ScientificIdentityCanonical(
    const CandidateUnit& candidate,
    const Pair::ObjectiveProvenance& controlObjective,
    const Pair::ObjectiveProvenance& objectiveUnderReview)
{
    std::ostringstream output;
    output << "replication_candidate_unit_v1;"
           << "symbol=" << candidate.symbol << ';'
           << "prediction_horizon=" << candidate.predictionHorizon << ';'
           << "replication_unit_identity="
           << candidate.replicationUnitIdentity << ';'
           << "control_objective_hash=" << controlObjective.hash << ';'
           << "objective_under_review_hash=" << objectiveUnderReview.hash
           << ';';
    return output.str();
}

std::string ScientificIdentity(
    const CandidateUnit& candidate,
    const Pair::ObjectiveProvenance& controlObjective,
    const Pair::ObjectiveProvenance& objectiveUnderReview)
{
    return TrainingObjective::DeterministicHash(ScientificIdentityCanonical(
        candidate, controlObjective, objectiveUnderReview));
}

Plan MakePlan(
    const Policy::Evaluation& evaluation,
    std::vector<CandidateUnit> candidateUniverse,
    std::size_t requestedCandidateLimit,
    CandidateConstraints constraints)
{
    Plan result;
    result.phase4ENextAction = evaluation.nextAction;
    result.objectiveUnderReview = evaluation.objectiveUnderReview;
    result.existingValidReplicationCount = evaluation.validPairCount;
    result.existingDistinctSymbolCount = evaluation.distinctSymbolCount;
    result.existingDistinctHorizonCount = evaluation.distinctHorizonCount;
    result.requestedCandidateLimit = requestedCandidateLimit;

    switch (evaluation.nextAction)
    {
        case Policy::NextAction::WaitForValidResult:
            result.status = PlanStatus::NoActionWait;
            Add(result.reasons, "phase4e_wait_for_valid_result");
            return result;
        case Policy::NextAction::RepairComparability:
            result.status = PlanStatus::NoActionRepair;
            Add(result.reasons, "phase4e_repair_comparability");
            return result;
        case Policy::NextAction::StopObjective:
            result.status = PlanStatus::NoActionStop;
            Add(result.reasons, "phase4e_stop_objective");
            return result;
        case Policy::NextAction::EligibleForCoefficientExploration:
            result.status = PlanStatus::NoActionCoefficientEligible;
            Add(result.reasons,
                "coefficient_exploration_outside_phase4f_scope");
            return result;
        case Policy::NextAction::Replicate:
            result.status = PlanStatus::PlanReplication;
            break;
        case Policy::NextAction::ReplicateBroader:
            result.status = PlanStatus::PlanBroaderReplication;
            break;
    }

    std::set<std::string> representedUnits;
    std::set<std::string> representedSymbols;
    std::set<int> representedHorizons;
    std::set<std::string> controlIdentities;
    for (const Policy::PairSummary& summary : evaluation.pairEvidence)
    {
        if (!summary.replicationUnitIdentity.empty())
            representedUnits.insert(summary.replicationUnitIdentity);
        if (summary.disposition != Pair::Disposition::InvalidComparison &&
            summary.disposition != Pair::Disposition::Incomplete)
        {
            representedSymbols.insert(summary.symbol);
            representedHorizons.insert(summary.horizon);
        }
        if (ValidObjective(summary.controlObjective))
            controlIdentities.insert(summary.controlObjective.hash);
    }

    if (!ValidObjective(evaluation.objectiveUnderReview) ||
        controlIdentities.size() != 1)
    {
        result.status = PlanStatus::InsufficientCandidateDiversity;
        Add(result.reasons, "phase4e_scientific_identity_context_invalid");
        return result;
    }
    for (const Policy::PairSummary& summary : evaluation.pairEvidence)
    {
        if (summary.controlObjective.hash == *controlIdentities.begin())
        {
            result.controlObjective = summary.controlObjective;
            break;
        }
    }

    std::sort(candidateUniverse.begin(), candidateUniverse.end(),
              [](const CandidateUnit& left, const CandidateUnit& right)
              {
                  return std::tie(left.replicationUnitIdentity,
                                  left.replicationUnitCanonical,
                                  left.symbol,
                                  left.predictionHorizon) <
                      std::tie(right.replicationUnitIdentity,
                               right.replicationUnitCanonical,
                               right.symbol,
                               right.predictionHorizon);
              });

    std::vector<PlannedCandidate> qualified;
    for (std::size_t begin = 0; begin < candidateUniverse.size();)
    {
        std::size_t end = begin + 1;
        while (end < candidateUniverse.size() &&
               candidateUniverse[end].replicationUnitIdentity ==
                   candidateUniverse[begin].replicationUnitIdentity)
            ++end;

        bool conflict = false;
        for (std::size_t index = begin + 1; index < end; ++index)
            conflict = conflict ||
                !(candidateUniverse[index] == candidateUniverse[begin]);
        const CandidateUnit& candidate = candidateUniverse[begin];
        if (conflict)
        {
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::CandidateIdentityConflict);
            begin = end;
            continue;
        }
        if (end - begin > 1)
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::DuplicateCandidateIdentity);

        if (!ValidSymbol(candidate.symbol))
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::InvalidSymbol);
        else if (candidate.predictionHorizon <= 0)
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::InvalidHorizon);
        else if (candidate.replicationUnitCanonical.empty())
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::InvalidReplicationUnitCanonical);
        else if (!TaggedHash(candidate.replicationUnitIdentity) ||
                 TrainingObjective::DeterministicHash(
                     candidate.replicationUnitCanonical) !=
                     candidate.replicationUnitIdentity)
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::InvalidReplicationUnitIdentity);
        else if (!Allowed(constraints.allowedSymbols, candidate.symbol))
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::SymbolNotAllowed);
        else if (!Allowed(constraints.allowedHorizons,
                          candidate.predictionHorizon))
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::HorizonNotAllowed);
        else if (representedUnits.contains(candidate.replicationUnitIdentity))
            AddExcluded(result.excludedCandidates,
                        candidate.replicationUnitIdentity,
                        ExclusionReason::DuplicatesExistingReplicationUnit);
        else
        {
            const bool newSymbol =
                !representedSymbols.contains(candidate.symbol);
            const bool newHorizon =
                !representedHorizons.contains(candidate.predictionHorizon);
            DiversityClass diversity =
                DiversityClass::NoSymbolOrHorizonDiversity;
            if (newSymbol && newHorizon)
                diversity = DiversityClass::AddNewSymbolAndHorizon;
            else if (newSymbol)
                diversity = DiversityClass::AddNewSymbol;
            else if (newHorizon)
                diversity = DiversityClass::AddNewHorizon;

            if (evaluation.nextAction ==
                    Policy::NextAction::ReplicateBroader &&
                diversity == DiversityClass::NoSymbolOrHorizonDiversity)
                AddExcluded(result.excludedCandidates,
                            candidate.replicationUnitIdentity,
                            ExclusionReason::DoesNotAddRequiredDiversity);
            else
            {
                PlannedCandidate planned;
                planned.unit = candidate;
                planned.diversityClass = diversity;
                Add(planned.reasons, DiversityReason(diversity));
                if (HasReason(evaluation,
                              "one_not_promising_result_requires_one_diverse_confirmation"))
                    Add(planned.reasons,
                        CandidateReason::ConfirmNegativeOnDiverseUnit);
                if (HasReason(evaluation,
                              "replicate_classification_concern"))
                    Add(planned.reasons,
                        CandidateReason::ReplicateClassificationConcern);
                if (HasReason(evaluation,
                              "replicate_profitability_metric_disagreement"))
                    Add(planned.reasons,
                        CandidateReason::ReplicateProfitabilityDisagreement);
                if (HasReason(evaluation,
                              "replicate_weak_or_ambiguous_materiality"))
                    Add(planned.reasons,
                        CandidateReason::ReplicateWeakMateriality);
                if (planned.reasons.size() == 1)
                    Add(planned.reasons,
                        CandidateReason::ConservativeGenericReplication);
                planned.scientificIdentityCanonical =
                    ScientificIdentityCanonical(candidate,
                        result.controlObjective,
                        result.objectiveUnderReview);
                planned.scientificIdentity = TrainingObjective::DeterministicHash(
                    planned.scientificIdentityCanonical);
                qualified.push_back(std::move(planned));
            }
        }
        begin = end;
    }

    std::sort(qualified.begin(), qualified.end(),
              [](const PlannedCandidate& left,
                 const PlannedCandidate& right)
              {
                  return std::tuple{
                      DiversityRank(left.diversityClass), left.unit.symbol,
                      left.unit.predictionHorizon,
                      left.unit.replicationUnitIdentity} <
                      std::tuple{
                          DiversityRank(right.diversityClass),
                          right.unit.symbol, right.unit.predictionHorizon,
                          right.unit.replicationUnitIdentity};
              });
    std::sort(result.excludedCandidates.begin(),
              result.excludedCandidates.end(),
              [](const ExcludedCandidate& left,
                 const ExcludedCandidate& right)
              {
                  return std::tuple{left.replicationUnitIdentity,
                                    static_cast<int>(left.reason)} <
                      std::tuple{right.replicationUnitIdentity,
                                 static_cast<int>(right.reason)};
              });

    result.requiredDiversityCanBeSatisfied = !qualified.empty();
    if (requestedCandidateLimit == 0)
    {
        result.status = PlanStatus::InsufficientCandidateDiversity;
        Add(result.reasons, "requested_candidate_limit_is_zero");
        return result;
    }
    const std::size_t selectedCount =
        std::min(requestedCandidateLimit, qualified.size());
    result.selectedCandidates.assign(qualified.begin(),
                                     qualified.begin() + selectedCount);
    if (result.selectedCandidates.empty())
    {
        result.status = PlanStatus::InsufficientCandidateDiversity;
        Add(result.reasons, "no_eligible_replication_candidate");
    }
    else if (evaluation.nextAction ==
             Policy::NextAction::ReplicateBroader)
        Add(result.reasons, "broader_replication_candidates_ranked");
    else
        Add(result.reasons, "replication_candidates_ranked");
    return result;
}

std::string PlanStatusText(PlanStatus value)
{
    switch (value)
    {
        case PlanStatus::NoActionWait: return "NO_ACTION_WAIT";
        case PlanStatus::NoActionRepair: return "NO_ACTION_REPAIR";
        case PlanStatus::NoActionStop: return "NO_ACTION_STOP";
        case PlanStatus::NoActionCoefficientEligible:
            return "NO_ACTION_COEFFICIENT_ELIGIBLE";
        case PlanStatus::PlanReplication: return "PLAN_REPLICATION";
        case PlanStatus::PlanBroaderReplication:
            return "PLAN_BROADER_REPLICATION";
        case PlanStatus::InsufficientCandidateDiversity:
            return "INSUFFICIENT_CANDIDATE_DIVERSITY";
    }
    throw std::invalid_argument("unknown_replication_candidate_plan_status");
}

std::string DiversityClassText(DiversityClass value)
{
    switch (value)
    {
        case DiversityClass::AddNewSymbolAndHorizon:
            return "ADD_NEW_SYMBOL_AND_HORIZON";
        case DiversityClass::AddNewSymbol: return "ADD_NEW_SYMBOL";
        case DiversityClass::AddNewHorizon: return "ADD_NEW_HORIZON";
        case DiversityClass::NoSymbolOrHorizonDiversity:
            return "NO_SYMBOL_OR_HORIZON_DIVERSITY";
    }
    throw std::invalid_argument("unknown_replication_candidate_diversity");
}

std::string CandidateReasonText(CandidateReason value)
{
    switch (value)
    {
        case CandidateReason::AddNewSymbolAndHorizon:
            return "ADD_NEW_SYMBOL_AND_HORIZON";
        case CandidateReason::AddNewSymbol: return "ADD_NEW_SYMBOL";
        case CandidateReason::AddNewHorizon: return "ADD_NEW_HORIZON";
        case CandidateReason::NoSymbolOrHorizonDiversity:
            return "NO_SYMBOL_OR_HORIZON_DIVERSITY";
        case CandidateReason::ConfirmNegativeOnDiverseUnit:
            return "CONFIRM_NEGATIVE_ON_DIVERSE_UNIT";
        case CandidateReason::ReplicateClassificationConcern:
            return "REPLICATE_CLASSIFICATION_CONCERN";
        case CandidateReason::ReplicateProfitabilityDisagreement:
            return "REPLICATE_PROFITABILITY_DISAGREEMENT";
        case CandidateReason::ReplicateWeakMateriality:
            return "REPLICATE_WEAK_MATERIALITY";
        case CandidateReason::ConservativeGenericReplication:
            return "CONSERVATIVE_GENERIC_REPLICATION";
    }
    throw std::invalid_argument("unknown_replication_candidate_reason");
}

std::string ExclusionReasonText(ExclusionReason value)
{
    switch (value)
    {
        case ExclusionReason::DuplicatesExistingReplicationUnit:
            return "DUPLICATES_EXISTING_REPLICATION_UNIT";
        case ExclusionReason::DuplicateCandidateIdentity:
            return "DUPLICATE_CANDIDATE_IDENTITY";
        case ExclusionReason::CandidateIdentityConflict:
            return "CANDIDATE_IDENTITY_CONFLICT";
        case ExclusionReason::InvalidSymbol: return "INVALID_SYMBOL";
        case ExclusionReason::InvalidHorizon: return "INVALID_HORIZON";
        case ExclusionReason::InvalidReplicationUnitCanonical:
            return "INVALID_REPLICATION_UNIT_CANONICAL";
        case ExclusionReason::InvalidReplicationUnitIdentity:
            return "INVALID_REPLICATION_UNIT_IDENTITY";
        case ExclusionReason::SymbolNotAllowed: return "SYMBOL_NOT_ALLOWED";
        case ExclusionReason::HorizonNotAllowed: return "HORIZON_NOT_ALLOWED";
        case ExclusionReason::DoesNotAddRequiredDiversity:
            return "DOES_NOT_ADD_REQUIRED_DIVERSITY";
    }
    throw std::invalid_argument("unknown_replication_candidate_exclusion");
}

std::string RenderMachineReadable(const Plan& plan)
{
    std::ostringstream output;
    output << "REPLICATION_CANDIDATE_PLAN,version=1"
           << ",phase4e_next_action="
           << Policy::NextActionText(plan.phase4ENextAction)
           << ",status=" << PlanStatusText(plan.status)
           << ",objective_under_review_hash="
           << (plan.objectiveUnderReview.hash.empty()
                   ? "NONE" : plan.objectiveUnderReview.hash)
           << ",control_objective_hash="
           << (plan.controlObjective.hash.empty()
                   ? "NONE" : plan.controlObjective.hash)
           << ",existing_valid_replication_count="
           << plan.existingValidReplicationCount
           << ",existing_distinct_symbol_count="
           << plan.existingDistinctSymbolCount
           << ",existing_distinct_horizon_count="
           << plan.existingDistinctHorizonCount
           << ",requested_candidate_limit="
           << plan.requestedCandidateLimit
           << ",selected_candidate_count="
           << plan.selectedCandidates.size()
           << ",required_diversity_can_be_satisfied="
           << (plan.requiredDiversityCanBeSatisfied ? "true" : "false")
           << ",reasons=" << Reasons(plan.reasons) << '\n';
    for (std::size_t index = 0; index < plan.selectedCandidates.size();
         ++index)
    {
        const PlannedCandidate& candidate = plan.selectedCandidates[index];
        output << "REPLICATION_CANDIDATE_SELECTED"
               << ",rank=" << index + 1
               << ",symbol=" << candidate.unit.symbol
               << ",prediction_horizon="
               << candidate.unit.predictionHorizon
               << ",replication_unit_identity="
               << candidate.unit.replicationUnitIdentity
               << ",scientific_identity="
               << candidate.scientificIdentity
               << ",diversity_class="
               << DiversityClassText(candidate.diversityClass)
               << ",reasons=" << Reasons(candidate.reasons) << '\n';
    }
    for (const ExcludedCandidate& candidate : plan.excludedCandidates)
        output << "REPLICATION_CANDIDATE_EXCLUDED"
               << ",replication_unit_identity="
               << (candidate.replicationUnitIdentity.empty()
                       ? "NONE" : candidate.replicationUnitIdentity)
               << ",reason=" << ExclusionReasonText(candidate.reason)
               << '\n';
    return output.str();
}

} // namespace EA::ReplicationCandidatePlanning
