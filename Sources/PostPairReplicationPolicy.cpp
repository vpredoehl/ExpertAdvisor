#include "PostPairReplicationPolicy.hpp"

#include "TrainingObjective.hpp"

#include <algorithm>
#include <cmath>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string_view>

namespace EA::PostPairReplicationPolicy
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
    if (value.canonical.empty() || !TaggedHash(value.hash) ||
        TrainingObjective::DeterministicHash(value.canonical) != value.hash)
        return false;
    try
    {
        return TrainingObjective::CanonicalText(
            TrainingObjective::ParseSupportedCanonicalText(value.canonical)) ==
            value.canonical;
    }
    catch (const std::exception&)
    {
        return false;
    }
}

bool ValidPolicy(const Policy& value)
{
    return value.minimumValidReplications > 1 &&
        value.minimumPromisingReplications > 1 &&
        value.minimumPromisingReplications <=
            value.minimumValidReplications &&
        std::isfinite(value.minimumPromisingFraction) &&
        value.minimumPromisingFraction > 0.5 &&
        value.minimumPromisingFraction <= 1.0 &&
        std::isfinite(value.maximumMixedFraction) &&
        value.maximumMixedFraction >= 0.0 &&
        value.maximumMixedFraction < 0.5 &&
        value.minimumDistinctSymbols > 1 &&
        value.minimumDistinctHorizons > 1 &&
        value.confirmatoryNotPromisingReplications > 1 &&
        value.maximumValidReplicationsBeforeStop >=
            value.minimumValidReplications &&
        (!value.requireReplicationOutsideInitialUnit ||
         (!value.initialSymbol.empty() && value.initialHorizon > 0));
}

void Add(std::vector<std::string>& values, std::string value)
{
    if (std::find(values.begin(), values.end(), value) == values.end())
        values.push_back(std::move(value));
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? TrainingObjective::CanonicalDouble(*value) : "NULL";
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

bool HasReason(const PairSummary& summary, std::string_view reason)
{
    return std::find(summary.interpretationReasons.begin(),
                     summary.interpretationReasons.end(), reason) !=
        summary.interpretationReasons.end();
}

std::string PrimaryMetricText(Pair::ProfitabilityPrimaryMetric value)
{
    switch (value)
    {
        case Pair::ProfitabilityPrimaryMetric::
                AggregateTerminalHorizonLogReturnSum:
            return "aggregate_terminal_horizon_log_return_sum";
        case Pair::ProfitabilityPrimaryMetric::
                AverageTerminalHorizonLogReturnPerActionablePrediction:
            return "average_terminal_horizon_log_return_per_actionable_prediction";
    }
    throw std::invalid_argument("unknown_primary_profitability_metric");
}

void RenderMetric(std::ostringstream& output,
                  std::string_view name,
                  const Pair::MetricDelta& metric)
{
    output << ',' << name << "_control=" << OptionalNumber(metric.control)
           << ',' << name << "_treatment="
           << OptionalNumber(metric.treatment)
           << ',' << name << "_delta="
           << OptionalNumber(metric.treatmentMinusControl);
}

} // namespace

Policy ConservativeFirstAuxiliaryScreening(
    std::string initialSymbol,
    int initialHorizon)
{
    Policy result;
    result.initialSymbol = std::move(initialSymbol);
    result.initialHorizon = initialHorizon;
    return result;
}

std::string PolicyCanonicalText(const Policy& policy)
{
    if (!ValidPolicy(policy))
        throw std::invalid_argument("replication_policy_invalid");
    std::string result = "post_pair_replication_policy_v1;";
    const auto append = [&result](std::string_view name,
                                  std::string_view value)
    {
        result.append(name);
        result.push_back('=');
        result.append(value);
        result.push_back(';');
    };
    append("minimum_valid_replications",
           std::to_string(policy.minimumValidReplications));
    append("minimum_promising_replications",
           std::to_string(policy.minimumPromisingReplications));
    append("minimum_promising_fraction",
           TrainingObjective::CanonicalDouble(
               policy.minimumPromisingFraction));
    append("maximum_mixed_fraction",
           TrainingObjective::CanonicalDouble(policy.maximumMixedFraction));
    append("maximum_not_promising_for_eligibility",
           std::to_string(policy.maximumNotPromisingForEligibility));
    append("minimum_distinct_symbols",
           std::to_string(policy.minimumDistinctSymbols));
    append("minimum_distinct_horizons",
           std::to_string(policy.minimumDistinctHorizons));
    append("confirmatory_not_promising_replications",
           std::to_string(policy.confirmatoryNotPromisingReplications));
    append("maximum_valid_replications_before_stop",
           std::to_string(policy.maximumValidReplicationsBeforeStop));
    append("initial_symbol", policy.initialSymbol);
    append("initial_horizon", std::to_string(policy.initialHorizon));
    append("require_replication_outside_initial_unit",
           policy.requireReplicationOutsideInitialUnit ? "true" : "false");
    return result;
}

std::string PolicyIdentity(const Policy& policy)
{
    return TrainingObjective::DeterministicHash(PolicyCanonicalText(policy));
}

PairSummary Summarize(
    std::string pairIdentity,
    std::string replicationUnitCanonical,
    const Pair::ArmEvidence& control,
    const Pair::ArmEvidence& treatment,
    const Pair::MaterialityPolicy& materialityPolicy,
    const Pair::ComparisonResult& comparison)
{
    PairSummary result;
    result.pairIdentity = std::move(pairIdentity);
    result.replicationUnitCanonical = std::move(replicationUnitCanonical);
    result.replicationUnitIdentity = TrainingObjective::DeterministicHash(
        result.replicationUnitCanonical);
    result.symbol = control.configuration.symbol;
    result.horizon = control.configuration.predictionHorizon;
    result.controlObjective = control.configuration.experimentObjective;
    result.treatmentObjective = treatment.configuration.experimentObjective;
    result.primaryProfitabilityMetric =
        materialityPolicy.primaryProfitabilityMetric;
    result.materialityPolicyCanonical =
        Pair::MaterialityPolicyCanonicalText(materialityPolicy);
    result.materialityPolicyIdentity =
        Pair::MaterialityPolicyIdentity(materialityPolicy);
    result.primaryProfitability =
        materialityPolicy.primaryProfitabilityMetric ==
                Pair::ProfitabilityPrimaryMetric::
                    AggregateTerminalHorizonLogReturnSum
            ? comparison.aggregateProfitability
            : comparison.averageProfitability;
    result.aggregateProfitability = comparison.aggregateProfitability;
    result.averageProfitability = comparison.averageProfitability;
    result.inferenceAccuracy = comparison.inferenceAccuracy;
    result.acceptAccuracy = comparison.acceptAccuracy;
    result.acceptRate = comparison.acceptRate;
    result.predictedNeutralProportion =
        comparison.predictedNeutralProportion;
    result.leaderScore = comparison.leaderScore;
    result.disposition = comparison.disposition;
    result.invalidReasons = comparison.invalidReasons;
    result.incompleteReasons = comparison.incompleteReasons;
    result.interpretationReasons = comparison.interpretationReasons;
    return result;
}

Evaluation Evaluate(
    const Pair::ObjectiveProvenance& objectiveUnderReview,
    const Policy& policy,
    std::vector<PairSummary> comparisons)
{
    Evaluation result;
    result.objectiveUnderReview = objectiveUnderReview;

    if (!ValidObjective(objectiveUnderReview))
        Add(result.reasons, "objective_under_review_invalid");
    if (!ValidPolicy(policy))
        Add(result.reasons, "replication_policy_invalid");
    else
        result.replicationPolicyIdentity = PolicyIdentity(policy);

    std::sort(comparisons.begin(), comparisons.end(),
              [](const PairSummary& left, const PairSummary& right)
              {
                  return left.pairIdentity < right.pairIdentity;
              });

    for (std::size_t begin = 0; begin < comparisons.size();)
    {
        std::size_t end = begin + 1;
        while (end < comparisons.size() &&
               comparisons[end].pairIdentity == comparisons[begin].pairIdentity)
            ++end;
        bool conflict = false;
        for (std::size_t index = begin + 1; index < end; ++index)
        {
            if (!(comparisons[begin] == comparisons[index]))
            {
                conflict = true;
                break;
            }
        }
        if (conflict)
            Add(result.reasons, "duplicate_pair_identity_conflict");
        else
        {
            result.pairEvidence.push_back(comparisons[begin]);
            if (end - begin > 1)
                Add(result.reasons, "duplicate_pair_identity_ignored");
        }
        begin = end;
    }

    std::optional<std::string> commonMaterialityIdentity;
    std::optional<std::string> commonMaterialityCanonical;
    std::optional<Pair::ProfitabilityPrimaryMetric> commonPrimaryMetric;
    std::optional<Pair::ObjectiveProvenance> commonControlObjective;
    std::map<std::string, const PairSummary*> replicationUnits;
    std::vector<const PairSummary*> independentValid;

    for (const PairSummary& summary : result.pairEvidence)
    {
        const bool materialityIdentityValid =
            !summary.materialityPolicyCanonical.empty() &&
            TaggedHash(summary.materialityPolicyIdentity) &&
            TrainingObjective::DeterministicHash(
                summary.materialityPolicyCanonical) ==
                summary.materialityPolicyIdentity;
        if (!materialityIdentityValid)
            Add(result.reasons, "materiality_policy_identity_invalid");
        if (!commonMaterialityIdentity)
        {
            commonMaterialityIdentity = summary.materialityPolicyIdentity;
            commonMaterialityCanonical = summary.materialityPolicyCanonical;
            commonPrimaryMetric = summary.primaryProfitabilityMetric;
        }
        else if (*commonMaterialityIdentity !=
                     summary.materialityPolicyIdentity ||
                 *commonMaterialityCanonical !=
                     summary.materialityPolicyCanonical ||
                 *commonPrimaryMetric != summary.primaryProfitabilityMetric)
        {
            Add(result.reasons, "materiality_policy_identity_mismatch");
        }

        if (!ValidObjective(summary.controlObjective) ||
            !ValidObjective(summary.treatmentObjective))
            Add(result.reasons, "pair_objective_identity_invalid");
        if (!(summary.treatmentObjective == objectiveUnderReview))
            Add(result.reasons, "objective_under_review_mismatch");
        if (!commonControlObjective)
            commonControlObjective = summary.controlObjective;
        else if (!(*commonControlObjective == summary.controlObjective))
            Add(result.reasons, "control_objective_identity_mismatch");
        if (summary.pairIdentity.empty() ||
            summary.replicationUnitCanonical.empty() ||
            !TaggedHash(summary.replicationUnitIdentity) ||
            TrainingObjective::DeterministicHash(
                summary.replicationUnitCanonical) !=
                summary.replicationUnitIdentity ||
            summary.symbol.empty() || summary.horizon <= 0)
            Add(result.reasons, "pair_summary_identity_incomplete");

        if (summary.disposition == Pair::Disposition::InvalidComparison)
        {
            ++result.invalidCount;
            continue;
        }
        if (summary.disposition == Pair::Disposition::Incomplete)
        {
            ++result.incompleteCount;
            continue;
        }

        const auto [iterator, inserted] = replicationUnits.emplace(
            summary.replicationUnitIdentity, &summary);
        if (!inserted)
        {
            if (iterator->second->disposition != summary.disposition)
                Add(result.reasons,
                    "deterministic_replication_unit_disposition_conflict");
            else
                Add(result.reasons,
                    "non_independent_replication_unit_ignored");
            continue;
        }
        independentValid.push_back(&summary);
    }

    if (commonMaterialityIdentity)
        result.materialityPolicyIdentity = *commonMaterialityIdentity;

    std::set<std::string> symbols;
    std::set<int> horizons;
    std::set<std::string> negativeSymbols;
    std::set<int> negativeHorizons;
    bool outsideInitialUnit = false;
    bool profitabilityDisagreement = false;
    bool classificationConcern = false;
    bool weakMateriality = false;

    for (const PairSummary* summary : independentValid)
    {
        ++result.validPairCount;
        symbols.insert(summary->symbol);
        horizons.insert(summary->horizon);
        outsideInitialUnit = outsideInitialUnit ||
            summary->symbol != policy.initialSymbol ||
            summary->horizon != policy.initialHorizon;
        switch (summary->disposition)
        {
            case Pair::Disposition::Promising:
                ++result.promisingCount;
                break;
            case Pair::Disposition::Mixed:
                ++result.mixedCount;
                profitabilityDisagreement = profitabilityDisagreement ||
                    HasReason(*summary,
                        "profitability_primitives_disagree_or_are_ambiguous");
                classificationConcern = classificationConcern ||
                    HasReason(*summary,
                        "classification_degradation_policy_not_configured") ||
                    HasReason(*summary,
                        "classification_policy_metric_unavailable");
                weakMateriality = weakMateriality ||
                    HasReason(*summary,
                        "profitability_effect_below_materiality_or_directionally_ambiguous") ||
                    HasReason(*summary,
                        "primary_profitability_delta_unavailable");
                break;
            case Pair::Disposition::NotPromising:
                ++result.notPromisingCount;
                negativeSymbols.insert(summary->symbol);
                negativeHorizons.insert(summary->horizon);
                classificationConcern = classificationConcern ||
                    HasReason(*summary,
                        "classification_degradation_unacceptable");
                break;
            case Pair::Disposition::InvalidComparison:
            case Pair::Disposition::Incomplete:
                break;
        }
    }

    result.distinctSymbolCount = symbols.size();
    result.distinctHorizonCount = horizons.size();
    result.replicationDiversitySatisfied =
        result.distinctSymbolCount >= policy.minimumDistinctSymbols &&
        result.distinctHorizonCount >= policy.minimumDistinctHorizons &&
        (!policy.requireReplicationOutsideInitialUnit || outsideInitialUnit);

    const auto repairReason = std::find_if(
        result.reasons.begin(), result.reasons.end(), [](const std::string& value)
        {
            return value != "duplicate_pair_identity_ignored" &&
                value != "non_independent_replication_unit_ignored";
        });
    if (result.invalidCount > 0 || repairReason != result.reasons.end())
    {
        if (result.invalidCount > 0)
            Add(result.reasons,
                "invalid_comparisons_are_not_scientific_evidence");
        result.nextAction = NextAction::RepairComparability;
        return result;
    }
    if (result.incompleteCount > 0)
    {
        Add(result.reasons,
            "incomplete_comparisons_do_not_count_as_replications");
        result.nextAction = NextAction::WaitForValidResult;
        return result;
    }
    if (result.validPairCount == 0)
    {
        Add(result.reasons, "no_valid_pair_result");
        result.nextAction = NextAction::WaitForValidResult;
        return result;
    }

    const bool negativeDiversity =
        negativeSymbols.size() >= 2 || negativeHorizons.size() >= 2;
    if (result.notPromisingCount >=
            policy.confirmatoryNotPromisingReplications &&
        negativeDiversity)
    {
        Add(result.reasons,
            "not_promising_result_confirmed_across_diverse_units");
        result.nextAction = NextAction::StopObjective;
        return result;
    }

    const double validCount = static_cast<double>(result.validPairCount);
    const double promisingFraction =
        static_cast<double>(result.promisingCount) / validCount;
    const double mixedFraction =
        static_cast<double>(result.mixedCount) / validCount;
    const bool eligible =
        result.validPairCount >= policy.minimumValidReplications &&
        result.promisingCount >= policy.minimumPromisingReplications &&
        promisingFraction >= policy.minimumPromisingFraction &&
        mixedFraction <= policy.maximumMixedFraction &&
        result.notPromisingCount <=
            policy.maximumNotPromisingForEligibility &&
        result.replicationDiversitySatisfied;
    if (eligible)
    {
        Add(result.reasons,
            "conservative_replication_and_diversity_requirements_satisfied");
        result.nextAction = NextAction::EligibleForCoefficientExploration;
        return result;
    }

    if (result.validPairCount >=
        policy.maximumValidReplicationsBeforeStop)
    {
        Add(result.reasons,
            "screening_replication_cap_reached_without_eligibility");
        result.nextAction = NextAction::StopObjective;
        return result;
    }

    if (classificationConcern)
        Add(result.reasons, "replicate_classification_concern");
    if (profitabilityDisagreement)
        Add(result.reasons, "replicate_profitability_metric_disagreement");
    if (weakMateriality)
        Add(result.reasons, "replicate_weak_or_ambiguous_materiality");
    if (result.notPromisingCount == 1)
        Add(result.reasons,
            "one_not_promising_result_requires_one_diverse_confirmation");

    if (result.mixedCount > 0 && result.promisingCount == 0 &&
        result.notPromisingCount == 0)
    {
        Add(result.reasons, "targeted_replication_required");
        result.nextAction = NextAction::Replicate;
    }
    else if (!result.replicationDiversitySatisfied ||
        result.notPromisingCount > 0 ||
        (result.promisingCount > 0 &&
         result.validPairCount < policy.minimumValidReplications))
    {
        Add(result.reasons, "broader_symbol_horizon_evidence_required");
        result.nextAction = NextAction::ReplicateBroader;
    }
    else
    {
        Add(result.reasons, "targeted_replication_required");
        result.nextAction = NextAction::Replicate;
    }
    return result;
}

std::string NextActionText(NextAction value)
{
    switch (value)
    {
        case NextAction::WaitForValidResult: return "WAIT_FOR_VALID_RESULT";
        case NextAction::RepairComparability: return "REPAIR_COMPARABILITY";
        case NextAction::StopObjective: return "STOP_OBJECTIVE";
        case NextAction::Replicate: return "REPLICATE";
        case NextAction::ReplicateBroader: return "REPLICATE_BROADER";
        case NextAction::EligibleForCoefficientExploration:
            return "ELIGIBLE_FOR_COEFFICIENT_EXPLORATION";
    }
    throw std::invalid_argument("unknown_post_pair_next_action");
}

std::string RenderMachineReadable(const Evaluation& evaluation)
{
    std::ostringstream output;
    output << "TRAINING_OBJECTIVE_REPLICATION_POLICY,version=1"
           << ",objective_under_review="
           << (ValidObjective(evaluation.objectiveUnderReview)
                   ? TrainingObjective::ParseSupportedCanonicalText(
                         evaluation.objectiveUnderReview.canonical)
                         .objectiveIdentifier
                   : "INVALID")
           << ",objective_hash=" << evaluation.objectiveUnderReview.hash
           << ",valid_pair_count=" << evaluation.validPairCount
           << ",promising_count=" << evaluation.promisingCount
           << ",mixed_count=" << evaluation.mixedCount
           << ",not_promising_count=" << evaluation.notPromisingCount
           << ",invalid_count=" << evaluation.invalidCount
           << ",incomplete_count=" << evaluation.incompleteCount
           << ",distinct_symbol_count=" << evaluation.distinctSymbolCount
           << ",distinct_horizon_count="
           << evaluation.distinctHorizonCount
           << ",materiality_policy_identity="
           << (evaluation.materialityPolicyIdentity.empty()
                   ? "NONE" : evaluation.materialityPolicyIdentity)
           << ",replication_policy_identity="
           << (evaluation.replicationPolicyIdentity.empty()
                   ? "NONE" : evaluation.replicationPolicyIdentity)
           << ",replication_diversity_satisfied="
           << (evaluation.replicationDiversitySatisfied ? "true" : "false")
           << ",next_action=" << NextActionText(evaluation.nextAction)
           << ",reasons=" << Reasons(evaluation.reasons) << '\n';

    for (const PairSummary& summary : evaluation.pairEvidence)
    {
        output << "TRAINING_OBJECTIVE_REPLICATION_PAIR"
               << ",pair_identity=" << summary.pairIdentity
               << ",replication_unit_identity="
               << summary.replicationUnitIdentity
               << ",symbol=" << summary.symbol
               << ",horizon=" << summary.horizon
               << ",control_objective_hash="
               << summary.controlObjective.hash
               << ",treatment_objective_hash="
               << summary.treatmentObjective.hash
               << ",primary_profitability_metric="
               << PrimaryMetricText(summary.primaryProfitabilityMetric)
               << ",materiality_policy_identity="
               << summary.materialityPolicyIdentity
               << ",disposition="
               << Pair::DispositionText(summary.disposition);
        RenderMetric(output, "primary_profitability",
                     summary.primaryProfitability);
        RenderMetric(output, "aggregate_profitability",
                     summary.aggregateProfitability);
        RenderMetric(output, "average_profitability",
                     summary.averageProfitability);
        RenderMetric(output, "inference_accuracy", summary.inferenceAccuracy);
        RenderMetric(output, "accept_accuracy", summary.acceptAccuracy);
        RenderMetric(output, "accept_rate", summary.acceptRate);
        RenderMetric(output, "neutral_proportion",
                     summary.predictedNeutralProportion);
        RenderMetric(output, "leader_score", summary.leaderScore);
        output << ",invalid_reasons=" << Reasons(summary.invalidReasons)
               << ",incomplete_reasons="
               << Reasons(summary.incompleteReasons)
               << ",interpretation_reasons="
               << Reasons(summary.interpretationReasons) << '\n';
    }
    return output.str();
}

} // namespace EA::PostPairReplicationPolicy
