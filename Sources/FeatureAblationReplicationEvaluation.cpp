#include "FeatureAblationReplicationEvaluation.hpp"

#include "ModelInputContract.hpp"
#include "ModelInputExpansion.hpp"
#include "TrainingObjective.hpp"

#include <algorithm>
#include <cmath>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::FeatureAblationReplicationEvaluation
{
namespace
{

MetricSummary Summarize(std::vector<double> values, bool includeMedian)
{
    MetricSummary result;
    if (values.empty()) return result;
    result.valueCount = values.size();
    double sum = 0.0;
    for (const double value : values)
    {
        if (!std::isfinite(value))
            throw std::invalid_argument("replication_metric_nonfinite");
        sum += value;
        if (value > 0.0) ++result.positiveCount;
        else if (value < 0.0) ++result.negativeCount;
        else ++result.zeroCount;
    }
    if (!std::isfinite(sum))
        throw std::invalid_argument("replication_metric_sum_nonfinite");
    result.sum = sum;
    result.mean = sum / static_cast<double>(values.size());
    if (includeMedian)
    {
        std::sort(values.begin(), values.end());
        const std::size_t middle = values.size() / 2;
        result.median = values.size() % 2 == 0
            ? (values[middle - 1] + values[middle]) / 2.0
            : values[middle];
    }
    return result;
}

std::vector<double> Deltas(
    const std::vector<MemberEvaluation>& members,
    const Pair::MetricDelta Pair::ComparisonResult::* metric)
{
    std::vector<double> values;
    for (const auto& member : members)
    {
        if (!member.scientificallyValidComplete) continue;
        const auto& delta = member.comparison.*metric;
        if (!delta.controlMinusAblation)
            throw std::invalid_argument("complete_replication_metric_missing");
        values.push_back(*delta.controlMinusAblation);
    }
    return values;
}

void AppendReason(std::vector<std::string>& reasons, std::string value)
{
    if (std::find(reasons.begin(), reasons.end(), value) == reasons.end())
        reasons.push_back(std::move(value));
}

bool TaggedHash(const std::optional<std::string>& value)
{
    if (!value || value->size() != 24 || !value->starts_with("fnv1a64:"))
        return false;
    for (std::size_t index = 8; index < value->size(); ++index)
        if (!((*value)[index] >= '0' && (*value)[index] <= '9') &&
            !((*value)[index] >= 'a' && (*value)[index] <= 'f'))
            return false;
    return true;
}

bool HasInputIdentity(const MemberEvaluation& member,
                      int requiredLayoutVersion)
{
    return member.controlModelInputWidth == Pair::kCausalSurpriseModelInputWidth &&
        member.treatmentModelInputWidth ==
            Pair::kCausalSurpriseModelInputWidth &&
        member.controlModelInputLayoutVersion == requiredLayoutVersion &&
        member.treatmentModelInputLayoutVersion == requiredLayoutVersion;
}

bool HasCorrectedSnapshotIdentity(const MemberEvaluation& member)
{
    return member.controlEconomicCalendarSnapshotId &&
        member.treatmentEconomicCalendarSnapshotId &&
        *member.controlEconomicCalendarSnapshotId > 0 &&
        member.controlEconomicCalendarSnapshotId ==
            member.treatmentEconomicCalendarSnapshotId &&
        TaggedHash(member.controlEconomicCalendarSnapshotHash) &&
        TaggedHash(member.treatmentEconomicCalendarSnapshotHash) &&
        member.controlEconomicCalendarSnapshotHash ==
            member.treatmentEconomicCalendarSnapshotHash;
}

} // namespace

std::vector<std::pair<long long, long long>> ParseExperimentIdPairs(
    std::string_view text)
{
    if (text.empty())
        throw std::invalid_argument(
            "--compare-feature-ablation-replications requires at least one CONTROL_ID:TREATMENT_ID pair");
    std::vector<std::pair<long long, long long>> result;
    std::set<long long> experimentIds;
    std::size_t start = 0;
    while (start <= text.size())
    {
        const std::size_t end = text.find(',', start);
        const std::string_view token = text.substr(
            start, end == std::string_view::npos
                ? std::string_view::npos : end - start);
        if (token.empty())
            throw std::invalid_argument(
                "feature-ablation replication membership contains an empty pair");
        const auto pair = Pair::ParseExperimentIdPair(token);
        if (!experimentIds.insert(pair.first).second ||
            !experimentIds.insert(pair.second).second)
            throw std::invalid_argument(
                "feature-ablation replication experiment IDs must be globally unique");
        result.push_back(pair);
        if (end == std::string_view::npos) break;
        start = end + 1;
    }
    return result;
}

MemberEvaluation MakeMemberEvaluation(
    std::size_t ordinal,
    const Pair::ArmEvidence& control,
    const Pair::ArmEvidence& treatment,
    const Pair::ComparisonResult& comparison,
    bool evidenceOrderIsControlThenAblation)
{
    MemberEvaluation member;
    member.ordinal = ordinal;
    member.controlExperimentId =
        control.authoritative.configuration.experimentId;
    member.treatmentExperimentId =
        treatment.authoritative.configuration.experimentId;
    member.symbol = control.authoritative.configuration.symbol;
    member.predictionHorizon =
        control.authoritative.configuration.predictionHorizon;
    member.pairDisposition = comparison.disposition;
    member.evidenceClassification = comparison.evidenceClassification;
    member.pairEvaluationIdentityHash = evidenceOrderIsControlThenAblation
        ? Pair::EvaluationIdentityHash(control, treatment, comparison)
        : Pair::EvaluationIdentityHash(treatment, control, comparison);
    member.ablationIdentityHash = comparison.ablationIdentityHash;
    member.controlEconomicCalendarSnapshotId =
        control.extended.economicCalendarSnapshotId;
    member.controlEconomicCalendarSnapshotHash =
        control.extended.economicCalendarSnapshotHash;
    member.treatmentEconomicCalendarSnapshotId =
        treatment.extended.economicCalendarSnapshotId;
    member.treatmentEconomicCalendarSnapshotHash =
        treatment.extended.economicCalendarSnapshotHash;
    member.controlModelInputWidth =
        control.extended.configuredModelInputWidth;
    member.controlModelInputLayoutVersion =
        control.extended.configuredModelInputLayoutVersion;
    member.treatmentModelInputWidth =
        treatment.extended.configuredModelInputWidth;
    member.treatmentModelInputLayoutVersion =
        treatment.extended.configuredModelInputLayoutVersion;
    member.incompleteReasons = comparison.incompleteReasons;
    member.invalidReasons = comparison.invalidReasons;
    member.comparison = comparison;
    if (comparison.evidenceClassification ==
        Pair::EvidenceClassification::PreFixCausalSurpriseEvidence)
    {
        member.evidenceState = MemberEvidenceState::HistoricalPreFix;
        member.exclusionReasons.push_back(
            "semantic_layout_6_excluded_from_corrected_replication");
        return member;
    }
    if (comparison.evidenceClassification ==
        Pair::EvidenceClassification::IncompatibleOrInvalidEvidence)
    {
        member.evidenceState = MemberEvidenceState::Invalid;
        return member;
    }
    switch (comparison.disposition)
    {
        case Pair::Disposition::ComparableComplete:
            member.evidenceState = MemberEvidenceState::Complete;
            member.scientificallyValidComplete = true;
            break;
        case Pair::Disposition::ComparableIncomplete:
        case Pair::Disposition::MissingFinalInference:
            member.evidenceState = MemberEvidenceState::Incomplete;
            break;
        case Pair::Disposition::ProfitabilityEvidenceUnavailable:
            member.evidenceState =
                MemberEvidenceState::ProfitabilityUnavailable;
            break;
        case Pair::Disposition::IncompatibleConfiguration:
        case Pair::Disposition::AmbiguousFinalInference:
        case Pair::Disposition::InvalidAblationPair:
            member.evidenceState = MemberEvidenceState::Invalid;
            break;
    }
    return member;
}

std::string PolicyCanonicalText(const ReplicationPolicy& policy)
{
    if (policy.version != kReplicationPolicyVersion ||
        policy.minimumValidReplications != kMinimumValidReplications)
        throw std::invalid_argument("unsupported_replication_policy");
    return
        "feature_ablation_replication_policy_v1;"
        "minimum_valid_replications=3;"
        "primary_metrics=aggregate_profitability_delta_and_average_profitability_per_actionable_delta;"
        "promising=both_positive_majority_and_both_signed_sums_positive_and_no_integrity_failure;"
        "not_promising=both_negative_majority_and_both_signed_sums_negative_and_no_integrity_failure;"
        "mixed=sufficient_evidence_without_concordant_primary_decision_or_with_integrity_failure;"
        "insufficient_evidence=fewer_than_minimum_valid_replications;"
        "classification_metrics=corroboration_only;"
        "incomplete_or_invalid_members=never_counted_negative;";
}

std::string PolicyHash(const ReplicationPolicy& policy)
{
    return TrainingObjective::DeterministicHash(PolicyCanonicalText(policy));
}

std::string SoftwareReadinessCanonicalText(
    const SoftwareReadinessAudit& audit)
{
    const auto flag = [](bool value) { return value ? "true" : "false"; };
    std::ostringstream output;
    output << kEconomicEventSoftwareReadinessVersion << ';'
           << "version=" << audit.version << ';'
           << "causal_release_boundary=" << flag(audit.causalReleaseBoundary) << ';'
           << "consensus_provenance_retained=" << flag(audit.consensusProvenanceRetained) << ';'
           << "missing_consensus_explicit=" << flag(audit.missingConsensusExplicit) << ';'
           << "scalar_range_semantics_validated=" << flag(audit.scalarRangeSemanticsValidated) << ';'
           << "normalization_fail_closed=" << flag(audit.normalizationFailClosed) << ';'
           << "finite_feature_values_enforced=" << flag(audit.finiteFeatureValuesEnforced) << ';'
           << "input_semantic_identity_fail_closed=" << flag(audit.inputSemanticIdentityFailClosed) << ';'
           << "deterministic_feature_path_tested=" << flag(audit.deterministicFeaturePathTested) << ';'
           << "availability_diagnostics_present=" << flag(audit.availabilityDiagnosticsPresent) << ';'
           << "read_only_evaluation=" << flag(audit.readOnlyEvaluation) << ';'
           << "activation_mutation_path_absent=" << flag(audit.activationMutationPathAbsent) << ';'
           << "registered_model_input_contracts=true;"
           << "current_model_input_width=" << EA::kCurrentModelInputWidth << ';'
           << "current_semantic_layout_version="
           << EA::kModelInputSemanticLayoutVersion << ';';
    return output.str();
}

bool SoftwareReady(const SoftwareReadinessAudit& audit)
{
    return audit.version == 2 && audit.causalReleaseBoundary &&
        audit.consensusProvenanceRetained && audit.missingConsensusExplicit &&
        audit.scalarRangeSemanticsValidated && audit.normalizationFailClosed &&
        audit.finiteFeatureValuesEnforced &&
        audit.inputSemanticIdentityFailClosed &&
        audit.deterministicFeaturePathTested &&
        audit.availabilityDiagnosticsPresent && audit.readOnlyEvaluation &&
        audit.activationMutationPathAbsent &&
        EA::kCurrentModelInputWidth == 77 &&
        EA::kModelInputSemanticLayoutVersion == 7;
}

ReplicationEvaluation Evaluate(
    std::vector<MemberEvaluation> members,
    const ReplicationPolicy& policy,
    const SoftwareReadinessAudit& softwareAudit,
    EvidenceScope evidenceScope)
{
    ReplicationEvaluation result;
    result.evidenceScope = evidenceScope;
    result.policyCanonical = PolicyCanonicalText(policy);
    result.policyHash = TrainingObjective::DeterministicHash(
        result.policyCanonical);
    result.softwareReadinessCanonical =
        SoftwareReadinessCanonicalText(softwareAudit);
    result.softwareReadinessHash = TrainingObjective::DeterministicHash(
        result.softwareReadinessCanonical);
    result.softwareReady = SoftwareReady(softwareAudit);

    std::set<long long> experimentIds;
    std::string expectedAblationIdentity;
    for (std::size_t index = 0; index < members.size(); ++index)
    {
        auto& member = members[index];
        if (member.ordinal != index + 1 || member.controlExperimentId <= 0 ||
            member.treatmentExperimentId <= 0 ||
            !experimentIds.insert(member.controlExperimentId).second ||
            !experimentIds.insert(member.treatmentExperimentId).second)
        {
            throw std::invalid_argument(
                "replication_membership_identity_invalid");
        }
        if (!member.ablationIdentityHash.empty())
        {
            if (expectedAblationIdentity.empty())
                expectedAblationIdentity = member.ablationIdentityHash;
            else if (member.ablationIdentityHash != expectedAblationIdentity)
            {
                member.evidenceState = MemberEvidenceState::Invalid;
                member.scientificallyValidComplete = false;
                AppendReason(member.invalidReasons,
                             "replication_ablation_identity_mismatch");
            }
        }
        if (evidenceScope == EvidenceScope::CorrectedCausalSurprise)
        {
            if (member.evidenceClassification ==
                Pair::EvidenceClassification::PreFixCausalSurpriseEvidence)
            {
                if (!HasInputIdentity(
                        member,
                        Pair::kPreFixCausalSurpriseSemanticLayoutVersion))
                {
                    member.evidenceState = MemberEvidenceState::Invalid;
                    member.scientificallyValidComplete = false;
                    AppendReason(member.invalidReasons,
                        "pre_fix_causal_surprise_input_identity_invalid");
                }
                else
                {
                    member.evidenceState =
                        MemberEvidenceState::HistoricalPreFix;
                    member.scientificallyValidComplete = false;
                    AppendReason(member.exclusionReasons,
                        "semantic_layout_6_excluded_from_corrected_replication");
                }
            }
            else if (member.evidenceClassification ==
                     Pair::EvidenceClassification::
                         CorrectedCausalSurprisePairEvidence)
            {
                if (!HasInputIdentity(
                        member,
                        Pair::kCorrectedCausalSurpriseSemanticLayoutVersion))
                {
                    member.evidenceState = MemberEvidenceState::Invalid;
                    member.scientificallyValidComplete = false;
                    AppendReason(member.invalidReasons,
                        "corrected_causal_surprise_input_identity_invalid");
                }
                if (!HasCorrectedSnapshotIdentity(member))
                {
                    member.evidenceState = MemberEvidenceState::Invalid;
                    member.scientificallyValidComplete = false;
                    AppendReason(member.invalidReasons,
                        "corrected_causal_surprise_snapshot_identity_invalid");
                }
            }
            else
            {
                member.evidenceState = MemberEvidenceState::Invalid;
                member.scientificallyValidComplete = false;
                AppendReason(member.invalidReasons,
                    "corrected_causal_surprise_classification_missing_or_invalid");
            }
        }
    }

    std::ostringstream membership;
    membership << "feature_ablation_replication_membership_v1;declared_pair_count="
               << members.size() << ';';
    for (const auto& member : members)
        membership << "ordinal=" << member.ordinal
                   << ",control_experiment_id=" << member.controlExperimentId
                   << ",treatment_experiment_id=" << member.treatmentExperimentId
                   << ';';
    result.membershipIdentityCanonical = membership.str();
    result.membershipIdentityHash = TrainingObjective::DeterministicHash(
        result.membershipIdentityCanonical);

    std::set<std::string> symbols;
    std::set<int> horizons;
    std::set<std::string> economicCalendarCorpora;
    result.population.declaredPairCount = members.size();
    for (const auto& member : members)
    {
        if (!member.symbol.empty()) symbols.insert(member.symbol);
        if (member.predictionHorizon) horizons.insert(*member.predictionHorizon);
        const auto snapshot = [](const std::optional<long long>& id,
                                 const std::optional<std::string>& hash)
        {
            return id && hash
                ? std::to_string(*id) + "/" + *hash
                : std::string{"legacy_live_unbound"};
        };
        economicCalendarCorpora.insert(
            snapshot(member.controlEconomicCalendarSnapshotId,
                     member.controlEconomicCalendarSnapshotHash) + "|" +
            snapshot(member.treatmentEconomicCalendarSnapshotId,
                     member.treatmentEconomicCalendarSnapshotHash));
        switch (member.evidenceState)
        {
            case MemberEvidenceState::Complete:
                ++result.population.completeComparablePairCount;
                ++result.population.profitabilityAvailablePairCount;
                if (member.evidenceClassification ==
                    Pair::EvidenceClassification::
                        CorrectedCausalSurprisePairEvidence)
                    ++result.population.correctedValidPairCount;
                break;
            case MemberEvidenceState::Incomplete:
                ++result.population.incompletePairCount;
                break;
            case MemberEvidenceState::Invalid:
                ++result.population.invalidPairCount;
                result.evidenceIntegrityFailure = true;
                AppendReason(result.integrityReasons,
                             "invalid_replication_member");
                break;
            case MemberEvidenceState::MissingEvidence:
                ++result.population.missingEvidencePairCount;
                result.evidenceIntegrityFailure = true;
                AppendReason(result.integrityReasons,
                             "missing_replication_member_evidence");
                break;
            case MemberEvidenceState::ProfitabilityUnavailable:
                ++result.population.profitabilityUnavailablePairCount;
                break;
            case MemberEvidenceState::HistoricalPreFix:
                ++result.population.historicalPreFixPairCount;
                break;
        }
    }
    result.population.distinctSymbolCount = symbols.size();
    result.population.distinctHorizonCount = horizons.size();
    result.population.distinctEconomicCalendarCorpusCount =
        economicCalendarCorpora.size();

    result.aggregateProfitability = Summarize(
        Deltas(members, &Pair::ComparisonResult::aggregateProfitability), true);
    result.averageProfitability = Summarize(
        Deltas(members, &Pair::ComparisonResult::averageProfitability), true);
    result.inferenceAccuracy = Summarize(
        Deltas(members, &Pair::ComparisonResult::inferenceAccuracy), false);
    result.leaderScore = Summarize(
        Deltas(members, &Pair::ComparisonResult::leaderScore), false);
    result.neutralProportion = Summarize(
        Deltas(members, &Pair::ComparisonResult::neutralProportion), false);
    result.actionableCount = Summarize(
        Deltas(members, &Pair::ComparisonResult::actionableCount), false);
    result.population.profitabilityPositivePairCount =
        result.aggregateProfitability.positiveCount;
    result.population.profitabilityNegativePairCount =
        result.aggregateProfitability.negativeCount;
    result.population.profitabilityZeroPairCount =
        result.aggregateProfitability.zeroCount;

    const std::size_t valid = result.population.profitabilityAvailablePairCount;
    if (valid < static_cast<std::size_t>(policy.minimumValidReplications))
    {
        result.decision = ReplicationDecision::InsufficientEvidence;
    }
    else if (result.evidenceIntegrityFailure)
    {
        result.decision = ReplicationDecision::Mixed;
    }
    else
    {
        const bool aggregatePositiveMajority =
            result.aggregateProfitability.positiveCount > valid / 2;
        const bool averagePositiveMajority =
            result.averageProfitability.positiveCount > valid / 2;
        const bool aggregateNegativeMajority =
            result.aggregateProfitability.negativeCount > valid / 2;
        const bool averageNegativeMajority =
            result.averageProfitability.negativeCount > valid / 2;
        const bool bothSumsPositive =
            *result.aggregateProfitability.sum > 0.0 &&
            *result.averageProfitability.sum > 0.0;
        const bool bothSumsNegative =
            *result.aggregateProfitability.sum < 0.0 &&
            *result.averageProfitability.sum < 0.0;
        if (aggregatePositiveMajority && averagePositiveMajority &&
            bothSumsPositive)
            result.decision = ReplicationDecision::Promising;
        else if (aggregateNegativeMajority && averageNegativeMajority &&
                 bothSumsNegative)
            result.decision = ReplicationDecision::NotPromising;
        else
            result.decision = ReplicationDecision::Mixed;
    }

    if (!result.softwareReady)
        result.action = ProductionizationAction::BlockedSoftwareReadiness;
    else if (result.decision == ReplicationDecision::InsufficientEvidence)
        result.action = ProductionizationAction::AwaitReplication;
    else if (result.decision == ReplicationDecision::Promising)
        result.action = ProductionizationAction::EligibleForActivationReview;
    else
        result.action = ProductionizationAction::DoNotEnable;

    if (result.population.invalidPairCount != 0 ||
        result.population.missingEvidencePairCount != 0)
        result.evidenceClassification =
            EvidenceClassification::IncompatibleOrInvalidEvidence;
    else if (evidenceScope == EvidenceScope::CorrectedCausalSurprise)
    {
        if (result.population.correctedValidPairCount >=
            static_cast<std::size_t>(policy.minimumValidReplications))
            result.evidenceClassification = EvidenceClassification::
                CorrectedCausalSurpriseReplicationEvidence;
        else if (result.population.correctedValidPairCount != 0 ||
                 result.population.historicalPreFixPairCount == 0)
            result.evidenceClassification = EvidenceClassification::
                CorrectedCausalSurprisePairEvidence;
        else
            result.evidenceClassification =
                EvidenceClassification::PreFixCausalSurpriseEvidence;
    }

    result.members = std::move(members);
    for (const auto& member : result.members)
    {
        if (!member.comparison.canonicalAblatedFeatureSet.empty())
        {
            result.canonicalAblatedFeatureSet =
                member.comparison.canonicalAblatedFeatureSet;
            break;
        }
    }
    std::ostringstream identity;
    identity << "feature_ablation_replication_evaluation_v3;"
             << "membership_identity_hash=" << result.membershipIdentityHash << ';'
             << "policy_hash=" << result.policyHash << ';'
             << "ablation_identity_hash="
             << (expectedAblationIdentity.empty() ? "NULL" : expectedAblationIdentity)
             << ';'
             << "pair_evaluation_semantic_version=3;"
             << "evidence_scope=" << EvidenceScopeText(evidenceScope) << ';'
             << "evidence_classification="
             << EvidenceClassificationText(result.evidenceClassification) << ';'
             << "corrected_valid_pair_count="
             << result.population.correctedValidPairCount << ';'
             << "historical_pre_fix_pair_count="
             << result.population.historicalPreFixPairCount << ';'
             << "replication_evaluation_semantic_version="
             << kReplicationEvaluationVersion << ';'
             << "expected_ablation_mask="
             << (result.canonicalAblatedFeatureSet.empty()
                     ? "NULL" : result.canonicalAblatedFeatureSet) << ';'
             << "input_contract=validated_per_pair;"
             << "economic_calendar_snapshot=provenance_not_treatment;"
             << "software_readiness_hash=" << result.softwareReadinessHash << ';';
    for (const auto& member : result.members)
        identity << "ordinal=" << member.ordinal
                 << ",pair_evaluation_identity_hash="
                 << (member.pairEvaluationIdentityHash.empty()
                         ? "NULL" : member.pairEvaluationIdentityHash)
                 << ",evidence_state="
                 << MemberEvidenceStateText(member.evidenceState) << ';';
    result.evaluationIdentityCanonical = identity.str();
    result.evaluationIdentityHash = TrainingObjective::DeterministicHash(
        result.evaluationIdentityCanonical);
    return result;
}

std::string MemberEvidenceStateText(MemberEvidenceState value)
{
    switch (value)
    {
        case MemberEvidenceState::Complete: return "complete";
        case MemberEvidenceState::Incomplete: return "incomplete";
        case MemberEvidenceState::Invalid: return "invalid";
        case MemberEvidenceState::MissingEvidence: return "missing_evidence";
        case MemberEvidenceState::ProfitabilityUnavailable:
            return "profitability_unavailable";
        case MemberEvidenceState::HistoricalPreFix:
            return "historical_pre_fix";
    }
    throw std::invalid_argument("unknown_replication_member_evidence_state");
}

std::string EvidenceScopeText(EvidenceScope value)
{
    switch (value)
    {
        case EvidenceScope::GenericFeatureAblation:
            return "generic_feature_ablation";
        case EvidenceScope::CorrectedCausalSurprise:
            return "corrected_causal_surprise";
    }
    throw std::invalid_argument("unknown_replication_evidence_scope");
}

std::string EvidenceClassificationText(EvidenceClassification value)
{
    switch (value)
    {
        case EvidenceClassification::GenericFeatureAblationEvidence:
            return "generic_feature_ablation_evidence";
        case EvidenceClassification::PreFixCausalSurpriseEvidence:
            return "pre_fix_causal_surprise_evidence";
        case EvidenceClassification::CorrectedCausalSurprisePairEvidence:
            return "corrected_causal_surprise_pair_evidence";
        case EvidenceClassification::CorrectedCausalSurpriseReplicationEvidence:
            return "corrected_causal_surprise_replication_evidence";
        case EvidenceClassification::IncompatibleOrInvalidEvidence:
            return "incompatible_or_invalid_evidence";
    }
    throw std::invalid_argument(
        "unknown_replication_evidence_classification");
}

std::string ReplicationDecisionText(ReplicationDecision value)
{
    switch (value)
    {
        case ReplicationDecision::InsufficientEvidence:
            return "insufficient_evidence";
        case ReplicationDecision::Promising: return "promising";
        case ReplicationDecision::Mixed: return "mixed";
        case ReplicationDecision::NotPromising: return "not_promising";
    }
    throw std::invalid_argument("unknown_replication_decision");
}

std::string ProductionizationActionText(ProductionizationAction value)
{
    switch (value)
    {
        case ProductionizationAction::AwaitReplication:
            return "await_replication";
        case ProductionizationAction::EligibleForActivationReview:
            return "eligible_for_activation_review";
        case ProductionizationAction::DoNotEnable: return "do_not_enable";
        case ProductionizationAction::BlockedSoftwareReadiness:
            return "blocked_software_readiness";
    }
    throw std::invalid_argument("unknown_productionization_action");
}

int ExitCode(const ReplicationEvaluation& result)
{
    if (result.population.invalidPairCount != 0 ||
        result.population.missingEvidencePairCount != 0)
        return 3;
    if (result.population.incompletePairCount != 0 ||
        result.population.profitabilityUnavailablePairCount != 0)
        return 4;
    return 0;
}

} // namespace EA::FeatureAblationReplicationEvaluation
