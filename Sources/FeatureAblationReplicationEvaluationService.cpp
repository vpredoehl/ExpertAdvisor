#include "FeatureAblationReplicationEvaluationService.hpp"

#include "FeatureAblationPairEvaluationRepository.hpp"
#include "PairedTrainingObjectiveEvaluationRepository.hpp"

#include <iomanip>
#include <optional>
#include <sstream>
#include <stdexcept>

#include <pqxx/pqxx>

namespace EA::FeatureAblationReplicationEvaluation
{
namespace
{

namespace Shared = PairedTrainingObjectiveEvaluation;

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const bool safe =
            (character >= 'a' && character <= 'z') ||
            (character >= 'A' && character <= 'Z') ||
            (character >= '0' && character <= '9') ||
            character == '_' || character == '-' || character == '.' ||
            character == ':';
        if (!safe) character = '_';
    }
    return value.empty() ? "EMPTY" : value;
}

std::string Reasons(const std::vector<std::string>& values)
{
    if (values.empty()) return "NONE";
    std::ostringstream output;
    for (std::size_t index = 0; index < values.size(); ++index)
    {
        if (index != 0) output << '|';
        output << MachineText(values[index]);
    }
    return output.str();
}

std::string OptionalNumber(const std::optional<double>& value)
{
    if (!value) return "NULL";
    std::ostringstream output;
    output << std::setprecision(17) << *value;
    return output.str();
}

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

void PrintMetric(std::ostringstream& output,
                 std::string_view name,
                 const MetricSummary& metric)
{
    output << "FEATURE_ABLATION_REPLICATION_METRIC"
           << ",metric=" << name
           << ",value_count=" << metric.valueCount
           << ",sum=" << OptionalNumber(metric.sum)
           << ",mean=" << OptionalNumber(metric.mean)
           << ",median=" << OptionalNumber(metric.median)
           << ",positive_pair_count=" << metric.positiveCount
           << ",negative_pair_count=" << metric.negativeCount
           << ",zero_pair_count=" << metric.zeroCount << '\n';
}

struct ArmLoad
{
    std::optional<Pair::ArmEvidence> arm;
    std::optional<Shared::EvidenceLoadErrorKind> errorKind;
    std::string reason;
};

ArmLoad LoadArm(pqxx::transaction_base& transaction, long long experimentId)
{
    try
    {
        return {Pair::LoadAuthoritativeArmEvidence(transaction, experimentId),
                std::nullopt, {}};
    }
    catch (const Shared::EvidenceLoadError& error)
    {
        return {std::nullopt, error.kind(), error.reason()};
    }
    catch (const std::invalid_argument& error)
    {
        return {std::nullopt,
                Shared::EvidenceLoadErrorKind::ProvenanceContractFailure,
                error.what()};
    }
    catch (const pqxx::failure&)
    {
        throw;
    }
    catch (const std::runtime_error& error)
    {
        return {std::nullopt,
                Shared::EvidenceLoadErrorKind::ProvenanceContractFailure,
                error.what()};
    }
}

MemberEvaluation LoadMember(
    pqxx::transaction_base& transaction,
    std::size_t ordinal,
    const std::pair<long long, long long>& ids)
{
    const ArmLoad control = LoadArm(transaction, ids.first);
    const ArmLoad treatment = LoadArm(transaction, ids.second);
    if (control.arm && treatment.arm)
    {
        return MakeMemberEvaluation(
            ordinal, *control.arm, *treatment.arm,
            Pair::Compare(*control.arm, *treatment.arm));
    }

    MemberEvaluation member;
    member.ordinal = ordinal;
    member.controlExperimentId = ids.first;
    member.treatmentExperimentId = ids.second;
    const Pair::ArmEvidence* available = control.arm
        ? &*control.arm : treatment.arm ? &*treatment.arm : nullptr;
    if (available)
    {
        member.symbol = available->authoritative.configuration.symbol;
        member.predictionHorizon =
            available->authoritative.configuration.predictionHorizon;
    }
    const bool missing =
        control.errorKind == Shared::EvidenceLoadErrorKind::ExperimentNotFound ||
        treatment.errorKind == Shared::EvidenceLoadErrorKind::ExperimentNotFound;
    const bool ambiguous =
        control.errorKind == Shared::EvidenceLoadErrorKind::AmbiguousEvidence ||
        treatment.errorKind == Shared::EvidenceLoadErrorKind::AmbiguousEvidence;
    member.evidenceState = missing
        ? MemberEvidenceState::MissingEvidence
        : MemberEvidenceState::Invalid;
    member.pairDisposition = ambiguous
        ? Pair::Disposition::AmbiguousFinalInference
        : Pair::Disposition::IncompatibleConfiguration;
    if (!control.arm)
        member.invalidReasons.push_back("control_load_" + control.reason);
    if (!treatment.arm)
        member.invalidReasons.push_back("treatment_load_" + treatment.reason);
    return member;
}

} // namespace

std::string RenderComparisonOutput(const ReplicationEvaluation& result)
{
    std::ostringstream output;
    output << "FEATURE_ABLATION_REPLICATION_SET"
           << ",version=" << kReplicationEvaluationVersion
           << ",declared_pair_count=" << result.population.declaredPairCount
           << ",membership_identity_hash=" << result.membershipIdentityHash
           << ",evaluation_identity_hash=" << result.evaluationIdentityHash
           << ",read_only=true\n";
    output << "FEATURE_ABLATION_REPLICATION_POLICY"
           << ",version=" << kReplicationPolicyVersion
           << ",minimum_valid_replications=" << kMinimumValidReplications
           << ",policy_hash=" << result.policyHash
           << ",canonical=" << result.policyCanonical << '\n';
    for (const auto& member : result.members)
    {
        output << "FEATURE_ABLATION_REPLICATION_MEMBER"
               << ",ordinal=" << member.ordinal
               << ",control_experiment_id=" << member.controlExperimentId
               << ",treatment_experiment_id=" << member.treatmentExperimentId
               << ",symbol="
               << (member.symbol.empty() ? "NULL" : MachineText(member.symbol))
               << ",prediction_horizon="
               << (member.predictionHorizon
                       ? std::to_string(*member.predictionHorizon) : "NULL")
               << ",evidence_state="
               << MemberEvidenceStateText(member.evidenceState)
               << ",pair_disposition="
               << Pair::DispositionText(member.pairDisposition)
               << ",pair_evaluation_identity_hash="
               << (member.pairEvaluationIdentityHash.empty()
                       ? "NULL" : member.pairEvaluationIdentityHash)
               << ",ablation_identity_hash="
               << (member.ablationIdentityHash.empty()
                       ? "NULL" : member.ablationIdentityHash)
               << ",scientifically_valid_complete="
               << Boolean(member.scientificallyValidComplete)
               << ",incomplete_reasons=" << Reasons(member.incompleteReasons)
               << ",invalid_reasons=" << Reasons(member.invalidReasons)
               << '\n';
    }
    const auto& p = result.population;
    output << "FEATURE_ABLATION_REPLICATION_SUMMARY"
           << ",declared_pair_count=" << p.declaredPairCount
           << ",complete_comparable_pair_count="
           << p.completeComparablePairCount
           << ",incomplete_pair_count=" << p.incompletePairCount
           << ",invalid_pair_count=" << p.invalidPairCount
           << ",missing_evidence_pair_count=" << p.missingEvidencePairCount
           << ",profitability_unavailable_pair_count="
           << p.profitabilityUnavailablePairCount
           << ",profitability_available_pair_count="
           << p.profitabilityAvailablePairCount
           << ",profitability_positive_pair_count="
           << p.profitabilityPositivePairCount
           << ",profitability_negative_pair_count="
           << p.profitabilityNegativePairCount
           << ",profitability_zero_pair_count="
           << p.profitabilityZeroPairCount
           << ",profitability_positive_pair_fraction="
           << (p.profitabilityAvailablePairCount == 0
                   ? "NULL"
                   : OptionalNumber(static_cast<double>(
                         p.profitabilityPositivePairCount) /
                         static_cast<double>(p.profitabilityAvailablePairCount)))
           << ",distinct_symbol_count=" << p.distinctSymbolCount
           << ",distinct_horizon_count=" << p.distinctHorizonCount << '\n';
    PrintMetric(output, "aggregate_terminal_horizon_log_return_sum_delta",
                result.aggregateProfitability);
    PrintMetric(output,
                "average_terminal_horizon_log_return_per_actionable_prediction_delta",
                result.averageProfitability);
    PrintMetric(output, "inference_accuracy_delta", result.inferenceAccuracy);
    PrintMetric(output, "leader_score_delta", result.leaderScore);
    PrintMetric(output, "neutral_proportion_delta", result.neutralProportion);
    PrintMetric(output, "actionable_count_delta", result.actionableCount);
    output << "FEATURE_ABLATION_REPLICATION_DECISION"
           << ",replication_decision="
           << ReplicationDecisionText(result.decision)
           << ",evidence_integrity_failure="
           << Boolean(result.evidenceIntegrityFailure)
           << ",integrity_reasons=" << Reasons(result.integrityReasons)
           << '\n';
    output << "FEATURE_ABLATION_PRODUCTIONIZATION_GATE"
           << ",productionization_software_ready="
           << Boolean(result.softwareReady)
           << ",software_readiness_version=1"
           << ",software_readiness_hash=" << result.softwareReadinessHash
           << ",replication_decision="
           << ReplicationDecisionText(result.decision)
           << ",productionization_action="
           << ProductionizationActionText(result.action)
           << ",activation_performed=false"
           << ",software_success=true"
           << ",exit_code=" << ExitCode(result)
           << ",read_only=true\n";
    return output.str();
}

int RunComparisonCommand(const std::string& connectionString,
                         const ComparisonCommand& command,
                         std::ostream& output,
                         std::ostream& errors)
{
    (void)errors;
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    std::vector<MemberEvaluation> members;
    members.reserve(command.experimentIdPairs.size());
    for (std::size_t index = 0; index < command.experimentIdPairs.size(); ++index)
        members.push_back(LoadMember(transaction, index + 1,
                                     command.experimentIdPairs[index]));
    const ReplicationEvaluation result = Evaluate(
        std::move(members), command.policy, command.softwareAudit);
    output << RenderComparisonOutput(result);
    return ExitCode(result);
}

} // namespace EA::FeatureAblationReplicationEvaluation
