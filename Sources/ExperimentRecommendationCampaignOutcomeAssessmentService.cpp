#include "ExperimentRecommendationCampaignOutcomeAssessmentService.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignApproval.hpp"
#include "ExperimentRecommendationService.hpp"

#include <algorithm>
#include <locale>
#include <map>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using Consistency =
    RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using MemberEvidence =
    RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;

class StreamStateGuard
{
public:
    explicit StreamStateGuard(std::ostream& stream)
        : stream_(stream), flags_(stream.flags()),
          precision_(stream.precision()), locale_(stream.getloc()) {}

    ~StreamStateGuard()
    {
        try
        {
            stream_.flags(flags_);
            stream_.precision(precision_);
            stream_.imbue(locale_);
        }
        catch (...)
        {
        }
    }

private:
    std::ostream& stream_;
    const std::ios_base::fmtflags flags_;
    const std::streamsize precision_;
    const std::locale locale_;
};

template <typename Value>
void PrintOptional(std::ostream& output, const std::optional<Value>& value)
{
    if (value) output << *value;
    else output << "null";
}

void PrintOptionalNumber(
    std::ostream& output,
    const std::optional<double>& value)
{
    if (value) output << CanonicalRecommendationDouble(*value);
    else output << "null";
}

void PrintSafety(std::ostream& output)
{
    output << "transaction_read_only=true,snapshot_isolation=repeatable_read,"
              "advisory_locks_acquired=0,row_locks_acquired=0,rows_inserted=0,"
              "rows_updated=0,rows_deleted=0,sequences_advanced=0,"
              "scheduler_started=false,scheduler_signaled=false,"
              "scheduler_polled=false,workers_launched=false,"
              "automatic_follow_up=false,campaign_state_row_created=false,"
              "assessment_persisted=false,assessment_advisory=true,"
              "campaign_success_declared=false,follow_up_authorized=false";
}

Consistency InputConsistency(
    const RecommendationCampaignOutcomeAssessmentMember& member)
{
    return std::find(member.diagnostics.begin(), member.diagnostics.end(),
               RecommendationCampaignOutcomeAssessmentDiagnosticCode::
                   InputEvidenceInconsistent) == member.diagnostics.end()
        ? Consistency::Consistent
        : Consistency::Inconsistent;
}

void PrintDiagnostics(
    std::ostream& output,
    const std::vector<
        RecommendationCampaignOutcomeAssessmentDiagnosticCode>& diagnostics)
{
    if (diagnostics.empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < diagnostics.size(); ++index)
    {
        if (index != 0) output << '|';
        output << RecommendationCampaignOutcomeAssessmentDiagnosticCodeText(
            diagnostics[index]);
    }
}

void PrintContext(
    std::ostream& output,
    const Context* context,
    const char* prefix)
{
    output << ',' << prefix << "_symbol=";
    if (context) output << RecommendationMachineText(context->symbol);
    else output << "null";
    output << ',' << prefix << "_prediction_horizon=";
    if (context) output << context->predictionHorizon;
    else output << "null";
    output << ',' << prefix << "_threshold=";
    if (context) output << CanonicalRecommendationDouble(context->threshold);
    else output << "null";
    output << ',' << prefix << "_window_size=";
    if (context) output << context->windowSize;
    else output << "null";
    output << ',' << prefix << "_label_definition=";
    if (context)
        output << RecommendationMachineText(context->labelDefinition);
    else output << "null";
    output << ',' << prefix << "_inference_range_start=";
    if (context)
        output << RecommendationMachineText(context->inferenceRangeStart);
    else output << "null";
    output << ',' << prefix << "_inference_range_end=";
    if (context)
        output << RecommendationMachineText(context->inferenceRangeEnd);
    else output << "null";
}

std::optional<double> MetricValue(
    const MetricCollection* collection,
    const std::string& identity)
{
    if (!collection) return std::nullopt;
    const auto found = std::find_if(collection->metrics.begin(),
        collection->metrics.end(), [&](const auto& metric)
    {
        return metric.identity == identity;
    });
    return found == collection->metrics.end()
        ? std::nullopt
        : found->value;
}

void PrintMetrics(
    std::ostream& output,
    const MetricCollection* metrics)
{
    if (!metrics || metrics->metrics.empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < metrics->metrics.size(); ++index)
    {
        const auto& metric = metrics->metrics[index];
        if (index != 0) output << '|';
        output << RecommendationMachineText(metric.identity) << ':';
        PrintOptionalNumber(output, metric.value);
        output << ':'
               << RecommendationCampaignOutcomeAssessmentMetricComparisonSupportText(
                      metric.comparisonSupport);
    }
}

void PrintResultIdentities(
    std::ostream& output,
    const std::vector<
        RecommendationCampaignOutcomeAssessmentResultIdentity>* identities)
{
    if (!identities || identities->empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < identities->size(); ++index)
    {
        if (index != 0) output << '|';
        output << RecommendationMachineText((*identities)[index].kind)
               << ':' << (*identities)[index].id;
    }
}

void PrintComparisons(
    std::ostream& output,
    const std::vector<
        RecommendationCampaignOutcomeAssessmentMetricComparison>& comparisons)
{
    if (comparisons.empty())
    {
        output << "none";
        return;
    }
    for (std::size_t index = 0; index < comparisons.size(); ++index)
    {
        const auto& comparison = comparisons[index];
        if (index != 0) output << '|';
        output << RecommendationMachineText(comparison.metricIdentity) << ':'
               << RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
                      comparison.classification)
               << ':';
        PrintOptionalNumber(output, comparison.sourceValue);
        output << ':';
        PrintOptionalNumber(output, comparison.resultValue);
        output << ':';
        PrintOptionalNumber(output, comparison.delta);
        output << ':';
        if (comparison.contextDifferences.empty()) output << "none";
        else
        {
            for (std::size_t difference = 0;
                 difference < comparison.contextDifferences.size();
                 ++difference)
            {
                if (difference != 0) output << '+';
                output << RecommendationCampaignOutcomeAssessmentContextDifferenceText(
                    comparison.contextDifferences[difference]);
            }
        }
    }
}

void PrintAssessment(std::ostream& output, const Assessment& assessment)
{
    const auto& summary = assessment.summary;
    output << "RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT"
           << ",campaign_approval_id="
           << assessment.campaignIdentity.campaignApprovalId
           << ",campaign_approval_contract_version="
           << kRecommendationCampaignApprovalContractVersion
           << ",campaign_approval_identity_hash="
           << RecommendationMachineText(
                  assessment.campaignIdentity.identityHash)
           << ",materialization_id="
           << assessment.materializationIdentity.materializationId
           << ",materialization_contract_version="
           << assessment.materializationIdentity.contractVersion
           << ",materialization_campaign_approval_identity_hash="
           << RecommendationMachineText(
                  assessment.materializationIdentity.campaignIdentityHash)
           << ",materialization_identity_hash="
           << RecommendationMachineText(
                  assessment.materializationIdentity.identityHash)
           << ",campaign_status_contract_version="
           << kRecommendationCampaignStatusContractVersion
           << ",assessment_contract_version="
           << assessment.identity.contractVersion
           << ",assessment_identity_hash="
           << RecommendationMachineText(assessment.identity.hash)
           << ",observed_at="
           << RecommendationMachineText(assessment.observedAt)
           << ",aggregate_outcome="
           << RecommendationCampaignOutcomeAssessmentAggregateOutcomeText(
                  summary.outcome)
           << ",member_count=" << summary.memberCount
           << ",lifecycle_not_terminal_count="
           << summary.notTerminalLifecycleCount
           << ",lifecycle_succeeded_count="
           << summary.succeededLifecycleCount
           << ",lifecycle_failed_count=" << summary.failedLifecycleCount
           << ",lifecycle_cancelled_count="
           << summary.cancelledLifecycleCount
           << ",lifecycle_unknown_count=" << summary.unknownLifecycleCount
           << ",consistent_member_count=" << summary.consistentMemberCount
           << ",inconsistent_member_count="
           << summary.inconsistentMemberCount
           << ",not_ready_member_count=" << summary.notReadyMemberCount
           << ",succeeded_comparable_member_count="
           << summary.succeededComparableMemberCount
           << ",succeeded_context_changed_member_count="
           << summary.succeededContextChangedMemberCount
           << ",succeeded_metric_gap_member_count="
           << summary.succeededMetricGapMemberCount
           << ",terminal_failed_member_count="
           << summary.terminalFailedMemberCount
           << ",terminal_cancelled_member_count="
           << summary.terminalCancelledMemberCount
           << ",inconsistent_outcome_member_count="
           << summary.inconsistentOutcomeMemberCount
           << ",metric_comparison_count=" << summary.metricComparisonCount
           << ",comparable_metric_count=" << summary.comparableMetricCount
           << ",context_changed_metric_count="
           << summary.contextChangedMetricCount
           << ",missing_source_metric_count="
           << summary.missingSourceMetricCount
           << ",missing_result_metric_count="
           << summary.missingResultMetricCount
           << ",metric_value_unavailable_count="
           << summary.metricValueUnavailableCount
           << ",unsupported_metric_count=" << summary.unsupportedMetricCount
           << ",positive_delta_count=" << summary.positiveDeltaCount
           << ",zero_delta_count=" << summary.zeroDeltaCount
           << ",negative_delta_count=" << summary.negativeDeltaCount << ',';
    PrintSafety(output);
    output << '\n';

    for (const auto& member : assessment.members)
    {
        const auto* source = member.sourceEvidence
            ? &*member.sourceEvidence : nullptr;
        const auto* result = member.resultEvidence
            ? &*member.resultEvidence : nullptr;
        output << "RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT_MEMBER"
               << ",materialization_id="
               << assessment.materializationIdentity.materializationId
               << ",member_ordinal=" << member.identity.memberOrdinal
               << ",materialization_member_id="
               << member.identity.materializationMemberId
               << ",ranking_member_id=" << member.identity.rankingMemberId
               << ",recommendation_id=" << member.identity.recommendationId
               << ",source_experiment_id="
               << member.identity.sourceExperimentId
               << ",proposal_id=" << member.identity.proposalId
               << ",expected_experiment_id=";
        PrintOptional(output, member.identity.expectedExperimentId);
        output << ",lifecycle="
               << RecommendationCampaignOutcomeAssessmentLifecycleStateText(
                      member.lifecycle)
               << ",input_consistency="
               << RecommendationCampaignOutcomeAssessmentConsistencyStateText(
                      InputConsistency(member))
               << ",final_consistency="
               << RecommendationCampaignOutcomeAssessmentConsistencyStateText(
                      member.consistency)
               << ",outcome="
               << RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
                      member.outcome)
               << ",diagnostic=";
        PrintDiagnostics(output, member.diagnostics);
        output << ",source_evidence_present=" << (source ? "true" : "false")
               << ",source_model_id=";
        if (source) PrintOptional(output, source->sourceModelId);
        else output << "null";
        output << ",source_analysis_id=";
        if (source) PrintOptional(output, source->sourceAnalysisId);
        else output << "null";
        PrintContext(output, source ? &source->context : nullptr, "source");
        output << ",source_inference_accuracy=";
        PrintOptionalNumber(output, MetricValue(
            source ? &source->metrics : nullptr, "inference_accuracy"));
        output << ",source_leader_score=";
        PrintOptionalNumber(output, MetricValue(
            source ? &source->metrics : nullptr, "leader_score"));
        output << ",source_metrics=";
        PrintMetrics(output, source ? &source->metrics : nullptr);
        output << ",result_evidence_present=" << (result ? "true" : "false")
               << ",result_experiment_id=";
        if (result) PrintOptional(output, result->experimentId);
        else output << "null";
        output << ",result_model_id=";
        if (result) PrintOptional(output, result->modelId);
        else output << "null";
        output << ",result_identity_count="
               << (result ? result->resultIdentities.size() : 0)
               << ",result_identities=";
        PrintResultIdentities(output,
            result ? &result->resultIdentities : nullptr);
        output << ",result_context_present="
               << (result && result->context ? "true" : "false");
        PrintContext(output,
            result && result->context ? &*result->context : nullptr,
            "result");
        output << ",result_inference_accuracy=";
        PrintOptionalNumber(output, MetricValue(
            result ? &result->metrics : nullptr, "inference_accuracy"));
        output << ",result_leader_score=";
        PrintOptionalNumber(output, MetricValue(
            result ? &result->metrics : nullptr, "leader_score"));
        output << ",result_metrics=";
        PrintMetrics(output, result ? &result->metrics : nullptr);
        output << ",comparison_count=" << member.comparisons.size()
               << ",comparisons=";
        PrintComparisons(output, member.comparisons);
        output << ',';
        PrintSafety(output);
        output << '\n';
    }
}

} // namespace

RecommendationCampaignOutcomeAssessmentLifecycleState
RecommendationCampaignOutcomeAssessmentLifecycleFromCampaignStatus(
    RecommendationCampaignStatusTerminalResult terminalResult)
{
    switch (terminalResult)
    {
        case RecommendationCampaignStatusTerminalResult::notTerminal:
            return Lifecycle::NotTerminal;
        case RecommendationCampaignStatusTerminalResult::succeeded:
            return Lifecycle::Succeeded;
        case RecommendationCampaignStatusTerminalResult::failed:
            return Lifecycle::Failed;
        case RecommendationCampaignStatusTerminalResult::cancelled:
            return Lifecycle::Cancelled;
        case RecommendationCampaignStatusTerminalResult::unknown:
            return Lifecycle::Unknown;
    }
    throw std::invalid_argument(
        "campaign_outcome_assessment_status_lifecycle_invalid");
}

RecommendationCampaignOutcomeAssessment
BuildRecommendationCampaignOutcomeAssessmentFromEvidence(
    const RecommendationCampaignOutcomeAssessmentEvidenceSnapshot& evidence)
{
    if (evidence.statusSnapshot.request.materializationId !=
            evidence.materializationIdentity.materializationId ||
        evidence.statusSnapshot.materializationContractVersion !=
            evidence.materializationIdentity.contractVersion ||
        evidence.statusSnapshot.materializationIdentityHash !=
            evidence.materializationIdentity.identityHash ||
        evidence.statusSnapshot.members.size() !=
            evidence.scientificMembers.size())
        throw std::invalid_argument(
            "campaign_outcome_assessment_evidence_snapshot_mismatch");

    std::map<int, const
        RecommendationCampaignOutcomeAssessmentScientificMemberEvidence*>
        scientificByOrdinal;
    for (const auto& member : evidence.scientificMembers)
        if (member.memberOrdinal <= 0 ||
            !scientificByOrdinal.emplace(
                member.memberOrdinal, &member).second)
            throw std::invalid_argument(
                "campaign_outcome_assessment_scientific_member_invalid");

    std::vector<MemberEvidence> members;
    members.reserve(evidence.statusSnapshot.members.size());
    for (const auto& member : evidence.statusSnapshot.members)
    {
        const auto scientific = scientificByOrdinal.find(
            member.memberOrdinal);
        if (scientific == scientificByOrdinal.end())
            throw std::invalid_argument(
                "campaign_outcome_assessment_scientific_member_missing");
        const bool inconsistent = member.consistency ==
                RecommendationCampaignStatusConsistency::inconsistent ||
            scientific->second->inputConsistency ==
                Consistency::Inconsistent;
        members.emplace_back(
            RecommendationCampaignOutcomeAssessmentMemberIdentity(
                member.memberOrdinal, member.materializationMemberId,
                member.rankingMemberId, member.recommendationId,
                member.sourceExperimentId, member.proposalId,
                member.experimentId),
            RecommendationCampaignOutcomeAssessmentLifecycleFromCampaignStatus(
                member.terminalResult),
            scientific->second->sourceEvidence,
            scientific->second->resultEvidence,
            inconsistent ? Consistency::Inconsistent
                         : Consistency::Consistent);
    }
    return BuildRecommendationCampaignOutcomeAssessment(
        evidence.campaignIdentity, evidence.materializationIdentity,
        evidence.statusSnapshot.observedAt, members);
}

void WriteRecommendationCampaignOutcomeAssessment(
    std::ostream& output,
    const RecommendationCampaignOutcomeAssessment& assessment)
{
    StreamStateGuard state{output};
    std::ostringstream defaults;
    output.flags(defaults.flags());
    output.precision(defaults.precision());
    output.imbue(std::locale::classic());
    PrintAssessment(output, assessment);
}

int RunRecommendationCampaignOutcomeAssessmentCommand(
    const std::string& connectionString,
    const RecommendationCampaignOutcomeAssessmentRequest& request,
    std::ostream& output,
    std::ostream& errors)
{
    try
    {
        const auto normalized =
            NormalizeRecommendationCampaignOutcomeAssessmentRequest(request);
        pqxx::connection connection{connectionString};
        const auto evidence =
            ReadRecommendationCampaignOutcomeAssessmentEvidence(
                connection, normalized);
        WriteRecommendationCampaignOutcomeAssessment(output,
            BuildRecommendationCampaignOutcomeAssessmentFromEvidence(
                evidence));
        return 0;
    }
    catch (const std::exception& error)
    {
        errors << "RECOMMENDATION_CAMPAIGN_OUTCOME_ASSESSMENT_FAILED"
               << ",campaign_approval_id=null,"
                  "campaign_approval_contract_version="
               << kRecommendationCampaignApprovalContractVersion
               << ",campaign_approval_identity_hash=null,materialization_id="
               << request.materializationId
               << ",materialization_contract_version=null,"
                  "materialization_campaign_approval_identity_hash=null,"
                  "materialization_identity_hash=null,"
                  "campaign_status_contract_version="
               << kRecommendationCampaignStatusContractVersion
               << ","
                  "assessment_contract_version="
               << kRecommendationCampaignOutcomeAssessmentContractVersion
               << ",assessment_identity_hash=null,observed_at=null,"
                  "aggregate_outcome=inconsistent,diagnostic="
               << RecommendationMachineText(error.what())
               << ",member_count=0,";
        PrintSafety(errors);
        errors << '\n';
        return 1;
    }
}

} // namespace EA::ExperimentRecommendation
