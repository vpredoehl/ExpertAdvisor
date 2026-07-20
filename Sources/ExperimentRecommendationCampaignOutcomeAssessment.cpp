#include "ExperimentRecommendationCampaignOutcomeAssessment.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <cmath>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using AggregateOutcome =
    RecommendationCampaignOutcomeAssessmentAggregateOutcome;
using Classification =
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification;
using Comparison =
    RecommendationCampaignOutcomeAssessmentMetricComparison;
using Consistency = RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Diagnostic = RecommendationCampaignOutcomeAssessmentDiagnosticCode;
using Difference = RecommendationCampaignOutcomeAssessmentContextDifference;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using Member = RecommendationCampaignOutcomeAssessmentMember;
using MemberEvidence = RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MemberOutcome = RecommendationCampaignOutcomeAssessmentMemberOutcome;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;

void RequireText(const std::string& value, const char* error)
{
    if (value.empty() || value.find('\0') != std::string::npos)
        throw std::invalid_argument(error);
}

void RequireOptionalPositive(
    const std::optional<long long>& value,
    const char* error)
{
    if (value && *value <= 0) throw std::invalid_argument(error);
}

void RequireValidMetricSupport(MetricSupport value)
{
    switch (value)
    {
        case MetricSupport::NumericDelta:
        case MetricSupport::Unsupported:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_assessment_metric_support_invalid");
}

void RequireValidLifecycle(Lifecycle value)
{
    switch (value)
    {
        case Lifecycle::NotTerminal:
        case Lifecycle::Succeeded:
        case Lifecycle::Failed:
        case Lifecycle::Cancelled:
        case Lifecycle::Unknown:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_assessment_lifecycle_invalid");
}

void RequireValidConsistency(Consistency value)
{
    switch (value)
    {
        case Consistency::Consistent:
        case Consistency::Inconsistent:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_assessment_consistency_invalid");
}

std::vector<Metric> CanonicalMetrics(std::vector<Metric> metrics)
{
    std::vector<const Metric*> ordered;
    ordered.reserve(metrics.size());
    for (const auto& metric : metrics) ordered.push_back(&metric);
    std::sort(ordered.begin(), ordered.end(), [](const Metric* left,
                                              const Metric* right)
    {
        return left->identity < right->identity;
    });
    for (std::size_t index = 1; index < ordered.size(); ++index)
        if (ordered[index - 1]->identity == ordered[index]->identity)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_metric_identity_duplicate");
    std::vector<Metric> canonical;
    canonical.reserve(ordered.size());
    for (const auto* metric : ordered) canonical.push_back(*metric);
    return canonical;
}

std::vector<ResultIdentity> CanonicalResultIdentities(
    std::vector<ResultIdentity> identities)
{
    std::vector<const ResultIdentity*> ordered;
    ordered.reserve(identities.size());
    for (const auto& identity : identities) ordered.push_back(&identity);
    std::sort(ordered.begin(), ordered.end(),
        [](const ResultIdentity* left, const ResultIdentity* right)
    {
        return std::tie(left->kind, left->id) <
            std::tie(right->kind, right->id);
    });
    for (std::size_t index = 1; index < ordered.size(); ++index)
        if (*ordered[index - 1] == *ordered[index])
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_result_identity_duplicate");
    std::vector<ResultIdentity> canonical;
    canonical.reserve(ordered.size());
    for (const auto* identity : ordered) canonical.push_back(*identity);
    return canonical;
}

std::string LengthPrefixed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalLongLong(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string OptionalDouble(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "null";
}

void AppendContext(std::ostringstream& out, const Context& context)
{
    out << LengthPrefixed(context.symbol) << ',' << context.predictionHorizon
        << ',' << CanonicalRecommendationDouble(context.threshold) << ','
        << context.windowSize << ',' << LengthPrefixed(context.labelDefinition)
        << ',' << LengthPrefixed(context.inferenceRangeStart) << ','
        << LengthPrefixed(context.inferenceRangeEnd);
}

void AppendOptionalContext(
    std::ostringstream& out,
    const std::optional<Context>& context)
{
    if (!context)
    {
        out << "absent";
        return;
    }
    out << "present,";
    AppendContext(out, *context);
}

void AppendMetric(std::ostringstream& out, const Metric& metric)
{
    out << LengthPrefixed(metric.identity) << ','
        << OptionalDouble(metric.value) << ','
        << RecommendationCampaignOutcomeAssessmentMetricComparisonSupportText(
               metric.comparisonSupport);
}

void AppendMetrics(std::ostringstream& out, const MetricCollection& collection)
{
    out << collection.metrics.size();
    for (const auto& metric : collection.metrics)
    {
        out << ',';
        AppendMetric(out, metric);
    }
}

void AppendSourceEvidence(
    std::ostringstream& out,
    const std::optional<
        RecommendationCampaignOutcomeAssessmentSourceEvidence>& evidence)
{
    if (!evidence)
    {
        out << "absent";
        return;
    }
    out << "present," << evidence->sourceExperimentId << ','
        << OptionalLongLong(evidence->sourceModelId) << ','
        << OptionalLongLong(evidence->sourceAnalysisId) << ",context=";
    AppendContext(out, evidence->context);
    out << ",metrics=";
    AppendMetrics(out, evidence->metrics);
}

void AppendResultEvidence(
    std::ostringstream& out,
    const std::optional<
        RecommendationCampaignOutcomeAssessmentResultEvidence>& evidence)
{
    if (!evidence)
    {
        out << "absent";
        return;
    }
    out << "present," << OptionalLongLong(evidence->experimentId) << ','
        << OptionalLongLong(evidence->modelId)
        << ",result_identity_count=" << evidence->resultIdentities.size();
    for (const auto& identity : evidence->resultIdentities)
        out << ',' << LengthPrefixed(identity.kind) << ',' << identity.id;
    out << ",context=";
    AppendOptionalContext(out, evidence->context);
    out << ",metrics=";
    AppendMetrics(out, evidence->metrics);
}

void AppendComparison(std::ostringstream& out, const Comparison& comparison)
{
    out << LengthPrefixed(comparison.metricIdentity) << ','
        << RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
               comparison.classification)
        << ",source_value=" << OptionalDouble(comparison.sourceValue)
        << ",result_value=" << OptionalDouble(comparison.resultValue)
        << ",delta=" << OptionalDouble(comparison.delta)
        << ",context_difference_count="
        << comparison.contextDifferences.size();
    for (const auto difference : comparison.contextDifferences)
        out << ','
            << RecommendationCampaignOutcomeAssessmentContextDifferenceText(
                   difference);
}

void AppendSummary(
    std::ostringstream& out,
    const RecommendationCampaignOutcomeAssessmentSummary& summary)
{
    out << ";aggregate_outcome="
        << RecommendationCampaignOutcomeAssessmentAggregateOutcomeText(
               summary.outcome)
        << ";member_count=" << summary.memberCount
        << ";lifecycle_not_terminal_count="
        << summary.notTerminalLifecycleCount
        << ";lifecycle_succeeded_count=" << summary.succeededLifecycleCount
        << ";lifecycle_failed_count=" << summary.failedLifecycleCount
        << ";lifecycle_cancelled_count=" << summary.cancelledLifecycleCount
        << ";lifecycle_unknown_count=" << summary.unknownLifecycleCount
        << ";consistent_member_count=" << summary.consistentMemberCount
        << ";inconsistent_member_count=" << summary.inconsistentMemberCount
        << ";not_ready_member_count=" << summary.notReadyMemberCount
        << ";succeeded_comparable_member_count="
        << summary.succeededComparableMemberCount
        << ";succeeded_context_changed_member_count="
        << summary.succeededContextChangedMemberCount
        << ";succeeded_metric_gap_member_count="
        << summary.succeededMetricGapMemberCount
        << ";terminal_failed_member_count="
        << summary.terminalFailedMemberCount
        << ";terminal_cancelled_member_count="
        << summary.terminalCancelledMemberCount
        << ";inconsistent_outcome_member_count="
        << summary.inconsistentOutcomeMemberCount
        << ";metric_comparison_count=" << summary.metricComparisonCount
        << ";comparable_metric_count=" << summary.comparableMetricCount
        << ";context_changed_metric_count="
        << summary.contextChangedMetricCount
        << ";missing_source_metric_count="
        << summary.missingSourceMetricCount
        << ";missing_result_metric_count="
        << summary.missingResultMetricCount
        << ";metric_value_unavailable_count="
        << summary.metricValueUnavailableCount
        << ";unsupported_metric_count=" << summary.unsupportedMetricCount
        << ";positive_delta_count=" << summary.positiveDeltaCount
        << ";zero_delta_count=" << summary.zeroDeltaCount
        << ";negative_delta_count=" << summary.negativeDeltaCount;
}

std::string AssessmentCanonicalText(
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity& campaign,
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
        materialization,
    const RecommendationCampaignOutcomeAssessmentSummary& summary,
    const std::vector<Member>& members)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_outcome_assessment_v2"
        << ";assessment_contract_version="
        << kRecommendationCampaignOutcomeAssessmentContractVersion
        << ";campaign_approval_id=" << campaign.campaignApprovalId
        << ";campaign_identity="
        << LengthPrefixed(campaign.identityCanonical)
        << ";campaign_identity_hash=" << LengthPrefixed(campaign.identityHash)
        << ";materialization_id=" << materialization.materializationId
        << ";materialization_contract_version="
        << materialization.contractVersion
        << ";materialization_identity="
        << LengthPrefixed(materialization.identityCanonical)
        << ";materialization_identity_hash="
        << LengthPrefixed(materialization.identityHash);
    for (const auto& member : members)
    {
        out << ";member=" << member.identity.memberOrdinal << ','
            << member.identity.materializationMemberId << ','
            << member.identity.rankingMemberId << ','
            << member.identity.recommendationId << ','
            << member.identity.sourceExperimentId << ','
            << member.identity.proposalId << ','
            << OptionalLongLong(member.identity.expectedExperimentId)
            << ",lifecycle="
            << RecommendationCampaignOutcomeAssessmentLifecycleStateText(
                   member.lifecycle)
            << ",consistency="
            << RecommendationCampaignOutcomeAssessmentConsistencyStateText(
                   member.consistency)
            << ",outcome="
            << RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
                   member.outcome)
            << ",diagnostic_count=" << member.diagnostics.size();
        for (const auto diagnostic : member.diagnostics)
            out << ','
                << RecommendationCampaignOutcomeAssessmentDiagnosticCodeText(
                       diagnostic);
        out << ",source=";
        AppendSourceEvidence(out, member.sourceEvidence);
        out << ",result=";
        AppendResultEvidence(out, member.resultEvidence);
        out << ",comparison_count=" << member.comparisons.size();
        for (const auto& comparison : member.comparisons)
        {
            out << ',';
            AppendComparison(out, comparison);
        }
    }
    AppendSummary(out, summary);
    return out.str();
}

std::vector<Difference> ContextDifferences(
    const Context& source,
    const Context& result)
{
    std::vector<Difference> differences;
    if (source.symbol != result.symbol)
        differences.push_back(Difference::Symbol);
    if (source.predictionHorizon != result.predictionHorizon)
        differences.push_back(Difference::PredictionHorizon);
    if (source.threshold != result.threshold)
        differences.push_back(Difference::Threshold);
    if (source.windowSize != result.windowSize)
        differences.push_back(Difference::WindowSize);
    if (source.labelDefinition != result.labelDefinition)
        differences.push_back(Difference::LabelDefinition);
    if (source.inferenceRangeStart != result.inferenceRangeStart ||
        source.inferenceRangeEnd != result.inferenceRangeEnd)
        differences.push_back(Difference::InferenceRange);
    return differences;
}

void CanonicalizeDiagnostics(std::vector<Diagnostic>& diagnostics)
{
    std::sort(diagnostics.begin(), diagnostics.end(),
        [](Diagnostic left, Diagnostic right)
    {
        return static_cast<int>(left) < static_cast<int>(right);
    });
    diagnostics.erase(
        std::unique(diagnostics.begin(), diagnostics.end()),
        diagnostics.end());
}

struct WorkingMember
{
    const MemberEvidence* evidence;
    std::vector<Diagnostic> diagnostics;
};

struct MutableSummary
{
    int memberCount = 0;
    int notTerminalLifecycleCount = 0;
    int succeededLifecycleCount = 0;
    int failedLifecycleCount = 0;
    int cancelledLifecycleCount = 0;
    int unknownLifecycleCount = 0;
    int consistentMemberCount = 0;
    int inconsistentMemberCount = 0;
    int notReadyMemberCount = 0;
    int succeededComparableMemberCount = 0;
    int succeededContextChangedMemberCount = 0;
    int succeededMetricGapMemberCount = 0;
    int terminalFailedMemberCount = 0;
    int terminalCancelledMemberCount = 0;
    int inconsistentOutcomeMemberCount = 0;
    std::size_t metricComparisonCount = 0;
    std::size_t comparableMetricCount = 0;
    std::size_t contextChangedMetricCount = 0;
    std::size_t missingSourceMetricCount = 0;
    std::size_t missingResultMetricCount = 0;
    std::size_t metricValueUnavailableCount = 0;
    std::size_t unsupportedMetricCount = 0;
    std::size_t positiveDeltaCount = 0;
    std::size_t zeroDeltaCount = 0;
    std::size_t negativeDeltaCount = 0;
};

AggregateOutcome DeriveAggregateOutcome(const MutableSummary& counts)
{
    if (counts.inconsistentMemberCount > 0)
        return AggregateOutcome::Inconsistent;
    if (counts.terminalFailedMemberCount > 0)
        return AggregateOutcome::HasFailures;
    if (counts.terminalCancelledMemberCount > 0)
        return AggregateOutcome::HasCancellations;
    if (counts.notReadyMemberCount > 0)
        return AggregateOutcome::NotReady;
    if (counts.succeededContextChangedMemberCount > 0)
        return AggregateOutcome::SucceededContextChanged;
    if (counts.succeededMetricGapMemberCount > 0)
        return AggregateOutcome::SucceededMetricGap;
    return AggregateOutcome::SucceededComparable;
}

void CountMember(MutableSummary& counts, const Member& member)
{
    ++counts.memberCount;
    switch (member.lifecycle)
    {
        case Lifecycle::NotTerminal:
            ++counts.notTerminalLifecycleCount;
            break;
        case Lifecycle::Succeeded:
            ++counts.succeededLifecycleCount;
            break;
        case Lifecycle::Failed:
            ++counts.failedLifecycleCount;
            break;
        case Lifecycle::Cancelled:
            ++counts.cancelledLifecycleCount;
            break;
        case Lifecycle::Unknown:
            ++counts.unknownLifecycleCount;
            break;
    }
    if (member.consistency == Consistency::Consistent)
        ++counts.consistentMemberCount;
    else
        ++counts.inconsistentMemberCount;

    switch (member.outcome)
    {
        case MemberOutcome::NotReady:
            ++counts.notReadyMemberCount;
            break;
        case MemberOutcome::SucceededComparable:
            ++counts.succeededComparableMemberCount;
            break;
        case MemberOutcome::SucceededContextChanged:
            ++counts.succeededContextChangedMemberCount;
            break;
        case MemberOutcome::SucceededMetricGap:
            ++counts.succeededMetricGapMemberCount;
            break;
        case MemberOutcome::TerminalFailed:
            ++counts.terminalFailedMemberCount;
            break;
        case MemberOutcome::TerminalCancelled:
            ++counts.terminalCancelledMemberCount;
            break;
        case MemberOutcome::Inconsistent:
            ++counts.inconsistentOutcomeMemberCount;
            break;
    }

    for (const auto& comparison : member.comparisons)
    {
        ++counts.metricComparisonCount;
        switch (comparison.classification)
        {
            case Classification::Comparable:
                ++counts.comparableMetricCount;
                if (*comparison.delta > 0.0)
                    ++counts.positiveDeltaCount;
                else if (*comparison.delta < 0.0)
                    ++counts.negativeDeltaCount;
                else
                    ++counts.zeroDeltaCount;
                break;
            case Classification::ContextChanged:
                ++counts.contextChangedMetricCount;
                break;
            case Classification::MissingSourceMetric:
                ++counts.missingSourceMetricCount;
                break;
            case Classification::MissingResultMetric:
                ++counts.missingResultMetricCount;
                break;
            case Classification::MetricValueUnavailable:
                ++counts.metricValueUnavailableCount;
                break;
            case Classification::Unsupported:
                ++counts.unsupportedMetricCount;
                break;
        }
    }
}

} // namespace

RecommendationCampaignOutcomeAssessmentCampaignIdentity::
    RecommendationCampaignOutcomeAssessmentCampaignIdentity(
        long long campaignApprovalIdValue,
        std::string identityCanonicalValue,
        std::string identityHashValue)
    : campaignApprovalId(campaignApprovalIdValue),
      identityCanonical(std::move(identityCanonicalValue)),
      identityHash(std::move(identityHashValue))
{
    if (campaignApprovalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_campaign_id_invalid");
    RequireText(identityCanonical,
        "recommendation_campaign_outcome_assessment_campaign_identity_missing");
    if (identityHash != RecommendationCanonicalHash(identityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_campaign_identity_invalid");
}

RecommendationCampaignOutcomeAssessmentMaterializationIdentity::
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
        long long materializationIdValue,
        long long campaignApprovalIdValue,
        std::string campaignIdentityHashValue,
        int contractVersionValue,
        int memberCountValue,
        std::string identityCanonicalValue,
        std::string identityHashValue)
    : materializationId(materializationIdValue),
      campaignApprovalId(campaignApprovalIdValue),
      campaignIdentityHash(std::move(campaignIdentityHashValue)),
      contractVersion(contractVersionValue),
      memberCount(memberCountValue),
      identityCanonical(std::move(identityCanonicalValue)),
      identityHash(std::move(identityHashValue))
{
    if (materializationId <= 0 || campaignApprovalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_materialization_id_invalid");
    if (memberCount <= 0 ||
        memberCount > kMaximumRecommendationCampaignOutcomeAssessmentMembers)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_materialization_member_count_invalid");
    RequireText(campaignIdentityHash,
        "recommendation_campaign_outcome_assessment_campaign_identity_hash_missing");
    RequireText(identityCanonical,
        "recommendation_campaign_outcome_assessment_materialization_identity_missing");
    if (identityHash != RecommendationCanonicalHash(identityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_materialization_identity_invalid");
}

RecommendationCampaignOutcomeAssessmentMemberIdentity::
    RecommendationCampaignOutcomeAssessmentMemberIdentity(
        int memberOrdinalValue,
        long long materializationMemberIdValue,
        long long rankingMemberIdValue,
        long long recommendationIdValue,
        long long sourceExperimentIdValue,
        long long proposalIdValue,
        std::optional<long long> expectedExperimentIdValue)
    : memberOrdinal(memberOrdinalValue),
      materializationMemberId(materializationMemberIdValue),
      rankingMemberId(rankingMemberIdValue),
      recommendationId(recommendationIdValue),
      sourceExperimentId(sourceExperimentIdValue),
      proposalId(proposalIdValue),
      expectedExperimentId(expectedExperimentIdValue)
{
    if (memberOrdinal <= 0 || materializationMemberId <= 0 ||
        rankingMemberId <= 0 || recommendationId <= 0 ||
        sourceExperimentId <= 0 || proposalId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_member_identity_invalid");
    RequireOptionalPositive(expectedExperimentId,
        "recommendation_campaign_outcome_assessment_expected_experiment_id_invalid");
}

RecommendationCampaignOutcomeAssessmentMetric::
    RecommendationCampaignOutcomeAssessmentMetric(
        std::string identityValue,
        std::optional<double> valueValue,
        MetricSupport comparisonSupportValue)
    : identity(std::move(identityValue)),
      value(valueValue && *valueValue == 0.0
              ? std::optional<double>(0.0)
              : valueValue),
      comparisonSupport(comparisonSupportValue)
{
    RequireText(identity,
        "recommendation_campaign_outcome_assessment_metric_identity_invalid");
    if (value && !std::isfinite(*value))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_metric_value_nonfinite");
    RequireValidMetricSupport(comparisonSupport);
}

RecommendationCampaignOutcomeAssessmentMetricCollection::
    RecommendationCampaignOutcomeAssessmentMetricCollection(
        std::vector<Metric> metricsValue)
    : metrics(CanonicalMetrics(std::move(metricsValue)))
{
}

RecommendationCampaignOutcomeAssessmentComparisonContext::
    RecommendationCampaignOutcomeAssessmentComparisonContext(
        std::string symbolValue,
        int predictionHorizonValue,
        double thresholdValue,
        int windowSizeValue,
        std::string labelDefinitionValue,
        std::string inferenceRangeStartValue,
        std::string inferenceRangeEndValue)
    : symbol(std::move(symbolValue)),
      predictionHorizon(predictionHorizonValue),
      threshold(thresholdValue == 0.0 ? 0.0 : thresholdValue),
      windowSize(windowSizeValue),
      labelDefinition(std::move(labelDefinitionValue)),
      inferenceRangeStart(std::move(inferenceRangeStartValue)),
      inferenceRangeEnd(std::move(inferenceRangeEndValue))
{
    RequireText(symbol,
        "recommendation_campaign_outcome_assessment_context_symbol_invalid");
    RequireText(labelDefinition,
        "recommendation_campaign_outcome_assessment_context_label_definition_invalid");
    RequireText(inferenceRangeStart,
        "recommendation_campaign_outcome_assessment_context_inference_range_invalid");
    RequireText(inferenceRangeEnd,
        "recommendation_campaign_outcome_assessment_context_inference_range_invalid");
    if (predictionHorizon <= 0 || windowSize <= 0 || !std::isfinite(threshold))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_context_invalid");
}

RecommendationCampaignOutcomeAssessmentSourceEvidence::
    RecommendationCampaignOutcomeAssessmentSourceEvidence(
        long long sourceExperimentIdValue,
        std::optional<long long> sourceModelIdValue,
        std::optional<long long> sourceAnalysisIdValue,
        Context contextValue,
        MetricCollection metricsValue)
    : sourceExperimentId(sourceExperimentIdValue),
      sourceModelId(sourceModelIdValue),
      sourceAnalysisId(sourceAnalysisIdValue),
      context(std::move(contextValue)),
      metrics(std::move(metricsValue))
{
    if (sourceExperimentId <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_source_experiment_id_invalid");
    RequireOptionalPositive(sourceModelId,
        "recommendation_campaign_outcome_assessment_source_model_id_invalid");
    RequireOptionalPositive(sourceAnalysisId,
        "recommendation_campaign_outcome_assessment_source_analysis_id_invalid");
}

RecommendationCampaignOutcomeAssessmentResultIdentity::
    RecommendationCampaignOutcomeAssessmentResultIdentity(
        std::string kindValue,
        long long idValue)
    : kind(std::move(kindValue)), id(idValue)
{
    RequireText(kind,
        "recommendation_campaign_outcome_assessment_result_identity_kind_invalid");
    if (id <= 0)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_result_identity_id_invalid");
}

RecommendationCampaignOutcomeAssessmentResultEvidence::
    RecommendationCampaignOutcomeAssessmentResultEvidence(
        std::optional<long long> experimentIdValue,
        std::optional<long long> modelIdValue,
        std::vector<ResultIdentity> resultIdentitiesValue,
        std::optional<Context> contextValue,
        MetricCollection metricsValue)
    : experimentId(experimentIdValue),
      modelId(modelIdValue),
      resultIdentities(
          CanonicalResultIdentities(std::move(resultIdentitiesValue))),
      context(std::move(contextValue)),
      metrics(std::move(metricsValue))
{
    RequireOptionalPositive(experimentId,
        "recommendation_campaign_outcome_assessment_result_experiment_id_invalid");
    RequireOptionalPositive(modelId,
        "recommendation_campaign_outcome_assessment_result_model_id_invalid");
}

RecommendationCampaignOutcomeAssessmentMemberEvidence::
    RecommendationCampaignOutcomeAssessmentMemberEvidence(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identityValue,
        Lifecycle lifecycleValue,
        std::optional<
            RecommendationCampaignOutcomeAssessmentSourceEvidence>
            sourceEvidenceValue,
        std::optional<
            RecommendationCampaignOutcomeAssessmentResultEvidence>
            resultEvidenceValue,
        Consistency inputConsistencyValue)
    : identity(std::move(identityValue)),
      lifecycle(lifecycleValue),
      inputConsistency(inputConsistencyValue),
      sourceEvidence(std::move(sourceEvidenceValue)),
      resultEvidence(std::move(resultEvidenceValue))
{
    RequireValidLifecycle(lifecycle);
    RequireValidConsistency(inputConsistency);
}

RecommendationCampaignOutcomeAssessmentMetricComparison::
    RecommendationCampaignOutcomeAssessmentMetricComparison(
        std::string metricIdentityValue,
        Classification classificationValue,
        std::optional<double> sourceValueValue,
        std::optional<double> resultValueValue,
        std::optional<double> deltaValue,
        std::vector<Difference> contextDifferencesValue)
    : metricIdentity(std::move(metricIdentityValue)),
      classification(classificationValue),
      sourceValue(sourceValueValue),
      resultValue(resultValueValue),
      delta(deltaValue && *deltaValue == 0.0
              ? std::optional<double>(0.0)
              : deltaValue),
      contextDifferences(std::move(contextDifferencesValue))
{
    RequireText(metricIdentity,
        "recommendation_campaign_outcome_assessment_comparison_metric_identity_invalid");
    if ((sourceValue && !std::isfinite(*sourceValue)) ||
        (resultValue && !std::isfinite(*resultValue)) ||
        (delta && !std::isfinite(*delta)))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_comparison_number_nonfinite");
    const bool changed = classification == Classification::ContextChanged;
    if (changed != !contextDifferences.empty())
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_comparison_context_invalid");
    if (classification == Classification::Comparable)
    {
        if (!sourceValue || !resultValue || !delta)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_comparable_invalid");
    }
    else if (delta)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_assessment_noncomparable_delta_invalid");
}

RecommendationCampaignOutcomeAssessmentSummary::
    RecommendationCampaignOutcomeAssessmentSummary(
        AggregateOutcome outcomeValue,
        int memberCountValue,
        int notTerminalLifecycleCountValue,
        int succeededLifecycleCountValue,
        int failedLifecycleCountValue,
        int cancelledLifecycleCountValue,
        int unknownLifecycleCountValue,
        int consistentMemberCountValue,
        int inconsistentMemberCountValue,
        int notReadyMemberCountValue,
        int succeededComparableMemberCountValue,
        int succeededContextChangedMemberCountValue,
        int succeededMetricGapMemberCountValue,
        int terminalFailedMemberCountValue,
        int terminalCancelledMemberCountValue,
        int inconsistentOutcomeMemberCountValue,
        std::size_t metricComparisonCountValue,
        std::size_t comparableMetricCountValue,
        std::size_t contextChangedMetricCountValue,
        std::size_t missingSourceMetricCountValue,
        std::size_t missingResultMetricCountValue,
        std::size_t metricValueUnavailableCountValue,
        std::size_t unsupportedMetricCountValue,
        std::size_t positiveDeltaCountValue,
        std::size_t zeroDeltaCountValue,
        std::size_t negativeDeltaCountValue)
    : outcome(outcomeValue),
      memberCount(memberCountValue),
      notTerminalLifecycleCount(notTerminalLifecycleCountValue),
      succeededLifecycleCount(succeededLifecycleCountValue),
      failedLifecycleCount(failedLifecycleCountValue),
      cancelledLifecycleCount(cancelledLifecycleCountValue),
      unknownLifecycleCount(unknownLifecycleCountValue),
      consistentMemberCount(consistentMemberCountValue),
      inconsistentMemberCount(inconsistentMemberCountValue),
      notReadyMemberCount(notReadyMemberCountValue),
      succeededComparableMemberCount(succeededComparableMemberCountValue),
      succeededContextChangedMemberCount(
          succeededContextChangedMemberCountValue),
      succeededMetricGapMemberCount(succeededMetricGapMemberCountValue),
      terminalFailedMemberCount(terminalFailedMemberCountValue),
      terminalCancelledMemberCount(terminalCancelledMemberCountValue),
      inconsistentOutcomeMemberCount(inconsistentOutcomeMemberCountValue),
      metricComparisonCount(metricComparisonCountValue),
      comparableMetricCount(comparableMetricCountValue),
      contextChangedMetricCount(contextChangedMetricCountValue),
      missingSourceMetricCount(missingSourceMetricCountValue),
      missingResultMetricCount(missingResultMetricCountValue),
      metricValueUnavailableCount(metricValueUnavailableCountValue),
      unsupportedMetricCount(unsupportedMetricCountValue),
      positiveDeltaCount(positiveDeltaCountValue),
      zeroDeltaCount(zeroDeltaCountValue),
      negativeDeltaCount(negativeDeltaCountValue)
{
    const int lifecycleTotal = notTerminalLifecycleCount +
        succeededLifecycleCount + failedLifecycleCount +
        cancelledLifecycleCount + unknownLifecycleCount;
    const int consistencyTotal =
        consistentMemberCount + inconsistentMemberCount;
    const int outcomeTotal = notReadyMemberCount +
        succeededComparableMemberCount + succeededContextChangedMemberCount +
        succeededMetricGapMemberCount + terminalFailedMemberCount +
        terminalCancelledMemberCount + inconsistentOutcomeMemberCount;
    const std::size_t comparisonTotal = comparableMetricCount +
        contextChangedMetricCount + missingSourceMetricCount +
        missingResultMetricCount + metricValueUnavailableCount +
        unsupportedMetricCount;
    const std::size_t deltaTotal =
        positiveDeltaCount + zeroDeltaCount + negativeDeltaCount;
    if (memberCount <= 0 || lifecycleTotal != memberCount ||
        consistencyTotal != memberCount || outcomeTotal != memberCount ||
        comparisonTotal != metricComparisonCount ||
        deltaTotal != comparableMetricCount)
        throw std::logic_error(
            "recommendation_campaign_outcome_assessment_summary_invariant_failed");
}

RecommendationCampaignOutcomeAssessmentMember::
    RecommendationCampaignOutcomeAssessmentMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identityValue,
        Lifecycle lifecycleValue,
        Consistency consistencyValue,
        MemberOutcome outcomeValue,
        std::vector<Diagnostic> diagnosticsValue,
        std::optional<
            RecommendationCampaignOutcomeAssessmentSourceEvidence>
            sourceEvidenceValue,
        std::optional<
            RecommendationCampaignOutcomeAssessmentResultEvidence>
            resultEvidenceValue,
        std::vector<Comparison> comparisonsValue)
    : identity(std::move(identityValue)),
      lifecycle(lifecycleValue),
      consistency(consistencyValue),
      outcome(outcomeValue),
      diagnostics(std::move(diagnosticsValue)),
      sourceEvidence(std::move(sourceEvidenceValue)),
      resultEvidence(std::move(resultEvidenceValue)),
      comparisons(std::move(comparisonsValue))
{
    const bool inconsistent = consistency == Consistency::Inconsistent;
    if (inconsistent != !diagnostics.empty() ||
        inconsistent != (outcome == MemberOutcome::Inconsistent) ||
        (inconsistent && !comparisons.empty()))
        throw std::logic_error(
            "recommendation_campaign_outcome_assessment_member_invariant_failed");
}

RecommendationCampaignOutcomeAssessmentIdentity::
    RecommendationCampaignOutcomeAssessmentIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
    if (contractVersion !=
            kRecommendationCampaignOutcomeAssessmentContractVersion ||
        hash != RecommendationCanonicalHash(canonicalText))
        throw std::logic_error(
            "recommendation_campaign_outcome_assessment_identity_invariant_failed");
    RequireText(canonicalText,
        "recommendation_campaign_outcome_assessment_identity_missing");
}

RecommendationCampaignOutcomeAssessment::
    RecommendationCampaignOutcomeAssessment(
        RecommendationCampaignOutcomeAssessmentIdentity identityValue,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentityValue,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentityValue,
        std::string observedAtValue,
        RecommendationCampaignOutcomeAssessmentSummary summaryValue,
        std::vector<Member> membersValue)
    : identity(std::move(identityValue)),
      campaignIdentity(std::move(campaignIdentityValue)),
      materializationIdentity(std::move(materializationIdentityValue)),
      observedAt(std::move(observedAtValue)),
      summary(std::move(summaryValue)),
      members(std::move(membersValue))
{
    RequireText(observedAt,
        "recommendation_campaign_outcome_assessment_observed_at_invalid");
    if (summary.memberCount != static_cast<int>(members.size()))
        throw std::logic_error(
            "recommendation_campaign_outcome_assessment_member_summary_invariant_failed");
}

struct RecommendationCampaignOutcomeAssessmentBuilder
{
    static std::vector<Comparison> BuildComparisons(
        const MemberEvidence& evidence,
        bool& nonFiniteDelta)
    {
        const auto* sourceMetrics = evidence.sourceEvidence
            ? &evidence.sourceEvidence->metrics.metrics
            : nullptr;
        const auto& resultMetrics = evidence.resultEvidence->metrics.metrics;
        std::vector<Difference> differences;
        if (evidence.sourceEvidence)
            differences = ContextDifferences(
                evidence.sourceEvidence->context,
                *evidence.resultEvidence->context);

        std::vector<Comparison> comparisons;
        std::size_t sourceIndex = 0;
        std::size_t resultIndex = 0;
        while ((sourceMetrics && sourceIndex < sourceMetrics->size()) ||
            resultIndex < resultMetrics.size())
        {
            const Metric* source = sourceMetrics &&
                    sourceIndex < sourceMetrics->size()
                ? &(*sourceMetrics)[sourceIndex]
                : nullptr;
            const Metric* result = resultIndex < resultMetrics.size()
                ? &resultMetrics[resultIndex]
                : nullptr;
            if (source && result && source->identity == result->identity)
            {
                if (!differences.empty())
                    comparisons.push_back(Comparison(source->identity,
                        Classification::ContextChanged, source->value,
                        result->value, std::nullopt, differences));
                else if (!source->value || !result->value)
                    comparisons.push_back(Comparison(source->identity,
                        Classification::MetricValueUnavailable, source->value,
                        result->value, std::nullopt, {}));
                else if (source->comparisonSupport != MetricSupport::NumericDelta ||
                    result->comparisonSupport != MetricSupport::NumericDelta)
                    comparisons.push_back(Comparison(source->identity,
                        Classification::Unsupported, source->value,
                        result->value, std::nullopt, {}));
                else
                {
                    double delta = *result->value - *source->value;
                    if (!std::isfinite(delta))
                    {
                        nonFiniteDelta = true;
                        return {};
                    }
                    if (delta == 0.0) delta = 0.0;
                    comparisons.push_back(Comparison(source->identity,
                        Classification::Comparable, source->value,
                        result->value, delta, {}));
                }
                ++sourceIndex;
                ++resultIndex;
            }
            else if (source &&
                (!result || source->identity < result->identity))
            {
                comparisons.push_back(Comparison(source->identity,
                    Classification::MissingResultMetric, source->value,
                    std::nullopt, std::nullopt, {}));
                ++sourceIndex;
            }
            else
            {
                comparisons.push_back(Comparison(result->identity,
                    Classification::MissingSourceMetric, std::nullopt,
                    result->value, std::nullopt, {}));
                ++resultIndex;
            }
        }
        return comparisons;
    }

    static MemberOutcome DeriveMemberOutcome(
        Lifecycle lifecycle,
        const std::vector<Diagnostic>& diagnostics,
        const std::vector<Comparison>& comparisons)
    {
        if (!diagnostics.empty()) return MemberOutcome::Inconsistent;
        switch (lifecycle)
        {
            case Lifecycle::NotTerminal:
            case Lifecycle::Unknown:
                return MemberOutcome::NotReady;
            case Lifecycle::Failed:
                return MemberOutcome::TerminalFailed;
            case Lifecycle::Cancelled:
                return MemberOutcome::TerminalCancelled;
            case Lifecycle::Succeeded:
                break;
        }
        if (std::any_of(comparisons.begin(), comparisons.end(),
                [](const Comparison& comparison)
        {
            return comparison.classification == Classification::ContextChanged;
        }))
            return MemberOutcome::SucceededContextChanged;
        if (comparisons.empty() ||
            std::any_of(comparisons.begin(), comparisons.end(),
                [](const Comparison& comparison)
        {
            return comparison.classification != Classification::Comparable;
        }))
            return MemberOutcome::SucceededMetricGap;
        return MemberOutcome::SucceededComparable;
    }

    static RecommendationCampaignOutcomeAssessment Build(
        const RecommendationCampaignOutcomeAssessmentCampaignIdentity&
            campaignIdentity,
        const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
            materializationIdentity,
        const std::string& observedAt,
        const std::vector<MemberEvidence>& members)
    {
        RequireText(observedAt,
            "recommendation_campaign_outcome_assessment_observed_at_invalid");
        if (materializationIdentity.campaignApprovalId !=
                campaignIdentity.campaignApprovalId ||
            materializationIdentity.campaignIdentityHash !=
                campaignIdentity.identityHash)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_campaign_materialization_mismatch");
        if (materializationIdentity.contractVersion !=
            kRecommendationCampaignMaterializationContractVersion)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_materialization_version_unsupported");
        if (materializationIdentity.memberCount !=
            static_cast<int>(members.size()))
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_member_count_mismatch");
        if (members.empty() ||
            members.size() > static_cast<std::size_t>(
                                 kMaximumRecommendationCampaignOutcomeAssessmentMembers))
            throw std::invalid_argument(
                "recommendation_campaign_outcome_assessment_member_count_invalid");

        std::vector<const MemberEvidence*> ordered;
        ordered.reserve(members.size());
        for (const auto& member : members) ordered.push_back(&member);
        std::sort(ordered.begin(), ordered.end(),
            [](const MemberEvidence* left, const MemberEvidence* right)
        {
            return std::tie(left->identity.memberOrdinal,
                       left->identity.materializationMemberId,
                       left->identity.rankingMemberId,
                       left->identity.recommendationId,
                       left->identity.proposalId) <
                std::tie(right->identity.memberOrdinal,
                       right->identity.materializationMemberId,
                       right->identity.rankingMemberId,
                       right->identity.recommendationId,
                       right->identity.proposalId);
        });

        std::set<long long> materializationMemberIds;
        std::set<long long> rankingMemberIds;
        std::set<long long> recommendationIds;
        std::set<long long> proposalIds;
        std::vector<WorkingMember> working;
        working.reserve(ordered.size());
        for (std::size_t index = 0; index < ordered.size(); ++index)
        {
            const auto& evidence = *ordered[index];
            if (evidence.identity.memberOrdinal != static_cast<int>(index) + 1 ||
                !materializationMemberIds
                     .insert(evidence.identity.materializationMemberId)
                     .second ||
                !rankingMemberIds.insert(evidence.identity.rankingMemberId)
                     .second ||
                !recommendationIds.insert(evidence.identity.recommendationId)
                     .second ||
                !proposalIds.insert(evidence.identity.proposalId).second)
                throw std::invalid_argument(
                    "recommendation_campaign_outcome_assessment_member_identity_set_invalid");

            WorkingMember item{&evidence, {}};
            if (evidence.inputConsistency == Consistency::Inconsistent)
                item.diagnostics.push_back(
                    Diagnostic::InputEvidenceInconsistent);
            if (evidence.sourceEvidence &&
                evidence.sourceEvidence->sourceExperimentId !=
                    evidence.identity.sourceExperimentId)
                item.diagnostics.push_back(
                    Diagnostic::SourceExperimentIdentityMismatch);
            if ((evidence.lifecycle == Lifecycle::Succeeded ||
                    evidence.lifecycle == Lifecycle::Failed ||
                    evidence.lifecycle == Lifecycle::Cancelled) &&
                !evidence.identity.expectedExperimentId)
                item.diagnostics.push_back(
                    Diagnostic::ExpectedExperimentIdentityMissing);
            if (evidence.lifecycle != Lifecycle::Succeeded &&
                evidence.resultEvidence)
                item.diagnostics.push_back(
                    Diagnostic::ResultEvidenceUnexpectedForLifecycle);
            if (evidence.lifecycle == Lifecycle::Succeeded)
            {
                if (!evidence.resultEvidence)
                    item.diagnostics.push_back(Diagnostic::ResultEvidenceMissing);
                else
                {
                    if (!evidence.resultEvidence->experimentId)
                        item.diagnostics.push_back(
                            Diagnostic::ResultExperimentIdentityMissing);
                    else if (evidence.identity.expectedExperimentId &&
                        evidence.resultEvidence->experimentId !=
                            evidence.identity.expectedExperimentId)
                        item.diagnostics.push_back(
                            Diagnostic::ResultExperimentIdentityMismatch);
                    if (!evidence.resultEvidence->modelId)
                        item.diagnostics.push_back(
                            Diagnostic::ResultModelIdentityMissing);
                    if (evidence.resultEvidence->resultIdentities.empty())
                        item.diagnostics.push_back(
                            Diagnostic::ResultIdentityMissing);
                    if (!evidence.resultEvidence->context)
                        item.diagnostics.push_back(
                            Diagnostic::ResultContextMissing);
                }
            }
            working.push_back(std::move(item));
        }

        std::map<long long, std::vector<std::size_t>> expectedExperiments;
        std::map<long long, std::vector<std::size_t>> resultExperiments;
        std::map<long long, std::vector<std::size_t>> resultModels;
        std::map<std::pair<std::string, long long>, std::vector<std::size_t>>
            resultIdentities;
        for (std::size_t index = 0; index < working.size(); ++index)
        {
            const auto& expected =
                working[index].evidence->identity.expectedExperimentId;
            if (expected) expectedExperiments[*expected].push_back(index);
            const auto& result = working[index].evidence->resultEvidence;
            if (!result) continue;
            if (result->experimentId)
                resultExperiments[*result->experimentId].push_back(index);
            if (result->modelId)
                resultModels[*result->modelId].push_back(index);
            for (const auto& identity : result->resultIdentities)
                resultIdentities[{identity.kind, identity.id}].push_back(index);
        }
        for (const auto& [unused, indices] : expectedExperiments)
        {
            static_cast<void>(unused);
            if (indices.size() < 2) continue;
            for (const auto index : indices)
                working[index].diagnostics.push_back(
                    Diagnostic::ExpectedExperimentIdentityReused);
        }
        for (const auto& [unused, indices] : resultExperiments)
        {
            static_cast<void>(unused);
            if (indices.size() < 2) continue;
            for (const auto index : indices)
                working[index].diagnostics.push_back(
                    Diagnostic::ResultExperimentReused);
        }
        for (const auto& [unused, indices] : resultModels)
        {
            static_cast<void>(unused);
            if (indices.size() < 2) continue;
            for (const auto index : indices)
                working[index].diagnostics.push_back(
                    Diagnostic::ResultModelIdentityReused);
        }
        for (const auto& [unused, indices] : resultIdentities)
        {
            static_cast<void>(unused);
            if (indices.size() < 2) continue;
            for (const auto index : indices)
                working[index].diagnostics.push_back(
                    Diagnostic::ResultIdentityReused);
        }

        std::vector<Member> assessedMembers;
        assessedMembers.reserve(working.size());
        for (auto& item : working)
        {
            CanonicalizeDiagnostics(item.diagnostics);
            std::vector<Comparison> comparisons;
            if (item.diagnostics.empty() &&
                item.evidence->lifecycle == Lifecycle::Succeeded)
            {
                bool nonFiniteDelta = false;
                comparisons = BuildComparisons(*item.evidence, nonFiniteDelta);
                if (nonFiniteDelta)
                {
                    item.diagnostics.push_back(
                        Diagnostic::NumericDeltaNonFinite);
                    comparisons.clear();
                }
            }
            CanonicalizeDiagnostics(item.diagnostics);
            const Consistency consistency = item.diagnostics.empty()
                ? Consistency::Consistent
                : Consistency::Inconsistent;
            const MemberOutcome outcome = DeriveMemberOutcome(
                item.evidence->lifecycle, item.diagnostics, comparisons);
            assessedMembers.push_back(Member(item.evidence->identity,
                item.evidence->lifecycle, consistency, outcome,
                item.diagnostics, item.evidence->sourceEvidence,
                item.evidence->resultEvidence, std::move(comparisons)));
        }

        MutableSummary counts;
        for (const auto& member : assessedMembers) CountMember(counts, member);
        const AggregateOutcome aggregateOutcome =
            DeriveAggregateOutcome(counts);
        RecommendationCampaignOutcomeAssessmentSummary summary(
            aggregateOutcome, counts.memberCount,
            counts.notTerminalLifecycleCount, counts.succeededLifecycleCount,
            counts.failedLifecycleCount, counts.cancelledLifecycleCount,
            counts.unknownLifecycleCount, counts.consistentMemberCount,
            counts.inconsistentMemberCount, counts.notReadyMemberCount,
            counts.succeededComparableMemberCount,
            counts.succeededContextChangedMemberCount,
            counts.succeededMetricGapMemberCount,
            counts.terminalFailedMemberCount,
            counts.terminalCancelledMemberCount,
            counts.inconsistentOutcomeMemberCount,
            counts.metricComparisonCount, counts.comparableMetricCount,
            counts.contextChangedMetricCount,
            counts.missingSourceMetricCount,
            counts.missingResultMetricCount,
            counts.metricValueUnavailableCount,
            counts.unsupportedMetricCount, counts.positiveDeltaCount,
            counts.zeroDeltaCount, counts.negativeDeltaCount);
        std::string canonical = AssessmentCanonicalText(campaignIdentity,
            materializationIdentity, summary, assessedMembers);
        RecommendationCampaignOutcomeAssessmentIdentity identity(
            kRecommendationCampaignOutcomeAssessmentContractVersion,
            canonical, RecommendationCanonicalHash(canonical));
        return RecommendationCampaignOutcomeAssessment(std::move(identity),
            campaignIdentity, materializationIdentity, observedAt,
            std::move(summary), std::move(assessedMembers));
    }
};

RecommendationCampaignOutcomeAssessment BuildRecommendationCampaignOutcomeAssessment(
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity&
        campaignIdentity,
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
        materializationIdentity,
    const std::string& observedAt,
    const std::vector<MemberEvidence>& members)
{
    return RecommendationCampaignOutcomeAssessmentBuilder::Build(
        campaignIdentity, materializationIdentity, observedAt, members);
}

#define EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(name, type, ...)                 \
    std::string name(type value)                                             \
    {                                                                         \
        switch (value)                                                        \
        {                                                                     \
            __VA_ARGS__                                                       \
        }                                                                     \
        throw std::invalid_argument(                                          \
            "recommendation_campaign_outcome_assessment_enum_invalid");      \
    }
#define EA_OUTCOME_ASSESSMENT_TEXT_CASE(value, text) case value: return text;

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupportText,
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MetricSupport::NumericDelta,
        "numeric_delta")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MetricSupport::Unsupported, "unsupported"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentLifecycleStateText,
    RecommendationCampaignOutcomeAssessmentLifecycleState,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Lifecycle::NotTerminal, "not_terminal")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Lifecycle::Succeeded, "succeeded")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Lifecycle::Failed, "failed")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Lifecycle::Cancelled, "cancelled")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Lifecycle::Unknown, "unknown"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentConsistencyStateText,
    RecommendationCampaignOutcomeAssessmentConsistencyState,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Consistency::Consistent, "consistent")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Consistency::Inconsistent, "inconsistent"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentDiagnosticCodeText,
    RecommendationCampaignOutcomeAssessmentDiagnosticCode,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::InputEvidenceInconsistent,
        "input_evidence_inconsistent")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::SourceExperimentIdentityMismatch,
        "source_experiment_identity_mismatch")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ExpectedExperimentIdentityMissing,
        "expected_experiment_identity_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ExpectedExperimentIdentityReused,
        "expected_experiment_identity_reused")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(
        Diagnostic::ResultEvidenceUnexpectedForLifecycle,
        "result_evidence_unexpected_for_lifecycle")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultEvidenceMissing,
        "result_evidence_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultExperimentIdentityMissing,
        "result_experiment_identity_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultExperimentIdentityMismatch,
        "result_experiment_identity_mismatch")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultModelIdentityMissing,
        "result_model_identity_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultModelIdentityReused,
        "result_model_identity_reused")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultIdentityMissing,
        "result_identity_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultContextMissing,
        "result_context_missing")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultExperimentReused,
        "result_experiment_reused")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::ResultIdentityReused,
        "result_identity_reused")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Diagnostic::NumericDeltaNonFinite,
        "numeric_delta_nonfinite"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentMemberOutcomeText,
    RecommendationCampaignOutcomeAssessmentMemberOutcome,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::NotReady, "not_ready")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::SucceededComparable,
        "succeeded_comparable")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::SucceededContextChanged,
        "succeeded_context_changed")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::SucceededMetricGap,
        "succeeded_metric_gap")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::TerminalFailed,
        "terminal_failed")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::TerminalCancelled,
        "terminal_cancelled")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(MemberOutcome::Inconsistent,
        "inconsistent"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText,
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::Comparable, "comparable")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::ContextChanged,
        "context_changed")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::MissingSourceMetric,
        "missing_source_metric")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::MissingResultMetric,
        "missing_result_metric")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::MetricValueUnavailable,
        "metric_value_unavailable")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Classification::Unsupported,
        "unsupported"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentContextDifferenceText,
    RecommendationCampaignOutcomeAssessmentContextDifference,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::Symbol, "symbol")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::PredictionHorizon,
        "prediction_horizon")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::Threshold, "threshold")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::WindowSize, "window_size")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::LabelDefinition,
        "label_definition")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(Difference::InferenceRange,
        "inference_range"))

EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION(
    RecommendationCampaignOutcomeAssessmentAggregateOutcomeText,
    RecommendationCampaignOutcomeAssessmentAggregateOutcome,
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::Inconsistent,
        "inconsistent")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::HasFailures,
        "has_failures")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::HasCancellations,
        "has_cancellations")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::NotReady, "not_ready")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::SucceededContextChanged,
        "succeeded_context_changed")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::SucceededMetricGap,
        "succeeded_metric_gap")
    EA_OUTCOME_ASSESSMENT_TEXT_CASE(AggregateOutcome::SucceededComparable,
        "succeeded_comparable"))

#undef EA_OUTCOME_ASSESSMENT_TEXT_CASE
#undef EA_OUTCOME_ASSESSMENT_TEXT_FUNCTION

} // namespace EA::ExperimentRecommendation
