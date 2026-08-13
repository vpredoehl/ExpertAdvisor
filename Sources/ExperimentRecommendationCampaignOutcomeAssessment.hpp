#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int
    kRecommendationCampaignOutcomeAssessmentContractVersion = 2;
inline constexpr int
    kMaximumRecommendationCampaignOutcomeAssessmentMembers = 1000;

// These immutable value types describe a read-only, point-in-time,
// non-persistent, and non-authoritative evidence assessment. They intentionally
// contain no campaign success decision or follow-up authorization.
struct RecommendationCampaignOutcomeAssessmentCampaignIdentity
{
    const long long campaignApprovalId;
    const std::string identityCanonical;
    const std::string identityHash;

    RecommendationCampaignOutcomeAssessmentCampaignIdentity(
        long long campaignApprovalId,
        std::string identityCanonical,
        std::string identityHash);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentCampaignIdentity&) const =
        default;
};

struct RecommendationCampaignOutcomeAssessmentMaterializationIdentity
{
    const long long materializationId;
    const long long campaignApprovalId;
    const std::string campaignIdentityHash;
    const int contractVersion;
    const int memberCount;
    const std::string identityCanonical;
    const std::string identityHash;

    RecommendationCampaignOutcomeAssessmentMaterializationIdentity(
        long long materializationId,
        long long campaignApprovalId,
        std::string campaignIdentityHash,
        int contractVersion,
        int memberCount,
        std::string identityCanonical,
        std::string identityHash);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&)
        const = default;
};

struct RecommendationCampaignOutcomeAssessmentMemberIdentity
{
    const int memberOrdinal;
    const long long materializationMemberId;
    const long long rankingMemberId;
    const long long recommendationId;
    const long long sourceExperimentId;
    const long long proposalId;
    const std::optional<long long> expectedExperimentId;

    RecommendationCampaignOutcomeAssessmentMemberIdentity(
        int memberOrdinal,
        long long materializationMemberId,
        long long rankingMemberId,
        long long recommendationId,
        long long sourceExperimentId,
        long long proposalId,
        std::optional<long long> expectedExperimentId);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMemberIdentity&) const =
        default;
};

enum class RecommendationCampaignOutcomeAssessmentMetricComparisonSupport
{
    NumericDelta,
    Unsupported
};

struct RecommendationCampaignOutcomeAssessmentMetric
{
    const std::string identity;
    const std::optional<double> value;
    const RecommendationCampaignOutcomeAssessmentMetricComparisonSupport
        comparisonSupport;

    RecommendationCampaignOutcomeAssessmentMetric(
        std::string identity,
        std::optional<double> value,
        RecommendationCampaignOutcomeAssessmentMetricComparisonSupport
            comparisonSupport);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMetric&) const = default;
};

struct RecommendationCampaignOutcomeAssessmentMetricCollection
{
    // Always sorted by metric identity. Duplicate identities are rejected.
    const std::vector<RecommendationCampaignOutcomeAssessmentMetric> metrics;

    explicit RecommendationCampaignOutcomeAssessmentMetricCollection(
        std::vector<RecommendationCampaignOutcomeAssessmentMetric> metrics =
            {});
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMetricCollection&) const =
        default;
};

struct RecommendationCampaignOutcomeAssessmentComparisonContext
{
    const std::string symbol;
    const int predictionHorizon;
    const double threshold;
    const int windowSize;
    const std::string labelDefinition;
    const std::string inferenceRangeStart;
    const std::string inferenceRangeEnd;

    RecommendationCampaignOutcomeAssessmentComparisonContext(
        std::string symbol,
        int predictionHorizon,
        double threshold,
        int windowSize,
        std::string labelDefinition,
        std::string inferenceRangeStart,
        std::string inferenceRangeEnd);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentComparisonContext&) const =
        default;
};

struct RecommendationCampaignOutcomeAssessmentSourceEvidence
{
    const long long sourceExperimentId;
    const std::optional<long long> sourceModelId;
    const std::optional<long long> sourceAnalysisId;
    const RecommendationCampaignOutcomeAssessmentComparisonContext context;
    const RecommendationCampaignOutcomeAssessmentMetricCollection metrics;

    RecommendationCampaignOutcomeAssessmentSourceEvidence(
        long long sourceExperimentId,
        std::optional<long long> sourceModelId,
        std::optional<long long> sourceAnalysisId,
        RecommendationCampaignOutcomeAssessmentComparisonContext context,
        RecommendationCampaignOutcomeAssessmentMetricCollection metrics);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentSourceEvidence&) const =
        default;
};

struct RecommendationCampaignOutcomeAssessmentResultIdentity
{
    const std::string kind;
    const long long id;

    RecommendationCampaignOutcomeAssessmentResultIdentity(
        std::string kind,
        long long id);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentResultIdentity&) const =
        default;
};

struct RecommendationCampaignOutcomeAssessmentResultEvidence
{
    const std::optional<long long> experimentId;
    const std::optional<long long> modelId;
    // Always sorted by (kind, id). Exact duplicates are rejected.
    const std::vector<RecommendationCampaignOutcomeAssessmentResultIdentity>
        resultIdentities;
    const std::optional<
        RecommendationCampaignOutcomeAssessmentComparisonContext>
        context;
    const RecommendationCampaignOutcomeAssessmentMetricCollection metrics;

    RecommendationCampaignOutcomeAssessmentResultEvidence(
        std::optional<long long> experimentId,
        std::optional<long long> modelId,
        std::vector<RecommendationCampaignOutcomeAssessmentResultIdentity>
            resultIdentities,
        std::optional<RecommendationCampaignOutcomeAssessmentComparisonContext>
            context,
        RecommendationCampaignOutcomeAssessmentMetricCollection metrics);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentResultEvidence&) const =
        default;
};

enum class RecommendationCampaignOutcomeAssessmentLifecycleState
{
    NotTerminal,
    Succeeded,
    Failed,
    Cancelled,
    Unknown
};

enum class RecommendationCampaignOutcomeAssessmentConsistencyState
{
    Consistent,
    Inconsistent
};

struct RecommendationCampaignOutcomeAssessmentMemberEvidence
{
    const RecommendationCampaignOutcomeAssessmentMemberIdentity identity;
    const RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle;
    const RecommendationCampaignOutcomeAssessmentConsistencyState
        inputConsistency;
    const std::optional<RecommendationCampaignOutcomeAssessmentSourceEvidence>
        sourceEvidence;
    const std::optional<RecommendationCampaignOutcomeAssessmentResultEvidence>
        resultEvidence;

    RecommendationCampaignOutcomeAssessmentMemberEvidence(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identity,
        RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle,
        std::optional<RecommendationCampaignOutcomeAssessmentSourceEvidence>
            sourceEvidence,
        std::optional<RecommendationCampaignOutcomeAssessmentResultEvidence>
            resultEvidence,
        RecommendationCampaignOutcomeAssessmentConsistencyState
            inputConsistency =
                RecommendationCampaignOutcomeAssessmentConsistencyState::Consistent);
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMemberEvidence&) const =
        default;
};

enum class RecommendationCampaignOutcomeAssessmentDiagnosticCode
{
    InputEvidenceInconsistent,
    SourceExperimentIdentityMismatch,
    ExpectedExperimentIdentityMissing,
    ExpectedExperimentIdentityReused,
    ResultEvidenceUnexpectedForLifecycle,
    ResultEvidenceMissing,
    ResultExperimentIdentityMissing,
    ResultExperimentIdentityMismatch,
    ResultModelIdentityMissing,
    ResultModelIdentityReused,
    ResultIdentityMissing,
    ResultContextMissing,
    ResultExperimentReused,
    ResultIdentityReused,
    NumericDeltaNonFinite
};

enum class RecommendationCampaignOutcomeAssessmentMemberOutcome
{
    NotReady,
    SucceededComparable,
    SucceededContextChanged,
    SucceededMetricGap,
    TerminalFailed,
    TerminalCancelled,
    Inconsistent
};

enum class RecommendationCampaignOutcomeAssessmentMetricComparisonClassification
{
    Comparable,
    ContextChanged,
    MissingSourceMetric,
    MissingResultMetric,
    MetricValueUnavailable,
    Unsupported
};

enum class RecommendationCampaignOutcomeAssessmentContextDifference
{
    Symbol,
    PredictionHorizon,
    Threshold,
    WindowSize,
    LabelDefinition,
    InferenceRange
};

struct RecommendationCampaignOutcomeAssessmentBuilder;

struct RecommendationCampaignOutcomeAssessmentMetricComparison
{
    const std::string metricIdentity;
    const RecommendationCampaignOutcomeAssessmentMetricComparisonClassification
        classification;
    const std::optional<double> sourceValue;
    const std::optional<double> resultValue;
    const std::optional<double> delta;
    const std::vector<
        RecommendationCampaignOutcomeAssessmentContextDifference>
        contextDifferences;

    RecommendationCampaignOutcomeAssessmentMetricComparison(
        const RecommendationCampaignOutcomeAssessmentMetricComparison&) =
        default;
    RecommendationCampaignOutcomeAssessmentMetricComparison(
        RecommendationCampaignOutcomeAssessmentMetricComparison&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMetricComparison&) const =
        default;

private:
    RecommendationCampaignOutcomeAssessmentMetricComparison(
        std::string metricIdentity,
        RecommendationCampaignOutcomeAssessmentMetricComparisonClassification
            classification,
        std::optional<double> sourceValue,
        std::optional<double> resultValue,
        std::optional<double> delta,
        std::vector<RecommendationCampaignOutcomeAssessmentContextDifference>
            contextDifferences);
    friend struct RecommendationCampaignOutcomeAssessmentBuilder;
};

enum class RecommendationCampaignOutcomeAssessmentAggregateOutcome
{
    Inconsistent,
    HasFailures,
    HasCancellations,
    NotReady,
    SucceededContextChanged,
    SucceededMetricGap,
    SucceededComparable
};

struct RecommendationCampaignOutcomeAssessmentSummary
{
    const RecommendationCampaignOutcomeAssessmentAggregateOutcome outcome;

    const int memberCount;
    const int notTerminalLifecycleCount;
    const int succeededLifecycleCount;
    const int failedLifecycleCount;
    const int cancelledLifecycleCount;
    const int unknownLifecycleCount;
    const int consistentMemberCount;
    const int inconsistentMemberCount;

    const int notReadyMemberCount;
    const int succeededComparableMemberCount;
    const int succeededContextChangedMemberCount;
    const int succeededMetricGapMemberCount;
    const int terminalFailedMemberCount;
    const int terminalCancelledMemberCount;
    const int inconsistentOutcomeMemberCount;

    const std::size_t metricComparisonCount;
    const std::size_t comparableMetricCount;
    const std::size_t contextChangedMetricCount;
    const std::size_t missingSourceMetricCount;
    const std::size_t missingResultMetricCount;
    const std::size_t metricValueUnavailableCount;
    const std::size_t unsupportedMetricCount;
    const std::size_t positiveDeltaCount;
    const std::size_t zeroDeltaCount;
    const std::size_t negativeDeltaCount;

    RecommendationCampaignOutcomeAssessmentSummary(
        const RecommendationCampaignOutcomeAssessmentSummary&) = default;
    RecommendationCampaignOutcomeAssessmentSummary(
        RecommendationCampaignOutcomeAssessmentSummary&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentSummary&) const = default;

private:
    RecommendationCampaignOutcomeAssessmentSummary(
        RecommendationCampaignOutcomeAssessmentAggregateOutcome outcome,
        int memberCount,
        int notTerminalLifecycleCount,
        int succeededLifecycleCount,
        int failedLifecycleCount,
        int cancelledLifecycleCount,
        int unknownLifecycleCount,
        int consistentMemberCount,
        int inconsistentMemberCount,
        int notReadyMemberCount,
        int succeededComparableMemberCount,
        int succeededContextChangedMemberCount,
        int succeededMetricGapMemberCount,
        int terminalFailedMemberCount,
        int terminalCancelledMemberCount,
        int inconsistentOutcomeMemberCount,
        std::size_t metricComparisonCount,
        std::size_t comparableMetricCount,
        std::size_t contextChangedMetricCount,
        std::size_t missingSourceMetricCount,
        std::size_t missingResultMetricCount,
        std::size_t metricValueUnavailableCount,
        std::size_t unsupportedMetricCount,
        std::size_t positiveDeltaCount,
        std::size_t zeroDeltaCount,
        std::size_t negativeDeltaCount);
    friend struct RecommendationCampaignOutcomeAssessmentBuilder;
};

struct RecommendationCampaignOutcomeAssessmentMember
{
    const RecommendationCampaignOutcomeAssessmentMemberIdentity identity;
    const RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle;
    const RecommendationCampaignOutcomeAssessmentConsistencyState consistency;
    const RecommendationCampaignOutcomeAssessmentMemberOutcome outcome;
    // Empty for consistent members; otherwise sorted in fixed code order.
    const std::vector<RecommendationCampaignOutcomeAssessmentDiagnosticCode>
        diagnostics;
    const std::optional<RecommendationCampaignOutcomeAssessmentSourceEvidence>
        sourceEvidence;
    const std::optional<RecommendationCampaignOutcomeAssessmentResultEvidence>
        resultEvidence;
    // One entry per metric identity in the source/result union, sorted by identity.
    const std::vector<
        RecommendationCampaignOutcomeAssessmentMetricComparison>
        comparisons;

    RecommendationCampaignOutcomeAssessmentMember(
        const RecommendationCampaignOutcomeAssessmentMember&) = default;
    RecommendationCampaignOutcomeAssessmentMember(
        RecommendationCampaignOutcomeAssessmentMember&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentMember&) const = default;

private:
    RecommendationCampaignOutcomeAssessmentMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identity,
        RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle,
        RecommendationCampaignOutcomeAssessmentConsistencyState consistency,
        RecommendationCampaignOutcomeAssessmentMemberOutcome outcome,
        std::vector<RecommendationCampaignOutcomeAssessmentDiagnosticCode>
            diagnostics,
        std::optional<RecommendationCampaignOutcomeAssessmentSourceEvidence>
            sourceEvidence,
        std::optional<RecommendationCampaignOutcomeAssessmentResultEvidence>
            resultEvidence,
        std::vector<RecommendationCampaignOutcomeAssessmentMetricComparison>
            comparisons);
    friend struct RecommendationCampaignOutcomeAssessmentBuilder;
};

struct RecommendationCampaignOutcomeAssessmentIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignOutcomeAssessmentIdentity(
        const RecommendationCampaignOutcomeAssessmentIdentity&) = default;
    RecommendationCampaignOutcomeAssessmentIdentity(
        RecommendationCampaignOutcomeAssessmentIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomeAssessmentIdentity&) const = default;

private:
    RecommendationCampaignOutcomeAssessmentIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignOutcomeAssessmentBuilder;
};

struct RecommendationCampaignOutcomeAssessment
{
    static constexpr bool readOnly = true;
    static constexpr bool pointInTime = true;
    static constexpr bool persistent = false;
    static constexpr bool authoritative = false;
    static constexpr bool declaresCampaignSuccess = false;

    const RecommendationCampaignOutcomeAssessmentIdentity identity;
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity
        campaignIdentity;
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity
        materializationIdentity;
    const std::string observedAt;
    const RecommendationCampaignOutcomeAssessmentSummary summary;
    const std::vector<RecommendationCampaignOutcomeAssessmentMember> members;

    RecommendationCampaignOutcomeAssessment(
        const RecommendationCampaignOutcomeAssessment&) = default;
    RecommendationCampaignOutcomeAssessment(
        RecommendationCampaignOutcomeAssessment&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomeAssessment&) const = default;

private:
    RecommendationCampaignOutcomeAssessment(
        RecommendationCampaignOutcomeAssessmentIdentity identity,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentity,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentity,
        std::string observedAt,
        RecommendationCampaignOutcomeAssessmentSummary summary,
        std::vector<RecommendationCampaignOutcomeAssessmentMember> members);
    friend struct RecommendationCampaignOutcomeAssessmentBuilder;
};

RecommendationCampaignOutcomeAssessment BuildRecommendationCampaignOutcomeAssessment(
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity&
        campaignIdentity,
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity&
        materializationIdentity,
    const std::string& observedAt,
    const std::vector<RecommendationCampaignOutcomeAssessmentMemberEvidence>&
        members);

std::string RecommendationCampaignOutcomeAssessmentMetricComparisonSupportText(
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport value);
std::string RecommendationCampaignOutcomeAssessmentLifecycleStateText(
    RecommendationCampaignOutcomeAssessmentLifecycleState value);
std::string RecommendationCampaignOutcomeAssessmentConsistencyStateText(
    RecommendationCampaignOutcomeAssessmentConsistencyState value);
std::string RecommendationCampaignOutcomeAssessmentDiagnosticCodeText(
    RecommendationCampaignOutcomeAssessmentDiagnosticCode value);
std::string RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
    RecommendationCampaignOutcomeAssessmentMemberOutcome value);
std::string
RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification
        value);
std::string RecommendationCampaignOutcomeAssessmentContextDifferenceText(
    RecommendationCampaignOutcomeAssessmentContextDifference value);
std::string RecommendationCampaignOutcomeAssessmentAggregateOutcomeText(
    RecommendationCampaignOutcomeAssessmentAggregateOutcome value);

} // namespace EA::ExperimentRecommendation
