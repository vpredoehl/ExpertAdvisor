#pragma once

#include "ExperimentRecommendationCampaignOutcomeAssessment.hpp"

#include <cstddef>
#include <optional>
#include <string>
#include <vector>

namespace EA::ExperimentRecommendation
{

inline constexpr int kRecommendationCampaignOutcomePolicyContractVersion = 1;
inline constexpr int
    kRecommendationCampaignOutcomePolicyDecisionContractVersion = 1;

enum class RecommendationCampaignOutcomePolicyMetricDirection
{
    HigherIsFavorable,
    LowerIsFavorable
};

struct RecommendationCampaignOutcomePolicyMetricRuleInput
{
    std::string metricIdentity;
    RecommendationCampaignOutcomePolicyMetricDirection direction =
        RecommendationCampaignOutcomePolicyMetricDirection::HigherIsFavorable;
};

struct RecommendationCampaignOutcomePolicyInput
{
    int contractVersion =
        kRecommendationCampaignOutcomePolicyContractVersion;
    int minimumComparableMemberCount = 1;
    std::vector<RecommendationCampaignOutcomePolicyMetricRuleInput>
        requiredMetricRules{
            {"inference_accuracy",
                RecommendationCampaignOutcomePolicyMetricDirection::
                    HigherIsFavorable},
            {"leader_score",
                RecommendationCampaignOutcomePolicyMetricDirection::
                    HigherIsFavorable}};
};

// Narrow plain-data validation views used at the immutable Step 5a/5c
// boundary. They cannot construct a policy, assessment, or decision and add no
// persistence or other production capability.
struct RecommendationCampaignOutcomePolicyMemberValidationView
{
    int memberOrdinal = 0;
    long long materializationMemberId = 0;
    long long rankingMemberId = 0;
    long long recommendationId = 0;
    long long sourceExperimentId = 0;
    long long proposalId = 0;
    std::optional<long long> expectedExperimentId;
    std::vector<std::string> comparisonMetricIdentities;
};

struct RecommendationCampaignOutcomePolicyValidationView
{
    int policyContractVersion = 0;
    std::string policyCanonicalText;
    std::string policyIdentityHash;

    int assessmentContractVersion = 0;
    std::string assessmentCanonicalText;
    std::string assessmentIdentityHash;

    long long campaignApprovalId = 0;
    std::string campaignIdentityCanonical;
    std::string campaignIdentityHash;

    long long materializationId = 0;
    long long materializationCampaignApprovalId = 0;
    std::string materializationCampaignIdentityHash;
    int materializationContractVersion = 0;
    int materializationMemberCount = 0;
    std::string materializationIdentityCanonical;
    std::string materializationIdentityHash;

    int assessmentSummaryMemberCount = 0;
    std::vector<RecommendationCampaignOutcomePolicyMemberValidationView>
        members;
};

struct RecommendationCampaignOutcomePolicyMetricRule
{
    const std::string metricIdentity;
    const RecommendationCampaignOutcomePolicyMetricDirection direction;

    RecommendationCampaignOutcomePolicyMetricRule(
        std::string metricIdentity,
        RecommendationCampaignOutcomePolicyMetricDirection direction);
    bool operator==(
        const RecommendationCampaignOutcomePolicyMetricRule&) const = default;
};

struct RecommendationCampaignOutcomePolicyIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignOutcomePolicyIdentity(
        const RecommendationCampaignOutcomePolicyIdentity&) = default;
    RecommendationCampaignOutcomePolicyIdentity(
        RecommendationCampaignOutcomePolicyIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicyIdentity&) const = default;

private:
    RecommendationCampaignOutcomePolicyIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignOutcomePolicyBuilder;
};

struct RecommendationCampaignOutcomePolicy
{
    const int minimumComparableMemberCount;
    // Always sorted by metric identity. Duplicate identities are rejected.
    const std::vector<RecommendationCampaignOutcomePolicyMetricRule>
        requiredMetricRules;
    const RecommendationCampaignOutcomePolicyIdentity identity;

    RecommendationCampaignOutcomePolicy(
        const RecommendationCampaignOutcomePolicy&) = default;
    RecommendationCampaignOutcomePolicy(
        RecommendationCampaignOutcomePolicy&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicy&) const = default;

private:
    RecommendationCampaignOutcomePolicy(
        int minimumComparableMemberCount,
        std::vector<RecommendationCampaignOutcomePolicyMetricRule>
            requiredMetricRules,
        RecommendationCampaignOutcomePolicyIdentity identity);
    friend struct RecommendationCampaignOutcomePolicyBuilder;
};

enum class RecommendationCampaignOutcomePolicyEvidenceClassification
{
    Inconsistent,
    NotReady,
    TerminalFailed,
    TerminalCancelled,
    SucceededContextChanged,
    SucceededMetricGap,
    SucceededComparable
};

enum class RecommendationCampaignOutcomePolicyMetricJudgment
{
    NotEvaluable,
    Favorable,
    Neutral,
    Unfavorable
};

enum class RecommendationCampaignOutcomePolicyMemberInterpretation
{
    Inconclusive,
    Favorable,
    Neutral,
    Unfavorable,
    Mixed
};

enum class RecommendationCampaignOutcomePolicyEvidenceSufficiency
{
    Insufficient,
    Sufficient
};

enum class RecommendationCampaignOutcomePolicyCampaignInterpretation
{
    Inconclusive,
    Favorable,
    Neutral,
    Unfavorable,
    Mixed
};

enum class RecommendationCampaignOutcomePolicyFollowUpEligibility
{
    NotEligible,
    EligibleForOperatorReview
};

enum class RecommendationCampaignOutcomePolicyReason
{
    InconsistentMemberPresent,
    NotReadyMemberPresent,
    FailedMemberPresent,
    CancelledMemberPresent,
    ContextChangedMemberPresent,
    MetricGapMemberPresent,
    ContextChangedComparisonPresent,
    MissingSourceMetricPresent,
    MissingResultMetricPresent,
    MetricValueUnavailablePresent,
    UnsupportedMetricPresent,
    RequiredMetricMissing,
    RequiredMetricNotComparable,
    MinimumComparableMembersNotMet,
    FavorableMetricEvidencePresent,
    NeutralMetricEvidencePresent,
    UnfavorableMetricEvidencePresent,
    EvidenceInsufficient,
    FavorableCampaignInterpretationRequired,
    AllMembersSucceededComparable
};

struct RecommendationCampaignOutcomePolicyMetricEvaluation
{
    const std::string metricIdentity;
    const RecommendationCampaignOutcomePolicyMetricDirection direction;
    const std::optional<
        RecommendationCampaignOutcomeAssessmentMetricComparisonClassification>
        evidenceClassification;
    // This is copied from Step 5a. The policy never recomputes it.
    const std::optional<double> assessmentDelta;
    const RecommendationCampaignOutcomePolicyMetricJudgment judgment;

    RecommendationCampaignOutcomePolicyMetricEvaluation(
        const RecommendationCampaignOutcomePolicyMetricEvaluation&) = default;
    RecommendationCampaignOutcomePolicyMetricEvaluation(
        RecommendationCampaignOutcomePolicyMetricEvaluation&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicyMetricEvaluation&) const =
        default;

private:
    RecommendationCampaignOutcomePolicyMetricEvaluation(
        std::string metricIdentity,
        RecommendationCampaignOutcomePolicyMetricDirection direction,
        std::optional<
            RecommendationCampaignOutcomeAssessmentMetricComparisonClassification>
            evidenceClassification,
        std::optional<double> assessmentDelta,
        RecommendationCampaignOutcomePolicyMetricJudgment judgment);
    friend struct RecommendationCampaignOutcomePolicyDecisionBuilder;
};

struct RecommendationCampaignOutcomePolicyMember
{
    const RecommendationCampaignOutcomeAssessmentMemberIdentity identity;
    const RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle;
    const RecommendationCampaignOutcomeAssessmentConsistencyState consistency;
    const RecommendationCampaignOutcomeAssessmentMemberOutcome
        assessmentOutcome;
    const std::vector<RecommendationCampaignOutcomeAssessmentDiagnosticCode>
        assessmentDiagnostics;
    const RecommendationCampaignOutcomePolicyEvidenceClassification
        evidenceClassification;
    const RecommendationCampaignOutcomePolicyMemberInterpretation
        interpretation;
    const std::vector<RecommendationCampaignOutcomePolicyMetricEvaluation>
        metricEvaluations;

    RecommendationCampaignOutcomePolicyMember(
        const RecommendationCampaignOutcomePolicyMember&) = default;
    RecommendationCampaignOutcomePolicyMember(
        RecommendationCampaignOutcomePolicyMember&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicyMember&) const = default;

private:
    RecommendationCampaignOutcomePolicyMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identity,
        RecommendationCampaignOutcomeAssessmentLifecycleState lifecycle,
        RecommendationCampaignOutcomeAssessmentConsistencyState consistency,
        RecommendationCampaignOutcomeAssessmentMemberOutcome assessmentOutcome,
        std::vector<RecommendationCampaignOutcomeAssessmentDiagnosticCode>
            assessmentDiagnostics,
        RecommendationCampaignOutcomePolicyEvidenceClassification
            evidenceClassification,
        RecommendationCampaignOutcomePolicyMemberInterpretation interpretation,
        std::vector<RecommendationCampaignOutcomePolicyMetricEvaluation>
            metricEvaluations);
    friend struct RecommendationCampaignOutcomePolicyDecisionBuilder;
};

struct RecommendationCampaignOutcomePolicySummary
{
    const RecommendationCampaignOutcomePolicyEvidenceSufficiency
        evidenceSufficiency;
    const RecommendationCampaignOutcomePolicyCampaignInterpretation
        campaignInterpretation;
    const RecommendationCampaignOutcomePolicyFollowUpEligibility
        followUpEligibility;
    const bool followUpAuthorized;
    const int memberCount;
    const int inconsistentMemberCount;
    const int notReadyMemberCount;
    const int failedMemberCount;
    const int cancelledMemberCount;
    const int contextChangedMemberCount;
    const int metricGapMemberCount;
    const int comparableMemberCount;
    const std::size_t metricEvaluationCount;
    const std::size_t notEvaluableMetricCount;
    const std::size_t favorableMetricCount;
    const std::size_t neutralMetricCount;
    const std::size_t unfavorableMetricCount;
    const std::vector<RecommendationCampaignOutcomePolicyReason>
        evidenceReasons;
    const std::vector<RecommendationCampaignOutcomePolicyReason>
        interpretationReasons;
    const std::vector<RecommendationCampaignOutcomePolicyReason>
        followUpReasons;

    RecommendationCampaignOutcomePolicySummary(
        const RecommendationCampaignOutcomePolicySummary&) = default;
    RecommendationCampaignOutcomePolicySummary(
        RecommendationCampaignOutcomePolicySummary&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicySummary&) const = default;

private:
    RecommendationCampaignOutcomePolicySummary(
        RecommendationCampaignOutcomePolicyEvidenceSufficiency
            evidenceSufficiency,
        RecommendationCampaignOutcomePolicyCampaignInterpretation
            campaignInterpretation,
        RecommendationCampaignOutcomePolicyFollowUpEligibility
            followUpEligibility,
        bool followUpAuthorized,
        int memberCount,
        int inconsistentMemberCount,
        int notReadyMemberCount,
        int failedMemberCount,
        int cancelledMemberCount,
        int contextChangedMemberCount,
        int metricGapMemberCount,
        int comparableMemberCount,
        std::size_t metricEvaluationCount,
        std::size_t notEvaluableMetricCount,
        std::size_t favorableMetricCount,
        std::size_t neutralMetricCount,
        std::size_t unfavorableMetricCount,
        std::vector<RecommendationCampaignOutcomePolicyReason>
            evidenceReasons,
        std::vector<RecommendationCampaignOutcomePolicyReason>
            interpretationReasons,
        std::vector<RecommendationCampaignOutcomePolicyReason>
            followUpReasons);
    friend struct RecommendationCampaignOutcomePolicyDecisionBuilder;
};

struct RecommendationCampaignOutcomePolicyDecisionIdentity
{
    const int contractVersion;
    const std::string canonicalText;
    const std::string hash;

    RecommendationCampaignOutcomePolicyDecisionIdentity(
        const RecommendationCampaignOutcomePolicyDecisionIdentity&) = default;
    RecommendationCampaignOutcomePolicyDecisionIdentity(
        RecommendationCampaignOutcomePolicyDecisionIdentity&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicyDecisionIdentity&) const =
        default;

private:
    RecommendationCampaignOutcomePolicyDecisionIdentity(
        int contractVersion,
        std::string canonicalText,
        std::string hash);
    friend struct RecommendationCampaignOutcomePolicyDecisionBuilder;
};

struct RecommendationCampaignOutcomePolicyDecision
{
    static constexpr bool readOnly = true;
    static constexpr bool databaseFree = true;
    static constexpr bool persistent = false;
    static constexpr bool advisory = true;
    static constexpr bool authoritative = false;
    static constexpr bool declaresCampaignSuccess = false;
    static constexpr bool followUpAuthorizing = false;

    const RecommendationCampaignOutcomePolicyDecisionIdentity identity;
    const RecommendationCampaignOutcomePolicy policy;
    const int assessmentContractVersion;
    const std::string assessmentCanonicalText;
    const std::string assessmentIdentityHash;
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity
        campaignIdentity;
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity
        materializationIdentity;
    const std::string observedAt;
    const RecommendationCampaignOutcomePolicySummary summary;
    const std::vector<RecommendationCampaignOutcomePolicyMember> members;

    RecommendationCampaignOutcomePolicyDecision(
        const RecommendationCampaignOutcomePolicyDecision&) = default;
    RecommendationCampaignOutcomePolicyDecision(
        RecommendationCampaignOutcomePolicyDecision&&) = default;
    bool operator==(
        const RecommendationCampaignOutcomePolicyDecision&) const = default;

private:
    RecommendationCampaignOutcomePolicyDecision(
        RecommendationCampaignOutcomePolicyDecisionIdentity identity,
        RecommendationCampaignOutcomePolicy policy,
        int assessmentContractVersion,
        std::string assessmentCanonicalText,
        std::string assessmentIdentityHash,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentity,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentity,
        std::string observedAt,
        RecommendationCampaignOutcomePolicySummary summary,
        std::vector<RecommendationCampaignOutcomePolicyMember> members);
    friend struct RecommendationCampaignOutcomePolicyDecisionBuilder;
};

RecommendationCampaignOutcomePolicy BuildRecommendationCampaignOutcomePolicy(
    const RecommendationCampaignOutcomePolicyInput& input = {});

RecommendationCampaignOutcomePolicyDecision
ApplyRecommendationCampaignOutcomePolicy(
    const RecommendationCampaignOutcomePolicy& policy,
    const RecommendationCampaignOutcomeAssessment& assessment);

void ValidateRecommendationCampaignOutcomePolicyEvidence(
    const RecommendationCampaignOutcomePolicyValidationView& view);

std::string RecommendationCampaignOutcomePolicyMetricDirectionText(
    RecommendationCampaignOutcomePolicyMetricDirection value);
std::string RecommendationCampaignOutcomePolicyEvidenceClassificationText(
    RecommendationCampaignOutcomePolicyEvidenceClassification value);
std::string RecommendationCampaignOutcomePolicyMetricJudgmentText(
    RecommendationCampaignOutcomePolicyMetricJudgment value);
std::string RecommendationCampaignOutcomePolicyMemberInterpretationText(
    RecommendationCampaignOutcomePolicyMemberInterpretation value);
std::string RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
    RecommendationCampaignOutcomePolicyEvidenceSufficiency value);
std::string RecommendationCampaignOutcomePolicyCampaignInterpretationText(
    RecommendationCampaignOutcomePolicyCampaignInterpretation value);
std::string RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
    RecommendationCampaignOutcomePolicyFollowUpEligibility value);
std::string RecommendationCampaignOutcomePolicyReasonText(
    RecommendationCampaignOutcomePolicyReason value);

} // namespace EA::ExperimentRecommendation
