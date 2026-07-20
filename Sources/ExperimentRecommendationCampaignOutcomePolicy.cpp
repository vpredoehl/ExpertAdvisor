#include "ExperimentRecommendationCampaignOutcomePolicy.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <cmath>
#include <locale>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <utility>

namespace EA::ExperimentRecommendation
{
namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using AssessmentClassification =
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification;
using AssessmentConsistency =
    RecommendationCampaignOutcomeAssessmentConsistencyState;
using AssessmentMember = RecommendationCampaignOutcomeAssessmentMember;
using AssessmentMemberOutcome =
    RecommendationCampaignOutcomeAssessmentMemberOutcome;
using CampaignInterpretation =
    RecommendationCampaignOutcomePolicyCampaignInterpretation;
using EvidenceClassification =
    RecommendationCampaignOutcomePolicyEvidenceClassification;
using EvidenceSufficiency =
    RecommendationCampaignOutcomePolicyEvidenceSufficiency;
using FollowUpEligibility =
    RecommendationCampaignOutcomePolicyFollowUpEligibility;
using MemberInterpretation =
    RecommendationCampaignOutcomePolicyMemberInterpretation;
using MetricDirection =
    RecommendationCampaignOutcomePolicyMetricDirection;
using MetricEvaluation =
    RecommendationCampaignOutcomePolicyMetricEvaluation;
using MetricJudgment = RecommendationCampaignOutcomePolicyMetricJudgment;
using MetricRule = RecommendationCampaignOutcomePolicyMetricRule;
using PolicyMember = RecommendationCampaignOutcomePolicyMember;
using PolicyReason = RecommendationCampaignOutcomePolicyReason;

void RequireText(const std::string& value, const char* error)
{
    if (value.empty() || value.find('\0') != std::string::npos)
        throw std::invalid_argument(error);
}

void RequireValidDirection(MetricDirection value)
{
    switch (value)
    {
        case MetricDirection::HigherIsFavorable:
        case MetricDirection::LowerIsFavorable:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_policy_metric_direction_invalid");
}

void RequireValidMetricJudgment(MetricJudgment value)
{
    switch (value)
    {
        case MetricJudgment::NotEvaluable:
        case MetricJudgment::Favorable:
        case MetricJudgment::Neutral:
        case MetricJudgment::Unfavorable:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_policy_metric_judgment_invalid");
}

void RequireValidMemberInterpretation(MemberInterpretation value)
{
    switch (value)
    {
        case MemberInterpretation::Inconclusive:
        case MemberInterpretation::Favorable:
        case MemberInterpretation::Neutral:
        case MemberInterpretation::Unfavorable:
        case MemberInterpretation::Mixed:
            return;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_policy_member_interpretation_invalid");
}

std::string LengthPrefixed(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

std::string OptionalDouble(const std::optional<double>& value)
{
    return value ? CanonicalRecommendationDouble(*value) : "null";
}

std::string OptionalLongLong(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "null";
}

std::string OptionalClassification(
    const std::optional<AssessmentClassification>& value)
{
    return value
        ? RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
              *value)
        : "null";
}

void AddReason(std::vector<PolicyReason>& reasons, PolicyReason reason)
{
    if (std::find(reasons.begin(), reasons.end(), reason) == reasons.end())
        reasons.push_back(reason);
}

void CanonicalizeReasons(std::vector<PolicyReason>& reasons)
{
    std::sort(reasons.begin(), reasons.end(),
        [](PolicyReason left, PolicyReason right)
    {
        return static_cast<int>(left) < static_cast<int>(right);
    });
    reasons.erase(std::unique(reasons.begin(), reasons.end()), reasons.end());
}

std::vector<MetricRule> CanonicalRules(
    const std::vector<RecommendationCampaignOutcomePolicyMetricRuleInput>&
        inputs)
{
    if (inputs.empty())
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_metric_rules_missing");
    std::vector<const RecommendationCampaignOutcomePolicyMetricRuleInput*>
        ordered;
    ordered.reserve(inputs.size());
    for (const auto& input : inputs) ordered.push_back(&input);
    std::sort(ordered.begin(), ordered.end(),
        [](const auto* left, const auto* right)
    {
        return left->metricIdentity < right->metricIdentity;
    });
    for (std::size_t index = 1; index < ordered.size(); ++index)
        if (ordered[index - 1]->metricIdentity ==
            ordered[index]->metricIdentity)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_policy_metric_rule_duplicate");
    std::vector<MetricRule> rules;
    rules.reserve(inputs.size());
    for (const auto* input : ordered)
        rules.emplace_back(input->metricIdentity, input->direction);
    return rules;
}

std::string PolicyCanonicalText(
    int minimumComparableMemberCount,
    const std::vector<MetricRule>& rules)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_outcome_policy_v1"
        << ";policy_contract_version="
        << kRecommendationCampaignOutcomePolicyContractVersion
        << ";minimum_comparable_member_count="
        << minimumComparableMemberCount
        << ";required_metric_rule_count=" << rules.size();
    for (const auto& rule : rules)
        out << ";required_metric=" << LengthPrefixed(rule.metricIdentity)
            << ','
            << RecommendationCampaignOutcomePolicyMetricDirectionText(
                   rule.direction);
    out << ";all_members_consistent_required=true"
           ";all_members_terminal_required=true"
           ";all_successful_members_comparable_required=true"
           ";follow_up_requires_all_members_comparable_success=true"
           ";follow_up_requires_favorable_campaign_interpretation=true"
           ";follow_up_authorizing=false";
    return out.str();
}

RecommendationCampaignOutcomePolicyValidationView ValidationView(
    const RecommendationCampaignOutcomePolicy& policy,
    const Assessment& assessment)
{
    RecommendationCampaignOutcomePolicyValidationView view;
    view.policyContractVersion = policy.identity.contractVersion;
    view.policyCanonicalText = policy.identity.canonicalText;
    view.policyIdentityHash = policy.identity.hash;
    view.assessmentContractVersion = assessment.identity.contractVersion;
    view.assessmentCanonicalText = assessment.identity.canonicalText;
    view.assessmentIdentityHash = assessment.identity.hash;
    view.campaignApprovalId = assessment.campaignIdentity.campaignApprovalId;
    view.campaignIdentityCanonical =
        assessment.campaignIdentity.identityCanonical;
    view.campaignIdentityHash = assessment.campaignIdentity.identityHash;
    view.materializationId =
        assessment.materializationIdentity.materializationId;
    view.materializationCampaignApprovalId =
        assessment.materializationIdentity.campaignApprovalId;
    view.materializationCampaignIdentityHash =
        assessment.materializationIdentity.campaignIdentityHash;
    view.materializationContractVersion =
        assessment.materializationIdentity.contractVersion;
    view.materializationMemberCount =
        assessment.materializationIdentity.memberCount;
    view.materializationIdentityCanonical =
        assessment.materializationIdentity.identityCanonical;
    view.materializationIdentityHash =
        assessment.materializationIdentity.identityHash;
    view.assessmentSummaryMemberCount = assessment.summary.memberCount;
    view.members.reserve(assessment.members.size());
    for (const auto& member : assessment.members)
    {
        RecommendationCampaignOutcomePolicyMemberValidationView memberView;
        memberView.memberOrdinal = member.identity.memberOrdinal;
        memberView.materializationMemberId =
            member.identity.materializationMemberId;
        memberView.rankingMemberId = member.identity.rankingMemberId;
        memberView.recommendationId = member.identity.recommendationId;
        memberView.sourceExperimentId = member.identity.sourceExperimentId;
        memberView.proposalId = member.identity.proposalId;
        memberView.expectedExperimentId =
            member.identity.expectedExperimentId;
        memberView.comparisonMetricIdentities.reserve(
            member.comparisons.size());
        for (const auto& comparison : member.comparisons)
            memberView.comparisonMetricIdentities.push_back(
                comparison.metricIdentity);
        view.members.push_back(std::move(memberView));
    }
    return view;
}

void ValidateAssessment(
    const RecommendationCampaignOutcomePolicy& policy,
    const Assessment& assessment)
{
    ValidateRecommendationCampaignOutcomePolicyEvidence(
        ValidationView(policy, assessment));

    (void)RecommendationCampaignOutcomeAssessmentAggregateOutcomeText(
        assessment.summary.outcome);
    for (const auto& member : assessment.members)
    {
        (void)RecommendationCampaignOutcomeAssessmentLifecycleStateText(
            member.lifecycle);
        (void)RecommendationCampaignOutcomeAssessmentConsistencyStateText(
            member.consistency);
        (void)RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
            member.outcome);
        for (const auto diagnostic : member.diagnostics)
            (void)RecommendationCampaignOutcomeAssessmentDiagnosticCodeText(
                diagnostic);
        for (const auto& comparison : member.comparisons)
        {
            (void)RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
                comparison.classification);
            for (const auto difference : comparison.contextDifferences)
                (void)RecommendationCampaignOutcomeAssessmentContextDifferenceText(
                    difference);
        }
    }
}

EvidenceClassification EvidenceClassificationFrom(
    AssessmentMemberOutcome outcome)
{
    switch (outcome)
    {
        case AssessmentMemberOutcome::Inconsistent:
            return EvidenceClassification::Inconsistent;
        case AssessmentMemberOutcome::NotReady:
            return EvidenceClassification::NotReady;
        case AssessmentMemberOutcome::TerminalFailed:
            return EvidenceClassification::TerminalFailed;
        case AssessmentMemberOutcome::TerminalCancelled:
            return EvidenceClassification::TerminalCancelled;
        case AssessmentMemberOutcome::SucceededContextChanged:
            return EvidenceClassification::SucceededContextChanged;
        case AssessmentMemberOutcome::SucceededMetricGap:
            return EvidenceClassification::SucceededMetricGap;
        case AssessmentMemberOutcome::SucceededComparable:
            return EvidenceClassification::SucceededComparable;
    }
    throw std::invalid_argument(
        "recommendation_campaign_outcome_policy_assessment_outcome_invalid");
}

const RecommendationCampaignOutcomeAssessmentMetricComparison* FindComparison(
    const AssessmentMember& member,
    const std::string& metricIdentity)
{
    const auto found = std::lower_bound(member.comparisons.begin(),
        member.comparisons.end(), metricIdentity,
        [](const auto& comparison, const std::string& identity)
    {
        return comparison.metricIdentity < identity;
    });
    return found != member.comparisons.end() &&
            found->metricIdentity == metricIdentity
        ? &*found
        : nullptr;
}

MetricJudgment JudgeDelta(double delta, MetricDirection direction)
{
    if (delta == 0.0) return MetricJudgment::Neutral;
    const bool favorable = direction == MetricDirection::HigherIsFavorable
        ? delta > 0.0
        : delta < 0.0;
    return favorable ? MetricJudgment::Favorable
                     : MetricJudgment::Unfavorable;
}

MemberInterpretation InterpretMetricJudgments(
    const std::vector<MetricEvaluation>& evaluations)
{
    bool favorable = false;
    bool unfavorable = false;
    for (const auto& evaluation : evaluations)
    {
        switch (evaluation.judgment)
        {
            case MetricJudgment::NotEvaluable:
                return MemberInterpretation::Inconclusive;
            case MetricJudgment::Favorable:
                favorable = true;
                break;
            case MetricJudgment::Neutral:
                break;
            case MetricJudgment::Unfavorable:
                unfavorable = true;
                break;
        }
    }
    if (favorable && unfavorable) return MemberInterpretation::Mixed;
    if (unfavorable) return MemberInterpretation::Unfavorable;
    if (favorable) return MemberInterpretation::Favorable;
    return MemberInterpretation::Neutral;
}

void AddMemberReason(
    std::vector<PolicyReason>& reasons,
    EvidenceClassification classification)
{
    switch (classification)
    {
        case EvidenceClassification::Inconsistent:
            AddReason(reasons, PolicyReason::InconsistentMemberPresent);
            break;
        case EvidenceClassification::NotReady:
            AddReason(reasons, PolicyReason::NotReadyMemberPresent);
            break;
        case EvidenceClassification::TerminalFailed:
            AddReason(reasons, PolicyReason::FailedMemberPresent);
            break;
        case EvidenceClassification::TerminalCancelled:
            AddReason(reasons, PolicyReason::CancelledMemberPresent);
            break;
        case EvidenceClassification::SucceededContextChanged:
            AddReason(reasons, PolicyReason::ContextChangedMemberPresent);
            break;
        case EvidenceClassification::SucceededMetricGap:
            AddReason(reasons, PolicyReason::MetricGapMemberPresent);
            break;
        case EvidenceClassification::SucceededComparable:
            break;
    }
}

void AddComparisonReason(
    std::vector<PolicyReason>& reasons,
    AssessmentClassification classification)
{
    switch (classification)
    {
        case AssessmentClassification::Comparable:
            break;
        case AssessmentClassification::ContextChanged:
            AddReason(reasons,
                PolicyReason::ContextChangedComparisonPresent);
            break;
        case AssessmentClassification::MissingSourceMetric:
            AddReason(reasons, PolicyReason::MissingSourceMetricPresent);
            break;
        case AssessmentClassification::MissingResultMetric:
            AddReason(reasons, PolicyReason::MissingResultMetricPresent);
            break;
        case AssessmentClassification::MetricValueUnavailable:
            AddReason(reasons, PolicyReason::MetricValueUnavailablePresent);
            break;
        case AssessmentClassification::Unsupported:
            AddReason(reasons, PolicyReason::UnsupportedMetricPresent);
            break;
    }
}

struct MutableCounts
{
    int inconsistent = 0;
    int notReady = 0;
    int failed = 0;
    int cancelled = 0;
    int contextChanged = 0;
    int metricGap = 0;
    int comparable = 0;
    std::size_t notEvaluableMetrics = 0;
    std::size_t favorableMetrics = 0;
    std::size_t neutralMetrics = 0;
    std::size_t unfavorableMetrics = 0;
};

void CountMember(MutableCounts& counts, const PolicyMember& member)
{
    switch (member.evidenceClassification)
    {
        case EvidenceClassification::Inconsistent:
            ++counts.inconsistent;
            break;
        case EvidenceClassification::NotReady:
            ++counts.notReady;
            break;
        case EvidenceClassification::TerminalFailed:
            ++counts.failed;
            break;
        case EvidenceClassification::TerminalCancelled:
            ++counts.cancelled;
            break;
        case EvidenceClassification::SucceededContextChanged:
            ++counts.contextChanged;
            break;
        case EvidenceClassification::SucceededMetricGap:
            ++counts.metricGap;
            break;
        case EvidenceClassification::SucceededComparable:
            ++counts.comparable;
            break;
    }
    for (const auto& evaluation : member.metricEvaluations)
    {
        switch (evaluation.judgment)
        {
            case MetricJudgment::NotEvaluable:
                ++counts.notEvaluableMetrics;
                break;
            case MetricJudgment::Favorable:
                ++counts.favorableMetrics;
                break;
            case MetricJudgment::Neutral:
                ++counts.neutralMetrics;
                break;
            case MetricJudgment::Unfavorable:
                ++counts.unfavorableMetrics;
                break;
        }
    }
}

CampaignInterpretation DeriveCampaignInterpretation(
    const MutableCounts& counts,
    int memberCount,
    int minimumComparableMemberCount,
    const std::vector<PolicyMember>& members)
{
    if (counts.inconsistent > 0 || counts.notReady > 0 ||
        counts.cancelled > 0 || counts.contextChanged > 0 ||
        counts.metricGap > 0 ||
        (counts.comparable < minimumComparableMemberCount &&
            counts.failed != memberCount))
        return CampaignInterpretation::Inconclusive;
    if (counts.failed == memberCount)
        return CampaignInterpretation::Unfavorable;

    bool favorable = false;
    bool unfavorable = counts.failed > 0;
    for (const auto& member : members)
    {
        switch (member.interpretation)
        {
            case MemberInterpretation::Inconclusive:
                return CampaignInterpretation::Inconclusive;
            case MemberInterpretation::Favorable:
                favorable = true;
                break;
            case MemberInterpretation::Neutral:
                break;
            case MemberInterpretation::Unfavorable:
                unfavorable = true;
                break;
            case MemberInterpretation::Mixed:
                favorable = true;
                unfavorable = true;
                break;
        }
    }
    if (favorable && unfavorable) return CampaignInterpretation::Mixed;
    if (unfavorable) return CampaignInterpretation::Unfavorable;
    if (favorable) return CampaignInterpretation::Favorable;
    return CampaignInterpretation::Neutral;
}

void AppendReasons(
    std::ostringstream& out,
    const std::vector<PolicyReason>& reasons)
{
    out << reasons.size();
    for (const auto reason : reasons)
        out << ',' << RecommendationCampaignOutcomePolicyReasonText(reason);
}

void AppendMetricEvaluation(
    std::ostringstream& out,
    const MetricEvaluation& evaluation)
{
    out << LengthPrefixed(evaluation.metricIdentity) << ','
        << RecommendationCampaignOutcomePolicyMetricDirectionText(
               evaluation.direction)
        << ",assessment_classification="
        << OptionalClassification(evaluation.evidenceClassification)
        << ",assessment_delta="
        << OptionalDouble(evaluation.assessmentDelta)
        << ",policy_judgment="
        << RecommendationCampaignOutcomePolicyMetricJudgmentText(
               evaluation.judgment);
}

std::string DecisionCanonicalText(
    const RecommendationCampaignOutcomePolicy& policy,
    const Assessment& assessment,
    const RecommendationCampaignOutcomePolicySummary& summary,
    const std::vector<PolicyMember>& members)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_outcome_policy_decision_v1"
        << ";decision_contract_version="
        << kRecommendationCampaignOutcomePolicyDecisionContractVersion
        << ";policy_contract_version=" << policy.identity.contractVersion
        << ";policy_canonical="
        << LengthPrefixed(policy.identity.canonicalText)
        << ";policy_hash=" << LengthPrefixed(policy.identity.hash)
        << ";assessment_contract_version="
        << assessment.identity.contractVersion
        << ";assessment_canonical="
        << LengthPrefixed(assessment.identity.canonicalText)
        << ";assessment_hash=" << LengthPrefixed(assessment.identity.hash)
        << ";campaign_approval_id="
        << assessment.campaignIdentity.campaignApprovalId
        << ";campaign_identity="
        << LengthPrefixed(assessment.campaignIdentity.identityCanonical)
        << ";campaign_identity_hash="
        << LengthPrefixed(assessment.campaignIdentity.identityHash)
        << ";materialization_id="
        << assessment.materializationIdentity.materializationId
        << ";materialization_campaign_approval_id="
        << assessment.materializationIdentity.campaignApprovalId
        << ";materialization_campaign_identity_hash="
        << LengthPrefixed(
               assessment.materializationIdentity.campaignIdentityHash)
        << ";materialization_contract_version="
        << assessment.materializationIdentity.contractVersion
        << ";materialization_member_count="
        << assessment.materializationIdentity.memberCount
        << ";materialization_identity="
        << LengthPrefixed(
               assessment.materializationIdentity.identityCanonical)
        << ";materialization_identity_hash="
        << LengthPrefixed(assessment.materializationIdentity.identityHash);
    for (const auto& member : members)
    {
        out << ";member=" << member.identity.memberOrdinal << ','
            << member.identity.materializationMemberId << ','
            << member.identity.rankingMemberId << ','
            << member.identity.recommendationId << ','
            << member.identity.sourceExperimentId << ','
            << member.identity.proposalId
            << ",expected_experiment_id="
            << OptionalLongLong(member.identity.expectedExperimentId)
            << ",assessment_lifecycle="
            << RecommendationCampaignOutcomeAssessmentLifecycleStateText(
                   member.lifecycle)
            << ",assessment_consistency="
            << RecommendationCampaignOutcomeAssessmentConsistencyStateText(
                   member.consistency)
            << ",assessment_outcome="
            << RecommendationCampaignOutcomeAssessmentMemberOutcomeText(
                   member.assessmentOutcome)
            << ",assessment_diagnostic_count="
            << member.assessmentDiagnostics.size();
        for (const auto diagnostic : member.assessmentDiagnostics)
            out << ','
                << RecommendationCampaignOutcomeAssessmentDiagnosticCodeText(
                       diagnostic);
        out << ",evidence_classification="
            << RecommendationCampaignOutcomePolicyEvidenceClassificationText(
                   member.evidenceClassification)
            << ",policy_interpretation="
            << RecommendationCampaignOutcomePolicyMemberInterpretationText(
                   member.interpretation)
            << ",metric_evaluation_count="
            << member.metricEvaluations.size();
        for (const auto& evaluation : member.metricEvaluations)
        {
            out << ',';
            AppendMetricEvaluation(out, evaluation);
        }
    }
    out << ";evidence_sufficiency="
        << RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
               summary.evidenceSufficiency)
        << ";campaign_interpretation="
        << RecommendationCampaignOutcomePolicyCampaignInterpretationText(
               summary.campaignInterpretation)
        << ";follow_up_eligibility="
        << RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
               summary.followUpEligibility)
        << ";follow_up_authorized="
        << (summary.followUpAuthorized ? "true" : "false")
        << ";member_count=" << summary.memberCount
        << ";inconsistent_member_count=" << summary.inconsistentMemberCount
        << ";not_ready_member_count=" << summary.notReadyMemberCount
        << ";failed_member_count=" << summary.failedMemberCount
        << ";cancelled_member_count=" << summary.cancelledMemberCount
        << ";context_changed_member_count="
        << summary.contextChangedMemberCount
        << ";metric_gap_member_count=" << summary.metricGapMemberCount
        << ";comparable_member_count=" << summary.comparableMemberCount
        << ";metric_evaluation_count=" << summary.metricEvaluationCount
        << ";not_evaluable_metric_count="
        << summary.notEvaluableMetricCount
        << ";favorable_metric_count=" << summary.favorableMetricCount
        << ";neutral_metric_count=" << summary.neutralMetricCount
        << ";unfavorable_metric_count=" << summary.unfavorableMetricCount
        << ";evidence_reasons=";
    AppendReasons(out, summary.evidenceReasons);
    out << ";interpretation_reasons=";
    AppendReasons(out, summary.interpretationReasons);
    out << ";follow_up_reasons=";
    AppendReasons(out, summary.followUpReasons);
    return out.str();
}

} // namespace

void ValidateRecommendationCampaignOutcomePolicyEvidence(
    const RecommendationCampaignOutcomePolicyValidationView& view)
{
    if (view.policyContractVersion !=
        kRecommendationCampaignOutcomePolicyContractVersion)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_version_unsupported");
    RequireText(view.policyCanonicalText,
        "recommendation_campaign_outcome_policy_identity_missing");
    if (view.policyIdentityHash !=
        RecommendationCanonicalHash(view.policyCanonicalText))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_identity_invalid");

    if (view.assessmentContractVersion !=
        kRecommendationCampaignOutcomeAssessmentContractVersion)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_assessment_version_unsupported");
    RequireText(view.assessmentCanonicalText,
        "recommendation_campaign_outcome_policy_assessment_identity_missing");
    if (view.assessmentIdentityHash !=
        RecommendationCanonicalHash(view.assessmentCanonicalText))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_assessment_identity_invalid");

    RequireText(view.campaignIdentityCanonical,
        "recommendation_campaign_outcome_policy_campaign_identity_missing");
    if (view.campaignApprovalId <= 0 ||
        view.campaignIdentityHash !=
            RecommendationCanonicalHash(view.campaignIdentityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_campaign_identity_invalid");

    RequireText(view.materializationIdentityCanonical,
        "recommendation_campaign_outcome_policy_materialization_identity_missing");
    if (view.materializationId <= 0 ||
        view.materializationCampaignApprovalId <= 0 ||
        view.materializationContractVersion <= 0 ||
        view.materializationIdentityHash != RecommendationCanonicalHash(
            view.materializationIdentityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_materialization_identity_invalid");
    if (view.materializationCampaignApprovalId != view.campaignApprovalId ||
        view.materializationCampaignIdentityHash !=
            view.campaignIdentityHash)
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_campaign_materialization_mismatch");

    if (view.materializationMemberCount !=
            static_cast<int>(view.members.size()) ||
        view.assessmentSummaryMemberCount !=
            static_cast<int>(view.members.size()) ||
        view.members.empty() ||
        view.members.size() > static_cast<std::size_t>(
            kMaximumRecommendationCampaignOutcomeAssessmentMembers))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_assessment_shape_invalid");

    std::set<long long> materializationMemberIds;
    std::set<long long> rankingMemberIds;
    std::set<long long> recommendationIds;
    std::set<long long> proposalIds;
    for (std::size_t index = 0; index < view.members.size(); ++index)
    {
        const auto& member = view.members[index];
        if (member.memberOrdinal != static_cast<int>(index) + 1 ||
            member.materializationMemberId <= 0 ||
            member.rankingMemberId <= 0 || member.recommendationId <= 0 ||
            member.sourceExperimentId <= 0 || member.proposalId <= 0 ||
            (member.expectedExperimentId &&
                *member.expectedExperimentId <= 0) ||
            !materializationMemberIds
                 .insert(member.materializationMemberId).second ||
            !rankingMemberIds.insert(member.rankingMemberId).second ||
            !recommendationIds.insert(member.recommendationId).second ||
            !proposalIds.insert(member.proposalId).second)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_policy_member_identity_set_invalid");
        std::set<std::string> comparisonIdentities;
        for (const auto& identity : member.comparisonMetricIdentities)
        {
            RequireText(identity,
                "recommendation_campaign_outcome_policy_comparison_identity_invalid");
            if (!comparisonIdentities.insert(identity).second)
                throw std::invalid_argument(
                    "recommendation_campaign_outcome_policy_comparison_identity_duplicate");
        }
    }
}

RecommendationCampaignOutcomePolicyMetricRule::
    RecommendationCampaignOutcomePolicyMetricRule(
        std::string metricIdentityValue,
        MetricDirection directionValue)
    : metricIdentity(std::move(metricIdentityValue)),
      direction(directionValue)
{
    RequireText(metricIdentity,
        "recommendation_campaign_outcome_policy_metric_identity_invalid");
    RequireValidDirection(direction);
}

RecommendationCampaignOutcomePolicyIdentity::
    RecommendationCampaignOutcomePolicyIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
    if (contractVersion !=
            kRecommendationCampaignOutcomePolicyContractVersion ||
        hash != RecommendationCanonicalHash(canonicalText))
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_identity_invariant_failed");
}

RecommendationCampaignOutcomePolicy::RecommendationCampaignOutcomePolicy(
    int minimumComparableMemberCountValue,
    std::vector<MetricRule> requiredMetricRulesValue,
    RecommendationCampaignOutcomePolicyIdentity identityValue)
    : minimumComparableMemberCount(minimumComparableMemberCountValue),
      requiredMetricRules(std::move(requiredMetricRulesValue)),
      identity(std::move(identityValue))
{
    if (minimumComparableMemberCount <= 0 ||
        minimumComparableMemberCount >
            kMaximumRecommendationCampaignOutcomeAssessmentMembers ||
        requiredMetricRules.empty())
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_invariant_failed");
}

RecommendationCampaignOutcomePolicyMetricEvaluation::
    RecommendationCampaignOutcomePolicyMetricEvaluation(
        std::string metricIdentityValue,
        MetricDirection directionValue,
        std::optional<AssessmentClassification> evidenceClassificationValue,
        std::optional<double> assessmentDeltaValue,
        MetricJudgment judgmentValue)
    : metricIdentity(std::move(metricIdentityValue)),
      direction(directionValue),
      evidenceClassification(evidenceClassificationValue),
      assessmentDelta(assessmentDeltaValue && *assessmentDeltaValue == 0.0
              ? std::optional<double>{0.0}
              : assessmentDeltaValue),
      judgment(judgmentValue)
{
    RequireText(metricIdentity,
        "recommendation_campaign_outcome_policy_metric_evaluation_identity_invalid");
    RequireValidDirection(direction);
    RequireValidMetricJudgment(judgment);
    if (evidenceClassification)
        (void)RecommendationCampaignOutcomeAssessmentMetricComparisonClassificationText(
            *evidenceClassification);
    if (assessmentDelta && !std::isfinite(*assessmentDelta))
        throw std::invalid_argument(
            "recommendation_campaign_outcome_policy_metric_delta_nonfinite");
    const bool evaluable = judgment != MetricJudgment::NotEvaluable;
    if (evaluable !=
            (evidenceClassification ==
                    AssessmentClassification::Comparable &&
                assessmentDelta.has_value()))
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_metric_evaluation_invariant_failed");
}

RecommendationCampaignOutcomePolicyMember::
    RecommendationCampaignOutcomePolicyMember(
        RecommendationCampaignOutcomeAssessmentMemberIdentity identityValue,
        RecommendationCampaignOutcomeAssessmentLifecycleState lifecycleValue,
        AssessmentConsistency consistencyValue,
        AssessmentMemberOutcome assessmentOutcomeValue,
        std::vector<RecommendationCampaignOutcomeAssessmentDiagnosticCode>
            assessmentDiagnosticsValue,
        EvidenceClassification evidenceClassificationValue,
        MemberInterpretation interpretationValue,
        std::vector<MetricEvaluation> metricEvaluationsValue)
    : identity(std::move(identityValue)),
      lifecycle(lifecycleValue),
      consistency(consistencyValue),
      assessmentOutcome(assessmentOutcomeValue),
      assessmentDiagnostics(std::move(assessmentDiagnosticsValue)),
      evidenceClassification(evidenceClassificationValue),
      interpretation(interpretationValue),
      metricEvaluations(std::move(metricEvaluationsValue))
{
    (void)RecommendationCampaignOutcomePolicyEvidenceClassificationText(
        evidenceClassification);
    RequireValidMemberInterpretation(interpretation);
}

RecommendationCampaignOutcomePolicySummary::
    RecommendationCampaignOutcomePolicySummary(
        EvidenceSufficiency evidenceSufficiencyValue,
        CampaignInterpretation campaignInterpretationValue,
        FollowUpEligibility followUpEligibilityValue,
        bool followUpAuthorizedValue,
        int memberCountValue,
        int inconsistentMemberCountValue,
        int notReadyMemberCountValue,
        int failedMemberCountValue,
        int cancelledMemberCountValue,
        int contextChangedMemberCountValue,
        int metricGapMemberCountValue,
        int comparableMemberCountValue,
        std::size_t metricEvaluationCountValue,
        std::size_t notEvaluableMetricCountValue,
        std::size_t favorableMetricCountValue,
        std::size_t neutralMetricCountValue,
        std::size_t unfavorableMetricCountValue,
        std::vector<PolicyReason> evidenceReasonsValue,
        std::vector<PolicyReason> interpretationReasonsValue,
        std::vector<PolicyReason> followUpReasonsValue)
    : evidenceSufficiency(evidenceSufficiencyValue),
      campaignInterpretation(campaignInterpretationValue),
      followUpEligibility(followUpEligibilityValue),
      followUpAuthorized(followUpAuthorizedValue),
      memberCount(memberCountValue),
      inconsistentMemberCount(inconsistentMemberCountValue),
      notReadyMemberCount(notReadyMemberCountValue),
      failedMemberCount(failedMemberCountValue),
      cancelledMemberCount(cancelledMemberCountValue),
      contextChangedMemberCount(contextChangedMemberCountValue),
      metricGapMemberCount(metricGapMemberCountValue),
      comparableMemberCount(comparableMemberCountValue),
      metricEvaluationCount(metricEvaluationCountValue),
      notEvaluableMetricCount(notEvaluableMetricCountValue),
      favorableMetricCount(favorableMetricCountValue),
      neutralMetricCount(neutralMetricCountValue),
      unfavorableMetricCount(unfavorableMetricCountValue),
      evidenceReasons(std::move(evidenceReasonsValue)),
      interpretationReasons(std::move(interpretationReasonsValue)),
      followUpReasons(std::move(followUpReasonsValue))
{
    const int classifiedMembers = inconsistentMemberCount +
        notReadyMemberCount + failedMemberCount + cancelledMemberCount +
        contextChangedMemberCount + metricGapMemberCount +
        comparableMemberCount;
    const std::size_t classifiedMetrics = notEvaluableMetricCount +
        favorableMetricCount + neutralMetricCount + unfavorableMetricCount;
    if (memberCount <= 0 || classifiedMembers != memberCount ||
        classifiedMetrics != metricEvaluationCount || followUpAuthorized)
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_summary_invariant_failed");
}

RecommendationCampaignOutcomePolicyDecisionIdentity::
    RecommendationCampaignOutcomePolicyDecisionIdentity(
        int contractVersionValue,
        std::string canonicalTextValue,
        std::string hashValue)
    : contractVersion(contractVersionValue),
      canonicalText(std::move(canonicalTextValue)),
      hash(std::move(hashValue))
{
    if (contractVersion !=
            kRecommendationCampaignOutcomePolicyDecisionContractVersion ||
        hash != RecommendationCanonicalHash(canonicalText))
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_decision_identity_invariant_failed");
}

RecommendationCampaignOutcomePolicyDecision::
    RecommendationCampaignOutcomePolicyDecision(
        RecommendationCampaignOutcomePolicyDecisionIdentity identityValue,
        RecommendationCampaignOutcomePolicy policyValue,
        int assessmentContractVersionValue,
        std::string assessmentCanonicalTextValue,
        std::string assessmentIdentityHashValue,
        RecommendationCampaignOutcomeAssessmentCampaignIdentity
            campaignIdentityValue,
        RecommendationCampaignOutcomeAssessmentMaterializationIdentity
            materializationIdentityValue,
        std::string observedAtValue,
        RecommendationCampaignOutcomePolicySummary summaryValue,
        std::vector<PolicyMember> membersValue)
    : identity(std::move(identityValue)),
      policy(std::move(policyValue)),
      assessmentContractVersion(assessmentContractVersionValue),
      assessmentCanonicalText(std::move(assessmentCanonicalTextValue)),
      assessmentIdentityHash(std::move(assessmentIdentityHashValue)),
      campaignIdentity(std::move(campaignIdentityValue)),
      materializationIdentity(std::move(materializationIdentityValue)),
      observedAt(std::move(observedAtValue)),
      summary(std::move(summaryValue)),
      members(std::move(membersValue))
{
    if (assessmentContractVersion !=
            kRecommendationCampaignOutcomeAssessmentContractVersion ||
        assessmentIdentityHash !=
            RecommendationCanonicalHash(assessmentCanonicalText) ||
        summary.memberCount != static_cast<int>(members.size()))
        throw std::logic_error(
            "recommendation_campaign_outcome_policy_decision_invariant_failed");
}

struct RecommendationCampaignOutcomePolicyBuilder
{
    static RecommendationCampaignOutcomePolicy Build(
        const RecommendationCampaignOutcomePolicyInput& input)
    {
        if (input.contractVersion !=
            kRecommendationCampaignOutcomePolicyContractVersion)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_policy_version_unsupported");
        if (input.minimumComparableMemberCount <= 0 ||
            input.minimumComparableMemberCount >
                kMaximumRecommendationCampaignOutcomeAssessmentMembers)
            throw std::invalid_argument(
                "recommendation_campaign_outcome_policy_minimum_evidence_invalid");
        auto rules = CanonicalRules(input.requiredMetricRules);
        std::string canonical = PolicyCanonicalText(
            input.minimumComparableMemberCount, rules);
        RecommendationCampaignOutcomePolicyIdentity identity(
            kRecommendationCampaignOutcomePolicyContractVersion,
            canonical, RecommendationCanonicalHash(canonical));
        return RecommendationCampaignOutcomePolicy(
            input.minimumComparableMemberCount, std::move(rules),
            std::move(identity));
    }
};

struct RecommendationCampaignOutcomePolicyDecisionBuilder
{
    static RecommendationCampaignOutcomePolicyDecision Build(
        const RecommendationCampaignOutcomePolicy& policy,
        const Assessment& assessment)
    {
        ValidateAssessment(policy, assessment);

        std::vector<PolicyMember> members;
        members.reserve(assessment.members.size());
        std::vector<PolicyReason> evidenceReasons;
        std::vector<PolicyReason> interpretationReasons;
        bool requiredMetricMissing = false;
        bool requiredMetricNotComparable = false;

        for (const auto& assessed : assessment.members)
        {
            const auto classification =
                EvidenceClassificationFrom(assessed.outcome);
            AddMemberReason(interpretationReasons, classification);
            switch (classification)
            {
                case EvidenceClassification::Inconsistent:
                case EvidenceClassification::NotReady:
                case EvidenceClassification::TerminalCancelled:
                case EvidenceClassification::SucceededContextChanged:
                case EvidenceClassification::SucceededMetricGap:
                    AddMemberReason(evidenceReasons, classification);
                    break;
                case EvidenceClassification::TerminalFailed:
                case EvidenceClassification::SucceededComparable:
                    break;
            }
            for (const auto& comparison : assessed.comparisons)
            {
                AddComparisonReason(
                    interpretationReasons, comparison.classification);
                AddComparisonReason(evidenceReasons, comparison.classification);
            }

            std::vector<MetricEvaluation> evaluations;
            evaluations.reserve(policy.requiredMetricRules.size());
            const bool successfulEvidence =
                classification ==
                    EvidenceClassification::SucceededContextChanged ||
                classification ==
                    EvidenceClassification::SucceededMetricGap ||
                classification ==
                    EvidenceClassification::SucceededComparable;
            for (const auto& rule : policy.requiredMetricRules)
            {
                const auto* comparison =
                    FindComparison(assessed, rule.metricIdentity);
                if (!comparison)
                {
                    if (successfulEvidence) requiredMetricMissing = true;
                    evaluations.push_back(MetricEvaluation(
                        rule.metricIdentity, rule.direction, std::nullopt,
                        std::nullopt, MetricJudgment::NotEvaluable));
                    continue;
                }
                if (comparison->classification !=
                        AssessmentClassification::Comparable ||
                    !comparison->delta)
                {
                    if (successfulEvidence)
                        requiredMetricNotComparable = true;
                    evaluations.push_back(MetricEvaluation(
                        rule.metricIdentity, rule.direction,
                        comparison->classification, std::nullopt,
                        MetricJudgment::NotEvaluable));
                    continue;
                }
                evaluations.push_back(MetricEvaluation(rule.metricIdentity,
                    rule.direction, comparison->classification,
                    comparison->delta,
                    JudgeDelta(*comparison->delta, rule.direction)));
            }

            MemberInterpretation interpretation =
                MemberInterpretation::Inconclusive;
            if (classification == EvidenceClassification::TerminalFailed)
                interpretation = MemberInterpretation::Unfavorable;
            else if (classification ==
                EvidenceClassification::SucceededComparable)
                interpretation = InterpretMetricJudgments(evaluations);

            members.push_back(PolicyMember(assessed.identity, assessed.lifecycle,
                assessed.consistency, assessed.outcome,
                assessed.diagnostics, classification, interpretation,
                std::move(evaluations)));
        }

        if (requiredMetricMissing)
        {
            AddReason(evidenceReasons, PolicyReason::RequiredMetricMissing);
            AddReason(interpretationReasons,
                PolicyReason::RequiredMetricMissing);
        }
        if (requiredMetricNotComparable)
        {
            AddReason(evidenceReasons,
                PolicyReason::RequiredMetricNotComparable);
            AddReason(interpretationReasons,
                PolicyReason::RequiredMetricNotComparable);
        }

        MutableCounts counts;
        for (const auto& member : members) CountMember(counts, member);
        if (counts.comparable < policy.minimumComparableMemberCount)
        {
            AddReason(evidenceReasons,
                PolicyReason::MinimumComparableMembersNotMet);
            AddReason(interpretationReasons,
                PolicyReason::MinimumComparableMembersNotMet);
        }
        if (counts.favorableMetrics > 0)
            AddReason(interpretationReasons,
                PolicyReason::FavorableMetricEvidencePresent);
        if (counts.neutralMetrics > 0)
            AddReason(interpretationReasons,
                PolicyReason::NeutralMetricEvidencePresent);
        if (counts.unfavorableMetrics > 0)
            AddReason(interpretationReasons,
                PolicyReason::UnfavorableMetricEvidencePresent);
        CanonicalizeReasons(evidenceReasons);
        CanonicalizeReasons(interpretationReasons);

        const EvidenceSufficiency sufficiency = evidenceReasons.empty()
            ? EvidenceSufficiency::Sufficient
            : EvidenceSufficiency::Insufficient;
        const CampaignInterpretation campaignInterpretation =
            DeriveCampaignInterpretation(counts,
                static_cast<int>(members.size()),
                policy.minimumComparableMemberCount, members);
        const bool eligible =
            sufficiency == EvidenceSufficiency::Sufficient &&
            counts.comparable == static_cast<int>(members.size()) &&
            campaignInterpretation == CampaignInterpretation::Favorable;
        std::vector<PolicyReason> followUpReasons;
        if (eligible)
            AddReason(followUpReasons,
                PolicyReason::AllMembersSucceededComparable);
        else
        {
            if (sufficiency == EvidenceSufficiency::Insufficient)
                AddReason(followUpReasons,
                    PolicyReason::EvidenceInsufficient);
            if (campaignInterpretation != CampaignInterpretation::Favorable)
                AddReason(followUpReasons,
                    PolicyReason::FavorableCampaignInterpretationRequired);
            for (const auto& member : members)
                AddMemberReason(
                    followUpReasons, member.evidenceClassification);
        }
        CanonicalizeReasons(followUpReasons);

        const std::size_t metricEvaluationCount = members.size() *
            policy.requiredMetricRules.size();
        RecommendationCampaignOutcomePolicySummary summary(sufficiency,
            campaignInterpretation,
            eligible ? FollowUpEligibility::EligibleForOperatorReview
                     : FollowUpEligibility::NotEligible,
            false, static_cast<int>(members.size()), counts.inconsistent,
            counts.notReady, counts.failed, counts.cancelled,
            counts.contextChanged, counts.metricGap, counts.comparable,
            metricEvaluationCount, counts.notEvaluableMetrics,
            counts.favorableMetrics, counts.neutralMetrics,
            counts.unfavorableMetrics, std::move(evidenceReasons),
            std::move(interpretationReasons),
            std::move(followUpReasons));
        std::string canonical = DecisionCanonicalText(
            policy, assessment, summary, members);
        RecommendationCampaignOutcomePolicyDecisionIdentity identity(
            kRecommendationCampaignOutcomePolicyDecisionContractVersion,
            canonical, RecommendationCanonicalHash(canonical));
        return RecommendationCampaignOutcomePolicyDecision(
            std::move(identity), policy, assessment.identity.contractVersion,
            assessment.identity.canonicalText, assessment.identity.hash,
            assessment.campaignIdentity,
            assessment.materializationIdentity, assessment.observedAt,
            std::move(summary), std::move(members));
    }
};

RecommendationCampaignOutcomePolicy BuildRecommendationCampaignOutcomePolicy(
    const RecommendationCampaignOutcomePolicyInput& input)
{
    return RecommendationCampaignOutcomePolicyBuilder::Build(input);
}

RecommendationCampaignOutcomePolicyDecision
ApplyRecommendationCampaignOutcomePolicy(
    const RecommendationCampaignOutcomePolicy& policy,
    const Assessment& assessment)
{
    return RecommendationCampaignOutcomePolicyDecisionBuilder::Build(
        policy, assessment);
}

#define EA_OUTCOME_POLICY_TEXT_FUNCTION(name, type, ...)                    \
    std::string name(type value)                                             \
    {                                                                         \
        switch (value)                                                        \
        {                                                                     \
            __VA_ARGS__                                                       \
        }                                                                     \
        throw std::invalid_argument(                                          \
            "recommendation_campaign_outcome_policy_enum_invalid");         \
    }
#define EA_OUTCOME_POLICY_TEXT_CASE(value, text) case value: return text;

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyMetricDirectionText,
    MetricDirection,
    EA_OUTCOME_POLICY_TEXT_CASE(MetricDirection::HigherIsFavorable,
        "higher_is_favorable")
    EA_OUTCOME_POLICY_TEXT_CASE(MetricDirection::LowerIsFavorable,
        "lower_is_favorable"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyEvidenceClassificationText,
    EvidenceClassification,
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::Inconsistent,
        "inconsistent")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::NotReady,
        "not_ready")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::TerminalFailed,
        "terminal_failed")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::TerminalCancelled,
        "terminal_cancelled")
    EA_OUTCOME_POLICY_TEXT_CASE(
        EvidenceClassification::SucceededContextChanged,
        "succeeded_context_changed")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::SucceededMetricGap,
        "succeeded_metric_gap")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceClassification::SucceededComparable,
        "succeeded_comparable"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyMetricJudgmentText,
    MetricJudgment,
    EA_OUTCOME_POLICY_TEXT_CASE(MetricJudgment::NotEvaluable,
        "not_evaluable")
    EA_OUTCOME_POLICY_TEXT_CASE(MetricJudgment::Favorable, "favorable")
    EA_OUTCOME_POLICY_TEXT_CASE(MetricJudgment::Neutral, "neutral")
    EA_OUTCOME_POLICY_TEXT_CASE(MetricJudgment::Unfavorable, "unfavorable"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyMemberInterpretationText,
    MemberInterpretation,
    EA_OUTCOME_POLICY_TEXT_CASE(MemberInterpretation::Inconclusive,
        "inconclusive")
    EA_OUTCOME_POLICY_TEXT_CASE(MemberInterpretation::Favorable, "favorable")
    EA_OUTCOME_POLICY_TEXT_CASE(MemberInterpretation::Neutral, "neutral")
    EA_OUTCOME_POLICY_TEXT_CASE(MemberInterpretation::Unfavorable,
        "unfavorable")
    EA_OUTCOME_POLICY_TEXT_CASE(MemberInterpretation::Mixed, "mixed"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyEvidenceSufficiencyText,
    EvidenceSufficiency,
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceSufficiency::Insufficient,
        "insufficient")
    EA_OUTCOME_POLICY_TEXT_CASE(EvidenceSufficiency::Sufficient,
        "sufficient"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyCampaignInterpretationText,
    CampaignInterpretation,
    EA_OUTCOME_POLICY_TEXT_CASE(CampaignInterpretation::Inconclusive,
        "inconclusive")
    EA_OUTCOME_POLICY_TEXT_CASE(CampaignInterpretation::Favorable,
        "favorable")
    EA_OUTCOME_POLICY_TEXT_CASE(CampaignInterpretation::Neutral, "neutral")
    EA_OUTCOME_POLICY_TEXT_CASE(CampaignInterpretation::Unfavorable,
        "unfavorable")
    EA_OUTCOME_POLICY_TEXT_CASE(CampaignInterpretation::Mixed, "mixed"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyFollowUpEligibilityText,
    FollowUpEligibility,
    EA_OUTCOME_POLICY_TEXT_CASE(FollowUpEligibility::NotEligible,
        "not_eligible")
    EA_OUTCOME_POLICY_TEXT_CASE(
        FollowUpEligibility::EligibleForOperatorReview,
        "eligible_for_operator_review"))

EA_OUTCOME_POLICY_TEXT_FUNCTION(
    RecommendationCampaignOutcomePolicyReasonText,
    PolicyReason,
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::InconsistentMemberPresent,
        "inconsistent_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::NotReadyMemberPresent,
        "not_ready_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::FailedMemberPresent,
        "failed_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::CancelledMemberPresent,
        "cancelled_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::ContextChangedMemberPresent,
        "context_changed_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::MetricGapMemberPresent,
        "metric_gap_member_present")
    EA_OUTCOME_POLICY_TEXT_CASE(
        PolicyReason::ContextChangedComparisonPresent,
        "context_changed_comparison_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::MissingSourceMetricPresent,
        "missing_source_metric_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::MissingResultMetricPresent,
        "missing_result_metric_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::MetricValueUnavailablePresent,
        "metric_value_unavailable_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::UnsupportedMetricPresent,
        "unsupported_metric_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::RequiredMetricMissing,
        "required_metric_missing")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::RequiredMetricNotComparable,
        "required_metric_not_comparable")
    EA_OUTCOME_POLICY_TEXT_CASE(
        PolicyReason::MinimumComparableMembersNotMet,
        "minimum_comparable_members_not_met")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::FavorableMetricEvidencePresent,
        "favorable_metric_evidence_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::NeutralMetricEvidencePresent,
        "neutral_metric_evidence_present")
    EA_OUTCOME_POLICY_TEXT_CASE(
        PolicyReason::UnfavorableMetricEvidencePresent,
        "unfavorable_metric_evidence_present")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::EvidenceInsufficient,
        "evidence_insufficient")
    EA_OUTCOME_POLICY_TEXT_CASE(
        PolicyReason::FavorableCampaignInterpretationRequired,
        "favorable_campaign_interpretation_required")
    EA_OUTCOME_POLICY_TEXT_CASE(PolicyReason::AllMembersSucceededComparable,
        "all_members_succeeded_comparable"))

#undef EA_OUTCOME_POLICY_TEXT_CASE
#undef EA_OUTCOME_POLICY_TEXT_FUNCTION

} // namespace EA::ExperimentRecommendation
