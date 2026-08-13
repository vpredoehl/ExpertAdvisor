#include "ExperimentRecommendationCampaignOutcomePolicy.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <cassert>
#include <locale>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

using namespace EA::ExperimentRecommendation;

namespace
{

using Assessment = RecommendationCampaignOutcomeAssessment;
using AssessmentIdentity = RecommendationCampaignOutcomeAssessmentIdentity;
using CampaignInterpretation =
    RecommendationCampaignOutcomePolicyCampaignInterpretation;
using Classification =
    RecommendationCampaignOutcomeAssessmentMetricComparisonClassification;
using Consistency = RecommendationCampaignOutcomeAssessmentConsistencyState;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Decision = RecommendationCampaignOutcomePolicyDecision;
using Direction = RecommendationCampaignOutcomePolicyMetricDirection;
using EvidenceClassification =
    RecommendationCampaignOutcomePolicyEvidenceClassification;
using EvidenceSufficiency =
    RecommendationCampaignOutcomePolicyEvidenceSufficiency;
using FollowUpEligibility =
    RecommendationCampaignOutcomePolicyFollowUpEligibility;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using MemberEvidence = RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MemberIdentity = RecommendationCampaignOutcomeAssessmentMemberIdentity;
using MemberInterpretation =
    RecommendationCampaignOutcomePolicyMemberInterpretation;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricJudgment = RecommendationCampaignOutcomePolicyMetricJudgment;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using Policy = RecommendationCampaignOutcomePolicy;
using PolicyIdentity = RecommendationCampaignOutcomePolicyIdentity;
using PolicyInput = RecommendationCampaignOutcomePolicyInput;
using PolicyReason = RecommendationCampaignOutcomePolicyReason;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;
using ValidationView = RecommendationCampaignOutcomePolicyValidationView;

constexpr char kExpectedDefaultPolicyCanonical[] =
    "experiment_recommendation_campaign_outcome_policy_v1;policy_contract_version=1;minimum_c"
    "omparable_member_count=1;required_metric_rule_count=2;required_metric=18:inference_accur"
    "acy,higher_is_favorable;required_metric=12:leader_score,higher_is_favorable;all_members_"
    "consistent_required=true;all_members_terminal_required=true;all_successful_members_compa"
    "rable_required=true;follow_up_requires_all_members_comparable_success=true;follow_up_req"
    "uires_favorable_campaign_interpretation=true;follow_up_authorizing=false";
constexpr char kExpectedDefaultPolicyHash[] = "fnv1a64:83a08cff5aad41ad";
constexpr char kExpectedGoldenDecisionCanonical[] =
    "experiment_recommendation_campaign_outcome_policy_decision_v1;decision_contract_version="
    "1;policy_contract_version=1;policy_canonical=512:experiment_recommendation_campaign_outc"
    "ome_policy_v1;policy_contract_version=1;minimum_comparable_member_count=1;required_metri"
    "c_rule_count=2;required_metric=18:inference_accuracy,higher_is_favorable;required_metric"
    "=12:leader_score,higher_is_favorable;all_members_consistent_required=true;all_members_te"
    "rminal_required=true;all_successful_members_comparable_required=true;follow_up_requires_"
    "all_members_comparable_success=true;follow_up_requires_favorable_campaign_interpretation"
    "=true;follow_up_authorizing=false;policy_hash=24:fnv1a64:83a08cff5aad41ad;assessment_con"
    "tract_version=2;assessment_canonical=1700:experiment_recommendation_campaign_outcome_ass"
    "essment_v2;assessment_contract_version=2;campaign_approval_id=1;campaign_identity=1:c;ca"
    "mpaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materialization_id=2;materialization_co"
    "ntract_version=1;materialization_identity=1:m;materialization_identity_hash=24:fnv1a64:a"
    "f63e04c8601f358;member=1,3,4,5,6,7,8,lifecycle=succeeded,consistency=consistent,outcome="
    "succeeded_comparable,diagnostic_count=0,source=present,6,9,10,context=1:s,1,0,1,1:l,1:a,"
    "1:b,metrics=2,18:inference_accuracy,1,numeric_delta,12:leader_score,2,numeric_delta,resu"
    "lt=present,8,11,result_identity_count=1,1:r,12,context=present,1:s,1,0,1,1:l,1:a,1:b,met"
    "rics=2,18:inference_accuracy,2,numeric_delta,12:leader_score,2,numeric_delta,comparison_"
    "count=2,18:inference_accuracy,comparable,source_value=1,result_value=2,delta=1,context_d"
    "ifference_count=0,12:leader_score,comparable,source_value=2,result_value=2,delta=0,conte"
    "xt_difference_count=0;aggregate_outcome=succeeded_comparable;member_count=1;lifecycle_no"
    "t_terminal_count=0;lifecycle_succeeded_count=1;lifecycle_failed_count=0;lifecycle_cancel"
    "led_count=0;lifecycle_unknown_count=0;consistent_member_count=1;inconsistent_member_coun"
    "t=0;not_ready_member_count=0;succeeded_comparable_member_count=1;succeeded_context_chang"
    "ed_member_count=0;succeeded_metric_gap_member_count=0;terminal_failed_member_count=0;ter"
    "minal_cancelled_member_count=0;inconsistent_outcome_member_count=0;metric_comparison_cou"
    "nt=2;comparable_metric_count=2;context_changed_metric_count=0;missing_source_metric_coun"
    "t=0;missing_result_metric_count=0;metric_value_unavailable_count=0;unsupported_metric_co"
    "unt=0;positive_delta_count=1;zero_delta_count=1;negative_delta_count=0;assessment_hash=2"
    "4:fnv1a64:b93180b7b3627b7c;campaign_approval_id=1;campaign_identity=1:c;campaign_identit"
    "y_hash=24:fnv1a64:af63de4c8601eff2;materialization_id=2;materialization_campaign_approva"
    "l_id=1;materialization_campaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materializatio"
    "n_contract_version=1;materialization_member_count=1;materialization_identity=1:m;materia"
    "lization_identity_hash=24:fnv1a64:af63e04c8601f358;member=1,3,4,5,6,7,expected_experimen"
    "t_id=8,assessment_lifecycle=succeeded,assessment_consistency=consistent,assessment_outco"
    "me=succeeded_comparable,assessment_diagnostic_count=0,evidence_classification=succeeded_"
    "comparable,policy_interpretation=favorable,metric_evaluation_count=2,18:inference_accura"
    "cy,higher_is_favorable,assessment_classification=comparable,assessment_delta=1,policy_ju"
    "dgment=favorable,12:leader_score,higher_is_favorable,assessment_classification=comparabl"
    "e,assessment_delta=0,policy_judgment=neutral;evidence_sufficiency=sufficient;campaign_in"
    "terpretation=favorable;follow_up_eligibility=eligible_for_operator_review;follow_up_auth"
    "orized=false;member_count=1;inconsistent_member_count=0;not_ready_member_count=0;failed_"
    "member_count=0;cancelled_member_count=0;context_changed_member_count=0;metric_gap_member"
    "_count=0;comparable_member_count=1;metric_evaluation_count=2;not_evaluable_metric_count="
    "0;favorable_metric_count=1;neutral_metric_count=1;unfavorable_metric_count=0;evidence_re"
    "asons=0;interpretation_reasons=2,favorable_metric_evidence_present,neutral_metric_eviden"
    "ce_present;follow_up_reasons=1,all_members_succeeded_comparable";
constexpr char kExpectedGoldenDecisionHash[] = "fnv1a64:bde8fbe1fa40c2f2";

static_assert(Decision::readOnly);
static_assert(Decision::databaseFree);
static_assert(!Decision::persistent);
static_assert(Decision::advisory);
static_assert(!Decision::authoritative);
static_assert(!Decision::declaresCampaignSuccess);
static_assert(!Decision::followUpAuthorizing);
static_assert(!std::is_default_constructible_v<Policy>);
static_assert(!std::is_default_constructible_v<Decision>);
static_assert(!std::is_aggregate_v<Policy>);
static_assert(!std::is_aggregate_v<Decision>);
static_assert(!std::is_copy_assignable_v<Policy>);
static_assert(!std::is_copy_assignable_v<Decision>);
static_assert(!std::is_constructible_v<PolicyIdentity,
    int, std::string, std::string>);
static_assert(!std::is_constructible_v<AssessmentIdentity,
    int, std::string, std::string>);

class CommaNumpunct : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
};

template <typename Function>
void AssertInvalidArgument(Function&& function)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::invalid_argument&)
    {
        threw = true;
    }
    assert(threw);
}

RecommendationCampaignOutcomeAssessmentCampaignIdentity Campaign()
{
    const std::string canonical = "policy-test-campaign-approval";
    return {41, canonical, RecommendationCanonicalHash(canonical)};
}

RecommendationCampaignOutcomeAssessmentMaterializationIdentity
Materialization(int count)
{
    const std::string canonical =
        "policy-test-materialization:" + std::to_string(count);
    return {71, Campaign().campaignApprovalId, Campaign().identityHash,
        kRecommendationCampaignMaterializationContractVersion, count,
        canonical, RecommendationCanonicalHash(canonical)};
}

Context ComparisonContext(std::string symbol = "eurusd")
{
    return {std::move(symbol), 12, 0.001, 128,
        "inference_eval_result_label_v1;label_rule_id=1;target_type=1",
        "2025-01-01", "2026-01-01"};
}

Metric Numeric(std::string identity, std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

Metric Unsupported(std::string identity, std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::Unsupported};
}

long long SourceExperiment(int ordinal) { return 500 + ordinal; }
long long ResultExperiment(int ordinal) { return 900 + ordinal; }

MemberIdentity Identity(int ordinal,
    std::optional<long long> expectedExperimentId)
{
    return {ordinal, 100 + ordinal, 200 + ordinal, 300 + ordinal,
        SourceExperiment(ordinal), 400 + ordinal, expectedExperimentId};
}

SourceEvidence Source(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparisonContext())
{
    return {SourceExperiment(ordinal), 600 + ordinal, 700 + ordinal,
        std::move(context), MetricCollection(std::move(metrics))};
}

ResultEvidence Result(
    int ordinal,
    std::vector<Metric> metrics,
    Context context = ComparisonContext())
{
    return {ResultExperiment(ordinal), 1000 + ordinal,
        {ResultIdentity("experiment_analysis_result", 2000 + ordinal)},
        std::move(context), MetricCollection(std::move(metrics))};
}

std::vector<Metric> SourceMetrics()
{
    return {Numeric("inference_accuracy", 0.70),
        Numeric("leader_score", 0.80)};
}

std::vector<Metric> FavorableMetrics()
{
    return {Numeric("inference_accuracy", 0.75),
        Numeric("leader_score", 0.82)};
}

std::vector<Metric> UnfavorableMetrics()
{
    return {Numeric("inference_accuracy", 0.65),
        Numeric("leader_score", 0.78)};
}

std::vector<Metric> NeutralMetrics()
{
    return {Numeric("inference_accuracy", 0.70),
        Numeric("leader_score", 0.80)};
}

MemberEvidence Succeeded(
    int ordinal,
    std::vector<Metric> sourceMetrics,
    std::vector<Metric> resultMetrics,
    Context sourceContext = ComparisonContext(),
    Context resultContext = ComparisonContext())
{
    return {Identity(ordinal, ResultExperiment(ordinal)),
        Lifecycle::Succeeded,
        Source(ordinal, std::move(sourceMetrics), std::move(sourceContext)),
        Result(ordinal, std::move(resultMetrics), std::move(resultContext))};
}

MemberEvidence Comparable(int ordinal)
{
    return Succeeded(ordinal, SourceMetrics(), FavorableMetrics());
}

MemberEvidence Terminal(int ordinal, Lifecycle lifecycle)
{
    return {Identity(ordinal, ResultExperiment(ordinal)), lifecycle,
        std::nullopt, std::nullopt};
}

Assessment Assess(
    std::vector<MemberEvidence> members,
    std::string observedAt = "2026-07-20 12:00:00+00")
{
    return BuildRecommendationCampaignOutcomeAssessment(Campaign(),
        Materialization(static_cast<int>(members.size())), observedAt,
        members);
}

Assessment GoldenAssessment()
{
    const std::string campaignCanonical = "c";
    const RecommendationCampaignOutcomeAssessmentCampaignIdentity campaign{
        1, campaignCanonical, RecommendationCanonicalHash(campaignCanonical)};
    const std::string materializationCanonical = "m";
    const RecommendationCampaignOutcomeAssessmentMaterializationIdentity
        materialization{2, campaign.campaignApprovalId,
            campaign.identityHash,
            kRecommendationCampaignMaterializationContractVersion, 1,
            materializationCanonical,
            RecommendationCanonicalHash(materializationCanonical)};
    const Context context{"s", 1, 0.0, 1, "l", "a", "b"};
    const MemberEvidence member{MemberIdentity{1, 3, 4, 5, 6, 7, 8},
        Lifecycle::Succeeded,
        SourceEvidence{6, 9, 10, context,
            MetricCollection{std::vector<Metric>{
                Numeric("inference_accuracy", 1.0),
                Numeric("leader_score", 2.0)}}},
        ResultEvidence{8, 11, {ResultIdentity{"r", 12}}, context,
            MetricCollection{std::vector<Metric>{
                Numeric("inference_accuracy", 2.0),
                Numeric("leader_score", 2.0)}}}};
    return BuildRecommendationCampaignOutcomeAssessment(campaign,
        materialization, "2026-07-20 00:00:00+00", {member});
}

ValidationView ValidationEvidence(
    const Policy& policy,
    const Assessment& assessment)
{
    ValidationView view;
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
        for (const auto& comparison : member.comparisons)
            memberView.comparisonMetricIdentities.push_back(
                comparison.metricIdentity);
        view.members.push_back(std::move(memberView));
    }
    return view;
}

Decision Decide(
    std::vector<MemberEvidence> members,
    const Policy& policy = BuildRecommendationCampaignOutcomePolicy())
{
    return ApplyRecommendationCampaignOutcomePolicy(
        policy, Assess(std::move(members)));
}

bool HasReason(
    const std::vector<PolicyReason>& reasons,
    PolicyReason reason)
{
    return std::find(reasons.begin(), reasons.end(), reason) != reasons.end();
}

const RecommendationCampaignOutcomePolicyMetricEvaluation& Evaluation(
    const Decision& decision,
    int memberOrdinal,
    const std::string& metricIdentity)
{
    const auto& member = decision.members.at(
        static_cast<std::size_t>(memberOrdinal - 1));
    const auto found = std::find_if(member.metricEvaluations.begin(),
        member.metricEvaluations.end(), [&](const auto& evaluation)
    {
        return evaluation.metricIdentity == metricIdentity;
    });
    assert(found != member.metricEvaluations.end());
    return *found;
}

} // namespace

int main()
{
    const Policy defaultPolicy =
        BuildRecommendationCampaignOutcomePolicy();
    assert(defaultPolicy.identity.contractVersion ==
        kRecommendationCampaignOutcomePolicyContractVersion);
    assert(defaultPolicy.identity.hash ==
        RecommendationCanonicalHash(defaultPolicy.identity.canonicalText));
    assert(defaultPolicy.requiredMetricRules.size() == 2);
    assert(defaultPolicy.requiredMetricRules[0].metricIdentity ==
        "inference_accuracy");
    assert(defaultPolicy.requiredMetricRules[1].metricIdentity ==
        "leader_score");

    const Assessment goldenAssessment = GoldenAssessment();
    const Decision goldenDecision =
        ApplyRecommendationCampaignOutcomePolicy(
            defaultPolicy, goldenAssessment);
    assert(defaultPolicy.identity.canonicalText ==
        kExpectedDefaultPolicyCanonical);
    assert(defaultPolicy.identity.hash == kExpectedDefaultPolicyHash);
    assert(goldenDecision.identity.canonicalText ==
        kExpectedGoldenDecisionCanonical);
    assert(goldenDecision.identity.hash == kExpectedGoldenDecisionHash);

    const ValidationView validEvidence =
        ValidationEvidence(defaultPolicy, goldenAssessment);
    ValidateRecommendationCampaignOutcomePolicyEvidence(validEvidence);
    ValidationView malformedEvidence = validEvidence;
    malformedEvidence.policyIdentityHash = "malformed";
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.assessmentIdentityHash = "malformed";
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.campaignIdentityHash = "malformed";
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.materializationIdentityHash = "malformed";
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.materializationCampaignApprovalId = 999;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.materializationCampaignIdentityHash = "mismatch";
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.assessmentContractVersion = 999;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.members[0].expectedExperimentId = 0;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });
    malformedEvidence = validEvidence;
    malformedEvidence.members[0].sourceExperimentId = 0;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });

    malformedEvidence = validEvidence;
    malformedEvidence.members[0].memberOrdinal = 2;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });

    const auto evidenceWithDistinctSecondMember = [&]
    {
        ValidationView evidence = validEvidence;
        auto second = evidence.members.front();
        second.memberOrdinal = 2;
        second.materializationMemberId += 1000;
        second.rankingMemberId += 1000;
        second.recommendationId += 1000;
        second.sourceExperimentId += 1000;
        second.proposalId += 1000;
        if (second.expectedExperimentId)
            *second.expectedExperimentId += 1000;
        evidence.members.push_back(std::move(second));
        evidence.materializationMemberCount = 2;
        evidence.assessmentSummaryMemberCount = 2;
        return evidence;
    };

    malformedEvidence = evidenceWithDistinctSecondMember();
    malformedEvidence.members[1].materializationMemberId =
        malformedEvidence.members[0].materializationMemberId;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });

    malformedEvidence = evidenceWithDistinctSecondMember();
    malformedEvidence.members[1].rankingMemberId =
        malformedEvidence.members[0].rankingMemberId;
    ValidateRecommendationCampaignOutcomePolicyEvidence(malformedEvidence);

    malformedEvidence = evidenceWithDistinctSecondMember();
    malformedEvidence.members[1].recommendationId =
        malformedEvidence.members[0].recommendationId;
    ValidateRecommendationCampaignOutcomePolicyEvidence(malformedEvidence);

    malformedEvidence = evidenceWithDistinctSecondMember();
    malformedEvidence.members[1].proposalId =
        malformedEvidence.members[0].proposalId;
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });

    malformedEvidence = validEvidence;
    malformedEvidence.members[0].comparisonMetricIdentities.push_back(
        malformedEvidence.members[0].comparisonMetricIdentities.front());
    AssertInvalidArgument([&]
    {
        ValidateRecommendationCampaignOutcomePolicyEvidence(
            malformedEvidence);
    });

    // These externally constructible identity inputs reject malformed hashes
    // before an immutable Step 5a assessment can exist.
    AssertInvalidArgument([]
    {
        (void)RecommendationCampaignOutcomeAssessmentCampaignIdentity{
            1, "c", "malformed"};
    });
    AssertInvalidArgument([]
    {
        (void)RecommendationCampaignOutcomeAssessmentMaterializationIdentity{
            2, 1, "campaign-hash",
            kRecommendationCampaignMaterializationContractVersion, 1,
            "m", "malformed"};
    });

    PolicyInput reorderedPolicyInput;
    std::reverse(reorderedPolicyInput.requiredMetricRules.begin(),
        reorderedPolicyInput.requiredMetricRules.end());
    assert(BuildRecommendationCampaignOutcomePolicy(reorderedPolicyInput) ==
        defaultPolicy);
    PolicyInput duplicatePolicyInput;
    duplicatePolicyInput.requiredMetricRules.push_back(
        duplicatePolicyInput.requiredMetricRules.front());
    AssertInvalidArgument([&]
    {
        (void)BuildRecommendationCampaignOutcomePolicy(
            duplicatePolicyInput);
    });
    PolicyInput unsupportedVersion;
    unsupportedVersion.contractVersion = 999;
    AssertInvalidArgument([&]
    {
        (void)BuildRecommendationCampaignOutcomePolicy(unsupportedVersion);
    });
    PolicyInput invalidMinimum;
    invalidMinimum.minimumComparableMemberCount = 0;
    AssertInvalidArgument([&]
    {
        (void)BuildRecommendationCampaignOutcomePolicy(invalidMinimum);
    });
    PolicyInput emptyRules;
    emptyRules.requiredMetricRules.clear();
    AssertInvalidArgument([&]
    {
        (void)BuildRecommendationCampaignOutcomePolicy(emptyRules);
    });
    PolicyInput invalidDirection;
    invalidDirection.requiredMetricRules[0].direction =
        static_cast<Direction>(999);
    AssertInvalidArgument([&]
    {
        (void)BuildRecommendationCampaignOutcomePolicy(invalidDirection);
    });
    AssertInvalidArgument([]
    {
        (void)RecommendationCampaignOutcomePolicyMetricDirectionText(
            static_cast<Direction>(999));
    });
    AssertInvalidArgument([]
    {
        (void)RecommendationCampaignOutcomePolicyCampaignInterpretationText(
            static_cast<CampaignInterpretation>(999));
    });
    AssertInvalidArgument([]
    {
        (void)RecommendationCampaignOutcomePolicyReasonText(
            static_cast<PolicyReason>(999));
    });

    // A complete comparable campaign is interpreted without turning the
    // advisory eligibility result into authority.
    const Decision favorable = Decide({Comparable(1)});
    assert(favorable.assessmentContractVersion ==
        kRecommendationCampaignOutcomeAssessmentContractVersion);
    assert(favorable.assessmentIdentityHash ==
        RecommendationCanonicalHash(favorable.assessmentCanonicalText));
    assert(favorable.campaignIdentity == Campaign());
    assert(favorable.materializationIdentity == Materialization(1));
    assert(favorable.summary.evidenceSufficiency ==
        EvidenceSufficiency::Sufficient);
    assert(favorable.summary.campaignInterpretation ==
        CampaignInterpretation::Favorable);
    assert(favorable.summary.followUpEligibility ==
        FollowUpEligibility::EligibleForOperatorReview);
    assert(!favorable.summary.followUpAuthorized);
    assert(favorable.summary.comparableMemberCount == 1);
    assert(favorable.summary.favorableMetricCount == 2);
    assert(favorable.members[0].evidenceClassification ==
        EvidenceClassification::SucceededComparable);
    assert(favorable.members[0].interpretation ==
        MemberInterpretation::Favorable);
    assert(Evaluation(favorable, 1, "inference_accuracy").judgment ==
        MetricJudgment::Favorable);
    assert(favorable.identity.canonicalText.find(
        "follow_up_authorized=false") != std::string::npos);
    assert(favorable.identity.canonicalText.find(
        "assessment_canonical=") != std::string::npos);
    assert(favorable.identity.hash ==
        RecommendationCanonicalHash(favorable.identity.canonicalText));

    const Decision mixedMetrics = Decide({Succeeded(1, SourceMetrics(),
        {Numeric("inference_accuracy", 0.75),
            Numeric("leader_score", 0.78)})});
    assert(mixedMetrics.summary.evidenceSufficiency ==
        EvidenceSufficiency::Sufficient);
    assert(mixedMetrics.summary.campaignInterpretation ==
        CampaignInterpretation::Mixed);
    assert(mixedMetrics.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);
    assert(HasReason(mixedMetrics.summary.followUpReasons,
        PolicyReason::FavorableCampaignInterpretationRequired));
    assert(mixedMetrics.members[0].interpretation ==
        MemberInterpretation::Mixed);
    assert(mixedMetrics.summary.favorableMetricCount == 1);
    assert(mixedMetrics.summary.unfavorableMetricCount == 1);

    const Decision neutral = Decide({Succeeded(1, SourceMetrics(),
        NeutralMetrics())});
    assert(neutral.summary.campaignInterpretation ==
        CampaignInterpretation::Neutral);
    assert(neutral.summary.evidenceSufficiency ==
        EvidenceSufficiency::Sufficient);
    assert(neutral.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);
    assert(HasReason(neutral.summary.followUpReasons,
        PolicyReason::FavorableCampaignInterpretationRequired));
    assert(neutral.summary.neutralMetricCount == 2);
    assert(Evaluation(neutral, 1, "leader_score").assessmentDelta ==
        std::optional<double>{0.0});

    const Decision unfavorable = Decide({Succeeded(1, SourceMetrics(),
        UnfavorableMetrics())});
    assert(unfavorable.summary.campaignInterpretation ==
        CampaignInterpretation::Unfavorable);
    assert(unfavorable.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);
    assert(HasReason(unfavorable.summary.followUpReasons,
        PolicyReason::FavorableCampaignInterpretationRequired));
    assert(unfavorable.summary.unfavorableMetricCount == 2);

    PolicyInput lowerIsFavorableInput;
    lowerIsFavorableInput.requiredMetricRules = {
        {"loss", Direction::LowerIsFavorable}};
    const Policy lowerIsFavorable =
        BuildRecommendationCampaignOutcomePolicy(lowerIsFavorableInput);
    const Decision lowerResult = Decide({Succeeded(1,
        {Numeric("loss", 1.0)}, {Numeric("loss", 0.5)})},
        lowerIsFavorable);
    assert(lowerResult.summary.campaignInterpretation ==
        CampaignInterpretation::Favorable);
    assert(Evaluation(lowerResult, 1, "loss").assessmentDelta ==
        std::optional<double>{-0.5});
    assert(Evaluation(lowerResult, 1, "loss").judgment ==
        MetricJudgment::Favorable);

    PolicyInput minimumTwoInput;
    minimumTwoInput.minimumComparableMemberCount = 2;
    const Decision minimumNotMet = Decide({Comparable(1)},
        BuildRecommendationCampaignOutcomePolicy(minimumTwoInput));
    assert(minimumNotMet.summary.evidenceSufficiency ==
        EvidenceSufficiency::Insufficient);
    assert(minimumNotMet.summary.campaignInterpretation ==
        CampaignInterpretation::Inconclusive);
    assert(HasReason(minimumNotMet.summary.evidenceReasons,
        PolicyReason::MinimumComparableMembersNotMet));

    // A metric absent from the Step 5a union stays explicitly absent; policy
    // evaluation does not invent or recompute it.
    const Decision requiredMetricMissing = Decide({Succeeded(1,
        {Numeric("inference_accuracy", 0.70)},
        {Numeric("inference_accuracy", 0.75)})});
    assert(requiredMetricMissing.members[0].evidenceClassification ==
        EvidenceClassification::SucceededComparable);
    assert(requiredMetricMissing.summary.evidenceSufficiency ==
        EvidenceSufficiency::Insufficient);
    assert(HasReason(requiredMetricMissing.summary.evidenceReasons,
        PolicyReason::RequiredMetricMissing));
    assert(!Evaluation(requiredMetricMissing, 1, "leader_score")
        .evidenceClassification);
    assert(!Evaluation(requiredMetricMissing, 1, "leader_score")
        .assessmentDelta);
    assert(Evaluation(requiredMetricMissing, 1, "leader_score").judgment ==
        MetricJudgment::NotEvaluable);

    const Decision contextChanged = Decide({Succeeded(1, SourceMetrics(),
        FavorableMetrics(), ComparisonContext(),
        ComparisonContext("gbpusd"))});
    assert(contextChanged.summary.contextChangedMemberCount == 1);
    assert(contextChanged.summary.campaignInterpretation ==
        CampaignInterpretation::Inconclusive);
    assert(HasReason(contextChanged.summary.evidenceReasons,
        PolicyReason::ContextChangedComparisonPresent));
    assert(Evaluation(contextChanged, 1, "leader_score")
        .evidenceClassification ==
        std::optional<Classification>{Classification::ContextChanged});

    const Decision missingSource = Decide({Succeeded(1,
        {Numeric("leader_score", 0.80)}, FavorableMetrics())});
    assert(missingSource.summary.metricGapMemberCount == 1);
    assert(HasReason(missingSource.summary.evidenceReasons,
        PolicyReason::MissingSourceMetricPresent));

    const Decision missingResult = Decide({Succeeded(1, SourceMetrics(),
        {Numeric("inference_accuracy", 0.75)})});
    assert(HasReason(missingResult.summary.evidenceReasons,
        PolicyReason::MissingResultMetricPresent));

    const Decision unavailable = Decide({Succeeded(1, SourceMetrics(),
        {Numeric("inference_accuracy", 0.75),
            Numeric("leader_score", std::nullopt)})});
    assert(HasReason(unavailable.summary.evidenceReasons,
        PolicyReason::MetricValueUnavailablePresent));

    const Decision unsupported = Decide({Succeeded(1,
        {Numeric("inference_accuracy", 0.70),
            Unsupported("leader_score", 0.80)},
        FavorableMetrics())});
    assert(HasReason(unsupported.summary.evidenceReasons,
        PolicyReason::UnsupportedMetricPresent));

    const Decision failed = Decide({Terminal(1, Lifecycle::Failed)});
    assert(failed.summary.failedMemberCount == 1);
    assert(failed.members[0].evidenceClassification ==
        EvidenceClassification::TerminalFailed);
    assert(failed.members[0].interpretation ==
        MemberInterpretation::Unfavorable);
    assert(failed.summary.evidenceSufficiency ==
        EvidenceSufficiency::Insufficient);
    assert(failed.summary.campaignInterpretation ==
        CampaignInterpretation::Unfavorable);
    assert(failed.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);

    const Decision cancelled = Decide({Terminal(1, Lifecycle::Cancelled)});
    assert(cancelled.summary.cancelledMemberCount == 1);
    assert(cancelled.summary.campaignInterpretation ==
        CampaignInterpretation::Inconclusive);
    assert(HasReason(cancelled.summary.evidenceReasons,
        PolicyReason::CancelledMemberPresent));

    const Decision notReady = Decide(
        {Terminal(1, Lifecycle::NotTerminal)});
    assert(notReady.summary.notReadyMemberCount == 1);
    assert(notReady.summary.campaignInterpretation ==
        CampaignInterpretation::Inconclusive);

    const Decision inconsistent = Decide({MemberEvidence(
        Identity(1, std::nullopt), Lifecycle::Unknown, std::nullopt,
        std::nullopt, Consistency::Inconsistent)});
    assert(inconsistent.summary.inconsistentMemberCount == 1);
    assert(inconsistent.summary.evidenceSufficiency ==
        EvidenceSufficiency::Insufficient);
    assert(inconsistent.members[0].assessmentDiagnostics.size() == 1);
    assert(inconsistent.identity.canonicalText.find(
        "assessment_diagnostic_count=1,input_evidence_inconsistent") !=
        std::string::npos);

    // Mixed members remain visible as separate evidence classes. A failure is
    // adverse evidence, but it can never satisfy the all-comparable condition
    // for follow-up operator review.
    const Decision successAndFailure = Decide(
        {Comparable(2), Terminal(1, Lifecycle::Failed)});
    assert(successAndFailure.summary.evidenceSufficiency ==
        EvidenceSufficiency::Sufficient);
    assert(successAndFailure.summary.campaignInterpretation ==
        CampaignInterpretation::Mixed);
    assert(successAndFailure.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);
    assert(HasReason(successAndFailure.summary.followUpReasons,
        PolicyReason::FailedMemberPresent));

    const Decision favorableAndUnfavorable = Decide({
        Succeeded(2, SourceMetrics(), UnfavorableMetrics()),
        Comparable(1)});
    assert(favorableAndUnfavorable.summary.comparableMemberCount == 2);
    assert(favorableAndUnfavorable.summary.campaignInterpretation ==
        CampaignInterpretation::Mixed);

    // Aggregate precedence is fixed policy behavior. The minimum comparable
    // member count is only an advisory evidence-coverage threshold, never a
    // claim of statistical sufficiency or campaign success.
    struct AggregateCase
    {
        const char* name;
        std::vector<MemberEvidence> members;
        int minimumComparableMemberCount;
        CampaignInterpretation expected;
    };
    const std::vector<AggregateCase> aggregateCases{
        {"all favorable", {Comparable(1), Comparable(2)}, 1,
            CampaignInterpretation::Favorable},
        {"all neutral",
            {Succeeded(1, SourceMetrics(), NeutralMetrics()),
                Succeeded(2, SourceMetrics(), NeutralMetrics())},
            1, CampaignInterpretation::Neutral},
        {"all unfavorable",
            {Succeeded(1, SourceMetrics(), UnfavorableMetrics()),
                Succeeded(2, SourceMetrics(), UnfavorableMetrics())},
            1, CampaignInterpretation::Unfavorable},
        {"favorable and unfavorable",
            {Comparable(1),
                Succeeded(2, SourceMetrics(), UnfavorableMetrics())},
            1, CampaignInterpretation::Mixed},
        {"favorable and neutral",
            {Comparable(1),
                Succeeded(2, SourceMetrics(), NeutralMetrics())},
            1, CampaignInterpretation::Favorable},
        {"unfavorable and neutral",
            {Succeeded(1, SourceMetrics(), UnfavorableMetrics()),
                Succeeded(2, SourceMetrics(), NeutralMetrics())},
            1, CampaignInterpretation::Unfavorable},
        {"favorable and failed",
            {Comparable(1), Terminal(2, Lifecycle::Failed)}, 1,
            CampaignInterpretation::Mixed},
        {"all failed",
            {Terminal(1, Lifecycle::Failed),
                Terminal(2, Lifecycle::Failed)},
            1, CampaignInterpretation::Unfavorable},
        {"cancelled present",
            {Comparable(1), Terminal(2, Lifecycle::Cancelled)}, 1,
            CampaignInterpretation::Inconclusive},
        {"not-ready present",
            {Comparable(1), Terminal(2, Lifecycle::NotTerminal)}, 1,
            CampaignInterpretation::Inconclusive},
        {"inconsistent present",
            {Comparable(1),
                MemberEvidence(Identity(2, std::nullopt),
                    Lifecycle::Unknown, std::nullopt, std::nullopt,
                    Consistency::Inconsistent)},
            1, CampaignInterpretation::Inconclusive},
        {"context-changed present",
            {Comparable(1),
                Succeeded(2, SourceMetrics(), FavorableMetrics(),
                    ComparisonContext(), ComparisonContext("gbpusd"))},
            1, CampaignInterpretation::Inconclusive},
        {"metric-gap present",
            {Comparable(1), Succeeded(2, SourceMetrics(),
                {Numeric("inference_accuracy", 0.75)})},
            1, CampaignInterpretation::Inconclusive},
        {"comparable count below minimum", {Comparable(1)}, 2,
            CampaignInterpretation::Inconclusive},
        {"all failed below minimum", {Terminal(1, Lifecycle::Failed)}, 2,
            CampaignInterpretation::Unfavorable}};
    for (const auto& aggregateCase : aggregateCases)
    {
        PolicyInput input;
        input.minimumComparableMemberCount =
            aggregateCase.minimumComparableMemberCount;
        const Decision decision = Decide(aggregateCase.members,
            BuildRecommendationCampaignOutcomePolicy(input));
        (void)aggregateCase.name;
        assert(decision.summary.campaignInterpretation ==
            aggregateCase.expected);
        assert(!decision.summary.followUpAuthorized);
    }

    // Eligibility is intentionally narrower than evidence sufficiency or a
    // merely conclusive interpretation.
    for (const Decision* decision : {&favorable, &neutral, &unfavorable,
             &mixedMetrics, &contextChanged, &successAndFailure})
        assert(!decision->summary.followUpAuthorized);
    assert(favorable.summary.followUpEligibility ==
        FollowUpEligibility::EligibleForOperatorReview);
    assert(contextChanged.summary.followUpEligibility ==
        FollowUpEligibility::NotEligible);

    // Step 5a canonicalizes member input. Step 5c consumes that immutable
    // ordering and therefore produces the same policy identity.
    const Assessment orderedAssessment = Assess({Comparable(1), Comparable(2)});
    const Assessment shuffledAssessment = Assess({Comparable(2), Comparable(1)});
    const Decision ordered = ApplyRecommendationCampaignOutcomePolicy(
        defaultPolicy, orderedAssessment);
    const Decision shuffled = ApplyRecommendationCampaignOutcomePolicy(
        defaultPolicy, shuffledAssessment);
    assert(ordered.members == shuffled.members);
    assert(ordered.summary == shuffled.summary);
    assert(ordered.identity == shuffled.identity);

    const Assessment laterObservation = Assess({Comparable(1)},
        "2026-07-20 12:05:00+00");
    const Decision later = ApplyRecommendationCampaignOutcomePolicy(
        defaultPolicy, laterObservation);
    assert(favorable.observedAt != later.observedAt);
    assert(favorable.assessmentIdentityHash == later.assessmentIdentityHash);
    assert(favorable.identity == later.identity);

    // Canonical identities are independent of the global locale.
    const std::locale originalLocale = std::locale();
    std::locale::global(std::locale(originalLocale, new CommaNumpunct));
    const Policy localizedPolicy =
        BuildRecommendationCampaignOutcomePolicy();
    const Decision localized = Decide({Comparable(1)}, localizedPolicy);
    std::locale::global(originalLocale);
    assert(localizedPolicy.identity == defaultPolicy.identity);
    assert(localized.identity == favorable.identity);

    const Decision changedEvidence = Decide({Succeeded(1, SourceMetrics(),
        {Numeric("inference_accuracy", 0.76),
            Numeric("leader_score", 0.82)})});
    assert(changedEvidence.assessmentIdentityHash !=
        favorable.assessmentIdentityHash);
    assert(changedEvidence.identity.hash != favorable.identity.hash);
    assert(minimumNotMet.policy.identity.hash != defaultPolicy.identity.hash);
    assert(minimumNotMet.identity.hash != favorable.identity.hash);

    // Ambiguous top-level membership is rejected by the immutable Step 5a
    // boundary before policy can interpret it.
    AssertInvalidArgument([]
    {
        (void)Assess({Comparable(1), Comparable(1)});
    });

    assert(RecommendationCampaignOutcomePolicyEvidenceSufficiencyText(
        EvidenceSufficiency::Sufficient) == "sufficient");
    assert(RecommendationCampaignOutcomePolicyCampaignInterpretationText(
        CampaignInterpretation::Mixed) == "mixed");
    assert(RecommendationCampaignOutcomePolicyFollowUpEligibilityText(
        FollowUpEligibility::EligibleForOperatorReview) ==
        "eligible_for_operator_review");
    assert(RecommendationCampaignOutcomePolicyReasonText(
        PolicyReason::EvidenceInsufficient) == "evidence_insufficient");

    return 0;
}
