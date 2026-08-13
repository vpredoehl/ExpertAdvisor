#include "ExperimentRecommendationCampaignFollowUpProposal.hpp"

#include "ExperimentRecommendation.hpp"
#include "ExperimentRecommendationCampaignMaterialization.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
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
using CampaignIdentity =
    RecommendationCampaignOutcomeAssessmentCampaignIdentity;
using CampaignInterpretation =
    RecommendationCampaignOutcomePolicyCampaignInterpretation;
using Context = RecommendationCampaignOutcomeAssessmentComparisonContext;
using Decision = RecommendationCampaignOutcomePolicyDecision;
using DecisionIdentity =
    RecommendationCampaignOutcomePolicyDecisionIdentity;
using Direction = RecommendationCampaignOutcomePolicyMetricDirection;
using EvidenceSufficiency =
    RecommendationCampaignOutcomePolicyEvidenceSufficiency;
using FollowUpEligibility =
    RecommendationCampaignOutcomePolicyFollowUpEligibility;
using Lifecycle = RecommendationCampaignOutcomeAssessmentLifecycleState;
using MaterializationIdentity =
    RecommendationCampaignOutcomeAssessmentMaterializationIdentity;
using MemberEvidence = RecommendationCampaignOutcomeAssessmentMemberEvidence;
using MemberIdentity = RecommendationCampaignOutcomeAssessmentMemberIdentity;
using Metric = RecommendationCampaignOutcomeAssessmentMetric;
using MetricCollection =
    RecommendationCampaignOutcomeAssessmentMetricCollection;
using MetricSupport =
    RecommendationCampaignOutcomeAssessmentMetricComparisonSupport;
using Policy = RecommendationCampaignOutcomePolicy;
using PolicyIdentity = RecommendationCampaignOutcomePolicyIdentity;
using PolicyInput = RecommendationCampaignOutcomePolicyInput;
using Proposal = RecommendationCampaignFollowUpProposal;
using ProposalIdentity = RecommendationCampaignFollowUpProposalIdentity;
using ProposalMember = RecommendationCampaignFollowUpProposalMember;
using ProposalReason = RecommendationCampaignFollowUpProposalReason;
using ResultEvidence =
    RecommendationCampaignOutcomeAssessmentResultEvidence;
using ResultIdentity =
    RecommendationCampaignOutcomeAssessmentResultIdentity;
using SourceEvidence =
    RecommendationCampaignOutcomeAssessmentSourceEvidence;
using ValidationView =
    RecommendationCampaignFollowUpProposalValidationView;

// Filled from a separately inspected Phase 6A result. The expected hash is a
// fixed literal and is never derived from this expected canonical literal.
constexpr char kExpectedGoldenProposalCanonical[] =
    "experiment_recommendation_campaign_follow_up_proposal_v1;proposal_contract_version=1;read_only=t"
    "rue;database_free=true;persistent=false;advisory=true;authoritative=false;approved=false;activat"
    "ed=false;execution_authorized=false;follow_up_authorized=false;scheduler_work=false;declares_cam"
    "paign_success=false;assessment_contract_version=2;assessment_canonical=1700:experiment_recommend"
    "ation_campaign_outcome_assessment_v2;assessment_contract_version=2;campaign_approval_id=1;campai"
    "gn_identity=1:c;campaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materialization_id=2;material"
    "ization_contract_version=1;materialization_identity=1:m;materialization_identity_hash=24:fnv1a64"
    ":af63e04c8601f358;member=1,3,4,5,6,7,8,lifecycle=succeeded,consistency=consistent,outcome=succee"
    "ded_comparable,diagnostic_count=0,source=present,6,9,10,context=1:s,1,0,1,1:l,1:a,1:b,metrics=2,"
    "18:inference_accuracy,1,numeric_delta,12:leader_score,2,numeric_delta,result=present,8,11,result"
    "_identity_count=1,1:r,12,context=present,1:s,1,0,1,1:l,1:a,1:b,metrics=2,18:inference_accuracy,2"
    ",numeric_delta,12:leader_score,2,numeric_delta,comparison_count=2,18:inference_accuracy,comparab"
    "le,source_value=1,result_value=2,delta=1,context_difference_count=0,12:leader_score,comparable,s"
    "ource_value=2,result_value=2,delta=0,context_difference_count=0;aggregate_outcome=succeeded_comp"
    "arable;member_count=1;lifecycle_not_terminal_count=0;lifecycle_succeeded_count=1;lifecycle_faile"
    "d_count=0;lifecycle_cancelled_count=0;lifecycle_unknown_count=0;consistent_member_count=1;incons"
    "istent_member_count=0;not_ready_member_count=0;succeeded_comparable_member_count=1;succeeded_con"
    "text_changed_member_count=0;succeeded_metric_gap_member_count=0;terminal_failed_member_count=0;t"
    "erminal_cancelled_member_count=0;inconsistent_outcome_member_count=0;metric_comparison_count=2;c"
    "omparable_metric_count=2;context_changed_metric_count=0;missing_source_metric_count=0;missing_re"
    "sult_metric_count=0;metric_value_unavailable_count=0;unsupported_metric_count=0;positive_delta_c"
    "ount=1;zero_delta_count=1;negative_delta_count=0;assessment_hash=24:fnv1a64:b93180b7b3627b7c;pol"
    "icy_contract_version=1;policy_canonical=512:experiment_recommendation_campaign_outcome_policy_v1"
    ";policy_contract_version=1;minimum_comparable_member_count=1;required_metric_rule_count=2;requir"
    "ed_metric=18:inference_accuracy,higher_is_favorable;required_metric=12:leader_score,higher_is_fa"
    "vorable;all_members_consistent_required=true;all_members_terminal_required=true;all_successful_m"
    "embers_comparable_required=true;follow_up_requires_all_members_comparable_success=true;follow_up"
    "_requires_favorable_campaign_interpretation=true;follow_up_authorizing=false;policy_hash=24:fnv1"
    "a64:83a08cff5aad41ad;policy_decision_contract_version=1;policy_decision_canonical=4023:experimen"
    "t_recommendation_campaign_outcome_policy_decision_v1;decision_contract_version=1;policy_contract"
    "_version=1;policy_canonical=512:experiment_recommendation_campaign_outcome_policy_v1;policy_cont"
    "ract_version=1;minimum_comparable_member_count=1;required_metric_rule_count=2;required_metric=18"
    ":inference_accuracy,higher_is_favorable;required_metric=12:leader_score,higher_is_favorable;all_"
    "members_consistent_required=true;all_members_terminal_required=true;all_successful_members_compa"
    "rable_required=true;follow_up_requires_all_members_comparable_success=true;follow_up_requires_fa"
    "vorable_campaign_interpretation=true;follow_up_authorizing=false;policy_hash=24:fnv1a64:83a08cff"
    "5aad41ad;assessment_contract_version=2;assessment_canonical=1700:experiment_recommendation_campa"
    "ign_outcome_assessment_v2;assessment_contract_version=2;campaign_approval_id=1;campaign_identity"
    "=1:c;campaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materialization_id=2;materialization_con"
    "tract_version=1;materialization_identity=1:m;materialization_identity_hash=24:fnv1a64:af63e04c86"
    "01f358;member=1,3,4,5,6,7,8,lifecycle=succeeded,consistency=consistent,outcome=succeeded_compara"
    "ble,diagnostic_count=0,source=present,6,9,10,context=1:s,1,0,1,1:l,1:a,1:b,metrics=2,18:inferenc"
    "e_accuracy,1,numeric_delta,12:leader_score,2,numeric_delta,result=present,8,11,result_identity_c"
    "ount=1,1:r,12,context=present,1:s,1,0,1,1:l,1:a,1:b,metrics=2,18:inference_accuracy,2,numeric_de"
    "lta,12:leader_score,2,numeric_delta,comparison_count=2,18:inference_accuracy,comparable,source_v"
    "alue=1,result_value=2,delta=1,context_difference_count=0,12:leader_score,comparable,source_value"
    "=2,result_value=2,delta=0,context_difference_count=0;aggregate_outcome=succeeded_comparable;memb"
    "er_count=1;lifecycle_not_terminal_count=0;lifecycle_succeeded_count=1;lifecycle_failed_count=0;l"
    "ifecycle_cancelled_count=0;lifecycle_unknown_count=0;consistent_member_count=1;inconsistent_memb"
    "er_count=0;not_ready_member_count=0;succeeded_comparable_member_count=1;succeeded_context_change"
    "d_member_count=0;succeeded_metric_gap_member_count=0;terminal_failed_member_count=0;terminal_can"
    "celled_member_count=0;inconsistent_outcome_member_count=0;metric_comparison_count=2;comparable_m"
    "etric_count=2;context_changed_metric_count=0;missing_source_metric_count=0;missing_result_metric"
    "_count=0;metric_value_unavailable_count=0;unsupported_metric_count=0;positive_delta_count=1;zero"
    "_delta_count=1;negative_delta_count=0;assessment_hash=24:fnv1a64:b93180b7b3627b7c;campaign_appro"
    "val_id=1;campaign_identity=1:c;campaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materializatio"
    "n_id=2;materialization_campaign_approval_id=1;materialization_campaign_identity_hash=24:fnv1a64:"
    "af63de4c8601eff2;materialization_contract_version=1;materialization_member_count=1;materializati"
    "on_identity=1:m;materialization_identity_hash=24:fnv1a64:af63e04c8601f358;member=1,3,4,5,6,7,exp"
    "ected_experiment_id=8,assessment_lifecycle=succeeded,assessment_consistency=consistent,assessmen"
    "t_outcome=succeeded_comparable,assessment_diagnostic_count=0,evidence_classification=succeeded_c"
    "omparable,policy_interpretation=favorable,metric_evaluation_count=2,18:inference_accuracy,higher"
    "_is_favorable,assessment_classification=comparable,assessment_delta=1,policy_judgment=favorable,"
    "12:leader_score,higher_is_favorable,assessment_classification=comparable,assessment_delta=0,poli"
    "cy_judgment=neutral;evidence_sufficiency=sufficient;campaign_interpretation=favorable;follow_up_"
    "eligibility=eligible_for_operator_review;follow_up_authorized=false;member_count=1;inconsistent_"
    "member_count=0;not_ready_member_count=0;failed_member_count=0;cancelled_member_count=0;context_c"
    "hanged_member_count=0;metric_gap_member_count=0;comparable_member_count=1;metric_evaluation_coun"
    "t=2;not_evaluable_metric_count=0;favorable_metric_count=1;neutral_metric_count=1;unfavorable_met"
    "ric_count=0;evidence_reasons=0;interpretation_reasons=2,favorable_metric_evidence_present,neutra"
    "l_metric_evidence_present;follow_up_reasons=1,all_members_succeeded_comparable;policy_decision_h"
    "ash=24:fnv1a64:bde8fbe1fa40c2f2;campaign_approval_id=1;campaign_identity=1:c;campaign_identity_h"
    "ash=24:fnv1a64:af63de4c8601eff2;materialization_id=2;materialization_campaign_approval_id=1;mate"
    "rialization_campaign_identity_hash=24:fnv1a64:af63de4c8601eff2;materialization_contract_version="
    "1;materialization_member_count=1;materialization_identity=1:m;materialization_identity_hash=24:f"
    "nv1a64:af63e04c8601f358;member_count=1;member=1,3,4,5,6,7,expected_experiment_id=8;evidence_suff"
    "iciency=sufficient;campaign_interpretation=favorable;follow_up_eligibility=eligible_for_operator"
    "_review;decision_follow_up_authorized=false;proposal_reason_count=1,eligible_favorable_policy_de"
    "cision";
constexpr char kExpectedGoldenProposalHash[] = "fnv1a64:23d15f2960ec08f5";

static_assert(Proposal::readOnly);
static_assert(Proposal::databaseFree);
static_assert(!Proposal::persistent);
static_assert(Proposal::advisory);
static_assert(!Proposal::authoritative);
static_assert(!Proposal::approved);
static_assert(!Proposal::activated);
static_assert(!Proposal::executionAuthorized);
static_assert(!Proposal::followUpAuthorized);
static_assert(!Proposal::schedulerWork);
static_assert(!Proposal::declaresCampaignSuccess);
static_assert(!std::is_default_constructible_v<ProposalIdentity>);
static_assert(!std::is_default_constructible_v<Proposal>);
static_assert(!std::is_aggregate_v<ProposalIdentity>);
static_assert(!std::is_aggregate_v<Proposal>);
static_assert(!std::is_copy_assignable_v<ProposalIdentity>);
static_assert(!std::is_copy_assignable_v<Proposal>);
static_assert(!std::is_constructible_v<ProposalIdentity,
    int, std::string, std::string>);
static_assert(!std::is_constructible_v<ProposalMember, MemberIdentity>);
static_assert(!std::is_constructible_v<AssessmentIdentity,
    int, std::string, std::string>);
static_assert(!std::is_constructible_v<PolicyIdentity,
    int, std::string, std::string>);
static_assert(!std::is_constructible_v<DecisionIdentity,
    int, std::string, std::string>);

class CommaNumpunct : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
};

template <typename Function>
void AssertExceptionReason(Function&& function, const std::string& expected)
{
    bool threw = false;
    try
    {
        function();
    }
    catch (const std::exception& error)
    {
        threw = true;
        assert(error.what() == expected);
    }
    assert(threw);
}

CampaignIdentity Campaign(
    long long campaignApprovalId = 1,
    std::string canonical = "c")
{
    const std::string hash = RecommendationCanonicalHash(canonical);
    return {campaignApprovalId, std::move(canonical), hash};
}

MaterializationIdentity Materialization(
    const CampaignIdentity& campaign,
    int memberCount,
    long long materializationId = 2,
    std::string canonical = "m")
{
    const std::string hash = RecommendationCanonicalHash(canonical);
    return {materializationId, campaign.campaignApprovalId,
        campaign.identityHash,
        kRecommendationCampaignMaterializationContractVersion, memberCount,
        std::move(canonical), hash};
}

Context ComparisonContext(std::string symbol = "s")
{
    return {std::move(symbol), 1, 0.0, 1, "l", "a", "b"};
}

Metric Numeric(std::string identity, std::optional<double> value)
{
    return {std::move(identity), value, MetricSupport::NumericDelta};
}

MemberIdentity Identity(int ordinal, long long recommendationOffset = 0)
{
    return {ordinal, 2 + ordinal, 3 + ordinal,
        4 + ordinal + recommendationOffset, 5 + ordinal, 6 + ordinal,
        7 + ordinal};
}

std::vector<Metric> SourceMetrics(
    double inferenceAccuracy = 1.0,
    double leaderScore = 2.0,
    bool reverse = false)
{
    if (reverse)
        return {Numeric("leader_score", leaderScore),
            Numeric("inference_accuracy", inferenceAccuracy)};
    return {Numeric("inference_accuracy", inferenceAccuracy),
        Numeric("leader_score", leaderScore)};
}

std::vector<Metric> ResultMetrics(
    double inferenceAccuracy,
    double leaderScore,
    bool reverse = false)
{
    if (reverse)
        return {Numeric("leader_score", leaderScore),
            Numeric("inference_accuracy", inferenceAccuracy)};
    return {Numeric("inference_accuracy", inferenceAccuracy),
        Numeric("leader_score", leaderScore)};
}

MemberEvidence Succeeded(
    int ordinal,
    std::vector<Metric> sourceMetrics,
    std::vector<Metric> resultMetrics,
    Context sourceContext = ComparisonContext(),
    Context resultContext = ComparisonContext(),
    long long recommendationOffset = 0)
{
    return {Identity(ordinal, recommendationOffset), Lifecycle::Succeeded,
        SourceEvidence{5 + ordinal, 8 + ordinal, 9 + ordinal,
            std::move(sourceContext),
            MetricCollection(std::move(sourceMetrics))},
        ResultEvidence{7 + ordinal, 10 + ordinal,
            {ResultIdentity{"r", 11 + ordinal}},
            std::move(resultContext),
            MetricCollection(std::move(resultMetrics))}};
}

MemberEvidence FavorableMember(
    int ordinal,
    bool reverseMetrics = false,
    double favorableInferenceAccuracy = 2.0)
{
    return Succeeded(ordinal, SourceMetrics(1.0, 2.0, reverseMetrics),
        ResultMetrics(
            favorableInferenceAccuracy, 2.0, reverseMetrics));
}

Assessment AssessWith(
    const CampaignIdentity& campaign,
    const MaterializationIdentity& materialization,
    std::vector<MemberEvidence> members,
    std::string observedAt = "2026-07-20 00:00:00+00")
{
    return BuildRecommendationCampaignOutcomeAssessment(campaign,
        materialization, observedAt, members);
}

Assessment Assess(
    std::vector<MemberEvidence> members,
    std::string observedAt = "2026-07-20 00:00:00+00")
{
    const CampaignIdentity campaign = Campaign();
    const MaterializationIdentity materialization =
        Materialization(campaign, static_cast<int>(members.size()));
    return AssessWith(
        campaign, materialization, std::move(members), std::move(observedAt));
}

Policy PolicyWithMinimum(int minimumComparableMemberCount)
{
    PolicyInput input;
    input.minimumComparableMemberCount = minimumComparableMemberCount;
    return BuildRecommendationCampaignOutcomePolicy(input);
}

Policy LowerIsFavorablePolicy()
{
    PolicyInput input;
    input.requiredMetricRules = {
        {"inference_accuracy", Direction::LowerIsFavorable},
        {"leader_score", Direction::LowerIsFavorable}};
    return BuildRecommendationCampaignOutcomePolicy(input);
}

Decision Decide(
    const Assessment& assessment,
    const Policy& policy = BuildRecommendationCampaignOutcomePolicy())
{
    return ApplyRecommendationCampaignOutcomePolicy(policy, assessment);
}

Proposal Propose(
    const Assessment& assessment,
    const Decision& decision)
{
    return BuildRecommendationCampaignFollowUpProposal(
        assessment, decision);
}

void AssertRefused(
    const Assessment& assessment,
    const Decision& decision,
    const std::string& expectedReason)
{
    AssertExceptionReason([&]
    {
        (void)Propose(assessment, decision);
    }, expectedReason);
}

} // namespace

int main()
{
    const Assessment favorableAssessment = Assess({FavorableMember(1)});
    const Decision favorableDecision = Decide(favorableAssessment);
    const Proposal favorable = Propose(
        favorableAssessment, favorableDecision);

    // Eligible favorable input produces one immutable advisory candidate and
    // copies exact upstream identities without approval or execution power.
    assert(favorable.identity.contractVersion ==
        kRecommendationCampaignFollowUpProposalContractVersion);
    assert(favorable.assessmentContractVersion ==
        favorableAssessment.identity.contractVersion);
    assert(favorable.assessmentCanonicalText ==
        favorableAssessment.identity.canonicalText);
    assert(favorable.assessmentIdentityHash ==
        favorableAssessment.identity.hash);
    assert(favorable.policyContractVersion ==
        favorableDecision.policy.identity.contractVersion);
    assert(favorable.policyCanonicalText ==
        favorableDecision.policy.identity.canonicalText);
    assert(favorable.policyIdentityHash ==
        favorableDecision.policy.identity.hash);
    assert(favorable.policyDecisionContractVersion ==
        favorableDecision.identity.contractVersion);
    assert(favorable.policyDecisionCanonicalText ==
        favorableDecision.identity.canonicalText);
    assert(favorable.policyDecisionIdentityHash ==
        favorableDecision.identity.hash);
    assert(favorable.campaignIdentity ==
        favorableAssessment.campaignIdentity);
    assert(favorable.materializationIdentity ==
        favorableAssessment.materializationIdentity);
    assert(favorable.memberCount == 1);
    assert(favorable.members.size() == 1);
    assert(favorable.members.front().identity ==
        favorableAssessment.members.front().identity);
    assert(favorable.summary.evidenceSufficiency ==
        favorableDecision.summary.evidenceSufficiency);
    assert(favorable.summary.campaignInterpretation ==
        CampaignInterpretation::Favorable);
    assert(favorable.summary.followUpEligibility ==
        FollowUpEligibility::EligibleForOperatorReview);
    assert(!favorable.summary.followUpAuthorized);
    assert(favorable.summary.reasons ==
        std::vector<ProposalReason>{
            ProposalReason::EligibleFavorablePolicyDecision});

    // Runtime proof mirrors every compile-time non-authority semantic.
    assert(favorable.readOnly);
    assert(favorable.databaseFree);
    assert(!favorable.persistent);
    assert(favorable.advisory);
    assert(!favorable.authoritative);
    assert(!favorable.approved);
    assert(!favorable.activated);
    assert(!favorable.executionAuthorized);
    assert(!favorable.followUpAuthorized);
    assert(!favorable.schedulerWork);
    assert(!favorable.declaresCampaignSuccess);

    assert(favorable.identity.canonicalText ==
        kExpectedGoldenProposalCanonical);
    assert(favorable.identity.hash == kExpectedGoldenProposalHash);
    assert(favorable.identity.hash ==
        RecommendationCanonicalHash(favorable.identity.canonicalText));

    const ValidationView aligned =
        MakeRecommendationCampaignFollowUpProposalValidationView(
            favorableAssessment, favorableDecision);
    ValidateRecommendationCampaignFollowUpProposalInput(aligned);

    // Campaign mismatch is distinguished before the enclosing assessment
    // canonical mismatch.
    const CampaignIdentity otherCampaign = Campaign(2, "other-campaign");
    const MaterializationIdentity otherCampaignMaterialization =
        Materialization(otherCampaign, 1);
    const Assessment otherCampaignAssessment = AssessWith(otherCampaign,
        otherCampaignMaterialization, {FavorableMember(1)});
    AssertRefused(favorableAssessment, Decide(otherCampaignAssessment),
        "campaign_identity_mismatch");

    const CampaignIdentity sameCampaign = Campaign();
    const MaterializationIdentity otherMaterialization =
        Materialization(sameCampaign, 1, 3, "other-materialization");
    const Assessment otherMaterializationAssessment = AssessWith(sameCampaign,
        otherMaterialization, {FavorableMember(1)});
    AssertRefused(favorableAssessment, Decide(otherMaterializationAssessment),
        "materialization_identity_mismatch");

    const MaterializationIdentity sharedTwoMemberMaterialization =
        Materialization(sameCampaign, 2);
    const Assessment membersA = AssessWith(sameCampaign,
        sharedTwoMemberMaterialization,
        {FavorableMember(1), FavorableMember(2)});
    const Assessment membersB = AssessWith(sameCampaign,
        sharedTwoMemberMaterialization,
        {Succeeded(1, SourceMetrics(), ResultMetrics(2.0, 2.0),
             ComparisonContext(), ComparisonContext(), 100),
            FavorableMember(2)});
    AssertRefused(membersA, Decide(membersB),
        "member_identity_mismatch");

    const Assessment changedCanonicalAssessment = Assess(
        {FavorableMember(1, false, 3.0)});
    AssertRefused(changedCanonicalAssessment, favorableDecision,
        "assessment_identity_mismatch");

    ValidationView malformed = aligned;
    malformed.assessmentIdentityHash = "fnv1a64:0000000000000000";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "assessment_identity_mismatch");
    malformed = aligned;
    malformed.decisionAssessmentIdentityHash =
        "fnv1a64:0000000000000000";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "assessment_identity_mismatch");

    malformed = aligned;
    malformed.policyCanonicalText += "x";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_identity_mismatch");
    malformed = aligned;
    malformed.policyIdentityHash = "fnv1a64:0000000000000000";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_identity_mismatch");

    malformed = aligned;
    malformed.policyDecisionCanonicalText += "x";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_decision_identity_mismatch");
    malformed = aligned;
    malformed.policyDecisionIdentityHash =
        "fnv1a64:0000000000000000";
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_decision_identity_mismatch");

    // Exact non-favorable Step 5C decisions fail only on copied Step 5C
    // classifications; Phase 6A does not reopen its truth tables.
    const Assessment neutralAssessment = Assess({Succeeded(1,
        SourceMetrics(), ResultMetrics(1.0, 2.0))});
    AssertRefused(neutralAssessment, Decide(neutralAssessment),
        "decision_not_eligible_for_operator_review");
    const Assessment unfavorableAssessment = Assess({Succeeded(1,
        SourceMetrics(), ResultMetrics(0.0, 1.0))});
    AssertRefused(unfavorableAssessment, Decide(unfavorableAssessment),
        "decision_not_eligible_for_operator_review");
    const Assessment mixedAssessment = Assess({Succeeded(1,
        SourceMetrics(), ResultMetrics(2.0, 1.0))});
    AssertRefused(mixedAssessment, Decide(mixedAssessment),
        "decision_not_eligible_for_operator_review");
    const Assessment inconclusiveAssessment = Assess({Succeeded(1,
        SourceMetrics(), ResultMetrics(2.0, 2.0), ComparisonContext(),
        ComparisonContext("t"))});
    AssertRefused(inconclusiveAssessment, Decide(inconclusiveAssessment),
        "decision_not_eligible_for_operator_review");
    const Decision insufficientDecision = Decide(
        favorableAssessment, PolicyWithMinimum(2));
    assert(insufficientDecision.summary.evidenceSufficiency ==
        EvidenceSufficiency::Insufficient);
    AssertRefused(favorableAssessment, insufficientDecision,
        "decision_not_eligible_for_operator_review");

    malformed = aligned;
    malformed.followUpAuthorized = true;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "follow_up_authorization_must_be_false");
    malformed = aligned;
    malformed.followUpEligibility = FollowUpEligibility::NotEligible;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "decision_not_eligible_for_operator_review");
    malformed = aligned;
    malformed.campaignInterpretation = CampaignInterpretation::Neutral;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "favorable_campaign_interpretation_required");

    // A lower-is-favorable policy proves that Phase 6A trusts the exact Step
    // 5C eligibility result instead of re-deriving direction or metric truth.
    const Assessment decreasingAssessment = Assess({Succeeded(1,
        SourceMetrics(2.0, 3.0), ResultMetrics(1.0, 2.0))});
    const Decision decreasingDecision = Decide(
        decreasingAssessment, LowerIsFavorablePolicy());
    assert(decreasingDecision.summary.campaignInterpretation ==
        CampaignInterpretation::Favorable);
    assert(decreasingDecision.summary.followUpEligibility ==
        FollowUpEligibility::EligibleForOperatorReview);
    const Proposal decreasing = Propose(
        decreasingAssessment, decreasingDecision);
    assert(decreasing.summary.followUpEligibility ==
        decreasingDecision.summary.followUpEligibility);

    // Step 5A canonical order is the only member/metric order consumed here.
    const Assessment orderedAssessment = Assess(
        {FavorableMember(1), FavorableMember(2)});
    const Assessment shuffledAssessment = Assess(
        {FavorableMember(2, true), FavorableMember(1, true)});
    const Decision orderedDecision = Decide(orderedAssessment);
    const Decision shuffledDecision = Decide(shuffledAssessment);
    assert(orderedAssessment.identity == shuffledAssessment.identity);
    assert(orderedDecision.identity == shuffledDecision.identity);
    assert(Propose(orderedAssessment, orderedDecision).identity ==
        Propose(shuffledAssessment, shuffledDecision).identity);

    const Assessment laterAssessment = Assess({FavorableMember(1)},
        "2026-07-20 00:05:00+00");
    const Decision laterDecision = Decide(laterAssessment);
    assert(favorableAssessment.observedAt != laterAssessment.observedAt);
    assert(favorableAssessment.identity == laterAssessment.identity);
    assert(favorableDecision.identity == laterDecision.identity);
    assert(favorable.identity ==
        Propose(laterAssessment, laterDecision).identity);

    const std::locale originalLocale = std::locale();
    std::locale::global(std::locale(originalLocale, new CommaNumpunct));
    const Assessment localizedAssessment = Assess({FavorableMember(1)});
    const Decision localizedDecision = Decide(localizedAssessment);
    const Proposal localized = Propose(
        localizedAssessment, localizedDecision);
    std::locale::global(originalLocale);
    assert(localized.identity == favorable.identity);

    const Proposal changedEvidence = Propose(changedCanonicalAssessment,
        Decide(changedCanonicalAssessment));
    assert(changedEvidence.assessmentIdentityHash !=
        favorable.assessmentIdentityHash);
    assert(changedEvidence.identity.hash != favorable.identity.hash);
    assert(changedEvidence.policyDecisionIdentityHash !=
        favorable.policyDecisionIdentityHash);

    const Assessment twoMemberAssessment = Assess(
        {FavorableMember(1), FavorableMember(2)});
    const Decision minimumOneDecision = Decide(
        twoMemberAssessment, PolicyWithMinimum(1));
    const Decision minimumTwoDecision = Decide(
        twoMemberAssessment, PolicyWithMinimum(2));
    const Proposal minimumOne = Propose(
        twoMemberAssessment, minimumOneDecision);
    const Proposal minimumTwo = Propose(
        twoMemberAssessment, minimumTwoDecision);
    assert(minimumOne.policyIdentityHash != minimumTwo.policyIdentityHash);
    assert(minimumOne.policyDecisionIdentityHash !=
        minimumTwo.policyDecisionIdentityHash);
    assert(minimumOne.identity.hash != minimumTwo.identity.hash);

    // IEEE negative zero is normalized upstream, while non-finite evidence is
    // rejected upstream before Phase 6A can inherit a malformed identity.
    const Assessment negativeZeroAssessment = Assess({Succeeded(1,
        SourceMetrics(-0.0, 2.0), ResultMetrics(0.0, 3.0))});
    const Assessment positiveZeroAssessment = Assess({Succeeded(1,
        SourceMetrics(0.0, 2.0), ResultMetrics(0.0, 3.0))});
    assert(negativeZeroAssessment.identity == positiveZeroAssessment.identity);
    assert(Propose(negativeZeroAssessment, Decide(negativeZeroAssessment))
            .identity ==
        Propose(positiveZeroAssessment, Decide(positiveZeroAssessment))
            .identity);
    AssertExceptionReason([]
    {
        (void)Assess({Succeeded(1, SourceMetrics(),
            ResultMetrics(std::numeric_limits<double>::infinity(), 3.0))});
    }, "recommendation_campaign_outcome_assessment_metric_value_nonfinite");

    ValidateRecommendationCampaignFollowUpProposalCanonicalSize(
        kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes);
    AssertExceptionReason([]
    {
        ValidateRecommendationCampaignFollowUpProposalCanonicalSize(
            kMaximumRecommendationCampaignFollowUpProposalCanonicalTextBytes +
            1U);
    }, "proposal_canonical_size_exceeded");

    const CampaignIdentity largeCampaign = Campaign(
        9, std::string(400000, 'x'));
    const MaterializationIdentity largeMaterialization =
        Materialization(largeCampaign, 1, 10, "large-materialization");
    const Assessment largeAssessment = AssessWith(largeCampaign,
        largeMaterialization, {FavorableMember(1)});
    const Decision largeDecision = Decide(largeAssessment);
    assert(largeDecision.summary.followUpEligibility ==
        FollowUpEligibility::EligibleForOperatorReview);
    AssertRefused(largeAssessment, largeDecision,
        "proposal_canonical_size_exceeded");

    malformed = aligned;
    malformed.assessmentContractVersion = 3;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "assessment_contract_unsupported");
    malformed = aligned;
    malformed.policyContractVersion = 2;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_contract_unsupported");
    malformed = aligned;
    malformed.policyDecisionContractVersion = 2;
    AssertExceptionReason([&]
    {
        ValidateRecommendationCampaignFollowUpProposalInput(malformed);
    }, "policy_decision_contract_unsupported");

    assert(RecommendationCampaignFollowUpProposalReasonText(
        ProposalReason::EligibleFavorablePolicyDecision) ==
        "eligible_favorable_policy_decision");

    return 0;
}
