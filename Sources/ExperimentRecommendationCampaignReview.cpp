#include "ExperimentRecommendationCampaignReview.hpp"

#include "ExperimentRecommendation.hpp"

#include <algorithm>
#include <locale>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>

namespace EA::ExperimentRecommendation
{
namespace
{

std::string LengthText(const std::string& value)
{
    return std::to_string(value.size()) + ":" + value;
}

bool IsSelected(const RecommendationCampaignPlanCandidate& candidate)
{
    return candidate.decision == RecommendationCampaignDecision::include;
}

void ValidatePlan(const RecommendationCampaignPlan& plan)
{
    if (plan.contractVersion != kRecommendationCampaignPlanContractVersion)
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_contract_unsupported");
    if (plan.identityCanonical.empty() ||
        plan.identityHash != RecommendationCanonicalHash(plan.identityCanonical))
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_identity_invalid");
    if (plan.policyCanonical !=
            RecommendationCampaignPlanningPolicyCanonicalText(plan.policy) ||
        plan.policyHash != RecommendationCanonicalHash(plan.policyCanonical) ||
        plan.scopeCanonical !=
            RecommendationCampaignPlanningScopeCanonicalText(plan.scope))
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_policy_or_scope_invalid");
    if (!RecommendationCampaignPlanOrderingIsDeterministic(plan))
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_order_invalid");
    if (plan.summary.candidateCount !=
            static_cast<int>(plan.candidates.size()) ||
        plan.summary.selectedCount < 0 || plan.summary.excludedCount < 0 ||
        plan.summary.selectedCount + plan.summary.excludedCount !=
            plan.summary.candidateCount)
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_summary_invalid");

    int selected = 0;
    int excluded = 0;
    for (const auto& candidate : plan.candidates)
    {
        const bool selectedShape =
            candidate.decision == RecommendationCampaignDecision::include &&
            candidate.reasons.size() == 1 &&
            candidate.reasons.front() == RecommendationCampaignReason::selected;
        const bool excludedShape =
            candidate.decision == RecommendationCampaignDecision::exclude &&
            !candidate.reasons.empty() &&
            std::find(
                candidate.reasons.begin(), candidate.reasons.end(),
                RecommendationCampaignReason::selected) ==
                candidate.reasons.end();
        if (!selectedShape && !excludedShape)
            throw std::invalid_argument(
                "recommendation_campaign_review_candidate_shape_invalid");
        selected += selectedShape ? 1 : 0;
        excluded += excludedShape ? 1 : 0;
    }
    if (selected != plan.summary.selectedCount ||
        excluded != plan.summary.excludedCount)
        throw std::invalid_argument(
            "recommendation_campaign_review_plan_counts_invalid");
}

RecommendationCampaignReviewCandidate ReviewCandidate(
    const RecommendationCampaignPlanCandidate& candidate)
{
    RecommendationCampaignReviewCandidate value;
    value.ordinal = candidate.ordinal;
    value.decision = candidate.decision;
    value.recommendationId = candidate.input.recommendationId;
    value.sourceExperimentId = candidate.input.sourceExperimentId;
    value.rankingMemberId = candidate.input.rankingMemberId;
    value.rankingPosition = candidate.input.rankingPosition;
    value.symbol = candidate.input.symbol;
    value.predictionHorizon = candidate.input.predictionHorizon;
    value.family = candidate.input.family;
    value.campaignDonchian20Mode = candidate.input.campaignDonchian20Mode;
    value.reasons = candidate.reasons;
    return value;
}

struct CoverageCounts
{
    int considered = 0;
    int selected = 0;
    int excluded = 0;
};

std::string ReasonsCanonicalText(
    const std::vector<RecommendationCampaignReason>& reasons)
{
    std::ostringstream out;
    out << "count=" << reasons.size();
    for (std::size_t index = 0; index < reasons.size(); ++index)
        out << ";reason[" << index << "]="
            << RecommendationCampaignReasonText(reasons[index]);
    return out.str();
}

std::string ReviewIdentityCanonicalText(
    const RecommendationCampaignReview& review)
{
    std::ostringstream out;
    out.imbue(std::locale::classic());
    out << "experiment_recommendation_campaign_review_v1"
        << ";contract_version=" << review.contractVersion
        << ";plan_identity="
        << LengthText(review.campaignPlanIdentityCanonical)
        << ";plan_identity_hash="
        << LengthText(review.campaignPlanIdentityHash)
        << ";policy_hash=" << LengthText(review.policyIdentityHash)
        << ";scope=" << LengthText(review.scopeCanonical)
        << ";candidate_count=" << review.summary.candidateCount
        << ";selected_count=" << review.summary.selectedCount
        << ";excluded_count=" << review.summary.excludedCount
        << ";duplicate_group_count=" << review.summary.duplicateGroupCount
        << ";duplicate_candidate_count="
        << review.summary.duplicateCandidateCount
        << ";considered_family_count="
        << review.summary.consideredFamilyCount
        << ";selected_family_count=" << review.summary.selectedFamilyCount
        << ";considered_symbol_count="
        << review.summary.consideredSymbolCount
        << ";selected_symbol_count=" << review.summary.selectedSymbolCount
        << ";considered_horizon_count="
        << review.summary.consideredHorizonCount
        << ";selected_horizon_count=" << review.summary.selectedHorizonCount
        << ";ordering_verified="
        << (review.summary.deterministicOrderingVerified ? 1 : 0);

    const auto appendCandidate = [&](const char* group, std::size_t index,
                                     const RecommendationCampaignReviewCandidate& value)
    {
        out << ';' << group << '[' << index << "]="
            << LengthText(
                   "ordinal=" + std::to_string(value.ordinal) +
                   ";decision=" +
                   RecommendationCampaignDecisionText(value.decision) +
                   ";recommendation_id=" +
                   std::to_string(value.recommendationId) +
                   ";source_experiment_id=" +
                   std::to_string(value.sourceExperimentId) +
                   ";ranking_member_id=" +
                   std::to_string(value.rankingMemberId) +
                   ";ranking_position=" +
                   std::to_string(value.rankingPosition) +
                   ";symbol=" + LengthText(value.symbol) +
                   ";horizon=" + std::to_string(value.predictionHorizon) +
                   ";family=" + LengthText(value.family) +
                   ";campaign_donchian20_arm=" +
                   (value.campaignDonchian20Mode
                        ? Donchian20ModeText(*value.campaignDonchian20Mode)
                        : "preserve") +
                   ";duplicate=" + (value.duplicate ? "1" : "0") +
                   ";duplicate_hash=" +
                   (value.duplicateIdentityHash
                        ? LengthText(*value.duplicateIdentityHash)
                        : "NULL") +
                   ";reasons=" + LengthText(ReasonsCanonicalText(value.reasons)));
    };
    out << ";selected_count_detail=" << review.selected.size();
    for (std::size_t index = 0; index < review.selected.size(); ++index)
        appendCandidate("selected", index, review.selected[index]);
    out << ";excluded_count_detail=" << review.excluded.size();
    for (std::size_t index = 0; index < review.excluded.size(); ++index)
        appendCandidate("excluded", index, review.excluded[index]);

    out << ";duplicate_count_detail=" << review.duplicateGroups.size();
    for (std::size_t index = 0; index < review.duplicateGroups.size(); ++index)
    {
        const auto& group = review.duplicateGroups[index];
        out << ";duplicate[" << index << "].hash="
            << LengthText(group.recommendationInvocationIdentityHash)
            << ";duplicate[" << index << "].member_count="
            << group.recommendationIds.size();
        for (std::size_t member = 0;
             member < group.recommendationIds.size(); ++member)
            out << ";duplicate[" << index << "].recommendation[" << member
                << "]=" << group.recommendationIds[member];
    }

    const auto appendTextCoverage = [&](const char* group, const auto& values)
    {
        out << ';' << group << "_count=" << values.size();
        for (std::size_t index = 0; index < values.size(); ++index)
            out << ';' << group << '[' << index << "]="
                << LengthText(values[index].value)
                << ';' << group << '[' << index << "].considered="
                << values[index].consideredCount
                << ';' << group << '[' << index << "].selected="
                << values[index].selectedCount
                << ';' << group << '[' << index << "].excluded="
                << values[index].excludedCount;
    };
    appendTextCoverage("family", review.familyCoverage);
    appendTextCoverage("symbol", review.symbolCoverage);
    out << ";horizon_count=" << review.horizonCoverage.size();
    for (std::size_t index = 0; index < review.horizonCoverage.size(); ++index)
    {
        const auto& value = review.horizonCoverage[index];
        out << ";horizon[" << index << "]=" << value.horizon
            << ";horizon[" << index << "].considered="
            << value.consideredCount
            << ";horizon[" << index << "].selected=" << value.selectedCount
            << ";horizon[" << index << "].excluded=" << value.excludedCount;
    }
    return out.str();
}

} // namespace

RecommendationCampaignReview ReviewRecommendationCampaignPlan(
    const RecommendationCampaignPlan& plan)
{
    ValidatePlan(plan);

    RecommendationCampaignReview review;
    review.campaignPlanIdentityCanonical = plan.identityCanonical;
    review.campaignPlanIdentityHash = plan.identityHash;
    review.policyIdentityHash = plan.policyHash;
    review.scopeCanonical = plan.scopeCanonical;
    review.generatedAt = plan.generatedAt;
    review.summary.candidateCount = plan.summary.candidateCount;
    review.summary.selectedCount = plan.summary.selectedCount;
    review.summary.excludedCount = plan.summary.excludedCount;
    review.summary.deterministicOrderingVerified = true;

    std::map<std::string, std::vector<std::size_t>> duplicateMembers;
    std::map<std::string, CoverageCounts> familyCounts;
    std::map<std::string, CoverageCounts> symbolCounts;
    std::map<int, CoverageCounts> horizonCounts;
    std::set<std::string> selectedFamilies;
    std::set<std::string> selectedSymbols;
    std::set<int> selectedHorizons;

    std::vector<RecommendationCampaignReviewCandidate> all;
    all.reserve(plan.candidates.size());
    for (std::size_t index = 0; index < plan.candidates.size(); ++index)
    {
        const auto& candidate = plan.candidates[index];
        all.push_back(ReviewCandidate(candidate));
        if (!candidate.input.recommendationInvocationCanonical.empty())
            duplicateMembers[
                candidate.input.recommendationInvocationCanonical +
                ";campaign_donchian20_arm=" +
                (candidate.input.campaignDonchian20Mode
                     ? Donchian20ModeText(*candidate.input.campaignDonchian20Mode)
                     : "preserve")]
                .push_back(index);
        const bool selected = IsSelected(candidate);
        auto count = [selected](auto& counts)
        {
            ++counts.considered;
            if (selected) ++counts.selected;
            else ++counts.excluded;
        };
        count(familyCounts[candidate.input.family]);
        count(symbolCounts[candidate.input.symbol]);
        count(horizonCounts[candidate.input.predictionHorizon]);
        if (selected)
        {
            selectedFamilies.insert(candidate.input.family);
            selectedSymbols.insert(candidate.input.symbol);
            selectedHorizons.insert(candidate.input.predictionHorizon);
        }
    }

    for (const auto& [canonical, members] : duplicateMembers)
    {
        if (members.size() < 2) continue;
        RecommendationCampaignReviewDuplicateGroup group;
        group.recommendationInvocationIdentityHash =
            RecommendationCanonicalHash(canonical);
        for (const std::size_t index : members)
        {
            all[index].duplicate = true;
            all[index].duplicateIdentityHash =
                group.recommendationInvocationIdentityHash;
            group.recommendationIds.push_back(all[index].recommendationId);
        }
        review.summary.duplicateCandidateCount +=
            static_cast<int>(members.size());
        review.duplicateGroups.push_back(std::move(group));
    }
    std::sort(
        review.duplicateGroups.begin(), review.duplicateGroups.end(),
        [](const auto& left, const auto& right)
        {
            if (left.recommendationInvocationIdentityHash !=
                right.recommendationInvocationIdentityHash)
                return left.recommendationInvocationIdentityHash <
                       right.recommendationInvocationIdentityHash;
            return left.recommendationIds < right.recommendationIds;
        });
    review.summary.duplicateGroupCount =
        static_cast<int>(review.duplicateGroups.size());

    for (auto& candidate : all)
    {
        if (candidate.decision == RecommendationCampaignDecision::include)
            review.selected.push_back(std::move(candidate));
        else
            review.excluded.push_back(std::move(candidate));
    }

    for (const auto& [value, counts] : familyCounts)
        review.familyCoverage.push_back({
            value, counts.considered, counts.selected, counts.excluded});
    for (const auto& [value, counts] : symbolCounts)
        review.symbolCoverage.push_back({
            value, counts.considered, counts.selected, counts.excluded});
    for (const auto& [value, counts] : horizonCounts)
        review.horizonCoverage.push_back({
            value, counts.considered, counts.selected, counts.excluded});

    review.summary.consideredFamilyCount =
        static_cast<int>(review.familyCoverage.size());
    review.summary.selectedFamilyCount =
        static_cast<int>(selectedFamilies.size());
    review.summary.consideredSymbolCount =
        static_cast<int>(review.symbolCoverage.size());
    review.summary.selectedSymbolCount =
        static_cast<int>(selectedSymbols.size());
    review.summary.consideredHorizonCount =
        static_cast<int>(review.horizonCoverage.size());
    review.summary.selectedHorizonCount =
        static_cast<int>(selectedHorizons.size());

    review.identityCanonical = ReviewIdentityCanonicalText(review);
    review.identityHash = RecommendationCanonicalHash(review.identityCanonical);
    return review;
}

} // namespace EA::ExperimentRecommendation
