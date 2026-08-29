#include "ProfitabilityVerification.hpp"

#include <algorithm>
#include <cmath>
#include <iterator>
#include <map>
#include <set>
#include <stdexcept>
#include <string_view>
#include <tuple>

namespace EA::ProfitabilityVerification
{
namespace
{

void AppendField(std::string& output,
                 std::string_view name,
                 const std::string& value)
{
    output.append(name);
    output.push_back('=');
    output.append(std::to_string(value.size()));
    output.push_back(':');
    output.append(value);
    output.push_back(';');
}

std::string Number(double value)
{
    if (!std::isfinite(value))
        throw std::invalid_argument("nonfinite_profitability_calibration_value");
    char buffer[64];
    const int length = std::snprintf(buffer, sizeof(buffer), "%.17g",
                                     value == 0.0 ? 0.0 : value);
    if (length <= 0 || static_cast<std::size_t>(length) >= sizeof(buffer))
        throw std::runtime_error("profitability_calibration_number_failure");
    return {buffer, static_cast<std::size_t>(length)};
}

bool SameWeight(double left, double right)
{
    return std::abs(left - right) <= 1e-12;
}

std::string IdVector(const std::vector<long long>& ids)
{
    std::string value;
    for (const long long id : ids)
    {
        if (!value.empty()) value.push_back(':');
        value += std::to_string(id);
    }
    return value.empty() ? "NONE" : value;
}

CalibrationMovementStatistics Movement(const std::vector<int>& deltas)
{
    CalibrationMovementStatistics result;
    result.count = static_cast<int>(deltas.size());
    if (deltas.empty()) return result;
    std::vector<int> absolute;
    absolute.reserve(deltas.size());
    long long sum = 0;
    long long absoluteSum = 0;
    for (const int delta : deltas)
    {
        sum += delta;
        absoluteSum += std::abs(delta);
        absolute.push_back(std::abs(delta));
        delta > 0 ? ++result.movedUp
            : delta < 0 ? ++result.movedDown : ++result.unchanged;
        result.maximumUpwardMovement =
            std::max(result.maximumUpwardMovement, delta);
        result.maximumDownwardMovement =
            std::max(result.maximumDownwardMovement, -delta);
    }
    std::sort(absolute.begin(), absolute.end());
    result.meanRankDelta = static_cast<double>(sum) / deltas.size();
    result.meanAbsoluteRankMovement =
        static_cast<double>(absoluteSum) / deltas.size();
    const std::size_t middle = absolute.size() / 2;
    result.medianAbsoluteRankMovement = absolute.size() % 2 == 0
        ? (static_cast<double>(absolute[middle - 1]) + absolute[middle]) / 2.0
        : static_cast<double>(absolute[middle]);
    const std::size_t p90Index = static_cast<std::size_t>(
        std::ceil(0.90 * static_cast<double>(absolute.size()))) - 1;
    result.p90AbsoluteRankMovement = absolute[p90Index];
    return result;
}

const CalibrationTopN& TopN(const ProfitabilityCalibrationWeight& weight, int n)
{
    const auto found = std::find_if(weight.topN.begin(), weight.topN.end(),
        [n](const auto& value) { return value.n == n; });
    if (found == weight.topN.end())
        throw std::logic_error("profitability_calibration_top_n_missing");
    return *found;
}

ProfitabilityCalibrationPairwise Pairwise(
    const WeightedShadowRanking& left,
    const WeightedShadowRanking& right)
{
    if (left.controlSnapshotId != right.controlSnapshotId ||
        left.sourceEvaluationRunId != right.sourceEvaluationRunId ||
        left.candidates.size() != right.candidates.size())
        throw std::invalid_argument("incompatible_profitability_pairwise_rankings");
    ProfitabilityCalibrationPairwise result;
    result.leftWeight = left.shadowWeight;
    result.rightWeight = right.shadowWeight;
    std::map<long long, int> leftRanks;
    std::map<long long, int> rightRanks;
    for (const auto& candidate : left.candidates)
        leftRanks.emplace(candidate.source.recommendationId, candidate.shadowRank);
    for (const auto& candidate : right.candidates)
        rightRanks.emplace(candidate.source.recommendationId, candidate.shadowRank);
    if (leftRanks.size() != rightRanks.size())
        throw std::invalid_argument("incompatible_profitability_pairwise_membership");
    long long absoluteSum = 0;
    for (const auto& [recommendationId, leftRank] : leftRanks)
    {
        const auto found = rightRanks.find(recommendationId);
        if (found == rightRanks.end())
            throw std::invalid_argument(
                "incompatible_profitability_pairwise_membership");
        const int difference = std::abs(leftRank - found->second);
        absoluteSum += difference;
        if (difference > 0) ++result.ordinalChanges;
        if (difference > result.largestOrdinalDifference)
        {
            result.largestOrdinalDifference = difference;
            result.largestOrdinalDifferenceRecommendationIds = {recommendationId};
        }
        else if (difference == result.largestOrdinalDifference && difference > 0)
            result.largestOrdinalDifferenceRecommendationIds.push_back(
                recommendationId);
    }
    result.meanAbsoluteOrdinalDifference = leftRanks.empty() ? 0.0
        : static_cast<double>(absoluteSum) / leftRanks.size();
    for (const int n : {5, 10, 20})
    {
        std::set<long long> leftTop;
        std::set<long long> rightTop;
        for (const auto& [recommendationId, rank] : leftRanks)
            if (rank <= n) leftTop.insert(recommendationId);
        for (const auto& [recommendationId, rank] : rightRanks)
            if (rank <= n) rightTop.insert(recommendationId);
        std::vector<long long> overlap;
        std::set_intersection(leftTop.begin(), leftTop.end(),
                              rightTop.begin(), rightTop.end(),
                              std::back_inserter(overlap));
        std::vector<long long> difference;
        std::set_symmetric_difference(leftTop.begin(), leftTop.end(),
                                      rightTop.begin(), rightTop.end(),
                                      std::back_inserter(difference));
        result.topNOverlap.emplace(n, static_cast<int>(overlap.size()));
        result.topNDifferenceRecommendationIds.emplace(n, std::move(difference));
    }
    result.canonical = "campaign_profitability_calibration_pairwise_v1;";
    AppendField(result.canonical, "left_weight", Number(result.leftWeight));
    AppendField(result.canonical, "right_weight", Number(result.rightWeight));
    AppendField(result.canonical, "ordinal_changes",
                std::to_string(result.ordinalChanges));
    AppendField(result.canonical, "mean_absolute_ordinal_difference",
                Number(result.meanAbsoluteOrdinalDifference));
    AppendField(result.canonical, "largest_ordinal_difference",
                std::to_string(result.largestOrdinalDifference));
    AppendField(result.canonical, "largest_difference_ids",
                IdVector(result.largestOrdinalDifferenceRecommendationIds));
    for (const int n : {5, 10, 20})
    {
        AppendField(result.canonical, "top_" + std::to_string(n) + "_overlap",
                    std::to_string(result.topNOverlap.at(n)));
        AppendField(result.canonical,
                    "top_" + std::to_string(n) + "_difference_ids",
                    IdVector(result.topNDifferenceRecommendationIds.at(n)));
    }
    result.hash = InferenceProfitability::DeterministicHash(result.canonical);
    return result;
}

} // namespace

std::string CoverageRecoveryClassText(CoverageRecoveryClass value)
{
    switch (value)
    {
        case CoverageRecoveryClass::available: return "available";
        case CoverageRecoveryClass::recoverableHistoricalAbsence:
            return "recoverable_historical_absence";
        case CoverageRecoveryClass::contextMismatch: return "context_mismatch";
        case CoverageRecoveryClass::noExactFinalInference:
            return "no_exact_final_inference";
        case CoverageRecoveryClass::legacyIncompleteProvenance:
            return "legacy_incomplete_provenance";
        case CoverageRecoveryClass::invalidIncompleteProvenance:
            return "invalid_incomplete_provenance";
        case CoverageRecoveryClass::otherUnavailable:
            return "other_unavailable";
    }
    throw std::logic_error("unknown_profitability_coverage_recovery_class");
}

std::vector<double> Phase10ProfitabilityCalibrationWeights()
{
    std::vector<double> weights;
    weights.reserve(21);
    weights.push_back(0.0);
    for (int step = 1; step <= 20; ++step)
        weights.push_back(static_cast<double>(step) * 0.0025);
    return weights;
}

ProfitabilityCalibrationReport BuildProfitabilityCalibrationReport(
    const std::vector<WeightedShadowRanking>& rankings)
{
    if (rankings.empty() || !SameWeight(rankings.front().shadowWeight, 0.0))
        throw std::invalid_argument("profitability_calibration_control_missing");
    ProfitabilityCalibrationReport report;
    report.controlSnapshotId = rankings.front().controlSnapshotId;
    report.sourceEvaluationRunId = rankings.front().sourceEvaluationRunId;
    std::set<double> uniqueWeights;
    for (const auto& ranking : rankings)
    {
        if (ranking.controlSnapshotId != report.controlSnapshotId ||
            ranking.sourceEvaluationRunId != report.sourceEvaluationRunId ||
            ranking.candidates.size() != rankings.front().candidates.size() ||
            !uniqueWeights.insert(ranking.shadowWeight).second)
            throw std::invalid_argument("invalid_profitability_calibration_sweep");

        ProfitabilityCalibrationWeight point;
        point.weight = ranking.shadowWeight;
        point.totalMembers = static_cast<int>(ranking.candidates.size());
        point.rankingHash = ranking.hash;
        std::vector<int> all;
        std::vector<int> positive;
        std::vector<int> negative;
        std::vector<int> unavailable;
        for (const auto& candidate : ranking.candidates)
        {
            all.push_back(candidate.rankDelta);
            switch (candidate.source.profitabilitySign)
            {
                case ProfitabilitySign::positive:
                    ++point.validProfitabilityMembers;
                    ++point.positiveProfitabilityMembers;
                    positive.push_back(candidate.rankDelta);
                    break;
                case ProfitabilitySign::negative:
                    ++point.validProfitabilityMembers;
                    ++point.negativeProfitabilityMembers;
                    negative.push_back(candidate.rankDelta);
                    break;
                case ProfitabilitySign::zero:
                case ProfitabilitySign::zeroActionable:
                    ++point.validProfitabilityMembers;
                    ++point.zeroProfitabilityMembers;
                    break;
                case ProfitabilitySign::unavailable:
                    ++point.unavailableMembers;
                    unavailable.push_back(candidate.rankDelta);
                    break;
                case ProfitabilitySign::invalid:
                    throw std::invalid_argument(
                        "invalid_evidence_in_profitability_calibration");
            }
        }
        point.totalMovement = Movement(all);
        point.positiveMovement = Movement(positive);
        point.negativeMovement = Movement(negative);
        point.unavailableMovement = Movement(unavailable);
        for (const int n : {5, 10, 20})
        {
            CalibrationTopN top;
            top.n = n;
            std::set<long long> controlIds;
            std::set<long long> shadowIds;
            for (const auto& candidate : ranking.candidates)
            {
                if (candidate.source.currentRank <= n)
                    controlIds.insert(candidate.source.recommendationId);
                if (candidate.shadowRank > n) continue;
                shadowIds.insert(candidate.source.recommendationId);
                switch (candidate.source.profitabilitySign)
                {
                    case ProfitabilitySign::positive: ++top.positiveCount; break;
                    case ProfitabilitySign::negative: ++top.negativeCount; break;
                    case ProfitabilitySign::zero:
                    case ProfitabilitySign::zeroActionable: ++top.zeroCount; break;
                    case ProfitabilitySign::unavailable:
                        ++top.unavailableCount; break;
                    case ProfitabilitySign::invalid:
                        throw std::invalid_argument(
                            "invalid_evidence_in_profitability_calibration");
                }
            }
            top.memberRecommendationIds.assign(shadowIds.begin(), shadowIds.end());
            std::set_difference(shadowIds.begin(), shadowIds.end(),
                                controlIds.begin(), controlIds.end(),
                                std::back_inserter(top.entrantRecommendationIds));
            std::set_difference(controlIds.begin(), controlIds.end(),
                                shadowIds.begin(), shadowIds.end(),
                                std::back_inserter(top.exitingRecommendationIds));
            top.entered = static_cast<int>(top.entrantRecommendationIds.size());
            top.exited = static_cast<int>(top.exitingRecommendationIds.size());
            top.retained = static_cast<int>(shadowIds.size()) - top.entered;
            top.validEvidenceCoverageCount =
                top.positiveCount + top.negativeCount + top.zeroCount;
            top.validEvidenceCoveragePercentage = shadowIds.empty() ? 0.0
                : 100.0 * top.validEvidenceCoverageCount /
                    static_cast<double>(shadowIds.size());
            point.topN.push_back(std::move(top));
        }
        point.canonical = "campaign_profitability_calibration_weight_v1;";
        AppendField(point.canonical, "weight", Number(point.weight));
        AppendField(point.canonical, "ranking_hash", point.rankingHash);
        AppendField(point.canonical, "total_members",
                    std::to_string(point.totalMembers));
        AppendField(point.canonical, "valid_members",
                    std::to_string(point.validProfitabilityMembers));
        AppendField(point.canonical, "unavailable_members",
                    std::to_string(point.unavailableMembers));
        AppendField(point.canonical, "mean_absolute_movement",
                    Number(point.totalMovement.meanAbsoluteRankMovement));
        for (const auto& top : point.topN)
        {
            const std::string prefix = "top_" + std::to_string(top.n) + "_";
            AppendField(point.canonical, prefix + "members",
                        IdVector(top.memberRecommendationIds));
            AppendField(point.canonical, prefix + "positive",
                        std::to_string(top.positiveCount));
            AppendField(point.canonical, prefix + "valid_coverage",
                        std::to_string(top.validEvidenceCoverageCount));
        }
        point.hash = InferenceProfitability::DeterministicHash(point.canonical);
        report.weights.push_back(std::move(point));
    }

    const auto rankingAt = [&](double weight) -> const WeightedShadowRanking& {
        const auto found = std::find_if(rankings.begin(), rankings.end(),
            [weight](const auto& ranking) {
                return SameWeight(ranking.shadowWeight, weight);
            });
        if (found == rankings.end())
            throw std::invalid_argument(
                "profitability_calibration_anchor_weight_missing");
        return *found;
    };
    report.anchorPairwise.push_back(Pairwise(rankingAt(0.01), rankingAt(0.025)));
    report.anchorPairwise.push_back(Pairwise(rankingAt(0.025), rankingAt(0.05)));
    report.anchorPairwise.push_back(Pairwise(rankingAt(0.01), rankingAt(0.05)));

    const auto& control = report.weights.front();
    int bestTop5 = 0;
    int bestTop10 = 0;
    for (const auto& point : report.weights)
    {
        bestTop5 = std::max(bestTop5, TopN(point, 5).positiveCount);
        bestTop10 = std::max(bestTop10, TopN(point, 10).positiveCount);
    }
    for (const auto& point : report.weights)
    {
        if (!report.responseCurve.firstBestTop5Weight &&
            TopN(point, 5).positiveCount == bestTop5)
            report.responseCurve.firstBestTop5Weight = point.weight;
        if (!report.responseCurve.firstTop10AtLeastNineWeight &&
            TopN(point, 10).positiveCount >= 9)
            report.responseCurve.firstTop10AtLeastNineWeight = point.weight;
        if (!report.responseCurve.firstBestTop10Weight &&
            TopN(point, 10).positiveCount == bestTop10)
            report.responseCurve.firstBestTop10Weight = point.weight;
        if (!report.responseCurve.firstTop20ImprovementWeight &&
            TopN(point, 20).positiveCount > TopN(control, 20).positiveCount)
            report.responseCurve.firstTop20ImprovementWeight = point.weight;
    }

    for (const int n : {5, 10, 20})
    {
        std::size_t begin = 0;
        while (begin < report.weights.size())
        {
            std::size_t end = begin;
            while (end + 1 < report.weights.size() &&
                   TopN(report.weights[end + 1], n).memberRecommendationIds ==
                       TopN(report.weights[begin], n).memberRecommendationIds)
                ++end;
            report.responseCurve.stabilityRegions.push_back({
                n, report.weights[begin].weight, report.weights[end].weight,
                TopN(report.weights[begin], n).memberRecommendationIds});
            begin = end + 1;
        }
    }
    for (std::size_t index = 1; index < report.weights.size(); ++index)
    {
        bool changed = false;
        for (const int n : {5, 10, 20})
            changed = changed ||
                TopN(report.weights[index - 1], n).memberRecommendationIds !=
                TopN(report.weights[index], n).memberRecommendationIds;
        if (changed)
            report.responseCurve.membershipDiscontinuityWeights.push_back(
                report.weights[index].weight);
    }

    const auto at005 = std::find_if(report.weights.begin(), report.weights.end(),
        [](const auto& point) { return SameWeight(point.weight, 0.05); });
    if (at005 == report.weights.end())
        throw std::invalid_argument("profitability_calibration_anchor_weight_missing");
    const int requiredTop10 = std::max(9, bestTop10 - 1);
    const int requiredTop20 = TopN(control, 20).positiveCount + 1;
    const auto effective = [&](const ProfitabilityCalibrationWeight& point) {
        return point.weight > 0.0 &&
            TopN(point, 5).positiveCount == bestTop5 &&
            TopN(point, 10).positiveCount >= requiredTop10 &&
            TopN(point, 20).positiveCount >= requiredTop20 &&
            point.positiveMovement.movedDown == 0 &&
            point.negativeMovement.movedUp == 0 &&
            point.totalMovement.meanAbsoluteRankMovement <
                at005->totalMovement.meanAbsoluteRankMovement;
    };
    for (std::size_t index = 0; index < report.weights.size(); ++index)
    {
        if (!effective(report.weights[index])) continue;
        report.responseCurve.minimumEffectiveWeight = report.weights[index].weight;
        std::size_t end = index;
        while (end + 1 < report.weights.size() && effective(report.weights[end + 1]))
            ++end;
        report.responseCurve.minimumEffectiveRegionEnd = report.weights[end].weight;
        break;
    }
    if (report.responseCurve.minimumEffectiveWeight &&
        report.responseCurve.minimumEffectiveRegionEnd)
    {
        report.responseCurve.provisional0025InsideMinimumEffectiveRegion =
            0.025 + 1e-12 >= *report.responseCurve.minimumEffectiveWeight &&
            0.025 - 1e-12 <= *report.responseCurve.minimumEffectiveRegionEnd;
        report.responseCurve.assessment =
            "provisional_minimum_effective_region_identified";
    }
    else
        report.responseCurve.assessment = "no_minimum_effective_region_identified";

    report.responseCurve.canonical =
        "campaign_profitability_calibration_response_curve_v1;";
    const auto optionalWeight = [](const std::optional<double>& value) {
        return value ? Number(*value) : std::string{"NULL"};
    };
    AppendField(report.responseCurve.canonical, "first_best_top5_weight",
                optionalWeight(report.responseCurve.firstBestTop5Weight));
    AppendField(report.responseCurve.canonical, "first_top10_at_least_9_weight",
                optionalWeight(report.responseCurve.firstTop10AtLeastNineWeight));
    AppendField(report.responseCurve.canonical, "first_best_top10_weight",
                optionalWeight(report.responseCurve.firstBestTop10Weight));
    AppendField(report.responseCurve.canonical, "first_top20_improvement_weight",
                optionalWeight(report.responseCurve.firstTop20ImprovementWeight));
    AppendField(report.responseCurve.canonical, "minimum_effective_weight",
                optionalWeight(report.responseCurve.minimumEffectiveWeight));
    AppendField(report.responseCurve.canonical, "minimum_effective_region_end",
                optionalWeight(report.responseCurve.minimumEffectiveRegionEnd));
    AppendField(report.responseCurve.canonical, "assessment",
                report.responseCurve.assessment);
    report.responseCurve.hash = InferenceProfitability::DeterministicHash(
        report.responseCurve.canonical);

    report.canonical = "campaign_profitability_calibration_report_v1;";
    AppendField(report.canonical, "control_snapshot_id",
                std::to_string(report.controlSnapshotId));
    AppendField(report.canonical, "source_evaluation_run_id",
                std::to_string(report.sourceEvaluationRunId));
    AppendField(report.canonical, "response_curve",
                report.responseCurve.canonical);
    for (std::size_t index = 0; index < report.weights.size(); ++index)
        AppendField(report.canonical, "weight[" + std::to_string(index) + "]",
                    report.weights[index].canonical);
    for (std::size_t index = 0; index < report.anchorPairwise.size(); ++index)
        AppendField(report.canonical, "pairwise[" + std::to_string(index) + "]",
                    report.anchorPairwise[index].canonical);
    report.hash = InferenceProfitability::DeterministicHash(report.canonical);
    return report;
}

} // namespace EA::ProfitabilityVerification
