#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <locale>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "../Sources/ExperimentRecommendationScoring.hpp"
#include "../Sources/ExperimentRecommendation.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{
class CommaNumpunct : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
};

RecommendationScoringInput Input(long long id = 1)
{
    RecommendationScoringInput input;
    input.recommendationId = id;
    input.sourceExperimentId = 184;
    input.sourcePredictionHorizon = 12;
    input.sourceRankWithinGroup = 1;
    input.sourceLeaderScore = 0.75;
    input.sourceInferenceAccuracy = 0.70;
    input.sourcePredictedNeutralProportion = 0.35;
    input.sourceEvidenceCount = 5000;
    input.changedParameter = kCoreLrMult;
    input.sourceValueCanonical = "1";
    input.proposedValueCanonical = "1.1";
    input.absoluteDelta = 0.1;
    input.relativeDelta = 0.1;
    input.generationOrdinal = 1;
    input.structuralRank = 1;
    input.semanticCanonicalText = "semantic_" + std::to_string(id);
    input.invocationCanonicalText = "invocation_" + std::to_string(id);
    input.recommendationPolicyCanonicalText = "recommendation_policy";
    input.duplicateType = "no_duplicate";
    input.recommendationStatus = "proposed";
    return input;
}

bool Throws(const std::string& text)
{
    try { (void)ParseRecommendationScoringPolicy(text); }
    catch (const std::invalid_argument&) { return true; }
    return false;
}

const RecommendationScoreComponent& Component(
    const RecommendationScoreResult& score, const std::string& name)
{
    const auto found = std::find_if(score.components.begin(),
        score.components.end(), [&](const auto& component) {
            return component.componentName == name;
        });
    assert(found != score.components.end());
    return *found;
}

std::vector<RankedRecommendationScore> RankingPair()
{
    RecommendationScoringInput first = Input(20);
    RecommendationScoringInput second = Input(10);
    first.semanticCanonicalText = second.semanticCanonicalText = "semantic";
    first.recommendationPolicyCanonicalText =
        second.recommendationPolicyCanonicalText = "policy";
    const RecommendationScoringPolicy policy;
    RecommendationScoreResult firstScore =
        ScoreExperimentRecommendation(policy, first);
    RecommendationScoreResult secondScore =
        ScoreExperimentRecommendation(policy, second);
    return {{first, firstScore, 0, 0, 0},
            {second, secondScore, 0, 0, 0}};
}

void RequireFirst(std::vector<RankedRecommendationScore> values,
                  long long recommendationId)
{
    const auto ranked = RankRecommendationScores(std::move(values));
    assert(ranked.front().input.recommendationId == recommendationId);
}
} // namespace

int main()
{
    const RecommendationScoringPolicy defaults;
    assert(!ValidateRecommendationScoringPolicy(defaults));
    const std::string canonical =
        RecommendationScoringPolicyCanonicalText(defaults);
    const std::string expectedCanonical =
        "experiment_recommendation_scoring_policy_v1"
        ";scoring_version=1"
        ";leader_score_weight=0.25"
        ";inference_accuracy_weight=0.25"
        ";evidence_strength_weight=0.15"
        ";neutral_balance_weight=0.1"
        ";structural_distance_weight=0.15"
        ";parameter_preference_weight=0.05"
        ";source_rank_weight=0.05"
        ";horizon_change_penalty_weight=0.05"
        ";relative_mutation_penalty_weight=0.1"
        ";minimum_evidence_count=1"
        ";evidence_saturation_count=5000"
        ";preferred_neutral_proportion=0.3333333333333333"
        ";maximum_neutral_proportion=0.8"
        ";maximum_relative_mutation=0.5"
        ";maximum_absolute_structural_distance=1"
        ";allow_missing_neutral_proportion=1"
        ";score_floor=0"
        ";score_ceiling=1"
        ";core_lr_preference=1"
        ";head_lr_preference=1"
        ";label_threshold_preference=1"
        ";prediction_horizon_preference=1";
    assert(canonical == expectedCanonical);
    assert(RecommendationScoringPolicyHash(defaults) ==
           "fnv1a64:2ad294a71969b770");
    const auto reordered = ParseRecommendationScoringPolicy(
        " score_ceiling = 1 , leader_score_weight = 0.25 ");
    assert(RecommendationScoringPolicyCanonicalText(reordered) == canonical);
    assert(Throws("unknown=1"));
    assert(Throws("leader_score_weight=1,leader_score_weight=2"));
    assert(Throws("leader_score_weight=-1"));
    assert(Throws("leader_score_weight=nan"));
    assert(Throws("score_floor=1,score_ceiling=1"));
    assert(Throws("minimum_evidence_count=0"));
    assert(Throws("minimum_evidence_count=10,evidence_saturation_count=9"));

    const RecommendationScoreResult base =
        ScoreExperimentRecommendation(defaults, Input());
    assert(base.valid && base.components.size() == 9);
    assert(std::isfinite(base.finalScore));
    assert(base.finalScore >= 0.0 && base.finalScore <= 1.0);
    RecommendationScoringInput approved = Input();
    approved.recommendationStatus = "approved";
    assert(ScoreExperimentRecommendation(defaults, approved).valid);
    RecommendationScoringInput rejected = Input();
    rejected.recommendationStatus = "rejected";
    assert(!ScoreExperimentRecommendation(defaults, rejected).valid);
    RecommendationScoringInput expired = Input();
    expired.recommendationStatus = "expired";
    assert(!ScoreExperimentRecommendation(defaults, expired).valid);

    RecommendationScoringInput input = Input();
    input.sourceEvidenceCount = defaults.minimumEvidenceCount;
    auto score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "evidence_strength").normalizedValue == 0.0);
    input.sourceEvidenceCount = defaults.evidenceSaturationCount;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "evidence_strength").normalizedValue == 1.0);
    input.sourceEvidenceCount = defaults.evidenceSaturationCount + 100;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "evidence_strength").normalizedValue == 1.0);
    input.sourceEvidenceCount = 0;
    assert(!ScoreExperimentRecommendation(defaults, input).valid);

    input = Input();
    input.sourcePredictedNeutralProportion =
        defaults.preferredNeutralProportion;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "neutral_balance").normalizedValue == 1.0);
    input.sourcePredictedNeutralProportion = defaults.maximumNeutralProportion;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "neutral_balance").normalizedValue == 0.0);
    input.sourcePredictedNeutralProportion.reset();
    score = ScoreExperimentRecommendation(defaults, input);
    assert(score.valid);
    assert(Component(score, "neutral_balance").normalizedValue == 0.5);
    RecommendationScoringPolicy requireNeutral = defaults;
    requireNeutral.allowMissingNeutralProportion = false;
    assert(!ScoreExperimentRecommendation(requireNeutral, input).valid);

    input = Input();
    input.absoluteDelta = 9.0;
    input.relativeDelta = 0.125;
    assert(DeriveRecommendationScoringDistance(input) == 0.125);
    input.relativeDelta.reset();
    assert(DeriveRecommendationScoringDistance(input) == 9.0);

    input = Input();
    input.absoluteDelta = 0.0;
    input.relativeDelta = 0.0;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "structural_proximity").normalizedValue == 1.0);
    assert(Component(score, "relative_mutation_penalty").normalizedValue == 0.0);
    input = Input();
    input.relativeDelta = defaults.maximumRelativeMutation;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "structural_proximity").normalizedValue == 0.0);
    assert(Component(score, "relative_mutation_penalty").normalizedValue == 1.0);

    input = Input();
    input.changedParameter = kPredictionHorizon;
    input.horizonDelta = 6;
    score = ScoreExperimentRecommendation(defaults, input);
    assert(Component(score, "horizon_change_penalty").normalizedValue == 0.5);
    assert(Component(base, "horizon_change_penalty").normalizedValue == 0.0);

    double positives = 0.0, penalties = 0.0, positiveWeights = 0.0;
    for (const auto& component : base.components)
    {
        if (component.penalty) penalties += component.weightedContribution;
        else { positives += component.weightedContribution; positiveWeights += component.weight; }
    }
    assert(std::abs(base.rawPositiveScore - positives / positiveWeights) < 1e-15);
    assert(std::abs(base.rawPenaltyScore - penalties / positiveWeights) < 1e-15);
    assert(std::abs(base.rawTotalScore -
                    (base.rawPositiveScore - base.rawPenaltyScore)) < 1e-15);
    assert(ScoreExperimentRecommendation(defaults, Input()).finalScore == base.finalScore);

    RecommendationScoringPolicy clamped = defaults;
    clamped.horizonChangePenaltyWeight = 100.0;
    input = Input();
    input.changedParameter = kPredictionHorizon;
    input.horizonDelta = 120;
    assert(ScoreExperimentRecommendation(clamped, input).finalScore == 0.0);
    clamped.scoreFloor = 0.2;
    assert(ScoreExperimentRecommendation(clamped, input).finalScore == 0.2);
    clamped = defaults;
    clamped.scoreCeiling = 0.5;
    assert(ScoreExperimentRecommendation(clamped, Input()).finalScore == 0.5);

    input = Input();
    input.sourceLeaderScore = std::numeric_limits<double>::quiet_NaN();
    assert(!ScoreExperimentRecommendation(defaults, input).valid);
    input = Input();
    input.absoluteDelta = std::numeric_limits<double>::infinity();
    assert(!ScoreExperimentRecommendation(defaults, input).valid);

    // Each documented comparator field is independently authoritative. The
    // pair deliberately gives recommendation ID 10 the final fallback
    // advantage so every earlier tie-break must override it when changed.
    auto pair = RankingPair();
    pair[0].score.finalScore += 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].score.rawPositiveScore += 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].score.rawPenaltyScore -= 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].input.sourceLeaderScore += 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].input.sourceInferenceAccuracy += 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].input.sourceEvidenceCount += 1;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].score.structuralDistance -= 0.01;
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].input.semanticCanonicalText = "a";
    pair[1].input.semanticCanonicalText = "b";
    RequireFirst(pair, 20);
    pair = RankingPair();
    pair[0].input.recommendationPolicyCanonicalText = "a";
    pair[1].input.recommendationPolicyCanonicalText = "b";
    RequireFirst(pair, 20);
    pair = RankingPair();
    RequireFirst(pair, 10);

    pair = RankingPair();
    pair[0].input.changedParameter = kPredictionHorizon;
    pair[1].input.changedParameter = kCoreLrMult;
    RequireFirst(pair, 10); // no hidden parameter-order tie-break

    std::vector<RankedRecommendationScore> rankable;
    for (long long id : {3LL, 1LL, 2LL})
    {
        input = Input(id);
        input.semanticCanonicalText = "same";
        rankable.push_back({input,
            ScoreExperimentRecommendation(defaults, input), 0, 0, 0});
    }
    const auto ranked = RankRecommendationScores(rankable);
    assert(ranked[0].input.recommendationId == 1);
    assert(ranked[1].input.recommendationId == 2);
    assert(ranked[2].input.recommendationId == 3);
    assert(ranked[0].scoreRank == 1 && ranked[0].tieGroup == 1);
    assert(ranked[1].scoreRank == 1 && ranked[1].tieGroup == 1);
    assert(ranked[2].rankingOrdinal == 3);
    std::reverse(rankable.begin(), rankable.end());
    const auto reverseRanked = RankRecommendationScores(rankable);
    for (std::size_t index = 0; index < ranked.size(); ++index)
    {
        assert(reverseRanked[index].input.recommendationId ==
               ranked[index].input.recommendationId);
        assert(reverseRanked[index].scoreRank == ranked[index].scoreRank);
        assert(reverseRanked[index].tieGroup == ranked[index].tieGroup);
        assert(reverseRanked[index].rankingOrdinal ==
               ranked[index].rankingOrdinal);
    }

    const std::locale previous = std::locale();
    std::locale::global(std::locale(previous, new CommaNumpunct));
    assert(RecommendationScoringPolicyCanonicalText(defaults) == canonical);
    assert(ScoreExperimentRecommendation(defaults, Input()).finalScore ==
           base.finalScore);
    std::locale::global(previous);

    assert(RecommendationPolicyHash(RecommendationPolicy{}) ==
           "fnv1a64:5d38b796e2380a45");
    return 0;
}
