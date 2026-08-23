#include "../Sources/ExperimentRecommendationRanking.hpp"
#include "../Sources/ExperimentRecommendation.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

using namespace EA::ExperimentRecommendation;

namespace
{

RecommendationScoreComponent Component(const std::string& name,
                                       double contribution,
                                       bool penalty)
{
    return {name, "reason", "input", 0.5, 0.2, contribution, penalty,
            "explanation"};
}

RecommendationRankingEvaluation Evaluation(
    long long id,
    RecommendationEvaluationDisposition disposition,
    std::optional<double> score,
    const std::string& semantic,
    const std::string& family = "core_lr_mult")
{
    const RecommendationEvaluationPolicy policy;
    const auto scoringSemantic =
        RecommendationScoringSemanticIdentityForPolicy(policy.scoringPolicy);
    const auto evaluationSemantic =
        RecommendationEvaluationSemanticIdentityForPolicy(policy);
    RecommendationRankingEvaluation value;
    value.evaluationResultId = id;
    value.evaluationRunId = 10;
    value.recommendationId = 100 + id;
    value.recommendationScanId = 20;
    value.sourceExperimentId = 30;
    value.sourceModelId = 40;
    value.sourceAnalysisId = 50;
    value.symbol = "EURUSD";
    value.horizon = 12;
    value.family = family;
    value.sourceValueCanonical = "1";
    value.proposedValueCanonical = "1.25";
    value.recommendationSemanticHash = semantic;
    value.evaluationIdentityCanonical = "evaluation_canonical_" +
        std::to_string(id);
    value.evaluationIdentityHash = RecommendationEvaluationCanonicalHash(
        value.evaluationIdentityCanonical);
    value.evaluationPolicyCanonical =
        RecommendationEvaluationPolicyCanonicalText(policy);
    value.evaluationPolicyHash = RecommendationEvaluationPolicyHash(policy);
    value.evaluationVersion = 1;
    value.evaluatorVersion = 1;
    value.scoringPolicyCanonical =
        RecommendationScoringPolicyCanonicalText(policy.scoringPolicy);
    value.scoringPolicyHash = RecommendationScoringPolicyHash(
        policy.scoringPolicy);
    value.scoringVersion = 1;
    value.scoringSemanticIdentity = scoringSemantic;
    value.evaluationSemanticIdentity = evaluationSemantic;
    value.disposition = disposition;
    value.reasonCode = RecommendationEvaluationDispositionText(disposition);
    value.explanation = "explanation";
    value.finalScore = score;
    value.eligibility = score ? RecommendationEligibility::eligible
                              : RecommendationEligibility::ineligible;
    if (score)
        value.components = {Component("leader_quality", 0.3, false),
                            Component("horizon_change_penalty", 0.1, true)};
    value.componentCount = static_cast<int>(value.components.size());
    return value;
}

RecommendationRankingScope RunScope(long long id = 10)
{
    RecommendationRankingScope scope;
    scope.type = RecommendationRankingScopeType::evaluationRun;
    scope.evaluationRunId = id;
    return scope;
}

void ApplyPolicy(RecommendationRankingEvaluation& value,
                 const RecommendationEvaluationPolicy& policy)
{
    value.evaluationPolicyCanonical =
        RecommendationEvaluationPolicyCanonicalText(policy);
    value.evaluationPolicyHash = RecommendationEvaluationPolicyHash(policy);
    value.evaluationVersion = policy.evaluationVersion;
    value.evaluatorVersion = policy.evaluatorVersion;
    value.scoringPolicyCanonical =
        RecommendationScoringPolicyCanonicalText(policy.scoringPolicy);
    value.scoringPolicyHash = RecommendationScoringPolicyHash(
        policy.scoringPolicy);
    value.scoringVersion = policy.scoringPolicy.scoringVersion;
    value.scoringSemanticIdentity =
        RecommendationScoringSemanticIdentityForPolicy(policy.scoringPolicy);
    value.evaluationSemanticIdentity =
        RecommendationEvaluationSemanticIdentityForPolicy(policy);
}

bool Throws(const auto& operation)
{
    try { operation(); }
    catch (const std::exception&) { return true; }
    return false;
}

} // namespace

int main()
{
    RecommendationRankingPolicy policy;
    assert(!ValidateRecommendationRankingPolicy(policy));
    assert(ValidateRecommendationRankingPolicy({2}));
    const std::string policyCanonical =
        RecommendationRankingPolicyCanonicalText(policy);
    assert(policyCanonical ==
        "experiment_recommendation_ranking_policy_v1;ranking_version=1;"
        "bucket_order=advisory_ready,blocked,non_actionable;"
        "ready_order=score_desc,semantic_hash,evaluation_hash,result_id;"
        "blocked_order=pending,active,completed,semantic_hash,evaluation_hash,result_id;"
        "non_actionable_order=insufficient,stale,unsupported,invalid,semantic_hash,evaluation_hash,result_id;"
        "inclusion=all_persisted_dispositions_before_global_limit;"
        "limit_application=global_after_order;grouping=single_scope;"
        "component_summary=per_penalty_class,contribution_desc,component_name;"
        "comparison=matching_evaluation_policy,scoring_policy,evaluator,family");
    assert(RecommendationRankingCanonicalHash(policyCanonical) ==
           RecommendationRankingCanonicalHash(policyCanonical));

    const RecommendationRankingScope run = RunScope();
    assert(!ValidateRecommendationRankingScope(run));
    assert(RecommendationRankingScopeTypeText(run.type) == "evaluation_run");
    assert(RecommendationRankingScopeValueText(run) == "10");
    RecommendationRankingScope symbolHorizon;
    symbolHorizon.type = RecommendationRankingScopeType::symbolHorizon;
    symbolHorizon.symbol = "EURUSD";
    symbolHorizon.horizon = 12;
    assert(!ValidateRecommendationRankingScope(symbolHorizon));
    assert(RecommendationRankingScopeCanonicalText(symbolHorizon).find(
        "symbol=6:EURUSD") != std::string::npos);
    RecommendationRankingScope scan;
    scan.type = RecommendationRankingScopeType::recommendationScan;
    scan.recommendationScanId = 20;
    RecommendationRankingScope symbol;
    symbol.type = RecommendationRankingScopeType::symbol;
    symbol.symbol = "NULL;type=global,=%:\xc3\xa9";
    RecommendationRankingScope horizon;
    horizon.type = RecommendationRankingScopeType::horizon;
    horizon.horizon = 12;
    RecommendationRankingScope family;
    family.type = RecommendationRankingScopeType::family;
    family.family = "core:lr=mult;%";
    RecommendationRankingScope global;
    global.type = RecommendationRankingScopeType::global;
    const std::vector<RecommendationRankingScope> allScopes = {
        run, scan, symbol, horizon, family, symbolHorizon, global};
    std::set<std::string> scopeCanonicals;
    for (const auto& scope : allScopes)
    {
        assert(!ValidateRecommendationRankingScope(scope));
        scopeCanonicals.insert(RecommendationRankingScopeCanonicalText(scope));
    }
    assert(scopeCanonicals.size() == allScopes.size());
    assert(RecommendationRankingScopeCanonicalText(symbol).find(
        "symbol=" + std::to_string(symbol.symbol->size()) + ":" +
        *symbol.symbol) != std::string::npos);
    assert(RecommendationRankingScopeCanonicalText(global).find(
        "type=global") != std::string::npos);
    RecommendationRankingScope ambiguous = run;
    ambiguous.symbol = "EURUSD";
    assert(ValidateRecommendationRankingScope(ambiguous));
    RecommendationRankingScope unsafeScope;
    unsafeScope.type = RecommendationRankingScopeType::symbol;
    unsafeScope.symbol = "EUR\nUSD";
    assert(ValidateRecommendationRankingScope(unsafeScope));
    unsafeScope.symbol = " EURUSD";
    assert(ValidateRecommendationRankingScope(unsafeScope));
    unsafeScope.symbol = "EURUSD ";
    assert(ValidateRecommendationRankingScope(unsafeScope));

    assert(RecommendationRankingBucketForDisposition(
        RecommendationEvaluationDisposition::advisoryReady) ==
        RecommendationRankingBucket::advisoryReady);
    assert(RecommendationRankingBucketForDisposition(
        RecommendationEvaluationDisposition::blockedActiveDuplicate) ==
        RecommendationRankingBucket::blocked);
    assert(RecommendationRankingBucketForDisposition(
        RecommendationEvaluationDisposition::staleSourceEvidence) ==
        RecommendationRankingBucket::nonActionable);
    assert(RecommendationRankingBucketForDisposition(
        RecommendationEvaluationDisposition::unsupportedRecommendationFamily) ==
        RecommendationRankingBucket::nonActionable);
    assert(ParseRecommendationRankingBucket("blocked") ==
           RecommendationRankingBucket::blocked);
    assert(!ParseRecommendationRankingBucket("unknown"));
    const auto emptyRanking = RankRecommendationEvaluationEvidence(
        policy, {}, kMaximumRecommendationRankingMembers);
    assert(emptyRanking.empty());
    assert(RecommendationRankingMembershipCanonicalText({}) ==
           "experiment_recommendation_ranking_membership_v1;count=0");
    const auto emptySemantics =
        ValidateRecommendationRankingPopulationSemantics({});
    assert(emptySemantics.state ==
           RecommendationRankingPopulationSemanticState::empty);
    assert(!emptySemantics.scoringIdentity && !emptySemantics.evaluationIdentity);

    auto homogeneousLeft = Evaluation(101,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "one");
    auto homogeneousRight = Evaluation(102,
        RecommendationEvaluationDisposition::advisoryReady, 0.6, "two");
    homogeneousRight.evaluationRunId = 11;
    const auto homogeneous = ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, homogeneousRight});
    assert(homogeneous.state ==
           RecommendationRankingPopulationSemanticState::verifiedHomogeneous);
    assert(homogeneous.distinctScoringIdentityCount == 1);
    assert(homogeneous.distinctEvaluationIdentityCount == 1);

    RecommendationEvaluationPolicy alternatePolicy;
    alternatePolicy.scoringPolicy.leaderScoreWeight = 0.30;
    auto incompatible = homogeneousRight;
    incompatible.finalScore = homogeneousLeft.finalScore;
    ApplyPolicy(incompatible, alternatePolicy);
    const auto heterogeneous = ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, incompatible});
    assert(heterogeneous.reason == "heterogeneous_scoring_semantics");
    assert(heterogeneous.evaluationRunIds == std::vector<long long>({10, 11}));
    assert(Throws([&] {
        RankRecommendationEvaluationEvidence(
            policy, {homogeneousLeft, incompatible}, 1);
    }));
    incompatible.recommendationId = homogeneousLeft.recommendationId;
    assert(Throws([&] {
        RankRecommendationEvaluationEvidence(
            policy, {homogeneousLeft, incompatible}, 2);
    }));

    auto inconsistentHash = homogeneousRight;
    inconsistentHash.scoringSemanticIdentity.hash = "fnv1a64:0000000000000000";
    assert(ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, inconsistentHash}).reason ==
        "invalid_scoring_semantic_provenance");
    auto sameHashDifferentCanonical = homogeneousRight;
    sameHashDifferentCanonical.scoringSemanticIdentity.canonical += ";different";
    sameHashDifferentCanonical.scoringSemanticIdentity.hash =
        homogeneousLeft.scoringSemanticIdentity.hash;
    assert(ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, sameHashDifferentCanonical}).reason ==
        "invalid_scoring_semantic_provenance");
    auto inconsistentEvaluationHash = homogeneousRight;
    inconsistentEvaluationHash.evaluationSemanticIdentity.hash =
        "fnv1a64:0000000000000000";
    assert(ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, inconsistentEvaluationHash}).reason ==
        "invalid_evaluation_semantic_provenance");
    auto inconsistentEvaluationCanonical = homogeneousRight;
    inconsistentEvaluationCanonical.evaluationSemanticIdentity.canonical +=
        ";different";
    inconsistentEvaluationCanonical.evaluationSemanticIdentity.hash =
        RecommendationEvaluationCanonicalHash(
            inconsistentEvaluationCanonical.evaluationSemanticIdentity.canonical);
    assert(ValidateRecommendationRankingPopulationSemantics(
        {homogeneousLeft, inconsistentEvaluationCanonical}).reason ==
        "invalid_evaluation_semantic_provenance");

    const std::string homogeneousIdentity =
        RecommendationRankingSnapshotIdentityCanonicalText(
            policy, run, 1, {homogeneousLeft});
    const std::string alternateIdentity =
        RecommendationRankingSnapshotIdentityCanonicalText(
            policy, run, 1, {incompatible});
    assert(homogeneousIdentity != alternateIdentity);

    for (const auto& protectedScope : allScopes)
        assert(Throws([&] {
            RecommendationRankingSnapshotIdentityCanonicalText(
                policy, protectedScope, 1,
                {homogeneousLeft, incompatible});
        }));

    std::vector<RecommendationRankingEvaluation> inputs = {
        Evaluation(8, RecommendationEvaluationDisposition::invalidPersistedEvidence,
                   std::nullopt, "z"),
        Evaluation(7, RecommendationEvaluationDisposition::completedDuplicate,
                   std::nullopt, "z"),
        Evaluation(6, RecommendationEvaluationDisposition::staleSourceEvidence,
                   std::nullopt, "z"),
        Evaluation(9,
                   RecommendationEvaluationDisposition::unsupportedRecommendationFamily,
                   std::nullopt, "z"),
        Evaluation(5, RecommendationEvaluationDisposition::blockedActiveDuplicate,
                   std::nullopt, "z"),
        Evaluation(4, RecommendationEvaluationDisposition::insufficientEvidence,
                   std::nullopt, "z"),
        Evaluation(3, RecommendationEvaluationDisposition::blockedPendingDuplicate,
                   std::nullopt, "z"),
        Evaluation(2, RecommendationEvaluationDisposition::advisoryReady,
                   0.8, "b"),
        Evaluation(1, RecommendationEvaluationDisposition::advisoryReady,
                   0.8, "a")};
    const auto ranked = RankRecommendationEvaluationEvidence(policy, inputs, 100);
    assert(ranked.size() == inputs.size());
    assert(ranked[0].evaluation.evaluationResultId == 1);
    assert(ranked[1].evaluation.evaluationResultId == 2);
    assert(ranked[2].evaluation.evaluationResultId == 3);
    assert(ranked[3].evaluation.evaluationResultId == 5);
    assert(ranked[4].evaluation.evaluationResultId == 7);
    assert(ranked[5].evaluation.evaluationResultId == 4);
    assert(ranked[6].evaluation.evaluationResultId == 6);
    assert(ranked[7].evaluation.evaluationResultId == 9);
    assert(ranked[8].evaluation.evaluationResultId == 8);
    assert(ranked[0].bucketRank == 1 && ranked[1].bucketRank == 2);
    assert(ranked[2].bucketRank == 1 && ranked[5].bucketRank == 1);
    assert(ranked[0].topPositiveComponent == "leader_quality");
    assert(ranked[0].topPenaltyComponent == "horizon_change_penalty");

    auto topTie = Evaluation(40,
        RecommendationEvaluationDisposition::advisoryReady, 0.6, "tie");
    topTie.components = {Component("z_positive", 0.2, false),
                         Component("a_positive", 0.2, false),
                         Component("z_penalty", 0.1, true),
                         Component("a_penalty", 0.1, true)};
    topTie.componentCount = static_cast<int>(topTie.components.size());
    const auto topTieRanked = RankRecommendationEvaluationEvidence(
        policy, {topTie}, 1);
    assert(topTieRanked.front().topPositiveComponent == "a_positive");
    assert(topTieRanked.front().topPenaltyComponent == "a_penalty");

    auto fallbackLeft = Evaluation(30,
        RecommendationEvaluationDisposition::advisoryReady, 0.5, "same");
    auto fallbackRight = Evaluation(29,
        RecommendationEvaluationDisposition::advisoryReady, 0.5, "same");
    fallbackLeft.evaluationIdentityCanonical = "shared_identity";
    fallbackRight.evaluationIdentityCanonical = "shared_identity";
    fallbackLeft.evaluationIdentityHash = RecommendationEvaluationCanonicalHash(
        fallbackLeft.evaluationIdentityCanonical);
    fallbackRight.evaluationIdentityHash = fallbackLeft.evaluationIdentityHash;
    const auto collisionOrdered = RankRecommendationEvaluationEvidence(
        policy, {fallbackLeft, fallbackRight}, 2);
    assert(collisionOrdered[0].evaluation.evaluationResultId == 29);
    assert(collisionOrdered[1].evaluation.evaluationResultId == 30);
    fallbackLeft.recommendationSemanticHash = "b";
    fallbackRight.recommendationSemanticHash = "a";
    assert(RankRecommendationEvaluationEvidence(
        policy, {fallbackLeft, fallbackRight}, 2)[0].evaluation.evaluationResultId == 29);

    auto positiveZero = Evaluation(31,
        RecommendationEvaluationDisposition::advisoryReady, 0.0, "same");
    auto negativeZero = Evaluation(32,
        RecommendationEvaluationDisposition::advisoryReady, -0.0, "same");
    positiveZero.evaluationIdentityCanonical = "signed_zero_identity";
    negativeZero.evaluationIdentityCanonical = "signed_zero_identity";
    positiveZero.evaluationIdentityHash = RecommendationEvaluationCanonicalHash(
        positiveZero.evaluationIdentityCanonical);
    negativeZero.evaluationIdentityHash = positiveZero.evaluationIdentityHash;
    assert(CanonicalRecommendationDouble(*positiveZero.finalScore) == "0");
    assert(CanonicalRecommendationDouble(*negativeZero.finalScore) == "0");
    assert(RankRecommendationEvaluationEvidence(
        policy, {negativeZero, positiveZero}, 2)[0].evaluation.evaluationResultId == 31);

    std::reverse(inputs.begin(), inputs.end());
    const auto reordered = RankRecommendationEvaluationEvidence(policy, inputs, 100);
    for (std::size_t index = 0; index < ranked.size(); ++index)
        assert(ranked[index].evaluation.evaluationResultId ==
               reordered[index].evaluation.evaluationResultId);
    assert(RankRecommendationEvaluationEvidence(policy, inputs, 2).size() == 2);

    const std::string identity =
        RecommendationRankingSnapshotIdentityCanonicalText(
            policy, run, 100, inputs);
    const std::string membership =
        RecommendationRankingMembershipCanonicalText(inputs);
    const auto populationSemantics =
        ValidateRecommendationRankingPopulationSemantics(inputs);
    assert(identity ==
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            policy, run, 100, membership, populationSemantics));
    assert(identity == RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, inputs));
    auto membershipChanged = inputs;
    membershipChanged.pop_back();
    assert(identity != RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, membershipChanged));
    assert(identity != RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, RunScope(11), 100, inputs));
    assert(identity != RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 99, inputs));
    auto identityReordered = inputs;
    std::reverse(identityReordered.begin(), identityReordered.end());
    assert(membership ==
           RecommendationRankingMembershipCanonicalText(identityReordered));
    assert(identity == RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, identityReordered));
    auto displayChanged = inputs;
    displayChanged.front().symbol = "CHANGED_DISPLAY_ONLY";
    assert(identity == RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, displayChanged));
    auto evaluationIdentityChanged = inputs;
    evaluationIdentityChanged.front().evaluationIdentityCanonical +=
        ";member[0].evaluation_result_id=1;NULL;global;\xc3\xa9";
    evaluationIdentityChanged.front().evaluationIdentityHash =
        RecommendationEvaluationCanonicalHash(
            evaluationIdentityChanged.front().evaluationIdentityCanonical);
    assert(identity != RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, evaluationIdentityChanged));
    auto resultIdChanged = inputs;
    resultIdChanged.front().evaluationResultId = 10000;
    assert(identity != RecommendationRankingSnapshotIdentityCanonicalText(
                           policy, run, 100, resultIdChanged));
    auto sharedCanonicalLeft = Evaluation(1,
        RecommendationEvaluationDisposition::advisoryReady, 0.5, "a");
    auto sharedCanonicalRight = Evaluation(10,
        RecommendationEvaluationDisposition::advisoryReady, 0.5, "b");
    sharedCanonicalLeft.evaluationIdentityCanonical = "shared;member[1]=10";
    sharedCanonicalRight.evaluationIdentityCanonical =
        sharedCanonicalLeft.evaluationIdentityCanonical;
    sharedCanonicalLeft.evaluationIdentityHash =
        RecommendationEvaluationCanonicalHash(
            sharedCanonicalLeft.evaluationIdentityCanonical);
    sharedCanonicalRight.evaluationIdentityHash =
        sharedCanonicalLeft.evaluationIdentityHash;
    const std::string sharedMembership =
        RecommendationRankingMembershipCanonicalText(
            {sharedCanonicalRight, sharedCanonicalLeft});
    assert(sharedMembership.find("count=2") != std::string::npos);
    assert(sharedMembership.find("evaluation_result_id=1;") != std::string::npos);
    assert(sharedMembership.ends_with("evaluation_result_id=10"));
    assert(RecommendationRankingMembershipCanonicalText({}).find("count=0") !=
           std::string::npos);
    assert(Throws([&] {
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            policy, run, 100, "", populationSemantics);
    }));
    const std::vector<std::string> malformedMemberships = {
        "not_a_membership",
        "experiment_recommendation_ranking_membership_v1;count=00",
        "experiment_recommendation_ranking_membership_v1;count=0;trailing",
        "experiment_recommendation_ranking_membership_v1;count=1",
        "experiment_recommendation_ranking_membership_v1;count=1;"
        "member[0].evaluation_identity=1:x;"
        "member[0].evaluation_result_id=01",
        "experiment_recommendation_ranking_membership_v1;count=1;"
        "member[0].evaluation_identity=2:x;"
        "member[0].evaluation_result_id=1"};
    for (const auto& malformedMembership : malformedMemberships)
        assert(Throws([&] {
            RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
                policy, run, 100, malformedMembership, populationSemantics);
        }));
    const std::string unsortedMembership =
        "experiment_recommendation_ranking_membership_v1;count=2;"
        "member[0].evaluation_identity=1:z;member[0].evaluation_result_id=2;"
        "member[1].evaluation_identity=1:a;member[1].evaluation_result_id=1";
    assert(Throws([&] {
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            policy, run, 100, unsortedMembership, populationSemantics);
    }));
    const std::string duplicateResultMembership =
        "experiment_recommendation_ranking_membership_v1;count=2;"
        "member[0].evaluation_identity=1:a;member[0].evaluation_result_id=1;"
        "member[1].evaluation_identity=1:b;member[1].evaluation_result_id=1";
    assert(Throws([&] {
        RecommendationRankingSnapshotIdentityCanonicalTextFromMembership(
            policy, run, 100, duplicateResultMembership, populationSemantics);
    }));
    assert(Throws([&] {
        auto invalid = inputs.front();
        invalid.evaluationIdentityCanonical.clear();
        RecommendationRankingMembershipCanonicalText({invalid});
    }));
    assert(Throws([&] {
        auto invalid = inputs.front();
        invalid.finalScore = std::numeric_limits<double>::quiet_NaN();
        invalid.disposition = RecommendationEvaluationDisposition::advisoryReady;
        invalid.eligibility = RecommendationEligibility::eligible;
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        invalid.components.front().weightedContribution =
            std::numeric_limits<double>::infinity();
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        invalid.components.clear();
        invalid.componentCount = 0;
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::insufficientEvidence,
            std::nullopt, "x");
        invalid.components = {Component("unexpected", 0.1, false)};
        invalid.componentCount = 1;
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        invalid.finalScore = -std::numeric_limits<double>::infinity();
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        invalid.components.front().componentName =
            invalid.components.back().componentName;
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        auto invalid = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        --invalid.componentCount;
        RankRecommendationEvaluationEvidence(policy, {invalid}, 10);
    }));
    assert(Throws([&] {
        const auto duplicate = Evaluation(20,
            RecommendationEvaluationDisposition::advisoryReady, 0.5, "x");
        RankRecommendationEvaluationEvidence(policy, {duplicate, duplicate}, 10);
    }));
    assert(Throws([&] {
        std::vector<RecommendationRankingEvaluation> oversized;
        oversized.reserve(kMaximumRecommendationRankingInputs + 1);
        for (int index = 0; index <= kMaximumRecommendationRankingInputs; ++index)
            oversized.push_back(Evaluation(1000 + index,
                RecommendationEvaluationDisposition::insufficientEvidence,
                std::nullopt, "semantic_" + std::to_string(index)));
        RankRecommendationEvaluationEvidence(policy, oversized,
                                              kMaximumRecommendationRankingMembers);
    }));
    assert(Throws([&] {
        std::vector<RecommendationRankingEvaluation> oversized;
        oversized.reserve(kMaximumRecommendationRankingInputs + 1);
        for (int index = 0; index <= kMaximumRecommendationRankingInputs; ++index)
            oversized.push_back(Evaluation(3000 + index,
                RecommendationEvaluationDisposition::insufficientEvidence,
                std::nullopt, "semantic_" + std::to_string(index)));
        RecommendationRankingMembershipCanonicalText(oversized);
    }));

    const auto left = Evaluation(20,
        RecommendationEvaluationDisposition::advisoryReady, 0.8, "left");
    auto right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right");
    right.components.front().weightedContribution = 0.2;
    auto comparison = CompareRecommendationEvaluations(left, right);
    assert(comparison.state == RecommendationComparisonState::comparable);
    assert(std::fabs(*comparison.scoreDelta - 0.1) < 1e-12);
    assert(comparison.componentDifferences.size() == 2);
    assert(comparison.largestPositiveDifference == "leader_quality");
    comparison = CompareRecommendationEvaluations(left, right, 1, 2, true);
    assert(comparison.state == RecommendationComparisonState::leftRankedHigher);
    assert(comparison.rankDelta == -1);
    comparison = CompareRecommendationEvaluations(left, right, 2, 1, true);
    assert(comparison.state == RecommendationComparisonState::rightRankedHigher);
    assert(comparison.rankDelta == 1);
    assert(Throws([&] {
        CompareRecommendationEvaluations(left, right, 0, 1, true);
    }));
    comparison = CompareRecommendationEvaluations(left, right, 1, 2, false);
    assert(comparison.state == RecommendationComparisonState::incomparableScope);
    right.scoringVersion = 2;
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::invalidScoringProvenance);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right");
    right.evaluationPolicyCanonical = "different_evaluation_policy";
    right.evaluationPolicyHash = left.evaluationPolicyHash;
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::invalidEvaluationProvenance);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right");
    right.evaluationPolicyHash = "different_hash_but_same_canonical";
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::invalidEvaluationProvenance);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right");
    right.evaluationIdentityHash = "fnv1a64:0000000000000000";
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::invalidEvaluationProvenance);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right");
    right.evaluatorVersion = 2;
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::invalidEvaluationProvenance);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.7, "right",
        "head_lr_mult");
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::incomparableFamily);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::insufficientEvidence,
        std::nullopt, "right");
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::incomparableMissingScore);
    right = Evaluation(21,
        RecommendationEvaluationDisposition::advisoryReady, 0.8, "right");
    assert(CompareRecommendationEvaluations(left, right).state ==
           RecommendationComparisonState::identicalScoreTie);
    right.components.pop_back();
    right.componentCount = static_cast<int>(right.components.size());
    comparison = CompareRecommendationEvaluations(left, right);
    assert(comparison.state == RecommendationComparisonState::identicalScoreTie);
    const auto missingPenalty = std::find_if(
        comparison.componentDifferences.begin(),
        comparison.componentDifferences.end(),
        [](const RecommendationComponentDifference& difference) {
            return difference.componentName == "horizon_change_penalty";
        });
    assert(missingPenalty != comparison.componentDifferences.end());
    assert(missingPenalty->leftContribution);
    assert(!missingPenalty->rightContribution);
    assert(!missingPenalty->contributionDelta);
    assert(!comparison.largestPenaltyDifference);
    assert(Throws([&] {
        auto mismatch = Evaluation(21,
            RecommendationEvaluationDisposition::advisoryReady, 0.8, "right");
        mismatch.components.back().penalty = false;
        CompareRecommendationEvaluations(left, mismatch);
    }));
    assert(Throws([&] {
        auto duplicate = Evaluation(21,
            RecommendationEvaluationDisposition::advisoryReady, 0.8, "right");
        duplicate.components.push_back(duplicate.components.front());
        duplicate.componentCount = static_cast<int>(duplicate.components.size());
        CompareRecommendationEvaluations(left, duplicate);
    }));
    auto blocked = Evaluation(22,
        RecommendationEvaluationDisposition::blockedActiveDuplicate,
        std::nullopt, "blocked");
    assert(CompareRecommendationEvaluations(left, blocked, 1, 2, true).state ==
           RecommendationComparisonState::incomparableMissingScore);

    return 0;
}
