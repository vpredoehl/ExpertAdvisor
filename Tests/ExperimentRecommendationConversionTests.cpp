#include <cassert>
#include <cmath>
#include <iostream>
#include <limits>
#include <locale>
#include <string>

#include "../Sources/ExperimentRecommendationConversion.hpp"

using namespace EA::ExperimentRecommendation;

namespace
{

class CommaDecimalPoint final : public std::numpunct<char>
{
protected:
    char do_decimal_point() const override { return ','; }
    char do_thousands_sep() const override { return '.'; }
    std::string do_grouping() const override { return "\1"; }
};

ExperimentInvocationConfiguration SourceInvocation()
{
    ExperimentInvocationConfiguration invocation;
    invocation.configuration.symbol = "eurusd";
    invocation.configuration.predictionHorizon = 12;
    invocation.configuration.labelThreshold = 0.001;
    invocation.configuration.coreLrMult = 1.0;
    invocation.configuration.headLrMult = 5.0;
    invocation.configuration.targetEpochs = 120;
    invocation.configuration.trainStartDate = "2010-01-01";
    invocation.configuration.trainEndDate = "2025-01-01";
    invocation.configuration.inferStartDate = "2025-01-01";
    invocation.configuration.inferEndDate = "2026-01-01";
    invocation.checkpointInterval = 15;
    invocation.resumeModelId = 77;
    return invocation;
}

void SetRecommendationIdentity(RecommendationConversionRequest& request)
{
    assert(request.mutations.size() == 1);
    ExperimentInvocationConfiguration proposed = request.sourceInvocation;
    const RecommendationConversionMutation& mutation = request.mutations.front();
    if (mutation.family == kCoreLrMult)
        proposed.configuration.coreLrMult = std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kHeadLrMult)
        proposed.configuration.headLrMult = std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kLabelThreshold)
        proposed.configuration.labelThreshold = std::stod(mutation.proposedValueCanonical);
    else if (mutation.family == kPredictionHorizon)
        proposed.configuration.predictionHorizon =
            std::stoi(mutation.proposedValueCanonical);
    else
        return;
    const auto semantic = BuildRecommendationCandidateIdentity(
        proposed.configuration);
    const auto invocation = BuildRecommendationInvocationIdentity(proposed);
    request.recommendationSemanticCanonical = semantic.canonicalText;
    request.recommendationSemanticHash = semantic.hash;
    request.recommendationInvocationCanonical = invocation.canonicalText;
    request.recommendationInvocationHash = invocation.hash;
}

RecommendationConversionRequest ValidRequest(
    const std::string& family = kCoreLrMult)
{
    RecommendationConversionRequest request;
    request.recommendationExists = true;
    request.recommendationId = 42;
    request.recommendationStatus = RecommendationStatus::approved;
    request.sourceExperimentId = 17;
    request.recommendationSourceExperimentId = 17;
    request.sourceInvocation = SourceInvocation();

    request.reviewAuthorization.present = true;
    request.reviewAuthorization.recommendationId = 42;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::approve;
    request.reviewAuthorization.resultingStatus = RecommendationStatus::approved;
    request.reviewAuthorization.latestActionEffective = true;
    request.reviewAuthorization.authorizationCanonical =
        "review_authorization_v1;operator=7:analyst;note=8:a,b:c=%";
    request.reviewAuthorization.authorizationHash = RecommendationCanonicalHash(
        request.reviewAuthorization.authorizationCanonical);

    request.evaluation.state = RecommendationConversionEvidenceState::completed;
    request.evaluation.valid = true;
    request.evaluation.recommendationId = 42;
    request.evaluation.sourceExperimentId = 17;
    request.evaluation.eligibility = RecommendationEligibility::eligible;
    request.evaluation.disposition =
        RecommendationEvaluationDisposition::advisoryReady;
    request.evaluation.evaluationIdentityCanonical =
        "evaluation_identity_v1;value=ready";
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    request.evaluation.evaluationPolicyCanonical =
        "evaluation_policy_v1;policy=stable";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);

    request.score.state = RecommendationConversionEvidenceState::completed;
    request.score.valid = true;
    request.score.recommendationId = 42;
    request.score.finalScore = 0.75;
    request.score.scoringPolicyCanonical = "scoring_policy_v1;policy=stable";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;

    if (family == kCoreLrMult)
        request.mutations.push_back({family, "1", "1.25"});
    else if (family == kHeadLrMult)
        request.mutations.push_back({family, "5", "4.5"});
    else if (family == kLabelThreshold)
        request.mutations.push_back({family, "0.001", "0.0011"});
    else if (family == kPredictionHorizon)
        request.mutations.push_back({family, "12", "16"});
    else
        request.mutations.push_back({family, "1", "2"});
    SetRecommendationIdentity(request);
    return request;
}

RecommendationConversionRankingProvenance Ranking()
{
    RecommendationConversionRankingProvenance ranking;
    ranking.snapshotIdentityCanonical = "ranking_snapshot_v1;scope=global";
    ranking.snapshotIdentityHash = RecommendationCanonicalHash(
        ranking.snapshotIdentityCanonical);
    ranking.memberIdentityCanonical = "ranking_member_v1;recommendation=42";
    ranking.memberIdentityHash = RecommendationCanonicalHash(
        ranking.memberIdentityCanonical);
    ranking.bucket = RecommendationRankingBucket::advisoryReady;
    ranking.bucketRank = 1;
    return ranking;
}

void ExpectReason(const RecommendationConversionRequest& request,
                  RecommendationConversionReason reason)
{
    const RecommendationConversionResult result =
        BuildProposedExperimentSpecification(request);
    assert(!result.eligibility.eligible);
    assert(result.eligibility.reason == reason);
    assert(!result.proposal);
    assert(!RecommendationConversionReasonText(reason).empty());
    assert(!result.eligibility.explanation.empty());
}

void AssertPreservedExcept(
    const ExperimentInvocationConfiguration& source,
    const ExperimentInvocationConfiguration& proposed,
    RecommendationMutationParameter changed)
{
    assert(source.checkpointInterval == proposed.checkpointInterval);
    assert(source.resumeModelId == proposed.resumeModelId);
    assert(source.configuration.symbol == proposed.configuration.symbol);
    assert(source.configuration.targetEpochs == proposed.configuration.targetEpochs);
    assert(source.configuration.trainStartDate == proposed.configuration.trainStartDate);
    assert(source.configuration.trainEndDate == proposed.configuration.trainEndDate);
    assert(source.configuration.inferStartDate == proposed.configuration.inferStartDate);
    assert(source.configuration.inferEndDate == proposed.configuration.inferEndDate);
    if (changed != RecommendationMutationParameter::predictionHorizon)
        assert(source.configuration.predictionHorizon ==
               proposed.configuration.predictionHorizon);
    if (changed != RecommendationMutationParameter::labelThreshold)
        assert(source.configuration.labelThreshold ==
               proposed.configuration.labelThreshold);
    if (changed != RecommendationMutationParameter::coreLrMult)
        assert(source.configuration.coreLrMult == proposed.configuration.coreLrMult);
    if (changed != RecommendationMutationParameter::headLrMult)
        assert(source.configuration.headLrMult == proposed.configuration.headLrMult);
}

std::string Identity(const RecommendationConversionRequest& request)
{
    const RecommendationConversionResult result =
        BuildProposedExperimentSpecification(request);
    assert(result.eligibility.eligible);
    assert(result.proposal);
    assert(result.proposal->conversionIdentityHash ==
           RecommendationCanonicalHash(
               result.proposal->conversionIdentityCanonical));
    return result.proposal->conversionIdentityHash;
}

} // namespace

int main()
{
    const struct MutationCase
    {
        const char* family;
        RecommendationMutationParameter parameter;
    } mutationCases[] = {
        {kCoreLrMult, RecommendationMutationParameter::coreLrMult},
        {kHeadLrMult, RecommendationMutationParameter::headLrMult},
        {kLabelThreshold, RecommendationMutationParameter::labelThreshold},
        {kPredictionHorizon, RecommendationMutationParameter::predictionHorizon}};
    for (const MutationCase& test : mutationCases)
    {
        const RecommendationConversionRequest request = ValidRequest(test.family);
        const RecommendationConversionResult result =
            BuildProposedExperimentSpecification(request);
        assert(result.eligibility.eligible);
        assert(result.eligibility.reason == RecommendationConversionReason::eligible);
        assert(result.proposal);
        assert(result.proposal->changedParameter == test.parameter);
        AssertPreservedExcept(
            request.sourceInvocation, result.proposal->proposedInvocation,
            test.parameter);
        assert(result.proposal->conversionIdentityCanonical.starts_with(
            "experiment_recommendation_conversion_identity_v1"));
    }

    RecommendationConversionRequest request = ValidRequest();
    request.reviewAuthorization.present = false;
    ExpectReason(request, RecommendationConversionReason::manualAuthorizationMissing);

    request = ValidRequest();
    request.reviewAuthorization.superseded = true;
    request.reviewAuthorization.latestAction = RecommendationReviewAction::reject;
    ExpectReason(
        request,
        RecommendationConversionReason::manualAuthorizationSupersededOrIneffective);

    request = ValidRequest();
    request.ranking = Ranking();
    request.reviewAuthorization.present = false;
    ExpectReason(request, RecommendationConversionReason::manualAuthorizationMissing);

    request = ValidRequest();
    request.evaluation.disposition =
        RecommendationEvaluationDisposition::blockedActiveDuplicate;
    request.evaluation.eligibility = RecommendationEligibility::ineligible;
    ExpectReason(request, RecommendationConversionReason::recommendationBlocked);

    request = ValidRequest();
    request.evaluation.state = RecommendationConversionEvidenceState::pending;
    ExpectReason(
        request, RecommendationConversionReason::evaluationMissingOrIncomplete);
    request = ValidRequest();
    request.evaluation.valid = false;
    ExpectReason(request, RecommendationConversionReason::evaluationInvalid);

    request = ValidRequest();
    request.score.state = RecommendationConversionEvidenceState::missing;
    ExpectReason(request, RecommendationConversionReason::scoreMissingOrIncomplete);
    request = ValidRequest();
    request.score.finalScore = std::numeric_limits<double>::infinity();
    ExpectReason(request, RecommendationConversionReason::scoreInvalid);

    request = ValidRequest("future_mutation");
    ExpectReason(request, RecommendationConversionReason::unsupportedMutationFamily);

    request = ValidRequest();
    request.mutations.front().proposedValueCanonical = "1.250";
    ExpectReason(request, RecommendationConversionReason::malformedProposedValue);
    request = ValidRequest();
    request.mutations.front().proposedValueCanonical = "0";
    ExpectReason(
        request, RecommendationConversionReason::proposedValueOutsideAllowedRange);

    request = ValidRequest();
    request.mutations.front().sourceValueCanonical = "0.75";
    ExpectReason(request, RecommendationConversionReason::sourceValueMismatch);
    request = ValidRequest();
    request.mutations.front().proposedValueCanonical = "1";
    ExpectReason(
        request, RecommendationConversionReason::proposedValueEqualsSourceValue);

    request = ValidRequest();
    request.mutations.push_back({kHeadLrMult, "5", "5.5"});
    ExpectReason(request, RecommendationConversionReason::multipleMutationsDetected);

    request = ValidRequest();
    const auto first = BuildProposedExperimentSpecification(request);
    assert(first.proposal);
    request.existingConversionCanonicals.push_back(
        first.proposal->conversionIdentityCanonical);
    ExpectReason(request, RecommendationConversionReason::duplicateConversionIdentity);

    request = ValidRequest();
    request.evaluation.evaluationIdentityHash = "fnv1a64:0000000000000000";
    ExpectReason(request, RecommendationConversionReason::inconsistentProvenance);
    request = ValidRequest();
    request.recommendationSourceExperimentId = 18;
    ExpectReason(
        request,
        RecommendationConversionReason::inconsistentSourceExperimentIdentity);

    request = ValidRequest();
    request.sourceInvocation.configuration.coreLrMult.reset();
    ExpectReason(
        request, RecommendationConversionReason::sourceConfigurationIncomplete);

    request = ValidRequest();
    request.recommendationExists = false;
    ExpectReason(request, RecommendationConversionReason::recommendationMissing);
    request = ValidRequest();
    request.recommendationStatus = RecommendationStatus::proposed;
    ExpectReason(
        request,
        RecommendationConversionReason::recommendationLifecycleNotConvertible);

    const RecommendationConversionRequest deterministic = ValidRequest();
    const auto deterministicA = BuildProposedExperimentSpecification(deterministic);
    const auto deterministicB = BuildProposedExperimentSpecification(deterministic);
    assert(deterministicA.proposal && deterministicB.proposal);
    assert(deterministicA.proposal->conversionIdentityCanonical ==
           deterministicB.proposal->conversionIdentityCanonical);
    assert(deterministicA.proposal->conversionIdentityHash ==
           deterministicB.proposal->conversionIdentityHash);

    const std::locale originalLocale = std::locale();
    std::locale::global(std::locale(
        std::locale::classic(), new CommaDecimalPoint));
    const std::string localeIdentity = Identity(deterministic);
    std::locale::global(originalLocale);
    assert(localeIdentity == Identity(deterministic));

    request = ValidRequest();
    const std::string baseIdentity = Identity(request);
    request.evaluation.evaluationIdentityCanonical += ";revision=2";
    request.evaluation.evaluationIdentityHash = RecommendationCanonicalHash(
        request.evaluation.evaluationIdentityCanonical);
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.evaluation.evaluationPolicyCanonical += ";revision=2";
    request.evaluation.evaluationPolicyHash = RecommendationCanonicalHash(
        request.evaluation.evaluationPolicyCanonical);
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.score.scoringPolicyCanonical += ";revision=2";
    request.score.scoringPolicyHash = RecommendationCanonicalHash(
        request.score.scoringPolicyCanonical);
    request.evaluation.scoringPolicyHash = request.score.scoringPolicyHash;
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.reviewAuthorization.authorizationCanonical += ";revision=2";
    request.reviewAuthorization.authorizationHash = RecommendationCanonicalHash(
        request.reviewAuthorization.authorizationCanonical);
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.mutations.front().proposedValueCanonical = "1.5";
    SetRecommendationIdentity(request);
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.sourceInvocation.checkpointInterval = 16;
    SetRecommendationIdentity(request);
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.recommendationId = 43;
    request.reviewAuthorization.recommendationId = 43;
    request.evaluation.recommendationId = 43;
    request.score.recommendationId = 43;
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.sourceExperimentId = 18;
    request.recommendationSourceExperimentId = 18;
    request.evaluation.sourceExperimentId = 18;
    assert(Identity(request) != baseIdentity);
    request = ValidRequest();
    request.sourceInvocation.configuration.coreLrMult = 1.1;
    request.mutations.front().sourceValueCanonical = "1.1";
    SetRecommendationIdentity(request);
    assert(Identity(request) != baseIdentity);

    request = ValidRequest();
    request.ranking = Ranking();
    const std::string rankedIdentity = Identity(request);
    request.ranking->bucketRank = 99;
    request.ranking->bucket = RecommendationRankingBucket::blocked;
    assert(Identity(request) == rankedIdentity);
    assert(rankedIdentity == baseIdentity);

    request = ValidRequest();
    request.ranking = Ranking();
    const auto ranked = BuildProposedExperimentSpecification(request);
    assert(ranked.proposal);
    assert(ranked.proposal->rankingSnapshotIdentityHash ==
           request.ranking->snapshotIdentityHash);
    request.ranking.reset();
    const auto unranked = BuildProposedExperimentSpecification(request);
    assert(unranked.proposal);
    assert(!unranked.proposal->rankingSnapshotIdentityHash);
    assert(ranked.proposal->conversionIdentityHash ==
           unranked.proposal->conversionIdentityHash);

    assert(RecommendationConversionEvidenceStateText(
               RecommendationConversionEvidenceState::completed) == "completed");
    assert(RecommendationCanonicalHash("stable") ==
           RecommendationCanonicalHash("stable"));
    std::cout << "ExperimentRecommendationConversionTests passed\n";
}
