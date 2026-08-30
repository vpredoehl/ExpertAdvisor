#include "../Sources/ProfitabilityVerification.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <locale>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

namespace Verification = EA::ProfitabilityVerification;
namespace Profitability = EA::InferenceProfitability;

namespace
{

Verification::CampaignProfitabilityOutcomeJob Job(
    long long experimentId,
    long long modelId,
    std::vector<long long> recommendationIds,
    bool compatible = true)
{
    Verification::CampaignProfitabilityOutcomeJob job;
    job.validationCohortIdentityHash =
        Verification::kPhase12ValidationCohortIdentityHash;
    job.rankingSnapshotId = Verification::kPhase12RankingSnapshotId;
    job.sourceEvaluationRunId = Verification::kPhase12SourceEvaluationRunId;
    job.sourceExperimentId = experimentId;
    job.sourceModelId = modelId;
    job.outcomeStart = Verification::kPhase12OutcomeStart;
    job.outcomeEnd = Verification::kPhase12OutcomeEnd;
    job.featureSemanticHash = Profitability::DeterministicHash(
        "feature:" + std::to_string(modelId));
    job.modelLineageHash = Profitability::DeterministicHash(
        "lineage:" + std::to_string(modelId));
    job.modelArtifactContentHash = Profitability::DeterministicHash(
        "artifact:" + std::to_string(modelId));
    job.metricDefinitionCanonical = Profitability::kMetricDefinitionCanonical;
    job.metricDefinitionHash = Profitability::MetricDefinitionHash();
    job.recommendationIds = std::move(recommendationIds);
    job.compatible = compatible;
    job.compatibilityState = compatible ? "compatible" :
        "model_lineage_ambiguous";
    job.hash = Profitability::DeterministicHash(
        "job:" + std::to_string(experimentId) + ":" +
        std::to_string(modelId));
    return job;
}

Verification::CampaignProfitabilityForwardValidationTopN Top(
    int n,
    std::vector<long long> control,
    std::vector<long long> candidate,
    std::vector<long long> retained,
    std::vector<long long> entrants,
    std::vector<long long> exits)
{
    Verification::CampaignProfitabilityForwardValidationTopN top;
    top.n = n;
    top.controlRecommendationIds = std::move(control);
    top.candidateRecommendationIds = std::move(candidate);
    top.retainedRecommendationIds = std::move(retained);
    top.candidateOnlyEntrants = std::move(entrants);
    top.controlOnlyExits = std::move(exits);
    top.hash = Profitability::DeterministicHash("top:" + std::to_string(n));
    return top;
}

Verification::CampaignProfitabilityOutcomePreparation Preparation()
{
    Verification::CampaignProfitabilityOutcomePreparation preparation;
    preparation.artifactSha256 = Verification::kPhase12ArtifactSha256;
    preparation.artifactIdentityVerified = true;
    preparation.hash = Verification::kPhase12PreparationIdentityHash;
    preparation.jobs = {
        Job(150, 499, {341, 340, 343, 342}, false),
        Job(140, 559, {352, 353, 354}),
        Job(158, 639, {348, 349, 350, 351}),
        Job(146, 649, {356, 355, 357, 358}),
        Job(396, 999, {359, 361, 360}),
        Job(412, 1015, {378, 380, 379}),
        Job(414, 1019, {372, 371, 373, 374}),
        Job(418, 1027, {382, 381, 384, 383}),
        Job(419, 1029, {368, 370, 369}),
        Job(426, 1076, {401, 400, 403, 402}),
        Job(420, 1090, {397, 396, 398, 399}),
        Job(432, 1134, {344, 345, 347, 346}),
        Job(436, 1161, {392, 393, 395, 394}),
        Job(472, 1276, {413, 415, 414}),
        Job(526, 1546, {404, 406, 405}),
        Job(557, 1615, {365, 367, 366}),
        Job(561, 1619, {375, 377, 376}),
        Job(594, 1654, {388, 389, 391, 390}),
        Job(558, 1658, {407, 409, 408}),
        Job(560, 1660, {416, 418, 417}),
        Job(559, 1664, {385, 386, 387}),
        Job(564, 1676, {362, 364, 363}),
        Job(602, 1702, {410, 411, 412})};
    preparation.topN = {
        Top(5,
            {359, 404, 416, 361, 360},
            {416, 404, 418, 417, 410},
            {404, 416}, {410, 417, 418}, {359, 360, 361}),
        Top(10,
            {359, 404, 416, 361, 360, 406, 405, 378, 410, 418},
            {416, 404, 418, 417, 410, 406, 405, 359, 407, 411},
            {359, 404, 405, 406, 410, 416, 418},
            {407, 411, 417}, {360, 361, 378}),
        Top(20,
            {359, 404, 416, 361, 360, 406, 405, 378, 410, 418,
             417, 368, 380, 379, 411, 412, 370, 369, 362, 407},
            {416, 404, 418, 417, 410, 406, 405, 359, 407, 411,
             412, 361, 360, 378, 368, 409, 408, 362, 380, 379},
            {359, 360, 361, 362, 368, 378, 379, 380, 404, 405,
             406, 407, 410, 411, 412, 416, 417, 418},
            {408, 409}, {369, 370})};
    return preparation;
}

Verification::CampaignProfitabilityProspectiveOutcome Outcome(
    const Verification::CampaignProfitabilityOutcomeJob& job,
    double aggregateReturn,
    std::uint64_t actionableCount = 10)
{
    Verification::CampaignProfitabilityProspectiveOutcome outcome;
    outcome.resultId = job.sourceModelId;
    outcome.validationCohortIdentityHash =
        Verification::kPhase12ValidationCohortIdentityHash;
    outcome.rankingSnapshotId = Verification::kPhase12RankingSnapshotId;
    outcome.sourceEvaluationRunId =
        Verification::kPhase12SourceEvaluationRunId;
    outcome.sourceExperimentId = job.sourceExperimentId;
    outcome.sourceModelId = job.sourceModelId;
    outcome.outcomeStart = Verification::kPhase12OutcomeStart;
    outcome.outcomeEnd = Verification::kPhase12OutcomeEnd;
    outcome.jobIdentityHash = job.hash;
    outcome.featureSemanticHash = job.featureSemanticHash;
    outcome.modelLineageHash = job.modelLineageHash;
    outcome.modelArtifactContentHash = job.modelArtifactContentHash;
    outcome.metricDefinitionCanonical = Profitability::kMetricDefinitionCanonical;
    outcome.metricDefinitionHash = Profitability::MetricDefinitionHash();
    outcome.sourceContentHash = Profitability::DeterministicHash(
        "source:" + std::to_string(job.sourceModelId));
    outcome.predictionCount = actionableCount + 2;
    outcome.actionableCount = actionableCount;
    if (aggregateReturn > 0.0)
    {
        outcome.winningActionableCount = actionableCount;
        outcome.grossPositiveReturn = aggregateReturn;
    }
    else if (aggregateReturn < 0.0)
    {
        outcome.losingActionableCount = actionableCount;
        outcome.grossNegativeReturn = aggregateReturn;
    }
    outcome.aggregateReturn = aggregateReturn;
    if (actionableCount != 0)
        outcome.averageReturn = aggregateReturn /
            static_cast<double>(actionableCount);
    const auto number = [](double value) {
        std::ostringstream rendered;
        rendered.imbue(std::locale::classic());
        rendered << std::setprecision(17) << (value == 0.0 ? 0.0 : value);
        return rendered.str();
    };
    outcome.outcomeIdentityCanonical =
        "campaign_profitability_prospective_outcome_v1;job_hash=" + job.hash +
        ";cohort_hash=" + job.validationCohortIdentityHash +
        ";source_experiment_id=" + std::to_string(job.sourceExperimentId) +
        ";source_model_id=" + std::to_string(job.sourceModelId) +
        ";outcome_start=" + job.outcomeStart + ";outcome_end=" +
        job.outcomeEnd + ";metric_definition_hash=" +
        job.metricDefinitionHash + ";source_content_hash=" +
        outcome.sourceContentHash + ";prediction_count=" +
        std::to_string(outcome.predictionCount) + ";actionable_count=" +
        std::to_string(outcome.actionableCount) +
        ";winning_actionable_count=" +
        std::to_string(outcome.winningActionableCount) +
        ";losing_actionable_count=" +
        std::to_string(outcome.losingActionableCount) +
        ";aggregate_return=" + number(outcome.aggregateReturn) +
        ";average_return=" +
        (outcome.averageReturn ? number(*outcome.averageReturn) : "NULL");
    outcome.outcomeIdentityHash = Profitability::DeterministicHash(
        outcome.outcomeIdentityCanonical);
    return outcome;
}

const Verification::CampaignProfitabilityOutcomeJob& FindJob(
    const Verification::CampaignProfitabilityOutcomePreparation& preparation,
    long long modelId)
{
    const auto found = std::find_if(
        preparation.jobs.begin(), preparation.jobs.end(),
        [modelId](const auto& job) { return job.sourceModelId == modelId; });
    assert(found != preparation.jobs.end());
    return *found;
}

Verification::CampaignProfitabilityProspectiveComparisonRequest Request()
{
    Verification::CampaignProfitabilityProspectiveComparisonRequest request;
    request.validationCohortIdentityHash =
        Verification::kPhase12ValidationCohortIdentityHash;
    request.phase11ArtifactSha256 = Verification::kPhase12ArtifactSha256;
    request.phase12PreparationArtifactSha256 =
        Verification::kPhase12PreparationArtifactSha256;
    request.phase12PreparationIdentityHash =
        Verification::kPhase12PreparationIdentityHash;
    request.metricDefinitionCanonical = Profitability::kMetricDefinitionCanonical;
    request.metricDefinitionHash = Profitability::MetricDefinitionHash();
    request.outcomeStart = Verification::kPhase12OutcomeStart;
    request.outcomeEnd = Verification::kPhase12OutcomeEnd;
    request.currentDate = "2026-10-01";
    request.preparation = Preparation();
    for (const auto& [modelId, value] : std::vector<std::pair<long long, double>>{
             {999, 0.1}, {1015, 0.2}, {1029, 0.3}, {1658, 0.4},
             {1660, 0.5}, {1702, 0.6}})
        request.outcomes.push_back(Outcome(
            FindJob(request.preparation, modelId), value));
    return request;
}

bool Near(double left, double right)
{
    return std::abs(left - right) < 1e-12;
}

const Verification::CampaignProfitabilityProspectiveTopNComparison& TopAt(
    const Verification::CampaignProfitabilityProspectiveComparison& comparison,
    int n)
{
    const auto found = std::find_if(
        comparison.topN.begin(), comparison.topN.end(),
        [n](const auto& top) { return top.n == n; });
    assert(found != comparison.topN.end());
    return *found;
}

} // namespace

int main()
{
    const auto complete =
        Verification::BuildCampaignProfitabilityProspectiveComparison(Request());
    assert(complete.final);
    assert(complete.readiness == Verification::
        ProspectiveComparisonReadiness::comparisonComplete);
    assert(complete.fullFrozenCohortCoverage.requiredCount == 23);
    assert(complete.fullFrozenCohortCoverage.coveredCount == 6);
    assert((complete.fullFrozenCohortCoverage.incompatibleSourceModelIds ==
            std::vector<long long>{499}));

    const auto& top5 = TopAt(complete, 5);
    assert((top5.retainedRecommendationIds ==
            std::vector<long long>{404, 416}));
    assert((top5.entrantRecommendationIds ==
            std::vector<long long>{410, 417, 418}));
    assert((top5.exitRecommendationIds ==
            std::vector<long long>{359, 360, 361}));
    assert((top5.entrantSourceModelIds ==
            std::vector<long long>{1660, 1702}));
    assert((top5.exitSourceModelIds == std::vector<long long>{999}));
    assert(top5.entrantContribution->recommendationCount == 3);
    assert(top5.entrantContribution->uniqueSourceModelCount == 2);
    assert(top5.exitContribution->recommendationCount == 3);
    assert(top5.exitContribution->uniqueSourceModelCount == 1);
    assert(Near(top5.entrantContribution->aggregateReturn, 1.6));
    assert(Near(top5.exitContribution->aggregateReturn, 0.3));
    assert(Near(*top5.candidateMinusControlIncrementalProfitability, 1.3));
    assert(top5.entrantContribution->actionableCount == 30);
    assert(top5.exitContribution->actionableCount == 30);

    const auto& top10 = TopAt(complete, 10);
    assert((top10.entrantRecommendationIds ==
            std::vector<long long>{407, 411, 417}));
    assert((top10.exitRecommendationIds ==
            std::vector<long long>{360, 361, 378}));
    assert((top10.entrantSourceModelIds ==
            std::vector<long long>{1658, 1660, 1702}));
    assert((top10.exitSourceModelIds ==
            std::vector<long long>{999, 1015}));
    assert(Near(*top10.candidateMinusControlIncrementalProfitability, 1.1));

    const auto& top20 = TopAt(complete, 20);
    assert((top20.entrantRecommendationIds ==
            std::vector<long long>{408, 409}));
    assert((top20.exitRecommendationIds ==
            std::vector<long long>{369, 370}));
    assert((top20.entrantSourceModelIds ==
            std::vector<long long>{1658}));
    assert((top20.exitSourceModelIds == std::vector<long long>{1029}));
    assert(top20.entrantContribution->recommendationCount == 2);
    assert(top20.entrantContribution->uniqueSourceModelCount == 1);
    assert(Near(*top20.candidateMinusControlIncrementalProfitability, 0.2));

    // Common members need no outcome to compute the incremental statistic.
    assert(std::find(
        complete.fullFrozenCohortCoverage.coveredSourceModelIds.begin(),
        complete.fullFrozenCohortCoverage.coveredSourceModelIds.end(),
        1546) ==
        complete.fullFrozenCohortCoverage.coveredSourceModelIds.end());

    auto missingEntrant = Request();
    std::erase_if(missingEntrant.outcomes,
        [](const auto& outcome) { return outcome.sourceModelId == 1702; });
    const auto entrantPending =
        Verification::BuildCampaignProfitabilityProspectiveComparison(
            missingEntrant);
    assert(!TopAt(entrantPending, 5).final);
    assert(TopAt(entrantPending, 5).readiness == Verification::
        ProspectiveComparisonReadiness::incompleteChangedSelectionCoverage);

    auto missingExit = Request();
    std::erase_if(missingExit.outcomes,
        [](const auto& outcome) { return outcome.sourceModelId == 999; });
    const auto exitPending =
        Verification::BuildCampaignProfitabilityProspectiveComparison(
            missingExit);
    assert(!TopAt(exitPending, 5).final);
    assert(TopAt(exitPending, 5).readiness == Verification::
        ProspectiveComparisonReadiness::incompleteChangedSelectionCoverage);

    // A complete earlier cutoff must not mask an incomplete later cutoff in
    // the overall readiness state.
    auto onlyTop10Missing = Request();
    std::erase_if(onlyTop10Missing.outcomes,
        [](const auto& outcome) { return outcome.sourceModelId == 1015; });
    const auto partiallyComplete =
        Verification::BuildCampaignProfitabilityProspectiveComparison(
            onlyTop10Missing);
    assert(TopAt(partiallyComplete, 5).final);
    assert(!TopAt(partiallyComplete, 10).final);
    assert(TopAt(partiallyComplete, 20).final);
    assert(partiallyComplete.readiness == Verification::
        ProspectiveComparisonReadiness::incompleteChangedSelectionCoverage);

    // Lower-ranked model 559 and incompatible model 499 are intentionally
    // absent and do not block any changed-selection comparison.
    assert(TopAt(complete, 5).final);
    assert(TopAt(complete, 10).final);
    assert(TopAt(complete, 20).final);

    auto badPhase11Artifact = Request();
    badPhase11Artifact.phase11ArtifactSha256 = std::string(64, '0');
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badPhase11Artifact).readiness == Verification::
           ProspectiveComparisonReadiness::artifactIdentityMismatch);
    auto badPhase12Artifact = Request();
    badPhase12Artifact.phase12PreparationArtifactSha256 = std::string(64, '0');
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badPhase12Artifact).readiness == Verification::
           ProspectiveComparisonReadiness::artifactIdentityMismatch);
    auto badCohort = Request();
    badCohort.validationCohortIdentityHash = "fnv1a64:0000000000000000";
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badCohort).readiness == Verification::
           ProspectiveComparisonReadiness::cohortIdentityMismatch);
    auto badMetric = Request();
    badMetric.metricDefinitionHash = "fnv1a64:0000000000000000";
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badMetric).readiness == Verification::
           ProspectiveComparisonReadiness::metricIdentityMismatch);
    auto badWindow = Request();
    badWindow.outcomeEnd = "2026-10-01";
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badWindow).readiness == Verification::
           ProspectiveComparisonReadiness::outcomeWindowIdentityMismatch);

    auto positive = Request();
    assert(*TopAt(Verification::BuildCampaignProfitabilityProspectiveComparison(
                      positive), 5)
                .candidateMinusControlIncrementalProfitability > 0.0);
    auto negative = Request();
    for (auto& outcome : negative.outcomes)
    {
        if (outcome.sourceModelId == 999)
            outcome = Outcome(FindJob(negative.preparation, 999), 1.0);
    }
    assert(*TopAt(Verification::BuildCampaignProfitabilityProspectiveComparison(
                      negative), 5)
                .candidateMinusControlIncrementalProfitability < 0.0);
    auto zero = Request();
    for (auto& outcome : zero.outcomes)
    {
        if (outcome.sourceModelId == 999)
            outcome = Outcome(FindJob(zero.preparation, 999), 1.6 / 3.0);
    }
    assert(Near(*TopAt(
        Verification::BuildCampaignProfitabilityProspectiveComparison(zero), 5)
        .candidateMinusControlIncrementalProfitability, 0.0));

    auto reordered = Request();
    std::reverse(reordered.outcomes.begin(), reordered.outcomes.end());
    std::reverse(reordered.preparation.jobs.begin(),
                 reordered.preparation.jobs.end());
    std::reverse(reordered.preparation.topN.begin(),
                 reordered.preparation.topN.end());
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               reordered).hash == complete.hash);

    auto badMembership = Request();
    std::swap(badMembership.preparation.topN[0].candidateOnlyEntrants[0],
              badMembership.preparation.topN[0].candidateOnlyEntrants[1]);
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badMembership).readiness == Verification::
           ProspectiveComparisonReadiness::cohortIdentityMismatch);

    auto badSourceMapping = Request();
    for (auto& job : badSourceMapping.preparation.jobs)
        if (job.sourceModelId == 1702) job.recommendationIds = {411, 412};
    for (auto& job : badSourceMapping.preparation.jobs)
        if (job.sourceModelId == 1658)
            job.recommendationIds.push_back(410);
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badSourceMapping).readiness == Verification::
           ProspectiveComparisonReadiness::cohortIdentityMismatch);

    auto badOutcomeCanonical = Request();
    badOutcomeCanonical.outcomes.front().outcomeIdentityCanonical +=
        ";not_the_persisted_contract";
    badOutcomeCanonical.outcomes.front().outcomeIdentityHash =
        Profitability::DeterministicHash(
            badOutcomeCanonical.outcomes.front().outcomeIdentityCanonical);
    assert(Verification::BuildCampaignProfitabilityProspectiveComparison(
               badOutcomeCanonical).readiness == Verification::
           ProspectiveComparisonReadiness::artifactIdentityMismatch);

    auto productionPending = Request();
    productionPending.currentDate = "2026-08-29";
    productionPending.outcomes.clear();
    const auto pending =
        Verification::BuildCampaignProfitabilityProspectiveComparison(
            productionPending);
    assert(!pending.final);
    assert(pending.readiness ==
        Verification::ProspectiveComparisonReadiness::pendingOutcomes);
    for (const auto& top : pending.topN)
    {
        assert(!top.final);
        assert(!top.candidateMinusControlIncrementalProfitability);
        assert(top.readiness ==
            Verification::ProspectiveComparisonReadiness::pendingOutcomes);
    }

    static_assert(Verification::kLiveProfitabilityRankingWeight == 0.0);
    static_assert(Verification::kLiveProfitabilityScoreContribution == 0.0);
}
