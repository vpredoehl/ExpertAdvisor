#include "../Sources/ProfitabilityVerification.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace Verification = EA::ProfitabilityVerification;
namespace Profitability = EA::InferenceProfitability;
namespace Recommendation = EA::ExperimentRecommendation;

namespace
{

template <typename Function>
bool Throws(Function&& function)
{
    try
    {
        function();
        return false;
    }
    catch (const std::exception&)
    {
        return true;
    }
}

Profitability::Observation Observation(long long id,
                                       double average,
                                       std::uint64_t actionable)
{
    Profitability::Observation value;
    value.observationId = id;
    value.statistics.predictionCount = actionable;
    value.statistics.actionableCount = actionable;
    value.statistics.aggregateTerminalHorizonLogReturnSum =
        average * static_cast<double>(actionable);
    value.averageTerminalHorizonLogReturnPerActionablePrediction = average;
    value.observationIdentityHash = Profitability::DeterministicHash(
        "shadow-observation-" + std::to_string(id));
    return value;
}

Profitability::Observation ZeroActionableObservation(long long id)
{
    auto value = Observation(id, 0.0, 0);
    value.averageTerminalHorizonLogReturnPerActionablePrediction.reset();
    return value;
}

Verification::EvidenceResult Evidence(
    long long id,
    const std::optional<Profitability::Observation>& observation,
    Verification::EvidenceState state = Verification::EvidenceState::valid)
{
    Verification::EvidenceResult value;
    value.experimentId = id;
    value.state = state;
    value.reason = state == Verification::EvidenceState::valid
        ? "valid" : "explicitly_unavailable";
    value.observation = observation;
    value.evidenceIdentityHash = Profitability::DeterministicHash(
        "shadow-evidence-" + std::to_string(id));
    return value;
}

Verification::ShadowCandidate Candidate(
    long long id,
    int rank,
    double score,
    const Verification::EvidenceResult& evidence)
{
    Verification::ShadowCandidate value;
    value.rankingMemberId = 1000 + id;
    value.recommendationId = 2000 + id;
    value.recommendationEvaluationResultId = 3000 + id;
    value.recommendationEvaluationRunId = 6;
    value.sourceExperimentId = 4000 + id;
    value.sourceModelId = 5000 + id;
    value.symbol = "EURUSDRMP";
    value.horizon = 5;
    value.currentRank = rank;
    value.currentScore = score;
    value.profitability = evidence;
    return value;
}

const Verification::WeightedShadowCandidate& ByRecommendation(
    const Verification::WeightedShadowRanking& ranking,
    long long recommendationId)
{
    const auto found = std::find_if(
        ranking.candidates.begin(), ranking.candidates.end(),
        [recommendationId](const auto& candidate) {
            return candidate.source.recommendationId == recommendationId;
        });
    assert(found != ranking.candidates.end());
    return *found;
}

bool Close(double left, double right)
{
    return std::abs(left - right) <= 1e-15;
}

} // namespace

int main()
{
    using Recommendation::ProfitabilityShadowNormalizationInput;
    using Recommendation::ProfitabilityShadowNormalizationPolicy;
    using Recommendation::ProfitabilityShadowNormalizationState;

    assert(Verification::kMaximumPhase9ProfitabilityShadowWeight == 0.05);
    assert((Verification::ParseProfitabilityShadowWeights("0.05,0.01,0.025") ==
            std::vector<double>{0.01, 0.025, 0.05}));
    assert((Verification::ParseProfitabilityShadowWeights("0,0.01") ==
            std::vector<double>{0.0, 0.01}));
    for (const std::string malformed : {
             "", ",", "0.01,", " 0.01", "not-a-number", "nan", "inf",
             "-0.01", "0.0500000001", "0.01,0.010"})
        assert(Throws([&] {
            (void)Verification::ParseProfitabilityShadowWeights(malformed);
        }));

    ProfitabilityShadowNormalizationPolicy normalizationPolicy;
    normalizationPolicy.minimumAnalyzablePopulationSize = 2;
    normalizationPolicy.supportHalfSaturationActionableCount = 100;
    const std::vector<ProfitabilityShadowNormalizationInput> normalizationInputs{
        {11, 100, -0.02, Profitability::DeterministicHash("negative")},
        {12, 100, 0.01, Profitability::DeterministicHash("positive-tie-a")},
        {13, 300, 0.01, Profitability::DeterministicHash("positive-tie-b")},
        {14, 0, std::nullopt, Profitability::DeterministicHash("zero-actionable")}};
    const auto normalization =
        Recommendation::AnalyzeProfitabilityShadowNormalization(
            normalizationInputs, normalizationPolicy);
    assert(normalization.populatedEvidenceCount == 4);
    assert(normalization.analyzableEvidenceCount == 3);
    assert(normalization.zeroActionableCount == 1);
    assert(normalization.policyCanonical.find(
        "unavailable=excluded_from_population_and_no_contribution") !=
        std::string::npos);
    assert(normalization.policyCanonical.find(
        "zero_actionable=valid_zero_aggregate_average_undefined_no_contribution") !=
        std::string::npos);
    assert(normalization.policyHash ==
        Recommendation::ProfitabilityShadowNormalizationPolicyHash(
            normalizationPolicy));

    const auto result = [&](long long observationId) -> const auto& {
        const auto found = std::find_if(
            normalization.results.begin(), normalization.results.end(),
            [observationId](const auto& value) {
                return value.profitabilityObservationId == observationId;
            });
        assert(found != normalization.results.end());
        return *found;
    };
    assert(result(11).state == ProfitabilityShadowNormalizationState::available);
    assert(*result(11).normalizedProfitabilityValue < 0.0);
    assert(*result(12).normalizedProfitabilityValue > 0.0);
    assert(result(12).empiricalMidrankPercentile ==
           result(13).empiricalMidrankPercentile);
    assert(Close(result(11).supportReliability, 0.5));
    assert(Close(result(13).supportReliability, 0.75));
    assert(result(14).state ==
           ProfitabilityShadowNormalizationState::zeroActionable);
    assert(!result(14).rawProfitabilityMetric);
    assert(!result(14).empiricalMidrankPercentile);
    assert(!result(14).normalizedProfitabilityValue);

    auto reversedInputs = normalizationInputs;
    std::reverse(reversedInputs.begin(), reversedInputs.end());
    const auto reversedNormalization =
        Recommendation::AnalyzeProfitabilityShadowNormalization(
            reversedInputs, normalizationPolicy);
    assert(reversedNormalization.canonical == normalization.canonical);
    assert(reversedNormalization.hash == normalization.hash);

    const auto negative = Observation(21, -0.02, 100);
    const auto positive = Observation(22, 0.01, 100);
    const auto positiveTie = Observation(23, 0.01, 300);
    const auto zeroActionable = ZeroActionableObservation(24);
    std::vector<Verification::ShadowCandidate> candidates{
        Candidate(1, 1, 0.8, Evidence(1, negative)),
        Candidate(2, 2, 0.8, Evidence(
            2, std::nullopt, Verification::EvidenceState::unavailable)),
        Candidate(3, 3, 0.8, Evidence(3, positive)),
        Candidate(4, 4, 0.7, Evidence(4, zeroActionable)),
        Candidate(5, 5, 0.6, Evidence(5, positiveTie)),
        Candidate(6, 6, 0.5, Evidence(3, positive))};
    const std::string snapshotHash =
        Profitability::DeterministicHash("control-snapshot-5");

    const auto control = Verification::BuildWeightedShadowRanking(
        candidates, 5, 6, snapshotHash, 0.0, normalizationPolicy);
    assert(control.candidates.size() == candidates.size());
    for (std::size_t index = 0; index < control.candidates.size(); ++index)
    {
        assert(control.candidates[index].source.currentRank ==
               static_cast<int>(index + 1));
        assert(control.candidates[index].shadowRank ==
               static_cast<int>(index + 1));
        assert(control.candidates[index].rankDelta == 0);
        assert(control.candidates[index].shadowFinalScore ==
               *control.candidates[index].source.currentScore);
    }

    const auto at001 = Verification::BuildWeightedShadowRanking(
        candidates, 5, 6, snapshotHash, 0.01, normalizationPolicy);
    const auto at0025 = Verification::BuildWeightedShadowRanking(
        candidates, 5, 6, snapshotHash, 0.025, normalizationPolicy);
    const auto at005 = Verification::BuildWeightedShadowRanking(
        candidates, 5, 6, snapshotHash, 0.05, normalizationPolicy);
    const auto& negative001 = ByRecommendation(at001, 2001);
    const auto& unavailable001 = ByRecommendation(at001, 2002);
    const auto& positive001 = ByRecommendation(at001, 2003);
    const auto& zeroActionable001 = ByRecommendation(at001, 2004);
    assert(*negative001.profitabilityContribution < 0.0);
    assert(*positive001.profitabilityContribution > 0.0);
    assert(!unavailable001.profitabilityContribution);
    assert(!unavailable001.normalizedProfitabilityValue);
    assert(unavailable001.shadowFinalScore ==
           *unavailable001.source.currentScore);
    assert(!zeroActionable001.profitabilityContribution);
    assert(!zeroActionable001.normalizedProfitabilityValue);
    assert(zeroActionable001.shadowFinalScore ==
           *zeroActionable001.source.currentScore);
    assert(positive001.shadowRank < unavailable001.shadowRank);
    assert(unavailable001.shadowRank < negative001.shadowRank);

    assert(*ByRecommendation(at001, 2003).profitabilityContribution <=
           *ByRecommendation(at0025, 2003).profitabilityContribution);
    assert(*ByRecommendation(at0025, 2003).profitabilityContribution <=
           *ByRecommendation(at005, 2003).profitabilityContribution);
    assert(*ByRecommendation(at001, 2001).profitabilityContribution >=
           *ByRecommendation(at0025, 2001).profitabilityContribution);
    assert(*ByRecommendation(at0025, 2001).profitabilityContribution >=
           *ByRecommendation(at005, 2001).profitabilityContribution);
    assert(ByRecommendation(at0025, 2003).normalizedProfitabilityValue ==
           ByRecommendation(at0025, 2006).normalizedProfitabilityValue);

    auto reversedCandidates = candidates;
    std::reverse(reversedCandidates.begin(), reversedCandidates.end());
    const auto repeated = Verification::BuildWeightedShadowRanking(
        reversedCandidates, 5, 6, snapshotHash, 0.025, normalizationPolicy);
    assert(repeated.canonical == at0025.canonical);
    assert(repeated.hash == at0025.hash);
    assert(at0025.policyCanonical.find(
        "order=shadow_score_desc,control_rank_asc,ranking_member_id_asc") !=
        std::string::npos);
    assert(at0025.policyCanonical.find("activation=disabled") !=
           std::string::npos);
    assert(at0025.policyCanonical.find("database_write=false") !=
           std::string::npos);
    assert(at0025.policyCanonical.find("live_profitability_weight=0") !=
           std::string::npos);
    assert(at0025.policyCanonical.find(
        "repeated_candidate_evidence=deduplicated_by_observation_identity") !=
        std::string::npos);

    auto conflictingRepeatedEvidence = candidates;
    conflictingRepeatedEvidence.back().profitability.evidenceIdentityHash =
        Profitability::DeterministicHash("conflicting-repeated-evidence");
    assert(Throws([&] {
        (void)Verification::BuildWeightedShadowRanking(
            conflictingRepeatedEvidence, 5, 6, snapshotHash, 0.01,
            normalizationPolicy);
    }));

    assert(Throws([&] {
        (void)Verification::BuildWeightedShadowRanking(
            candidates, 5, 6, snapshotHash,
            std::numeric_limits<double>::quiet_NaN(), normalizationPolicy);
    }));
    assert(Throws([&] {
        (void)Verification::BuildWeightedShadowRanking(
            candidates, 5, 6, snapshotHash,
            std::numeric_limits<double>::infinity(), normalizationPolicy);
    }));
    assert(Throws([&] {
        (void)Verification::BuildWeightedShadowRanking(
            candidates, 5, 6, snapshotHash, -0.01, normalizationPolicy);
    }));
    assert(Throws([&] {
        (void)Verification::BuildWeightedShadowRanking(
            candidates, 5, 6, snapshotHash, 0.0500000001,
            normalizationPolicy);
    }));

    const auto phase10Weights =
        Verification::Phase10ProfitabilityCalibrationWeights();
    assert(phase10Weights.size() == 21);
    assert(phase10Weights.front() == 0.0);
    assert(Close(phase10Weights[1], 0.0025));
    assert(Close(phase10Weights[10], 0.025));
    assert(Close(phase10Weights.back(), 0.05));

    std::vector<Verification::ShadowCandidate> calibrationCandidates;
    for (int rank = 1; rank <= 25; ++rank)
    {
        Verification::EvidenceResult evidence;
        if (rank == 4 || rank == 5)
            evidence = Evidence(
                100 + rank, std::nullopt,
                Verification::EvidenceState::unavailable);
        else
        {
            const bool positive = (rank >= 6 && rank <= 18) || rank >= 23;
            const double average = positive
                ? 0.001 * static_cast<double>(rank)
                : -0.001 * static_cast<double>(rank);
            evidence = Evidence(
                100 + rank, Observation(100 + rank, average, 100));
        }
        calibrationCandidates.push_back(Candidate(
            100 + rank, rank,
            1.0 - 0.001 * static_cast<double>(rank), evidence));
    }
    std::vector<Verification::WeightedShadowRanking> sweep;
    for (const double weight : phase10Weights)
        sweep.push_back(Verification::BuildWeightedShadowRanking(
            calibrationCandidates, 5, 6, snapshotHash, weight));
    const auto calibration =
        Verification::BuildProfitabilityCalibrationReport(sweep);
    assert(calibration.weights.size() == 21);
    assert(calibration.anchorPairwise.size() == 3);
    assert(calibration.weights.front().totalMovement.movedUp == 0);
    assert(calibration.weights.front().totalMovement.unchanged == 25);
    assert(calibration.weights.front().totalMovement.movedDown == 0);
    assert(calibration.weights.front().validProfitabilityMembers == 23);
    assert(calibration.weights.front().unavailableMembers == 2);
    assert(calibration.weights[1].totalMovement.meanAbsoluteRankMovement >= 0.0);
    assert(calibration.weights.back().totalMovement.p90AbsoluteRankMovement >=
           calibration.weights.back().totalMovement.medianAbsoluteRankMovement);
    assert(calibration.responseCurve.firstBestTop5Weight);
    assert(calibration.responseCurve.firstTop10AtLeastNineWeight);
    assert(calibration.responseCurve.firstBestTop10Weight);
    assert(calibration.responseCurve.firstTop20ImprovementWeight);
    assert(calibration.responseCurve.minimumEffectiveWeight);
    assert(calibration.responseCurve.minimumEffectiveRegionEnd);
    assert(!calibration.responseCurve.stabilityRegions.empty());
    assert(!calibration.responseCurve.membershipDiscontinuityWeights.empty());
    assert(!calibration.hash.empty());

    std::vector<Verification::WeightedShadowRanking> repeatedSweep;
    for (const double weight : phase10Weights)
    {
        auto input = calibrationCandidates;
        std::reverse(input.begin(), input.end());
        repeatedSweep.push_back(Verification::BuildWeightedShadowRanking(
            input, 5, 6, snapshotHash, weight));
    }
    const auto repeatedCalibration =
        Verification::BuildProfitabilityCalibrationReport(repeatedSweep);
    assert(repeatedCalibration.canonical == calibration.canonical);
    assert(repeatedCalibration.hash == calibration.hash);

    auto invalidSweep = sweep;
    invalidSweep.erase(invalidSweep.begin() + 4);
    assert(Throws([&] {
        (void)Verification::BuildProfitabilityCalibrationReport(invalidSweep);
    }));

    assert(Verification::kPhase11PrecommittedProfitabilityWeight == 0.025);
    assert(Verification::TemporalCohortClassificationText(
               Verification::TemporalCohortClassification::
                   admissibleTemporalHoldout) ==
           "admissible_temporal_holdout");
    assert(Verification::TemporalCohortClassificationText(
               Verification::TemporalCohortClassification::
                   futureInformationLeakage) ==
           "future_information_leakage");
    Verification::CampaignProfitabilityShadowSource forwardSource;
    forwardSource.controlSnapshotId = 5;
    forwardSource.sourceEvaluationRunId = 6;
    forwardSource.controlSnapshotIdentityHash = snapshotHash;
    forwardSource.persistedMemberCount =
        static_cast<int>(candidates.size());
    forwardSource.candidates = candidates;
    const auto precommit = Verification::
        BuildCampaignProfitabilityForwardValidationPrecommit(
            forwardSource, "2026-08-28T16:30:10.000000Z",
            "2026-08-29", "2027-08-29");
    assert(precommit.controlWeight == 0.0);
    assert(precommit.candidateWeight == 0.025);
    assert(precommit.members.size() == candidates.size());
    assert(precommit.topN.size() == 3);
    assert(precommit.topN[0].n == 5);
    assert(precommit.topN[1].n == 10);
    assert(precommit.topN[2].n == 20);
    assert(!precommit.hash.empty());
    assert(precommit.canonical.find("precommitted_candidate_weight=5:0.025") !=
           std::string::npos);
    assert(precommit.canonical.find("activation=5:false") !=
           std::string::npos);
    assert(precommit.canonical.find("live_profitability_weight=1:0") !=
           std::string::npos);
    auto reversedForwardSource = forwardSource;
    std::reverse(reversedForwardSource.candidates.begin(),
                 reversedForwardSource.candidates.end());
    const auto repeatedPrecommit = Verification::
        BuildCampaignProfitabilityForwardValidationPrecommit(
            reversedForwardSource, "2026-08-28T16:30:10.000000Z",
            "2026-08-29", "2027-08-29");
    assert(repeatedPrecommit.canonical == precommit.canonical);
    assert(repeatedPrecommit.hash == precommit.hash);
    for (const auto& member : precommit.members)
        assert(member.controlRank == member.source.currentRank);
    assert(Throws([&] {
        (void)Verification::
            BuildCampaignProfitabilityForwardValidationPrecommit(
                forwardSource, "2026-08-28T16:30:10.000000Z",
                "2026-08-28", "2027-08-29");
    }));
    assert(Throws([&] {
        (void)Verification::
            BuildCampaignProfitabilityForwardValidationPrecommit(
                forwardSource, "2026-08-28T16:30:10.000000Z",
                "2027-08-29", "2027-08-29");
    }));
    assert(Throws([&] {
        (void)Verification::
            BuildCampaignProfitabilityForwardValidationPrecommit(
                forwardSource, "2026-08-28T16:30:10.000000Z",
                "2026-02-29", "2027-08-29");
    }));
    return 0;
}
