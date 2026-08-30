#include "ProfitabilityVerificationService.hpp"

#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ProfitabilityVerificationRepository.hpp"

#include <algorithm>
#include <cmath>
#include <chrono>
#include <ctime>
#include <iomanip>
#include <limits>
#include <locale>
#include <map>
#include <optional>
#include <set>
#include <sstream>
#include <stdexcept>

#include <pqxx/pqxx>

namespace EA::ProfitabilityVerification
{
namespace
{

namespace Recommendation = EA::ExperimentRecommendation;

std::string Boolean(bool value)
{
    return value ? "true" : "false";
}

std::string OptionalId(const std::optional<long long>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string OptionalCount(const std::optional<std::uint64_t>& value)
{
    return value ? std::to_string(*value) : "NULL";
}

std::string Number(double value)
{
    if (!std::isfinite(value)) return "NONFINITE";
    std::ostringstream output;
    output.imbue(std::locale::classic());
    output << std::setprecision(17) << (value == 0.0 ? 0.0 : value);
    return output.str();
}

std::string OptionalNumber(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

std::string MachineText(std::string value)
{
    for (char& character : value)
    {
        const unsigned char byte = static_cast<unsigned char>(character);
        const bool safe = (byte >= 'a' && byte <= 'z') ||
            (byte >= 'A' && byte <= 'Z') ||
            (byte >= '0' && byte <= '9') || character == '_' ||
            character == '-' || character == '.' || character == ':';
        if (!safe) character = '_';
    }
    return value.empty() ? "NULL" : value;
}

std::string CurrentUtcDate()
{
    const std::time_t now = std::time(nullptr);
    std::tm utc{};
    gmtime_r(&now, &utc);
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%d");
    return output.str();
}

const char* OutcomeSafetyFields()
{
    return ",activation=false,live_profitability_weight=0"
           ",production_ranking_modified=false,recommendation_modified=false"
           ",ranking_snapshot_modified=false,training_started=false"
           ",experiment_created=false,experiment_queued=false"
           ",scheduler_modified=false";
}

void RenderEvidence(std::ostream& output, const EvidenceResult& result)
{
    const auto& observation = result.observation;
    const std::optional<std::uint64_t> predictionCount = observation
        ? std::optional<std::uint64_t>{observation->statistics.predictionCount}
        : std::nullopt;
    const std::optional<std::uint64_t> actionableCount = observation
        ? std::optional<std::uint64_t>{observation->statistics.actionableCount}
        : std::nullopt;
    output << "PROFITABILITY_EVIDENCE"
           << ",experiment_id=" << result.experimentId
           << ",final_model_id=" << OptionalId(result.finalModelId)
           << ",final_inference_result_id="
           << OptionalId(result.finalInferenceEvalResultId)
           << ",profitability_observation_id="
           << (observation ? std::to_string(observation->observationId) : "NULL")
           << ",inference_scope="
           << (observation
                   ? InferenceProfitability::ScopeText(
                         observation->provenance.scope) : "final")
           << ",checkpoint_eval_id="
           << (observation
                   ? OptionalId(observation->provenance.checkpointEvalId) : "NULL")
           << ",metric_definition_canonical="
           << InferenceProfitability::kMetricDefinitionCanonical
           << ",metric_definition_hash="
           << InferenceProfitability::MetricDefinitionHash()
           << ",inference_start="
           << (observation ? observation->provenance.inferenceStart : "NULL")
           << ",inference_end="
           << (observation ? observation->provenance.inferenceEnd : "NULL")
           << ",prediction_count=" << OptionalCount(predictionCount)
           << ",actionable_count=" << OptionalCount(actionableCount)
           << ",aggregate_terminal_horizon_log_return_sum="
           << (observation
                   ? Number(observation->statistics
                         .aggregateTerminalHorizonLogReturnSum) : "NULL")
           << ",average_terminal_horizon_log_return_per_actionable_prediction="
           << (observation
                   ? OptionalNumber(observation
                         ->averageTerminalHorizonLogReturnPerActionablePrediction)
                   : "NULL")
           << ",observation_identity_hash="
           << (observation ? observation->observationIdentityHash : "NULL")
           << ",evidence_identity_hash=" << result.evidenceIdentityHash
           << ",evidence_status=" << EvidenceStateText(result.state)
           << ",reason=" << MachineText(result.reason)
           << ",read_only=true\n";
}

struct Coverage
{
    std::size_t candidateCount = 0;
    std::size_t validCount = 0;
    std::size_t unavailableCount = 0;
    std::size_t incompleteCount = 0;
    std::size_t invalidCount = 0;
    std::size_t zeroActionableCount = 0;
    std::size_t positiveAggregateCount = 0;
    std::size_t negativeAggregateCount = 0;
    std::size_t zeroAggregateCount = 0;
    std::size_t positiveAverageCount = 0;
    std::size_t negativeAverageCount = 0;
    std::size_t zeroAverageCount = 0;
    std::map<std::string, std::size_t> metricDefinitions;
};

Coverage Summarize(const ShadowRanking& shadow)
{
    Coverage coverage;
    coverage.candidateCount = shadow.candidates.size();
    for (const auto& candidate : shadow.candidates)
    {
        const auto& result = candidate.profitability;
        if (result.state == EvidenceState::valid)
            ++coverage.validCount;
        else if (result.state == EvidenceState::unavailable)
            ++coverage.unavailableCount;
        else if (result.state == EvidenceState::incomplete)
            ++coverage.incompleteCount;
        else
            ++coverage.invalidCount;
        if (!result.observation) continue;
        const auto& observation = *result.observation;
        ++coverage.metricDefinitions[observation.metricDefinitionHash];
        if (observation.statistics.actionableCount == 0)
            ++coverage.zeroActionableCount;
        const double aggregate =
            observation.statistics.aggregateTerminalHorizonLogReturnSum;
        aggregate > 0.0 ? ++coverage.positiveAggregateCount
            : aggregate < 0.0 ? ++coverage.negativeAggregateCount
                              : ++coverage.zeroAggregateCount;
        if (observation
                .averageTerminalHorizonLogReturnPerActionablePrediction)
        {
            const double average = *observation
                .averageTerminalHorizonLogReturnPerActionablePrediction;
            average > 0.0 ? ++coverage.positiveAverageCount
                : average < 0.0 ? ++coverage.negativeAverageCount
                                : ++coverage.zeroAverageCount;
        }
    }
    return coverage;
}

struct MovementSummary
{
    int count = 0;
    long long deltaSum = 0;
    int minimum = std::numeric_limits<int>::max();
    int maximum = std::numeric_limits<int>::min();
    int movedUp = 0;
    int unchanged = 0;
    int movedDown = 0;

    void Add(int delta)
    {
        ++count;
        deltaSum += delta;
        minimum = std::min(minimum, delta);
        maximum = std::max(maximum, delta);
        delta > 0 ? ++movedUp : delta < 0 ? ++movedDown : ++unchanged;
    }
};

void RenderMovement(std::ostream& output,
                    long long snapshotId,
                    double weight,
                    const std::string& dimension,
                    const std::string& value,
                    const MovementSummary& summary)
{
    output << "CAMPAIGN_PROFITABILITY_SHADOW_MOVEMENT"
           << ",control_snapshot_id=" << snapshotId
           << ",shadow_weight=" << Number(weight)
           << ",dimension=" << dimension
           << ",value=" << MachineText(value)
           << ",member_count=" << summary.count
           << ",mean_rank_delta="
           << (summary.count == 0 ? "NULL" :
               Number(static_cast<double>(summary.deltaSum) / summary.count))
           << ",min_rank_delta="
           << (summary.count == 0 ? "NULL" : std::to_string(summary.minimum))
           << ",max_rank_delta="
           << (summary.count == 0 ? "NULL" : std::to_string(summary.maximum))
           << ",moved_up=" << summary.movedUp
           << ",unchanged=" << summary.unchanged
           << ",moved_down=" << summary.movedDown << '\n';
}

std::string IdList(const std::set<long long>& ids)
{
    std::string value;
    for (const long long id : ids)
    {
        if (!value.empty()) value += ':';
        value += std::to_string(id);
    }
    return value.empty() ? "NONE" : value;
}

std::string IdList(const std::vector<long long>& ids)
{
    std::string value;
    for (const long long id : ids)
    {
        if (!value.empty()) value += ':';
        value += std::to_string(id);
    }
    return value.empty() ? "NONE" : value;
}

std::string OptionalWeight(const std::optional<double>& value)
{
    return value ? Number(*value) : "NULL";
}

bool SameWeight(double left, double right)
{
    return std::abs(left - right) <= 1e-12;
}

} // namespace

int RunVerificationCommand(const std::string& connectionString,
                           const std::vector<long long>& experimentIds,
                           std::ostream& output,
                           std::ostream& errors)
{
    (void)errors;
    if (experimentIds.empty())
        throw std::invalid_argument("profitability_experiment_set_empty");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    std::vector<EvidenceResult> results;
    results.reserve(experimentIds.size());
    for (const long long experimentId : experimentIds)
    {
        results.push_back(
            LoadAndVerifyExactFinalEvidence(transaction, experimentId));
        RenderEvidence(output, results.back());
    }
    const int exitCode = ExitCode(results);
    output << "PROFITABILITY_VERIFICATION_SUMMARY"
           << ",declared_experiment_count=" << results.size()
           << ",valid_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return r.state == EvidenceState::valid; })
           << ",unavailable_or_incomplete_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return r.state == EvidenceState::unavailable ||
                      r.state == EvidenceState::incomplete; })
           << ",invalid_or_ambiguous_count="
           << std::count_if(results.begin(), results.end(), [](const auto& r) {
                  return EvidenceStateIsInvalid(r.state); })
           << ",software_success=true,read_only=true,exit_code="
           << exitCode << '\n';
    return exitCode;
}

int RunCampaignReadinessCommand(const std::string& connectionString,
                                long long rankingSnapshotId,
                                std::ostream& output,
                                std::ostream& errors)
{
    (void)errors;
    if (rankingSnapshotId <= 0)
        throw std::invalid_argument("invalid_campaign_ranking_snapshot_id");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    if (!Recommendation::RecommendationCampaignPlanningSchemasExist(transaction) ||
        !InferenceProfitability::SchemaExists(transaction))
        throw std::runtime_error("profitability_readiness_schema_unavailable");

    Recommendation::RecommendationCampaignPlanningPolicy planningPolicy;
    planningPolicy.maximumCandidatesConsidered =
        Recommendation::kMaximumRecommendationCampaignCandidates;
    Recommendation::RecommendationCampaignPlanningScope scope;
    scope.rankingSnapshotId = rankingSnapshotId;
    const auto input = Recommendation::LoadRecommendationCampaignPlanInput(
        transaction, planningPolicy, scope);

    std::vector<ShadowCandidate> candidates;
    candidates.reserve(input.candidates.size());
    for (const auto& source : input.candidates)
    {
        EvidenceResult evidence = LoadAndVerifyExactFinalEvidence(
            transaction, source.sourceExperimentId);
        evidence = EnforceFrozenCampaignEvidence(
            source.sourceModelId,
            source.finalProfitabilityEvidence,
            std::move(evidence));

        ShadowCandidate candidate;
        candidate.rankingMemberId = source.rankingMemberId;
        candidate.recommendationId = source.recommendationId;
        candidate.sourceExperimentId = source.sourceExperimentId;
        candidate.sourceModelId = source.sourceModelId;
        candidate.currentRank = source.rankingPosition;
        candidate.currentScore = source.rankingScore;
        candidate.leaderScore = source.leaderScore;
        candidate.inferenceAccuracy = source.inferenceAccuracy;
        candidate.predictedNeutralProportion =
            source.predictedNeutralProportion;
        candidate.profitability = std::move(evidence);
        candidates.push_back(std::move(candidate));
    }

    const ShadowRanking shadow = BuildShadowRanking(std::move(candidates));
    for (const auto& candidate : shadow.candidates)
    {
        const auto& observation = candidate.profitability.observation;
        output << "CAMPAIGN_PROFITABILITY_SHADOW_CANDIDATE"
               << ",ranking_snapshot_id=" << rankingSnapshotId
               << ",ranking_member_id=" << candidate.rankingMemberId
               << ",recommendation_id=" << candidate.recommendationId
               << ",source_experiment_id="
               << candidate.sourceExperimentId
               << ",source_model_id=" << OptionalId(candidate.sourceModelId)
               << ",current_rank=" << candidate.currentRank
               << ",current_score=" << OptionalNumber(candidate.currentScore)
               << ",leader_score=" << Number(candidate.leaderScore)
               << ",inference_accuracy="
               << Number(candidate.inferenceAccuracy)
               << ",predicted_neutral_proportion="
               << OptionalNumber(candidate.predictedNeutralProportion)
               << ",profitability_evidence_available="
               << Boolean(candidate.profitability.state == EvidenceState::valid)
               << ",profitability_evidence_status="
               << EvidenceStateText(candidate.profitability.state)
               << ",profitability_sign="
               << ProfitabilitySignText(candidate.profitabilitySign)
               << ",actionable_count="
               << (observation
                       ? std::to_string(
                             observation->statistics.actionableCount) : "NULL")
               << ",aggregate_terminal_horizon_log_return_sum="
               << (observation
                       ? Number(observation->statistics
                             .aggregateTerminalHorizonLogReturnSum) : "NULL")
               << ",average_terminal_horizon_log_return_per_actionable_prediction="
               << (observation
                       ? OptionalNumber(observation
                             ->averageTerminalHorizonLogReturnPerActionablePrediction)
                       : "NULL")
               << ",profitability_evidence_identity="
               << candidate.profitability.evidenceIdentityHash
               << ",profitability_shadow_rank="
               << candidate.profitabilityShadowRank
               << ",rank_delta=" << candidate.rankDelta
               << ",live_rank_authoritative=true"
               << ",candidate_hash=" << candidate.hash << '\n';
    }

    const Coverage coverage = Summarize(shadow);
    output << "CAMPAIGN_PROFITABILITY_COVERAGE"
           << ",ranking_snapshot_id=" << rankingSnapshotId
           << ",candidate_count=" << coverage.candidateCount
           << ",exact_final_profitability_count=" << coverage.validCount
           << ",missing_profitability_count=" << coverage.unavailableCount
           << ",incomplete_final_evidence_count=" << coverage.incompleteCount
           << ",invalid_or_ambiguous_count=" << coverage.invalidCount
           << ",zero_actionable_count=" << coverage.zeroActionableCount
           << ",positive_aggregate_count=" << coverage.positiveAggregateCount
           << ",negative_aggregate_count=" << coverage.negativeAggregateCount
           << ",zero_aggregate_count=" << coverage.zeroAggregateCount
           << ",positive_average_count=" << coverage.positiveAverageCount
           << ",negative_average_count=" << coverage.negativeAverageCount
           << ",zero_average_count=" << coverage.zeroAverageCount
           << ",final_evidence_count=" << coverage.validCount
           << ",checkpoint_evidence_count=0"
           << ",current_rank_changed=false,read_only=true\n";
    for (const auto& [hash, count] : coverage.metricDefinitions)
        output << "CAMPAIGN_PROFITABILITY_METRIC_DEFINITION"
               << ",metric_definition_hash=" << hash
               << ",candidate_count=" << count << '\n';

    const ReadinessGate gate = EvaluateReadinessGate(true, true, shadow);
    output << "CAMPAIGN_PROFITABILITY_SHADOW_POLICY"
           << ",version=" << shadow.policyVersion
           << ",policy_hash=" << shadow.policyHash
           << ",canonical=" << shadow.policyCanonical
           << ",shadow_ranking_hash=" << shadow.hash
           << ",live_profitability_weight=0"
           << ",live_profitability_score_contribution=0\n";
    output << "CAMPAIGN_PROFITABILITY_READINESS_GATE"
           << ",profitability_software_ready="
           << Boolean(gate.profitabilitySoftwareReady)
           << ",campaign_profitability_contract_ready="
           << Boolean(gate.campaignProfitabilityContractReady)
           << ",profitability_shadow_ranking_ready="
           << Boolean(gate.profitabilityShadowRankingReady)
           << ",campaign_profitability_activation_performed=false"
           << ",campaign_profitability_action="
           << ReadinessActionText(gate.action)
           << ",live_ranking_changed=false,read_only=true\n";

    std::vector<EvidenceResult> evidence;
    evidence.reserve(shadow.candidates.size());
    for (const auto& candidate : shadow.candidates)
        evidence.push_back(candidate.profitability);
    return ExitCode(evidence);
}

int RunCampaignShadowRankingCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    const std::vector<double>& shadowWeights,
    std::ostream& output,
    std::ostream& errors)
{
    (void)errors;
    if (rankingSnapshotId <= 0 || shadowWeights.empty())
        throw std::invalid_argument("invalid_campaign_profitability_shadow_request");
    std::set<double> validated;
    std::vector<double> requestedWeights;
    requestedWeights.reserve(shadowWeights.size());
    for (double weight : shadowWeights)
    {
        if (!std::isfinite(weight) || weight < 0.0 ||
            weight > kMaximumPhase9ProfitabilityShadowWeight ||
            !validated.insert(weight == 0.0 ? 0.0 : weight).second)
            throw std::invalid_argument("invalid_campaign_profitability_shadow_weight");
        requestedWeights.push_back(weight == 0.0 ? 0.0 : weight);
    }
    std::sort(requestedWeights.begin(), requestedWeights.end());

    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const CampaignProfitabilityShadowSource source =
        LoadCampaignProfitabilityShadowSource(transaction, rankingSnapshotId);
    std::vector<double> weights{0.0};
    for (double weight : requestedWeights)
        if (weight != 0.0) weights.push_back(weight);

    std::vector<WeightedShadowRanking> rankings;
    rankings.reserve(weights.size());
    for (double weight : weights)
        rankings.push_back(BuildWeightedShadowRanking(
            source.candidates, source.controlSnapshotId,
            source.sourceEvaluationRunId,
            source.controlSnapshotIdentityHash, weight));

    output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_SAFETY"
           << ",shadow_only=true,control_snapshot_id="
           << source.controlSnapshotId
           << ",source_evaluation_run_id=" << source.sourceEvaluationRunId
           << ",source_member_count=" << source.persistedMemberCount
           << ",live_rank_authoritative=true,live_profitability_weight=0"
           << ",live_profitability_score_contribution=0,activation=false"
           << ",database_write=false,experiment_created=false"
           << ",experiment_queued=false,scheduler_modified=false\n";

    for (const auto& ranking : rankings)
    {
        output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_POLICY"
               << ",control_snapshot_id=" << ranking.controlSnapshotId
               << ",source_evaluation_run_id=" << ranking.sourceEvaluationRunId
               << ",shadow_weight=" << Number(ranking.shadowWeight)
               << ",phase9_upper_bound="
               << Number(kMaximumPhase9ProfitabilityShadowWeight)
               << ",policy_version=" << ranking.policyVersion
               << ",policy_hash=" << ranking.policyHash
               << ",normalization_policy_hash="
               << ranking.normalization.policyHash
               << ",normalization_membership_hash="
               << ranking.normalization.membershipHash
               << ",shadow_ranking_hash=" << ranking.hash
               << ",activation=disabled,database_write=false\n";

        MovementSummary total;
        std::map<std::string, MovementSummary> byState;
        std::map<std::string, MovementSummary> byDirection;
        long long absoluteMovement = 0;
        int maximumUp = 0;
        int maximumDown = 0;
        std::set<long long> maximumUpIds;
        std::set<long long> maximumDownIds;
        for (const auto& candidate : ranking.candidates)
        {
            const auto& evidence = candidate.source.profitability;
            const auto& observation = evidence.observation;
            total.Add(candidate.rankDelta);
            byState[EvidenceStateText(evidence.state)].Add(candidate.rankDelta);
            byDirection[ProfitabilitySignText(
                candidate.source.profitabilitySign)].Add(candidate.rankDelta);
            absoluteMovement += std::abs(candidate.rankDelta);
            if (candidate.rankDelta > maximumUp)
            {
                maximumUp = candidate.rankDelta;
                maximumUpIds = {candidate.source.recommendationId};
            }
            else if (candidate.rankDelta == maximumUp && maximumUp > 0)
                maximumUpIds.insert(candidate.source.recommendationId);
            if (candidate.rankDelta < maximumDown)
            {
                maximumDown = candidate.rankDelta;
                maximumDownIds = {candidate.source.recommendationId};
            }
            else if (candidate.rankDelta == maximumDown && maximumDown < 0)
                maximumDownIds.insert(candidate.source.recommendationId);

            output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_MEMBER"
                   << ",control_snapshot_id=" << ranking.controlSnapshotId
                   << ",source_evaluation_run_id="
                   << ranking.sourceEvaluationRunId
                   << ",shadow_weight=" << Number(ranking.shadowWeight)
                   << ",control_rank=" << candidate.source.currentRank
                   << ",shadow_rank=" << candidate.shadowRank
                   << ",rank_delta=" << candidate.rankDelta
                   << ",recommendation_id="
                   << candidate.source.recommendationId
                   << ",recommendation_evaluation_result_id="
                   << candidate.source.recommendationEvaluationResultId
                   << ",ranking_member_id="
                   << candidate.source.rankingMemberId
                   << ",source_experiment_id="
                   << candidate.source.sourceExperimentId
                   << ",symbol=" << candidate.source.symbol
                   << ",horizon=" << candidate.source.horizon
                   << ",profitability_state="
                   << EvidenceStateText(evidence.state)
                   << ",profitability_observation_id="
                   << (observation ? std::to_string(observation->observationId)
                                   : "NULL")
                   << ",unavailable_reason="
                   << (observation ? "NULL" : MachineText(evidence.reason))
                   << ",profitability_direction="
                   << ProfitabilitySignText(candidate.source.profitabilitySign)
                   << ",actionable_count="
                   << (observation ? std::to_string(
                           observation->statistics.actionableCount) : "NULL")
                   << ",aggregate_return="
                   << (observation ? Number(observation->statistics
                           .aggregateTerminalHorizonLogReturnSum) : "NULL")
                   << ",average_return="
                   << (observation ? OptionalNumber(observation
                           ->averageTerminalHorizonLogReturnPerActionablePrediction)
                                   : "NULL")
                   << ",normalization_state="
                   << candidate.normalizationState
                   << ",normalization_reason="
                   << MachineText(candidate.normalizationReason)
                   << ",empirical_midrank_percentile="
                   << OptionalNumber(candidate.empiricalMidrankPercentile)
                   << ",bounded_candidate_metric="
                   << OptionalNumber(candidate.boundedCandidateMetric)
                   << ",support_reliability="
                   << OptionalNumber(candidate.supportReliability)
                   << ",normalized_profitability_value="
                   << OptionalNumber(candidate.normalizedProfitabilityValue)
                   << ",control_final_score="
                   << OptionalNumber(candidate.source.currentScore)
                   << ",profitability_contribution="
                   << OptionalNumber(candidate.profitabilityContribution)
                   << ",shadow_final_score="
                   << Number(candidate.shadowFinalScore)
                   << ",candidate_hash=" << candidate.hash << '\n';
        }
        output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_SUMMARY"
               << ",control_snapshot_id=" << ranking.controlSnapshotId
               << ",shadow_weight=" << Number(ranking.shadowWeight)
               << ",total_members=" << total.count
               << ",moved_up=" << total.movedUp
               << ",unchanged=" << total.unchanged
               << ",moved_down=" << total.movedDown
               << ",mean_absolute_rank_movement="
               << Number(static_cast<double>(absoluteMovement) / total.count)
               << ",maximum_upward_movement=" << maximumUp
               << ",maximum_upward_recommendation_ids="
               << IdList(maximumUpIds)
               << ",maximum_downward_movement=" << -maximumDown
               << ",maximum_downward_recommendation_ids="
               << IdList(maximumDownIds) << '\n';
        for (const auto& [state, summary] : byState)
            RenderMovement(output, ranking.controlSnapshotId,
                           ranking.shadowWeight, "profitability_state", state,
                           summary);
        for (const auto& [direction, summary] : byDirection)
            RenderMovement(output, ranking.controlSnapshotId,
                           ranking.shadowWeight, "profitability_direction",
                           direction, summary);

        for (const int n : {5, 10, 20})
        {
            std::set<long long> control;
            std::set<long long> shadow;
            int controlPositive = 0;
            int shadowPositive = 0;
            for (const auto& candidate : ranking.candidates)
            {
                if (candidate.source.currentRank <= n)
                {
                    control.insert(candidate.source.recommendationId);
                    if (candidate.source.profitabilitySign ==
                        ProfitabilitySign::positive) ++controlPositive;
                }
                if (candidate.shadowRank <= n)
                {
                    shadow.insert(candidate.source.recommendationId);
                    if (candidate.source.profitabilitySign ==
                        ProfitabilitySign::positive) ++shadowPositive;
                }
            }
            std::set<long long> entered;
            std::set<long long> exited;
            std::set_difference(shadow.begin(), shadow.end(),
                                control.begin(), control.end(),
                                std::inserter(entered, entered.end()));
            std::set_difference(control.begin(), control.end(),
                                shadow.begin(), shadow.end(),
                                std::inserter(exited, exited.end()));
            const std::size_t available = std::min<std::size_t>(
                static_cast<std::size_t>(n), ranking.candidates.size());
            output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_TOP_N"
                   << ",control_snapshot_id=" << ranking.controlSnapshotId
                   << ",shadow_weight=" << Number(ranking.shadowWeight)
                   << ",top_n=" << n
                   << ",retained=" << available - entered.size()
                   << ",entered=" << entered.size()
                   << ",exited=" << exited.size()
                   << ",entrant_ids=" << IdList(entered)
                   << ",exiting_ids=" << IdList(exited)
                   << ",control_positive_count=" << controlPositive
                   << ",shadow_positive_count=" << shadowPositive << '\n';
        }

        std::map<double, std::vector<const WeightedShadowCandidate*>> ties;
        for (const auto& candidate : ranking.candidates)
            ties[*candidate.source.currentScore].push_back(&candidate);
        int tieGroups = 0;
        int tiedMembers = 0;
        int profitabilityBrokenPairs = 0;
        bool consistent = true;
        int fallbackPairs = 0;
        for (const auto& [score, group] : ties)
        {
            (void)score;
            if (group.size() < 2) continue;
            ++tieGroups;
            tiedMembers += static_cast<int>(group.size());
            for (std::size_t i = 0; i < group.size(); ++i)
                for (std::size_t j = i + 1; j < group.size(); ++j)
                {
                    const auto left = group[i]->profitabilityContribution;
                    const auto right = group[j]->profitabilityContribution;
                    if (left && right && *left != *right)
                    {
                        ++profitabilityBrokenPairs;
                        const bool higherRanksFirst = (*left > *right) ==
                            (group[i]->shadowRank < group[j]->shadowRank);
                        consistent = consistent && higherRanksFirst;
                    }
                    else ++fallbackPairs;
                }
        }
        output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_TIES"
               << ",control_snapshot_id=" << ranking.controlSnapshotId
               << ",shadow_weight=" << Number(ranking.shadowWeight)
               << ",control_score_tie_groups=" << tieGroups
               << ",control_score_tied_members=" << tiedMembers
               << ",ties_broken_by_profitability_pairs="
               << profitabilityBrokenPairs
               << ",profitability_breaks_directionally_consistent="
               << Boolean(consistent)
               << ",fallback_tie_pairs=" << fallbackPairs << '\n';
    }

    std::map<long long, std::vector<const WeightedShadowCandidate*>> sensitivity;
    for (const auto& ranking : rankings)
        for (const auto& candidate : ranking.candidates)
            sensitivity[candidate.source.recommendationId].push_back(&candidate);
    int contributionViolations = 0;
    int rankNonmonotonic = 0;
    for (const auto& [recommendationId, values] : sensitivity)
    {
        bool contributionMonotonic = true;
        bool rankMovementMonotonic = true;
        for (std::size_t index = 1; index < values.size(); ++index)
        {
            const auto previous = values[index - 1]->profitabilityContribution;
            const auto current = values[index]->profitabilityContribution;
            if (previous && current)
            {
                if (values[index]->source.profitabilitySign ==
                        ProfitabilitySign::positive && *current < *previous)
                    contributionMonotonic = false;
                if (values[index]->source.profitabilitySign ==
                        ProfitabilitySign::negative && *current > *previous)
                    contributionMonotonic = false;
            }
            if (values[index]->source.profitabilitySign ==
                    ProfitabilitySign::positive &&
                values[index]->rankDelta < values[index - 1]->rankDelta)
                rankMovementMonotonic = false;
            if (values[index]->source.profitabilitySign ==
                    ProfitabilitySign::negative &&
                values[index]->rankDelta > values[index - 1]->rankDelta)
                rankMovementMonotonic = false;
        }
        if (!contributionMonotonic) ++contributionViolations;
        if (!rankMovementMonotonic) ++rankNonmonotonic;
        output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_SENSITIVITY_MEMBER"
               << ",recommendation_id=" << recommendationId
               << ",profitability_direction="
               << ProfitabilitySignText(values.front()->source.profitabilitySign)
               << ",contribution_monotonic="
               << Boolean(contributionMonotonic)
               << ",rank_movement_monotonic="
               << Boolean(rankMovementMonotonic);
        for (std::size_t index = 0; index < values.size(); ++index)
            output << ",weight_" << index << "=" << Number(weights[index])
                   << ",contribution_" << index << "="
                   << OptionalNumber(values[index]->profitabilityContribution)
                   << ",rank_delta_" << index << "="
                   << values[index]->rankDelta;
        output << '\n';
    }
    output << "CAMPAIGN_PROFITABILITY_WEIGHTED_SHADOW_SENSITIVITY_SUMMARY"
           << ",control_snapshot_id=" << source.controlSnapshotId
           << ",member_count=" << sensitivity.size()
           << ",contribution_monotonicity_violations="
           << contributionViolations
           << ",rank_movement_nonmonotonic_members=" << rankNonmonotonic
           << ",rank_nonmonotonicity_may_reflect_competing_movements=true"
           << ",shadow_only=true,activation=false,database_write=false\n";
    return 0;
}

int RunCampaignProfitabilityCalibrationCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    std::ostream& output,
    std::ostream& errors)
{
    (void)errors;
    if (rankingSnapshotId <= 0)
        throw std::invalid_argument(
            "invalid_campaign_profitability_calibration_snapshot_id");

    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const CampaignProfitabilityCoverageAudit coverage =
        LoadCampaignProfitabilityCoverageAudit(transaction, rankingSnapshotId);
    const CampaignProfitabilityShadowSource source =
        LoadCampaignProfitabilityShadowSource(transaction, rankingSnapshotId);
    const std::vector<double> weights = Phase10ProfitabilityCalibrationWeights();
    std::vector<WeightedShadowRanking> rankings;
    rankings.reserve(weights.size());
    for (const double weight : weights)
        rankings.push_back(BuildWeightedShadowRanking(
            source.candidates, source.controlSnapshotId,
            source.sourceEvaluationRunId, source.controlSnapshotIdentityHash,
            weight));
    const ProfitabilityCalibrationReport report =
        BuildProfitabilityCalibrationReport(rankings);
    if (rankings.front().candidates.size() != source.candidates.size())
        throw std::runtime_error("calibration_zero_weight_population_mismatch");
    for (const auto& candidate : rankings.front().candidates)
        if (candidate.shadowRank != candidate.source.currentRank ||
            candidate.rankDelta != 0)
            throw std::runtime_error(
                "calibration_zero_weight_control_reproduction_failed");

    const auto count = [&](const std::string& reason) {
        const auto found = coverage.reasonCounts.find(reason);
        return found == coverage.reasonCounts.end() ? std::size_t{0}
                                                    : found->second;
    };
    const std::size_t valid = count("valid_profitability_observation");
    std::size_t otherUnavailable = 0;
    for (const auto& [reason, reasonCount] : coverage.reasonCounts)
        if (reason != "valid_profitability_observation" &&
            reason != "no_profitability_observation" &&
            reason != "final_inference_context_mismatch" &&
            reason != "no_exact_final_inference_result")
            otherUnavailable += reasonCount;
    const auto recoveryCount = [&](const std::string& recoveryClass) {
        const auto found = coverage.recoveryClassCounts.find(recoveryClass);
        return found == coverage.recoveryClassCounts.end()
            ? std::size_t{0} : found->second;
    };
    const double overallCoverage = coverage.members.empty() ? 0.0
        : 100.0 * valid / static_cast<double>(coverage.members.size());

    output << "CAMPAIGN_PROFITABILITY_CALIBRATION_START"
           << ",control_snapshot_id=" << report.controlSnapshotId
           << ",source_evaluation_run_id=" << report.sourceEvaluationRunId
           << ",population_size=" << source.persistedMemberCount
           << ",grid_point_count=" << weights.size()
           << ",report_hash=" << report.hash
           << ",advisory=true,shadow_only=true,read_only=true"
           << ",activation=false,production_ranking_modified=false"
           << ",database_write=false,experiment_created=false"
           << ",experiment_queued=false,scheduler_modified=false\n";
    output << "CAMPAIGN_PROFITABILITY_COVERAGE_SUMMARY"
           << ",control_snapshot_id=" << coverage.controlSnapshotId
           << ",total_members=" << coverage.members.size()
           << ",valid_profitability_observation=" << valid
           << ",no_profitability_observation="
           << count("no_profitability_observation")
           << ",final_inference_context_mismatch="
           << count("final_inference_context_mismatch")
           << ",no_exact_final_inference_result="
           << count("no_exact_final_inference_result")
           << ",other_unavailable=" << otherUnavailable
           << ",invalid_incomplete_provenance="
           << recoveryCount("invalid_incomplete_provenance")
           << ",overall_valid_coverage_percentage=" << Number(overallCoverage)
           << ",before_valid_count=" << valid
           << ",after_valid_count=" << valid
           << ",backfill_performed=false,coverage_audit_hash="
           << coverage.hash << ",read_only=true\n";
    for (const auto& [reason, reasonCount] : coverage.reasonCounts)
        output << "CAMPAIGN_PROFITABILITY_COVERAGE_REASON"
               << ",control_snapshot_id=" << coverage.controlSnapshotId
               << ",reason=" << MachineText(reason)
               << ",member_count=" << reasonCount << '\n';
    for (const auto& member : coverage.members)
    {
        const auto& frozen = member.frozenEvidence;
        const std::string reason = member.validatedEvidence.state ==
                EvidenceState::valid
            ? "valid_profitability_observation"
            : member.validatedEvidence.reason;
        output << "CAMPAIGN_PROFITABILITY_COVERAGE_MEMBER"
               << ",control_snapshot_id=" << coverage.controlSnapshotId
               << ",control_rank=" << member.controlRank
               << ",recommendation_id=" << member.recommendationId
               << ",recommendation_evaluation_result_id="
               << member.recommendationEvaluationResultId
               << ",ranking_member_id=" << member.rankingMemberId
               << ",source_experiment_id=" << member.sourceExperimentId
               << ",source_model_id=" << OptionalId(member.sourceModelId)
               << ",symbol=" << MachineText(member.symbol)
               << ",horizon=" << member.horizon
               << ",frozen_provenance_version=" << frozen.provenanceVersion
               << ",frozen_final_inference_result_id="
               << OptionalId(frozen.finalInferenceEvalResultId)
               << ",frozen_profitability_observation_id="
               << OptionalId(frozen.profitabilityObservationId)
               << ",frozen_inference_scope="
               << MachineText(frozen.inferenceScope)
               << ",frozen_inference_start="
               << (frozen.inferenceStart ? MachineText(*frozen.inferenceStart)
                                         : "NULL")
               << ",frozen_inference_end="
               << (frozen.inferenceEnd ? MachineText(*frozen.inferenceEnd)
                                       : "NULL")
               << ",frozen_metric_definition_hash="
               << (frozen.metricDefinitionHash
                       ? *frozen.metricDefinitionHash : "NULL")
               << ",frozen_source_content_hash="
               << (frozen.sourceContentHash ? *frozen.sourceContentHash : "NULL")
               << ",frozen_observation_identity_hash="
               << (frozen.observationIdentityHash
                       ? *frozen.observationIdentityHash : "NULL")
               << ",unavailable_reason=" << MachineText(reason)
               << ",recovery_class="
               << CoverageRecoveryClassText(member.recoveryClass)
               << ",exact_final_inference_result_exists="
               << Boolean(member.exactFinalInferenceResultExists)
               << ",any_final_inference_result_exists="
               << Boolean(member.anyFinalInferenceResultExists)
               << ",exact_final_profitability_observation_exists="
               << Boolean(member.exactFinalProfitabilityObservationExists)
               << ",any_inference_profitability_observation_exists="
               << Boolean(member.anyInferenceProfitabilityObservationExists)
               << ",frozen_snapshot_backfill_permitted=false"
               << ",reconstruction_assessment="
               << MachineText(member.reconstructionAssessment)
               << ",member_hash=" << member.hash << '\n';
    }

    const auto renderMovement = [&](const char* prefix,
                                    const CalibrationMovementStatistics& value) {
        output << ',' << prefix << "_count=" << value.count
               << ',' << prefix << "_moved_up=" << value.movedUp
               << ',' << prefix << "_unchanged=" << value.unchanged
               << ',' << prefix << "_moved_down=" << value.movedDown
               << ',' << prefix << "_mean_rank_delta="
               << Number(value.meanRankDelta);
    };
    for (const auto& point : report.weights)
    {
        const bool anchor = SameWeight(point.weight, 0.01) ||
            SameWeight(point.weight, 0.025) || SameWeight(point.weight, 0.05);
        output << "CAMPAIGN_PROFITABILITY_CALIBRATION_SWEEP_POINT"
               << ",control_snapshot_id=" << report.controlSnapshotId
               << ",weight=" << Number(point.weight)
               << ",mandatory_anchor=" << Boolean(anchor)
               << ",total_members=" << point.totalMembers
               << ",valid_profitability_members="
               << point.validProfitabilityMembers
               << ",unavailable_members=" << point.unavailableMembers
               << ",positive_profitability_members="
               << point.positiveProfitabilityMembers
               << ",negative_profitability_members="
               << point.negativeProfitabilityMembers
               << ",zero_profitability_members="
               << point.zeroProfitabilityMembers
               << ",moved_up=" << point.totalMovement.movedUp
               << ",unchanged=" << point.totalMovement.unchanged
               << ",moved_down=" << point.totalMovement.movedDown
               << ",mean_absolute_rank_movement="
               << Number(point.totalMovement.meanAbsoluteRankMovement)
               << ",median_absolute_rank_movement="
               << Number(point.totalMovement.medianAbsoluteRankMovement)
               << ",p90_absolute_rank_movement="
               << Number(point.totalMovement.p90AbsoluteRankMovement)
               << ",maximum_upward_movement="
               << point.totalMovement.maximumUpwardMovement
               << ",maximum_downward_movement="
               << point.totalMovement.maximumDownwardMovement;
        renderMovement("positive", point.positiveMovement);
        renderMovement("negative", point.negativeMovement);
        renderMovement("unavailable", point.unavailableMovement);
        output << ",ranking_hash=" << point.rankingHash
               << ",point_hash=" << point.hash
               << ",activation=false,production_ranking_modified=false\n";
        if (anchor)
            output << "CAMPAIGN_PROFITABILITY_CALIBRATION_WEIGHT"
                   << ",control_snapshot_id=" << report.controlSnapshotId
                   << ",weight=" << Number(point.weight)
                   << ",point_hash=" << point.hash
                   << ",direct_anchor_comparison=true,activation=false\n";
        for (const auto& top : point.topN)
            output << "CAMPAIGN_PROFITABILITY_CALIBRATION_TOP_N"
                   << ",control_snapshot_id=" << report.controlSnapshotId
                   << ",weight=" << Number(point.weight)
                   << ",top_n=" << top.n
                   << ",retained=" << top.retained
                   << ",entered=" << top.entered
                   << ",exited=" << top.exited
                   << ",entrant_ids=" << IdList(top.entrantRecommendationIds)
                   << ",exiting_ids=" << IdList(top.exitingRecommendationIds)
                   << ",member_ids=" << IdList(top.memberRecommendationIds)
                   << ",positive_count=" << top.positiveCount
                   << ",negative_count=" << top.negativeCount
                   << ",zero_count=" << top.zeroCount
                   << ",unavailable_count=" << top.unavailableCount
                   << ",valid_evidence_coverage_count="
                   << top.validEvidenceCoverageCount
                   << ",valid_evidence_coverage_percentage="
                   << Number(top.validEvidenceCoveragePercentage) << '\n';
    }

    for (const auto& pair : report.anchorPairwise)
        output << "CAMPAIGN_PROFITABILITY_CALIBRATION_PAIRWISE"
               << ",control_snapshot_id=" << report.controlSnapshotId
               << ",left_weight=" << Number(pair.leftWeight)
               << ",right_weight=" << Number(pair.rightWeight)
               << ",top_5_overlap=" << pair.topNOverlap.at(5)
               << ",top_5_difference_recommendation_ids="
               << IdList(pair.topNDifferenceRecommendationIds.at(5))
               << ",top_10_overlap=" << pair.topNOverlap.at(10)
               << ",top_10_difference_recommendation_ids="
               << IdList(pair.topNDifferenceRecommendationIds.at(10))
               << ",top_20_overlap=" << pair.topNOverlap.at(20)
               << ",top_20_difference_recommendation_ids="
               << IdList(pair.topNDifferenceRecommendationIds.at(20))
               << ",ordinal_changes=" << pair.ordinalChanges
               << ",mean_absolute_ordinal_difference="
               << Number(pair.meanAbsoluteOrdinalDifference)
               << ",largest_ordinal_difference="
               << pair.largestOrdinalDifference
               << ",largest_ordinal_difference_recommendation_ids="
               << IdList(pair.largestOrdinalDifferenceRecommendationIds)
               << ",pairwise_hash=" << pair.hash << '\n';

    std::map<long long, std::pair<int, int>> rankRanges;
    for (const auto& ranking : rankings)
        for (const auto& candidate : ranking.candidates)
        {
            auto [found, inserted] = rankRanges.emplace(
                candidate.source.recommendationId,
                std::pair<int, int>{candidate.shadowRank, candidate.shadowRank});
            if (!inserted)
            {
                found->second.first =
                    std::min(found->second.first, candidate.shadowRank);
                found->second.second =
                    std::max(found->second.second, candidate.shadowRank);
            }
        }
    for (const auto& [recommendationId, range] : rankRanges)
        if (range.second - range.first >= 5)
            output << "CAMPAIGN_PROFITABILITY_CALIBRATION_SENSITIVITY"
                   << ",recommendation_id=" << recommendationId
                   << ",best_rank=" << range.first
                   << ",worst_rank=" << range.second
                   << ",rank_span=" << range.second - range.first
                   << ",material_rank_span_threshold=5\n";

    output << "CAMPAIGN_PROFITABILITY_CALIBRATION_RESPONSE_CURVE"
           << ",control_snapshot_id=" << report.controlSnapshotId
           << ",first_best_top_5_weight="
           << OptionalWeight(report.responseCurve.firstBestTop5Weight)
           << ",first_top_10_at_least_9_weight="
           << OptionalWeight(report.responseCurve.firstTop10AtLeastNineWeight)
           << ",first_best_top_10_weight="
           << OptionalWeight(report.responseCurve.firstBestTop10Weight)
           << ",first_top_20_improvement_weight="
           << OptionalWeight(report.responseCurve.firstTop20ImprovementWeight)
           << ",membership_discontinuity_weights=";
    for (std::size_t index = 0;
         index < report.responseCurve.membershipDiscontinuityWeights.size();
         ++index)
    {
        if (index > 0) output << ':';
        output << Number(
            report.responseCurve.membershipDiscontinuityWeights[index]);
    }
    output << ",response_curve_hash=" << report.responseCurve.hash << '\n';
    for (const auto& region : report.responseCurve.stabilityRegions)
        output << "CAMPAIGN_PROFITABILITY_CALIBRATION_STABILITY_REGION"
               << ",top_n=" << region.topN
               << ",first_weight=" << Number(region.firstWeight)
               << ",last_weight=" << Number(region.lastWeight)
               << ",grid_point_membership_stable=true"
               << ",member_ids=" << IdList(region.memberRecommendationIds)
               << '\n';
    output << "CAMPAIGN_PROFITABILITY_CALIBRATION_MINIMUM_EFFECTIVE_WEIGHT"
           << ",minimum_effective_weight="
           << OptionalWeight(report.responseCurve.minimumEffectiveWeight)
           << ",minimum_effective_region_end="
           << OptionalWeight(report.responseCurve.minimumEffectiveRegionEnd)
           << ",weight_0_025_inside_region="
           << Boolean(report.responseCurve.provisional0025InsideMinimumEffectiveRegion)
           << ",policy=smallest_weight_with_best_top5_top10_within_one_of_best_"
              "top20_improved_no_positive_down_no_negative_up_and_less_churn_"
              "than_0.05"
           << ",assessment=" << report.responseCurve.assessment
           << ",advisory_only=true,activation=false\n";
    output << "CAMPAIGN_PROFITABILITY_CALIBRATION_ASSESSMENT"
           << ",classification=provisional_in_sample_shadow_calibration"
           << ",minimum_effective_weight="
           << OptionalWeight(report.responseCurve.minimumEffectiveWeight)
           << ",production_decision=false"
           << ",temporal_out_of_sample_validation_required=true"
           << ",future_realized_profitability_not_proven=true"
           << ",activation=false,production_ranking_modified=false\n";

    const ProfitabilityCalibrationWeight* readinessPoint = &report.weights.front();
    if (report.responseCurve.minimumEffectiveWeight)
        for (const auto& point : report.weights)
            if (SameWeight(point.weight,
                           *report.responseCurve.minimumEffectiveWeight))
                readinessPoint = &point;
    const auto topAt = [&](int n) -> const CalibrationTopN& {
        const auto found = std::find_if(
            readinessPoint->topN.begin(), readinessPoint->topN.end(),
            [n](const auto& top) { return top.n == n; });
        if (found == readinessPoint->topN.end())
            throw std::logic_error("calibration_readiness_top_n_missing");
        return *found;
    };
    output << "CAMPAIGN_PROFITABILITY_PRODUCTION_READINESS"
           << ",control_snapshot_id=" << report.controlSnapshotId
           << ",assessment_weight=" << Number(readinessPoint->weight)
           << ",overall_valid_coverage_percentage=" << Number(overallCoverage)
           << ",top_5_valid_coverage_percentage="
           << Number(topAt(5).validEvidenceCoveragePercentage)
           << ",top_10_valid_coverage_percentage="
           << Number(topAt(10).validEvidenceCoveragePercentage)
           << ",top_20_valid_coverage_percentage="
           << Number(topAt(20).validEvidenceCoveragePercentage)
           << ",repository_coverage_policy_defined=false"
           << ",future_human_approved_coverage_policy_required=true"
           << ",activation_ready=false"
           << ",blockers=low_overall_coverage:missing_coverage_policy:"
              "temporal_out_of_sample_validation_required"
           << ",activation=false,production_ranking_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,database_write=false\n";
    return 0;
}

int RunCampaignProfitabilityTemporalValidationCommand(
    const std::string& connectionString,
    std::ostream& output,
    std::ostream& errors)
{
    (void)errors;
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const CampaignProfitabilityTemporalFeasibilityAudit audit =
        LoadCampaignProfitabilityTemporalFeasibilityAudit(transaction);
    const auto classificationCount = [&](const std::string& value) {
        const auto found = audit.classificationCounts.find(value);
        return found == audit.classificationCounts.end()
            ? std::size_t{0} : found->second;
    };
    const std::size_t admissible = classificationCount(
        "admissible_temporal_holdout");
    output << "CAMPAIGN_PROFITABILITY_TEMPORAL_VALIDATION_START"
           << ",protocol_version=1,precommitted_candidate_weight=0.025"
           << ",control_weight=0,sensitivity_weights=0.0225:0.0275"
           << ",sensitivity_non_selective=true,phase10_sweep_repeated=false"
           << ",cohort_count=" << audit.cohorts.size()
           << ",audit_hash=" << audit.hash
           << ",read_only=true,activation=false,live_profitability_weight=0"
           << ",production_ranking_modified=false"
           << ",recommendation_modified=false"
           << ",ranking_snapshot_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,worker_modified=false\n";
    for (const auto& cohort : audit.cohorts)
    {
        const bool accepted = cohort.classification ==
            TemporalCohortClassification::admissibleTemporalHoldout;
        output << (accepted
                ? "CAMPAIGN_PROFITABILITY_TEMPORAL_COHORT"
                : "CAMPAIGN_PROFITABILITY_TEMPORAL_COHORT_REJECTED")
               << ",ranking_snapshot_id=" << cohort.rankingSnapshotId
               << ",source_evaluation_run_id="
               << cohort.sourceEvaluationRunId
               << ",as_of_timestamp=" << cohort.asOfTimestamp
               << ",total_candidate_count=" << cohort.totalCandidateCount
               << ",valid_ranking_time_profitability_evidence_count="
               << cohort.validRankingTimeProfitabilityEvidenceCount
               << ",unavailable_ranking_time_evidence_count="
               << cohort.unavailableRankingTimeEvidenceCount
               << ",legitimate_subsequent_outcome_count="
               << cohort.legitimateSubsequentOutcomeCount
               << ",ranking_input_start="
               << (cohort.rankingInputStart
                       ? *cohort.rankingInputStart : "NULL")
               << ",ranking_input_end="
               << (cohort.rankingInputEnd ? *cohort.rankingInputEnd : "NULL")
               << ",outcome_start="
               << (cohort.outcomeStart ? *cohort.outcomeStart : "NULL")
               << ",outcome_end="
               << (cohort.outcomeEnd ? *cohort.outcomeEnd : "NULL")
               << ",point_in_time_provenance_violation_count="
               << cohort.pointInTimeProvenanceViolationCount
               << ",overlapping_input_and_outcome_count="
               << cohort.overlappingInputAndOutcomeCount
               << ",future_information_leakage_count="
               << cohort.futureInformationLeakageCount
               << ",context_or_identity_mismatch_count="
               << cohort.contextOrIdentityMismatchCount
               << ",ranking_population_reconstructable="
               << Boolean(cohort.rankingPopulationReconstructable)
               << ",zero_weight_control_reconstructed_exactly="
               << Boolean(cohort.exactControlReconstruction)
               << ",classification="
               << TemporalCohortClassificationText(cohort.classification)
               << ",reason=" << MachineText(cohort.reason)
               << ",leakage_check="
               << (cohort.futureInformationLeakageCount == 0 &&
                           cohort.pointInTimeProvenanceViolationCount == 0
                       ? "pass" : "fail_closed")
               << ",cohort_hash=" << cohort.hash << '\n';
        for (const auto& [reason, count] :
             cohort.rankingTimeUnavailableReasonCounts)
        {
            if (reason == "valid_profitability_observation") continue;
            output << "CAMPAIGN_PROFITABILITY_TEMPORAL_COVERAGE"
                   << ",ranking_snapshot_id=" << cohort.rankingSnapshotId
                   << ",unavailable_reason=" << MachineText(reason)
                   << ",member_count=" << count
                   << ",exact_final_only=true,checkpoint_substitution=false\n";
        }
    }
    const std::string assessment = admissible == 0
        ? "HISTORICAL_HOLDOUT_UNAVAILABLE_FORWARD_VALIDATION_REQUIRED"
        : "TEMPORAL_VALIDATION_INCONCLUSIVE";
    output << "CAMPAIGN_PROFITABILITY_TEMPORAL_AGGREGATE"
           << ",admissible_cohort_count=" << admissible
           << ",rejected_cohort_count=" << audit.cohorts.size() - admissible
           << ",insufficient_ranking_time_provenance_count="
           << classificationCount("insufficient_ranking_time_provenance")
           << ",insufficient_subsequent_outcome_count="
           << classificationCount("insufficient_subsequent_outcome")
           << ",overlapping_input_and_outcome_period_count="
           << classificationCount("overlapping_input_and_outcome_period")
           << ",future_information_leakage_count="
           << classificationCount("future_information_leakage")
           << ",context_or_identity_mismatch_count="
           << classificationCount("context_or_identity_mismatch")
           << ",other_fail_closed_count="
           << classificationCount("other_fail_closed")
           << ",top_n_outcome_comparison_emitted=false"
           << ",reason=no_admissible_temporal_holdout\n";
    output << "CAMPAIGN_PROFITABILITY_PHASE11_ASSESSMENT"
           << ",assessment=" << assessment
           << ",candidate_weight=0.025,weight_selected_from_holdout=false"
           << ",historical_top_5_result=unavailable"
           << ",historical_top_10_result=unavailable"
           << ",historical_top_20_result=unavailable"
           << ",production_activation_authority=false"
           << ",forward_validation_required=" << Boolean(admissible == 0)
           << ",activation=false,live_profitability_weight=0"
           << ",production_ranking_modified=false"
           << ",recommendation_modified=false"
           << ",ranking_snapshot_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,worker_modified=false\n";
    output << "CAMPAIGN_PROFITABILITY_PRODUCTION_READINESS"
           << ",phase=11,activation_ready=false"
           << ",repository_coverage_policy_defined=false"
           << ",blockers=no_admissible_temporal_holdout:"
              "forward_validation_not_completed:coverage_policy_not_approved"
           << ",activation=false,live_profitability_weight=0"
           << ",production_ranking_modified=false"
           << ",recommendation_modified=false"
           << ",ranking_snapshot_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,worker_modified=false"
           << ",database_write=false\n";
    return 0;
}

int RunCampaignProfitabilityForwardValidationPrecommitCommand(
    const std::string& connectionString,
    long long rankingSnapshotId,
    const std::string& expectedOutcomeStart,
    const std::string& expectedOutcomeEnd,
    std::ostream& output,
    std::ostream& errors)
{
    (void)errors;
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const CampaignProfitabilityTemporalCohort cohort =
        LoadCampaignProfitabilityTemporalCohort(transaction, rankingSnapshotId);
    if (!cohort.rankingPopulationReconstructable ||
        !cohort.exactControlReconstruction ||
        cohort.pointInTimeProvenanceViolationCount != 0)
        throw std::runtime_error(
            "forward_validation_source_snapshot_not_point_in_time_reconstructable");
    const CampaignProfitabilityShadowSource source =
        LoadCampaignProfitabilityShadowSource(transaction, rankingSnapshotId);
    const auto precommit = BuildCampaignProfitabilityForwardValidationPrecommit(
        source, cohort.asOfTimestamp, expectedOutcomeStart, expectedOutcomeEnd);

    output << "CAMPAIGN_PROFITABILITY_FORWARD_VALIDATION_PRECOMMIT"
           << ",protocol_version=" << precommit.protocolVersion
           << ",validation_cohort_identity_hash=" << precommit.hash
           << ",ranking_snapshot_id=" << precommit.rankingSnapshotId
           << ",source_evaluation_run_id="
           << precommit.sourceEvaluationRunId
           << ",decision_timestamp=" << precommit.decisionTimestamp
           << ",control_weight=0,precommitted_candidate_weight=0.025"
           << ",weight_selected_from_future_outcome=false"
           << ",expected_outcome_start=" << precommit.expectedOutcomeStart
           << ",expected_outcome_end=" << precommit.expectedOutcomeEnd
           << ",control_ranking_hash=" << precommit.controlRankingHash
           << ",candidate_ranking_hash=" << precommit.candidateRankingHash
           << ",member_count=" << precommit.members.size()
           << ",external_immutable_artifact_commit_required=true"
           << ",database_persistence=false,activation=false"
           << ",live_profitability_weight=0"
           << ",production_ranking_modified=false"
           << ",recommendation_modified=false"
           << ",ranking_snapshot_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,worker_modified=false\n";
    for (const auto& member : precommit.members)
    {
        const auto& observation = member.source.profitability.observation;
        output << "CAMPAIGN_PROFITABILITY_ASOF_MEMBER"
               << ",validation_cohort_identity_hash=" << precommit.hash
               << ",ranking_member_id=" << member.source.rankingMemberId
               << ",recommendation_id=" << member.source.recommendationId
               << ",evaluation_result_id="
               << member.source.recommendationEvaluationResultId
               << ",source_experiment_id="
               << member.source.sourceExperimentId
               << ",source_model_id=" << OptionalId(member.source.sourceModelId)
               << ",symbol=" << MachineText(member.source.symbol)
               << ",horizon=" << member.source.horizon
               << ",control_rank=" << member.controlRank
               << ",candidate_rank=" << member.candidateRank
               << ",rank_delta=" << member.rankDelta
               << ",ranking_time_profitability_state="
               << EvidenceStateText(member.source.profitability.state)
               << ",ranking_time_profitability_sign="
               << ProfitabilitySignText(member.source.profitabilitySign)
               << ",ranking_time_profitability_value="
               << (observation
                       ? OptionalNumber(observation->
                           averageTerminalHorizonLogReturnPerActionablePrediction)
                       : "NULL")
               << ",ranking_time_profitability_observation_id="
               << (observation ? std::to_string(observation->observationId)
                               : "NULL")
               << ",ranking_time_profitability_observation_identity_hash="
               << (member.rankingTimeProfitabilityObservationIdentityHash.empty()
                       ? "NULL"
                       : member.rankingTimeProfitabilityObservationIdentityHash)
               << ",evidence_available_at_selection_time="
               << Boolean(member.evidenceAvailableAtSelectionTime)
               << ",subsequent_outcome_identity=PENDING"
               << ",subsequent_outcome_start="
               << precommit.expectedOutcomeStart
               << ",subsequent_outcome_end=" << precommit.expectedOutcomeEnd
               << ",leakage_check=pending_future_strictly_subsequent_outcome"
               << ",member_hash=" << member.hash << '\n';
    }
    for (const auto& top : precommit.topN)
        output << "CAMPAIGN_PROFITABILITY_TEMPORAL_TOP_N"
               << ",validation_cohort_identity_hash=" << precommit.hash
               << ",top_n=" << top.n
               << ",control_selected_ids="
               << IdList(top.controlRecommendationIds)
               << ",candidate_selected_ids="
               << IdList(top.candidateRecommendationIds)
               << ",retained_ids=" << IdList(top.retainedRecommendationIds)
               << ",candidate_only_entrants="
               << IdList(top.candidateOnlyEntrants)
               << ",control_only_exits=" << IdList(top.controlOnlyExits)
               << ",outcome_status=pending,top_n_hash=" << top.hash << '\n';
    output << "CAMPAIGN_PROFITABILITY_PHASE11_ASSESSMENT"
           << ",assessment=HISTORICAL_HOLDOUT_UNAVAILABLE_FORWARD_VALIDATION_REQUIRED"
           << ",forward_precommit_prepared=true"
           << ",future_outcome_computation_launched=false"
           << ",production_activation_authority=false"
           << ",activation=false,live_profitability_weight=0"
           << ",production_ranking_modified=false"
           << ",recommendation_modified=false"
           << ",ranking_snapshot_modified=false"
           << ",experiment_created=false,experiment_queued=false"
           << ",scheduler_modified=false,worker_modified=false"
           << ",database_write=false\n";
    return 0;
}

int RunCampaignProfitabilityOutcomePreparationCommand(
    const std::string& connectionString,
    const std::string& validationCohortIdentityHash,
    std::ostream& output,
    std::ostream& errors,
    const std::string& artifactPath,
    const std::string& currentDateOverride)
{
    (void)errors;
    if (validationCohortIdentityHash !=
        kPhase12ValidationCohortIdentityHash)
        throw std::invalid_argument("phase12_validation_cohort_hash_mismatch");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const auto preparation = LoadCampaignProfitabilityOutcomePreparation(
        transaction,
        currentDateOverride.empty() ? CurrentUtcDate() : currentDateOverride,
        artifactPath);

    const std::string common =
        std::string{",validation_cohort_identity_hash="} +
        kPhase12ValidationCohortIdentityHash +
        ",ranking_snapshot_id=5,source_evaluation_run_id=6";
    output << "CAMPAIGN_PROFITABILITY_OUTCOME_SUMMARY" << common
           << ",protocol_version=1,artifact_path="
           << MachineText(preparation.artifactPath)
           << ",artifact_sha256=" << preparation.artifactSha256
           << ",artifact_identity_verified="
           << Boolean(preparation.artifactIdentityVerified)
           << ",unique_source_model_job_count=" << preparation.jobs.size()
           << ",recommendation_count=79,outcome_start="
           << kPhase12OutcomeStart << ",outcome_end=" << kPhase12OutcomeEnd
           << ",metric_hash=" << InferenceProfitability::MetricDefinitionHash()
           << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
           << ",preparation_hash=" << preparation.hash
           << ",future_outcome_execution_launched=false"
           << OutcomeSafetyFields() << '\n';

    std::size_t compatibleCount = 0;
    std::size_t waitingCount = 0;
    std::size_t readyCount = 0;
    for (const auto& job : preparation.jobs)
    {
        compatibleCount += job.compatible ? 1U : 0U;
        waitingCount += job.readiness ==
            OutcomeJobReadiness::waitingForOutcomeData ? 1U : 0U;
        readyCount += job.readiness == OutcomeJobReadiness::readyToExecute
            ? 1U : 0U;
        output << "CAMPAIGN_PROFITABILITY_OUTCOME_JOB" << common
               << ",source_experiment_id=" << job.sourceExperimentId
               << ",source_model_id=" << job.sourceModelId
               << ",symbol=" << MachineText(job.symbol)
               << ",horizon=" << job.horizon
               << ",original_train_start=" << job.originalTrainStart
               << ",original_train_end=" << job.originalTrainEnd
               << ",original_final_inference_start="
               << job.originalInferenceStart
               << ",original_final_inference_end=" << job.originalInferenceEnd
               << ",outcome_start=" << job.outcomeStart
               << ",outcome_end=" << job.outcomeEnd
               << ",threshold=" << Number(job.threshold)
               << ",label_rule_id=" << job.labelRuleId
               << ",target_type=" << job.targetType
               << ",window_size=" << job.windowSize
               << ",input_width=" << job.inputWidth
               << ",feature_semantic_hash=" << job.featureSemanticHash
               << ",model_lineage_hash=" << job.modelLineageHash
               << ",model_artifact_content_hash="
               << job.modelArtifactContentHash
               << ",model_parameter_row_count=" << job.modelParameterRowCount
               << ",recommendation_ids=" << IdList(job.recommendationIds)
               << ",recommendation_count=" << job.recommendationIds.size()
               << ",top_n_participation="
               << MachineText(job.topNParticipation)
               << ",model_artifact_available=" << Boolean(job.modelExists)
               << ",exact_model_experiment_link="
               << Boolean(job.exactModelExperimentLink)
               << ",exact_final_source_model="
               << Boolean(job.exactFinalSourceModel)
               << ",exact_original_final_inference="
               << Boolean(job.exactOriginalFinalInference)
               << ",checkpoint_substitution="
               << Boolean(job.checkpointSubstitution)
               << ",compatibility_state="
               << MachineText(job.compatibilityState)
               << ",metric_hash=" << job.metricDefinitionHash
               << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
               << ",job_hash=" << job.hash << OutcomeSafetyFields() << '\n';
        output << "CAMPAIGN_PROFITABILITY_OUTCOME_JOB_READINESS" << common
               << ",source_experiment_id=" << job.sourceExperimentId
               << ",source_model_id=" << job.sourceModelId
               << ",symbol=" << MachineText(job.symbol)
               << ",horizon=" << job.horizon
               << ",outcome_start=" << job.outcomeStart
               << ",outcome_end=" << job.outcomeEnd
               << ",metric_hash=" << job.metricDefinitionHash
               << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
               << ",job_hash=" << job.hash
               << ",readiness=" << OutcomeJobReadinessText(job.readiness)
               << ",current_date=" << preparation.currentDate
               << ",market_data_coverage_checked=false"
               << ",reason="
               << (job.readiness == OutcomeJobReadiness::waitingForOutcomeData
                       ? "outcome_end_has_not_occurred"
                       : MachineText(job.compatibilityState))
               << OutcomeSafetyFields() << '\n';
        output << "CAMPAIGN_PROFITABILITY_OUTCOME_EXECUTION_COMMAND" << common
               << ",source_experiment_id=" << job.sourceExperimentId
               << ",source_model_id=" << job.sourceModelId
               << ",symbol=" << MachineText(job.symbol)
               << ",horizon=" << job.horizon
               << ",outcome_start=" << job.outcomeStart
               << ",outcome_end=" << job.outcomeEnd
               << ",metric_hash=" << job.metricDefinitionHash
               << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
               << ",job_hash=" << job.hash
               << ",command=./LSTM_Release%20"
                  "--run-frozen-model-outcome-inference%3D"
               << job.validationCohortIdentityHash << "%2C"
               << job.sourceExperimentId << "%2C" << job.sourceModelId
               << "%2C" << job.outcomeStart << "%2C" << job.outcomeEnd
               << "%2C" << job.hash
               << OutcomeSafetyFields() << '\n';
    }

    for (const auto& top : preparation.topN)
    {
        std::map<long long, long long> recommendationModel;
        for (const auto& job : preparation.jobs)
            for (long long recommendationId : job.recommendationIds)
                recommendationModel[recommendationId] = job.sourceModelId;
        const auto modelList = [&](const std::vector<long long>& ids) {
            std::vector<long long> models;
            for (long long id : ids) models.push_back(recommendationModel.at(id));
            std::sort(models.begin(), models.end());
            models.erase(std::unique(models.begin(), models.end()), models.end());
            return IdList(models);
        };
        output << "CAMPAIGN_PROFITABILITY_OUTCOME_SELECTION_MAPPING" << common
               << ",source_experiment_id=ALL,source_model_id=ALL"
               << ",symbol=MULTIPLE,horizon=MULTIPLE"
               << ",outcome_start=" << kPhase12OutcomeStart
               << ",outcome_end=" << kPhase12OutcomeEnd
               << ",metric_hash=" << InferenceProfitability::MetricDefinitionHash()
               << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
               << ",top_n=" << top.n
               << ",retained_recommendation_ids="
               << IdList(top.retainedRecommendationIds)
               << ",candidate_entrant_recommendation_ids="
               << IdList(top.candidateOnlyEntrants)
               << ",control_exit_recommendation_ids="
               << IdList(top.controlOnlyExits)
               << ",retained_unique_source_model_ids="
               << modelList(top.retainedRecommendationIds)
               << ",entrant_unique_source_model_ids="
               << modelList(top.candidateOnlyEntrants)
               << ",exit_unique_source_model_ids="
               << modelList(top.controlOnlyExits)
               << ",duplicate_recommendations_are_independent=false"
               << ",phase13_contract=common_member_contribution_plus_entrant_"
                  "contribution_minus_exit_contribution"
               << ",candidate_minus_control_incremental_profitability=PENDING"
               << ",outcome_coverage=PENDING,top_n_hash=" << top.hash
               << OutcomeSafetyFields() << '\n';
    }
    output << "CAMPAIGN_PROFITABILITY_OUTCOME_SUMMARY" << common
           << ",source_experiment_id=ALL,source_model_id=ALL"
           << ",symbol=MULTIPLE,horizon=MULTIPLE,outcome_start="
           << kPhase12OutcomeStart << ",outcome_end=" << kPhase12OutcomeEnd
           << ",metric_hash=" << InferenceProfitability::MetricDefinitionHash()
           << ",source_content_hash=PENDING,outcome_identity_hash=PENDING"
           << ",unique_job_count=" << preparation.jobs.size()
           << ",compatible_job_count=" << compatibleCount
           << ",waiting_for_outcome_data_count=" << waitingCount
           << ",ready_to_execute_count=" << readyCount
           << ",completed_outcome_count=0"
           << ",proof_future_outcome_not_executed=true"
           << OutcomeSafetyFields() << '\n';
    return 0;
}

int RunCampaignProfitabilityProspectiveComparisonCommand(
    const std::string& connectionString,
    const std::string& validationCohortIdentityHash,
    std::ostream& output,
    std::ostream& errors,
    const std::string& phase11ArtifactPath,
    const std::string& phase12PreparationArtifactPath,
    const std::string& currentDateOverride)
{
    (void)errors;
    if (validationCohortIdentityHash !=
        kPhase12ValidationCohortIdentityHash)
        throw std::invalid_argument("phase13_validation_cohort_hash_mismatch");
    pqxx::connection connection{connectionString};
    pqxx::read_transaction transaction{connection};
    transaction.exec("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ;");
    const auto comparison = LoadCampaignProfitabilityProspectiveComparison(
        transaction,
        currentDateOverride.empty() ? CurrentUtcDate() : currentDateOverride,
        phase11ArtifactPath, phase12PreparationArtifactPath);

    const auto coverageFields = [](const CampaignProfitabilitySourceCoverage& c,
                                   const std::string& prefix) {
        std::ostringstream rendered;
        rendered << prefix << "required_source_model_ids="
                 << IdList(c.requiredSourceModelIds)
                 << ',' << prefix << "covered_source_model_ids="
                 << IdList(c.coveredSourceModelIds)
                 << ',' << prefix << "missing_source_model_ids="
                 << IdList(c.missingSourceModelIds)
                 << ',' << prefix << "incompatible_source_model_ids="
                 << IdList(c.incompatibleSourceModelIds)
                 << ',' << prefix << "required_source_model_count="
                 << c.requiredCount
                 << ',' << prefix << "covered_source_model_count="
                 << c.coveredCount
                 << ',' << prefix << "coverage_percentage="
                 << Number(c.percentage);
        return rendered.str();
    };
    const auto reasons = [](const std::vector<std::string>& values) {
        std::string rendered;
        for (const auto& value : values)
        {
            if (!rendered.empty()) rendered.push_back(':');
            rendered += MachineText(value);
        }
        return rendered.empty() ? std::string{"NONE"} : rendered;
    };
    const std::string common =
        std::string{",validation_cohort_identity_hash="} +
        comparison.validationCohortIdentityHash +
        ",ranking_snapshot_id=5,source_evaluation_run_id=6" +
        ",phase11_artifact_sha256=" + comparison.phase11ArtifactSha256 +
        ",phase12_preparation_artifact_sha256=" +
        comparison.phase12PreparationArtifactSha256 +
        ",phase12_preparation_identity_hash=" +
        comparison.phase12PreparationIdentityHash +
        ",metric_hash=" + comparison.metricDefinitionHash +
        ",outcome_start=" + comparison.outcomeStart +
        ",outcome_end=" + comparison.outcomeEnd;

    output << "CAMPAIGN_PROFITABILITY_PROSPECTIVE_COMPARISON_SUMMARY"
           << common << ",current_date=" << comparison.currentDate
           << ",readiness="
           << ProspectiveComparisonReadinessText(comparison.readiness)
           << ",blocking_reasons=" << reasons(comparison.blockingReasons)
           << ",final=" << Boolean(comparison.final)
           << ",comparison_identity_hash=" << comparison.hash
           << OutcomeSafetyFields() << ",database_write=false\n";
    output << "CAMPAIGN_PROFITABILITY_PROSPECTIVE_FULL_COHORT_COVERAGE"
           << common << ',' << coverageFields(
                  comparison.fullFrozenCohortCoverage, "")
           << ",coverage_scope=all_23_frozen_source_models"
           << ",known_incompatible_source_model_id=499"
           << ",known_incompatible_reason=model_lineage_ambiguous"
           << OutcomeSafetyFields() << ",database_write=false\n";

    for (const auto& top : comparison.topN)
    {
        std::string duplicateRelationships;
        for (const auto& [modelId, recommendationIds] :
             top.recommendationIdsBySourceModel)
        {
            if (recommendationIds.size() < 2) continue;
            if (!duplicateRelationships.empty())
                duplicateRelationships.push_back('|');
            duplicateRelationships += std::to_string(modelId) + ':' +
                IdList(recommendationIds);
        }
        if (duplicateRelationships.empty()) duplicateRelationships = "NONE";
        output << "CAMPAIGN_PROFITABILITY_PROSPECTIVE_TOP_N" << common
               << ",top_n=" << top.n
               << ",control_recommendation_ids="
               << IdList(top.controlRecommendationIds)
               << ",candidate_recommendation_ids="
               << IdList(top.candidateRecommendationIds)
               << ",retained_recommendation_ids="
               << IdList(top.retainedRecommendationIds)
               << ",entrant_recommendation_ids="
               << IdList(top.entrantRecommendationIds)
               << ",exit_recommendation_ids="
               << IdList(top.exitRecommendationIds)
               << ",control_recommendation_count="
               << top.controlRecommendationIds.size()
               << ",candidate_recommendation_count="
               << top.candidateRecommendationIds.size()
               << ",control_unique_source_model_ids="
               << IdList(top.controlSourceModelIds)
               << ",candidate_unique_source_model_ids="
               << IdList(top.candidateSourceModelIds)
               << ",retained_unique_source_model_ids="
               << IdList(top.retainedSourceModelIds)
               << ",entrant_unique_source_model_ids="
               << IdList(top.entrantSourceModelIds)
               << ",exit_unique_source_model_ids="
               << IdList(top.exitSourceModelIds)
               << ",duplicate_source_relationships="
               << duplicateRelationships
               << ",economic_weighting=recommendation_selection_slot"
               << ",statistical_evidence_unit=unique_source_model"
               << ",duplicate_recommendations_are_independent=false"
               << ",common_member_contribution=cancelled_not_recomputed"
               << ",common_member_contribution_cancels=true"
               << ',' << coverageFields(
                      top.changedSelectionCoverage, "changed_selection_")
               << ",entrant_aggregate_return="
               << (top.entrantContribution
                       ? Number(top.entrantContribution->aggregateReturn)
                       : "PENDING")
               << ",entrant_prediction_count="
               << (top.entrantContribution
                       ? std::to_string(top.entrantContribution->predictionCount)
                       : "PENDING")
               << ",entrant_actionable_count="
               << (top.entrantContribution
                       ? std::to_string(top.entrantContribution->actionableCount)
                       : "PENDING")
               << ",entrant_average_return_per_actionable_prediction="
               << (top.entrantContribution
                       ? OptionalNumber(top.entrantContribution->
                             averageReturnPerActionablePrediction)
                       : "PENDING")
               << ",exit_aggregate_return="
               << (top.exitContribution
                       ? Number(top.exitContribution->aggregateReturn)
                       : "PENDING")
               << ",exit_prediction_count="
               << (top.exitContribution
                       ? std::to_string(top.exitContribution->predictionCount)
                       : "PENDING")
               << ",exit_actionable_count="
               << (top.exitContribution
                       ? std::to_string(top.exitContribution->actionableCount)
                       : "PENDING")
               << ",exit_average_return_per_actionable_prediction="
               << (top.exitContribution
                       ? OptionalNumber(top.exitContribution->
                             averageReturnPerActionablePrediction)
                       : "PENDING")
               << ",candidate_minus_control_incremental_profitability="
               << (top.candidateMinusControlIncrementalProfitability
                       ? Number(*top.candidateMinusControlIncrementalProfitability)
                       : "PENDING")
               << ",incremental_per_actionable_profitability=NOT_DEFINED"
               << ",readiness="
               << ProspectiveComparisonReadinessText(top.readiness)
               << ",blocking_reasons=" << reasons(top.blockingReasons)
               << ",final=" << Boolean(top.final)
               << ",top_n_comparison_hash=" << top.hash
               << OutcomeSafetyFields() << ",database_write=false\n";
    }
    if (comparison.final) return 0;
    if (comparison.readiness ==
            ProspectiveComparisonReadiness::pendingOutcomes ||
        comparison.readiness == ProspectiveComparisonReadiness::
            incompleteChangedSelectionCoverage)
        return 4;
    return 3;
}

} // namespace EA::ProfitabilityVerification
