#include "ProfitabilityVerificationService.hpp"

#include "ExperimentRecommendationCampaignPlanningRepository.hpp"
#include "ProfitabilityVerificationRepository.hpp"

#include <algorithm>
#include <cmath>
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

} // namespace EA::ProfitabilityVerification
